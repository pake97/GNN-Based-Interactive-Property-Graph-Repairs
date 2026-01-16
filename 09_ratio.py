#!/usr/bin/env python3
import argparse, json, math
from pathlib import Path
from bisect import bisect_left
import polars as pl
import numpy as np
# ---------- utilities ----------

def build_users_sorted(users: pl.DataFrame):
    """Return users sorted by skill + arrays for fast queries."""
    u = users.select("user_id","skills","cost","capacity").sort("skills")
    uid = u["user_id"].to_list()
    s   = [float(x) for x in u["skills"].to_list()]    # skills
    c   = [float(x) for x in u["cost"].to_list()]        # per-assignment cost
    CC  = [float(x) for x in u["capacity"].to_list()]       # capacity
    return u, uid, s, c, CC

def build_sparse_table_min_idx(vals):
    """Sparse table (argmin indices) for range-min queries over vals."""
    n = len(vals)
    import math
    K = (n).bit_length()
    st = [[0]*n for _ in range(K)]
    for i in range(n):
        st[0][i] = i
    k = 1
    while (1 << k) <= n:
        half = 1 << (k-1)
        for i in range(0, n - (1<<k) + 1):
            i1 = st[k-1][i]
            i2 = st[k-1][i + half]
            st[k][i] = i1 if vals[i1] <= vals[i2] else i2
        k += 1
    return st

def rmq_argmin(vals, st, L, R):
    """Return index of min(vals[L:R]) with 0<=L<R<=n."""
    if L >= R: return None
    import math
    k = (R - L).bit_length() - 1
    i1 = st[k][L]
    i2 = st[k][R - (1<<k)]
    return i1 if vals[i1] <= vals[i2] else i2

def band_indices(skill_sorted, d):
    """Given difficulty d, return index ranges for the 3 bands in the skill-sorted array."""
    # p=1 for s > d-0.2  ⇒ s >= (d-0.2) is fine (ties won’t matter)
    # p=.66 for d-0.4 < s <= d-0.2  ⇒ s in [d-0.4, d-0.2)
    # p=.5  for d-0.6 < s <= d-0.4  ⇒ s in [d-0.6, d-0.4)
    s = skill_sorted
    iH  = bisect_left(s, d - 0.2)   # [iH, n)
    iM1 = bisect_left(s, d - 0.4)   # [iM1, iH)
    iM2 = bisect_left(s, d - 0.6)   # [iM2, iM1)
    return iM2, iM1, iH  # M2=[iM2,iM1), M1=[iM1,iH), H=[iH,n)

def best_in_range_excl(costs, st, L, R, excl_idx=None):
    """Argmin of costs[L:R], optionally excluding excl_idx if it falls inside."""
    if L >= R: return None
    j = rmq_argmin(costs, st, L, R)
    if excl_idx is None or j != excl_idx:
        return j
    # need second best: check left and right halves around excl_idx
    j1 = rmq_argmin(costs, st, L, excl_idx)
    j2 = rmq_argmin(costs, st, excl_idx+1, R)
    if j1 is None: return j2
    if j2 is None: return j1
    return j1 if costs[j1] <= costs[j2] else j2

# ---------- inner (pair-free) ----------

def assign_per_node_pairfree(users_sorted_df: pl.DataFrame,
                             uid, skills, costs,
                             violations: pl.DataFrame,
                             lam: float, metric = "difficulty") -> pl.DataFrame:
    """Exact budget-only inner: per node pick user maximizing p - lam*c among the 3 bands."""
    st = build_sparse_table_min_idx(costs)  # RMQ over costs
    rows = []
    for v, d in zip(violations["id"].to_list(),
                    violations[metric].to_list()):
        iM2, iM1, iH = band_indices(skills, float(d))
        best = None  # (score, p, cost, user_idx)
        # H band (p=1)
        j = best_in_range_excl(costs, st, iH, len(costs))
        if j is not None:
            p = 1.0; sc = p - lam*costs[j]
            best = (sc, p, costs[j], j)
        # M1 (p=.66)
        j = best_in_range_excl(costs, st, iM1, iH)
        if j is not None:
            p = 0.66; sc = p - lam*costs[j]
            if (best is None) or (sc > best[0]): best = (sc, p, costs[j], j)
        # M2 (p=.5)
        j = best_in_range_excl(costs, st, iM2, iM1)
        if j is not None:
            p = 0.5; sc = p - lam*costs[j]
            if (best is None) or (sc > best[0]): best = (sc, p, costs[j], j)
        if best is None:
            # no band qualifies -> p=0 with global cheapest cost
            j = rmq_argmin(costs, st, 0, len(costs))
            p = 0.0; sc = - lam*costs[j]; best = (sc, p, costs[j], j)
        sc, p, c, j = best
        rows.append((v, uid[j], p, c, float(d), sc, j))
    return pl.DataFrame(rows, schema=["id","user_id","p","cost",metric,"score","user_idx"], orient="row")

# ---------- capacity repair (pair-free) ----------

def repair_capacity_pairfree(assign: pl.DataFrame,
                             users_sorted_df: pl.DataFrame,
                             uid, skills, costs, CC_list,
                             violations: pl.DataFrame,
                             lam: float, metric = "difficulty"):
    """
    Greedily move lowest-score nodes off overloaded users to next-best users
    (by p - lam*c using the same 3 bands), while respecting CC.
    """
    st = build_sparse_table_min_idx(costs)
    # current loads per user_idx
    load = {j:0.0 for j in range(len(uid))}
    for uidx, d in zip(assign["user_idx"].to_list(), assign[metric].to_list()):
        load[uidx] += float(d)

    # map from user_id to index for quick CC lookup (here we use arrays directly)
    # repeated until all within CC or no move possible
    df = assign
    while True:
        # find most overloaded user_idx
        over_j = None; over_amt = 0.0
        for j in range(len(uid)):
            ov = load[j] - CC_list[j]
            if ov > over_amt + 1e-9:
                over_amt, over_j = ov, j
        if over_j is None:
            return df  # all within CC

        # candidate nodes currently on over_j, ascending by score (cheapest to move first)
        mine = df.filter(pl.col("user_idx")==over_j).sort("score", descending=False)

        moved = False
        for r in mine.iter_rows(named=True):
            v = r["id"]; d = float(r[metric]); cur_j = int(r["user_idx"])
            # consider alternatives in each band, excluding cur_j
            iM2, iM1, iH = band_indices(skills, d)
            candidates = []
            # H
            jH = best_in_range_excl(costs, st, iH, len(costs), excl_idx=cur_j)
            if jH is not None: candidates.append((1.0, jH))
            # M1
            j1 = best_in_range_excl(costs, st, iM1, iH, excl_idx=cur_j)
            if j1 is not None: candidates.append((0.66, j1))
            # M2
            j2 = best_in_range_excl(costs, st, iM2, iM1, excl_idx=cur_j)
            if j2 is not None: candidates.append((0.5, j2))

            # pick best feasible alt by score and capacity room
            cand_best = None
            for p, j2 in candidates:
                if load[j2] + d <= CC_list[j2] + 1e-9:
                    sc = p - lam*costs[j2]
                    if (cand_best is None) or (sc > cand_best[0]):
                        cand_best = (sc, p, j2)
            if cand_best is None:
                continue  # try next node

            sc2, p2, j2 = cand_best
            # commit move v: cur_j -> j2
            load[cur_j] -= d
            load[j2]    += d
            df = (df.filter(pl.col("id") != v)
                    .vstack(pl.DataFrame([{
                        "id": v,
                        "user_id": uid[j2],
                        "p": p2,
                        "cost": costs[j2],
                        metric: d,
                        "score": sc2,
                        "user_idx": j2,
                    }], orient="row")))
            moved = True
            break

        if not moved:
            # no legal move -> structurally infeasible w.r.t. CC
            return df



def EQ(skill, difficulty):
    if skill - difficulty > -0.2:
        return 1
    elif skill - difficulty > -0.4 and skill - difficulty <= -0.2:
        return 0.66
    elif skill - difficulty > -0.6 and skill - difficulty <= -0.4:
        return 0.5
    else:
        return 0


def score_all_violations_for_user(user_row, viols_df, K):
    """
    Returns a NumPy array p_i(v) for all violations v for this user i.

    Args:
      user_row: a dict-like row from users DataFrame (Polars .iter_rows(named=True))
      viols_df: Polars DataFrame of violations
      metric_col: string name of the difficulty column D_v (e.g., args.metric)
      theta: slope/scale for EQ if using logistic/exp
      kind: which EQ form to use ('logistic', 'hinge', 'exp')
    """
    # difficulties D_v
    D = viols_df['difficulty'].to_numpy()  # shape (|V|,)
    # user skill K_i
    # compute vector of scores p_i(v) = EQ(D_v, K_i)
    
    
    
    
    return np.array([EQ(K, float(d)) for d in D])  # shape (|V|,)


def compute_Vi_fractional(assignment_df, users_df):
    """
    Returns:
      V: list of per-user upper bounds V_i (fractional knapsack on all violations)
      C: list of costs c_i aligned with users_df
    Expects:
      users_df has columns: user_id, cost, CC (capacity) and user skill params
      viols has column 'D' or the name in `metric` giving D_v
    NOTE: You must plug in your scoring for p_i(v)=EQ(D_v,K_i).
    """
    import numpy as np
    V, C = [], []
    
    # You need a function that returns p_i(v) for all viols given user i.
    # Example: score_all_violations_for_user(i, users_df_row, viols) -> np.array of p per node
    for row in assignment_df.iter_rows(named=True):
        c_i  = float(row["cost"])
        user_skill = users_df.filter(pl.col("user_id")==row["user_id"]).select("skills").to_numpy()[0,0]
        print(user_skill)
        violatons = assignment_df.filter(pl.col("user_id")==row["user_id"])
        p = score_all_violations_for_user(row, violatons, user_skill)  # <--- implement using your EQ()
        w = violatons["difficulty"].to_numpy()  # weights = difficulties D_v
        # fractional knapsack by value/weight
        idx = np.argsort(-(p / np.maximum(w, 1e-12)))
        w_sorted = w[idx]; p_sorted = p[idx]
        cap = users_df.filter(pl.col("user_id")==row["user_id"]).select("capacity").to_numpy()[0,0]; value = 0.0
        for val, wt in zip(p_sorted, w_sorted):
            if cap <= 1e-12 or wt <= 0: break
            take = min(1.0, cap / wt)
            value += val * take
            cap -= wt * take
        V.append(float(value)); C.append(c_i)
    return V, C



# ---------- outer loop ----------

def main():
    ap = argparse.ArgumentParser()
    
    ap.add_argument("--budget", type=float, required=True)
    ap.add_argument("--lam0", type=float, default=0.0)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--step0", type=float, default=1.0)
    ap.add_argument("--save-assign-every", type=int, default=0)
    ap.add_argument('--dataset', type=str, required=True, help='Dataset name or path (string)')
    ap.add_argument('--theta', type=float, required=True, help='Theta')
    ap.add_argument('--metric', type=str, required=True, help='DIfficulty metric')
    
    args = ap.parse_args()

    
    dataset = args.dataset
    theta = args.theta
    metric = args.metric
    budget = args.budget
    outpath = f"{dataset}/approx_{metric}_{str(theta)}_{str(budget)}/"
    outdir = Path(outpath); outdir.mkdir(parents=True, exist_ok=True)

    users = pl.read_ipc(f"./{dataset}/users.feather", memory_map=False)
    viols = pl.read_ipc(f"{dataset}/grdgs/{str(theta)}_nodes.feather", memory_map=False)
    
    

    

    users_sorted_df, uid, skills, costs, CC_list = build_users_sorted(users)

    lam = 0.0 if args.lam0 is None else args.lam0
    history = []
    best = None
    Z_star = 0.0
    
   
    for t in range(args.iters):
        # for lam in [0,0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,1.0]: 
        # inner (pair-free)
        assign = assign_per_node_pairfree(users_sorted_df, uid, skills, costs, viols, lam, metric)        
        print(budget,lam)
        # metrics
        spent  = float(assign["cost"].sum())             # relaxed spend (pre-repair)
        spentF = float(assign["cost"].sum())        # feasible spend (same here)
        qualF  = float(assign["p"].sum())
        # feasibility: budget only here
        feas_budget = spentF <= args.budget + 1e-12
        
        
        UCB_dec = 0
        for a in assign.iter_rows(named=True):
            UCB_dec += max(0.0, a["p"] - lam * a["cost"])
        UCB_dec += lam * args.budget
        
        
        users = assign["user_id"].to_list()
        
        users_subgraph = {}
        
        for u in users:
            
            users_subgraph[u] = { 'cost' : users_sorted_df.filter(pl.col("user_id")==u)["cost"].sum(), 'quality': assign.filter(pl.col("user_id")==u)["p"].sum()}
            users_subgraph[u]['ratio'] = users_subgraph[u]['quality']/users_subgraph[u]['cost'] if users_subgraph[u]['cost']>0 else 0
        
        
        users_sorted = sorted(users_subgraph.items(), key=lambda x: x[1]['quality']/x[1]['cost'], reverse=True)
        
        total_cost = np.sum([x[1]['cost'] for x in users_sorted])
        
        if total_cost<budget:
            best_lambda = 0
            UCB_CERT = np.sum([x[1]['quality'] for x in users_sorted])
            print(UCB_CERT)
            print(best_lambda)
            break
        else: 
            #find index of first user such that sum of costs is <= budget
            cum_cost = 0
            for i, x in enumerate(users_sorted):
                cum_cost += x[1]['cost']
                if cum_cost > budget:
                    break
            best_lambda = users_sorted[i-1][1]['ratio'] if i>0 else 0
            UCB_CERT = np.sum([x[1]['quality'] for x in users_sorted[:i]]) - best_lambda * (budget - np.sum([x[1]['cost'] for x in users_sorted[:i]])) + best_lambda * budget
            print(UCB_CERT)
            print(best_lambda)
            break
        
        # Z = assign["p"].sum()
        # C = assign["cost"].sum()
        # UB = Z + lam * (args.budget - C)  # classic dual expression
        # Z_star =max(Z_star, float(Z)) if best is not None else float(Z)
        # UB_min = min(UB, history[-1]["UB"]) if len(history) > 0 else UB
        
        

        #print(f"iter {t:3d}: lam={lam:.4f} step={args.step0 / math.sqrt(t + 1.0):.4f} spent={spent:.4f} qual={qualF:.4f} Z*={Z_star:.4f} UB={UB:.4f} UBmin={UB_min:.4f} gap={gap if gap is not None else 'NA':.6f} ratio={ratio if ratio is not None else 'NA':.6f} feas_budget={feas_budget}")
        
        # UB_lam = 0
        
        # for a in assign.iter_rows(named=True):
        #     UB_lam += max(0.0, a["p"] - lam * a["cost"])
        # UB_lam += lam * args.budget
        
        
        # g = spent - args.budget
        # step = args.step0 / math.sqrt(t + 1.0)
        # lam = max(0.0, lam + step * g)
        # print("Z_stars:" , Z_star)
        # print("UB_mins:", UB_min)
        # print("gaps:", gap)
        # print("ratios:", ratio)
    
    #     # subgradient update (budget only): g = spent - B  (use relaxed spent for dual)
    #     g = spent - args.budget
    #     step = args.step0 / math.sqrt(t + 1.0)
    #     lam = max(0.0, lam + step * g)

    #     if args.save_assign_every and (t % args.save_assign_every == 0):
    #         assign.write_ipc(outdir / f"assign_{t:03d}.feather", compression="zstd")

    #     history.append({"iter": t, "lambda": lam, "spent_rel": spent,
    #                     "spent_feas": spentF, "quality_feas": qualF, "step": step})



    #     # Priced inner value your selector achieved at current lambda
    #     V_priced_t = qualF - lam * spentF
    #     UB_t = V_priced_t + lam * budget   # classic dual expression
    #     # One-time (before loop): V_list, C_list = compute_Vi_fractional(...)
    #     # Here, get the decoupled inner UB at current lambda:
        
        
        
    #     if 'V_list' not in locals():
    #         V_list, C_list = compute_Vi_fractional(assign_feas, users_sorted_df)

    #     print(len(V_list), len(C_list))
        
    #     UB_inner_dec_t = sum(max(0.0, V_i - lam * c_i) for V_i, c_i in zip(V_list, C_list))
    #     print(UB_inner_dec_t)
    #     print(V_priced_t)
        
    #     beta_hat_t = 0.0 if UB_inner_dec_t <= 1e-12 else max(0.0, min(1.0, V_priced_t / UB_inner_dec_t))
    #     print(beta_hat_t)
    #     # Certified per-iter UB using beta_hat_t (safe lower bound on true beta)
    #     UB_cert_t = (UB_t / max(beta_hat_t, 1e-12))  # avoid divide-by-zero
        
    #     # Track bests across the run
    #     UB_min_priced = min(UB_min_priced, UB_t) if 'UB_min_priced' in locals() else UB_t
    #     UB_min_cert   = min(UB_min_cert, UB_cert_t) if 'UB_min_cert' in locals() else UB_cert_t
    #     beta_hat_min  = min(beta_hat_min, beta_hat_t) if 'beta_hat_min' in locals() else beta_hat_t

    #     # (optional) log
    #     history[-1].update({"V_priced": V_priced_t,
    #                         "UB_t": UB_t,
    #                         "UB_inner_dec": UB_inner_dec_t,
    #                         "beta_hat": beta_hat_t,
    #                         "UB_cert_t": UB_cert_t})


    #     # simple stop if near budget
    #     if abs(g) < 1e-6 and step < 1e-4:
    #         break

    # Z_star = best["quality"] if best is not None else 0.0
    # UB_cert = UB_min_cert if 'UB_min_cert' in locals() else None
    # gap = (UB_cert - Z_star) / UB_cert if (UB_cert and UB_cert > 0) else None
    # ratio = Z_star / UB_cert if (UB_cert and UB_cert > 0) else None

    # summary = {
    #     "best_feasible": best,
    #     "final_lambda": lam,
    #     "iterations": len(history),
    #     "UB_priced_min": UB_min_priced if 'UB_min_priced' in locals() else None,
    #     "beta_hat_min": beta_hat_min if 'beta_hat_min' in locals() else None,
    #     "UB_cert": UB_cert,
    #     "cert_gap": gap,
    #     "cert_ratio": ratio,
    # }
    # (outdir / "summary.json").write_text(json.dumps(summary, indent=2))



if __name__ == "__main__":
    main()
