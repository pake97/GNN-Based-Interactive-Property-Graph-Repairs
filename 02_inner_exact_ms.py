import polars as pl
import numpy as np
import argparse
import math
from pathlib import Path
import json



def per_node_inner_exact_no_eq(users: pl.DataFrame,
                               violations: pl.DataFrame,
                               lam: float,
                               alpha_df: pl.DataFrame | None = None,
                               metric: str = "difficulty",
                               ) -> tuple[pl.DataFrame, dict]:
    """
    Exact per-node assignment without precomputed eq pairs.
    users: [user_id, skill, c, CC]
    violations: [violation_id, difficulty]
    lam: lambda multiplier (float)
    alpha_df: optional [user_id, alpha] (capacity multipliers). If None -> zeros.

    Returns:
      assign: per-node DF [violation_id, user_id, p, c, difficulty, score]
      info:   dict with totals (spent, total_quality, score_sum, num_users_used, ...)
    """
    # --- inputs to numpy arrays ---
    u_ids   = users["user_id"].to_list()
    u_skill = users["skills"].to_numpy().astype(np.float64, copy=False)
    u_c     = users["cost"].to_numpy().astype(np.float64, copy=False)
    u_CC    = users["capacity"].to_numpy().astype(np.float64, copy=False)

    if alpha_df is None:
        u_alpha = np.zeros(len(u_ids)).astype(np.float64, copy=False)
    else:
        # map alpha by user_id
        a_map = dict(zip(alpha_df["user_id"].to_list(), alpha_df["alpha"].to_list()))
        u_alpha = np.array([float(a_map.get(uid, 0.0)) for uid in u_ids]).astype(np.float64, copy=False)

    v_ids = violations["id"].to_list()
    v_D   = violations[metric].to_numpy().astype(np.float64, copy=False)

    # Precompute intercept/slope for reduced term: lam*c + D*alpha
    intercept = lam * u_c           # shape [U]
    slope     = u_alpha             # shape [U]

    rows = []
    spent = 0.0
    total_p = 0.0
    score_sum = 0.0
    chosen_users = set()

    U = len(u_ids)

    for vid, d in zip(v_ids, v_D):
        # Feasible users (can at least hold this node wrt CC)
        feas = (u_CC >= d)

        if not np.any(feas):
            # Should not happen if precheck had nodes_missing==0
            # Fallback: allow all users (will violate CC structurally); still choose someone to keep coverage
            feas = np.ones(U).astype(bool)

        # Bands by skill - difficulty
        z = u_skill - d
        band1 = feas & (z > -0.2)                    # p = 1.0
        band2 = feas & (z > -0.4) & (z <= -0.2)      # p = 0.66
        band3 = feas & (z > -0.6) & (z <= -0.4)      # p = 0.5
        band0 = feas & ~(band1 | band2 | band3)      # p = 0.0

        best_score = -np.inf
        best_idx   = None
        best_p     = 0.0

        # helper to evaluate a band
        def consider(mask: np.ndarray, p_val: float):
            nonlocal best_score, best_idx, best_p
            if not np.any(mask):
                return
            # cost term to minimize = intercept + slope * d
            cost = intercept[mask] + slope[mask] * d
            k = np.argmin(cost)
            idx = np.flatnonzero(mask)[k]
            score = p_val - cost[k]
            if score > best_score:
                best_score = score
                best_idx = idx
                best_p = p_val

        # Try positive bands first
        consider(band1, 1.0)
        consider(band2, 0.66)
        consider(band3, 0.5)
        # Fallback to band0 only if nothing selected (should be rare)
        if best_idx is None:
            consider(band0, 0.0)
            # if still None, just pick the absolute best by cost among feas
            if best_idx is None:
                cost = intercept[feas] + slope[feas] * d
                k = np.argmin(cost)
                best_idx = np.flatnonzero(feas)[k]
                best_p = 0.0
                best_score = best_p - cost[k]

        uid = u_ids[best_idx]
        c   = float(u_c[best_idx])

        rows.append({
            "id": vid,
            "user_id": uid,
            "p": float(best_p),
            "cost": c,
            metric: float(d),
            "score": float(best_score),
        })
        spent += c
        total_p += best_p
        score_sum += best_score
        chosen_users.add(uid)

    assign = pl.DataFrame(rows, schema=["id","user_id","p","cost",metric,"score"])
    info = {
        "lambda": lam,
        "spent": spent,
        "total_quality": total_p,
        "score_sum": score_sum,
        "num_nodes": assign.height,
        "num_users_used": len(chosen_users),
    }
    return assign, info


import polars as pl

def _EQ_expr(skill_col: str, d_lit: float):
    # returns a Polars Expr producing the step EQ(skill - d)
    z = pl.col(skill_col) - pl.lit(d_lit)
    return (
        pl.when(z > -0.2).then(1.0)
        .when((z > -0.4) & (z <= -0.2)).then(0.66)
        .when((z > -0.6) & (z <= -0.4)).then(0.5)
        .otherwise(0.0)
        .cast(pl.Float64)
    )

def repair_capacity_no_eq(
    assign: pl.DataFrame,
    users: pl.DataFrame,
    violations: pl.DataFrame,
    lam: float,
    alpha_df: pl.DataFrame | None,
    *,
    id_col: str = "id",
    user_col: str = "user_id",
    p_col: str = "p",
    cost_col: str = "cost",
    metric_col: str = "difficulty",
    capacity_col: str = "capacity",
    skill_col: str = "skill",
) -> pl.DataFrame:
    """
    Greedily reassign overflow from overloaded users to next-best users
    by reduced score computed on-the-fly (no df_eq needed). Keeps full coverage.

    assign: per-node assignment with columns [id_col, user_col, p_col, cost_col, metric_col, "score"]
    users : [user_col, skill_col, cost_col, capacity_col]
    violations: [id_col, metric_col]
    lam   : lambda multiplier
    alpha_df: optional [user_col, "alpha"] multipliers; if None -> zero alphas

    Returns a new DataFrame with the same schema as `assign`.
    """
    # Normalize working frame to the exact columns we use (to avoid vstack width mismatches)
    cur = assign.select([id_col, user_col, p_col, cost_col, metric_col, "score"])

    # alpha per user (defaults to 0)
    if alpha_df is None:
        alpha_df = users.select(pl.col(user_col), pl.lit(0.0).alias("alpha"))
    else:
        if "alpha" not in alpha_df.columns:
            alpha_df = alpha_df.rename({alpha_df.columns[-1]: "alpha"})
        alpha_df = alpha_df.select(pl.col(user_col), pl.col("alpha"))

    # capacity map
    cap_map = dict(zip(users[user_col].to_list(), users[capacity_col].to_list()))

    # Pre-join users + alpha once (static info); per-node we only plug in d and compute EQ, score
    users_alpha = users.join(alpha_df, on=user_col, how="left").with_columns(
        pl.col("alpha").fill_null(0.0)
    )

    while True:
        # current loads
        loads = cur.group_by(user_col).agg(pl.sum(metric_col).alias("load"))

        over = (
            loads.join(users.select(user_col, capacity_col), on=user_col)
                 .with_columns((pl.col("load") - pl.col(capacity_col)).alias("overflow"))
                 .filter(pl.col("overflow") > 1e-9)
        )
        if over.is_empty():
            return cur  # all capacities satisfied

        # pick most overloaded user
        ou = over.sort("overflow", descending=True).row(0, named=True)[user_col]

        # nodes currently on 'ou', try moving lowest reduced-score first
        mine = cur.filter(pl.col(user_col) == ou).sort("score", descending=False)

        # Build a Python map for quick load lookups
        loads_map = dict(zip(loads[user_col].to_list(), loads["load"].to_list()))

        moved = False
        for r in mine.iter_rows(named=True):
            vid = r[id_col]
            dv  = float(r[metric_col])

            # Compute alternative users' EQ and reduced scores for this dv
            # score_alt = EQ(skill, dv) - lam*cost - alpha*dv
            alt = (
                users_alpha
                .with_columns([
                    _EQ_expr(skill_col, dv).alias("_p_alt"),
                    (pl.col(cost_col).cast(pl.Float64)).alias("_c_alt"),
                    (pl.col("alpha").cast(pl.Float64)).alias("_alpha_alt"),
                ])
                .with_columns((pl.col("_p_alt") - lam*pl.col("_c_alt") - pl.lit(dv)*pl.col("_alpha_alt")).alias("_score_alt"))
                .select([user_col, "_p_alt", "_c_alt", "_score_alt"])
                .filter(pl.col(user_col) != ou)                      # cannot keep on same user
                .sort("_score_alt", descending=True)                 # try best-scoring alt first
            )

            # Try to place on first alt user that fits capacity
            for nb in alt.iter_rows(named=True):
                u2 = nb[user_col]
                # capacity check with current loads
                load_u2 = float(loads_map.get(u2, 0.0))
                if load_u2 + dv <= float(cap_map[u2]) + 1e-9:
                    # Build a single-row DF with the SAME columns/order as `cur`
                    row_df = pl.DataFrame([{
                        id_col: vid,
                        user_col: u2,
                        p_col: float(nb["_p_alt"]),
                        cost_col: float(nb["_c_alt"]),
                        metric_col: dv,
                        "score": float(nb["_score_alt"]),
                    }]).select(cur.columns)

                    # Commit move: remove old row for vid, append new
                    cur = pl.concat([cur.filter(pl.col(id_col) != vid), row_df], how="vertical")

                    # Update loads map (so we can possibly place another in this pass)
                    loads_map[ou] = float(loads_map.get(ou, 0.0)) - dv
                    loads_map[u2] = float(loads_map.get(u2, 0.0)) + dv

                    moved = True
                    break
            if moved:
                break

        if not moved:
            # No legal move exists given current capacities ⇒ structurally infeasible to repair
            return cur


def maybe_dual_value(info: dict, lam: float,
                     users: pl.DataFrame,
                     alpha_df: pl.DataFrame | None,
                     B: float | None):
    """
    Dual:  L(λ,α) = sum_v max_i s_i(v) + λ B + sum_i α_i CC_i
                   = info['score_sum'] + λ B + Σ α_i CC_i
    """
    dual = None
    if B is not None:
        dual = info["score_sum"] + lam * float(B)
        if alpha_df is not None and "capacity" in users.columns:
            cc_map = dict(zip(users["user_id"].to_list(), users["capacity"].to_list()))
            for r in alpha_df.iter_rows(named=True):
                dual += float(r["alpha"]) * float(cc_map.get(r["user_id"], 0.0))
    return dual


# ---------- dual value helper (optional, but nice to log) ----------
def dual_value(info: dict, lam: float, B: float,
               users: pl.DataFrame, alpha_df: pl.DataFrame | None):
    """
    L(λ,α) = Σ_v max_i s_i(v) + λ B + Σ_i α_i CC_i
           = info['score_sum'] + λ B + Σ_i α_i CC_i
    """
    dv = info["score_sum"] + lam * float(B)
    if alpha_df is not None and "capacity" in users.columns:
        cc_map = dict(zip(users["user_id"].to_list(), users["capacity"].to_list()))
        for r in alpha_df.iter_rows(named=True):
            dv += float(r["alpha"]) * float(cc_map.get(r["user_id"], 0.0))
    return dv




def main():
    ap = argparse.ArgumentParser()
    
    
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--alpha", default=None, help="optional alpha.feather with [user_id, alpha]")
    ap.add_argument("--budget", type=float, default=None, help="optional B for dual calc")
    
    ap.add_argument('--dataset', type=str, required=True, help='Dataset name or path (string)')
    ap.add_argument('--theta', type=float, required=True, help='Theta')
    ap.add_argument('--metric', type=str, required=True, help='DIfficulty metric')
    ap.add_argument("--lam0", type=float, default=0.0, help="initial lambda")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--step0", type=float, default=1.0, help="base step size for diminishing steps")
    ap.add_argument("--cap-mults", action="store_true",
                    help="use capacity multipliers alpha_i (needs users.CC)")
    ap.add_argument("--save-assign-every", type=int, default=0,
                    help="if >0, write assign_t.feather every k iterations")
    
    args = ap.parse_args()
    
    dataset = args.dataset
    theta = args.theta
    metric = args.metric
    budget = args.budget if args.budget is not None else 1e12  # large budget if none given
    outpath = f"{dataset}/run_{metric}_{str(theta)}_{str(budget)}/"
    outdir = Path(outpath); outdir.mkdir(parents=True, exist_ok=True)

    users = pl.read_ipc(f"./{dataset}/users.feather", memory_map=False)
    violations = pl.read_ipc(f"{dataset}/grdgs/{str(theta)}_nodes.feather", memory_map=False)
    print(violations.schema)
    alpha_df = pl.read_ipc(args.alpha) if args.alpha else None

#     assign, info = per_node_inner_exact(df_eq, users, violations, args.lam, alpha_df)

#     # attach dual (optional)
#     dual = maybe_dual_value(info, args.lam, users, alpha_df, args.budget)
#     if dual is not None:
#         info["dual_value"] = float(dual)

#     # save outputs (Feather/IPC)
#     assign.select(["id","user_id","p","cost",metric,"score"]).write_ipc(
#         outdir / f"assign_{args.iter:03d}.feather", compression="zstd"
#     )
#     # pl.DataFrame([info]).write_ipc(outdir / f"iterinfo_{args.iter:03d}.feather", compression="zstd")

#     # # also drop a tiny JSON summary for quick peeks
#     # (outdir / f"iterinfo_{args.iter:03d}.json").write_text(json.dumps(info, indent=2))

#     # after computing `assign, info`
# # info has keys: lambda, spent, total_quality, score_sum, loads (dict), num_nodes, num_users_used

#     # 1) write a flat one-row table (drop/serialize 'loads')
#     info_flat = {k: v for k, v in info.items() if k != "loads"}
#     pl.DataFrame([info_flat]).write_ipc(
#         outdir / f"iterinfo_{args.iter:03d}.feather",
#         compression="zstd"
#     )

#     # 2) write loads as a separate feather (one row per user)
#     loads_rows = [{"user_id": uid, "load": float(val)} for uid, val in info["loads"].items()]
#     pl.DataFrame(loads_rows).write_ipc(
#         outdir / f"loads_{args.iter:03d}.feather",
#         compression="zstd"
#     )

#     # 3) (optional) also dump everything (including loads) to JSON for easy peeking
#     (outdir / f"iterinfo_{args.iter:03d}.json").write_text(json.dumps(info, indent=2))
#     print(info)
    
    if args.cap_mults and "capacity" not in users.columns:
        raise ValueError("--cap-mults set but users has no 'CC' column")

    # init multipliers
    lam = float(args.lam0)
    alpha_df = None
    if args.cap_mults:
        alpha_df = users.select(pl.col("user_id"), pl.lit(0.0).alias("alpha").cast(pl.Float64))

    history = []
    best_feasible = None

    for t in range(args.iters):
        # inner solve
        assign, info = per_node_inner_exact_no_eq(users, violations, lam, alpha_df,  metric=metric)

        # save assignment periodically
        if args.save_assign_every and (t % args.save_assign_every == 0):
            assign.write_ipc(outdir / f"assign_{t:03d}.feather", compression="zstd")
        # capacity repair (only if enforcing capacities)
        assign_feas = assign
        if args.cap_mults:
            assign_feas = repair_capacity_no_eq(
            assign, users, violations, lam, alpha_df,
            id_col="id",
            user_col="user_id",
            p_col="p",
            cost_col="cost",
            metric_col=metric,   # or your chosen metric column
            capacity_col="capacity",
            skill_col="skills",
        )


        spent_feas = float(assign_feas["cost"].sum())
        feasible_budget = spent_feas <= args.budget + 1e-12  # always true if budget is huge
        feasible_caps = True
        if args.cap_mults:
            loads_tbl = (assign_feas.group_by("user_id")
                                    .agg(pl.sum(metric).alias("load")))
            cc_map = dict(zip(users["user_id"].to_list(), users["capacity"].to_list()))
            feasible_caps = all(
                float(loads_tbl.filter(pl.col("user_id")==uid)["load"].fill_null(0.0).sum())
                <= float(cc_map[uid]) + 1e-9
                for uid in users["user_id"].to_list()
            )

        if feasible_budget and (not args.cap_mults or feasible_caps):
            quality_feas = float(assign_feas["p"].sum())
            if (best_feasible is None) or (quality_feas > best_feasible["quality"] + 1e-12):
                best_feasible = {"iter": t, "quality": quality_feas,
                                "spent": spent_feas, "lambda": lam}
                assign_feas.write_ipc(outdir / "best_assign.feather", compression="zstd")
                (assign_feas.group_by("user_id")
                            .agg([pl.sum(metric).alias("load"),
                                pl.count().alias("num_nodes"),
                                pl.sum("cost").alias("cost_sum"),
                                pl.sum("p").alias("quality_sum")])
                            .write_ipc(outdir / "best_per_user.feather", compression="zstd"))
        # compute dual (nice to have)
        dv = dual_value(info, lam, args.budget, users, alpha_df)

        # # feasibility check for snapshot:
        # feasible_budget = info["spent"] <= args.budget + 1e-12
        # feasible_caps = True
        # if args.cap_mults:
        #     # compute loads and compare to CC
        #     loads_tbl = (assign.group_by("user_id")
        #                        .agg(pl.sum(metric).alias("load")))
        #     loads = dict(zip(loads_tbl["user_id"].to_list(), loads_tbl["load"].to_list()))
        #     cc_map = dict(zip(users["user_id"].to_list(), users["capacity"].to_list()))
        #     feasible_caps = all(loads.get(uid, 0.0) <= float(cc_map[uid]) + 1e-9
        #                         for uid in users["user_id"].to_list())

        # if feasible_budget and (not args.cap_mults or feasible_caps):
        #     if (best_feasible is None) or (info["total_quality"] > best_feasible["quality"] + 1e-12):
        #         best_feasible = {
        #             "iter": t,
        #             "quality": info["total_quality"],
        #             "spent": info["spent"],
        #             "lambda": lam,
        #         }
        #         assign.write_ipc(outdir / "best_assign.feather", compression="zstd")
        #         # also per-user summary
        #         (assign.group_by("user_id")
        #                .agg([pl.sum(metric).alias("load"),
        #                      pl.count().alias("num_nodes"),
        #                      pl.sum("cost").alias("cost_sum"),
        #                      pl.sum("p").alias("quality_sum")])
        #                .write_ipc(outdir / "best_per_user.feather", compression="zstd"))

        # subgradients
        g_lam = info["spent"] - args.budget
        step = args.step0 / math.sqrt(t + 1.0)
        lam = max(0.0, lam + step * g_lam)

        if args.cap_mults:
            # build loads against CC
            loads_tbl = (assign.group_by("user_id")
                               .agg(pl.sum(metric).alias("load")))
            loads = dict(zip(loads_tbl["user_id"].to_list(), loads_tbl["load"].to_list()))
            cc_map = dict(zip(users["user_id"].to_list(), users["capacity"].to_list()))
            # update alphas
            if alpha_df is None:
                alpha_df = users.select(pl.col("user_id"), pl.lit(0.0).alias("alpha").cast(pl.Float64))
            # convert to dict for update
            alpha_map = dict(zip(alpha_df["user_id"].to_list(), alpha_df["alpha"].to_list()))
            for uid in users["user_id"].to_list():
                g_alpha = loads.get(uid, 0.0) - float(cc_map[uid])
                alpha_map[uid] = max(0.0, alpha_map.get(uid, 0.0) + step * g_alpha)
            alpha_df = pl.DataFrame({"user_id": list(alpha_map.keys()),
                                     "alpha": list(alpha_map.values())})


        if args.cap_mults:
            max_over = float(
                (assign.group_by("user_id")
                    .agg(pl.sum(metric).alias("load"))
                    .join(users.select("user_id","capacity"), on="user_id")
                    .with_columns((pl.col("load") - pl.col("capacity")).alias("overflow"))
                    .select(pl.max("overflow").alias("max_over")).fill_null(0.0)
                    .item())
            )
            print(max_over)
        else:
            max_over = 0.0

        # add to history row:

        # log history row (flat only)
        history.append({
            "iter": t,
            "lambda": lam,
            "spent": info["spent"],
            "total_quality": info["total_quality"],
            "dual_value": dv,
            "subgrad_lambda": g_lam,
            "step": step,
            "feasible_budget": feasible_budget,
            "feasible_caps": feasible_caps,
            "num_users_used": info["num_users_used"],
        })

        # simple early stop: near-feasible on budget & tiny step
        if abs(g_lam) < 1e-6 and step < 1e-4:
            break

    # write history + summary
    pl.DataFrame(history).write_ipc(outdir / "history.feather", compression="zstd")

    summary = {
        "best_feasible": best_feasible,
        "final_lambda": lam,
        "iters_run": len(history),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()


