#!/usr/bin/env python3
import argparse, json, math
import polars as pl
from pathlib import Path


import polars as pl

def repair_capacity(assign, users, df_eq, violations, lam, alpha_df, metric="difficulty"):
    # keep only the columns we operate on
    
    cur = assign.select(["id", "user_id", "p", "cost", metric, "score"])
    
    if alpha_df is None:
        alpha_df = users.select(pl.col("user_id"), pl.lit(0.0).alias("alpha"))

    scores = (
        df_eq
        .join(users.select("user_id","cost"), on="user_id")
        .join(violations.select("id",metric), on="id")
        .join(alpha_df.select("user_id","alpha"), on="user_id", how="left")
        .with_columns(pl.col("alpha").fill_null(0.0))
        .with_columns((pl.col("p") - lam*pl.col("cost") - pl.col("alpha")*pl.col(metric)).alias("score"))
        .select(["id","user_id","p","cost",metric,"score"])  # <- normalize here too
    )

    CC = dict(zip(users["user_id"].to_list(), users["capacity"].to_list()))

    while True:
        loads = cur.group_by("user_id").agg(pl.sum(metric).alias("load"))
        over = (
            loads.join(users.select("user_id","capacity"), on="user_id")
                 .with_columns((pl.col("load") - pl.col("capacity")).alias("overflow"))
                 .filter(pl.col("overflow") > 1e-9)
        )
        if over.is_empty():
            return cur

        ou = over.sort("overflow", descending=True).row(0, named=True)["user_id"]
        mine = cur.filter(pl.col("user_id")==ou).sort("score", descending=False)

        moved = False
        for r in mine.iter_rows(named=True):
            
            v = r["id"]; dv = float(r[metric])
            alt = (scores.filter(pl.col("id")==v)
                         .filter(pl.col("user_id")!=ou)
                         .sort("score", descending=True))
            for nb in alt.iter_rows(named=True):
                u2 = nb["user_id"]
                load_u2 = float(loads.filter(pl.col("user_id")==u2)["load"].fill_null(0.0).sum())
                if load_u2 + dv <= float(CC[u2]) + 1e-9:
                    # build a single-row DF with the SAME columns as `cur`
                    row_df = pl.DataFrame([{
                        "id": v,
                        "user_id": u2,
                        "p": float(nb["p"]),
                        "cost": float(nb["cost"]),
                        metric: float(nb[metric]),
                        "score": float(nb["score"]),
                    }]).select(cur.columns)  # ensure order/width match

                    cur = pl.concat([cur.filter(pl.col("id") != v), row_df], how="vertical")
                    moved = True
                    break
            if moved:
                break

        if not moved:
            return cur








def per_node_inner_exact(df_eq: pl.DataFrame,
                         users: pl.DataFrame,
                         violations: pl.DataFrame,
                         lam: float,
                         metric: str,
                         alpha_df: pl.DataFrame | None = None) -> tuple[pl.DataFrame, dict]:
    """
    df_eq      : [violation_id, user_id, p]
    users      : [user_id, c, CC]  (CC optional if no capacity multipliers)
    violations : [violation_id, difficulty]
    lam        : float
    alpha_df   : optional [user_id, alpha] (default zeros)
    """
    # ensure columns exist with consistent names/dtypes
    
    if metric not in violations.columns:
        raise ValueError("violations must have column 'difficulty'")
    if "cost" not in users.columns:
        raise ValueError("users must have per-assignment cost column 'c'")

    # join costs, difficulties
    df = (
        df_eq
        .join(users.select("user_id","cost","capacity"), on="user_id", how="left")
        .join(violations.select("id",metric), on="id", how="left")
    )

    # attach alpha (capacity multipliers) or zeros
    if alpha_df is None:
        df = df.with_columns(pl.lit(0.0).alias("alpha"))
        #alpha_df = pl.DataFrame({"user_id": users["user_id"], "alpha": pl.lit(0.0)})
        alpha_df = users.select(
    pl.col("user_id"),
    pl.lit(0.0).cast(pl.Float64).alias("alpha"),
)
    else:
        if not {"user_id","alpha"} <= set(alpha_df.columns):
            raise ValueError("alpha_df must have columns ['user_id','alpha']")
        df = df.join(alpha_df.select("user_id","alpha"), on="user_id", how="left").with_columns(
            pl.col("alpha").fill_null(0.0)
        )

    # reduced score s = p - lam*c - alpha * difficulty
    df = df.with_columns(
        (pl.col("p") - lam * pl.col("cost") - pl.col("alpha") * pl.col(metric)).alias("score")
    )

    # choose best user per node (deterministic tie-break: higher p, lower cost, smaller user_id)
    df_sorted = df.sort(
        by=["id", "score", "p", "cost", "user_id"],
        descending=[False, True, True, False, False],
    )
    assign = df_sorted.unique(subset=["id"], keep="first")

    # metrics
    spent = float(assign["cost"].sum())
    total_quality = float(assign["p"].sum())
    score_sum = float(assign["score"].sum())  # = sum_v max_i s_i(v)

    # per-user loads (sum of difficulties); and selected node counts
    by_user = (assign.group_by("user_id")
                     .agg([pl.sum(metric).alias("load"),
                           pl.len().alias("num_nodes")]))

    loads = dict(zip(by_user["user_id"].to_list(), by_user["load"].to_list()))

    info = {
        "lambda": lam,
        "spent": spent,
        "total_quality": total_quality,
        "score_sum": score_sum,
        "loads": loads,
        "num_nodes": assign.height,
        
        "num_users_used": int(assign["user_id"].n_unique()),

    }
    return assign, info

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
    
    
    df_eq = pl.read_ipc(f"{dataset}/eq_{metric}_eqs.feather", memory_map=False)
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
        assign, info = per_node_inner_exact(df_eq, users, violations, lam,metric, alpha_df)

        # save assignment periodically
        if args.save_assign_every and (t % args.save_assign_every == 0):
            assign.write_ipc(outdir / f"assign_{t:03d}.feather", compression="zstd")
        # capacity repair (only if enforcing capacities)
        assign_feas = assign
        if args.cap_mults:
            assign_feas = repair_capacity(assign, users, df_eq, violations, lam, alpha_df,  metric=metric)

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
                                pl.len().alias("num_nodes"),
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


