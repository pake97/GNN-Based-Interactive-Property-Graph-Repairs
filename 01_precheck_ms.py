

import polars as pl
import argparse
import sys
#!/usr/bin/env python3
import math
from pathlib import Path

from bisect import bisect_right

def EQ(skill, difficulty):
    if skill - difficulty > -0.2:
        return 1
    elif skill - difficulty > -0.4 and skill - difficulty <= -0.2:
        return 0.66
    elif skill - difficulty > -0.6 and skill - difficulty <= -0.4:
        return 0.5
    else:
        return 0
    
    
def EQ_expr(skill_col: str = "skills", diff_col: str = "difficulty"):
    import polars as pl
    z = pl.col(skill_col) - pl.col(diff_col)
    return (
        pl.when(z > -0.2).then(pl.lit(1.0))
        .when((z > -0.4) & (z <= -0.2)).then(pl.lit(0.66))
        .when((z > -0.6) & (z <= -0.4)).then(pl.lit(0.5))
        .otherwise(pl.lit(0.0))
    )

    
    


def build_eq_shards(users: pl.DataFrame, df_nodes: pl.DataFrame, outdir: Path,
                    shard_max_rows: int = 2_000_000):
    """
    Stream per-user. For user with skill s and CC, emit all violations with:
      d <= min(CC, s+0.6) and assign p in {1,0.66,0.5} by the bands:
        p=1     : d <  s+0.2
        p=0.66  : s+0.2 <= d <= s+0.4
        p=0.5   : s+0.4 <  d <= s+0.6
    Writes shards eq_shard_XXX.feather. Returns (pairs_kept, covered_nodes_set).
    """
    # violations sorted by difficulty for fast slicing
    df_nodes_sorted = df_nodes.sort("difficulty")
    v_ids = df_nodes_sorted["id"].to_list()
    v_d   = df_nodes_sorted["difficulty"].to_list()
    nV = len(v_d)

    buf = []
    shard_idx = 0
    pairs_kept = 0
    covered = set()

    for r in users.iter_rows(named=True):
        uid   = r["user_id"]
        s     = float(r["skills"])
        cc    = float(r["capacity"])

        d1 = min(cc, s + 0.2)
        d2 = min(cc, s + 0.4)
        d3 = min(cc, s + 0.6)

        # indexes: <= thresholds (use bisect_right for inclusive upper bound)
        i1 = bisect_right(v_d, d1)                 # [0, i1) -> p=1
        i2 = bisect_right(v_d, d2)                 # [i1, i2) -> p=0.66
        i3 = bisect_right(v_d, d3)                 # [i2, i3) -> p=0.5

        if i3 == 0:
            continue  # this user can't take any violation with p>0

        # update coverage
        for k in range(i3):
            covered.add(v_ids[k])

        # emit rows (avoid per-row dict allocations when possible)
        if i1 > 0:
            ids = v_ids[0:i1]
            buf.extend((ids[j], uid, 1.0) for j in range(len(ids)))
        if i2 > i1:
            ids = v_ids[i1:i2]
            buf.extend((ids[j], uid, 0.66) for j in range(len(ids)))
        if i3 > i2:
            ids = v_ids[i2:i3]
            buf.extend((ids[j], uid, 0.5) for j in range(len(ids)))

        # flush shard
        if len(buf) >= shard_max_rows:
            df = pl.DataFrame(buf, schema=["id","user_id","p"])
            df.write_ipc(outdir / f"eq_shard_{shard_idx:03d}.feather", compression="zstd")
            pairs_kept += len(buf)
            buf.clear()
            shard_idx += 1

    # final flush
    if buf:
        df = pl.DataFrame(buf, schema=["id","user_id","p"])
        df.write_ipc(outdir / f"eq_shard_{shard_idx:03d}.feather", compression="zstd")
        pairs_kept += len(buf)
        buf.clear()

    return pairs_kept, covered

def lb_budget_without_pairs(users: pl.DataFrame, df_nodes: pl.DataFrame) -> float:
    """
    LB_budget = sum_v min_i c_i  subject to  difficulty_v <= min(CC_i, skill_i+0.6).
    Doable in O(V log U): sort users by threshold t_i = min(CC_i, skill_i+0.6),
    precompute suffix-min of c_i, then binary search per violation.
    """
    u = users.select([
        pl.col("user_id"),
        pl.col("cost").cast(pl.Float64).alias("cost"),
        (pl.min_horizontal(pl.col("capacity"), pl.col("skills") + 0.6)).alias("t"),
    ]).sort("t")  # ascending by threshold

    t = u["t"].to_list()
    c = u["cost"].to_list()
    n = len(t)
    # suffix min of c
    suf_min = [math.inf]*n
    m = math.inf
    for i in range(n-1, -1, -1):
        m = c[i] if c[i] < m else m
        suf_min[i] = m

    # for each violation difficulty d, find first i with t[i] >= d
    v_d = df_nodes["difficulty"].to_list()
    import bisect
    LB = 0.0
    for d in v_d:
        j = bisect.bisect_left(t, float(d))
        if j == n:  # no user can take this node (shouldn't happen if you checked nodes_missing)
            return math.inf
        LB += suf_min[j]
    return LB

def exclusive_overload(eq_scan: pl.LazyFrame, users: pl.DataFrame, df_nodes: pl.DataFrame) -> pl.DataFrame:
    """
    Check nodes with exactly one feasible user and whether their total difficulty
    exceeds that user's CC. Works lazily across shards.
    Returns a (possibly empty) DF of offending users.
    """
    # count feasible users per node
    k_per_node = (eq_scan.group_by("id").agg(pl.len().alias("k")))
    # join to keep only k==1 rows with user_id, then sum difficulty per user
    # We need difficulties; bring df_nodes lazily too
    excl = (k_per_node.filter(pl.col("k")==1)
                      .join(eq_scan, on="id")
                      .join(df_nodes.lazy(), on="id")
                      .group_by("user_id")
                      .agg(pl.sum("difficulty").alias("excl_load"))
                      .join(users.lazy().select("user_id","capacity"), on="user_id")
                      .with_columns((pl.col("excl_load") - pl.col("capacity")).alias("excl_over"))
                      .filter(pl.col("excl_over") > 1e-9)
             )
    return excl.collect(streaming=True)

def main():
    parser = argparse.ArgumentParser(description="Process arguments for theta, users, and dataset.")
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name or path (string)')
    parser.add_argument('--theta', type=float, required=True, help='Theta')
    parser.add_argument('--metric', type=str, required=True, help='DIfficulty metric')
    parser.add_argument("--concat", action="store_true", help="concatenate shards into eq.feather at the end")
    parser.add_argument("--shard-max-rows", type=int, default=2_000_000)
    
    
    args = parser.parse_args()
    
    dataset = args.dataset
    theta = args.theta
    metric = args.metric
    
    
    # Read CSV with polars
    df_nodes = pl.read_ipc(f"{dataset}/grdgs/{str(theta)}_nodes.feather")
    
    


    users = pl.read_ipc(f"{dataset}/users.feather") 
    
    
    outpath = f"{dataset}/run_{metric}_{str(theta)}_pre/"
    outdir = Path(outpath); outdir.mkdir(parents=True, exist_ok=True)
    

    # 0) Feasibility: each node must have at least one feasible user (threshold >= d)
    # We can check coverage after building shards; but we can also do a quick LB now:
    LB_budget = lb_budget_without_pairs(users, df_nodes)

    # 1) Build eq shards (p>0 only), streaming
    pairs_kept, covered = build_eq_shards(users, df_nodes, outdir, shard_max_rows=args.shard_max_rows)
    nodes_missing = df_nodes.height - len(covered)

    # 2) Optional: concat shards to a single eq.feather (lazy, streaming)
    eq_path = outdir / "eq.feather"
    if args.concat:
        shards = sorted(outdir.glob("eq_shard_*.feather"))
        if shards:
            lf = pl.concat([pl.scan_ipc(str(p)) for p in shards])  # lazy concatenate
            lf.sink_ipc(eq_path, compression="zstd")               # stream to single file

    # 3) Capacity LB and exclusive overload (lazy over shards)
    sumD  = float(df_nodes["difficulty"].sum())
    sumCC = float(users["capacity"].sum())
    excl_over_df = exclusive_overload(pl.concat([pl.scan_ipc(str(p)) for p in sorted(outdir.glob("eq_shard_*.feather"))]),
                                      users, df_nodes)

    # 4) Save precheck summary
    pre = pl.DataFrame([{
        "num_users": users.height,
        "num_violations": df_nodes.height,
        "pairs_kept": pairs_kept,
        "nodes_missing": nodes_missing,
        "LB_budget": LB_budget,
        "sumD": sumD,
        "sumCC": sumCC,
        "capacity_LB_ok": sumD <= sumCC + 1e-12,
        "has_missing_nodes": nodes_missing > 0,
        "exclusive_overload_count": excl_over_df.height,
    }])
    pre.write_ipc(outdir / "precheck.feather", compression="zstd")
    if excl_over_df.height > 0:
        excl_over_df.write_ipc(outdir / "exclusive_overload.feather", compression="zstd")

    print(pre)

if __name__ == "__main__":
    main()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process arguments for theta, users, and dataset.")
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name or path (string)')
    parser.add_argument('--theta', type=float, required=True, help='Theta')
    parser.add_argument('--metric', type=str, required=True, help='DIfficulty metric')
    
    
    
    args = parser.parse_args()
    
    dataset = args.dataset
    theta = args.theta
    metric = args.metric
    
    
    # Read CSV with polars
    df_nodes = pl.read_ipc(f"{dataset}/grdgs/{str(theta)}_nodes.feather")
    
    


    users = pl.read_ipc(f"{dataset}/users.feather")
    

    # all pairs
    cross =df_nodes.join(users, how="cross")

    # capacity-feasible pairs only (optional but recommended)
    cross = cross.filter(pl.col(metric) <= pl.col("capacity"))

    # expected quality
    df_eq = (cross.with_columns(p = EQ_expr("skills", metric))
                .filter(pl.col("p") > 0.0)        # drop zero-EQ pairs to shrink
                .select("id", "user_id", "p"))


    df_eq.write_ipc(f"{dataset}/eq_{metric}_eqs.feather", compression="zstd")


    # 1) every node covered by at least one user
    covered_once = df_eq.select("id").unique()
    missing = df_nodes.join(covered_once, on="id", how="anti")
    num_missing = missing.height

    # 2) budget lower bound (per-assignment cost)
    LB_budget = (
        df_eq.join(users.select("user_id","cost"), on="user_id")
            .group_by("id")
            .agg(pl.min("cost").alias("cmin"))
            .select(pl.col("cmin").sum().alias("LB_budget"))
            .item()
    )

    # 3) capacity lower bound
    sumD  = float(df_nodes["difficulty"].sum())
    sumCC = float(users["capacity"].sum())


    exclusive_over = (
    df_eq.group_by("id").agg(pl.count().alias("k"))
      .filter(pl.col("k")==1)                 # nodes with exactly one feasible user
      .join(df_eq, on="id")            # bring that unique user_id
      .join(df_nodes, on="id")          # bring difficulty
      .group_by("user_id").agg(pl.sum("difficulty").alias("excl_load"))
      .join(users.select("user_id","capacity"), on="user_id")
      .with_columns((pl.col("excl_load") - pl.col("capacity")).alias("excl_over"))
      .filter(pl.col("excl_over") > 1e-9)
)


    precheck = pl.DataFrame([{
        "num_users": users.height,
        "num_violations": df_nodes.height,
        "pairs_kept": df_eq.height,
        "nodes_missing": num_missing,
        "LB_budget": LB_budget,
        "sumD": sumD,
        "sumCC": sumCC,
        "capacity_LB_ok": sumD <= sumCC + 1e-12,
        "has_missing_nodes": num_missing > 0,
        "exclusive_overload_count": exclusive_over.height,
    }])

    precheck.write_ipc(f"{dataset}/{metric}_{theta}_precheck.feather", compression="zstd")
    print(precheck.to_dict())
    if num_missing > 0:
        print("Infeasible: some nodes have no feasible user. Example:\n", missing.head(5))

    if exclusive_over.height > 0:
        exclusive_over.write_ipc(OUT/"exclusive_overload.feather", compression="zstd")
        print("Infeasible: exclusive-user overload detected. See exclusive_overload.feather")
        sys.exit(1)   # treat as fatal; remove if you prefer a warning
    else:
        print("Precheck OK. No exclusive-user overload.")

        
    
    
    
    
    
    
    
