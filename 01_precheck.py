

import polars as pl
import argparse
import sys

def EQ(skill, difficulty):
    if skill - difficulty > -0.2:
        return 1
    elif skill - difficulty > -0.4 and skill - difficulty <= -0.2:
        return 0.66
    elif skill - difficulty > -0.6 and skill - difficulty <= -0.4:
        return 0.5
    else:
        return 0
    
    
def EQ_expr(skill_col: str = "skill", diff_col: str = "difficulty"):
    import polars as pl
    z = pl.col(skill_col) - pl.col(diff_col)
    return (
        pl.when(z > -0.2).then(pl.lit(1.0))
        .when((z > -0.4) & (z <= -0.2)).then(pl.lit(0.66))
        .when((z > -0.6) & (z <= -0.4)).then(pl.lit(0.5))
        .otherwise(pl.lit(0.0))
    )

    
    



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
    sumD  = float(df_nodes[metric].sum())
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

        
    
    
    
    
    
    
    
