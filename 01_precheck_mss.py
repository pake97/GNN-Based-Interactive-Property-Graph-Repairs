

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

    
    

# 02_inner_pairfree.py
import argparse, math
from bisect import bisect_left, bisect_right
from pathlib import Path
import polars as pl

def assign_per_node_pairfree(users_sorted: pl.DataFrame,
                             violations: pl.DataFrame,
                             lam: float, metric: str = "difficulty") -> pl.DataFrame:
    """
    users_sorted: asc by skill; has columns:
        user_id, skill, c, CC, suf_min_idx, suf_min_cost
    violations: violation_id, difficulty
    Returns per-node assignment: violation_id, user_id, p, c, difficulty, score
    """
    s = users_sorted["skills"].to_list()
    c = users_sorted["cost"].to_list()
    uid = users_sorted["user_id"].to_list()
    suf_idx = users_sorted["suf_min_idx"].to_list()
    suf_cost = users_sorted["suf_min_cost"].to_list()
    n = len(s)

    rows = []
    for v, d in zip(violations["id"].to_list(),
                    violations[metric].to_list()):
        # thresholds
        tH  = d - 0.2
        tM1 = d - 0.4
        tM2 = d - 0.6

        # indices in skill (first index with skill >= τ)
        iH  = bisect_left(s, tH)
        iM1 = bisect_left(s, tM1)
        iM2 = bisect_left(s, tM2)

        # band H: [iH, n)
        best = None
        if iH < n:
            j = suf_idx[iH]; cost = suf_cost[iH]; p = 1.0
            score = p - lam*cost
            best = (score, p, cost, uid[j])

        # band M1: [iM1, iH) (only if iM1 < iH)
        if iM1 < iH:
            # min cost in a finite subarray -> take the min of suf_min_cost at iM1,
            # but if its argmin index >= iH, we need a scan. To stay memory-safe,
            # do a tiny linear scan over this short range; typical bands are small.
            j0 = suf_idx[iM1]; c0 = suf_cost[iM1]
            if j0 < iH:
                cost, jmin = c0, j0
            else:
                # fallback scan over [iM1, iH)
                cost, jmin = c[iM1], iM1
                for k in range(iM1+1, iH):
                    if c[k] < cost:
                        cost, jmin = c[k], k
            pM1 = 0.66; score = pM1 - lam*cost
            if (best is None) or (score > best[0]):
                best = (score, pM1, cost, uid[jmin])

        # band M2: [iM2, iM1)
        if iM2 < iM1:
            j0 = suf_idx[iM2]; c0 = suf_cost[iM2]
            if j0 < iM1:
                cost, jmin = c0, j0
            else:
                cost, jmin = c[iM2], iM2
                for k in range(iM2+1, iM1):
                    if c[k] < cost:
                        cost, jmin = c[k], k
            pM2 = 0.5; score = pM2 - lam*cost
            if (best is None) or (score > best[0]):
                best = (score, pM2, cost, uid[jmin])

        # If no band qualifies (all thresholds beyond max skill), fallback: no p>0 -> choose none.
        # But with your data, s spans [0,1], so iM2 < n usually holds.
        if best is None:
            # assign to cheapest overall as a last resort (p=0)
            j = suf_idx[0]; cost = suf_cost[0]; p = 0.0
            score = - lam*cost
            best = (score, p, cost, uid[j])

        score, p, cost, u = best
        rows.append((v, u, p, cost, float(d), float(score)))

    return pl.DataFrame(rows,
                        schema=["id","user_id","p","cost",metric,"score"],
                        orient="row")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name or path (string)')
    parser.add_argument('--theta', type=float, required=True, help='Theta')
    parser.add_argument('--metric', type=str, required=True, help='DIfficulty metric')
    parser.add_argument("--lam", type=float, default=0.0)
    args = parser.parse_args()
    
    dataset = args.dataset
    theta = args.theta
    metric = args.metric
    lam = args.lam
    users_sorted = pl.read_ipc(f"{dataset}/users_sorted.feather", memory_map=False)
    violations   = pl.read_ipc(f"{dataset}/grdgs/{str(theta)}_nodes.feather", memory_map=False)
    

    assign = assign_per_node_pairfree(users_sorted, violations,lam, metric)
    outpath = f"./{dataset}/run_{metric}_{str(theta)}_pre/"
    outdir = Path(outpath); outdir.mkdir(parents=True, exist_ok=True)

    assign.write_ipc(outdir/ 'pre.feather', compression="zstd")
    print(assign.shape)

if __name__ == "__main__":
    main()