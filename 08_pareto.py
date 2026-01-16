import numpy as np
import pandas as pd
from typing import Iterable, List, Tuple, Union, Sequence

def pareto_front(
    df: pd.DataFrame,
    objectives: Sequence[str],
    maximize: Union[bool, Sequence[bool]] = False,
    return_indices: bool = True,
    chunk_size: int = 4096,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the Pareto-optimal (non-dominated) rows of `df` over given `objectives`.

    Parameters
    ----------
    df : pd.DataFrame
        Source data.
    objectives : sequence of str
        Column names to optimize.
    maximize : bool or sequence of bool, default False
        If a single bool: applies to all objectives.
        If a sequence: per-objective True means maximize that objective, False means minimize.
    return_indices : bool, default True
        Whether to also return the integer indices (positional) of the Pareto-optimal rows.
    chunk_size : int, default 4096
        Size of comparison blocks to limit memory during pairwise checks.

    Returns
    -------
    mask : np.ndarray (bool, shape (n_rows,))
        True for Pareto-optimal rows, False otherwise.
    idx  : np.ndarray (int) 
        Positional indices of Pareto-optimal rows (only if return_indices=True).

    Notes
    -----
    - Definition used: row A dominates row B if A is *no worse* than B in all objectives
      and *strictly better* in at least one objective, considering min/max senses.
    - Handles ties gracefully (equal points are mutually non-dominating).
    - Works for any number of objectives (>=1).
    - Complexity is O(n^2) worst case but implemented with chunking to avoid
      allocating a full n×n matrix; suitable for thousands to low tens of thousands of rows.
    """

    if len(objectives) == 0:
        raise ValueError("Provide at least one objective column.")

    # Build matrix X (n, m)
    X = df.loc[:, objectives].to_numpy()
    n, m = X.shape

    # Normalize direction: convert all objectives to a *minimize* problem.
    if isinstance(maximize, bool):
        maximize = [maximize] * m
    maximize = np.asarray(maximize, dtype=bool)
    if maximize.size != m:
        raise ValueError("Length of `maximize` must match number of objectives.")
    # Flip sign for maximize so that lower is always better
    X = np.where(maximize, -X, X)

    # Keep a candidate set; we'll mark dominated points as False
    is_candidate = np.ones(n, dtype=bool)

    # Optional: lexicographic pre-sort to put 'good' points earlier → a little pruning
    order = np.lexsort(tuple(X[:, k] for k in range(m - 1, -1, -1)))
    X_sorted = X[order]

    # For mapping back to original positions
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(n)

    # Streaming pairwise dominance check in chunks
    for i in range(n):
        if not is_candidate[order[i]]:
            continue  # already dominated

        xi = X_sorted[i]  # shape (m,)

        # Compare xi against only *later* points (j > i) — earlier points cannot be dominated by xi
        # We'll process in blocks to limit memory.
        start = i + 1
        while start < n:
            end = min(start + chunk_size, n)
            block = X_sorted[start:end]  # shape (b, m)

            # Dominance test: xi <= block in all dims, and xi < block in any dim
            not_worse_all = np.all(xi <= block, axis=1)
            strictly_better_any = np.any(xi < block, axis=1)
            dominated = not_worse_all & strictly_better_any

            # Mark dominated (map back to original indices)
            if np.any(dominated):
                dominated_positions = order[start:end][dominated]
                is_candidate[dominated_positions] = False

            start = end

        # Early exit: if remaining candidates count (after i) is zero, we can break
        # (Cheap check: if no True after position i+1 in sorted space)
        if not np.any(is_candidate[order[i+1:]]):
            break

    mask = is_candidate
    # mask is in original order; ensure we only keep non-dominated
    if return_indices:
        idx = np.flatnonzero(mask)
        return mask, idx
    return mask, np.array([], dtype=int)


# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    # Toy data: minimize "cost", maximize "quality", minimize "time"
    df = pd.read_csv("faers/exp2_quality_summary.csv")
    
    # #dataset,metric,theta,budget,avg_f1,num_repairs,total_cost,unique_users,combined_f1
    mask, idx = pareto_front(df, objectives=["combined_f1","total_cost","unique_users"], maximize=[True, False, False])
    pareto_df = df.loc[mask].copy()

    print("Pareto indices:", idx.tolist())
    print(pareto_df.sort_values(["combined_f1","total_cost","unique_users"]).to_string(index=False))



    df1 = pd.read_csv("finbench/exp2_quality_summary.csv")
    
     
    mask, idx = pareto_front(df1, objectives=["combined_f1","total_cost","unique_users"], maximize=[True, False, False])
    pareto_df = df1.loc[mask].copy()

    print("Pareto indices:", idx.tolist())
    print(pareto_df.sort_values(["combined_f1","total_cost","unique_users"]).to_string(index=False))


    df2 = pd.read_csv("icij/exp2_quality_summary.csv")
    
    mask, idx = pareto_front(df2, objectives=["combined_f1","total_cost","unique_users"], maximize=[True, False, False])
    pareto_df = df2.loc[mask].copy()

    print("Pareto indices:", idx.tolist())
    print(pareto_df.sort_values(["combined_f1","total_cost","unique_users"]).to_string(index=False))
    
    
    df3 = pd.read_csv("snb/exp2_quality_summary.csv")
    
    mask, idx = pareto_front(df3, objectives=["combined_f1","total_cost","unique_users"], maximize=[True, False, False])
    pareto_df = df3.loc[mask].copy()

    print("Pareto indices:", idx.tolist())
    print(pareto_df.sort_values(["combined_f1","total_cost","unique_users"]).to_string(index=False))
    
    


    dff = pd.concat([df, df1, df2, df3], ignore_index=True)


    #dataset,metric,theta,budget,avg_f1,num_repairs,total_cost,unique_users,combined_f1
    mask, idx = pareto_front(dff, objectives=["combined_f1","total_cost","unique_users"], maximize=[True, False, False])
    pareto_df = dff.loc[mask].copy()

    print("Pareto indices:", idx.tolist())
    print(pareto_df.sort_values(["combined_f1","total_cost","unique_users"]).to_string(index=False))
    print(pareto_df.sort_values(["theta"]).to_string(index=False))
