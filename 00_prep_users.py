# prep_users.py
import polars as pl
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process arguments for theta, users, and dataset.")
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name or path (string)')
    
    
    
    
    args = parser.parse_args()
    
    dataset = args.dataset
    
    
    
    
        
    

    users = pl.read_ipc(f"{dataset}/users.feather", memory_map=False).select(
            "user_id","skills","cost","capacity"
        ).sort("skills")  # ascending by skill

    # suffix argmin of cost; keep argmin index and value
    c = users["cost"].to_list()
    n = len(c)
    suf_min_idx = [0]*n
    suf_min_val = [0.0]*n
    best_i, best_v = n-1, c[-1]
    suf_min_idx[-1] = best_i; suf_min_val[-1] = float(best_v)
    for i in range(n-2, -1, -1):
        if c[i] <= best_v:
            best_i, best_v = i, c[i]
        suf_min_idx[i] = best_i
        suf_min_val[i] = float(best_v)

    users = users.with_columns([
        pl.Series("suf_min_idx", suf_min_idx),
        pl.Series("suf_min_cost", suf_min_val),
    ])
    users.write_ipc(f"{dataset}/users_sorted.feather", compression="zstd")
