from dataclasses import dataclass
from typing import Dict, List, Set, Callable, Tuple, Optional
from heapq import nlargest
from collections import deque
import argparse
import polars as pl
import numpy as np
import joblib
import warnings
import json
warnings.filterwarnings("ignore", message="X does not have valid feature names")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process arguments for theta, users, and dataset.")
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name or path (string)')
    parser.add_argument('--theta', type=float, required=True, help='Theta value (float)')
    parser.add_argument('--budget', type=float, required=True, help='DIfficulty metric')
    parser.add_argument('--metric', type=str, required=True, help='DIfficulty metric')
    
    
    args = parser.parse_args()
    
    dataset = args.dataset
    theta = args.theta
    budget = args.budget
    metric = args.metric
    
    # Read CSV with polars
    df_nodes = pl.read_ipc(f"{dataset}/grdgs/{str(theta)}_nodes.feather")
    user_model = joblib.load('super_repair_type_model.pkl')
    df_users = pl.read_ipc(f"./{dataset}/users.feather")
    
    
    df_file = pl.read_ipc(f"{dataset}/gap_{metric}_{str(theta)}_{str(budget)}/best_assign.feather")
    types_df = pl.read_csv(
        f"{dataset}/grdgs/node_types.csv",
        has_header=False,
        new_columns=["id", "type"]
    )

    df_joined = df_file.join(types_df, on="id", how="left")
    
    f1_map = {0:1,2: 0.5,1: 0.66,3: 0}  
    f1 = []
    f1_1=[]    
    f1_2=[]
    f1_3=[]

    for assignment in df_joined.iter_rows(named=True):
        user = assignment['user_id']
        node = assignment['id']
        cost = assignment['cost']
        real_difficulty = df_nodes.filter(pl.col("id")==node).select('real_difficulty').to_numpy()[0][0]
        
        
        skill = df_users.filter(pl.col("user_id")==user).select("skills").to_numpy()[0][0]        
            
        res = f1_map[user_model.predict([[skill, real_difficulty]])[0]]
        f1.append(res)
        if(assignment['type']==1):
            f1_1.append(res)
        if(assignment['type']==2):
            f1_2.append(res)
        if(assignment['type']==3):
            f1_3.append(res)
    unique_users = df_file.select("user_id").unique()
    total_cost = df_file.select(pl.col("cost").sum()).to_numpy()[0][0] 
    
    tmp = df_joined.filter(pl.col("type") == 1)

    unique_users_1 = tmp.select(pl.col("user_id").n_unique()).item()
    total_cost_1 = tmp.select(pl.col("cost").sum()).item()

    tmp = df_joined.filter(pl.col("type") == 2)

    unique_users_2 = tmp.select(pl.col("user_id").n_unique()).item()
    total_cost_2 = tmp.select(pl.col("cost").sum()).item()
    tmp = df_joined.filter(pl.col("type") == 3)

    unique_users_3 = tmp.select(pl.col("user_id").n_unique()).item()
    total_cost_3 = tmp.select(pl.col("cost").sum()).item()
    results = {
    "avg_f1": float(np.mean(f1)),
    "max_f1": float(np.max(f1)),
    "min_f1": float(np.min(f1)),
    "std_f1": float(np.std(f1)),
    "num_repairs": int(len(f1)),
    "1_avg_f1": float(np.mean(f1_1)),
    "1_max_f1": float(np.max(f1_1)),
    "1_min_f1": float(np.min(f1_1)),
    "1_std_f1": float(np.std(f1_1)),
    "1_num_repairs": int(len(f1_1)),
    "2_avg_f1": float(np.mean(f1_2)),
    "2_max_f1": float(np.max(f1_2)),
    "2_min_f1": float(np.min(f1_2)),
    "2_std_f1": float(np.std(f1_2)),
    "2_num_repairs": int(len(f1_2)),
    "3_avg_f1": float(np.mean(f1_3)),
    "3_max_f1": float(np.max(f1_3)),
    "3_min_f1": float(np.min(f1_3)),
    "3_std_f1": float(np.std(f1_3)),
    "3_num_repairs": int(len(f1_3)),
    "total_cost": float(total_cost),
    "unique_users": int(unique_users.height),
    "1_total_cost": float(total_cost_1),
    "1_unique_users": int(unique_users_1),
    "2_total_cost": float(total_cost_2),
    "2_unique_users": int(unique_users_2),
    "3_total_cost": float(total_cost_3),
    "3_unique_users": int(unique_users_3)
    
    }

    with open(f"{dataset}/gap_{metric}_{str(theta)}_{str(budget)}/quality_report.json", "w") as f:
        json.dump(results, f, indent=4)
    
    
    