import polars as pl
import argparse
import pickle
import joblib
import json
import ast
import pandas as pd
import networkx as nx
import random
import numpy as np
import warnings
import time
import concurrent.futures
import math






    



def generate_users(num_users, min_skill=0.01, max_skill=0.99):
    users = []
    for i in range(num_users):
        skill = random.uniform(0,1)  # Random skill between 0.01 and 0.99
        capacity = random.uniform(5, 20)  

        # higher the skill, higher the cost using a linear relationship and no random         
        cost = 1 + skill * (20 - 1)
        #
        normalized_cost = (cost - 1) / (20 - 1)

        #cost = random.randint(min(1,capacity), 20)  # Random cost between 1 and 20
        
        
        efficiency = capacity / cost if cost > 0 else 0

        users.append({
            'user_id': i,
            'capacity': round(capacity, 2),
            'cost': normalized_cost,
            'efficiency': round(efficiency, 4),
            'skills': round(skill, 4)
        })
    return users



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process arguments for theta, users, and dataset.")
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name or path (string)')
    
    args = parser.parse_args()
    
    dataset = args.dataset
    
    # Read CSV with polars
    df = pl.read_csv(f"./data/{dataset}/grdg_nodes.csv", low_memory=False, null_values=["None", "null", "NaN", "nan", "N/A", "NA", ""])
    users = generate_users(df.shape[0])

    # Convert to Polars DataFrame
    users_df = pl.DataFrame(users)
    
    users_df.write_ipc(f"{dataset}/users.feather")
    

    # print(f"Saved Polars DataFrame to {args.output_pkl}")


