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


import argparse
import ast
import json
import os
from typing import Dict, List, Tuple, Any

import polars as pl



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process arguments for theta, users, and dataset.")
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name or path (string)')
    #parser.add_argument('--metric', type=str, required=True, help='Dataset name or path (string)')
    
    args = parser.parse_args()
    
    dataset = args.dataset
    #metric = args.metric
    users = pl.read_ipc(f"./{dataset}/users.feather")
    
    user_model = joblib.load('super_repair_type_model.pkl')
            
    
    # # Read CSV with polars
    # df_nodes = pl.read_csv(f"./data/{dataset}/grdg_nodes.csv", low_memory=False, null_values=["None", "null", "NaN", "nan", "N/A", "NA", ""])

    
    # grdg = load_graph(df_nodes, df_edges)
    for metric in ['difficulty', 'normalized_cs_cl', 'normalized_pagerank', 'normalized_degree']:
        for theta in [1.0]:
            violations = pl.read_ipc(f"{dataset}/grdgs/{theta}_nodes.feather")  
            users_violations = []
            for v in violations.iter_rows(named=True):
                violation_difficulty = v[metric]
                for u in users.iter_rows(named=True):
                    skill = u['skills']
                    #EQ = np.sum(user_model.predict_proba([[skill, violation_difficulty]])[0]*np.array([1,0.66,0.5,0]))
                    EQ = skill - violation_difficulty
                    users_violations.append({
                        'user_id': u['user_id'],
                        'violation_id': v['id'],
                        'eq': EQ,
                    })
            
            # i need a df that is a matrix : users x violations with the eq as value
            df_users_violations = pl.DataFrame(users_violations)
            df_users_violations = df_users_violations.pivot(index='user_id', columns='violation_id', values='eq')
            df_users_violations = df_users_violations.fill_null(0.0)
            df_users_violations = df_users_violations.with_columns(pl.col(pl.Int64).cast(pl.Float64))
            df_users_violations = df_users_violations.with_columns(pl.col(pl.Float64).round(4))
            df_users_violations.write_ipc(f"{dataset}/eq_{metric}_eqs.feather")            
    

    



