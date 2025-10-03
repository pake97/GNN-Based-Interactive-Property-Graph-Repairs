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

def build_edge(row):
    return (row["source"], row["target"], {"weight": row["weight"]})


def load_graph(nodes, edges):
    grdg = nx.Graph()
    
    
    print(f"Loaded {len(nodes)} nodes from grdg_nodes.csv")
    
    
    alpha=0.505985
    beta=0.195751
    # iterate over each row and print it 
    for row in nodes.iter_rows(named=True):
        min_cl = nodes['cs_cl'].min()
        max_cl = nodes['cs_cl'].max()
        min_degree = nodes['degree'].min()
        max_degree = nodes['degree'].max()
        min_pr = nodes['pagerank'].min()
        max_pr = nodes['pagerank'].max()
        
        num_nodes = len(json.loads(row["nodes"]))
        num_edges = len(ast.literal_eval(row["edges"]))
         
        
        grdg.add_node(row["id"], difficulty=row['difficulty'],degree=row['degree'],sum_node_conf=row['sum_node_conf'],sum_edge_score=row['sum_edge_score'],real_difficulty=alpha*row['avg_node_confidence']+beta*row['avg_edge_prediction_score'], nodes=json.loads(row["nodes"]), edges=ast.literal_eval(row["edges"]), num_nodes=len(json.loads(row["nodes"])), num_edges=len(ast.literal_eval(row["edges"])), edge_info=ast.literal_eval(row["edge_info_dict"]), normalized_cs_cl=(row['cs_cl'] - min_cl) / (max_cl - min_cl) if max_cl != min_cl else 0, normalized_degree=(row['degree'] - min_degree) / (max_degree - min_degree) if max_degree != min_degree else 0, normalized_pagerank=(row['pagerank'] - min_pr) / (max_pr - min_pr) if max_pr != min_pr else 0)
        
    print(f"Loaded nodes into the graph.")
    
    # iterate over each row and print it
    
    edge_iter = edges.iter_rows(named=True)
    edge_list = [build_edge(row) for row in edge_iter]

    grdg.add_edges_from(edge_list)
    # for index, row in edges.iterrows():
    #     grdg.add_edge(row["source"], row["target"], weight=row["weight"])
    print(f"Loaded edges into the graph.")
    return grdg

def gnn_pass(graph, theta, f1_score=[], gnn_f1_score=[]):
    
    G = graph.copy()
    for n,d in G.nodes(data=True):
        deleted = False
        if(len(d["edge_info"])==0):
            graph.remove_node(n)
        for edge in d["edge_info"].keys():
            #print(d["edge_info"][edge])
            if d["edge_info"][edge][0]>theta:
                deleted = True
                if d["edge_info"][edge][1]:
                    f1_score.append(1)
                    gnn_f1_score.append(1)
                    #print("deleted correctly")
                else:
                    f1_score.append(0)    
                    gnn_f1_score.append(0)    
        if(deleted):
            graph.remove_node(n)
    return f1_score, gnn_f1_score




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process arguments for theta, users, and dataset.")
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name or path (string)')
    
    args = parser.parse_args()
    
    dataset = args.dataset
    
    # Read CSV with polars
    df_nodes = pl.read_csv(f"./data/{dataset}/grdg_nodes.csv", low_memory=False, null_values=["None", "null", "NaN", "nan", "N/A", "NA", ""])
    df_edges = pl.read_csv(f"./data/{dataset}/grdg_edges.csv", low_memory=False, null_values=["None", "null", "NaN", "nan", "N/A", "NA", ""])
    
    

    f1_score = []
    grdg = load_graph(df_nodes, df_edges)
    for theta in [1.0]:
        if theta == 1.0:
            rows = []
            for node_id, attrs in grdg.nodes(data=True):
                row = {"id": node_id}
                for k, v in attrs.items():
                    # dump lists, tuples, dicts, etc. as JSON text
                    if isinstance(v, (dict, list, tuple)):
                        row[k] = str(v)
                    else:
                        row[k] = v
                rows.append(row)
            df_nodes = pl.DataFrame(rows)
            rows = [(u, v, d.get("weight", 1.0)) for u, v, d in grdg.edges(data=True)]
            df_edges = pl.DataFrame(rows, schema=["source", "target", "weight"])
            df_nodes.write_ipc(f"{dataset}/grdgs/{theta}_nodes.feather")
            df_edges.write_ipc(f"{dataset}/grdgs/{theta}_edges.feather")       
            with open (f"{dataset}/grdgs/{theta}_f1.txt", "w") as f:
                f.write(f"F1 Score: {np.mean(f1_score)}\n")
                f.write(f"GNN F1 Score: {np.mean(gnn_f1_score)}\n")
                f.write(f"Delete nodes: {len(f1_score)}\n")
            
        else:
            f1_score, gnn_f1_score = gnn_pass(grdg, theta)
            rows = []
            for node_id, attrs in grdg.nodes(data=True):
                row = {"id": node_id}
                for k, v in attrs.items():
                    # dump lists, tuples, dicts, etc. as JSON text
                    if isinstance(v, (dict, list, tuple)):
                        row[k] = str(v)
                    else:
                        row[k] = v
                rows.append(row)

            df_nodes = pl.DataFrame(rows)
            rows = [(u, v, d.get("weight", 1.0)) for u, v, d in grdg.edges(data=True)]
            df_edges = pl.DataFrame(rows, schema=["source", "target", "weight"])
            df_nodes.write_ipc(f"{dataset}/grdgs/{theta}_nodes.feather")
            df_edges.write_ipc(f"{dataset}/grdgs/{theta}_edges.feather")       
            with open (f"{dataset}/grdgs/{theta}_f1.txt", "w") as f:
                f.write(f"F1 Score: {np.mean(f1_score)}\n")
                f.write(f"GNN F1 Score: {np.mean(gnn_f1_score)}\n")
                f.write(f"Delete nodes: {len(f1_score)}\n")
    
    
    
    
    #df.write_ipc(f"{dataset}/users.feather")
    



