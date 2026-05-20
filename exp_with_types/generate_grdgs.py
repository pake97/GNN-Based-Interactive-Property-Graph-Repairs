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
import csv

import argparse
import ast
import json
import os
from typing import Dict, List, Tuple, Any

import polars as pl

def build_edge(row):
    return (row["source"], row["target"], {"weight": row["weight"]})


def load_graph(nodes, edges, dataset):
    grdg = nx.Graph()
    
    
    print(f"Loaded {len(nodes)} nodes from grdg_nodes.csv")
    min_difficulty = nodes.select(pl.col("difficulty")).min()[0, 0]
    max_difficulty = nodes.select(pl.col("difficulty")).max()[0, 0]
    
    alpha=0.505985
    beta=0.195751
    # iterate over each row and print it 
    for row in nodes.iter_rows(named=True):
        
        grdg.add_node(row["id"],violation_type=row['violation_type'],normalized_difficulty=(row['difficulty']-min_difficulty)/(max_difficulty-min_difficulty) if max_difficulty != min_difficulty else 0,difficulty=row['difficulty'],sum_node_conf=row['sum_node_conf'],sum_edge_score=row['sum_edge_score'],real_difficulty=alpha*row['avg_node_confidence']+beta*row['avg_edge_prediction_score'], nodes=json.loads(row["nodes"]), edges=ast.literal_eval(row["edges"]), num_nodes=len(json.loads(row["nodes"])), num_edges=len(ast.literal_eval(row["edges"])), edge_info=ast.literal_eval(row["edge_info_dict"]))
        
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
        
        if(d["difficulty"]<(1-theta)):
            
            #we need the edges whose score is les then theta     
            edges_below_theta = [{edge:d["edge_info"][edge][0]} for edge in d["edge_info"].keys()]
            toDelete = min(data, key=lambda d: next(iter(d.values())))
            
            
            #after deleting the edge, we check if the node is isolated, if yes we delete it
            if d["edge_info"][toDelete.keys()[0]][1]:
                f1_score.append(1)
                gnn_f1_score.append(1)
                #print("deleted correctly")
            else:
                f1_score.append(0)    
                gnn_f1_score.append(0)    
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
    gnn_f1_score = []
    grdg = load_graph(df_nodes, df_edges, dataset)    
    
    print(f"Graph loaded with {grdg.number_of_nodes()} nodes and {grdg.number_of_edges()} edges.")
    
    
    gnn_f1=[]
    gnn_f1_1=[]
    gnn_f1_2=[] 
    gnn_f1_3=[]
    nodes_type_1=[]
    nodes_type_2=[]
    nodes_type_3=[]

    for node,data in grdg.nodes(data=True):
        
        if(grdg.nodes[node].get("violation_type", 1)==1):
                nodes_type_1.append(node)

        if(grdg.nodes[node].get("violation_type", 1)==2):
            nodes_type_2.append(node)

        if(grdg.nodes[node].get("violation_type", 1)==3):
            nodes_type_3.append(node)
    
    lists = [nodes_type_1, nodes_type_2, nodes_type_3]

    with open(f"{dataset}/grdgs/node_types.csv", 'w', newline='') as f:
        writer = csv.writer(f)

        for i, lst in enumerate(lists, start=1):
            for item in lst:
                writer.writerow([item, i])    

    for theta in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]:
        print(theta)
        G=grdg.copy()
        
        
        nodes_to_remove = [
            node for node, attrs in G.nodes(data=True)
            if attrs.get("normalized_difficulty", float("inf")) > theta
        ]
        
        

        nodes_to_remove_1 = [
            node for node, attrs in G.nodes(data=True)
            if attrs.get("normalized_difficulty", float("inf")) > theta and attrs.get("violation_type",1)==1
        ]
        nodes_to_remove_2 = [
            node for node, attrs in G.nodes(data=True)
            if attrs.get("normalized_difficulty", float("inf")) > theta and attrs.get("violation_type",1)==2
        ]
        nodes_to_remove_3 = [
            node for node, attrs in G.nodes(data=True)
            if attrs.get("normalized_difficulty", float("inf")) > theta and attrs.get("violation_type",1)==3
        ]

        for node in nodes_to_remove:
            edge_info = G.nodes[node].get("edge_info", {})

            if not edge_info:
                gnn_f1_score.append(0)
                G.remove_node(node)
                continue

            best_edge, best_tuple = min(
                edge_info.items(),
                key=lambda item: item[1][0]
            )

            is_correct = best_tuple[1]
            gnn_f1.append(1 if is_correct else 0)
            if(G.nodes[node].get("violation_type", 1)==1):
                gnn_f1_1.append(1 if is_correct else 0)


            if(G.nodes[node].get("violation_type", 1)==2):
                gnn_f1_2.append(1 if is_correct else 0)


            if(G.nodes[node].get("violation_type", 1)==3):
                gnn_f1_3.append(1 if is_correct else 0)


            G.remove_node(node)

        print(len(nodes_to_remove))    
        print(np.mean(gnn_f1_1))
        print(np.mean(gnn_f1_2))
        print(np.mean(gnn_f1_3))
        
        rows = []
        for node_id, attrs in G.nodes(data=True):
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
            f.write(f"GNN F1 Score type 1: {np.mean(gnn_f1_1)}\n")
            f.write(f"GNN F1 Score type 2: {np.mean(gnn_f1_2)}\n")
            f.write(f"GNN F1 Score type 3: {np.mean(gnn_f1_3)}\n")
            f.write(f"GNN F1 Score AVG: {np.mean(gnn_f1)}\n")
            f.write(f"Delete nodes: {len(nodes_to_remove)}\n")
            f.write(f"Delete nodes 1: {len(nodes_to_remove_1)}\n")
            f.write(f"Delete nodes 2: {len(nodes_to_remove_2)}\n")
            f.write(f"Delete nodes 3: {len(nodes_to_remove_3)}\n")


    # for theta in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]:
    #     print(theta)
    #     count = sum(
    #         1 for _, attrs in grdg.nodes(data=True)
    #         if attrs.get("difficulty", float("inf")) > theta
    #     )
    #     print(count)
        # if theta == 1.0:
        #     rows = []
        #     for node_id, attrs in grdg.nodes(data=True):
        #         row = {"id": node_id}
        #         for k, v in attrs.items():
        #             # dump lists, tuples, dicts, etc. as JSON text
        #             if isinstance(v, (dict, list, tuple)):
        #                 row[k] = str(v)
        #             else:
        #                 row[k] = v
        #         rows.append(row)
        #     df_nodes = pl.DataFrame(rows)
        #     rows = [(u, v, d.get("weight", 1.0)) for u, v, d in grdg.edges(data=True)]
        #     df_edges = pl.DataFrame(rows, schema=["source", "target", "weight"])
        #     df_nodes.write_ipc(f"{dataset}/grdgs/{theta}_nodes.feather")
        #     df_edges.write_ipc(f"{dataset}/grdgs/{theta}_edges.feather")       
        #     with open (f"{dataset}/grdgs/{theta}_f1.txt", "w") as f:
        #         f.write(f"F1 Score: {np.mean(f1_score)}\n")
        #         f.write(f"GNN F1 Score: {np.mean(gnn_f1_score)}\n")
        #         f.write(f"Delete nodes: {len(f1_score)}\n")
            
        # else:
        #     print("with theta = "+str(theta)+" i should be here")
        #     f1_score, gnn_f1_score = gnn_pass(grdg, theta)
        #     print("and i should have called the pass")
        #     rows = []
        #     for node_id, attrs in grdg.nodes(data=True):
        #         row = {"id": node_id}
        #         for k, v in attrs.items():
        #             # dump lists, tuples, dicts, etc. as JSON text
        #             if isinstance(v, (dict, list, tuple)):
        #                 row[k] = str(v)
        #             else:
        #                 row[k] = v
        #         rows.append(row)

        #     df_nodes = pl.DataFrame(rows)
        #     rows = [(u, v, d.get("weight", 1.0)) for u, v, d in grdg.edges(data=True)]
        #     df_edges = pl.DataFrame(rows, schema=["source", "target", "weight"])
        #     df_nodes.write_ipc(f"{dataset}/grdgs/{theta}_nodes.feather")
        #     df_edges.write_ipc(f"{dataset}/grdgs/{theta}_edges.feather")       
        #     with open (f"{dataset}/grdgs/{theta}_f1.txt", "w") as f:
        #         f.write(f"F1 Score: {np.mean(f1_score)}\n")
        #         f.write(f"GNN F1 Score: {np.mean(gnn_f1_score)}\n")
        #         f.write(f"Delete nodes: {len(f1_score)}\n")
    
    
    
    
    #df.write_ipc(f"{dataset}/users.feather")
    



