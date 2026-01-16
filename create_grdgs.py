import argparse
import joblib
import json
import ast
import pandas as pd
import networkx as nx
import ast 
import networkx as nx
import random
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
random.seed(41)  # For reproducibility






def load_graph(dataset):
    grdg = nx.Graph()
    nodes = pd.read_csv("./data/"+dataset+"/grdg_nodes.csv")
    # iterate over each row and print it 
    for index, row in nodes.iterrows():
        grdg.add_node(row["id"], difficulty=row["difficulty"],real_difficulty=row['real_difficulty'],degree=row['degree'],pagerank=row['pagerank'], nodes=json.loads(row["nodes"]), edges=ast.literal_eval(row["edges"]), num_nodes=len(json.loads(row["nodes"])), num_edges=len(ast.literal_eval(row["edges"])), edge_info=ast.literal_eval(row["edge_info_dict"]))
        
    edges = pd.read_csv("./data/"+dataset+"/grdg_edges.csv")
    # iterate over each row and print it 
    for index, row in edges.iterrows():
        grdg.add_edge(row["source"], row["target"], weight=row["weight"])
    return grdg




def gnn_pass(graph, theta, f1_score=[], gnn_f1_score=[]):
    to_delete = []
    for n,d in graph.nodes(data=True):
        deleted = False
        for edge in d["edge_info"].keys():
            print(d["edge_info"][edge])
            if d["edge_info"][edge][0]>theta:
                deleted = True
                if d["edge_info"][edge][1]:
                    f1_score.append(1)
                    gnn_f1_score.append(1)
                    print("deleted correctly")
                else:
                    f1_score.append(0)    
                    gnn_f1_score.append(0)    
        if(deleted):
            to_delete.append(n)
    graph.remove_nodes_from(to_delete)
    return f1_score, gnn_f1_score


def main(theta, users, dataset):
    f1_score = []
    gnn_f1_score = []

    
    grdg = load_graph(dataset)    

    f1_score, gnn_f1_score=gnn_pass(grdg, theta, f1_score, gnn_f1_score)

    nodes_data = []
    for node, data in grdg.nodes(data=True):
        row = {"id": node, "type": data.get("type", "unknown")}
        row.update(data)
        nodes_data.append(row)

    edges_data = []
    for u, v, data in grdg.edges(data=True):
        row = {"source": u, "target": v}
        row.update(data)
        edges_data.append(row)

    nodes_df = pd.DataFrame(nodes_data)
    edges_df = pd.DataFrame(edges_data)

    nodes_df.to_csv("./data/"+dataset+"/"+str(theta)+"/grdg_nodes.csv", index=False)
    edges_df.to_csv("./data/"+dataset+"/"+str(theta)+"/grdg_edges.csv", index=False)
    


    
    

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process arguments for theta, users, and dataset.")

    parser.add_argument('--theta', type=float, required=True, help='Theta value (float)')
    parser.add_argument('--users', type=int, required=True, help='Number of users (int)')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name or path (string)')

    args = parser.parse_args()

    # Access the arguments
    theta = args.theta
    users = args.users
    dataset = args.dataset




    print(f"Theta: {theta}")
    print(f"Users: {users}")
    print(f"Dataset: {dataset}")
    main(theta, users, dataset)



