

from scipy.optimize import linear_sum_assignment

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
import time
import concurrent.futures
import math
import pygame

warnings.filterwarnings("ignore", category=UserWarning)
random.seed(41)  # For reproducibility

def normalized_difference_similarity(x, y):
    """
    Computes the normalized difference similarity between two scalar values.
    Returns a value between 0 and 1, where 1 means identical and 0 means maximum difference.
    """
    if x == y:
        return 1.0  # Perfect similarity

    max_val = max(abs(x), abs(y))
    if max_val == 0:
        return 1.0  # Both are zero, treat as identical

    similarity = 1 - abs(x - y) / max_val
    return max(0.0, similarity)  # Ensure the result is not negative



def load_user_models(path):
    """
    function to load user models from a pkl file.
    """
    # Load the model from file
    loaded_model = joblib.load(path)

    # X_test = pd.DataFrame({'Skill': [0.8], 'Estimated_Difficulty': [0.8]})
    # # Now you can use it to predict
    # y_pred = loaded_model.predict(X_test)
    # print(y_pred)
    # #['O' 'W' 'S']
    # y_pred_proba = loaded_model.predict_proba(X_test)
    # print(y_pred_proba[0][0])
    return loaded_model



# def generate_users(num_users, min_capacity, max_capacity, min_cost, max_cost):
#     users = []
#     for i in range(num_users):
#         # capacity = random.randint(min_capacity, max_capacity)
#         # # cost must be at least 1 and less than capacity
        
#         # #skills between 0.7 and 1
#         # #skills = random.uniform(0.7, 1)
#         # skills = random.uniform(0, 1)
#         # cost = random.randint(min_cost, min(max_cost, capacity - 1))
#         # efficiency = capacity / cost
        
        
#         capacity = random.randint(min_capacity, max_capacity)
        
#         # Skill in (0.01, 0.99) to avoid logit singularity
#         skill = random.uniform(0.01, 0.99)
        
#         # Logit transformation
#         logit = math.log(skill / (1 - skill))  # can be negative or positive

#         # Normalize logit to a [0, 1] scale using sigmoid
#         scaled_logit = 1 / (1 + math.exp(-logit))  # this returns skill again, but shows how to normalize

#         # Or use positive-only cost: rescale logit to [1, capacity - 1]
#         # Shift and scale logit from [-4.6, 4.6] → [1, capacity - 1]
#         # Note: logit(0.01) ≈ -4.6; logit(0.99) ≈ +4.6

#         normalized_logit = (logit + 4.6) / (2 * 4.6)  # now in [0, 1]
#         cost = int(1 + normalized_logit * (capacity - 2))  # ensure cost ∈ [1, capacity - 1]

#         efficiency = capacity / cost if cost > 0 else 0
#         users.append({
#             'user_id': i,
#             'capacity': capacity,
#             'cost': cost,
#             'efficiency': efficiency,
#             'skills': skill    
#         })
#     return users    
import random
import math


def predict(diff, skill): 
    if (diff-skill)> -0.2:
        return 'O'
    elif (diff-skill)> -0.4 and (diff-skill)<=-0.2:
        return 'S'
    else:
        return 'W'


def generate_users(num_users, min_skill=0.01, max_skill=0.99):
    users = []
    for i in range(num_users):
        # Skill in (0.01, 0.99) to avoid logit extremes
        mu = 0.5     # Mean
        sigma = 0.1  # Standard deviation
        
        skill = random.gauss(mu, sigma)
        
        
        skill = min(max(skill, 0), 1)  # clamp to [0.01, 0.99]
        skill = random.uniform(0,1)  # Random skill between 0.01 and 0.99
        #capacity = random.gauss(20, 10)
        
        #capacity = max(skill, min(capacity, 20*skill))  # clamp to [skill, 20*skill]
        capacity = random.uniform(5, 20)  
    #   # Compute logit-based cost from skill only
    #     logit = math.log(skill / (1 - skill))  # ranges ~[-4.6, 4.6]
    #     normalized_logit = (logit + 4.6) / (2 * 4.6)  # now in [0,1]
    #     raw_cost = 1 + normalized_logit * (max_cost - 1)  # scaled to [1, max_cost]
    #     cost = int(raw_cost)
    #     # Clamp cost to be strictly less than capacity
    #     cost = min(cost, int(capacity) - 1)
    #     cost = max(1, cost)  # ensure cost is at least 1
    #     efficiency = capacity / cost if cost > 0 else 0
    
        # if skill <= 0.5: 
        #     cost = 5
        # elif skill>0.5 and skill< 0.65:
        #     cost = 10
        # elif skill>=0.65 and skill< 0.8:
        #     cost = 20
        # elif skill>=0.8 and skill< 0.9:
        #     cost = 30
        # elif skill>0.9:
        #     cost = 50
        cost = random.randint(min(1,capacity), 20)  # Random cost between 1 and 20
        efficiency = capacity / cost if cost > 0 else 0

        users.append({
            'user_id': i,
            'capacity': round(capacity, 2),
            'cost': cost,
            'efficiency': round(efficiency, 4),
            'skills': round(skill, 4)
        })
    return users




def load_graph(dataset, metric):
    grdg = nx.Graph()
    print(f"Loading graph from dataset: {dataset} with metric: {metric}")
    nodes = pd.read_csv("./data/"+dataset+"/grdg_nodes.csv", low_memory=False)
    print(f"Loaded {len(nodes)} nodes from grdg_nodes.csv")
    
    
    alpha=0.505985
    beta=0.195751
    # iterate over each row and print it 
    for index, row in nodes.iterrows():
        if metric=='difficulty':
            difficulty = row["difficulty"]
        else:
            max_metric = nodes[metric].max()
            min_metric = nodes[metric].min()
            difficulty=(row[metric]-min_metric)/(max_metric-min_metric)
            
        grdg.add_node(row["id"], locked=False, difficulty=difficulty,degree=row['degree'],sum_node_conf=row['sum_node_conf'],sum_edge_score=row['sum_edge_score'],real_difficulty=alpha*row['avg_node_confidence']+beta*row['avg_edge_prediction_score'], nodes=json.loads(row["nodes"]), edges=ast.literal_eval(row["edges"]), num_nodes=len(json.loads(row["nodes"])), num_edges=len(ast.literal_eval(row["edges"])), edge_info=ast.literal_eval(row["edge_info_dict"]))
    # print(f"Loaded nodes into the graph.")
    # edges = pd.read_csv("./data/"+dataset+"/grdg_edges.csv", low_memory=False)
    # # iterate over each row and print it
    
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     edge_list = list(executor.map(build_edge, edges.to_dict(orient="records")))

    # grdg.add_edges_from(edge_list) 
    # # for index, row in edges.iterrows():
    # #     grdg.add_edge(row["source"], row["target"], weight=row["weight"])
    # print(f"Loaded edges into the graph.")
    return grdg

def build_edge(row):
    return (row["source"], row["target"], {"weight": row["weight"]})


def assign_uQAR(G, users, user_model, metric):
    """
    Assigns nodes to users using a greedy, takeover-based algorithm.
    Args:
        G: networkx.Graph with node attribute 'difficulty'
        users: list of dicts, each with 'user_id', 'capacity', 'cost'
    Returns:
        node_assignment: dict {node: user_id}
        total_cost: int
    """

    # Build a mapping from user_id to user info for quick lookup
    user_dict = {user['user_id']: user for user in users}
    user_remaining_capacity = {user['user_id']: user['capacity'] for user in users}
    node_assignment = {node: None for node in G.nodes()}
    print("sorting nodes by degree")
    nodes_sorted = sorted(G.nodes, key=lambda n: G.nodes[n]['difficulty'], reverse=True)
    print("nodes sorted by degree")
    
    
    print("sorting users by efficiency and nodes by metric")
    users_sorted = sorted(users, key=lambda u: u['skills'], reverse=True)
    print("users sorted by efficiency")
    
    
    similarity_matrix = np.zeros((len(nodes_sorted), len(users_sorted)))

    # Build similarity matrix
    for i, task in enumerate(nodes_sorted):
        for j, user in enumerate(users_sorted):
            similarity_matrix[i][j] = normalized_difference_similarity( G.nodes[task]['difficulty'], user['skills'])
    

   

    # similarity_matrix is already built
    cost_matrix = 1 - similarity_matrix  # lower cost = higher similarity
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    assignments = list(zip(row_ind, col_ind))
    for task_i, user_j in assignments:
        node_assignment[nodes_sorted[task_i]] = users_sorted[user_j]['user_id']

    

    # Compute total cost
    user_node_counts = {user['user_id']: 0 for user in users}
    for assigned_user in node_assignment.values():
        if assigned_user is not None:
            user_node_counts[assigned_user] += 1

    total_cost = 0
    for user_id, count in user_node_counts.items():
        cost = user_dict[user_id]['cost']
        total_cost += count * cost
    
        # Compute edge cut: edges that connect nodes assigned to different users
    edge_cut = 0
    for u, v in G.edges():
        uid_u = node_assignment[u]
        uid_v = node_assignment[v]
        if uid_u is not None and uid_v is not None and uid_u != uid_v:
            edge_cut += 1

    return node_assignment, total_cost, edge_cut




def gnn_pass(graph, theta, f1_score=[], gnn_f1_score=[]):
    to_delete = []
    for n,d in graph.nodes(data=True):
        deleted = False
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
            to_delete.append(n)
    graph.remove_nodes_from(to_delete)
    return f1_score, gnn_f1_score



def remove_nodes_with_low_difficulty(graph, mindiff):
    """
    Removes nodes with difficulty below a certain threshold.
    Args:
        graph: networkx.Graph
        mindiff: float, minimum difficulty threshold
    Returns:
        networkx.Graph with low-difficulty nodes removed
    """
    to_remove = [n for n, d in graph.nodes(data=True) if d['real_difficulty'] < mindiff]
    graph.remove_nodes_from(to_remove)
    print(f"Removed {len(to_remove)} nodes with difficulty < {mindiff}")
    return graph


def main(theta, users, dataset, user_model, metric, mindiff):
    f1_score = []
    gnn_f1_score = []
    users_f1_score = []
    
    print("Loading graph from dataset:", dataset)
    grdg = load_graph(dataset, metric)    
    print("Graph loaded successfully.")
    if mindiff>0:
        grdg = remove_nodes_with_low_difficulty(grdg, mindiff)
    print(f"Generating {users} users with capacity between 5 and 50, cost between 1 and 5")
    #users = generate_users(users, 2, 5, 1, 5)
    users = generate_users(grdg.number_of_nodes())
    print(f"Generated {len(users)} users.")

    if theta<1:
        t0 = time.time()
        f1_score, gnn_f1_score=gnn_pass(grdg, theta, f1_score, gnn_f1_score)
        t1 = time.time()
        print(f"GNN pass took {t1-t0} seconds")
    node_assignment = {}
    edge_cut=grdg.number_of_edges()
    total_cost=0
    
    print(f"Graph has {grdg.number_of_nodes()} nodes and {grdg.number_of_edges()} edges")
    t0 = time.time()
    if(len(grdg.nodes)<=len(users)):
        node_assignment, total_cost, edge_cut = assign_uQAR(grdg, users, user_model, "difficulty")
    t1 = time.time()
    print(f"Greedy assignment took {t1-t0} seconds")  
    
    print(f"Assigned: {len(set(node_assignment.values()))} of the {len(users)} users")
    print("AU:", len(set(node_assignment.values())))
    print("Edge cut:", edge_cut)
    print("Total cost:", total_cost)
    



    f1_map = {0:1,2: 0,1: 0.66,3: 0.33}   
    #f1_map = {"O":1,"W": 0,"SS": 0.66,"S": 0.33}   

    assignments = {'assignment':[]}
    for node in node_assignment.keys():
        res = f1_map[user_model.predict([[users[node_assignment[node]]['skills'], grdg.nodes[node]['real_difficulty']]])[0]]
        assignments['assignment'].append({
            'difficulty':grdg.nodes[node]['difficulty'],
            'real_difficulty': grdg.nodes[node]['real_difficulty'],
            'skill': users[node_assignment[node]]['skills'],
            'f1_score': res
        })
        
        #res = f1_map[predict(grdg.nodes[node]['real_difficulty'], users[node_assignment[node]]['skills'])]
        users_f1_score.append(res)
        f1_score.append(res)
    
    df = pd.DataFrame(assignments['assignment'])
    df.to_csv(f"{metric}assignment.csv", index=False)
    print("F1 Score:", np.mean(f1_score) if len(f1_score) > 0 else 'nan')
    print("GNN F1 Score:", np.mean(gnn_f1_score) if len(gnn_f1_score) > 0 else 'nan')
    print("Users F1 Score:", np.mean(users_f1_score) if len(users_f1_score) > 0 else 'nan') 
    


    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process arguments for theta, users, and dataset.")

    parser.add_argument('--theta', type=float, required=True, help='Theta value (float)')
    parser.add_argument('--users', type=int, required=True, help='Number of users (int)')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name or path (string)')
    parser.add_argument('--metric', type=str, default='Difficulty', help='Difficulty Measure')
    parser.add_argument('--mindiff', type=str, help='Min difficulty of violations')
    args = parser.parse_args()

    # Access the arguments
    theta = args.theta
    users = args.users
    dataset = args.dataset
    metric = args.metric
    mindiff = float(args.mindiff)
    print(f"Loading user model from super_repair_type_model.pkl")
    user_model = load_user_models('super_repair_type_model.pkl')
    print(f"Model loaded successfully.")

    print(f"Theta: {theta}")
    print(f"Users: {users}")
    print(f"Dataset: {dataset}")
    print(f"Difficulty Metric: {metric}")
    main(theta, users, dataset, user_model, metric, mindiff)


    # pygame.mixer.init()
    # pygame.mixer.music.load("mixkit-winning-a-coin-video-game-2069.wav")
    # pygame.mixer.music.play()

    # while pygame.mixer.music.get_busy():
    #     pygame.time.Clock().tick(10)


