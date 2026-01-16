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



def generate_users(num_users, min_capacity, max_capacity, min_cost, max_cost):
    users = []
    for i in range(num_users):
        capacity = random.randint(min_capacity, max_capacity)
        # cost must be at least 1 and less than capacity
        cost = random.randint(min_cost, min(max_cost, capacity - 1))
        #skills between 0.7 and 1
        skills = random.uniform(0.7, 1)
        efficiency = capacity / cost
        users.append({
            'user_id': i,
            'capacity': capacity,
            'cost': cost,
            'efficiency': efficiency,
            'skills': skills    
        })
    return users    



def load_graph(dataset):
    grdg = nx.Graph()
    nodes = pd.read_csv("./data/"+dataset+"/grdg_nodes.csv")
    # iterate over each row and print it 
    for index, row in nodes.iterrows():
        
        grdg.add_node(row["id"], difficulty=row["difficulty"], nodes=json.loads(row["nodes"]), edges=ast.literal_eval(row["edges"]), num_nodes=len(json.loads(row["nodes"])), num_edges=len(ast.literal_eval(row["edges"])), edge_info=ast.literal_eval(row["edge_info_dict"]))
        
    edges = pd.read_csv("./data/"+dataset+"/grdg_edges.csv")
    # iterate over each row and print it 
    for index, row in edges.iterrows():
        grdg.add_edge(row["source"], row["target"], weight=row["weight"])
    return grdg



def uQAR_degree_assignment(G, users, user_model):
    """
    Assigns nodes to users by ascending degree, 
    """

    # Build a mapping from user_id to user info for quick lookup
    user_dict = {user['user_id']: user for user in users}
    user_remaining_capacity = {user['user_id']: user['capacity'] for user in users}
    node_assignment = {node: None for node in G.nodes()}
    users_sorted = sorted(users, key=lambda u: u['efficiency'], reverse=True)
    
    print(f"Users sorted by efficiency")
    # 1. Loop over nodes and assign to the lowest-cost eligible user
    for node in sorted(G.nodes, key=lambda n: G.nodes[n]['degree'], reverse=True):
        difficulty = G.nodes[node]['degree']
        # Find eligible users (enough capacity)
        eligible_users = [u for u in users_sorted if user_remaining_capacity[u['user_id']] >= difficulty and user_model.predict([[u['skills'], difficulty]])[0]==0]
        if eligible_users:
            # Assign to user with lowest cost
            chosen_user = min(eligible_users, key=lambda u: u['cost'])
            #chosen_user = max(eligible_users, key=lambda u: u['efficiency'])
            node_assignment[node] = chosen_user['user_id']
            user_remaining_capacity[chosen_user['user_id']] -= difficulty
        else:
            print(f"No eligible user for node.")
            exp_qualities = [{idx : np.sum(user_model.predict_proba([[u['skills'],difficulty]])*[1,0.33,0.66])} for idx, u in enumerate(users_sorted) if user_remaining_capacity[u['user_id']] >= difficulty]
            selected_user = max(exp_qualities, key=lambda x: list(x.values())[0])
            chosen_user = users_sorted[list(selected_user.keys())[0]]
            node_assignment[node] = chosen_user['user_id']
            user_remaining_capacity[chosen_user['user_id']] -= difficulty
    
    print(f"Initial assignment done. users assigned to nodes.")

    # 2. Growing phase: users try to expand their component
    changed = True
    while changed:
        changed = False
        for user in sorted(users, key=lambda u: u['cost']):  # Lower cost users first
        #for user in users_sorted:  # Lower cost users first
            user_id = user['user_id']
            # Get nodes currently assigned to this user
            user_nodes = {n for n, uid in node_assignment.items() if uid == user_id}
            if not user_nodes:
                continue
            # Find neighbors (assigned to others or unassigned)
            neighbors = set()
            for n in user_nodes:
                neighbors.update(G.neighbors(n))
            neighbors -= user_nodes
            for nb in neighbors:
                nb_difficulty = G.nodes[nb]['degree']
                current_owner = node_assignment[nb]
                # Can only "eat" if user has enough capacity
                if user_remaining_capacity[user_id] >= nb_difficulty and user_model.predict([[user['skills'], nb_difficulty]])[0]==0:
                    # If unassigned, just take it
                    if current_owner is None:
                        node_assignment[nb] = user_id
                        user_remaining_capacity[user_id] -= nb_difficulty
                        changed = True
                    # If assigned to a higher-cost user, "eat" it and its component
                    elif user_dict[current_owner]['cost'] > user['cost']:
                        # Find all nodes in the component of the higher-cost user connected to nb
                        to_steal = set()
                        stack = [nb]
                        while stack:
                            curr = stack.pop()
                            if node_assignment[curr] == current_owner and curr not in to_steal:
                                to_steal.add(curr)
                                stack.extend([nbr for nbr in G.neighbors(curr) if node_assignment[nbr] == current_owner])
                        # Check if user has enough capacity for all
                        total_difficulty = sum(G.nodes[n]['degree'] for n in to_steal)
                        if user_remaining_capacity[user_id] >= total_difficulty:
                            for n in to_steal:
                                node_assignment[n] = user_id
                            user_remaining_capacity[user_id] -= total_difficulty
                            user_remaining_capacity[current_owner] += total_difficulty
                            changed = True

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


def main(theta, users, dataset, user_model):
    f1_score = []
    gnn_f1_score = []
    users_f1_score = []
    
    grdg = load_graph(dataset)    
    users = generate_users(users, 5, 50, 1, 5)

    f1_score, gnn_f1_score=gnn_pass(grdg, theta, f1_score, gnn_f1_score)

    node_assignment = {}
    edge_cut=grdg.number_of_edges()
    total_cost=0
    
    print(f"Graph has {grdg.number_of_nodes()} nodes and {grdg.number_of_edges()} edges")
    if(len(grdg.nodes)<=len(users)):
        node_assignment, total_cost, edge_cut = uQAR_degree_assignment(grdg, users, user_model)
    
        
    
    print(f"Assigned: {len(set(node_assignment.values()))} up to {len(users)} nodes")
    print("AU:", len(set(node_assignment.values())))
    print("Edge cut:", edge_cut)
    print("Total cost:", total_cost)
    



    f1_map = {0:1,1: 0,2: 0.33}   

    for node in node_assignment.keys():
        res = f1_map[user_model.predict([[users[node_assignment[node]]['skills'], grdg.nodes[node]['real_difficulty']]])[0]]
        users_f1_score.append(res)
        f1_score.append(res)
    
    
    print("F1 Score:", np.mean(f1_score))
    print("GNN F1 Score:", np.mean(gnn_f1_score))
    print("Users F1 Score:", np.mean(users_f1_score))
    

    
    

    
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

    user_model = load_user_models('repair_type_model.pkl')


    print(f"Theta: {theta}")
    print(f"Users: {users}")
    print(f"Dataset: {dataset}")
    main(theta, users, dataset, user_model)



