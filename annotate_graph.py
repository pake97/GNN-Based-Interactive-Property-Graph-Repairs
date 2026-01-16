
import pandas as pd 
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import pickle
import networkx as nx
from collections import defaultdict
from itertools import combinations
import json
import numpy as np
import torch
import numpy as np
import resource



def df_to_nx(df): 
    G = nx.DiGraph() #create a directed graph
    #iterate over the rows of the dataframe
    for i, row in df.iterrows():
        
        #if row['id] is not empty, it is a node , the node type is in the column "_labels" but we need to strip ":", the we add all the properties (the columns) that are not NaN to the node. 
        if pd.notna(row['_id']):
            node = int(row['_id'])
            node_type = row['_labels'].split(':')[1]
            #add the node to the graph
            G.add_node(node, type=node_type)
            #add the properties to the node
            for col in df.columns:
                if col not in ['_id', '_labels'] and pd.notna(row[col]):
                    G.nodes[node][col] = row[col]
        #if row['id] is empty, it is a relation , the relation type is in the column "_type" but we need to strip ":", source node is in column '_start' and target in column "_end" the we add all the properties (the columns) that are not NaN to the relation.        
        else:
            source = int(row['_start'])
            target = int(row['_end'])
            rel_type = row['_type']
            #add the relation to the graph
            G.add_edge(source, target, type=rel_type)
            #add the properties to the relation
            for col in df.columns:
                if pd.notna(row[col]):
                    G.edges[source, target][col] = row[col]
                    
    return G
    




def compute_node_confidence(x_dict):
    """
    Compute confidence scores for each node in x_dict based on proximity 
    to its nearest same-type neighbor in latent space.
    
    Args:
        x_dict (Dict[str, Tensor]): node_type → [num_nodes, embedding_dim]

    Returns:
        confidence_dict (Dict[str, Tensor]): node_type → [num_nodes] tensor of confidence scores
    """



    confidence_dict = {}

    for node_type, embeddings in x_dict.items():
        embeddings = embeddings.detach().cpu()

        if embeddings.size(0) < 2:
            confidence_dict[node_type] = torch.ones(embeddings.size(0))
            continue

        # Efficient pairwise L2 distances [N, N]
        dists = torch.cdist(embeddings, embeddings, p=2)

        # Mask diagonal (self-comparisons)
        dists.fill_diagonal_(float('inf'))

        # Get minimum distance to another same-type node
        min_dists, _ = torch.min(dists, dim=1)

        # Compute confidence
        confidence = torch.exp(-min_dists)
        confidence_dict[node_type] = confidence



    return confidence_dict


def annotate_graph_with_predictions_and_confidence(
    nx_graph,
    edge_label_index_dict,
    predictions,
    node_index_map,
    confidence_dict
):
    """
    Annotates a NetworkX graph with:
    - GNN prediction scores on test edges
    - Node confidence scores based on embedding similarity

    Args:
        nx_graph (nx.Graph or nx.DiGraph): Original graph.
        edge_label_index_dict (dict): edge_type → [2, num_edges] tensor of test edge indices.
        predictions (dict): edge_type → tensor of prediction scores.
        node_index_map (dict): node_type → {tensor_index: original_node_id}
        confidence_dict (dict): node_type → [num_nodes] tensor of node confidence scores.
    """

    # 👉 Annotate nodes with confidence
    for node_type, idx_to_id in node_index_map.items():
        if node_type not in confidence_dict:
            continue

        scores = confidence_dict[node_type]
        for tensor_idx, original_id in idx_to_id.items():
            if nx_graph.has_node(original_id):
                nx_graph.nodes[original_id]['node_confidence'] = float(scores[tensor_idx])

    print("Keys in predictions:", list(predictions.keys()))
    print("Keys in edge_label_index_dict:", list(edge_label_index_dict.keys()))

    # 👉 Annotate edges with prediction scores
    for edge_type, edge_index in edge_label_index_dict.items():
        if edge_type not in predictions:
            continue

        scores = predictions[edge_type].detach().cpu().numpy()

        if edge_index.size(1) != len(scores):
            print(f"⚠️ Mismatch in edge count for {edge_type}")
            continue

        src_type, _, dst_type = edge_type
        src_indices = edge_index[0].tolist()
        dst_indices = edge_index[1].tolist()

        for src_idx, dst_idx, score in zip(src_indices, dst_indices, scores):
            src_id = node_index_map[src_type][src_idx]
            dst_id = node_index_map[dst_type][dst_idx]

            if nx_graph.has_edge(src_id, dst_id):
                nx_graph[src_id][dst_id]['prediction_score'] = float(score)
            elif nx_graph.has_edge(dst_id, src_id):  # for undirected
                nx_graph[dst_id][src_id]['prediction_score'] = float(score)
            else:
                nx_graph.add_edge(src_id, dst_id, prediction_score=float(score))


def invert_node_index_map(node_index_map):
    return {
        node_type: {v: k for k, v in index_map.items()}
        for node_type, index_map in node_index_map.items()
    }



class GRDG(nx.Graph):
    def __init__(self, original_graph, alpha=0.2, beta=0.1):
        super().__init__()
        self.original_graph = original_graph
        self.alpha = 0.20786
        self.beta = 0.534601
        #Alpha: 0.5059845592208605
        #Beta: 0.1957506318635603
        self._violation_features = {}
        self._build_grdg()
        
        
    def _build_grdg(self):
        # Step 1: Collect all violation subgraphs
        violations = defaultdict(lambda: {"nodes": set(), "edges": set()})

        for node, data in self.original_graph.nodes(data=True):
            if(data.get("isViolation") is True):
                for v_id in json.loads(data.get("violationId")):
                    violations[v_id]["nodes"].add(node)

        for u, v, data in graph.edges(data=True):
            if(data.get("isViolation.1") is True):
                for v_id in json.loads(data.get("violationId.1")):
                    violations[v_id]["edges"].add((u, v))

        # Step 2: Create hypervertices
        for v_id, content in violations.items():
            node_ids = list(content["nodes"])
            edge_ids = list(content["edges"])
            cognitive_load = self._compute_cognitive_load(node_ids, edge_ids)
            node_confidences = [self.original_graph.nodes[n].get("node_confidence", 1) for n in node_ids]
            edge_scores = [self.original_graph.get_edge_data(u, v).get("prediction_score", 1) for u, v in edge_ids]
            labels = [self.original_graph.get_edge_data(u, v).get("toDelete", False) for u, v in edge_ids]
            edge_info_dict = {
                (u, v): (
                    self.original_graph.get_edge_data(u, v).get("prediction_score", 1),
                    self.original_graph.get_edge_data(u, v).get("toDelete", False)
                )
                for (u, v) in edge_ids
            }

            avg_node_conf = sum(node_confidences) / len(node_confidences) if node_confidences else 0.0
            avg_edge_score = sum(edge_scores) / len(edge_scores) if edge_scores else 0.0
            sum_node_conf = sum(node_confidences)
            sum_edge_score = sum(edge_scores)
            real_difficulty = avg_node_conf * self.alpha + avg_edge_score * self.beta
            self._violation_features[v_id] = (avg_node_conf, avg_edge_score)
            self.add_node(v_id,
                          type="hypervertex",
                          difficulty=0,
                          cognitive_load=cognitive_load,
                          nodes=node_ids,
                          edge_info_dict=edge_info_dict,
                          real_difficulty=real_difficulty,
                          avg_sum_node_conf=0.5*sum_node_conf+0.5*sum_edge_score,
                          avg_node_confidence=avg_node_conf,
                          avg_edge_prediction_score=avg_edge_score,
                          sum_node_conf=sum_node_conf, 
                          sum_edge_score = sum_edge_score,
                          edges=edge_ids)
        
        alpha, beta = 0.5, 0.5
        self.compute_difficulty_with_weights(alpha, beta)
        

        # Step 3: Add weighted edges between hypervertices based on shared nodes
        for v1, v2 in combinations(self.nodes, 2):
            shared_nodes = set(self.nodes[v1]["nodes"]) & set(self.nodes[v2]["nodes"])
            if shared_nodes:
                sum_conf = sum(
                    self.original_graph.nodes[n].get("node_confidence", 0.0)
                    for n in shared_nodes
                )
                weight = 1 / sum_conf if sum_conf > 0 else float("inf")
                self.add_edge(v1, v2, weight=weight)

        self.compute_node_metrics()

    def compute_difficulty_with_weights(self, alpha, beta):
        """
        Compute and assign difficulty to each hypervertex using the given alpha and beta.
        Difficulty is defined as:
            difficulty = alpha * avg_node_confidence + beta * avg_edge_prediction_score
        """
        for v_id, (avg_node_conf, avg_edge_score) in self._violation_features.items():
            difficulty = alpha * avg_node_conf + beta * avg_edge_score
            self.nodes[v_id]["difficulty"] = difficulty

    def compute_difficulties_with_weights(self, alpha, beta, idx):
        """
        Compute and assign difficulty to each hypervertex using the given alpha and beta.
        Difficulty is defined as:
            difficulty = alpha * avg_node_confidence + beta * avg_edge_prediction_score
        """
        for v_id, (avg_node_conf, avg_edge_score) in self._violation_features.items():
            difficulty = alpha * avg_node_conf + beta * avg_edge_score
            self.nodes[v_id]["difficulty"+str(idx)] = difficulty


    def _compute_cognitive_load(self, node_ids, edge_ids):
        n = len(node_ids)
        m = len(edge_ids)
        density = (2 * m) / (n * (n - 1)) if n > 1 else 0.0
        cognitive_load = m * density

        return cognitive_load

    def compute_node_metrics(self):
        """
        Computes degree and PageRank for each node and stores them as attributes:
        - 'degree': number of edges connected to the node
        - 'pagerank': importance score from PageRank algorithm
        """
        degrees = dict(self.degree())
        pageranks = nx.pagerank(self)
        

        for node in self.nodes:
            self.nodes[node]["degree"] = degrees.get(node, 0)
            self.nodes[node]["pagerank"] = pageranks.get(node, 0.0)
             # Compute cs_cl only for hypervertices (which have 'nodes' and 'edges' attributes)
            
                
            node_ids = self.nodes[node].get("nodes", [])
            edge_ids = self.nodes[node].get("edge_info_dict", [])

            num_properties = sum(len(self.original_graph.nodes[n].keys()) for n in node_ids)
            cs_cl = len(node_ids) + len(edge_ids.keys()) + num_properties

            self.nodes[node]["cs_cl"] = cs_cl

    def plot_difficulty_distribution(self, bins=30, kde=False):

        difficulties = [self.nodes[v]["difficulty"] for v in self.nodes if "difficulty" in self.nodes[v]]

        plt.figure(figsize=(8, 5))
        if kde:
            sns.kdeplot(difficulties, fill=True, bw_adjust=0.3)
            plt.title("KDE of Violation Difficulties")
        else:
            plt.hist(difficulties, bins=bins, edgecolor='black')
            plt.title("Histogram of Violation Difficulties")

        plt.xlabel("Difficulty")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def set_difficulty(self, v_id, value, edge=False, target=None):
        if edge:
            if self.has_edge(v_id, target):
                self[v_id][target]["difficulty"] = value
        else:
            if v_id in self.nodes:
                self.nodes[v_id]["difficulty"] = value

    def get_violation_subgraph(self, v_id):
        if v_id not in self.nodes:
            return None
        nodes = self.nodes[v_id].get("nodes", [])
        return self.original_graph.subgraph(nodes).copy()
    
    def get_average_difficulty(self):
        difficulties = [self.nodes[v]["difficulty"] for v in self.nodes if "difficulty" in self.nodes[v]]
        return np.mean(difficulties) if difficulties else 0.0 , np.std(difficulties) if difficulties else 0.0
    
    def save_as_csv(self, dir_path): 
        nodes_data = []
        for node, data in self.nodes(data=True):
            row = {"id": node, "type": data.get("type", "unknown")}
            row.update(data)
            nodes_data.append(row)

        edges_data = []
        for u, v, data in self.edges(data=True):
            row = {"source": u, "target": v}
            row.update(data)
            edges_data.append(row)

        nodes_df = pd.DataFrame(nodes_data)
        edges_df = pd.DataFrame(edges_data)

        nodes_df.to_csv(dir_path+'/grdg_nodes.csv', index=False)
        edges_df.to_csv(dir_path+'/grdg_edges.csv', index=False)
        print(f"Saved GRDG nodes to {dir_path}/grdg_nodes.csv")
        print(f"Saved GRDG edges to {dir_path}/grdg_edges.csv")





#grdg.plot_difficulty_distribution()  # KDE curve



if __name__ == "__main__":
    
    

    dir_path = sys.argv[1]
    file_name = sys.argv[2]
    




    
    
    


    print(f"Processing dataset from {dir_path} with file {file_name}...")
    x_dict = torch.load(dir_path+'/x_dict.pt',weights_only=False)
    print("Loaded node embeddings from x_dict.pt.")
    confidence_scores = compute_node_confidence(x_dict)
    
    print("Computed node confidence scores.")
    # For example:
    # for ntype, scores in confidence_scores.items():
    #     print(f"{ntype} — mean confidence: {scores.mean():.4f}, min: {scores.min():.4f}")
    id_map={}
    with open(dir_path+'/node_map.pkl', 'rb') as f:
        id_map = pickle.load(f)
    inverted_node_index_map = invert_node_index_map(id_map)
    
    print("Inverted node index map.")
    data = pd.read_csv(f'{dir_path}/{file_name}',low_memory=False)

    graph = df_to_nx(data)
    
    nx.write_graphml(graph, dir_path+'/sw.graphml')    
    print("Converted DataFrame to NetworkX graph.")
    edge_scores = torch.load(dir_path+'/scores.pt',weights_only=False)
    edge_label_index_dict = torch.load(dir_path+'/edge_label_index_dict.pt',weights_only=False)
    annotate_graph_with_predictions_and_confidence(
        nx_graph=graph,
        edge_label_index_dict=edge_label_index_dict,
        predictions=edge_scores,
        node_index_map=inverted_node_index_map,
        confidence_dict=confidence_scores
    )
    print("Annotated graph with predictions and node confidence scores.")
    

    # bbedge_scores = []
    # for n in graph.edges(data=True):
        
    #     if 'prediction_score' in n[2]:
            
    #         bbedge_scores.append(n[2]['prediction_score'])    
        
    # print("mean edge score: ", np.mean(bbedge_scores))
    # print("std edge score: ", np.std(bbedge_scores))
    # print("min edge score: ", np.min(bbedge_scores))
    # print("max edge score: ", np.max(bbedge_scores))



    grdg = GRDG(graph)
    

    grdg.save_as_csv(dir_path=dir_path)    
    

    #print("Average difficulty of violations: ", grdg.get_average_difficulty())








### For star wars : 

# confidences = {
#  'Q1': {
#      'nodes':[770, 607,818],
#      'edges':[(607,770),(607,818)]
#  },
#  'Q2': {
#      'nodes':[634, 652,581],
#      'edges':[(634,652),(652,581)]
#  },
#  'Q3': {
#      'nodes':[580,596,590],
#      'edges':[(590,596),(596,580)]
#  },
#  'Q4': {
#      'nodes':[600,584],
#      'edges':[(600,584)]
#  },
#  'Q5': {
#      'nodes':[628,582],
#      'edges':[(628,582)]
#  },
#  'Q6': {
#     'nodes':[628,582],
#      'edges':[(628,582)]
#  },   
# }


# for q in confidences:
#     node_conf = []
#     edge_conf = []
    
#     print(f"Confidence for {q}:")
#     nodes = confidences[q]['nodes']
#     edges = confidences[q]['edges']
#     for node_id in nodes:
#         n = graph.nodes[node_id]
#         node_conf.append(n.get('node_confidence', 0))
#     for edge in edges:
        
#         u, v = edge
#         if graph.has_edge(u, v):
#             e=graph[u][v]
            
#             edge_conf.append(e.get('prediction_score', 0.9))
#         else:
#             print(f"Edge {edge} not found in graph.")
#     print(f"  Nodes: {np.mean(node_conf):.4f} ")
#     print(f"  Edges: {np.mean(edge_conf):.4f} ")
        
        





