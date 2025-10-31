
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import sys 
import pickle








# Helper to resolve a node ID to its type
def resolve_type(node_id, id_maps):
   for ntype, id_map in id_maps.items():
       if node_id in id_map:
           return ntype
   return None



def parse_heterogeneous_csv(csv_path, embed_model='all-MiniLM-L6-v2'):
    # embed_model='all-MiniLM-L6-v2'
    # csv_path = './data/finbench/finbench_with_violations.csv' 
    model = SentenceTransformer(embed_model)
    df = pd.read_csv(csv_path, low_memory=False)
    # strip ':' from columhn named _labels 

    df['_labels'] = df['_labels'].str.replace(':', '', regex=False)
    data = HeteroData()
    id_maps = defaultdict(dict)

    is_edge = df["_id"].isna()
    node_df = df[~is_edge].copy()
    edge_df = df[is_edge].copy()
    print(node_df)
    for node_type in node_df["_labels"].unique():
        
        sub_df = node_df[node_df["_labels"] == node_type].copy()
        
        
        # Drop completely empty columns
        sub_df = sub_df.dropna(axis=1, how='all')
        
        # Identify properties that have a _GT counterpart
        gt_cols = [col for col in sub_df.columns if col.endswith("_GT")]
        valid_props = [col[:-3] for col in gt_cols if col[:-3] in sub_df.columns]
        # Create ID mapping for this node type
        
        sub_df["_id"] = sub_df["_id"].astype(int)
        
        node_ids = sub_df["_id"].tolist()
        id_maps[node_type] = dict(zip(sub_df["_id"].tolist(), range(len(sub_df))))
        
        
        sub_df = sub_df[valid_props]
        features = []
        for col in sub_df.columns.to_list():

            col_data = sub_df[col]
            if col_data.astype(str).isin(['True', 'False']).all():
                # Boolean column
                #features.append(col_data.astype(str).map({'True': 1, 'False': 0}).astype(int).values[:, None])
                features.append(torch.tensor(
                    col_data.astype(str).map({'True': 1, 'False': 0}).astype(int).values[:, None],
                    dtype=torch.float
                ))
            elif col_data.dtype == object:
                # String column → embed
                embeddings = model.encode(col_data.astype(str).tolist(), show_progress_bar=False)
                features.append(torch.tensor(embeddings, dtype=torch.float))
            else:
                #features.append(torch.tensor(col_data.astype(float).values[:, None]))
                scaler = MinMaxScaler()
                scaled = scaler.fit_transform(col_data.astype(float).values.reshape(-1, 1))
                features.append(torch.tensor(scaled, dtype=torch.float))

        if features:
            data[node_type].x = torch.cat(features, dim=1)
        else:
            data[node_type].x = torch.empty((len(sub_df), 0))
        data[node_type].node_id = node_ids

    
    # First, annotate edge_df with inferred source and target node types
    edge_df["_src_type"] = edge_df["_start"].apply(lambda x: resolve_type(x,id_maps))
    edge_df["_dst_type"] = edge_df["_end"].apply(lambda x: resolve_type(x,id_maps))

    # Drop edges where we can't resolve both types
    edge_df = edge_df.dropna(subset=["_src_type", "_dst_type"])
    edge_df=edge_df[["_src_type", "_dst_type", "_type", "_start", "_end","toDelete", "isViolation.1"]]
    edge_df.columns=["_src_type", "_dst_type", "_type", "_start", "_end","toDelete", "isViolation"]
    edge_df['isViolation']=edge_df['isViolation'].fillna(False)
    print(edge_df.head())
    # Now group by full (src_type, rel_type, dst_type) tuple
    grouped = edge_df.groupby(["_src_type", "_type", "_dst_type"])
    print(grouped.head())






    for (src_type, rel_type, dst_type), group in grouped:
        src_ids = group["_start"].tolist()
        dst_ids = group["_end"].tolist()

        src_index = [id_maps[src_type][i] for i in src_ids]
        dst_index = [id_maps[dst_type][i] for i in dst_ids]

        edge_key = (src_type, rel_type, dst_type)
        print(edge_key)
        data[edge_key].edge_index = torch.tensor([src_index, dst_index], dtype=torch.long)

        
        
        labels = group["toDelete"].fillna(False).map({True: 0, False: 1}).astype(int)    
        data[edge_key].edge_label = torch.tensor(labels.values, dtype=torch.long)
        
        # Violation mask → test mask
        
        test_mask = group["isViolation"]
        data[edge_key].test_mask = torch.tensor(test_mask.values, dtype=torch.bool)


    return data, id_maps



# Save to file


# # Load later
# with open('data.pkl', 'rb') as f:
#     loaded_data = pickle.load(f)


if __name__ == "__main__":
    
    # read argument for file path
    embed_model='all-MiniLM-L6-v2'
    #csv_path = './data/finbench/finbench_with_violations.csv' 

    dir_path = sys.argv[1]
    file_name = sys.argv[2]
    
    csv_path = f'{dir_path}/{file_name}'
    # Example usage
    data , id_maps= parse_heterogeneous_csv(csv_path, embed_model)
    
    torch.save(data, csv_path.replace('.csv', '.pt'))
    with open(dir_path+'/node_map.pkl', 'wb') as f:
        pickle.dump(id_maps, f)
        
    #print(data)
    #print(id_maps)
    print("Data saved to", csv_path.replace('.csv', '.pt'))
    print("Node maps saved to", dir_path+'/node_map.pkl')