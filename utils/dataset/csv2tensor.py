import pandas as pd
import torch
from torch_geometric.data import HeteroData
from sentence_transformers import SentenceTransformer
from collections import defaultdict

def parse_hetero_csv(csv_path, embed_model='all-MiniLM-L6-v2'):
    model = SentenceTransformer(embed_model)

    df = pd.read_csv(csv_path).fillna('')  # Ensure empty strings are consistent
    is_edge = df["_id"] == ""

    node_df = df[~is_edge].copy()
    edge_df = df[is_edge].copy()

    data = HeteroData()
    id_maps = defaultdict(dict)
    reverse_id_maps = defaultdict(dict)

    # Get properties that have _GT counterparts
    gt_columns = [col for col in df.columns if col.endswith("_GT")]
    valid_columns = [col[:-3] for col in gt_columns if col[:-3] in df.columns]

    # ==== PROCESS NODES ====
    node_types = node_df["_labels"].unique()

    for ntype in node_types:
        sub_df = node_df[node_df["_labels"] == ntype].copy()
        ids = sub_df["_id"].tolist()
        id_map = {id_: idx for idx, id_ in enumerate(ids)}
        id_maps[ntype] = id_map
        reverse_id_maps[ntype] = {v: k for k, v in id_map.items()}

        features = []

        for col in valid_columns:
            if sub_df[col].dtype == bool or sub_df[col].isin(['True', 'False']).all():
                features.append(sub_df[col].astype(str).map({'True': 1, 'False': 0}).astype(int).values[:, None])
            elif sub_df[col].dtype == object:
                embeddings = model.encode(sub_df[col].astype(str).tolist(), show_progress_bar=False)
                features.append(torch.tensor(embeddings))
            else:
                features.append(torch.tensor(sub_df[col].astype(float).values[:, None]))

        # Embed _labels
        labels = pd.Categorical(sub_df["_labels"])
        features.append(torch.tensor(labels.codes[:, None]))

        data[ntype].x = torch.cat(features, dim=1) if features else torch.empty((len(sub_df), 0))
        data[ntype].node_id = ids

    # ==== PROCESS EDGES ====
    edge_groups = edge_df.groupby("_type")

    for etype, group in edge_groups:
        src_ids = group["_start"].tolist()
        dst_ids = group["_end"].tolist()

        # Infer src_type and dst_type from lookup
        # Try to detect based on node ID presence in maps
        src_type = next((nt for nt in id_maps if all(i in id_maps[nt] for i in src_ids if i)), None)
        dst_type = next((nt for nt in id_maps if all(i in id_maps[nt] for i in dst_ids if i)), None)

        if src_type is None or dst_type is None:
            continue  # skip bad edges

        src_idx = [id_maps[src_type][i] for i in src_ids]
        dst_idx = [id_maps[dst_type][i] for i in dst_ids]

        key = (src_type, etype, dst_type)
        edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)
        data[key].edge_index = edge_index

        # Link prediction label
        if "toDelete" in group.columns:
            data[key].edge_label = torch.tensor(group["toDelete"].astype(str).map({'True': 1, 'False': 0}).astype(int).values)

        # Violation mask
        if "isViolation" in group.columns:
            mask = group["isViolation"].astype(str).map({'True': True, 'False': False}).astype(bool)
            data[key].test_mask = torch.tensor(mask.values, dtype=torch.bool)
        else:
            data[key].test_mask = torch.zeros(len(group), dtype=torch.bool)

    return data
