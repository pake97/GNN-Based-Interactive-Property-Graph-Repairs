
from torch_geometric.data import HeteroData
import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv, Linear
from torch_geometric.data import HeteroData
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.loader import NeighborLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv
from torch.nn import Linear
from torch_geometric.utils import negative_sampling
from torch.nn import BCEWithLogitsLoss, BCELoss
from sklearn.metrics import accuracy_score
import sys
import os
import torch


def custom_link_split(data: HeteroData, neg_ratio=0.3, min_test_edges=10):
    train_data = HeteroData()
    test_data = HeteroData()

    # Copy node features
    for ntype in data.node_types:
        train_data[ntype].x = data[ntype].x
        train_data[ntype].num_nodes = data[ntype].num_nodes
        test_data[ntype].x = data[ntype].x
        test_data[ntype].num_nodes = data[ntype].num_nodes

    for edge_type in data.edge_types:
        edge_index = data[edge_type].edge_index
        edge_label = data[edge_type].edge_label

        # Check if test_mask is valid
        if "test_mask" in data[edge_type] and data[edge_type].test_mask.sum() > 0:
            test_mask = data[edge_type].test_mask
        else:
            # Fallback random test split
            num_edges = edge_index.size(1)
            perm = torch.randperm(num_edges)
            test_size = max(min_test_edges, int(0.1 * num_edges))
            test_mask = torch.zeros(num_edges, dtype=torch.bool)
            test_mask[perm[:test_size]] = True

        # Split
        pos_train_idx = (~test_mask).nonzero(as_tuple=False).view(-1)
        pos_test_idx = test_mask.nonzero(as_tuple=False).view(-1)

        if len(pos_train_idx) == 0 or len(pos_test_idx) == 0:
            print(f"⚠️ Skipping {edge_type} due to no train or test positives.")
            continue

        train_pos_edges = edge_index[:, pos_train_idx]
        test_pos_edges = edge_index[:, pos_test_idx]

        train_pos_labels = edge_label[pos_train_idx]
        test_pos_labels = edge_label[pos_test_idx]

        # Negative sampling
        num_train_neg = int(train_pos_edges.size(1) * neg_ratio)
        num_test_neg = int(test_pos_edges.size(1) * neg_ratio)

        neg_train_edges = negative_sampling(
            edge_index=train_pos_edges,
            num_nodes=(data[edge_type[0]].num_nodes, data[edge_type[2]].num_nodes),
            num_neg_samples=num_train_neg
        )
        neg_test_edges = negative_sampling(
            edge_index=test_pos_edges,
            num_nodes=(data[edge_type[0]].num_nodes, data[edge_type[2]].num_nodes),
            num_neg_samples=num_test_neg
        )

        train_data[edge_type].edge_index = torch.cat([train_pos_edges, neg_train_edges], dim=1)
        train_data[edge_type].edge_label = torch.cat([train_pos_labels, torch.zeros(num_train_neg, dtype=torch.long)], dim=0)
        train_data[edge_type].edge_label_index = torch.cat([train_pos_edges, neg_train_edges], dim=1)

        test_data[edge_type].edge_index = torch.cat([test_pos_edges, neg_test_edges], dim=1)
        test_data[edge_type].edge_label = torch.cat([test_pos_labels, torch.zeros(num_test_neg, dtype=torch.long)], dim=0)
        test_data[edge_type].edge_label_index = torch.cat([test_pos_edges, neg_test_edges], dim=1)

    print("✅ Link split completed.")
    return train_data, test_data



class HeteroGAT(torch.nn.Module):
    def __init__(self, metadata, input_dims, hidden_dim, heads=4):
        super().__init__()
        self.metadata = metadata
        self.hidden_dim = hidden_dim
        self.heads = heads

        # 1. Per-node-type input projection
        self.node_proj = torch.nn.ModuleDict({
            ntype: Linear(in_dim, hidden_dim)
            for ntype, in_dim in input_dims.items()
        })

        # 2. Shared GAT layer across all edge types
        self.convs = HeteroConv({
            edge_type: GATConv((hidden_dim, hidden_dim), hidden_dim, heads=heads, concat=True, add_self_loops=False)
            for edge_type in metadata[1]
        }, aggr="sum")

        # 3. Per-edge-type link predictor
        self.edge_predictor = torch.nn.ModuleDict({
            '__'.join(edge_type): Linear(hidden_dim * heads * 2, 1)
            for edge_type in metadata[1]
        })

    def forward(self, x_dict, edge_index_dict):
        # 1. Project node features to shared hidden size
        x_dict = {
            ntype: self.node_proj[ntype](x.float())
            for ntype, x in x_dict.items()
        }

        # 2. Apply GATConv
        x_dict = self.convs(x_dict, edge_index_dict)
        x_dict = {k: F.elu(v) for k, v in x_dict.items()}
        return x_dict

    def predict_links(self, x_dict, edge_label_index_dict):
        scores = {}
        for edge_type, edge_index in edge_label_index_dict.items():
            src_type, _, dst_type = edge_type

            # Skip if embeddings were not returned
            if src_type not in x_dict or dst_type not in x_dict:
                print(f"⚠️ Skipping {edge_type} — embeddings missing for {src_type} or {dst_type}")
                continue

            src, dst = edge_index
            src_emb = x_dict[src_type][src]
            dst_emb = x_dict[dst_type][dst]
            edge_input = torch.cat([src_emb, dst_emb], dim=-1)
            edge_type_str = '__'.join(edge_type)

            logits = self.edge_predictor[edge_type_str](edge_input)
            scores[edge_type] = torch.sigmoid(logits).squeeze()
        return scores



# def train(model, data, optimizer, device='cpu', epochs=10):
#     print("\n🚀 Starting training...")
#     model.to(device)
#     data = data.to(device)
#     #loss_fn = BCEWithLogitsLoss()
#     loss_fn = BCELoss()

#     for epoch in range(1, epochs + 1):
#         model.train()
#         optimizer.zero_grad()

#         x_dict = model(data.x_dict, data.edge_index_dict)
#         x_dict = {k: v.float() for k, v in x_dict.items()}  # force dtype consistency
#         total_loss = 0
#         skipped = 0
        
#         for edge_type in data.edge_types:
#             if 'edge_label' not in data[edge_type]:
#                 continue

#             if edge_type not in data.edge_label_index_dict:
#                 continue
            
#             edge_label = data[edge_type].edge_label.float()
#             edge_index = data[edge_type].edge_label_index

#             src, dst = edge_index
#             src_emb = x_dict[edge_type[0]][src]
#             dst_emb = x_dict[edge_type[2]][dst]
#             edge_input = torch.cat([src_emb, dst_emb], dim=-1)

#             logits = model.edge_predictor['__'.join(edge_type)](edge_input).squeeze()
            
            
#             logits = torch.sigmoid(logits)
            
#             loss = loss_fn(logits, edge_label)
#             total_loss += loss

#         total_loss.backward()
#         optimizer.step()

#         print(f"Epoch {epoch:02d} | Loss: {total_loss.item():.4f}")

#     print("✅ Training finished.")
from torch.nn import BCEWithLogitsLoss

def train(model, data, optimizer, epochs=50, device='cpu'):
    model.to(device)
    data = data.to(device)
    loss_fn = BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        x_dict = model(data.x_dict, data.edge_index_dict)
        x_dict = {k: v.float() for k, v in x_dict.items()}  # force dtype consistency

        total_loss = 0
        skipped = 0

        for edge_type in data.edge_types:
            if 'edge_label' not in data[edge_type]:
                continue

            if edge_type not in data.edge_label_index_dict:
                continue

            edge_label_index = data[edge_type].edge_label_index
            edge_label = data[edge_type].edge_label.float()

            src_type, _, dst_type = edge_type
            if src_type not in x_dict or dst_type not in x_dict:
                skipped += 1
                continue

            src, dst = edge_label_index
            src_emb = x_dict[src_type][src]
            dst_emb = x_dict[dst_type][dst]
            edge_input = torch.cat([src_emb, dst_emb], dim=-1)

            edge_type_str = '__'.join(edge_type)
            logits = model.edge_predictor[edge_type_str](edge_input).squeeze()

            loss = loss_fn(logits, edge_label)
            total_loss += loss

        total_loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1:02d} | Loss: {total_loss.item():.4f} | Skipped edge types: {skipped}")



def evaluate(model, test_data, device='cpu'):
    print("\n🧪 Evaluating model...")
    model.eval()
    test_data = test_data.to(device)

    x_dict = model(test_data.x_dict, test_data.edge_index_dict)

    for edge_type in test_data.edge_types:
        if 'edge_index' in test_data[edge_type] and 'edge_label' in test_data[edge_type]:
            test_data[edge_type].edge_label_index = test_data[edge_type].edge_index

    scores = model.predict_links(x_dict, test_data.edge_label_index_dict)

    for edge_type, preds in scores.items():
        preds_binary = (preds > 0.5).long()
        labels = test_data[edge_type].edge_label.long()
        acc = accuracy_score(labels.cpu(), preds_binary.cpu())
        print(f"✅ Accuracy for {edge_type}: {acc:.4f}")
    return x_dict, scores

def save_model(model, path):
    print(f"\n💾 Saving model to {path}...")
    torch.save(model.state_dict(), path)
    print("✅ Model saved.")


def main(file_path, output_dir):
    


    print(f"📂 Loading data from {file_path}...")
    data = torch.load(file_path, weights_only=False)
    train_data, test_data = custom_link_split(data)
    input_dims = {ntype: train_data[ntype].x.size(1) for ntype in train_data.node_types}
    for edge_type in test_data.edge_types:
        edge_index = test_data[edge_type].edge_index
        if edge_index.dtype != torch.long:
            print(f"⚠️ Casting {edge_type} edge_index to long")
            test_data[edge_type].edge_index = edge_index.long()
    
    model = HeteroGAT(metadata=train_data.metadata(), input_dims=input_dims, hidden_dim=64)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train(model, train_data, optimizer, epochs=50, device='cpu')  # or 'cuda' if applicable

    x_dict, scores =  evaluate(model, test_data, device='cpu')
    save_model(model, os.path.join(output_dir, 'hetero_gat_model.pt'))
    # Save node embeddings (x_dict) and edge scores (scores)
    torch.save(x_dict, os.path.join(output_dir, 'x_dict.pt'))
    torch.save(scores, os.path.join(output_dir, 'scores.pt'))
    torch.save(test_data.edge_label_index_dict, os.path.join(output_dir, 'edge_label_index_dict.pt'))
    print("✅ All tasks completed.")

if __name__ == '__main__':
    dir_path = sys.argv[1]
    file_name = sys.argv[2]
    file_path = f'{dir_path}/{file_name}'
    main(file_path, dir_path)
