# models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv

class HeteroGAT(nn.Module):
    # ... [rest of the __init__ method]

    def forward(self, x_dict, edge_index_dict):
        # Debug: Print initial node features
        print("Initial node features:")
        for node_type, x in x_dict.items():
            print(f" - {node_type}: {x.shape}")

        # Transform node features to hidden dimension
        x_dict = {
            node_type: self.node_emb[node_type](x)
            for node_type, x in x_dict.items()
        }

        # Debug: Print transformed node features
        print("Transformed node features:")
        for node_type, x in x_dict.items():
            print(f" - {node_type}: {x.shape}")

        # Convolutional Layers
        for idx, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.elu(x) for key, x in x_dict.items()}
            
            # Debug: Print node features after convolution
            print(f"After Conv Layer {idx+1}:")
            for node_type, x in x_dict.items():
                print(f" - {node_type}: {x.shape}")

        # Output Layer (Classification)
        out = self.linear(x_dict['tweet'])
        return out
