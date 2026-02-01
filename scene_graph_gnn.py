"""
Graph Neural Network for Scene Graph relationship prediction.
Uses message passing to reason about object relationships with global context.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SceneGraphGNN(nn.Module):
    """
    Graph Neural Network for predicting object relationships in scene graphs.
    
    Architecture:
    1. Node Embedding: Encode object features
    2. Edge Embedding: Encode spatial relationships
    3. Message Passing: Exchange information between connected nodes
    4. Edge Prediction: Predict probability of each relationship
    """
    
    def __init__(self, node_dim=7, edge_dim=5, hidden_dim=64, num_layers=2):
        super().__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Node encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Edge encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Message passing layers
        self.conv_layers = nn.ModuleList([
            GraphConvLayer(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Edge prediction head
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass through GNN.
        
        Args:
            x: [N, node_dim] - node features
            edge_index: [2, E] - edge connectivity (source, target)
            edge_attr: [E, edge_dim] - edge features
        
        Returns:
            edge_probs: [E, 1] - probability of each edge being a true relationship
        """
        # Encode nodes and edges
        x = self.node_encoder(x)  # [N, hidden_dim]
        edge_attr_encoded = self.edge_encoder(edge_attr)  # [E, hidden_dim]
        
        # Message passing to update node representations
        for conv in self.conv_layers:
            x = conv(x, edge_index, edge_attr_encoded)
        
        # Edge prediction
        # For each edge, concatenate source node, target node, and edge features
        if edge_index.shape[1] > 0:
            edge_features = torch.cat([
                x[edge_index[0]],  # Source node features
                x[edge_index[1]],  # Target node features
                edge_attr          # Original edge spatial features
            ], dim=1)  # [E, hidden_dim*2 + edge_dim]
            
            edge_probs = self.edge_predictor(edge_features)
        else:
            edge_probs = torch.empty((0, 1), dtype=torch.float32, device=x.device)
        
        return edge_probs


class GraphConvLayer(nn.Module):
    """
    Graph convolutional layer with edge-conditioned message passing.
    
    For each node, aggregates messages from neighboring nodes,
    where messages are conditioned on edge features.
    """
    
    def __init__(self, in_dim, out_dim):
        super().__init__()
        
        # Message network: combines source node + edge features
        self.message_net = nn.Sequential(
            nn.Linear(in_dim + in_dim, out_dim),  # source + edge
            nn.ReLU()
        )
        
        # Update network: combines current node + aggregated messages
        self.update_net = nn.Sequential(
            nn.Linear(in_dim + out_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
    
    def forward(self, x, edge_index, edge_attr):
        """
        Message passing step.
        
        Args:
            x: [N, in_dim] - current node features
            edge_index: [2, E] - edge connectivity
            edge_attr: [E, edge_dim] - edge features (encoded)
        
        Returns:
            x_new: [N, out_dim] - updated node features
        """
        N = x.size(0)
        
        if edge_index.shape[1] == 0:
            # No edges, just apply self-update
            return F.relu(self.update_net(torch.cat([x, torch.zeros_like(x)], dim=1)))
        
        # Aggregate messages for each node
        messages = []
        for i in range(N):
            # Find incoming edges to node i
            incoming_mask = edge_index[1] == i
            
            if incoming_mask.sum() == 0:
                # No incoming edges, use zero message
                messages.append(torch.zeros(self.message_net[-2].out_features, 
                                           dtype=x.dtype, device=x.device))
                continue
            
            # Get source nodes and edges
            source_indices = edge_index[0][incoming_mask]
            source_nodes = x[source_indices]
            edge_feats = edge_attr[incoming_mask]
            
            # Compute messages from each source
            combined = torch.cat([source_nodes, edge_feats], dim=1)
            node_messages = self.message_net(combined)
            
            # Aggregate (mean pooling)
            aggregated_message = node_messages.mean(dim=0)
            messages.append(aggregated_message)
        
        messages = torch.stack(messages)  # [N, out_dim]
        
        # Update node features
        combined_update = torch.cat([x, messages], dim=1)
        x_new = self.update_net(combined_update)
        
        # Residual connection
        if x.shape == x_new.shape:
            x_new = x_new + x
        
        return F.relu(x_new)


if __name__ == '__main__':
    # Test GNN
    print("Testing SceneGraphGNN...")
    
    # Simulate graph data
    N = 4  # 2 cubes + 2 paras
    E = 4  # 2 paras * 2 cubes
    
    x = torch.randn(N, 7)  # Node features
    edge_index = torch.tensor([[2, 2, 3, 3],  # Sources (paras)
                               [0, 1, 0, 1]], dtype=torch.long)  # Targets (cubes)
    edge_attr = torch.randn(E, 5)  # Edge features
    
    # Create model
    model = SceneGraphGNN(node_dim=7, edge_dim=5, hidden_dim=64, num_layers=2)
    
    # Forward pass
    edge_probs = model(x, edge_index, edge_attr)
    
    print(f"Input:")
    print(f"  Nodes: {x.shape}")
    print(f"  Edge index: {edge_index.shape}")
    print(f"  Edge features: {edge_attr.shape}")
    print(f"\nOutput:")
    print(f"  Edge probabilities: {edge_probs.shape}")
    print(f"  Probabilities: {edge_probs.squeeze().tolist()}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    print("\n✅ SceneGraphGNN test passed!")
