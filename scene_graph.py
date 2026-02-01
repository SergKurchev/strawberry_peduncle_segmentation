"""
Scene Graph representation for object relationship modeling.
Converts detections into graph structure with nodes (objects) and edges (relationships).
"""

import numpy as np
import torch
import cv2


class SceneGraph:
    """
    Represents a scene as a graph of objects and potential relationships.
    
    Nodes: Objects (cubes and parallelepipeds)
    Edges: Potential para->cube relationships
    """
    
    def __init__(self, boxes, labels, masks, image_size):
        """
        Args:
            boxes: [N, 4] numpy array - bounding boxes in xyxy format
            labels: [N] numpy array - class labels (1=cube, 2=parallelepiped)
            masks: [N, H, W] numpy array - binary segmentation masks
            image_size: (H, W) tuple - image dimensions
        """
        self.boxes = boxes
        self.labels = labels
        self.masks = masks
        self.image_size = image_size
        
        self.N = len(boxes)
        
        # Build graph components
        self.node_features = self._build_node_features()
        self.edge_index, self.edge_features = self._build_edges()
    
    def _build_node_features(self):
        """
        Create feature vector for each node (object).
        
        Node features (7D per node):
        - Class (one-hot): [2] - [cube, parallelepiped]
        - Bbox center (normalized): [2] - [cx, cy]
        - Bbox size (normalized): [2] - [w, h]
        - Mask area (normalized): [1] - area ratio
        
        Returns:
            node_features: [N, 7] numpy array
        """
        H, W = self.image_size
        node_features = []
        
        for i in range(self.N):
            bbox = self.boxes[i]
            label = self.labels[i]
            mask = self.masks[i] if self.masks is not None else None
            
            # === Class (one-hot) ===
            class_vec = [1, 0] if label == 1 else [0, 1]  # [cube, para]
            
            # === Bbox center (normalized) ===
            cx = (bbox[0] + bbox[2]) / 2 / W
            cy = (bbox[1] + bbox[3]) / 2 / H
            
            # === Bbox size (normalized) ===
            w = (bbox[2] - bbox[0]) / W
            h = (bbox[3] - bbox[1]) / H
            
            # === Mask area (normalized) ===
            if mask is not None and mask.sum() > 0:
                area = mask.sum() / (H * W)
            else:
                area = w * h  # Fallback to bbox area
            
            features = class_vec + [cx, cy, w, h, area]
            node_features.append(features)
        
        return np.array(node_features, dtype=np.float32)
    
    def _build_edges(self):
        """
        Build directed edges from all parallelepipeds to all cubes.
        
        Edge features are spatial relationships (same as Phase 2):
        - Vertical score
        - Horizontal overlap
        - Centeredness
        - Size ratio
        - Mask IoU
        
        Returns:
            edge_index: [2, E] numpy array - [source_nodes, target_nodes]
            edge_features: [E, 5] numpy array - spatial features
        """
        from affinity_net import compute_spatial_features
        
        # Separate parallelepipeds and cubes
        para_indices = np.where(self.labels == 2)[0]
        cube_indices = np.where(self.labels == 1)[0]
        
        edge_index = []
        edge_features = []
        
        # Create edges: para -> cube
        for para_idx in para_indices:
            for cube_idx in cube_indices:
                # Edge direction: para_idx -> cube_idx
                edge_index.append([para_idx, cube_idx])
                
                # Compute spatial features for this edge
                spatial = compute_spatial_features(
                    self.boxes[para_idx],
                    self.boxes[cube_idx],
                    self.masks[para_idx] if self.masks is not None else None,
                    self.masks[cube_idx] if self.masks is not None else None,
                    self.image_size
                )
                edge_features.append(spatial)
        
        # Handle empty case
        if len(edge_index) == 0:
            return np.array([[], []], dtype=np.int64), np.array([], dtype=np.float32).reshape(0, 5)
        
        # Convert to numpy arrays
        edge_index = np.array(edge_index, dtype=np.int64).T  # [2, E]
        edge_features = np.array(edge_features, dtype=np.float32)  # [E, 5]
        
        return edge_index, edge_features
    
    def to_torch(self, device='cpu'):
        """
        Convert graph to PyTorch tensors.
        
        Returns:
            dict with:
                'x': [N, 7] - node features
                'edge_index': [2, E] - edge connectivity
                'edge_attr': [E, 5] - edge features
        """
        return {
            'x': torch.from_numpy(self.node_features).to(device),
            'edge_index': torch.from_numpy(self.edge_index).long().to(device),
            'edge_attr': torch.from_numpy(self.edge_features).to(device)
        }
    
    def num_nodes(self):
        """Number of nodes in graph."""
        return self.N
    
    def num_edges(self):
        """Number of edges in graph."""
        return self.edge_index.shape[1] if self.edge_index.size > 0 else 0


if __name__ == '__main__':
    # Test scene graph construction
    print("Testing SceneGraph...")
    
    # Simulate detections
    boxes = np.array([
        [100, 100, 150, 150],  # Cube 0
        [200, 200, 250, 250],  # Cube 1
        [120, 80, 140, 95],    # Para 2
        [220, 180, 240, 195],  # Para 3
    ])
    labels = np.array([1, 1, 2, 2])
    masks = None
    image_size = (480, 640)
    
    graph = SceneGraph(boxes, labels, masks, image_size)
    
    print(f"Nodes: {graph.num_nodes()}")
    print(f"Edges: {graph.num_edges()}")
    print(f"Node features shape: {graph.node_features.shape}")
    print(f"Edge index shape: {graph.edge_index.shape}")
    print(f"Edge features shape: {graph.edge_features.shape}")
    
    # Convert to PyTorch
    graph_torch = graph.to_torch()
    print(f"\nPyTorch tensors:")
    print(f"  x: {graph_torch['x'].shape}")
    print(f"  edge_index: {graph_torch['edge_index'].shape}")
    print(f"  edge_attr: {graph_torch['edge_attr'].shape}")
    
    print("\n✅ SceneGraph test passed!")
