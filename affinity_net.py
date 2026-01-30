"""
AffinityNet - Learnable network for predicting object associations.
Simplified version using ONLY geometric features (no pretrained model required).

Input: 5D spatial features (vertical_dist, overlap, centeredness, size_ratio, mask_iou)
Output: Affinity score [0, 1]
"""

import torch
import torch.nn as nn


class AffinityNet(nn.Module):
    """
    MLP for predicting association affinity between parallelepiped and cube.
    Uses only geometric/spatial features - no visual features needed.
    """
    
    def __init__(self, spatial_dim=5, hidden_dims=[32, 16]):
        super().__init__()
        
        layers = []
        prev_dim = spatial_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.extend([
            nn.Linear(prev_dim, 1),
            nn.Sigmoid()
        ])
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, spatial_features):
        """
        Args:
            spatial_features: [B, 5] - batch of spatial feature vectors
        
        Returns:
            affinity: [B, 1] - probability of association
        """
        return self.network(spatial_features)
    
    def predict_matrix(self, spatial_matrix):
        """
        Predict affinity for all pairs.
        
        Args:
            spatial_matrix: [N_para, N_cube, 5] - spatial features for all pairs
        
        Returns:
            affinity_matrix: [N_para, N_cube] - affinity scores
        """
        N_para, N_cube, _ = spatial_matrix.shape
        
        # Flatten to [N_para * N_cube, 5]
        spatial_flat = spatial_matrix.reshape(-1, 5)
        
        # Forward pass
        affinity_flat = self.forward(spatial_flat)  # [N_para * N_cube, 1]
        
        # Reshape to matrix
        affinity_matrix = affinity_flat.reshape(N_para, N_cube)
        
        return affinity_matrix


def compute_spatial_features(para_bbox, cube_bbox, para_mask, cube_mask, image_size):
    """
    Compute 5D spatial feature vector for a single para-cube pair.
    
    Args:
        para_bbox: [x1, y1, x2, y2] - parallelepiped bounding box
        cube_bbox: [x1, y1, x2, y2] - cube bounding box
        para_mask: [H, W] - binary mask for parallelepiped
        cube_mask: [H, W] - binary mask for cube
        image_size: (H, W) - image dimensions
    
    Returns:
        features: [5] - numpy array of spatial features
    """
    import numpy as np
    
    H, W = image_size
    
    # Unpack bounding boxes
    px1, py1, px2, py2 = para_bbox
    cx1, cy1, cx2, cy2 = cube_bbox
    
    # === FEATURE 1: Vertical distance (normalized) ===
    # Distance from parallelepiped bottom to cube top
    vertical_dist = abs(py2 - cy1) / H
    # Normalize: closer = higher score
    vertical_score = max(0, 1.0 - vertical_dist * 5.0)
    
    # === FEATURE 2: Horizontal overlap ===
    overlap_left = max(px1, cx1)
    overlap_right = min(px2, cx2)
    overlap_width = max(0, overlap_right - overlap_left)
    para_width = px2 - px1
    horizontal_overlap = overlap_width / (para_width + 1e-6)
    
    # === FEATURE 3: Centeredness ===
    # How centered is para over cube horizontally
    para_center_x = (px1 + px2) / 2
    cube_center_x = (cx1 + cx2) / 2
    cube_width = cx2 - cx1
    offset = abs(para_center_x - cube_center_x)
    centeredness = max(0, 1.0 - offset / (cube_width / 2 + 1e-6))
    
    # === FEATURE 4: Size ratio ===
    para_area = (px2 - px1) * (py2 - py1)
    cube_area = (cx2 - cx1) * (cy2 - cy1)
    size_ratio = para_area / (cube_area + 1e-6)
    # Clip to reasonable range
    size_ratio = min(size_ratio, 1.0)
    
    # === FEATURE 5: Mask IoU ===
    if para_mask is not None and cube_mask is not None:
        intersection = np.logical_and(para_mask, cube_mask).sum()
        union = np.logical_or(para_mask, cube_mask).sum()
        mask_iou = intersection / (union + 1e-6)
    else:
        mask_iou = 0.0
    
    features = np.array([
        vertical_score,
        horizontal_overlap,
        centeredness,
        size_ratio,
        mask_iou
    ], dtype=np.float32)
    
    return features


def compute_spatial_features_batch(para_bboxes, cube_bboxes, para_masks, cube_masks, image_size):
    """
    Compute spatial features for all para-cube pairs.
    
    Args:
        para_bboxes: [N_para, 4] - parallelepiped bounding boxes
        cube_bboxes: [N_cube, 4] - cube bounding boxes
        para_masks: [N_para, H, W] - parallelepiped masks
        cube_masks: [N_cube, H, W] - cube masks
        image_size: (H, W)
    
    Returns:
        spatial_matrix: [N_para, N_cube, 5] - spatial features
    """
    import numpy as np
    
    N_para = len(para_bboxes)
    N_cube = len(cube_bboxes)
    
    spatial_matrix = np.zeros((N_para, N_cube, 5), dtype=np.float32)
    
    for i in range(N_para):
        for j in range(N_cube):
            spatial_matrix[i, j] = compute_spatial_features(
                para_bboxes[i],
                cube_bboxes[j],
                para_masks[i] if para_masks is not None else None,
                cube_masks[j] if cube_masks is not None else None,
                image_size
            )
    
    return spatial_matrix


if __name__ == '__main__':
    # Test
    model = AffinityNet()
    
    # Test single sample
    spatial = torch.randn(1, 5)
    output = model(spatial)
    print(f"Single sample output: {output.item():.4f}")
    
    # Test batch
    spatial_batch = torch.randn(10, 5)
    outputs = model(spatial_batch)
    print(f"Batch outputs shape: {outputs.shape}")
    
    # Test matrix prediction
    spatial_matrix = torch.randn(3, 4, 5)  # 3 paras, 4 cubes
    affinity_matrix = model.predict_matrix(spatial_matrix)
    print(f"Affinity matrix shape: {affinity_matrix.shape}")
    print(f"Affinity matrix:\n{affinity_matrix}")
