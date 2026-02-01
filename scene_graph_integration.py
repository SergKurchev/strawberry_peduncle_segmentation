"""
Integration utilities for Scene Graph GNN.
Provides inference and evaluation functions for trained GNN models.
"""

import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import cv2

from scene_graph import SceneGraph
from scene_graph_gnn import SceneGraphGNN


def predict_associations_with_gnn(gnn_model, boxes, labels, masks, image_size, device='cuda', threshold=0.5):
    """
    Predict object associations using trained Scene Graph GNN.
    
    Args:
        gnn_model: Trained SceneGraphGNN model
        boxes: [N, 4] numpy array - bounding boxes in xyxy format
        labels: [N] numpy array - class labels (1=cube, 2=parallelepiped)
        masks: [N, H, W] numpy array - binary segmentation masks
        image_size: (H, W) tuple
        device: 'cuda' or 'cpu'
        threshold: Minimum confidence for association
    
    Returns:
        associations: dict {para_idx: cube_idx}
        affinity_matrix: [N_para, N_cube] numpy array - GNN predictions
    """
    gnn_model.eval()
    gnn_model = gnn_model.to(device)
    
    # Build scene graph
    graph = SceneGraph(boxes, labels, masks, image_size)
    graph_data = graph.to_torch(device=device)
    
    # GNN prediction
    with torch.no_grad():
        edge_probs = gnn_model(
            graph_data['x'],
            graph_data['edge_index'],
            graph_data['edge_attr']
        )  # [E, 1]
    
    # Reshape to matrix
    para_indices = np.where(labels == 2)[0]
    cube_indices = np.where(labels == 1)[0]
    
    N_para = len(para_indices)
    N_cube = len(cube_indices)
    
    if N_para == 0 or N_cube == 0:
        return {}, np.array([])
    
    affinity_matrix = edge_probs.cpu().numpy().reshape(N_para, N_cube)
    
    # Hungarian matching for optimal assignment
    row_ind, col_ind = linear_sum_assignment(-affinity_matrix)
    
    associations = {}
    for para_local_idx, cube_local_idx in zip(row_ind, col_ind):
        if affinity_matrix[para_local_idx, cube_local_idx] > threshold:
            para_global_idx = int(para_indices[para_local_idx])
            cube_global_idx = int(cube_indices[cube_local_idx])
            associations[para_global_idx] = cube_global_idx
    
    return associations, affinity_matrix


def evaluate_scene_graph_gnn(gnn_model, val_dataset, device='cuda'):
    """
    Evaluate GNN accuracy on validation dataset.
    
    Returns:
        metrics: dict with accuracy, precision, recall, F1
    """
    gnn_model.eval()
    gnn_model = gnn_model.to(device)
    
    total_correct = 0
    total_positives = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for i in range(len(val_dataset)):
        graph_data, edge_labels, metadata = val_dataset[i]
        
        graph_data = {k: v.to(device) for k, v in graph_data.items()}
        edge_labels = edge_labels.to(device)
        
        # Predict
        with torch.no_grad():
            edge_probs = gnn_model(
                graph_data['x'],
                graph_data['edge_index'],
                graph_data['edge_attr']
            ).squeeze()
        
        # Binary predictions
        predictions = (edge_probs > 0.5).float()
        
        # Metrics
        total_correct += (predictions == edge_labels).sum().item()
        total_positives += len(edge_labels)
        
        true_positives += ((predictions == 1) & (edge_labels == 1)).sum().item()
        false_positives += ((predictions == 1) & (edge_labels == 0)).sum().item()
        false_negatives += ((predictions == 0) & (edge_labels == 1)).sum().item()
    
    accuracy = total_correct / total_positives
    precision = true_positives / (true_positives + false_positives + 1e-6)
    recall = true_positives / (true_positives + false_negatives + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    return metrics


def visualize_scene_graph_predictions(image, boxes, labels, associations, 
                                      ground_truth=None, save_path=None):
    """
    Visualize scene graph predictions.
    
    Args:
        image: [H, W, 3] numpy array - RGB image
        boxes: [N, 4] numpy array - bounding boxes
        labels: [N] numpy array - class labels
        associations: dict {para_idx: cube_idx} - predicted
        ground_truth: dict {para_idx: cube_idx} - ground truth (optional)
        save_path: str - path to save visualization
    """
    vis_image = image.copy()
    
    # Draw bounding boxes
    for idx, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = box.astype(int)
        
        if label == 1:  # Cube
            color = (255, 0, 0)  # Red
            text = f"C{idx}"
        else:  # Parallelepiped
            color = (0, 255, 0)  # Green
            text = f"P{idx}"
        
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis_image, text, (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Draw association lines
    for para_idx, cube_idx in associations.items():
        para_box = boxes[para_idx]
        cube_box = boxes[cube_idx]
        
        # Centers
        para_center = (int((para_box[0] + para_box[2]) / 2),
                      int((para_box[1] + para_box[3]) / 2))
        cube_center = (int((cube_box[0] + cube_box[2]) / 2),
                      int((cube_box[1] + cube_box[3]) / 2))
        
        # Color: green if correct, red if wrong
        if ground_truth is not None:
            is_correct = ground_truth.get(para_idx) == cube_idx
            line_color = (0, 255, 0) if is_correct else (0, 0, 255)
        else:
            line_color = (255, 255, 0)  # Yellow if no GT
        
        cv2.line(vis_image, para_center, cube_center, line_color, 2)
    
    # Display
    plt.figure(figsize=(14, 10))
    plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
    
    title = f"Scene Graph Predictions: {len(associations)} associations"
    if ground_truth is not None:
        correct = sum(1 for p, c in associations.items() if ground_truth.get(p) == c)
        title += f" ({correct}/{len(associations)} correct)"
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    
    plt.show()
    
    return vis_image


if __name__ == '__main__':
    # Example usage
    print("Testing Scene Graph Integration...")
    
    # Load trained GNN
    gnn_model = SceneGraphGNN(node_dim=7, edge_dim=5, hidden_dim=64, num_layers=2)
    # gnn_model.load_state_dict(torch.load('best_scene_graph_gnn.pth'))
    
    # Simulate detections
    boxes = np.array([
        [100, 100, 150, 150],  # Cube 0
        [200, 200, 250, 250],  # Cube 1
        [120, 80, 140, 95],    # Para 2 (should match cube 0)
        [220, 180, 240, 195],  # Para 3 (should match cube 1)
    ])
    labels = np.array([1, 1, 2, 2])
    masks = None
    image_size = (480, 640)
    
    # Predict
    associations, affinity_matrix = predict_associations_with_gnn(
        gnn_model, boxes, labels, masks, image_size, device='cpu'
    )
    
    print(f"Predicted associations: {associations}")
    print(f"Affinity matrix shape: {affinity_matrix.shape}")
    print(f"Affinity matrix:\n{affinity_matrix}")
    
    print("\n✅ Scene Graph Integration test passed!")
