"""
Integration utilities for using AffinityNet with segmentation models.
Works with both Mask R-CNN and YOLOv11.
"""

import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

from affinity_net import AffinityNet, compute_spatial_features_batch


def predict_associations_with_affinity(affinity_model, boxes, labels, masks, image_size, device='cuda'):
    """
    Predict associations using trained AffinityNet.
    Works with outputs from Mask R-CNN or YOLO.
    
    Args:
        affinity_model: Trained AffinityNet instance
        boxes: [N, 4] tensor - bounding boxes in xyxy format
        labels: [N] tensor - class labels (1=cube, 2=parallelepiped)
        masks: [N, H, W] tensor - binary segmentation masks  
        image_size: (H, W) tuple - image dimensions
        device: 'cuda' or 'cpu'
    
    Returns:
        associations: dict {para_idx: cube_idx} - predicted associations
        affinity_matrix: [N_para, N_cube] numpy array - affinity scores (for visualization)
    """
    affinity_model.eval()
    affinity_model = affinity_model.to(device)
    
    # Convert to numpy if tensors
    if torch.is_tensor(boxes):
        boxes = boxes.cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    if torch.is_tensor(masks):
        masks = masks.cpu().numpy()
    
    # Separate by class
    para_indices = np.where(labels == 2)[0]
    cube_indices = np.where(labels == 1)[0]
    
    if len(para_indices) == 0 or len(cube_indices) == 0:
        return {}, np.array([])
    
    # Extract sub-arrays
    para_boxes = boxes[para_indices]
    cube_boxes = boxes[cube_indices]
    para_masks = masks[para_indices] if masks is not None else None
    cube_masks = masks[cube_indices] if masks is not None else None
    
    # === COMPUTE SPATIAL FEATURES ===
    spatial_features = compute_spatial_features_batch(
        para_boxes, cube_boxes, para_masks, cube_masks, image_size
    )  # [N_para, N_cube, 5]
    
    # === AFFINITY PREDICTION ===
    spatial_tensor = torch.from_numpy(spatial_features).to(device)
    
    with torch.no_grad():
        affinity_matrix_tensor = affinity_model.predict_matrix(spatial_tensor)
        affinity_matrix = affinity_matrix_tensor.cpu().numpy()
    
    # === HUNGARIAN MATCHING ===
    row_ind, col_ind = linear_sum_assignment(-affinity_matrix)
    
    associations = {}
    threshold = 0.5
    
    for para_local_idx, cube_local_idx in zip(row_ind, col_ind):
        if affinity_matrix[para_local_idx, cube_local_idx] > threshold:
            para_global_idx = int(para_indices[para_local_idx])
            cube_global_idx = int(cube_indices[cube_local_idx])
            associations[para_global_idx] = cube_global_idx
    
    return associations, affinity_matrix


def compute_association_accuracy(predictions, ground_truth):
    """
    Compute accuracy of predicted associations.
    
    Args:
        predictions: dict {para_instance_id: predicted_cube_instance_id}
        ground_truth: dict {para_instance_id: gt_cube_instance_id}
    
    Returns:
        accuracy: float - ratio of correct predictions
        correct: int - number of correct predictions
        total: int - total number of parallelepipeds
    """
    correct = 0
    total = len(ground_truth)
    
    for para_id, gt_cube_id in ground_truth.items():
        pred_cube_id = predictions.get(para_id, None)
        
        if pred_cube_id == gt_cube_id:
            correct += 1
    
    accuracy = correct / (total + 1e-6)
    
    return accuracy, correct, total


def extract_ground_truth_associations(annotations):
    """
    Extract ground truth associations from COCO annotations.
    
    Args:
        annotations: list of annotation dicts for a single image
    
    Returns:
        gt_associations: dict {para_instance_id: cube_instance_id}
    """
    gt_associations = {}
    
    for ann in annotations:
        if ann['category_id'] == 2:  # Parallelepiped
            para_instance_id = ann['instance_id']
            cube_instance_id = ann['parent_id']
            gt_associations[para_instance_id] = cube_instance_id
    
    return gt_associations


def visualize_associations(image, boxes, labels, associations, save_path=None):
    """
    Visualize predicted associations by drawing lines between cubes and parallelepipeds.
    
    Args:
        image: [H, W, 3] numpy array - RGB image
        boxes: [N, 4] numpy array - bounding boxes
        labels: [N] numpy array - class labels
        associations: dict {para_idx: cube_idx}
        save_path: str - path to save visualization (optional)
    """
    import cv2
    import matplotlib.pyplot as plt
    
    vis_image = image.copy()
    
    # Draw bounding boxes
    for idx, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = box.astype(int)
        
        if label == 1:  # Cube
            color = (255, 0, 0)  # Red
            text = f"Cube {idx}"
        else:  # Parallelepiped
            color = (0, 255, 0)  # Green
            text = f"Para {idx}"
        
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis_image, text, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Draw association lines
    for para_idx, cube_idx in associations.items():
        para_box = boxes[para_idx]
        cube_box = boxes[cube_idx]
        
        # Center points
        para_center = ((para_box[0] + para_box[2]) // 2, 
                      (para_box[1] + para_box[3]) // 2)
        cube_center = ((cube_box[0] + cube_box[2]) // 2,
                      (cube_box[1] + cube_box[3]) // 2)
        
        cv2.line(vis_image, 
                (int(para_center[0]), int(para_center[1])),
                (int(cube_center[0]), int(cube_center[1])),
                (0, 255, 255), 2)  # Yellow line
    
    # Display
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
    plt.title(f"Associations: {len(associations)} links")
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    
    plt.show()
    
    return vis_image


# === EXAMPLE USAGE ===
if __name__ == '__main__':
    """
    Example integration with Mask R-CNN inference.
    """
    
    # Load AffinityNet
    affinity_model = AffinityNet()
    affinity_model.load_state_dict(torch.load('best_affinity_net.pth'))
    affinity_model.eval()
    
    # Simulate detections (would come from Mask R-CNN/YOLO)
    boxes = np.array([
        [100, 100, 150, 150],  # Cube 0
        [200, 200, 250, 250],  # Cube 1
        [120, 80, 140, 95],    # Para 2 (should match cube 0)
        [220, 180, 240, 195],  # Para 3 (should match cube 1)
    ])
    labels = np.array([1, 1, 2, 2])  # 1=cube, 2=para
    masks = None  # Would be actual masks
    image_size = (480, 640)
    
    # Predict associations
    associations, affinity_matrix = predict_associations_with_affinity(
        affinity_model, boxes, labels, masks, image_size
    )
    
    print(f"Predicted associations: {associations}")
    print(f"Affinity matrix:\n{affinity_matrix}")
