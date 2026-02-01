"""
Dataset for training Scene Graph GNN.
Each sample is a complete scene graph with ground truth relationships.
"""

import os
import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from scene_graph import SceneGraph


class SceneGraphDataset(Dataset):
    """
    Dataset for training Scene Graph GNN on object relationships.
    
    Each sample consists of:
    - Graph structure (nodes + edges)
    - Ground truth edge labels (which para belongs to which cube)
    """
    
    def __init__(self, coco_data, dataset_path, split='train', train_ratio=0.8):
        """
        Args:
            coco_data: Loaded COCO annotations dict
            dataset_path: Path to dataset directory
            split: 'train' or 'val'
            train_ratio: Ratio for train/val split
        """
        self.dataset_path = dataset_path
        self.images_path = os.path.join(dataset_path, 'images')
        self.masks_path = os.path.join(dataset_path, 'masks')
        
        # Split data
        all_images = coco_data['images']
        split_idx = int(len(all_images) * train_ratio)
        
        if split == 'train':
            self.images = all_images[:split_idx]
        else:
            self.images = all_images[split_idx:]
        
        self.image_id_to_info = {img['id']: img for img in self.images}
        
        # Create annotations mapping
        self.annotations = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id in self.image_id_to_info:
                if img_id not in self.annotations:
                    self.annotations[img_id] = []
                self.annotations[img_id].append(ann)
        
        # Filter images that have both parallelepipeds and cubes
        self.valid_images = []
        for img_id in self.annotations:
            anns = self.annotations[img_id]
            has_cube = any(a['category_id'] == 1 for a in anns)
            has_para = any(a['category_id'] == 2 for a in anns)
            if has_cube and has_para:
                self.valid_images.append(img_id)
        
        print(f"{split.upper()} dataset: {len(self.valid_images)} valid scenes "
              f"({len(self.images)} total images)")
    
    def __len__(self):
        return len(self.valid_images)
    
    def __getitem__(self, idx):
        """
        Returns:
            graph_data: dict with x, edge_index, edge_attr
            edge_labels: [E] tensor - ground truth for each edge (0 or 1)
            metadata: dict with image info for visualization
        """
        image_id = self.valid_images[idx]
        img_info = self.image_id_to_info[image_id]
        anns = self.annotations[image_id]
        
        # === EXTRACT DETECTIONS FROM ANNOTATIONS ===
        boxes = []
        labels = []
        masks = []
        instance_ids = []
        parent_ids = []
        
        for ann in anns:
            # Bounding box (convert COCO xywh to xyxy)
            bbox = ann['bbox']
            bbox_xyxy = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            boxes.append(bbox_xyxy)
            
            labels.append(ann['category_id'])
            instance_ids.append(ann['instance_id'])
            parent_ids.append(ann.get('parent_id', -1))
            
            # Load mask
            mask_path = os.path.join(self.masks_path, img_info['file_name'])
            mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            
            seg_color = ann['segmentation_color']
            
            if len(mask_img.shape) == 3:
                # RGB encoding
                mask = (mask_img[:, :, 2] == seg_color[0]) & \
                       (mask_img[:, :, 1] == seg_color[1])
            else:
                # Grayscale
                mask = mask_img == ann['instance_id']
            
            masks.append(mask.astype(np.uint8))
        
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        masks = np.array(masks)
        
        # === BUILD SCENE GRAPH ===
        graph = SceneGraph(boxes, labels, masks, (img_info['height'], img_info['width']))
        graph_data = graph.to_torch()
        
        # === CREATE EDGE LABELS ===
        # For each edge (para->cube), label is 1 if this para belongs to this cube
        edge_labels = []
        
        para_indices = np.where(labels == 2)[0]
        cube_indices = np.where(labels == 1)[0]
        
        for para_idx in para_indices:
            para_parent_id = parent_ids[para_idx]
            
            for cube_idx in cube_indices:
                cube_instance_id = instance_ids[cube_idx]
                
                # Ground truth: 1 if para's parent matches cube's instance ID
                label = 1.0 if para_parent_id == cube_instance_id else 0.0
                edge_labels.append(label)
        
        edge_labels = torch.tensor(edge_labels, dtype=torch.float32)
        
        # === METADATA FOR VISUALIZATION ===
        metadata = {
            'image_id': image_id,
            'image_path': os.path.join(self.images_path, img_info['file_name']),
            'boxes': boxes,
            'labels': labels,
            'instance_ids': instance_ids,
            'parent_ids': parent_ids
        }
        
        return graph_data, edge_labels, metadata


def collate_scene_graphs(batch):
    """
    Collate function for batching scene graphs.
    
    Since graphs have different sizes, we need custom batching.
    Returns a list of (graph_data, edge_labels, metadata) tuples.
    """
    # For simplicity, return as list (batch size will be processed sequentially)
    # In production, use torch_geometric.data.Batch for efficient batching
    return batch


if __name__ == '__main__':
    # Test dataset loading
    import json
    from torch.utils.data import DataLoader
    
    dataset_path = 'strawberry_peduncle_segmentation/dataset'
    
    with open(os.path.join(dataset_path, 'annotations.json'), 'r') as f:
        coco_data = json.load(f)
    
    train_dataset = SceneGraphDataset(coco_data, dataset_path, split='train')
    val_dataset = SceneGraphDataset(coco_data, dataset_path, split='val')
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)} scenes")
    print(f"  Val: {len(val_dataset)} scenes")
    
    # Test loading a sample
    graph_data, edge_labels, metadata = train_dataset[0]
    
    print(f"\nSample 0:")
    print(f"  Image ID: {metadata['image_id']}")
    print(f"  Nodes: {graph_data['x'].shape[0]}")
    print(f"  Edges: {graph_data['edge_index'].shape[1]}")
    print(f"  Edge labels: {edge_labels.shape}")
    print(f"  Positive edges: {edge_labels.sum().item():.0f}")
    print(f"  Negative edges: {(edge_labels == 0).sum().item():.0f}")
    
    # Test DataLoader
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,
                              collate_fn=collate_scene_graphs)
    
    batch = next(iter(train_loader))
    print(f"\nBatch size: {len(batch)}")
    
    print("\n✅ SceneGraphDataset test passed!")
