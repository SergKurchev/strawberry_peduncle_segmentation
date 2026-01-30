"""
Dataset for training AffinityNet.
Works directly with COCO annotations - NO pretrained model needed.
"""

import os
import json
import numpy as np
import cv2
from torch.utils.data import Dataset
import torch

from affinity_net import compute_spatial_features


class AffinityDataset(Dataset):
    """
    Dataset for training affinity network.
    Generates positive/negative pairs from COCO annotations.
    """
    
    def __init__(self, coco_data, dataset_path, split='train', train_ratio=0.8, neg_positive_ratio=2):
        """
        Args:
            coco_data: Loaded COCO annotations (dict)
            dataset_path: Path to dataset folder
            split: 'train' or 'val'
            train_ratio: Ratio of training data
            neg_positive_ratio: Ratio of negative to positive samples
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
        
        # Create image_id to annotations mapping
        self.annotations = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id in self.image_id_to_info:
                if img_id not in self.annotations:
                    self.annotations[img_id] = []
                self.annotations[img_id].append(ann)
        
        # Generate training pairs
        self.pairs = []
        self._generate_pairs(neg_positive_ratio)
        
        print(f"{split.upper()} dataset: {len(self.pairs)} pairs from {len(self.images)} images")
    
    def _generate_pairs(self, neg_positive_ratio):
        """Generate positive and negative training pairs."""
        
        for image_info in self.images:
            image_id = image_info['id']
            
            if image_id not in self.annotations:
                continue
            
            anns = self.annotations[image_id]
            
            # Separate by category
            paras = [ann for ann in anns if ann['category_id'] == 2]
            cubes = [ann for ann in anns if ann['category_id'] == 1]
            
            if len(paras) == 0 or len(cubes) == 0:
                continue
            
            # === POSITIVE PAIRS ===
            for para in paras:
                parent_id = para['parent_id']
                
                # Find matching cube by instance_id
                matching_cube = None
                for cube in cubes:
                    if cube['instance_id'] == parent_id:
                        matching_cube = cube
                        break
                
                if matching_cube is not None:
                    self.pairs.append({
                        'image_id': image_id,
                        'para': para,
                        'cube': matching_cube,
                        'label': 1  # Positive
                    })
            
            # === NEGATIVE PAIRS ===
            # For each para, create negative pairs with non-matching cubes
            num_positives = len([p for p in self.pairs if p['image_id'] == image_id])
            num_negatives_needed = num_positives * neg_positive_ratio
            
            negative_count = 0
            for para in paras:
                parent_id = para['parent_id']
                
                for cube in cubes:
                    if cube['instance_id'] != parent_id:
                        self.pairs.append({
                            'image_id': image_id,
                            'para': para,
                            'cube': cube,
                            'label': 0  # Negative
                        })
                        negative_count += 1
                        
                        if negative_count >= num_negatives_needed:
                            break
                
                if negative_count >= num_negatives_needed:
                    break
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        """
        Returns:
            spatial_features: [5] - geometric features
            label: 0 or 1 - ground truth association
        """
        pair = self.pairs[idx]
        
        image_id = pair['image_id']
        para_ann = pair['para']
        cube_ann = pair['cube']
        label = pair['label']
        
        # Get image info
        image_info = self.image_id_to_info[image_id]
        image_size = (image_info['height'], image_info['width'])
        
        # Get bounding boxes
        para_bbox = para_ann['bbox']  # [x, y, w, h] COCO format
        cube_bbox = cube_ann['bbox']
        
        # Convert COCO bbox [x, y, w, h] to [x1, y1, x2, y2]
        para_bbox_xyxy = [
            para_bbox[0], 
            para_bbox[1], 
            para_bbox[0] + para_bbox[2], 
            para_bbox[1] + para_bbox[3]
        ]
        cube_bbox_xyxy = [
            cube_bbox[0],
            cube_bbox[1],
            cube_bbox[0] + cube_bbox[2],
            cube_bbox[1] + cube_bbox[3]
        ]
        
        # Load masks
        mask_path = os.path.join(self.masks_path, image_info['file_name'])
        mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        
        # Extract binary masks using segmentation_color
        para_seg_color = para_ann['segmentation_color']
        cube_seg_color = cube_ann['segmentation_color']
        
        # Create binary masks
        if len(mask_img.shape) == 3:
            # RGB encoding: R=instance_id, G=category_id
            para_mask = (mask_img[:, :, 2] == para_seg_color[0]) & \
                       (mask_img[:, :, 1] == para_seg_color[1])
            cube_mask = (mask_img[:, :, 2] == cube_seg_color[0]) & \
                       (mask_img[:, :, 1] == cube_seg_color[1])
        else:
            # Grayscale - use instance_id only
            para_mask = mask_img == para_ann['instance_id']
            cube_mask = mask_img == cube_ann['instance_id']
        
        para_mask = para_mask.astype(np.uint8)
        cube_mask = cube_mask.astype(np.uint8)
        
        # Compute spatial features
        spatial_features = compute_spatial_features(
            para_bbox_xyxy,
            cube_bbox_xyxy,
            para_mask,
            cube_mask,
            image_size
        )
        
        return torch.from_numpy(spatial_features), torch.tensor(label, dtype=torch.float32)


if __name__ == '__main__':
    # Test dataset loading
    import json
    
    dataset_path = 'strawberry_peduncle_segmentation/dataset'
    
    with open(os.path.join(dataset_path, 'annotations.json'), 'r') as f:
        coco_data = json.load(f)
    
    train_dataset = AffinityDataset(coco_data, dataset_path, split='train')
    val_dataset = AffinityDataset(coco_data, dataset_path, split='val')
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Test loading a sample
    features, label = train_dataset[0]
    print(f"\nSample features shape: {features.shape}")
    print(f"Features: {features}")
    print(f"Label: {label.item()}")
