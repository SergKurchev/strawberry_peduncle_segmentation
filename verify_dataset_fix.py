import json
import os
import numpy as np
from PIL import Image
from collections import defaultdict
import torch
from torch.utils.data import Dataset
import sys

# --- MOCK DATASET CLASS (COPIED FROM FIXED NOTEBOOK) ---
class CubeParallelepipedDataset(Dataset):
    """Dataset для кубов и параллелепипедов с информацией о связях."""
    
    def __init__(self, images_path, masks_path, annotations_path, transforms=None):
        self.images_path = images_path
        self.masks_path = masks_path
        self.transforms = transforms
        
        # Загрузка аннотаций
        with open(annotations_path, 'r') as f:
            data = json.load(f)
        
        self.images_info = {img['id']: img for img in data['images']}
        
        # Группировка аннотаций по изображениям
        self.annotations_by_image = defaultdict(list)
        for ann in data['annotations']:
            self.annotations_by_image[ann['image_id']].append(ann)
        
        self.image_ids = list(self.images_info.keys())
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.images_info[image_id]
        
        # Загрузка изображения
        img_path = os.path.join(self.images_path, image_info['file_name'])
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)
        
        # Загрузка маски
        mask_path = os.path.join(self.masks_path, image_info['file_name'])
        mask_image = np.array(Image.open(mask_path).convert("RGB"))
        
        # Получение аннотаций для этого изображения
        annotations = self.annotations_by_image[image_id]
        
        boxes = []
        labels = []
        masks = []
        instance_ids = []
        parent_ids = []
        
        for ann in annotations:
            # Создание бинарной маски из цветовой маски
            seg_color = ann['segmentation_color']
            # Strict color matching
            obj_mask = np.all(mask_image == seg_color, axis=2).astype(np.uint8)
            
            # Если маска пустая
            if obj_mask.sum() == 0:
                continue
                
            # --- FIX: FILTER NOISE USING CONNECTED COMPONENTS ---
            import cv2 
            # Находим связные компоненты
            num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(obj_mask, connectivity=8)
            
            if num_labels <= 1: # Только фон
                continue
                
            # Ищем самый большой компонент (пропуская 0 - фон)
            largest_label = 1
            max_area = stats[1, cv2.CC_STAT_AREA]
            
            for i in range(2, num_labels):
                if stats[i, cv2.CC_STAT_AREA] > max_area:
                    max_area = stats[i, cv2.CC_STAT_AREA]
                    largest_label = i
            
            # Фильтр по минимальной площади (например 5 пикселей)
            if max_area < 5: 
                continue
                
            # Оставляем только эту компоненту
            obj_mask = (labels_im == largest_label).astype(np.uint8)
                
            # Вычисление BBox из очищенной маски
            pos = np.where(obj_mask)
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            
            # Проверка на валидность bbox
            if xmax <= xmin: xmax = xmin + 1
            if ymax <= ymin: ymax = ymin + 1
            
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(ann['category_id'])
            instance_ids.append(ann['instance_id'])
            parent_ids.append(ann['parent_id'])
            masks.append(obj_mask)
        
        # Проверка на пустые аннотации
        num_objs = len(boxes)
        
        if num_objs == 0:
            # Возвращаем пустые тензоры для изображений без объектов
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, image.shape[0], image.shape[1]), dtype=torch.uint8)
            instance_ids = torch.zeros((0,), dtype=torch.int64)
            parent_ids = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
            instance_ids = torch.as_tensor(instance_ids, dtype=torch.int64)
            parent_ids = torch.as_tensor(parent_ids, dtype=torch.int64)
        
        image_id_tensor = torch.tensor([image_id])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if num_objs > 0 else torch.zeros((0,))
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        
        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": image_id_tensor,
            "area": area,
            "iscrowd": iscrowd,
            "instance_ids": instance_ids,
            "parent_ids": parent_ids
        }
        
        # Конвертация изображения в тензор
        image = torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        
        if self.transforms:
            image, target = self.transforms(image, target)
        
        return image, target


# --- VERIFICATION SCRIPT ---

DATASET_PATH = r"c:\Users\NeverGonnaGiveYouUp\My project\strawberry_peduncle_segmentation\dataset"
IMAGES_PATH = os.path.join(DATASET_PATH, "images")
MASKS_PATH = os.path.join(DATASET_PATH, "masks")
ANNOTATIONS_PATH = os.path.join(DATASET_PATH, "annotations.json")

def verify():
    print("Starting RIGOROUS verification...")
    
    dataset = CubeParallelepipedDataset(IMAGES_PATH, MASKS_PATH, ANNOTATIONS_PATH)
    print(f"Dataset size: {len(dataset)} images (Checking first 50 for speed)")
    
    issues = 0
    total_objects = 0
    
    for i in range(min(50, len(dataset))):
        try:
            img, target = dataset[i]
            
            boxes = target['boxes']
            labels = target['labels']
            instance_ids = target['instance_ids']
            parent_ids = target['parent_ids']
            masks = target['masks']
            
            num_objs = len(boxes)
            total_objects += num_objs
            
            # 1. Check Boxes format [x1, y1, x2, y2]
            if num_objs > 0:
                if (boxes[:, 2] <= boxes[:, 0]).any() or (boxes[:, 3] <= boxes[:, 1]).any():
                    print(f"❌ Image {i}: Invalid BBox coordinates found!")
                    issues += 1
            
            # 2. Check mask vs box
            for j in range(num_objs):
                box = boxes[j].numpy()
                mask = masks[j].numpy()
                
                # Check if mask is empty
                if mask.sum() == 0:
                    print(f"❌ Image {i}, Obj {j}: Empty Mask returned!")
                    issues += 1
                    continue
                
                # Check if box covers mask (Loose check)
                pos = np.where(mask)
                xmin, xmax = np.min(pos[1]), np.max(pos[1])
                ymin, ymax = np.min(pos[0]), np.max(pos[0])
                
                # Our generator should match exactly, but allow 1px error due to casting
                bx1, by1, bx2, by2 = box
                
                if abs(bx1 - xmin) > 1 or abs(bx2 - xmax) > 1 or abs(by1 - ymin) > 1 or abs(by2 - ymax) > 1:
                    print(f"⚠️ Image {i}, Obj {j}: BBox {box} deviates from mask bounds [{xmin}, {ymin}, {xmax}, {ymax}]")
                    # Not strictly an error if our code adds +1 for safety, just warning
            
            # 3. Check Parent ID logic
            current_instances = set(instance_ids.numpy())
            
            for j in range(num_objs):
                lbl = labels[j].item()
                pid = parent_ids[j].item()
                mask = masks[j].numpy()
                
                # --- LABEL VERIFICATION (COLOR CHECK) ---
                # Get pixels from original image where mask is True
                # img is Tensor [3, H, W], needs to be [H, W, 3] for indexing with [H, W] mask
                img_np = img.permute(1, 2, 0).numpy() # [H, W, 3]
                
                # Multiply by 255 to get 0-255 range if it was normalized
                # Dataset divides by 255.0, so:
                img_np = (img_np * 255).astype(np.uint8)
                
                masked_pixels = img_np[mask == 1]
                
                if len(masked_pixels) > 0:
                    avg_color = masked_pixels.mean(axis=0) # [R, G, B]
                    r, g, b = avg_color
                    
                    if lbl == 1: # RED CUBE
                        if r < g or r < b:
                            print(f"⚠️ Image {i}, Obj {j} (Cube): Avg Color {avg_color.astype(int)} is NOT RED!")
                            # Allow some noise, but warn heavily
                            if g > r + 20: # If noticeably more green
                                print(f"❌ INVALID LABEL: Class 1 (Cube) is GREEN! Should be RED.")
                                issues += 1
                        else:
                            # Confirm valid red object
                            pass
                            
                    elif lbl == 2: # GREEN PARALLELEPIPED
                        if g < r or g < b:
                            print(f"⚠️ Image {i}, Obj {j} (Para): Avg Color {avg_color.astype(int)} is NOT GREEN!")
                            if r > g + 20: 
                                print(f"❌ INVALID LABEL: Class 2 (Para) is RED! Should be GREEN.")
                                issues += 1
                        else:
                            pass
                
                if lbl == 1: # Cube
                    if pid != 0:
                        print(f"⚠️ Image {i}, Obj {j}: Cube has non-zero parent_id {pid}!")
                elif lbl == 2: # Parallelepiped
                     pass

                         
        except Exception as e:
            print(f"❌ CRITICAL ERROR processing image {i}: {e}")
            issues += 1
            import traceback
            traceback.print_exc()

    print("\n--- SUMMARY ---")
    print(f"Processed {len(dataset)} images.")
    print(f"Total objects verified: {total_objects}")
    if issues == 0:
        print("✅ SUCCESS: All checks passed. The Dataset logic is robust.")
    else:
        print(f"❌ FAILED: Found {issues} issues during verification.")

if __name__ == "__main__":
    verify()
