import json
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict
import glob

# Constants
BASE_DIR = r"c:\Users\NeverGonnaGiveYouUp\My project\strawberry_peduncle_segmentation\dataset"
ANNOTATIONS_PATH = os.path.join(BASE_DIR, "annotations.json")
MASKS_DIR = os.path.join(BASE_DIR, "masks")
IMAGES_DIR = os.path.join(BASE_DIR, "images")

# Categories
CATEGORIES = {
    1: "RED CUBE",
    2: "GREEN PARA"
}

COLORS = {
    1: (255, 0, 0),   # Red
    2: (0, 255, 0)    # Green
}

# Helper to find BBox
def get_bbox_from_mask(mask_img_np, seg_color):
    obj_mask = np.all(mask_img_np == seg_color, axis=2).astype(np.uint8)
    if obj_mask.sum() == 0: return None
    import cv2
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(obj_mask, connectivity=8)
    if num_labels <= 1: return None
    largest_label = 1
    max_area = stats[1, cv2.CC_STAT_AREA]
    for i in range(2, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > max_area:
            max_area = stats[i, cv2.CC_STAT_AREA]
            largest_label = i
            
    # Increased Noise Threshold: 50
    if max_area < 50: return None
    
    obj_mask = (labels_im == largest_label).astype(np.uint8)
    pos = np.where(obj_mask)
    xmin, xmax = np.min(pos[1]), np.max(pos[1])
    ymin, ymax = np.min(pos[0]), np.max(pos[0])
    center = centroids[largest_label]
    return (xmin, ymin, xmax, ymax), center

def visualize_sample():
    print(f"Loading annotations from {ANNOTATIONS_PATH}...")
    with open(ANNOTATIONS_PATH, 'r') as f:
        data = json.load(f)

    # Group annotations
    anns_by_img = defaultdict(list)
    for ann in data['annotations']:
        anns_by_img[ann['image_id']].append(ann)
        
    img_info = data['images'][0]
    img_id = img_info['id']
    filename = img_info['file_name']
    
    img_path = os.path.join(IMAGES_DIR, filename)
    img = Image.open(img_path).convert("RGB")
    mask_path = os.path.join(MASKS_DIR, filename)
    mask_img = np.array(Image.open(mask_path).convert("RGB"))
    
    draw = ImageDraw.Draw(img)
    anns = anns_by_img[img_id]
    
    # Pass 1: Collect valid objects
    cube_objects = {}
    valid_objects = {}
    
    for ann in anns:
        res = get_bbox_from_mask(mask_img, ann['segmentation_color'])
        if res:
             valid_objects[ann['instance_id']] = (res, ann['category_id'])
             if ann['category_id'] == 1:
                 cube_objects[ann['instance_id']] = res
        else:
             print(f"Skipped obj {ann['instance_id']} (Noise)")

    # Pass 2: Draw
    for ann in anns:
        if ann['instance_id'] not in valid_objects: continue
        
        ((xmin, ymin, xmax, ymax), center), cat_id = valid_objects[ann['instance_id']]
        
        label = CATEGORIES.get(cat_id, "Unknown")
        color = COLORS.get(cat_id, (255, 255, 255))
        
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)
        draw.text((xmin, ymin - 10), f"{label} {ann['instance_id']}", fill=color)
        
        # SANITIZED ASSOCIATION
        if cat_id == 2 and ann['parent_id'] > 0:
            # Look in CUBE map
            if ann['parent_id'] in cube_objects:
                pcx, pcy = cube_objects[ann['parent_id']][1]
                ccx, ccy = center
                
                dist = ((pcx-ccx)**2 + (pcy-ccy)**2)**0.5
                if dist < 200:
                    draw.line([ccx, ccy, pcx, pcy], fill=(255, 255, 0), width=2)
                    print(f"Drawn link Child {ann['instance_id']} -> Parent {ann['parent_id']}")
                else:
                    print(f"REMOVED BAD LINK Child {ann['instance_id']} -> Parent {ann['parent_id']} (Dist {dist:.1f})")
            else:
                print(f"Parent {ann['parent_id']} for Child {ann['instance_id']} not found (or filtered)")

    img.save("visual_check_sanitized.png")
    print("Saved visual_check_sanitized.png")

if __name__ == "__main__":
    visualize_sample()
