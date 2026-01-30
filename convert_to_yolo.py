#!/usr/bin/env python3
"""
Конвертация COCO Instance Segmentation датасета в YOLO формат.
Создаёт текстовые аннотации с полигонами для YOLOv11-seg.
Оригинальный формат (PNG masks + COCO JSON) остаётся без изменений.
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


def mask_to_polygons(mask_path, instance_id, category_id):
    """
    Конвертирует PNG маску в YOLO полигоны.
    
    Args:
        mask_path: путь к PNG маске
        instance_id: ID инстанса
        category_id: ID категории
    
    Returns:
        list of polygons: каждый полигон это список нормализованных координат
    """
    # Читаем маску
    mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    
    if mask_img is None:
        print(f"⚠️ Warning: Could not read mask {mask_path}")
        return []
    
    # Проверяем формат RGB (channel encoding)
    # R канал = instance_id, G канал = category_id
    if len(mask_img.shape) == 3:
        r_channel = mask_img[: , :, 2]  # OpenCV uses BGR
        g_channel = mask_img[:, :, 1]
        
        # Создаём бинарную маску для данного instance + category
        binary_mask = (r_channel == instance_id) & (g_channel == category_id)
    else:
        # Grayscale mask - используем напрямую
        binary_mask = mask_img == instance_id
    
    # Приводим к uint8
    binary_mask = binary_mask.astype(np.uint8) * 255
    
    # Находим контуры
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return []
    
    height, width = binary_mask.shape
    polygons = []
    
    for contour in contours:
        # Минимум 3 точки для полигона
        if len(contour) < 3:
            continue
        
        # Упрощаем полигон (Douglas-Peucker)
        epsilon = 0.001 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) < 3:
            continue
        
        # Нормализуем координаты (0-1)
        polygon = []
        for point in approx:
            x_norm = point[0][0] / width
            y_norm = point[0][1] / height
            polygon.extend([x_norm, y_norm])
        
        polygons.append(polygon)
    
    return polygons


def convert_coco_to_yolo(dataset_path, output_path=None):
    """
    Конвертирует COCO датасет в YOLO формат.
    
    Args:
        dataset_path: путь к папке с датасетом (содержит images/, masks/, annotations.json)
        output_path: путь для сохранения YOLO датасета (по умолчанию: dataset_path/yolo)
    """
    dataset_path = Path(dataset_path)
    
    if output_path is None:
        output_path = dataset_path / "yolo"
    else:
        output_path = Path(output_path)
    
    # Пути
    annotations_path = dataset_path / "annotations.json"
    images_path = dataset_path / "images"
    masks_path = dataset_path / "masks"
    
    # Создаём структуру YOLO
    yolo_images_dir = output_path / "images"
    yolo_labels_dir = output_path / "labels"
    yolo_images_dir.mkdir(parents=True, exist_ok=True)
    yolo_labels_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📂 Reading annotations from: {annotations_path}")
    
    # Загружаем COCO аннотации
    with open(annotations_path, 'r') as f:
        coco_data = json.load(f)
    
    # Маппинг категорий (COCO ID -> YOLO class index)
    # В YOLO классы начинаются с 0
    category_mapping = {}
    for idx, category in enumerate(coco_data['categories']):
        category_mapping[category['id']] = idx
    
    print(f"\n📋 Category Mapping (COCO ID -> YOLO Class):")
    for category in coco_data['categories']:
        yolo_class = category_mapping[category['id']]
        print(f"   {category['id']} ({category['name']}) -> {yolo_class}")
    
    # Создаём dataset.yaml
    yaml_content = f"""# YOLO Dataset Configuration
# Generated from COCO format

path: {output_path.absolute()}
train: images  # Все изображения для обучения
val: images    # Можно разделить на train/val позже

# Classes
names:
"""
    
    for category in sorted(coco_data['categories'], key=lambda x: category_mapping[x['id']]):
        yolo_class = category_mapping[category['id']]
        yaml_content += f"  {yolo_class}: {category['name']}\n"
    
    yaml_path = output_path / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\n✅ Created dataset.yaml: {yaml_path}")
    
    # Группируем аннотации по изображениям
    image_annotations = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in image_annotations:
            image_annotations[image_id] = []
        image_annotations[image_id].append(ann)
    
    # Конвертируем каждое изображение
    print(f"\n🔄 Converting {len(coco_data['images'])} images...")
    
    for image_info in tqdm(coco_data['images']):
        image_id = image_info['id']
        filename = image_info['file_name']
        
        # Копируем изображение (создаём символическую ссылку или копируем)
        src_image = images_path / filename
        dst_image = yolo_images_dir / filename
        
        if src_image.exists():
            if not dst_image.exists():
                # Windows: копируем файл
                import shutil
                shutil.copy2(src_image, dst_image)
        else:
            print(f"⚠️ Warning: Image not found: {src_image}")
            continue
        
        # Создаём текстовый файл с аннотациями
        label_filename = filename.replace('.png', '.txt').replace('.jpg', '.txt')
        label_path = yolo_labels_dir / label_filename
        
        # Получаем аннотации для этого изображения
        annotations = image_annotations.get(image_id, [])
        
        with open(label_path, 'w') as f:
            for ann in annotations:
                instance_id = ann['instance_id']
                category_id = ann['category_id']
                yolo_class = category_mapping[category_id]
                
                # Получаем полигоны из маски
                mask_path = masks_path / filename
                polygons = mask_to_polygons(mask_path, instance_id, category_id)
                
                # Записываем каждый полигон как отдельную строку
                for polygon in polygons:
                    # Формат YOLO: class_id x1 y1 x2 y2 ... xn yn
                    line = f"{yolo_class} " + " ".join([f"{coord:.6f}" for coord in polygon])
                    f.write(line + "\n")
    
    print(f"\n✅ Conversion complete!")
    print(f"   Images: {yolo_images_dir}")
    print(f"   Labels: {yolo_labels_dir}")
    print(f"   Config: {yaml_path}")
    print(f"\n📝 Next steps:")
    print(f"   1. Train: yolo segment train data={yaml_path} model=yolo11l-seg.pt epochs=50")
    print(f"   2. Validate: yolo segment val data={yaml_path} model=runs/segment/train/weights/best.pt")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert COCO Instance Segmentation to YOLO format")
    parser.add_argument('dataset_path', type=str, help='Path to COCO dataset folder')
    parser.add_argument('--output', type=str, default=None, help='Output path for YOLO dataset')
    
    args = parser.parse_args()
    
    convert_coco_to_yolo(args.dataset_path, args.output)
