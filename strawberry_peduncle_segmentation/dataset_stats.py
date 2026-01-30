#!/usr/bin/env python3
"""
Dataset Statistics Analyzer
Анализирует датасет и выводит статистику для выявления проблем генерации
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image

def load_annotations(json_path):
    """Загрузка аннотаций из JSON файла"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_dataset(dataset_path):
    """Главная функция анализа датасета"""
    dataset_path = Path(dataset_path)
    annotations_path = dataset_path / "annotations.json"
    images_path = dataset_path / "images"
    masks_path = dataset_path / "masks"
    
    if not annotations_path.exists():
        print(f"❌ Файл annotations.json не найден: {annotations_path}")
        return
    
    data = load_annotations(annotations_path)
    
    print("=" * 60)
    print("📊 СТАТИСТИКА ДАТАСЕТА")
    print("=" * 60)
    
    # === 1. Общая информация ===
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])
    
    print(f"\n📁 Путь: {dataset_path}")
    print(f"🖼️  Изображений: {len(images)}")
    print(f"📝 Аннотаций: {len(annotations)}")
    print(f"🏷️  Категорий: {len(categories)}")
    
    for cat in categories:
        print(f"   - {cat['id']}: {cat['name']}")
    
    # === 2. Размеры изображений ===
    if images:
        sizes = set((img['width'], img['height']) for img in images)
        print(f"\n📐 Размеры изображений:")
        for w, h in sizes:
            count = sum(1 for img in images if img['width'] == w and img['height'] == h)
            print(f"   {w}x{h}: {count} изображений")
    
    # === 3. Статистика по категориям ===
    category_counts = defaultdict(int)
    category_names = {cat['id']: cat['name'] for cat in categories}
    
    for ann in annotations:
        category_counts[ann['category_id']] += 1
    
    print(f"\n📊 Объектов по категориям:")
    for cat_id, count in sorted(category_counts.items()):
        name = category_names.get(cat_id, f"unknown_{cat_id}")
        print(f"   {name} (id={cat_id}): {count}")
    
    # === 4. Статистика bbox ===
    bbox_widths = []
    bbox_heights = []
    bbox_areas = []
    
    for ann in annotations:
        bbox = ann.get('bbox', [0, 0, 0, 0])
        if len(bbox) >= 4:
            bbox_widths.append(bbox[2])
            bbox_heights.append(bbox[3])
            bbox_areas.append(ann.get('area', bbox[2] * bbox[3]))
    
    if bbox_widths:
        print(f"\n📏 Статистика Bounding Box:")
        print(f"   Ширина:  min={min(bbox_widths):.1f}, max={max(bbox_widths):.1f}, avg={np.mean(bbox_widths):.1f}")
        print(f"   Высота:  min={min(bbox_heights):.1f}, max={max(bbox_heights):.1f}, avg={np.mean(bbox_heights):.1f}")
        print(f"   Площадь: min={min(bbox_areas):.1f}, max={max(bbox_areas):.1f}, avg={np.mean(bbox_areas):.1f}")
    
    # === 5. Проверка parent_id связей ===
    cubes = [ann for ann in annotations if ann['category_id'] == 1]
    parallelepipeds = [ann for ann in annotations if ann['category_id'] == 2]
    
    print(f"\n🔗 Связи parent_id:")
    print(f"   Кубов: {len(cubes)}")
    print(f"   Параллелепипедов: {len(parallelepipeds)}")
    
    # Проверка корректности parent_id
    orphaned = 0
    valid_links = 0
    cube_instance_ids = set(ann['instance_id'] for ann in cubes)
    
    for para in parallelepipeds:
        if para['parent_id'] in cube_instance_ids:
            valid_links += 1
        else:
            orphaned += 1
    
    print(f"   Корректных связей: {valid_links}")
    if orphaned > 0:
        print(f"   ⚠️ Сирот (без родителя): {orphaned}")
    
    # === 6. Объектов на изображение ===
    objects_per_image = defaultdict(int)
    for ann in annotations:
        objects_per_image[ann['image_id']] += 1
    
    if objects_per_image:
        counts = list(objects_per_image.values())
        print(f"\n🔢 Объектов на изображение:")
        print(f"   min={min(counts)}, max={max(counts)}, avg={np.mean(counts):.1f}")
    
    # === 7. Проверка segmentation_color ===
    colors = set()
    for ann in annotations:
        color = tuple(ann.get('segmentation_color', [0, 0, 0]))
        colors.add(color)
    
    print(f"\n🎨 Уникальных цветов сегментации: {len(colors)}")
    
    # Проверяем чёрный цвет
    black_annotations = [ann for ann in annotations if ann.get('segmentation_color') == [0, 0, 0]]
    if black_annotations:
        print(f"   ⚠️ Аннотаций с чёрным цветом (0,0,0): {len(black_annotations)}")
    
    # === 8. Проверка файлов ===
    print(f"\n📂 Проверка файлов:")
    
    # Проверяем изображения
    missing_images = 0
    if images_path.exists():
        for img in images:
            img_file = images_path / img['file_name']
            if not img_file.exists():
                missing_images += 1
        print(f"   Изображения: {len(images) - missing_images}/{len(images)} существуют")
        if missing_images > 0:
            print(f"   ⚠️ Отсутствующих: {missing_images}")
    else:
        print(f"   ❌ Папка images не найдена")
    
    # Проверяем маски
    missing_masks = 0
    if masks_path.exists():
        for img in images:
            mask_file = masks_path / img['file_name']
            if not mask_file.exists():
                missing_masks += 1
        print(f"   Маски: {len(images) - missing_masks}/{len(images)} существуют")
        if missing_masks > 0:
            print(f"   ⚠️ Отсутствующих: {missing_masks}")
    else:
        print(f"   ❌ Папка masks не найдена")
    
    # === 9. Анализ масок (если существуют) ===
    if masks_path.exists():
        print(f"\n🎭 Анализ масок:")
        black_masks = 0
        non_black_masks = 0
        sample_masks = list(masks_path.glob("*.png"))[:min(10, len(list(masks_path.glob("*.png"))))]
        
        for mask_file in sample_masks:
            try:
                mask = np.array(Image.open(mask_file))
                if mask.max() == 0:
                    black_masks += 1
                else:
                    non_black_masks += 1
            except Exception as e:
                print(f"   ❌ Ошибка чтения {mask_file.name}: {e}")
        
        total_checked = black_masks + non_black_masks
        if total_checked > 0:
            print(f"   Проверено масок: {total_checked}")
            print(f"   Чёрных (пустых): {black_masks}")
            print(f"   С данными: {non_black_masks}")
            if black_masks > 0:
                print(f"   ⚠️ ПРОБЛЕМА: Маски полностью чёрные!")
    
    # === 10. Выявление проблем ===
    print(f"\n" + "=" * 60)
    print("🔍 ВЫЯВЛЕННЫЕ ПРОБЛЕМЫ:")
    print("=" * 60)
    
    problems = []
    
    # Проверка размера изображений
    if images:
        w, h = images[0]['width'], images[0]['height']
        if w < 1024 or h < 1024:
            problems.append(f"⚠️ Размер изображений {w}x{h} < 1024x1024 (рекомендуется)")
    
    # Проверка количества изображений
    if len(images) < 1000:
        problems.append(f"⚠️ Только {len(images)} изображений (рекомендуется 1000+)")
    
    # Проверка маленьких bbox
    tiny_bboxes = sum(1 for a in bbox_areas if a < 10)
    if tiny_bboxes > 0:
        problems.append(f"⚠️ {tiny_bboxes} очень маленьких bbox (area < 10 пикселей)")
    
    # Проверка сирот
    if orphaned > 0:
        problems.append(f"⚠️ {orphaned} параллелепипедов без валидного parent_id")
    
    # Проверка чёрных масок
    if masks_path.exists() and black_masks > 0:
        problems.append(f"❌ {black_masks}/{total_checked} проверенных масок полностью чёрные!")
    
    if problems:
        for p in problems:
            print(f"   {p}")
    else:
        print("   ✅ Проблем не обнаружено!")
    
    print("\n" + "=" * 60)

def main():
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        # По умолчанию ищем в текущей директории
        if os.path.exists("dataset"):
            dataset_path = "dataset"
        elif os.path.exists("strawberry_peduncle_segmentation/dataset"):
            dataset_path = "strawberry_peduncle_segmentation/dataset"
        else:
            dataset_path = "."
    
    analyze_dataset(dataset_path)

if __name__ == "__main__":
    main()
