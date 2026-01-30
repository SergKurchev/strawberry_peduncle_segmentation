#!/usr/bin/env python3
"""
Mask Visualizer - визуализация масок сегментации с усилением контраста
"""

import sys
import os
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def visualize_mask(mask_path, output_path=None, show=True):
    """Визуализация маски с усилением контраста"""
    mask = np.array(Image.open(mask_path))
    
    print(f"📁 Файл: {mask_path}")
    print(f"📐 Размер: {mask.shape}")
    print(f"🔢 Min: {mask.min()}, Max: {mask.max()}")
    print(f"🎨 Non-zero пикселей: {np.count_nonzero(mask)}")
    
    if len(mask.shape) == 3:
        r_channel = mask[:, :, 0]
        g_channel = mask[:, :, 1]
        b_channel = mask[:, :, 2] if mask.shape[2] > 2 else np.zeros_like(r_channel)
    else:
        r_channel = mask
        g_channel = np.zeros_like(mask)
        b_channel = np.zeros_like(mask)
    
    print(f"   R (instance_id): уникальные = {np.unique(r_channel)}")
    print(f"   G (category_id): уникальные = {np.unique(g_channel)}")
    
    # Создаём усиленную визуализацию
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # 1. Оригинальная маска (очень тёмная)
    axes[0].imshow(mask)
    axes[0].set_title('Оригинал (тёмно)')
    axes[0].axis('off')
    
    # 2. Усиленная маска (умноженная на 30)
    enhanced = np.clip(mask.astype(np.float32) * 30, 0, 255).astype(np.uint8)
    axes[1].imshow(enhanced)
    axes[1].set_title('Усиленная (x30)')
    axes[1].axis('off')
    
    # 3. Цветовая карта по instance_id
    cmap = plt.cm.get_cmap('tab20', r_channel.max() + 1)
    axes[2].imshow(r_channel, cmap=cmap, vmin=0, vmax=max(7, r_channel.max()))
    axes[2].set_title(f'Instance ID (R channel)')
    axes[2].axis('off')
    
    # 4. Категории (1 = куб, 2 = параллелепипед)
    category_colors = np.zeros((*g_channel.shape, 3), dtype=np.uint8)
    category_colors[g_channel == 1] = [255, 0, 0]  # Красный для кубов
    category_colors[g_channel == 2] = [0, 255, 0]  # Зелёный для параллелепипедов
    axes[3].imshow(category_colors)
    axes[3].set_title('Категории (1=красный куб, 2=зелёный пар.)')
    axes[3].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"💾 Сохранено: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

def visualize_all_masks(dataset_path, max_masks=5):
    """Визуализация нескольких масок"""
    masks_path = Path(dataset_path) / "masks"
    output_path = Path(dataset_path) / "mask_visualizations"
    output_path.mkdir(exist_ok=True)
    
    mask_files = sorted(masks_path.glob("*.png"))[:max_masks]
    
    print(f"\n🎭 Визуализация {len(mask_files)} масок...")
    print("=" * 50)
    
    for mask_file in mask_files:
        output_file = output_path / f"viz_{mask_file.name}"
        visualize_mask(mask_file, output_file, show=False)
        print()
    
    print(f"✅ Визуализации сохранены в: {output_path}")

def main():
    if len(sys.argv) > 1:
        # Если передан конкретный файл
        if os.path.isfile(sys.argv[1]):
            visualize_mask(sys.argv[1], show=True)
        else:
            # Передан путь к датасету
            visualize_all_masks(sys.argv[1])
    else:
        # По умолчанию
        if os.path.exists("dataset"):
            visualize_all_masks("dataset")
        else:
            print("Использование: python visualize_masks.py [mask_file.png | dataset_path]")

if __name__ == "__main__":
    main()
