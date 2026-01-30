# Synthetic Dataset for Cube-Parallelepiped Segmentation

## Описание

Система для генерации синтетического датасета в Unity и обучения модели сегментации с определением связей между объектами.

## Структура проекта

```
Assets/
├── Scripts/
│   ├── DatasetGenerator.cs      # Генерация объектов
│   ├── DatasetCapture.cs        # Захват датасета
│   ├── SegmentationId.cs        # Компонент ID
│   └── Editor/
│       └── DatasetCaptureEditor.cs  # Редактор инструментов
├── Shaders/
│   └── SegmentationShader.shader    # Шейдер для масок
└── Materials/

train_segmentation.ipynb   # Jupyter notebook для Kaggle
Dataset/                   # Сгенерированный датасет (после запуска)
├── images/               # RGB изображения
├── masks/                # Сегментационные маски
└── annotations.json      # COCO аннотации с parent_id
```

## Использование

### 1. Настройка сцены в Unity

1. Откройте Unity Editor
2. Перейдите в **Tools → Dataset Capture**
3. Нажмите **Create DatasetGenerator** - создаст объект генератора
4. Нажмите **Create DatasetCapture on Main Camera** - добавит компонент камеры

### 2. Настройка материалов

1. Создайте красный материал (Unlit/Color, красный цвет)
2. Создайте зелёный материал (Unlit/Color, зелёный цвет)
3. Назначьте материалы в компоненте DatasetGenerator
4. Назначьте SegmentationShader в компоненте DatasetCapture

### 3. Генерация объектов

1. Нажмите **Generate Structures** в окне Dataset Capture
2. Проверьте что объекты появились в области 1×1×1 метр
3. При необходимости настройте параметры генерации

### 4. Захват датасета

1. Войдите в **Play Mode** (нажмите Play)
2. Нажмите **Capture Dataset** в окне Dataset Capture
3. Дождитесь завершения (прогресс в Console)
4. Датасет сохранится в папку `Dataset/` рядом с проектом

### 5. Обучение модели

1. Загрузите датасет на Kaggle
2. Откройте `train_segmentation.ipynb`
3. Измените `DATASET_PATH` на путь к вашему датасету
4. Запустите все ячейки

## Формат аннотаций

COCO-формат с дополнительным полем `parent_id`:

```json
{
  "annotations": [
    {
      "id": 1,
      "category_id": 1,
      "instance_id": 1,
      "parent_id": 0,  // Куб - нет родителя
      "bbox": [x, y, w, h]
    },
    {
      "id": 2,
      "category_id": 2,
      "instance_id": 1,
      "parent_id": 1,  // Параллелепипед принадлежит кубу с instance_id=1
      "bbox": [x, y, w, h]
    }
  ]
}
```

## Параметры генерации

| Параметр | Значение | Описание |
|----------|----------|----------|
| Cube size | 3×3×3 см | Размер красного куба |
| Parallelepiped size | 0.1×0.1×2 см | Размер зелёного параллелепипеда |
| Spawn area | 1×1×1 м | Область генерации |
| Count | 5-10 | Количество структур |

## Лицензия

MIT
