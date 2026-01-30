# Strawberry Peduncle Segmentation Dataset

Synthetic dataset for instance segmentation with parent-child object relationships.

## Dataset Structure

```
dataset/
├── images/           # RGB images (PNG)
│   ├── 0000.png
│   └── ...
├── masks/            # Segmentation masks (PNG, instance ID as color)
│   ├── 0000.png
│   └── ...
└── annotations.json  # COCO format with parent_id
```

## Categories

| ID | Name | Description |
|----|------|-------------|
| 1 | red_cube | Red cube 3×3×3 cm |
| 2 | green_parallelepiped | Green stick 0.1×0.1×2 cm attached on top of cube |

## Annotation Format

COCO format with additional `parent_id` field for object relationships:

```json
{
  "images": [{
    "id": 0,
    "file_name": "0000.png",
    "width": 640,
    "height": 480
  }],
  "categories": [
    {"id": 1, "name": "red_cube"},
    {"id": 2, "name": "green_parallelepiped"}
  ],
  "annotations": [{
    "id": 1,
    "image_id": 0,
    "category_id": 1,
    "instance_id": 1,
    "parent_id": 0,
    "bbox": [x, y, w, h],
    "segmentation_color": [r, g, b]
  }, {
    "id": 2,
    "image_id": 0,
    "category_id": 2,
    "instance_id": 1,
    "parent_id": 1,
    "bbox": [x, y, w, h],
    "segmentation_color": [r, g, b]
  }]
}
```

- `parent_id = 0` — object has no parent (cubes)
- `parent_id = N` — object belongs to cube with `instance_id = N`

## Mask Format

Masks encode `instance_id` and `category_id` in RGB channels:
- R channel: `instance_id`
- G channel: `category_id`
- B channel: 0

To decode in Python:
```python
import cv2
mask = cv2.imread('masks/0000.png')
instance_ids = mask[:, :, 2]  # R channel (BGR in OpenCV)
category_ids = mask[:, :, 1]  # G channel
```

## Training

See `train_segmentation.ipynb` for Kaggle training notebook.

## Generation

Dataset generated using Unity with the scripts in this repository.

## License

MIT
