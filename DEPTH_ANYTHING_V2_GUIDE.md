# 📏 Depth Anything V2 + Unity Integration

## ✅ Ответ: ДА, можно использовать параметры камеры из Unity!

### Что такое Depth Anything V2?

**Depth Anything V2** - это state-of-the-art модель для **monocular depth estimation** (оценка глубины по одному изображению).

**Версии:**
- **V1** (2024): Базовая модель, относительная глубина
- **V2** (NeurIPS 2024): Улучшенная модель, **metric depth** (глубина в метрах)
- **V3**: Пока не выпущена

### Ключевые возможности V2:

1. **Metric Depth Estimation** - глубина в метрах (не относительная!)
2. **Два режима**:
   - Indoor (Hypersim): до 20 метров
   - Outdoor (Virtual KITTI): до 80 метров
3. **Использует camera intrinsics** для точных измерений
4. **Быстрая**: 10x быстрее чем Stable Diffusion модели

---

## 🎥 Unity Camera → Depth Anything V2

### Необходимые параметры из Unity:

```csharp
// Unity Camera
Camera camera = Camera.main;
float fov = camera.fieldOfView;  // Вертикальный FOV в градусах
int width = 1024;
int height = 1024;
```

### Вычисление Focal Length:

```python
import math

# Unity параметры
fov_vertical = 60.0  # градусы
image_height = 1024
image_width = 1024

# Формула
fov_rad = fov_vertical * math.pi / 180.0
focal_length_y = (image_height / 2.0) / math.tan(fov_rad / 2.0)
focal_length_x = focal_length_y  # Если aspect ratio = 1:1

# Результат для FOV=60°, 1024x1024:
# focal_length = 886.4 пикселей
```

### Использование в Depth Anything V2:

```python
from depth_anything_v2.dpt import DepthAnythingV2

# Инициализация модели
model = DepthAnythingV2(
    encoder='vitl',
    features=256,
    out_channels=[256, 512, 1024, 1024],
    max_depth=20.0  # 20m для indoor
)
model.load_state_dict(torch.load('depth_anything_v2_metric_hypersim_vitl.pth'))

# Inference
image_bgr = cv2.imread('unity_screenshot.png')
depth_map = model.infer_image(image_bgr)  # HxW array в МЕТРАХ!

# depth_map[y, x] = расстояние в метрах до пикселя (x, y)
```

---

## 📊 Как работает Metric Depth

### 1. Обучение модели:

Depth Anything V2 Metric обучена на:
- **Hypersim** (indoor): 77K синтетических изображений с ground truth depth
- **Virtual KITTI** (outdoor): 21K синтетических изображений

Эти датасеты содержат **точные depth maps** с известными camera intrinsics.

### 2. Inference:

```python
# Модель предсказывает depth в метрах напрямую
depth = model.infer_image(image)

# Можно конвертировать в point cloud
x, y = np.meshgrid(np.arange(width), np.arange(height))

# Используем focal length из Unity
x_3d = (x - width / 2) / focal_length_x * depth
y_3d = (y - height / 2) / focal_length_y * depth
z_3d = depth

points_3d = np.stack([x_3d, y_3d, z_3d], axis=-1)
```

### 3. Точность:

| Датасет | AbsRel ↓ | RMSE ↓ | δ1 ↑ |
|---------|----------|--------|------|
| Hypersim | 0.058 | 0.141 | 0.981 |
| Virtual KITTI | 0.048 | 0.387 | 0.992 |

- **AbsRel**: Средняя относительная ошибка (чем меньше, тем лучше)
- **RMSE**: Root Mean Square Error
- **δ1**: % пикселей с ошибкой < 25% (чем больше, тем лучше)

---

## 🔧 Интеграция с Unity

### Вариант 1: Экспорт параметров камеры

```csharp
// Unity Script для экспорта camera intrinsics
using UnityEngine;
using System.IO;

public class CameraIntrinsicsExporter : MonoBehaviour
{
    public Camera targetCamera;
    
    [ContextMenu("Export Camera Intrinsics")]
    public void ExportIntrinsics()
    {
        float fov = targetCamera.fieldOfView;
        int width = targetCamera.pixelWidth;
        int height = targetCamera.pixelHeight;
        
        // Вычисление focal length
        float fovRad = fov * Mathf.Deg2Rad;
        float focalLengthY = (height / 2.0f) / Mathf.Tan(fovRad / 2.0f);
        float focalLengthX = focalLengthY; // Square aspect
        
        // Сохранение в JSON
        var intrinsics = new
        {
            fov_vertical = fov,
            width = width,
            height = height,
            focal_length_x = focalLengthX,
            focal_length_y = focalLengthY,
            principal_point_x = width / 2.0f,
            principal_point_y = height / 2.0f
        };
        
        string json = JsonUtility.ToJson(intrinsics, true);
        File.WriteAllText("camera_intrinsics.json", json);
        
        Debug.Log($"Camera intrinsics exported: focal_length={focalLengthX:F2}px");
    }
}
```

### Вариант 2: Автоматическое сохранение при захвате

Добавить в `BatchDatasetCapture.cs`:

```csharp
private void SaveCameraIntrinsics(string basePath)
{
    var intrinsics = new
    {
        fov_vertical = mainCamera.fieldOfView,
        width = imageWidth,
        height = imageHeight,
        focal_length_x = CalculateFocalLength(),
        focal_length_y = CalculateFocalLength()
    };
    
    string json = JsonUtility.ToJson(intrinsics, true);
    File.WriteAllText(Path.Combine(basePath, "camera_intrinsics.json"), json);
}

private float CalculateFocalLength()
{
    float fovRad = mainCamera.fieldOfView * Mathf.Deg2Rad;
    return (imageHeight / 2.0f) / Mathf.Tan(fovRad / 2.0f);
}
```

---

## 🚀 Полный Pipeline

### 1. Unity → Генерация датасета

```
Unity BatchDatasetCapture
    ↓
Генерация 1000 изображений
    ↓
Сохранение:
  - images/*.png
  - masks/*.png
  - annotations.json
  - camera_intrinsics.json  ← НОВОЕ!
```

### 2. Python → Обучение Segmentation

```python
# train_segmentation.ipynb
Mask R-CNN обучается на:
  - Segmentation (кубы, параллелепипеды)
  - Определение связей
```

### 3. Python → Depth Estimation + Inference

```python
# depth_estimation_inference.ipynb

# Загрузка camera intrinsics
with open('camera_intrinsics.json') as f:
    intrinsics = json.load(f)

focal_length = intrinsics['focal_length_x']

# Depth estimation
depth_map = depth_model.infer_image(image)

# Segmentation
segmentation = segmentation_model(image)

# Для каждого объекта:
for obj in segmentation:
    # Извлекаем depth для пикселей объекта
    object_depths = depth_map[obj['mask']]
    
    # Вычисляем расстояние
    distance = np.median(object_depths)
    
    print(f"{obj['category']}: {distance:.2f}m")
```

---

## 📈 Преимущества этого подхода

### ✅ Плюсы:

1. **Точные метрические расстояния** (в метрах, не относительные)
2. **Не требует depth sensor** - работает с обычной RGB камерой
3. **Быстрый inference** (~30 FPS на GPU)
4. **Предобученная модель** - не нужно обучать с нуля
5. **Совместимость с Unity** - легко интегрировать

### ⚠️ Ограничения:

1. **Точность зависит от сцены**:
   - Лучше работает на indoor сценах (как наш датасет)
   - Хуже на outdoor с большими расстояниями
2. **Требует правильных camera intrinsics**:
   - Неправильный focal length → неправильные расстояния
3. **Monocular depth** - нет стерео информации:
   - Может ошибаться на текстурах без глубины
   - Проблемы с прозрачными объектами

---

## 🎯 Применение для вашего проекта

### Что вы получаете:

1. **Segmentation**: Mask R-CNN находит кубы и параллелепипеды
2. **Associations**: Определяет какой параллелепипед на каком кубе
3. **Depth**: Depth Anything V2 дает расстояние до каждого объекта
4. **3D Position**: Можно восстановить 3D координаты

### Пример результата:

```json
{
  "image": "00042.png",
  "objects": [
    {
      "id": 0,
      "category": "red_cube",
      "bbox": [512, 600, 580, 668],
      "distance": {
        "center": 0.85,  // метры
        "mean": 0.87,
        "min": 0.82,
        "max": 0.91
      }
    },
    {
      "id": 1,
      "category": "green_parallelepiped",
      "bbox": [530, 550, 562, 600],
      "parent_id": 0,  // Принадлежит кубу #0
      "distance": {
        "center": 0.83,  // Ближе чем куб (сверху)
        "mean": 0.84,
        "min": 0.81,
        "max": 0.86
      }
    }
  ]
}
```

---

## 📚 Ссылки

- **Depth Anything V2 Paper**: https://arxiv.org/abs/2406.09414
- **GitHub**: https://github.com/DepthAnything/Depth-Anything-V2
- **Metric Depth**: https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth
- **HuggingFace Demo**: https://huggingface.co/spaces/Depth-Anything/Depth-Anything-V2

---

## 🚀 Depth Anything V3 (НОВИНКА!)

**Статус**: ✅ **ВЫПУЩЕНА!** (январь 2026)

**GitHub**: https://github.com/ByteDance-Seed/Depth-Anything-3

### 🎯 Ключевые возможности V3:

**Depth Anything V3** - это революционное обновление, которое **превосходит V2** по всем метрикам!

#### 📦 Три серии моделей:

1. **DA3 Main Series** (Giant, Large, Base, Small):
   - 🌊 **Monocular Depth** - depth из одного изображения
   - 🌊 **Multi-View Depth** - consistent depth из нескольких изображений
   - 🎯 **Pose-Conditioned Depth** - depth с учётом camera pose
   - 📷 **Camera Pose Estimation** - оценка extrinsics и intrinsics
   - 🟡 **3D Gaussian Estimation** - прямое предсказание 3D Gaussians!

2. **DA3 Metric Series** (DA3Metric-Large):
   - Специализированная модель для **metric depth в метрах**
   - Формула: `metric_depth = focal * net_output / 300.0`
   - Идеально для вашего Unity проекта!

3. **DA3 Monocular Series** (DA3Mono-Large):
   - Высококачественная relative depth
   - Предсказывает depth напрямую (не disparity как V2)

#### 🌟 DA3 Nested Series:

**DA3NESTED-GIANT-LARGE-1.1** - комбинация any-view модели + metric модели:
- ✅ Автоматически оценивает camera pose
- ✅ Выдаёт depth **сразу в метрах** (не нужна формула!)
- ✅ Работает с одним или несколькими изображениями
- ✅ Лучшая точность на street scenes

### 🆚 V3 vs V2: Что лучше?

| Возможность | V2 | V3 |
|-------------|----|----|
| Monocular Depth | ✅ | ✅ Лучше |
| Metric Depth | ✅ | ✅ Лучше |
| Multi-View Depth | ❌ | ✅ **НОВОЕ!** |
| Camera Pose Estimation | ❌ | ✅ **НОВОЕ!** |
| 3D Gaussians | ❌ | ✅ **НОВОЕ!** |
| Архитектура | DINOv2 + DPT | Plain Transformer |
| Точность | Хорошая | **Лучшая** |

### 💡 Для вашего Unity проекта:

**Рекомендую использовать V3!** Вот почему:

#### Вариант 1: DA3METRIC-LARGE (простой)
```python
from depth_anything_3.api import DepthAnything3

model = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE")
model = model.to("cuda")

prediction = model.inference([image])

# Конвертация в метры с Unity focal length
focal_length = 886.4  # из Unity FOV=60°
metric_depth = focal_length * prediction.depth / 300.0

# metric_depth теперь в метрах!
```

#### Вариант 2: DA3NESTED-GIANT-LARGE-1.1 (продвинутый)
```python
model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE-1.1")
model = model.to("cuda")

prediction = model.inference([image])

# prediction.depth УЖЕ в метрах!
# prediction.intrinsics - оценённые camera intrinsics
# prediction.extrinsics - camera pose (если несколько изображений)

print(f"Depth range: {prediction.depth.min():.2f}m - {prediction.depth.max():.2f}m")
print(f"Estimated focal length: {prediction.intrinsics[0, 0, 0]:.2f}px")
```

### 🎁 Дополнительные возможности V3:

1. **Экспорт в разные форматы**:
   - `.glb` - 3D модели
   - `.ply` - point clouds
   - `.npz` - numpy arrays
   - 3D Gaussian Splatting videos

2. **Web UI** - Gradio интерфейс для визуализации

3. **CLI** - мощный command-line interface:
```bash
da3 auto assets/examples/SOH \
  --export-format glb \
  --export-dir output/
```

### 📊 Точность V3:

**AUC3 метрика** (чем выше, тем лучше):

| Dataset | V2 | V3 (Nested) |
|---------|-------|-------------|
| HiRoom | - | **84.4** |
| ETH3D | - | **52.6** |
| DTU | - | **93.9** |
| ScanNet++ | - | **89.4** |

### ⚠️ Важные детали:

1. **Используйте модели с суффиксом `-1.1`** - они исправляют баг обучения
2. **`use_ray_pose=True`** - медленнее, но точнее для camera pose
3. **Nested модели** выдают depth сразу в метрах (не нужна формула)

### 🔧 Установка V3:

```bash
pip install xformers torch>=2 torchvision
pip install git+https://github.com/ByteDance-Seed/Depth-Anything-3.git
```

---

## 🎯 Итоговая рекомендация для Unity проекта:

### Используйте **Depth Anything V3 (DA3NESTED-GIANT-LARGE-1.1)**!

**Преимущества**:
- ✅ Автоматически оценивает camera intrinsics (не нужно передавать из Unity!)
- ✅ Depth сразу в метрах
- ✅ Лучшая точность
- ✅ Может работать с несколькими изображениями для лучшей consistency
- ✅ Оценивает camera pose (полезно для multi-view)

**Альтернатива**: Если нужна скорость, используйте **DA3METRIC-LARGE** (быстрее, но нужен focal length из Unity)
