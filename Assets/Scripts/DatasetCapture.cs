using UnityEngine;
using UnityEngine.Rendering.Universal;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Globalization;

/// <summary>
/// Захват датасета: RGB изображения, маски сегментации и аннотации COCO
/// </summary>
public class DatasetCapture : MonoBehaviour
{
    [Header("Camera Settings")]
    public Camera mainCamera;
    
    [Tooltip("Радиус орбиты камеры вокруг центра сцены")]
    public float orbitRadius = 0.8f;
    
    [Tooltip("Центр орбиты камеры")]
    public Vector3 orbitCenter = Vector3.zero;

    [Header("Capture Settings")]
    [Tooltip("Количество точек по горизонтали (азимут)")]
    public int horizontalSteps = 8;
    
    [Tooltip("Количество точек по вертикали (элевация)")]
    public int verticalSteps = 4;
    
    [Tooltip("Минимальный угол элевации (градусы)")]
    public float minElevation = 15f;
    
    [Tooltip("Максимальный угол элевации (градусы)")]
    public float maxElevation = 75f;

    [Header("Output Settings")]
    public int imageWidth = 1024;
    public int imageHeight = 1024;
    public string outputFolder = "strawberry_peduncle_segmentation/dataset";

    [Header("Materials")]
    public Shader segmentationShader;

    private DatasetGenerator generator;
    private List<ImageAnnotation> allImages = new List<ImageAnnotation>();
    private List<ObjectAnnotation> allAnnotations = new List<ObjectAnnotation>();
    private int annotationIdCounter = 1;

    // Структуры для COCO формата
    [System.Serializable]
    public class ImageAnnotation
    {
        public int id;
        public string file_name;
        public int width;
        public int height;
    }

    [System.Serializable]
    public class ObjectAnnotation
    {
        public int id;
        public int image_id;
        public int category_id;
        public int instance_id;
        public int parent_id; // 0 for cubes, instance_id for parallelepipeds
        public float[] bbox;
        public float area;
        public int[] segmentation_color; // RGB values for mask lookup
    }

    [System.Serializable]
    public class Category
    {
        public int id;
        public string name;
    }

    [System.Serializable]
    public class CocoDataset
    {
        public List<ImageAnnotation> images;
        public List<ObjectAnnotation> annotations;
        public List<Category> categories;
    }

    void Start()
    {
        generator = FindAnyObjectByType<DatasetGenerator>();
        if (mainCamera == null)
            mainCamera = Camera.main;
    }

    [ContextMenu("Capture Dataset")]
    public void CaptureDataset()
    {
        StartCoroutine(CaptureDatasetCoroutine());
    }

    private IEnumerator CaptureDatasetCoroutine()
    {
        generator = FindAnyObjectByType<DatasetGenerator>();
        if (generator == null)
        {
            Debug.LogError("DatasetGenerator not found!");
            yield break;
        }

        // Автозагрузка шейдера если не назначен
        if (segmentationShader == null)
        {
            segmentationShader = Shader.Find("Custom/SegmentationShader");
            if (segmentationShader == null)
            {
                // Если кастомный не найден, используем Unlit/Color
                segmentationShader = Shader.Find("Unlit/Color");
            }
            if (segmentationShader == null)
            {
                Debug.LogError("Segmentation shader not found!");
                yield break;
            }
        }

        // Создание папок
        string basePath = Path.Combine(Application.dataPath, "..", outputFolder);
        string imagesPath = Path.Combine(basePath, "images");
        string masksPath = Path.Combine(basePath, "masks");
        string vizPath = Path.Combine(basePath, "mask_visualizations");

        Directory.CreateDirectory(imagesPath);
        Directory.CreateDirectory(masksPath);
        Directory.CreateDirectory(vizPath);

        allImages.Clear();
        allAnnotations.Clear();
        annotationIdCounter = 1;

        int imageId = 0;

        // Генерация позиций камеры
        for (int v = 0; v < verticalSteps; v++)
        {
            float elevation = Mathf.Lerp(minElevation, maxElevation, (float)v / (verticalSteps - 1));
            
            for (int h = 0; h < horizontalSteps; h++)
            {
                float azimuth = (360f / horizontalSteps) * h;
                
                // Вычисление позиции камеры
                Vector3 cameraPos = CalculateCameraPosition(azimuth, elevation);
                mainCamera.transform.position = cameraPos;
                mainCamera.transform.LookAt(orbitCenter);

                yield return new WaitForEndOfFrame();

                // Захват RGB
                string rgbFileName = $"{imageId:D4}.png";
                CaptureRGB(Path.Combine(imagesPath, rgbFileName));

                // Захват маски сегментации и визуализации
                string maskFileName = $"{imageId:D4}.png";
                CaptureSegmentationMaskWithVisualization(
                    Path.Combine(masksPath, maskFileName),
                    Path.Combine(vizPath, $"viz_{maskFileName}")
                );

                // Добавление аннотаций
                AddAnnotations(imageId, rgbFileName);

                Debug.Log($"Captured image {imageId} at azimuth={azimuth}, elevation={elevation}");
                imageId++;

                yield return null;
            }
        }

        // Сохранение аннотаций
        SaveAnnotations(basePath);
        
        Debug.Log($"Dataset capture complete! {imageId} images saved to {basePath}");
        Debug.Log($"Visualizations saved to {vizPath}");
    }

    private Vector3 CalculateCameraPosition(float azimuth, float elevation)
    {
        float azimuthRad = azimuth * Mathf.Deg2Rad;
        float elevationRad = elevation * Mathf.Deg2Rad;

        float x = orbitRadius * Mathf.Cos(elevationRad) * Mathf.Sin(azimuthRad);
        float y = orbitRadius * Mathf.Sin(elevationRad);
        float z = orbitRadius * Mathf.Cos(elevationRad) * Mathf.Cos(azimuthRad);

        return orbitCenter + new Vector3(x, y, z);
    }

    private void CaptureRGB(string filePath)
    {
        RenderTexture rt = new RenderTexture(imageWidth, imageHeight, 24);
        mainCamera.targetTexture = rt;
        
        Texture2D screenShot = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);
        mainCamera.Render();
        
        RenderTexture.active = rt;
        screenShot.ReadPixels(new Rect(0, 0, imageWidth, imageHeight), 0, 0);
        screenShot.Apply();
        
        byte[] bytes = screenShot.EncodeToPNG();
        File.WriteAllBytes(filePath, bytes);
        
        mainCamera.targetTexture = null;
        RenderTexture.active = null;
        DestroyImmediate(rt);
        DestroyImmediate(screenShot);
    }

    private void CaptureSegmentationMask(string filePath)
    {
        // Сохраняем оригинальные материалы
        Dictionary<Renderer, Material[]> originalMaterials = new Dictionary<Renderer, Material[]>();
        List<Material> tempMaterials = new List<Material>();
        
        // Находим URP Unlit шейдер
        Shader unlitShader = Shader.Find("Universal Render Pipeline/Unlit");
        if (unlitShader == null)
        {
            unlitShader = Shader.Find("Unlit/Color");
        }
        
        if (unlitShader == null)
        {
            Debug.LogError("Cannot find Unlit shader for segmentation masks!");
            return;
        }

        // Заменяем материалы на сегментационные
        foreach (var structure in generator.generatedStructures)
        {
            // Куб
            var cubeRenderer = structure.cube.GetComponent<Renderer>();
            originalMaterials[cubeRenderer] = cubeRenderer.sharedMaterials;
            
            Material cubeMaskMat = new Material(unlitShader);
            var cubeSegId = structure.cube.GetComponent<SegmentationId>();
            Color cubeColor = GetInstanceColor(cubeSegId.instanceId, cubeSegId.categoryId);
            cubeMaskMat.SetColor("_BaseColor", cubeColor);
            cubeMaskMat.SetColor("_Color", cubeColor);
            cubeRenderer.sharedMaterial = cubeMaskMat;
            tempMaterials.Add(cubeMaskMat);

            // Параллелепипед
            var paraRenderer = structure.parallelepiped.GetComponent<Renderer>();
            originalMaterials[paraRenderer] = paraRenderer.sharedMaterials;
            
            Material paraMaskMat = new Material(unlitShader);
            var paraSegId = structure.parallelepiped.GetComponent<SegmentationId>();
            Color paraColor = GetInstanceColor(paraSegId.instanceId, paraSegId.categoryId);
            paraMaskMat.SetColor("_BaseColor", paraColor);
            paraMaskMat.SetColor("_Color", paraColor);
            paraRenderer.sharedMaterial = paraMaskMat;
            tempMaterials.Add(paraMaskMat);
        }

        // Скрываем все другие объекты и устанавливаем чёрный фон
        Color originalBgColor = mainCamera.backgroundColor;
        CameraClearFlags originalClearFlags = mainCamera.clearFlags;
        int originalCullingMask = mainCamera.cullingMask;
        
        mainCamera.backgroundColor = Color.black;
        mainCamera.clearFlags = CameraClearFlags.SolidColor;
        
        // Отключаем пост-обработку если есть
        var urpData = mainCamera.GetUniversalAdditionalCameraData();
        bool originalPostProcessing = false;
        if (urpData != null)
        {
            originalPostProcessing = urpData.renderPostProcessing;
            urpData.renderPostProcessing = false;
        }

        // Рендерим маску
        RenderTexture rt = new RenderTexture(imageWidth, imageHeight, 24, RenderTextureFormat.ARGB32);
        rt.antiAliasing = 1;
        mainCamera.targetTexture = rt;
        
        Texture2D screenShot = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);
        mainCamera.Render();
        
        RenderTexture.active = rt;
        screenShot.ReadPixels(new Rect(0, 0, imageWidth, imageHeight), 0, 0);
        screenShot.Apply();
        
        byte[] bytes = screenShot.EncodeToPNG();
        File.WriteAllBytes(filePath, bytes);
        
        mainCamera.targetTexture = null;
        RenderTexture.active = null;
        DestroyImmediate(rt);
        DestroyImmediate(screenShot);

        // Восстанавливаем оригинальные материалы
        foreach (var kvp in originalMaterials)
        {
            kvp.Key.sharedMaterials = kvp.Value;
        }
        
        // Очищаем временные материалы
        foreach (var mat in tempMaterials)
        {
            DestroyImmediate(mat);
        }

        // Восстанавливаем настройки камеры
        mainCamera.backgroundColor = originalBgColor;
        mainCamera.clearFlags = originalClearFlags;
        mainCamera.cullingMask = originalCullingMask;
        
        if (urpData != null)
        {
            urpData.renderPostProcessing = originalPostProcessing;
        }
    }

    private void CaptureSegmentationMaskWithVisualization(string maskFilePath, string vizFilePath)
    {
        // Сохраняем оригинальные материалы
        Dictionary<Renderer, Material[]> originalMaterials = new Dictionary<Renderer, Material[]>();
        List<Material> tempMaterials = new List<Material>();
        
        // Находим URP Unlit шейдер
        Shader unlitShader = Shader.Find("Universal Render Pipeline/Unlit");
        if (unlitShader == null)
            unlitShader = Shader.Find("Unlit/Color");

        // Заменяем материалы на сегментационные
        foreach (var structure in generator.generatedStructures)
        {
            var cubeRenderer = structure.cube.GetComponent<Renderer>();
            originalMaterials[cubeRenderer] = cubeRenderer.sharedMaterials;
            
            Material cubeMaskMat = new Material(unlitShader);
            var cubeSegId = structure.cube.GetComponent<SegmentationId>();
            Color cubeColor = GetInstanceColor(cubeSegId.instanceId, cubeSegId.categoryId);
            cubeMaskMat.SetColor("_BaseColor", cubeColor);
            cubeMaskMat.SetColor("_Color", cubeColor);
            cubeRenderer.sharedMaterial = cubeMaskMat;
            tempMaterials.Add(cubeMaskMat);

            var paraRenderer = structure.parallelepiped.GetComponent<Renderer>();
            originalMaterials[paraRenderer] = paraRenderer.sharedMaterials;
            
            Material paraMaskMat = new Material(unlitShader);
            var paraSegId = structure.parallelepiped.GetComponent<SegmentationId>();
            Color paraColor = GetInstanceColor(paraSegId.instanceId, paraSegId.categoryId);
            paraMaskMat.SetColor("_BaseColor", paraColor);
            paraMaskMat.SetColor("_Color", paraColor);
            paraRenderer.sharedMaterial = paraMaskMat;
            tempMaterials.Add(paraMaskMat);
        }

        // Чёрный фон
        Color originalBgColor = mainCamera.backgroundColor;
        CameraClearFlags originalClearFlags = mainCamera.clearFlags;
        mainCamera.backgroundColor = Color.black;
        mainCamera.clearFlags = CameraClearFlags.SolidColor;
        
        var urpData = mainCamera.GetUniversalAdditionalCameraData();
        bool originalPostProcessing = false;
        if (urpData != null)
        {
            originalPostProcessing = urpData.renderPostProcessing;
            urpData.renderPostProcessing = false;
        }

        // Рендерим маску
        RenderTexture rt = new RenderTexture(imageWidth, imageHeight, 24, RenderTextureFormat.ARGB32);
        rt.antiAliasing = 1;
        mainCamera.targetTexture = rt;
        mainCamera.Render();
        
        RenderTexture.active = rt;
        Texture2D mask = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);
        mask.ReadPixels(new Rect(0, 0, imageWidth, imageHeight), 0, 0);
        mask.Apply();
        
        // Сохраняем оригинальную маску
        byte[] maskBytes = mask.EncodeToPNG();
        File.WriteAllBytes(maskFilePath, maskBytes);
        
        // Создаём визуализацию с усиленными цветами
        Color32[] pixels = mask.GetPixels32();
        for (int i = 0; i < pixels.Length; i++)
        {
            if (pixels[i].r > 0 || pixels[i].g > 0)
            {
                // Яркие цвета для визуализации
                int instanceId = pixels[i].r;
                int categoryId = pixels[i].g;
                
                if (categoryId == 1) // Куб - оттенки красного
                {
                    pixels[i] = new Color32(
                        (byte)(150 + (instanceId * 15) % 105),
                        (byte)(30 + (instanceId * 7) % 50),
                        (byte)(30 + (instanceId * 11) % 50),
                        255
                    );
                }
                else if (categoryId == 2) // Параллелепипед - оттенки зелёного
                {
                    pixels[i] = new Color32(
                        (byte)(30 + (instanceId * 7) % 50),
                        (byte)(150 + (instanceId * 15) % 105),
                        (byte)(30 + (instanceId * 11) % 50),
                        255
                    );
                }
            }
        }
        mask.SetPixels32(pixels);
        mask.Apply();
        
        byte[] vizBytes = mask.EncodeToPNG();
        File.WriteAllBytes(vizFilePath, vizBytes);
        
        mainCamera.targetTexture = null;
        RenderTexture.active = null;
        DestroyImmediate(rt);
        DestroyImmediate(mask);

        // Восстанавливаем материалы
        foreach (var kvp in originalMaterials)
        {
            kvp.Key.sharedMaterials = kvp.Value;
        }
        
        foreach (var mat in tempMaterials)
        {
            DestroyImmediate(mat);
        }

        mainCamera.backgroundColor = originalBgColor;
        mainCamera.clearFlags = originalClearFlags;
        
        if (urpData != null)
        {
            urpData.renderPostProcessing = originalPostProcessing;
        }
    }

    /// <summary>
    /// Возвращает уникальный цвет для instance_id + category_id
    /// </summary>
    private Color GetInstanceColor(int instanceId, int categoryId)
    {
        // Используем instanceId в красном канале, categoryId в зелёном
        // Это позволяет легко декодировать в Python
        float r = instanceId / 255f;
        float g = categoryId / 255f;
        float b = 0f;
        return new Color(r, g, b, 1f);
    }

    private void AddAnnotations(int imageId, string fileName)
    {
        // Добавляем информацию об изображении
        allImages.Add(new ImageAnnotation
        {
            id = imageId,
            file_name = fileName,
            width = imageWidth,
            height = imageHeight
        });

        // Для каждой структуры проверяем видимость и добавляем аннотации
        foreach (var structure in generator.generatedStructures)
        {
            var cubeSegId = structure.cube.GetComponent<SegmentationId>();
            var paraSegId = structure.parallelepiped.GetComponent<SegmentationId>();

            // Проверяем видимость куба
            Bounds cubeBounds = structure.cube.GetComponent<Renderer>().bounds;
            Rect cubeBBox = GetScreenBoundingBox(cubeBounds);
            
            if (cubeBBox.width > 0 && cubeBBox.height > 0 && IsVisible(structure.cube))
            {
                Color cubeColor = GetInstanceColor(cubeSegId.instanceId, cubeSegId.categoryId);
                allAnnotations.Add(new ObjectAnnotation
                {
                    id = annotationIdCounter++,
                    image_id = imageId,
                    category_id = cubeSegId.categoryId,
                    instance_id = cubeSegId.instanceId,
                    parent_id = 0,
                    bbox = new float[] { cubeBBox.x, cubeBBox.y, cubeBBox.width, cubeBBox.height },
                    area = cubeBBox.width * cubeBBox.height,
                    segmentation_color = new int[] { 
                        Mathf.RoundToInt(cubeColor.r * 255), 
                        Mathf.RoundToInt(cubeColor.g * 255), 
                        Mathf.RoundToInt(cubeColor.b * 255) 
                    }
                });
            }

            // Проверяем видимость параллелепипеда
            Bounds paraBounds = structure.parallelepiped.GetComponent<Renderer>().bounds;
            Rect paraBBox = GetScreenBoundingBox(paraBounds);
            
            if (paraBBox.width > 0 && paraBBox.height > 0 && IsVisible(structure.parallelepiped))
            {
                Color paraColor = GetInstanceColor(paraSegId.instanceId, paraSegId.categoryId);
                allAnnotations.Add(new ObjectAnnotation
                {
                    id = annotationIdCounter++,
                    image_id = imageId,
                    category_id = paraSegId.categoryId,
                    instance_id = paraSegId.instanceId,
                    parent_id = cubeSegId.instanceId, // Связь с родительским кубом
                    bbox = new float[] { paraBBox.x, paraBBox.y, paraBBox.width, paraBBox.height },
                    area = paraBBox.width * paraBBox.height,
                    segmentation_color = new int[] { 
                        Mathf.RoundToInt(paraColor.r * 255), 
                        Mathf.RoundToInt(paraColor.g * 255), 
                        Mathf.RoundToInt(paraColor.b * 255) 
                    }
                });
            }
        }
    }

    private Rect GetScreenBoundingBox(Bounds bounds)
    {
        Vector3[] corners = new Vector3[8];
        corners[0] = bounds.min;
        corners[1] = bounds.max;
        corners[2] = new Vector3(bounds.min.x, bounds.min.y, bounds.max.z);
        corners[3] = new Vector3(bounds.min.x, bounds.max.y, bounds.min.z);
        corners[4] = new Vector3(bounds.max.x, bounds.min.y, bounds.min.z);
        corners[5] = new Vector3(bounds.min.x, bounds.max.y, bounds.max.z);
        corners[6] = new Vector3(bounds.max.x, bounds.min.y, bounds.max.z);
        corners[7] = new Vector3(bounds.max.x, bounds.max.y, bounds.min.z);

        float minX = float.MaxValue, minY = float.MaxValue;
        float maxX = float.MinValue, maxY = float.MinValue;
        bool anyValid = false;

        foreach (var corner in corners)
        {
            Vector3 screenPoint = mainCamera.WorldToScreenPoint(corner);
            
            // Только если точка перед камерой
            if (screenPoint.z > 0)
            {
                anyValid = true;
                minX = Mathf.Min(minX, screenPoint.x);
                maxX = Mathf.Max(maxX, screenPoint.x);
                minY = Mathf.Min(minY, screenPoint.y);
                maxY = Mathf.Max(maxY, screenPoint.y);
            }
        }

        if (!anyValid)
            return new Rect(0, 0, 0, 0);

        // Ограничиваем рамками экрана
        minX = Mathf.Max(0, minX);
        minY = Mathf.Max(0, minY);
        maxX = Mathf.Min(imageWidth, maxX);
        maxY = Mathf.Min(imageHeight, maxY);

        // Конвертируем в формат COCO (y инвертирован)
        float y = imageHeight - maxY;
        
        return new Rect(minX, y, maxX - minX, maxY - minY);
    }

    private bool IsVisible(GameObject obj)
    {
        Vector3 screenPoint = mainCamera.WorldToViewportPoint(obj.transform.position);
        return screenPoint.x >= 0 && screenPoint.x <= 1 &&
               screenPoint.y >= 0 && screenPoint.y <= 1 &&
               screenPoint.z > 0;
    }

    private void SaveAnnotations(string basePath)
    {
        CocoDataset dataset = new CocoDataset
        {
            images = allImages,
            annotations = allAnnotations,
            categories = new List<Category>
            {
                new Category { id = 1, name = "red_cube" },
                new Category { id = 2, name = "green_parallelepiped" }
            }
        };

        string json = JsonUtility.ToJson(dataset, true);
        
        // JsonUtility не поддерживает сериализацию списков напрямую,
        // поэтому создаём JSON вручную
        StringBuilder sb = new StringBuilder();
        sb.AppendLine("{");
        
        // Images
        sb.AppendLine("  \"images\": [");
        for (int i = 0; i < allImages.Count; i++)
        {
            var img = allImages[i];
            sb.Append($"    {{\"id\": {img.id}, \"file_name\": \"{img.file_name}\", \"width\": {img.width}, \"height\": {img.height}}}");
            if (i < allImages.Count - 1) sb.AppendLine(",");
            else sb.AppendLine();
        }
        sb.AppendLine("  ],");

        // Categories
        sb.AppendLine("  \"categories\": [");
        sb.AppendLine("    {\"id\": 1, \"name\": \"red_cube\"},");
        sb.AppendLine("    {\"id\": 2, \"name\": \"green_parallelepiped\"}");
        sb.AppendLine("  ],");

        // Annotations
        sb.AppendLine("  \"annotations\": [");
        for (int i = 0; i < allAnnotations.Count; i++)
        {
            var ann = allAnnotations[i];
            // Используем InvariantCulture для правильного формата чисел (точки вместо запятых)
            string bboxStr = string.Format(CultureInfo.InvariantCulture, 
                "[{0:F2}, {1:F2}, {2:F2}, {3:F2}]", 
                ann.bbox[0], ann.bbox[1], ann.bbox[2], ann.bbox[3]);
            string areaStr = ann.area.ToString("F2", CultureInfo.InvariantCulture);
            
            sb.Append($"    {{\"id\": {ann.id}, \"image_id\": {ann.image_id}, \"category_id\": {ann.category_id}, ");
            sb.Append($"\"instance_id\": {ann.instance_id}, \"parent_id\": {ann.parent_id}, ");
            sb.Append($"\"bbox\": {bboxStr}, ");
            sb.Append($"\"area\": {areaStr}, ");
            sb.Append($"\"segmentation_color\": [{ann.segmentation_color[0]}, {ann.segmentation_color[1]}, {ann.segmentation_color[2]}]}}");
            if (i < allAnnotations.Count - 1) sb.AppendLine(",");
            else sb.AppendLine();
        }
        sb.AppendLine("  ]");
        
        sb.AppendLine("}");

        File.WriteAllText(Path.Combine(basePath, "annotations.json"), sb.ToString());
    }

    void OnDrawGizmosSelected()
    {
        // Визуализация орбиты камеры
        Gizmos.color = Color.cyan;
        
        int segments = 36;
        Vector3 prevPos = Vector3.zero;
        
        for (int i = 0; i <= segments; i++)
        {
            float angle = (360f / segments) * i;
            Vector3 pos = CalculateCameraPosition(angle, (minElevation + maxElevation) / 2f);
            
            if (i > 0)
                Gizmos.DrawLine(prevPos, pos);
            
            prevPos = pos;
        }

        // Центр орбиты
        Gizmos.color = Color.yellow;
        Gizmos.DrawWireSphere(orbitCenter, 0.05f);
    }
}
