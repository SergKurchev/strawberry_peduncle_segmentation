using UnityEngine;
using UnityEngine.Rendering.Universal;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Globalization;

/// <summary>
/// Улучшенный Batch генератор датасета:
/// - Рандомная камера (дистанция, углы, кропы)
/// - Фильтрация окклюзий (видны только незаслонённые части)
/// - Визуализации во время генерации
/// </summary>
public class BatchDatasetCapture : MonoBehaviour
{
    [Header("Capture Settings")]
    public int totalImages = 1000;
    public int imagesPerScene = 10;
    
    [Header("Camera Settings")]
    public Camera mainCamera;
    
    [Tooltip("Минимальный радиус камеры")]
    public float minOrbitRadius = 0.6f;
    
    [Tooltip("Максимальный радиус камеры")]
    public float maxOrbitRadius = 1.2f;
    
    public Vector3 orbitCenter = Vector3.zero;
    public float minElevation = 10f;
    public float maxElevation = 80f;
    
    [Header("Random Crop Settings")]
    [Tooltip("Вероятность кропа (0-1)")]
    [Range(0f, 1f)]
    public float cropProbability = 0.3f;
    
    [Tooltip("Минимальный размер кропа относительно полного изображения")]
    [Range(0.3f, 1f)]
    public float minCropScale = 0.5f;
    
    [Header("Output Settings")]
    public int imageWidth = 1024;
    public int imageHeight = 1024;
    public string outputFolder = "strawberry_peduncle_segmentation/dataset";
    
    [Header("Visualization")]
    public bool saveVisualizations = true;
    
    [Header("Occlusion Settings")]
    [Tooltip("Минимальный % видимости объекта для включения в аннотации")]
    [Range(0.1f, 1f)]
    public float minVisibilityRatio = 0.3f;

    private DatasetGenerator generator;
    private List<ImageAnnotation> allImages = new List<ImageAnnotation>();
    private List<ObjectAnnotation> allAnnotations = new List<ObjectAnnotation>();
    private int annotationIdCounter = 1;
    private bool isCapturing = false;
    
    // Текущие параметры кропа
    private Rect currentCropRect;
    private bool isCurrentlyCropped = false;

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
        public int parent_id;
        public float[] bbox;
        public float area;
        public int[] segmentation_color;
        public float visibility_ratio;
    }

    void Start()
    {
        generator = FindAnyObjectByType<DatasetGenerator>();
        if (mainCamera == null)
            mainCamera = Camera.main;
    }

    [ContextMenu("Start Batch Capture (1000 images)")]
    public void StartBatchCapture()
    {
        if (!isCapturing)
            StartCoroutine(BatchCaptureCoroutine());
    }

    private IEnumerator BatchCaptureCoroutine()
    {
        isCapturing = true;
        generator = FindAnyObjectByType<DatasetGenerator>();
        
        if (generator == null)
        {
            Debug.LogError("DatasetGenerator not found!");
            isCapturing = false;
            yield break;
        }

        // Создание папок
        string basePath = Path.Combine(Application.dataPath, "..", outputFolder);
        string imagesPath = Path.Combine(basePath, "images");
        string masksPath = Path.Combine(basePath, "masks");
        string vizPath = Path.Combine(basePath, "visualizations");
        
        Directory.CreateDirectory(imagesPath);
        Directory.CreateDirectory(masksPath);
        if (saveVisualizations)
            Directory.CreateDirectory(vizPath);

        allImages.Clear();
        allAnnotations.Clear();
        annotationIdCounter = 1;

        int imageCounter = 0;
        int sceneCounter = 0;
        
        Debug.Log($"Starting batch capture: {totalImages} images, regenerating every {imagesPerScene} shots");

        while (imageCounter < totalImages)
        {
            // Перегенерация объектов каждые imagesPerScene снимков
            generator.GenerateStructures();
            sceneCounter++;
            
            yield return null;

            for (int shot = 0; shot < imagesPerScene && imageCounter < totalImages; shot++)
            {
                // Рандомный ракурс камеры
                SetupRandomCamera();
                
                yield return null;

                string fileName = $"{imageCounter:D5}.png";
                
                // Рендерим сначала маску для расчёта окклюзий
                RenderTexture maskRT = RenderSegmentationMask();
                
                // Получаем реальные видимые пиксели для каждого объекта
                Dictionary<int, int> visiblePixels = CalculateVisiblePixels(maskRT);
                
                // Сохраняем RGB изображение
                CaptureRGBImage(Path.Combine(imagesPath, fileName));
                
                // Сохраняем маску
                SaveRenderTexture(maskRT, Path.Combine(masksPath, fileName));
                
                // Добавляем только видимые аннотации
                int actualWidth = isCurrentlyCropped ? (int)currentCropRect.width : imageWidth;
                int actualHeight = isCurrentlyCropped ? (int)currentCropRect.height : imageHeight;
                
                allImages.Add(new ImageAnnotation
                {
                    id = imageCounter,
                    file_name = fileName,
                    width = actualWidth,
                    height = actualHeight
                });
                
                AddAnnotationsWithOcclusion(imageCounter, visiblePixels);
                
                // Визуализация
                if (saveVisualizations && imageCounter < 50)
                {
                    SaveVisualization(Path.Combine(vizPath, $"viz_{fileName}"), maskRT);
                }
                
                RenderTexture.ReleaseTemporary(maskRT);
                
                imageCounter++;
                
                if (imageCounter % 100 == 0)
                    Debug.Log($"Progress: {imageCounter}/{totalImages} images ({sceneCounter} scenes)");
            }
        }

        SaveAnnotations(basePath);
        
        Debug.Log($"Batch capture complete! {imageCounter} images saved to {basePath}");
        isCapturing = false;
    }

    private void SetupRandomCamera()
    {
        // Устанавливаем near plane камеры для избежания клиппинга
        float originalNearPlane = mainCamera.nearClipPlane;
        mainCamera.nearClipPlane = 0.05f;
        
        // Рандомный радиус (дистанция) с проверкой
        // Минимум должен быть больше половины диагонали области генерации
        float sceneDiagonal = Mathf.Sqrt(
            generator.spawnAreaSize.x * generator.spawnAreaSize.x +
            generator.spawnAreaSize.y * generator.spawnAreaSize.y +
            generator.spawnAreaSize.z * generator.spawnAreaSize.z
        );
        float safeMinRadius = Mathf.Max(minOrbitRadius, sceneDiagonal * 0.6f);
        float radius = Random.Range(safeMinRadius, maxOrbitRadius);
        
        // Рандомные углы
        float azimuth = Random.Range(0f, 360f);
        float elevation = Random.Range(minElevation, maxElevation);
        
        // Добавляем случайный наклон камеры
        float tilt = Random.Range(-15f, 15f);
        
        // Позиционируем камеру
        float azimuthRad = azimuth * Mathf.Deg2Rad;
        float elevationRad = elevation * Mathf.Deg2Rad;

        float x = radius * Mathf.Cos(elevationRad) * Mathf.Sin(azimuthRad);
        float y = radius * Mathf.Sin(elevationRad);
        float z = radius * Mathf.Cos(elevationRad) * Mathf.Cos(azimuthRad);

        Vector3 cameraPosition = orbitCenter + new Vector3(x, y, z);
        
        // Проверяем что камера не внутри объектов
        // Если находим пересечение, увеличиваем радиус
        foreach (var structure in generator.generatedStructures)
        {
            Vector3 cubePos = structure.cube.transform.position;
            float distToCube = Vector3.Distance(cameraPosition, cubePos);
            
            // Размер куба ~0.03м, добавляем буфер
            if (distToCube < 0.1f)
            {
                // Отодвигаем камеру дальше
                Vector3 direction = (cameraPosition - orbitCenter).normalized;
                cameraPosition = orbitCenter + direction * (radius + 0.2f);
                break;
            }
        }
        
        mainCamera.transform.position = cameraPosition;
        mainCamera.transform.LookAt(orbitCenter);
        
        // Добавляем случайный roll
        mainCamera.transform.Rotate(Vector3.forward, tilt, Space.Self);
        
        // Случайное смещение точки взгляда
        Vector3 lookOffset = new Vector3(
            Random.Range(-0.05f, 0.05f),
            Random.Range(-0.05f, 0.05f),
            Random.Range(-0.05f, 0.05f)
        );
        mainCamera.transform.LookAt(orbitCenter + lookOffset);
        
        // Решаем, делать ли кроп
        isCurrentlyCropped = Random.value < cropProbability;
        
        if (isCurrentlyCropped)
        {
            // Рандомный кроп
            float cropScale = Random.Range(minCropScale, 1f);
            float cropWidth = imageWidth * cropScale;
            float cropHeight = imageHeight * cropScale;
            float cropX = Random.Range(0, imageWidth - cropWidth);
            float cropY = Random.Range(0, imageHeight - cropHeight);
            
            currentCropRect = new Rect(cropX, cropY, cropWidth, cropHeight);
        }
        else
        {
            currentCropRect = new Rect(0, 0, imageWidth, imageHeight);
        }
    }

    private RenderTexture RenderSegmentationMask()
    {
        Dictionary<Renderer, Material[]> originalMaterials = new Dictionary<Renderer, Material[]>();
        List<Material> tempMaterials = new List<Material>();
        
        Shader unlitShader = Shader.Find("Universal Render Pipeline/Unlit");
        if (unlitShader == null)
            unlitShader = Shader.Find("Unlit/Color");

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

        // Рендер
        RenderTexture rt = RenderTexture.GetTemporary(imageWidth, imageHeight, 24, RenderTextureFormat.ARGB32);
        rt.antiAliasing = 1;
        mainCamera.targetTexture = rt;
        mainCamera.Render();
        mainCamera.targetTexture = null;

        // Восстановление
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

        return rt;
    }

    private Dictionary<int, int> CalculateVisiblePixels(RenderTexture rt)
    {
        Dictionary<int, int> visiblePixels = new Dictionary<int, int>();
        
        RenderTexture.active = rt;
        Texture2D tex = new Texture2D(rt.width, rt.height, TextureFormat.RGB24, false);
        tex.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0);
        tex.Apply();
        RenderTexture.active = null;
        
        Color32[] pixels = tex.GetPixels32();
        
        foreach (var pixel in pixels)
        {
            if (pixel.r > 0 || pixel.g > 0)
            {
                // Создаём уникальный ключ из instance_id и category_id
                int key = pixel.r * 1000 + pixel.g;
                if (!visiblePixels.ContainsKey(key))
                    visiblePixels[key] = 0;
                visiblePixels[key]++;
            }
        }
        
        DestroyImmediate(tex);
        return visiblePixels;
    }

    private void CaptureRGBImage(string filePath)
    {
        RenderTexture rt = new RenderTexture(imageWidth, imageHeight, 24);
        mainCamera.targetTexture = rt;
        mainCamera.Render();
        
        RenderTexture.active = rt;
        
        Texture2D screenShot;
        if (isCurrentlyCropped)
        {
            // Читаем только кроп
            screenShot = new Texture2D((int)currentCropRect.width, (int)currentCropRect.height, TextureFormat.RGB24, false);
            screenShot.ReadPixels(currentCropRect, 0, 0);
        }
        else
        {
            screenShot = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);
            screenShot.ReadPixels(new Rect(0, 0, imageWidth, imageHeight), 0, 0);
        }
        screenShot.Apply();
        
        byte[] bytes = screenShot.EncodeToPNG();
        File.WriteAllBytes(filePath, bytes);
        
        mainCamera.targetTexture = null;
        RenderTexture.active = null;
        DestroyImmediate(rt);
        DestroyImmediate(screenShot);
    }

    private void SaveRenderTexture(RenderTexture rt, string filePath)
    {
        RenderTexture.active = rt;
        
        Texture2D screenShot;
        if (isCurrentlyCropped)
        {
            screenShot = new Texture2D((int)currentCropRect.width, (int)currentCropRect.height, TextureFormat.RGB24, false);
            screenShot.ReadPixels(currentCropRect, 0, 0);
        }
        else
        {
            screenShot = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);
            screenShot.ReadPixels(new Rect(0, 0, imageWidth, imageHeight), 0, 0);
        }
        screenShot.Apply();
        
        byte[] bytes = screenShot.EncodeToPNG();
        File.WriteAllBytes(filePath, bytes);
        
        RenderTexture.active = null;
        DestroyImmediate(screenShot);
    }

    private void SaveVisualization(string filePath, RenderTexture maskRT)
    {
        // Создаём усиленную визуализацию маски
        RenderTexture.active = maskRT;
        Texture2D mask;
        if (isCurrentlyCropped)
        {
            mask = new Texture2D((int)currentCropRect.width, (int)currentCropRect.height, TextureFormat.RGB24, false);
            mask.ReadPixels(currentCropRect, 0, 0);
        }
        else
        {
            mask = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);
            mask.ReadPixels(new Rect(0, 0, imageWidth, imageHeight), 0, 0);
        }
        mask.Apply();
        RenderTexture.active = null;
        
        // Усиливаем цвета для визуализации
        Color32[] pixels = mask.GetPixels32();
        for (int i = 0; i < pixels.Length; i++)
        {
            if (pixels[i].r > 0 || pixels[i].g > 0)
            {
                // Создаём яркий цвет на основе ID
                int hue = (pixels[i].r * 37 + pixels[i].g * 73) % 256;
                pixels[i] = new Color32(
                    (byte)((pixels[i].r * 30) % 256 + 50),
                    (byte)((pixels[i].g * 100) % 256 + 50),
                    (byte)(hue),
                    255
                );
            }
        }
        mask.SetPixels32(pixels);
        mask.Apply();
        
        byte[] bytes = mask.EncodeToPNG();
        File.WriteAllBytes(filePath, bytes);
        
        DestroyImmediate(mask);
    }

    private Color GetInstanceColor(int instanceId, int categoryId)
    {
        float r = instanceId / 255f;
        float g = categoryId / 255f;
        float b = 0f;
        return new Color(r, g, b, 1f);
    }

    private void AddAnnotationsWithOcclusion(int imageId, Dictionary<int, int> visiblePixels)
    {
        foreach (var structure in generator.generatedStructures)
        {
            var cubeSegId = structure.cube.GetComponent<SegmentationId>();
            var paraSegId = structure.parallelepiped.GetComponent<SegmentationId>();

            // Проверяем куб
            int cubeKey = cubeSegId.instanceId * 1000 + cubeSegId.categoryId;
            int cubeVisible = visiblePixels.ContainsKey(cubeKey) ? visiblePixels[cubeKey] : 0;
            
            // Рассчитываем теоретическую площадь
            Bounds cubeBounds = structure.cube.GetComponent<Renderer>().bounds;
            Rect cubeBBox = GetScreenBoundingBox(cubeBounds);
            float cubeTheoreticalArea = cubeBBox.width * cubeBBox.height;
            
            // Проверяем видимость
            float cubeVisibility = cubeTheoreticalArea > 0 ? cubeVisible / cubeTheoreticalArea : 0;
            
            if (cubeVisible > 10 && cubeVisibility >= minVisibilityRatio && IsInCropArea(cubeBBox))
            {
                // Корректируем bbox для кропа
                Rect adjustedBBox = AdjustBBoxForCrop(cubeBBox);
                
                if (adjustedBBox.width > 0 && adjustedBBox.height > 0)
                {
                    Color cubeColor = GetInstanceColor(cubeSegId.instanceId, cubeSegId.categoryId);
                    allAnnotations.Add(new ObjectAnnotation
                    {
                        id = annotationIdCounter++,
                        image_id = imageId,
                        category_id = cubeSegId.categoryId,
                        instance_id = cubeSegId.instanceId,
                        parent_id = 0,
                        bbox = new float[] { adjustedBBox.x, adjustedBBox.y, adjustedBBox.width, adjustedBBox.height },
                        area = cubeVisible,
                        segmentation_color = new int[] { 
                            Mathf.RoundToInt(cubeColor.r * 255), 
                            Mathf.RoundToInt(cubeColor.g * 255), 
                            Mathf.RoundToInt(cubeColor.b * 255) 
                        },
                        visibility_ratio = cubeVisibility
                    });
                }
            }

            // Проверяем параллелепипед
            int paraKey = paraSegId.instanceId * 1000 + paraSegId.categoryId;
            int paraVisible = visiblePixels.ContainsKey(paraKey) ? visiblePixels[paraKey] : 0;
            
            Bounds paraBounds = structure.parallelepiped.GetComponent<Renderer>().bounds;
            Rect paraBBox = GetScreenBoundingBox(paraBounds);
            float paraTheoreticalArea = paraBBox.width * paraBBox.height;
            
            float paraVisibility = paraTheoreticalArea > 0 ? paraVisible / paraTheoreticalArea : 0;
            
            if (paraVisible > 5 && paraVisibility >= minVisibilityRatio && IsInCropArea(paraBBox))
            {
                Rect adjustedBBox = AdjustBBoxForCrop(paraBBox);
                
                if (adjustedBBox.width > 0 && adjustedBBox.height > 0)
                {
                    Color paraColor = GetInstanceColor(paraSegId.instanceId, paraSegId.categoryId);
                    allAnnotations.Add(new ObjectAnnotation
                    {
                        id = annotationIdCounter++,
                        image_id = imageId,
                        category_id = paraSegId.categoryId,
                        instance_id = paraSegId.instanceId,
                        parent_id = cubeSegId.instanceId,
                        bbox = new float[] { adjustedBBox.x, adjustedBBox.y, adjustedBBox.width, adjustedBBox.height },
                        area = paraVisible,
                        segmentation_color = new int[] { 
                            Mathf.RoundToInt(paraColor.r * 255), 
                            Mathf.RoundToInt(paraColor.g * 255), 
                            Mathf.RoundToInt(paraColor.b * 255) 
                        },
                        visibility_ratio = paraVisibility
                    });
                }
            }
        }
    }

    private bool IsInCropArea(Rect bbox)
    {
        if (!isCurrentlyCropped) return true;
        
        // Проверяем пересечение с областью кропа
        return bbox.Overlaps(currentCropRect);
    }

    private Rect AdjustBBoxForCrop(Rect bbox)
    {
        if (!isCurrentlyCropped) return bbox;
        
        // Ограничиваем bbox областью кропа
        float x1 = Mathf.Max(bbox.x, currentCropRect.x);
        float y1 = Mathf.Max(bbox.y, currentCropRect.y);
        float x2 = Mathf.Min(bbox.x + bbox.width, currentCropRect.x + currentCropRect.width);
        float y2 = Mathf.Min(bbox.y + bbox.height, currentCropRect.y + currentCropRect.height);
        
        // Переводим в координаты кропа
        float newX = x1 - currentCropRect.x;
        float newY = y1 - currentCropRect.y;
        float newWidth = x2 - x1;
        float newHeight = y2 - y1;
        
        return new Rect(newX, newY, newWidth, newHeight);
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

        minX = Mathf.Max(0, minX);
        minY = Mathf.Max(0, minY);
        maxX = Mathf.Min(imageWidth, maxX);
        maxY = Mathf.Min(imageHeight, maxY);

        float y = imageHeight - maxY;
        
        return new Rect(minX, y, maxX - minX, maxY - minY);
    }

    private void SaveAnnotations(string basePath)
    {
        StringBuilder sb = new StringBuilder();
        sb.AppendLine("{");
        
        sb.AppendLine("  \"images\": [");
        for (int i = 0; i < allImages.Count; i++)
        {
            var img = allImages[i];
            sb.Append($"    {{\"id\": {img.id}, \"file_name\": \"{img.file_name}\", \"width\": {img.width}, \"height\": {img.height}}}");
            if (i < allImages.Count - 1) sb.AppendLine(",");
            else sb.AppendLine();
        }
        sb.AppendLine("  ],");

        sb.AppendLine("  \"categories\": [");
        sb.AppendLine("    {\"id\": 1, \"name\": \"red_cube\"},");
        sb.AppendLine("    {\"id\": 2, \"name\": \"green_parallelepiped\"}");
        sb.AppendLine("  ],");

        sb.AppendLine("  \"annotations\": [");
        for (int i = 0; i < allAnnotations.Count; i++)
        {
            var ann = allAnnotations[i];
            string bboxStr = string.Format(CultureInfo.InvariantCulture, 
                "[{0:F2}, {1:F2}, {2:F2}, {3:F2}]", 
                ann.bbox[0], ann.bbox[1], ann.bbox[2], ann.bbox[3]);
            string areaStr = ann.area.ToString("F2", CultureInfo.InvariantCulture);
            string visStr = ann.visibility_ratio.ToString("F2", CultureInfo.InvariantCulture);
            
            sb.Append($"    {{\"id\": {ann.id}, \"image_id\": {ann.image_id}, \"category_id\": {ann.category_id}, ");
            sb.Append($"\"instance_id\": {ann.instance_id}, \"parent_id\": {ann.parent_id}, ");
            sb.Append($"\"bbox\": {bboxStr}, ");
            sb.Append($"\"area\": {areaStr}, ");
            sb.Append($"\"visibility_ratio\": {visStr}, ");
            sb.Append($"\"segmentation_color\": [{ann.segmentation_color[0]}, {ann.segmentation_color[1]}, {ann.segmentation_color[2]}]}}");
            if (i < allAnnotations.Count - 1) sb.AppendLine(",");
            else sb.AppendLine();
        }
        sb.AppendLine("  ]");
        
        sb.AppendLine("}");

        File.WriteAllText(Path.Combine(basePath, "annotations.json"), sb.ToString());
    }
}
