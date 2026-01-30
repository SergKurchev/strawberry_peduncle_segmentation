using UnityEngine;
using System.Collections.Generic;

/// <summary>
/// Генерирует случайные структуры: красный куб (3×3×3 см) с зелёным параллелепипедом (0.1×0.1×2 см) сверху.
/// </summary>
public class DatasetGenerator : MonoBehaviour
{
    [Header("Generation Settings")]
    [Tooltip("Минимальное количество структур")]
    public int minStructures = 5;
    
    [Tooltip("Максимальное количество структур")]
    public int maxStructures = 10;
    
    [Tooltip("Размер области генерации (метры)")]
    public Vector3 spawnAreaSize = new Vector3(0.5f, 0.5f, 0.5f);
    
    [Tooltip("Центр области генерации")]
    public Vector3 spawnAreaCenter = Vector3.zero;

    [Header("Materials")]
    public Material redMaterial;
    public Material greenMaterial;

    [Header("Segmentation Materials")]
    public Material segmentationMaterial;

    // Хранение сгенерированных объектов
    [HideInInspector]
    public List<StructureData> generatedStructures = new List<StructureData>();

    [System.Serializable]
    public class StructureData
    {
        public int instanceId;
        public GameObject cube;
        public GameObject parallelepiped;
        public Color cubeSegmentColor;
        public Color parallelepipedSegmentColor;
    }

    private int nextInstanceId = 1;

    void Start()
    {
        GenerateStructures();
    }

    [ContextMenu("Regenerate Structures")]
    public void GenerateStructures()
    {
        ClearStructures();
        
        int count = Random.Range(minStructures, maxStructures + 1);
        
        for (int i = 0; i < count; i++)
        {
            CreateStructure();
        }
        
        Debug.Log($"Сгенерировано {count} структур");
    }

    public void ClearStructures()
    {
        foreach (var structure in generatedStructures)
        {
            if (structure.cube != null)
                DestroyImmediate(structure.cube);
        }
        generatedStructures.Clear();
        nextInstanceId = 1;
    }

    private void CreateStructure()
    {
        // Размеры в Unity единицах (метрах)
        float cubeSize = 0.03f; // 3 см
        Vector3 parallelepipedSize = new Vector3(0.003f, 0.02f, 0.003f); // 0.3×2×0.3 см (вытянут по Y)

        // Случайная позиция в области генерации
        Vector3 position = spawnAreaCenter + new Vector3(
            Random.Range(-spawnAreaSize.x / 2f, spawnAreaSize.x / 2f),
            Random.Range(-spawnAreaSize.y / 2f, spawnAreaSize.y / 2f),
            Random.Range(-spawnAreaSize.z / 2f, spawnAreaSize.z / 2f)
        );

        // Случайный поворот
        Quaternion rotation = Random.rotation;

        // Создание куба
        GameObject cube = GameObject.CreatePrimitive(PrimitiveType.Cube);
        cube.name = $"RedCube_{nextInstanceId}";
        cube.transform.position = position;
        cube.transform.rotation = rotation;
        cube.transform.localScale = new Vector3(cubeSize, cubeSize, cubeSize);

        // Создание параллелепипеда
        GameObject parallelepiped = GameObject.CreatePrimitive(PrimitiveType.Cube);
        parallelepiped.name = $"GreenParallelepiped_{nextInstanceId}";
        parallelepiped.transform.SetParent(cube.transform);
        
        // Позиция на верхней грани куба
        // Куб имеет размер 1, масштабирован до cubeSize
        // Верхняя грань на локальной Y = 0.5
        // Параллелепипед своим нижним основанием касается верхней грани
        float parallelepipedLocalY = 0.5f + (parallelepipedSize.y / 2f) / cubeSize;
        parallelepiped.transform.localPosition = new Vector3(0, parallelepipedLocalY, 0);
        parallelepiped.transform.localRotation = Quaternion.identity;
        parallelepiped.transform.localScale = new Vector3(
            parallelepipedSize.x / cubeSize,
            parallelepipedSize.y / cubeSize,
            parallelepipedSize.z / cubeSize
        );

        // Применение материалов
        var cubeRenderer = cube.GetComponent<Renderer>();
        var paraRenderer = parallelepiped.GetComponent<Renderer>();
        
        if (redMaterial != null)
            cubeRenderer.sharedMaterial = redMaterial;
        else
        {
            // Создаём новый красный материал
            Material redMat = new Material(Shader.Find("Universal Render Pipeline/Lit"));
            redMat.color = Color.red;
            cubeRenderer.sharedMaterial = redMat;
        }

        if (greenMaterial != null)
            paraRenderer.sharedMaterial = greenMaterial;
        else
        {
            // Создаём новый зелёный материал
            Material greenMat = new Material(Shader.Find("Universal Render Pipeline/Lit"));
            greenMat.color = Color.green;
            paraRenderer.sharedMaterial = greenMat;
        }

        // ID хранится в компоненте SegmentationId, теги не нужны

        // Сохранение данных структуры
        // Используем разные ID для куба и параллелепипеда
        int cubeIdValue = nextInstanceId;
        int paraIdValue = nextInstanceId + 1;

        StructureData data = new StructureData
        {
            instanceId = cubeIdValue, // Основной ID структуры (куба)
            cube = cube,
            parallelepiped = parallelepiped,
            cubeSegmentColor = GetSegmentationColor(cubeIdValue),
            parallelepipedSegmentColor = GetSegmentationColor(paraIdValue)
        };

        // Добавление компонента для хранения ID
        var cubeId = cube.AddComponent<SegmentationId>();
        cubeId.instanceId = cubeIdValue;
        cubeId.categoryId = 1; // red_cube
        cubeId.parentInstanceId = 0; // No parent

        var paraId = parallelepiped.AddComponent<SegmentationId>();
        paraId.instanceId = paraIdValue;
        paraId.categoryId = 2; // green_parallelepiped
        paraId.parentInstanceId = cubeIdValue; // Parent is the cube

        generatedStructures.Add(data);
        nextInstanceId += 2; // Увеличиваем на 2, так как создали 2 объекта
    }

    /// <summary>
    /// Генерирует уникальный цвет для сегментации по ID
    /// </summary>
    public static Color GetSegmentationColor(int id)
    {
        // Используем ID для генерации уникального цвета
        float r = ((id * 11) % 255) / 255f;
        float g = ((id * 23) % 255) / 255f;
        float b = ((id * 47) % 255) / 255f;
        return new Color(r, g, b, 1f);
    }

    void OnDrawGizmosSelected()
    {
        // Визуализация области генерации в редакторе
        Gizmos.color = new Color(1, 1, 0, 0.3f);
        Gizmos.DrawWireCube(spawnAreaCenter, spawnAreaSize);
    }
}
