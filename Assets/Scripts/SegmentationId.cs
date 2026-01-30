using UnityEngine;

/// <summary>
/// Компонент для хранения ID сегментации объекта
/// </summary>
public class SegmentationId : MonoBehaviour
{
    [Tooltip("Уникальный ID экземпляра (пара куб-параллелепипед имеет одинаковый instanceId)")]
    public int instanceId;

    [Tooltip("ID категории: 1 = red_cube, 2 = green_parallelepiped")]
    public int categoryId;

    [Tooltip("ID родительского экземпляра (для параллелепипеда - его куб, для куба - 0)")]
    public int parentInstanceId;

    /// <summary>
    /// Возвращает уникальный annotation ID для COCO формата
    /// </summary>
    public int GetAnnotationId()
    {
        // instanceId * 10 + categoryId дает уникальный ID для каждой аннотации
        return instanceId * 10 + categoryId;
    }
}
