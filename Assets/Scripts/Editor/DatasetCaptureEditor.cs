using UnityEngine;
using UnityEditor;

/// <summary>
/// Editor окно для управления генерацией датасета
/// </summary>
public class DatasetCaptureEditor : EditorWindow
{
    private DatasetGenerator generator;
    private DatasetCapture capture;
    private BatchDatasetCapture batchCapture;

    [MenuItem("Tools/Dataset Capture")]
    public static void ShowWindow()
    {
        GetWindow<DatasetCaptureEditor>("Dataset Capture");
    }

    void OnGUI()
    {
        GUILayout.Label("Dataset Generation Tools", EditorStyles.boldLabel);
        GUILayout.Space(10);

        // Поиск компонентов
        if (generator == null)
            generator = FindAnyObjectByType<DatasetGenerator>();
        if (capture == null)
            capture = FindAnyObjectByType<DatasetCapture>();
        if (batchCapture == null)
            batchCapture = FindAnyObjectByType<BatchDatasetCapture>();

        // Генерация объектов
        GUILayout.Label("1. Object Generation", EditorStyles.boldLabel);
        
        if (generator != null)
        {
            EditorGUILayout.ObjectField("Generator", generator, typeof(DatasetGenerator), true);
            
            if (GUILayout.Button("Generate Structures"))
            {
                generator.GenerateStructures();
            }

            if (GUILayout.Button("Clear Structures"))
            {
                generator.ClearStructures();
            }
        }
        else
        {
            EditorGUILayout.HelpBox("DatasetGenerator not found in scene. Add it to a GameObject.", MessageType.Warning);
            
            if (GUILayout.Button("Create DatasetGenerator"))
            {
                GameObject go = new GameObject("DatasetGenerator");
                go.AddComponent<DatasetGenerator>();
                generator = go.GetComponent<DatasetGenerator>();
            }
        }

        GUILayout.Space(20);

        // Batch захват датасета (1000 изображений)
        GUILayout.Label("2. Batch Dataset Capture (1000 images)", EditorStyles.boldLabel);
        
        if (batchCapture != null)
        {
            EditorGUILayout.ObjectField("Batch Capture", batchCapture, typeof(BatchDatasetCapture), true);
            
            EditorGUILayout.LabelField("Total Images:", batchCapture.totalImages.ToString());
            EditorGUILayout.LabelField("Images Per Scene:", batchCapture.imagesPerScene.ToString());
            EditorGUILayout.LabelField("Camera Radius:", $"{batchCapture.minOrbitRadius:F2} - {batchCapture.maxOrbitRadius:F2}");
            EditorGUILayout.LabelField("Crop Probability:", $"{batchCapture.cropProbability * 100:F0}%");
            
            GUILayout.Space(5);
            
            GUI.backgroundColor = Color.green;
            if (GUILayout.Button("▶ START BATCH CAPTURE (Play Mode)", GUILayout.Height(40)))
            {
                if (Application.isPlaying)
                {
                    batchCapture.StartBatchCapture();
                }
                else
                {
                    EditorUtility.DisplayDialog("Play Mode Required", 
                        "Please enter Play Mode before capturing the dataset.", "OK");
                }
            }
            GUI.backgroundColor = Color.white;
        }
        else
        {
            EditorGUILayout.HelpBox("BatchDatasetCapture not found. Add it to Main Camera.", MessageType.Warning);
            
            if (GUILayout.Button("Create BatchDatasetCapture on Main Camera"))
            {
                Camera mainCam = Camera.main;
                if (mainCam != null)
                {
                    batchCapture = mainCam.gameObject.AddComponent<BatchDatasetCapture>();
                    batchCapture.mainCamera = mainCam;
                }
                else
                {
                    EditorUtility.DisplayDialog("No Camera", 
                        "No main camera found in scene.", "OK");
                }
            }
        }

        GUILayout.Space(20);

        // Обычный захват датасета (32 изображения)
        GUILayout.Label("3. Quick Dataset Capture (32 images)", EditorStyles.boldLabel);
        
        if (capture != null)
        {
            EditorGUILayout.ObjectField("Capture", capture, typeof(DatasetCapture), true);
            
            if (GUILayout.Button("Capture Dataset (Play Mode Required)"))
            {
                if (Application.isPlaying)
                {
                    capture.CaptureDataset();
                }
                else
                {
                    EditorUtility.DisplayDialog("Play Mode Required", 
                        "Please enter Play Mode before capturing the dataset.", "OK");
                }
            }
        }
        else
        {
            EditorGUILayout.HelpBox("DatasetCapture not found in scene.", MessageType.Info);
            
            if (GUILayout.Button("Create DatasetCapture on Main Camera"))
            {
                Camera mainCam = Camera.main;
                if (mainCam != null)
                {
                    capture = mainCam.gameObject.AddComponent<DatasetCapture>();
                    capture.mainCamera = mainCam;
                }
            }
        }

        GUILayout.Space(20);

        // Информация
        GUILayout.Label("4. Output Info", EditorStyles.boldLabel);
        
        if (batchCapture != null)
        {
            EditorGUILayout.LabelField("Output Folder:", batchCapture.outputFolder);
            EditorGUILayout.LabelField("Image Size:", $"{batchCapture.imageWidth}x{batchCapture.imageHeight}");
            EditorGUILayout.LabelField("Save Visualizations:", batchCapture.saveVisualizations ? "Yes" : "No");
            EditorGUILayout.LabelField("Min Visibility:", $"{batchCapture.minVisibilityRatio * 100:F0}%");
        }
        else if (capture != null)
        {
            EditorGUILayout.LabelField("Output Folder:", capture.outputFolder);
            EditorGUILayout.LabelField("Image Size:", $"{capture.imageWidth}x{capture.imageHeight}");
            EditorGUILayout.LabelField("Total Images:", $"{capture.horizontalSteps * capture.verticalSteps}");
        }
    }
}
