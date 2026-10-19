using Hindsight.Playback;
using Hindsight.UI;
using TMPro;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace Hindsight.EditorTools
{
    /// <summary>
    /// Generates the materials and prefabs from code and adds the app to the open scene,
    /// so nothing has to be hand-edited. Safe to re-run; it overwrites the generated assets.
    /// Prefabs are only regenerated when missing, so swapped-in models are kept.
    /// </summary>
    public static class HindsightSetup
    {
        const string MaterialsFolder = "Assets/Art/Materials";
        const string PrefabsFolder = "Assets/Prefabs";
        const string UnlitPath = MaterialsFolder + "/Unlit.mat";
        const string BallMaterialPath = MaterialsFolder + "/Ball.mat";
        const string PlayerPrefabPath = PrefabsFolder + "/Player.prefab";
        const string BallPrefabPath = PrefabsFolder + "/Ball.prefab";
        const string UnlitShader = "Universal Render Pipeline/Unlit";

        static readonly Color BallColor = new Color32(0xFF, 0x7A, 0x1A, 0xFF);

        [MenuItem("Hindsight/Setup Scene")]
        public static void SetupScene()
        {
            if (!EnsureTmpResources()) return;

            Material unlit = EnsureMaterial(UnlitPath, Color.white);
            Material ballMaterial = EnsureMaterial(BallMaterialPath, BallColor);
            if (unlit == null || ballMaterial == null) return;

            PlayerView player = AssetDatabase.LoadAssetAtPath<PlayerView>(PlayerPrefabPath) ?? CreatePlayerPrefab(unlit);
            BallView ball = AssetDatabase.LoadAssetAtPath<BallView>(BallPrefabPath) ?? CreateBallPrefab(ballMaterial);

            AddAppToScene(unlit, player, ball);
        }

        [MenuItem("Hindsight/Regenerate Prefabs")]
        public static void RegeneratePrefabs()
        {
            if (!EnsureTmpResources()) return;
            Material unlit = EnsureMaterial(UnlitPath, Color.white);
            Material ballMaterial = EnsureMaterial(BallMaterialPath, BallColor);
            if (unlit == null || ballMaterial == null) return;
            CreatePlayerPrefab(unlit);
            CreateBallPrefab(ballMaterial);
            Debug.Log("[Hindsight] Regenerated Player and Ball prefabs.");
        }

        static bool EnsureTmpResources()
        {
            if (Resources.Load<TMP_Settings>("TMP Settings") != null) return true;
            EditorUtility.DisplayDialog("Hindsight setup",
                "TextMeshPro Essential Resources are not imported yet.\n\n" +
                "Use Window > TextMeshPro > Import TMP Essential Resources, then run Hindsight > Setup Scene again.",
                "OK");
            return false;
        }

        static Material EnsureMaterial(string path, Color color)
        {
            EnsureFolder(MaterialsFolder);
            var material = AssetDatabase.LoadAssetAtPath<Material>(path);
            if (material != null) return material;

            Shader shader = Shader.Find(UnlitShader);
            if (shader == null)
            {
                Debug.LogError($"[Hindsight] Shader '{UnlitShader}' not found. Is URP installed and active?");
                return null;
            }
            material = new Material(shader) { color = color };
            AssetDatabase.CreateAsset(material, path);
            return material;
        }

        static PlayerView CreatePlayerPrefab(Material unlit)
        {
            EnsureFolder(PrefabsFolder);
            var root = new GameObject("Player");
            var view = root.AddComponent<PlayerView>();

            // Body: 0.9 yd wide, 0.8 yd tall (cylinder primitive is 2 units tall).
            GameObject body = Primitive(PrimitiveType.Cylinder, "Body", root.transform, unlit);
            body.transform.localPosition = new Vector3(0f, 0.4f, 0f);
            body.transform.localScale = new Vector3(0.9f, 0.4f, 0.9f);

            // Facing marker on the front (+z) edge.
            GameObject facing = Primitive(PrimitiveType.Cube, "Facing", root.transform, unlit);
            facing.transform.localPosition = new Vector3(0f, 0.55f, 0.45f);
            facing.transform.localScale = new Vector3(0.3f, 0.3f, 0.3f);

            var label = new GameObject("Number", typeof(RectTransform));
            label.transform.SetParent(root.transform, false);
            label.transform.localPosition = new Vector3(0f, 0.85f, 0f);
            label.AddComponent<Billboard>();
            var text = label.AddComponent<TextMeshPro>();
            text.text = "00";
            text.fontSize = 5f;
            text.fontStyle = FontStyles.Bold;
            text.alignment = TextAlignmentOptions.Center;
            text.textWrappingMode = TextWrappingModes.NoWrap;
            text.rectTransform.sizeDelta = new Vector2(2f, 1f);

            var so = new SerializedObject(view);
            SetArray(so.FindProperty("jerseyRenderers"), body.GetComponent<Renderer>());
            SetArray(so.FindProperty("trimRenderers"), facing.GetComponent<Renderer>());
            so.FindProperty("numberLabel").objectReferenceValue = text;
            so.ApplyModifiedPropertiesWithoutUndo();

            return SavePrefab(root, PlayerPrefabPath).GetComponent<PlayerView>();
        }

        static BallView CreateBallPrefab(Material ballMaterial)
        {
            EnsureFolder(PrefabsFolder);
            var root = new GameObject("Ball");
            root.AddComponent<BallView>();
            // Exaggerated football (real one is ~0.3 yd long) so it reads at full-field zoom.
            GameObject mesh = Primitive(PrimitiveType.Sphere, "Mesh", root.transform, ballMaterial);
            mesh.transform.localScale = new Vector3(0.4f, 0.4f, 0.65f);
            return SavePrefab(root, BallPrefabPath).GetComponent<BallView>();
        }

        static void AddAppToScene(Material unlit, PlayerView player, BallView ball)
        {
            var app = Object.FindAnyObjectByType<HindsightApp>();
            if (app == null)
            {
                var go = new GameObject("Hindsight");
                Undo.RegisterCreatedObjectUndo(go, "Add Hindsight");
                app = go.AddComponent<HindsightApp>();
            }

            var so = new SerializedObject(app);
            so.FindProperty("unlitMaterial").objectReferenceValue = unlit;
            so.FindProperty("playerPrefab").objectReferenceValue = player;
            so.FindProperty("ballPrefab").objectReferenceValue = ball;
            so.ApplyModifiedProperties();

            // The app creates its own cameras; remove other scene cameras so they don't render over it.
            foreach (var cam in Object.FindObjectsByType<Camera>(FindObjectsInactive.Include))
            {
                if (cam.transform.IsChildOf(app.transform)) continue;
                Debug.Log($"[Hindsight] Removed scene camera '{cam.name}' (the app creates its own).");
                Undo.DestroyObjectImmediate(cam.gameObject);
            }

            EditorSceneManager.MarkSceneDirty(app.gameObject.scene);
            Selection.activeGameObject = app.gameObject;
            Debug.Log("[Hindsight] Scene ready. Save the scene and press Play.");
        }

        static GameObject Primitive(PrimitiveType type, string objectName, Transform parent, Material material)
        {
            GameObject go = GameObject.CreatePrimitive(type);
            go.name = objectName;
            go.transform.SetParent(parent, false);
            Object.DestroyImmediate(go.GetComponent<Collider>());
            var renderer = go.GetComponent<Renderer>();
            renderer.sharedMaterial = material;
            renderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            renderer.receiveShadows = false;
            return go;
        }

        static GameObject SavePrefab(GameObject root, string path)
        {
            GameObject prefab = PrefabUtility.SaveAsPrefabAsset(root, path);
            Object.DestroyImmediate(root);
            return prefab;
        }

        static void SetArray(SerializedProperty array, params Object[] values)
        {
            array.arraySize = values.Length;
            for (int i = 0; i < values.Length; i++)
                array.GetArrayElementAtIndex(i).objectReferenceValue = values[i];
        }

        static void EnsureFolder(string path)
        {
            if (AssetDatabase.IsValidFolder(path)) return;
            string parent = System.IO.Path.GetDirectoryName(path)?.Replace('\\', '/');
            if (!string.IsNullOrEmpty(parent)) EnsureFolder(parent);
            AssetDatabase.CreateFolder(parent, System.IO.Path.GetFileName(path));
        }
    }
}
