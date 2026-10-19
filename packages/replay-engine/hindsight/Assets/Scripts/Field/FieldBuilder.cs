using System.Collections.Generic;
using TMPro;
using UnityEngine;

namespace Hindsight.Field
{
    /// <summary>
    /// Builds a flat, low-poly field in code. 1 unit = 1 yard. The field spans
    /// x 0..120 (end zones 0..10 and 110..120) and z 0..53.33 (data y).
    /// All flat geometry is batched into a few meshes; colors come from instances of
    /// one URP Unlit material so the shader is always included in builds.
    /// </summary>
    public class FieldBuilder : MonoBehaviour
    {
        public const float Length = 120f;
        public const float Width = 160f / 3f;   // 53.33 yards
        public const float GoalLineLeft = 10f;
        public const float GoalLineRight = 110f;
        /// <summary>Inner edge of NFL hash marks, 70'9" from each sideline.</summary>
        public const float HashFromSideline = 70.75f / 3f;

        const float LineWidth = 4f / 36f;       // 4 inches
        const float BorderWidth = 0.6f;
        const float TickLength = 2f / 3f;       // 2 feet
        const float NumberCenterFromSideline = 8f; // tops of 2 yd numbers sit 9 yd in

        // Draw heights (stacked to avoid z-fighting).
        const float SurroundY = -0.02f;
        const float GrassY = 0f;
        const float EndZoneY = 0.003f;
        const float LineY = 0.01f;
        const float MarkerY = 0.015f;
        const float NumberY = 0.012f;

        [SerializeField] Color surroundColor = new Color32(0x2C, 0x5A, 0x35, 0xFF);
        [SerializeField] Color grassColor = new Color32(0x3B, 0x7D, 0x40, 0xFF);
        [SerializeField] Color endZoneColor = new Color32(0x2A, 0x66, 0x52, 0xFF);
        [SerializeField] Color lineColor = new Color32(0xF2, 0xF2, 0xF2, 0xFF);
        [SerializeField] Color scrimmageColor = new Color32(0x3D, 0x8B, 0xFF, 0xFF);
        [SerializeField] Color firstDownColor = new Color32(0xFF, 0xD4, 0x1F, 0xFF);
        [SerializeField] float numberFontSize = 22f;

        Material baseMaterial;
        Transform scrimmageLine;
        Transform firstDownLine;

        public void Build(Material unlit)
        {
            baseMaterial = unlit;

            var surround = new QuadMeshBuilder();
            surround.Rect(-6f, -6f, Length + 6f, Width + 6f, SurroundY);
            AddMesh("Surround", surround, surroundColor);

            var grass = new QuadMeshBuilder();
            grass.Rect(0f, 0f, Length, Width, GrassY);
            AddMesh("Grass", grass, grassColor);

            var endZones = new QuadMeshBuilder();
            endZones.Rect(0f, 0f, GoalLineLeft, Width, EndZoneY);
            endZones.Rect(GoalLineRight, 0f, Length, Width, EndZoneY);
            AddMesh("EndZones", endZones, endZoneColor);

            AddMesh("Lines", BuildLines(), lineColor);
            BuildNumbers();

            scrimmageLine = AddMesh("LineOfScrimmage", VerticalStripe(0.25f), scrimmageColor).transform;
            firstDownLine = AddMesh("FirstDownLine", VerticalStripe(0.25f), firstDownColor).transform;
            SetMarkers(-1f, -1f);
        }

        /// <summary>Position the line of scrimmage and line to gain (field x). Negative hides.</summary>
        public void SetMarkers(float lineOfScrimmage, float firstDown)
        {
            PlaceMarker(scrimmageLine, lineOfScrimmage);
            PlaceMarker(firstDownLine, firstDown);
        }

        static void PlaceMarker(Transform marker, float x)
        {
            if (marker == null) return;
            marker.gameObject.SetActive(x >= 0f);
            marker.localPosition = new Vector3(x, 0f, 0f);
        }

        QuadMeshBuilder BuildLines()
        {
            var b = new QuadMeshBuilder();
            float h = LineWidth / 2f;

            // Border around the whole field, drawn just outside the playing surface.
            b.Rect(-BorderWidth, -BorderWidth, Length + BorderWidth, 0f, LineY);
            b.Rect(-BorderWidth, Width, Length + BorderWidth, Width + BorderWidth, LineY);
            b.Rect(-BorderWidth, 0f, 0f, Width, LineY);
            b.Rect(Length, 0f, Length + BorderWidth, Width, LineY);

            // Yard lines every 5 yards, goal line to goal line. Goal lines are 8 inches wide.
            for (int x = (int)GoalLineLeft; x <= (int)GoalLineRight; x += 5)
            {
                float w = x == (int)GoalLineLeft || x == (int)GoalLineRight ? LineWidth : h;
                b.Rect(x - w, 0f, x + w, Width, LineY);
            }

            // One-yard ticks: along both sidelines and at both hash rows.
            float nearHash = HashFromSideline;
            float farHash = Width - HashFromSideline;
            for (int x = (int)GoalLineLeft + 1; x < (int)GoalLineRight; x++)
            {
                if (x % 5 == 0) continue;
                b.Rect(x - h, 0.11f, x + h, 0.11f + TickLength, LineY);
                b.Rect(x - h, Width - 0.11f - TickLength, x + h, Width - 0.11f, LineY);
                b.Rect(x - h, nearHash - TickLength, x + h, nearHash, LineY);
                b.Rect(x - h, farHash, x + h, farHash + TickLength, LineY);
            }
            return b;
        }

        // Yard numbers 10..50..10. Bottoms face the nearest sideline, as on a real field,
        // so the far-side numbers read upside down from the top-down camera.
        void BuildNumbers()
        {
            var root = new GameObject("YardNumbers").transform;
            root.SetParent(transform, false);
            for (int x = 20; x <= 100; x += 10)
            {
                int yard = x <= 60 ? x - 10 : 110 - x;
                AddNumber(root, yard, x, NumberCenterFromSideline, 0f);
                AddNumber(root, yard, x, Width - NumberCenterFromSideline, 180f);
            }
        }

        void AddNumber(Transform parent, int yard, float x, float z, float yaw)
        {
            var go = new GameObject($"Yard{yard}");
            go.transform.SetParent(parent, false);
            go.transform.SetLocalPositionAndRotation(new Vector3(x, NumberY, z), Quaternion.Euler(90f, yaw, 0f));
            var text = go.AddComponent<TextMeshPro>();
            text.text = yard.ToString();
            text.fontSize = numberFontSize;
            text.fontStyle = FontStyles.Bold;
            text.alignment = TextAlignmentOptions.Center;
            text.characterSpacing = 30f;
            text.color = lineColor;
            text.textWrappingMode = TextWrappingModes.NoWrap;
            text.rectTransform.sizeDelta = new Vector2(8f, 3f);
        }

        static QuadMeshBuilder VerticalStripe(float width)
        {
            var b = new QuadMeshBuilder();
            b.Rect(-width / 2f, 0f, width / 2f, Width, MarkerY);
            return b;
        }

        GameObject AddMesh(string meshName, QuadMeshBuilder builder, Color color)
        {
            var go = new GameObject(meshName);
            go.transform.SetParent(transform, false);
            go.AddComponent<MeshFilter>().sharedMesh = builder.ToMesh(meshName);
            var renderer = go.AddComponent<MeshRenderer>();
            renderer.sharedMaterial = MaterialFor(color);
            renderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            renderer.receiveShadows = false;
            return go;
        }

        readonly Dictionary<Color, Material> materials = new();

        Material MaterialFor(Color color)
        {
            if (materials.TryGetValue(color, out var m)) return m;
            m = new Material(baseMaterial) { name = $"Field {ColorUtility.ToHtmlStringRGB(color)}", color = color };
            materials[color] = m;
            return m;
        }

        void OnDestroy()
        {
            foreach (var m in materials.Values) Destroy(m);
        }

        /// <summary>Accumulates flat, upward-facing rectangles into one mesh.</summary>
        class QuadMeshBuilder
        {
            readonly List<Vector3> vertices = new();
            readonly List<int> triangles = new();

            /// <summary>Axis-aligned rectangle in field coordinates (x, data y -> z).</summary>
            public void Rect(float x0, float z0, float x1, float z1, float height)
            {
                int i = vertices.Count;
                vertices.Add(new Vector3(x0, height, z0));
                vertices.Add(new Vector3(x0, height, z1));
                vertices.Add(new Vector3(x1, height, z1));
                vertices.Add(new Vector3(x1, height, z0));
                triangles.AddRange(new[] { i, i + 1, i + 2, i, i + 2, i + 3 });
            }

            public Mesh ToMesh(string meshName)
            {
                var mesh = new Mesh { name = meshName };
                if (vertices.Count > 65000) mesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;
                mesh.SetVertices(vertices);
                mesh.SetTriangles(triangles, 0);
                var normals = new Vector3[vertices.Count];
                for (int n = 0; n < normals.Length; n++) normals[n] = Vector3.up;
                mesh.normals = normals;
                mesh.RecalculateBounds();
                return mesh;
            }
        }
    }
}
