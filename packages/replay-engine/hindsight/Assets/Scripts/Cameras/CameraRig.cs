using Hindsight.Field;
using UnityEngine;

namespace Hindsight.Cameras
{
    /// <summary>
    /// Two cameras: an orthographic top-down view framing the whole field (default) and a
    /// perspective "broadcast" view from the near sideline that follows the ball along x.
    /// Both are tagged MainCamera; only the active one is enabled, so Camera.main tracks it.
    /// </summary>
    [DefaultExecutionOrder(-100)] // position cameras before billboards read them
    public class CameraRig : MonoBehaviour
    {
        public enum View { TopDown, Broadcast }

        [Header("Top-down")]
        [Tooltip("Yards of space kept around the field.")]
        [SerializeField] float topDownMargin = 4f;

        [Header("Broadcast")]
        [SerializeField] float broadcastFov = 38f;
        [SerializeField] float broadcastHeight = 24f;
        [Tooltip("Distance behind the near sideline (data y = 0).")]
        [SerializeField] float broadcastSetback = 22f;
        [Tooltip("Field y the camera aims at.")]
        [SerializeField] float broadcastAimY = 22f;
        [SerializeField] float followSmoothTime = 0.35f;
        [SerializeField] float minFollowX = 25f;
        [SerializeField] float maxFollowX = 95f;

        [SerializeField] Color background = new Color32(0x16, 0x1C, 0x24, 0xFF);

        public View Current { get; private set; } = View.TopDown;
        public Camera TopDown { get; private set; }
        public Camera Broadcast { get; private set; }

        Transform target;
        float followX = FieldBuilder.Length / 2f;
        float followVelocity;

        public void Build()
        {
            TopDown = CreateCamera("TopDownCamera");
            TopDown.orthographic = true;
            TopDown.transform.SetPositionAndRotation(
                new Vector3(FieldBuilder.Length / 2f, 60f, FieldBuilder.Width / 2f),
                Quaternion.Euler(90f, 0f, 0f));

            Broadcast = CreateCamera("BroadcastCamera");
            Broadcast.orthographic = false;
            Broadcast.fieldOfView = broadcastFov;

            SetView(View.TopDown);
        }

        /// <summary>What the broadcast camera follows (normally the ball).</summary>
        public void SetTarget(Transform followTarget)
        {
            target = followTarget;
            if (target != null) followX = ClampX(target.position.x);
            followVelocity = 0f;
        }

        public void Toggle() => SetView(Current == View.TopDown ? View.Broadcast : View.TopDown);

        public void SetView(View view)
        {
            Current = view;
            TopDown.gameObject.SetActive(view == View.TopDown);
            Broadcast.gameObject.SetActive(view == View.Broadcast);
        }

        void LateUpdate()
        {
            if (TopDown == null) return;

            // Fit the whole field for any aspect ratio.
            float halfLength = FieldBuilder.Length / 2f + topDownMargin;
            float halfWidth = FieldBuilder.Width / 2f + topDownMargin;
            TopDown.orthographicSize = Mathf.Max(halfWidth, halfLength / Mathf.Max(TopDown.aspect, 0.01f));

            if (target != null)
                followX = Mathf.SmoothDamp(followX, ClampX(target.position.x), ref followVelocity, followSmoothTime);
            var position = new Vector3(followX, broadcastHeight, -broadcastSetback);
            var aim = new Vector3(followX, 0f, broadcastAimY);
            Broadcast.transform.SetPositionAndRotation(position, Quaternion.LookRotation(aim - position));
        }

        float ClampX(float x) => Mathf.Clamp(x, minFollowX, maxFollowX);

        Camera CreateCamera(string cameraName)
        {
            var go = new GameObject(cameraName) { tag = "MainCamera" };
            go.transform.SetParent(transform, false);
            var cam = go.AddComponent<Camera>();
            cam.clearFlags = CameraClearFlags.SolidColor;
            cam.backgroundColor = background;
            cam.nearClipPlane = 0.3f;
            cam.farClipPlane = 300f;
            return cam;
        }
    }
}
