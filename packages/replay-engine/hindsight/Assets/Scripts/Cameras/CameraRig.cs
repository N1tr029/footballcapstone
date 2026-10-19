using Hindsight.Field;
using UnityEngine;

namespace Hindsight.Cameras
{
    /// <summary>
    /// Three cameras: an orthographic top-down view framing the whole field (default), a
    /// perspective "broadcast" view from the near sideline that follows the ball along x,
    /// and a first-person view riding one player's helmet.
    /// All are tagged MainCamera; only the active one is enabled, so Camera.main tracks it.
    ///
    /// The player view is the one with a caveat. The data contains no head tracking and
    /// never will — no public football dataset has a gaze column — so this looks along the
    /// player's BODY orientation (o), which is what is measured. A receiver running a
    /// corner with his head turned back at the ball is drawn facing the corner, because
    /// that is the only thing the data knows. It is a useful view of where a player was and
    /// what was around him; it is not what he saw.
    /// </summary>
    [DefaultExecutionOrder(-100)] // position cameras before billboards read them
    public class CameraRig : MonoBehaviour
    {
        public enum View { TopDown, Broadcast, PlayerPov }

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

        [Header("Player view")]
        [SerializeField] float povFov = 72f;
        [Tooltip("Eye height in yards above the player's feet.")]
        [SerializeField] float povEyeHeight = 1.05f;
        [Tooltip("Forward offset so the player's own body is not filling the lens.")]
        [SerializeField] float povForwardOffset = 0.45f;
        [Tooltip("Yaw smoothing. 0 is rigid, which on 10 Hz data reads as snapping.")]
        [SerializeField] float povTurnSmoothTime = 0.12f;

        [SerializeField] Color background = new Color32(0x16, 0x1C, 0x24, 0xFF);

        public View Current { get; private set; } = View.TopDown;
        public Camera TopDown { get; private set; }
        public Camera Broadcast { get; private set; }
        public Camera PlayerPov { get; private set; }
        /// <summary>Whose helmet the player view is riding, or null.</summary>
        public Transform PovTarget => povTarget;

        Transform target;
        float followX = FieldBuilder.Length / 2f;
        float followVelocity;
        Transform povTarget;
        float povYaw;
        float povYawVelocity;
        bool povYawPrimed;

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

            PlayerPov = CreateCamera("PlayerPovCamera");
            PlayerPov.orthographic = false;
            PlayerPov.fieldOfView = povFov;
            // Closer than the shared default: the lens sits just in front of a body.
            PlayerPov.nearClipPlane = 0.05f;

            SetView(View.TopDown);
        }

        /// <summary>What the broadcast camera follows (normally the ball).</summary>
        public void SetTarget(Transform followTarget)
        {
            target = followTarget;
            if (target != null) followX = ClampX(target.position.x);
            followVelocity = 0f;
        }

        /// <summary>Whose helmet to ride in the player view. Null falls back to broadcast.</summary>
        public void SetPovTarget(Transform player)
        {
            povTarget = player;
            povYawPrimed = false;
            if (povTarget == null && Current == View.PlayerPov) SetView(View.Broadcast);
        }

        /// <summary>Cycle top-down -> broadcast -> player, skipping the player view when
        /// nobody has been selected to ride.</summary>
        public void Toggle()
        {
            View next = Current switch
            {
                View.TopDown => View.Broadcast,
                View.Broadcast => povTarget != null ? View.PlayerPov : View.TopDown,
                _ => View.TopDown,
            };
            SetView(next);
        }

        public void SetView(View view)
        {
            if (view == View.PlayerPov && povTarget == null) view = View.Broadcast;
            Current = view;
            TopDown.gameObject.SetActive(view == View.TopDown);
            Broadcast.gameObject.SetActive(view == View.Broadcast);
            PlayerPov.gameObject.SetActive(view == View.PlayerPov);
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

            UpdatePov();
        }

        void UpdatePov()
        {
            if (povTarget == null) return;

            // The player's facing. PlayerView sets rotation from the data's o with no
            // conversion, so +z is where the body is pointing.
            float yaw = povTarget.eulerAngles.y;
            if (!povYawPrimed)
            {
                povYaw = yaw;
                povYawVelocity = 0f;
                povYawPrimed = true;
            }
            else
            {
                // SmoothDampAngle, not SmoothDamp: a man turning past due north must not
                // spin the long way round through his own shoulder.
                povYaw = Mathf.SmoothDampAngle(povYaw, yaw, ref povYawVelocity, povTurnSmoothTime);
            }

            var rotation = Quaternion.Euler(0f, povYaw, 0f);
            var eye = povTarget.position
                      + Vector3.up * povEyeHeight
                      + rotation * Vector3.forward * povForwardOffset;
            PlayerPov.transform.SetPositionAndRotation(eye, rotation);
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
