using Hindsight.Data;
using TMPro;
using UnityEngine;

namespace Hindsight.Playback
{
    /// <summary>
    /// Visual for one player. Lives on the Player prefab; swap the prefab's meshes for nicer
    /// models and keep this component. The root's forward (+z) is the player's facing.
    /// </summary>
    public class PlayerView : MonoBehaviour
    {
        static readonly int BaseColor = Shader.PropertyToID("_BaseColor");

        [Tooltip("Renderers tinted with the jersey color.")]
        [SerializeField] Renderer[] jerseyRenderers;
        [Tooltip("Renderers tinted with the number/trim color (e.g. the facing marker).")]
        [SerializeField] Renderer[] trimRenderers;
        [SerializeField] TMP_Text numberLabel;

        MaterialPropertyBlock block;

        public PlayerInfo Info { get; private set; }

        public void Setup(PlayerInfo info, TeamColors.Kit kit)
        {
            Info = info;
            name = $"{info.club} #{info.jerseyNumber} {info.name}";
            Tint(jerseyRenderers, kit.Jersey);
            Tint(trimRenderers, kit.Number);
            if (numberLabel != null)
            {
                numberLabel.text = info.jerseyNumber.ToString();
                numberLabel.color = kit.Number;
            }
        }

        /// <param name="position">Unity world position.</param>
        /// <param name="yaw">Degrees about +Y; same convention as the data's o/dir.</param>
        public void SetPose(Vector3 position, float yaw)
        {
            transform.SetPositionAndRotation(position, Quaternion.Euler(0f, yaw, 0f));
        }

        void Tint(Renderer[] renderers, Color color)
        {
            if (renderers == null) return;
            block ??= new MaterialPropertyBlock();
            foreach (var r in renderers)
            {
                if (r == null) continue;
                r.GetPropertyBlock(block);
                block.SetColor(BaseColor, color);
                r.SetPropertyBlock(block);
            }
        }
    }
}
