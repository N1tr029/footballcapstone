using UnityEngine;

namespace Hindsight.UI
{
    /// <summary>Keeps this object facing the active camera (used for jersey numbers).</summary>
    public class Billboard : MonoBehaviour
    {
        void LateUpdate()
        {
            Camera cam = Camera.main;
            if (cam != null)
                transform.rotation = cam.transform.rotation;
        }
    }
}
