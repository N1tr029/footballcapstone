using UnityEngine;

namespace Hindsight.Playback
{
    /// <summary>
    /// Visual for the football. Lives on the Ball prefab. Points its long axis (+z) along
    /// its direction of travel while moving.
    /// </summary>
    public class BallView : MonoBehaviour
    {
        const float MinMoveForHeading = 0.02f;

        Vector3 lastPosition;
        bool hasLast;

        public void SetPosition(Vector3 position)
        {
            Vector3 delta = position - lastPosition;
            if (hasLast && delta.sqrMagnitude > MinMoveForHeading * MinMoveForHeading)
                transform.rotation = Quaternion.LookRotation(delta);
            transform.position = position;
            lastPosition = position;
            hasLast = true;
        }

        /// <summary>Forget the previous position, e.g. after a jump to another frame or play.</summary>
        public void ResetHeading()
        {
            hasLast = false;
            transform.rotation = Quaternion.identity;
        }
    }
}
