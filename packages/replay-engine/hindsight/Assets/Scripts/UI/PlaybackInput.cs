using Hindsight.Cameras;
using Hindsight.Data;
using Hindsight.Playback;
using UnityEngine;
using UnityEngine.InputSystem;

namespace Hindsight.UI
{
    /// <summary>
    /// Keyboard controls (Input System):
    ///   Space        play / pause
    ///   Left/Right   step one frame back / forward
    ///   1-4          speed 0.25x / 0.5x / 1x / 2x
    ///   R            restart
    ///   N or ]       next play       P or [   previous play
    ///   C            toggle top-down / broadcast camera
    /// </summary>
    public class PlaybackInput : MonoBehaviour
    {
        public const string Help =
            "Space play/pause   Left/Right step   1-4 speed   R restart   N/P next/prev play   C camera";

        PlaybackController playback;
        PlayLoader loader;
        CameraRig cameras;

        public void Init(PlaybackController playbackController, PlayLoader playLoader, CameraRig cameraRig)
        {
            playback = playbackController;
            loader = playLoader;
            cameras = cameraRig;
        }

        void Update()
        {
            Keyboard kb = Keyboard.current;
            if (kb == null || playback == null) return;

            if (kb.spaceKey.wasPressedThisFrame) playback.TogglePlay();
            if (kb.rightArrowKey.wasPressedThisFrame) playback.Step(+1);
            if (kb.leftArrowKey.wasPressedThisFrame) playback.Step(-1);
            if (kb.rKey.wasPressedThisFrame) playback.Restart();

            if (kb.digit1Key.wasPressedThisFrame || kb.numpad1Key.wasPressedThisFrame) playback.SetSpeedIndex(0);
            if (kb.digit2Key.wasPressedThisFrame || kb.numpad2Key.wasPressedThisFrame) playback.SetSpeedIndex(1);
            if (kb.digit3Key.wasPressedThisFrame || kb.numpad3Key.wasPressedThisFrame) playback.SetSpeedIndex(2);
            if (kb.digit4Key.wasPressedThisFrame || kb.numpad4Key.wasPressedThisFrame) playback.SetSpeedIndex(3);

            if (loader != null)
            {
                if (kb.nKey.wasPressedThisFrame || kb.rightBracketKey.wasPressedThisFrame) loader.Next();
                if (kb.pKey.wasPressedThisFrame || kb.leftBracketKey.wasPressedThisFrame) loader.Previous();
            }

            if (cameras != null && kb.cKey.wasPressedThisFrame) cameras.Toggle();
        }
    }
}
