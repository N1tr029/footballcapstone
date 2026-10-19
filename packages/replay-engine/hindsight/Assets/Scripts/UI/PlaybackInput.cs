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
    ///   C            cycle top-down / broadcast / player view
    ///   V            ride the next player        Shift+V   the previous one
    /// </summary>
    public class PlaybackInput : MonoBehaviour
    {
        public const string Help =
            "Space play/pause   Left/Right step   1-4 speed   R restart   N/P next/prev play   "
            + "C camera   V ride a player";

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

            if (cameras != null && kb.vKey.wasPressedThisFrame)
            {
                bool back = kb.leftShiftKey.isPressed || kb.rightShiftKey.isPressed;
                RidePlayer(back ? -1 : +1);
            }
        }

        /// <summary>Step the player view onto the next player, and switch to it.
        ///
        /// Holds an index rather than a reference because PlaybackController rebuilds its
        /// view list on every play; the index is re-clamped against the current list each
        /// time, so changing play while riding someone lands on a real player.</summary>
        void RidePlayer(int step)
        {
            var players = playback?.Players;
            if (players == null || players.Count == 0) return;

            riding = ((riding + step) % players.Count + players.Count) % players.Count;
            cameras.SetPovTarget(players[riding].transform);
            cameras.SetView(CameraRig.View.PlayerPov);
        }

        int riding = -1;
    }
}
