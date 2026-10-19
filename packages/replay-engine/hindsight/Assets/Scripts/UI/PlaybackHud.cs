using Hindsight.Data;
using Hindsight.Playback;
using TMPro;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.InputSystem.UI;
using UnityEngine.UI;

namespace Hindsight.UI
{
    /// <summary>
    /// Minimal screen overlay built in code: play context at the top, the play navigator bar
    /// at the bottom, and playback state just above it.
    /// </summary>
    public class PlaybackHud : MonoBehaviour
    {
        [SerializeField] float fontSize = 22f;
        [SerializeField] float navigatorHeight = 116f;

        const float Margin = 16f;

        PlaybackController playback;
        PlayLoader loader;
        TextMeshProUGUI header;
        TextMeshProUGUI status;
        TextMeshProUGUI simulatedBadge;
        string loadError;
        string hoverText;

        public void Init(PlaybackController playbackController, PlayLoader playLoader)
        {
            playback = playbackController;
            loader = playLoader;
            BuildCanvas();
            playback.PlayChanged += OnPlayChanged;
            loader.LoadFailed += message => loadError = message;
        }

        void OnDestroy()
        {
            if (playback != null) playback.PlayChanged -= OnPlayChanged;
        }

        void OnPlayChanged(PlayTimeline timeline)
        {
            loadError = null;
            PlayData p = timeline.Play;
            string matchup = p.possessionTeam == p.homeTeam
                ? $"{p.visitorTeam} @ <b>{p.homeTeam}</b>"
                : $"<b>{p.visitorTeam}</b> @ {p.homeTeam}";
            header.text =
                $"{matchup}   Q{p.quarter} {p.gameClock}   {Ordinal(p.down)} & {p.yardsToGo}   " +
                $"<size=80%>drive {p.driveNumber}, play {loader.CurrentIndex + 1}/{loader.Plays.Count}</size>\n" +
                $"<size=80%>{p.description}</size>";

            // A simulated play must never draw like a measured one. The description
            // says so too, but a description is read once and a badge is on every
            // frame — which is the standard docs/roles.md sets: "The simulation is
            // never presented as fact. Every simulated frame is labelled."
            ShowSimulated(p.simulated || AnyOrientationInferred(p));
        }

        static bool AnyOrientationInferred(PlayData play)
        {
            // Not only fully simulated plays: a track from video has every body angle
            // inferred from its own velocity, and a viewer reading a player's facing
            // off the screen deserves to know that nothing measured it.
            if (play.frames == null) return false;
            foreach (var frame in play.frames)
            {
                if (frame.players == null) continue;
                foreach (var p in frame.players)
                {
                    if (!p.OrientationMeasured) return true;
                }
            }
            return false;
        }

        void ShowSimulated(bool on)
        {
            if (simulatedBadge == null) return;
            simulatedBadge.gameObject.SetActive(on);
        }

        void Update()
        {
            if (status == null) return;
            if (loadError != null)
            {
                status.text = $"<color=#FF6B6B>{loadError}</color>";
                return;
            }
            PlayTimeline t = playback.Timeline;
            if (t == null)
            {
                status.text = "Loading...";
                return;
            }
            float frame = playback.CurrentFrame;
            string state = playback.IsPlaying ? "PLAYING" : "PAUSED";
            string evt = t.EventAt(frame) ?? "-";
            status.text =
                $"{state}  {playback.Speed:0.##}x   frame {t.FrameIdAt(frame)}/{t.Play.frames[t.FrameCount - 1].frameId}" +
                $"   {(frame - t.SnapIndex) / t.FrameRate:+0.0;-0.0}s from snap   {evt}" +
                (loader.IsLoading ? "   loading..." : "") +
                $"\n<size=75%><color=#AAB4C0>{hoverText ?? PlaybackInput.Help}</color></size>";
        }

        static string Ordinal(int n) => n switch { 1 => "1st", 2 => "2nd", 3 => "3rd", _ => $"{n}th" };

        void BuildCanvas()
        {
            var canvasGo = new GameObject("HUD", typeof(Canvas), typeof(CanvasScaler), typeof(GraphicRaycaster));
            canvasGo.transform.SetParent(transform, false);
            var canvas = canvasGo.GetComponent<Canvas>();
            canvas.renderMode = RenderMode.ScreenSpaceOverlay;
            var scaler = canvasGo.GetComponent<CanvasScaler>();
            scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
            scaler.referenceResolution = new Vector2(1920, 1080);
            scaler.matchWidthOrHeight = 0.5f;

            header = CreateText(canvas.transform, "Header", top: true);
            status = CreateText(canvas.transform, "Status", top: false);
            ((RectTransform)status.transform).anchoredPosition = new Vector2(0f, Margin + navigatorHeight + 6f);

            var navRect = new GameObject("PlayNavigator", typeof(RectTransform)).GetComponent<RectTransform>();
            navRect.SetParent(canvas.transform, false);
            navRect.anchorMin = Vector2.zero;
            navRect.anchorMax = new Vector2(1f, 0f);
            navRect.pivot = new Vector2(0.5f, 0f);
            navRect.anchoredPosition = new Vector2(0f, Margin);
            navRect.sizeDelta = new Vector2(-2f * Margin, navigatorHeight);
            navRect.gameObject.AddComponent<PlayNavigator>().Init(loader, text => hoverText = text);

            simulatedBadge = BuildSimulatedBadge(canvas.transform);
            simulatedBadge.gameObject.SetActive(false);

            // Clicks need an EventSystem; the Input System module picks up its default UI actions.
            if (EventSystem.current == null && FindAnyObjectByType<EventSystem>() == null)
                new GameObject("EventSystem", typeof(EventSystem), typeof(InputSystemUIInputModule))
                    .transform.SetParent(transform, false);
        }

        TextMeshProUGUI BuildSimulatedBadge(Transform parent)
        {
            // Top-centre, on its own, in a colour nothing else on the HUD uses. It is
            // deliberately not subtle: this is the one thing in the viewer that
            // separates a record of what happened from a model's guess about what
            // might have.
            var go = new GameObject("SimulatedBadge", typeof(RectTransform));
            go.transform.SetParent(parent, false);
            var rt = (RectTransform)go.transform;
            rt.anchorMin = new Vector2(0.5f, 1f);
            rt.anchorMax = new Vector2(0.5f, 1f);
            rt.pivot = new Vector2(0.5f, 1f);
            rt.anchoredPosition = new Vector2(0f, -Margin);
            rt.sizeDelta = new Vector2(760f, 64f);

            var background = go.AddComponent<Image>();
            background.color = new Color(0.72f, 0.12f, 0.12f, 0.88f);
            background.raycastTarget = false;

            var labelGo = new GameObject("Label", typeof(RectTransform));
            labelGo.transform.SetParent(go.transform, false);
            var labelRt = (RectTransform)labelGo.transform;
            labelRt.anchorMin = Vector2.zero;
            labelRt.anchorMax = Vector2.one;
            labelRt.sizeDelta = Vector2.zero;

            var label = labelGo.AddComponent<TextMeshProUGUI>();
            label.text = "SIMULATED — THIS DID NOT HAPPEN";
            label.fontSize = fontSize * 1.1f;
            label.fontStyle = FontStyles.Bold;
            label.color = Color.white;
            label.alignment = TextAlignmentOptions.Center;
            label.raycastTarget = false;
            return label;
        }

        TextMeshProUGUI CreateText(Transform parent, string objectName, bool top)
        {
            var go = new GameObject(objectName, typeof(RectTransform));
            go.transform.SetParent(parent, false);
            var rt = (RectTransform)go.transform;
            rt.anchorMin = new Vector2(0f, top ? 1f : 0f);
            rt.anchorMax = new Vector2(1f, top ? 1f : 0f);
            rt.pivot = new Vector2(0.5f, top ? 1f : 0f);
            rt.anchoredPosition = new Vector2(0f, top ? -16f : 16f);
            rt.sizeDelta = new Vector2(-48f, 90f);

            var text = go.AddComponent<TextMeshProUGUI>();
            text.fontSize = fontSize;
            text.color = Color.white;
            text.alignment = top ? TextAlignmentOptions.TopLeft : TextAlignmentOptions.BottomLeft;
            text.textWrappingMode = TextWrappingModes.Normal;
            text.overflowMode = TextOverflowModes.Ellipsis;
            text.raycastTarget = false;
            return text;
        }
    }
}
