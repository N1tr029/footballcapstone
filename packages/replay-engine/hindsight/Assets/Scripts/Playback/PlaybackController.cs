using System;
using System.Collections.Generic;
using Hindsight.Data;
using UnityEngine;

namespace Hindsight.Playback
{
    /// <summary>
    /// Replays a play exactly as recorded. Nothing is simulated: each Update turns elapsed
    /// playback time into a fractional frame and interpolates between the two nearest
    /// tracking frames. Data (x, y) maps to Unity (x, height, y); 1 unit = 1 yard.
    /// </summary>
    public class PlaybackController : MonoBehaviour
    {
        public static readonly float[] Speeds = { 0.25f, 0.5f, 1f, 2f };

        [SerializeField] PlayerView playerPrefab;
        [SerializeField] BallView ballPrefab;
        [Tooltip("Restart begins this many seconds before the snap. Negative = from the first frame.")]
        [SerializeField] float leadInSeconds = 1.5f;
        [Tooltip("Ball height when on the ground before the snap.")]
        [SerializeField] float ballGroundHeight = 0.15f;
        [Tooltip("Ball height while carried. Above the player bodies so it stays visible top-down.")]
        [SerializeField] float ballCarryHeight = 1.2f;

        public PlayTimeline Timeline { get; private set; }
        public bool IsPlaying { get; private set; }
        public int SpeedIndex { get; private set; } = 2;
        public float Speed => Speeds[SpeedIndex];
        /// <summary>Current fractional frame index into the timeline.</summary>
        public float CurrentFrame => Timeline == null ? 0f : elapsed * Timeline.FrameRate;
        public BallView Ball => ball;

        public event Action<PlayTimeline> PlayChanged;

        readonly List<PlayerView> views = new();
        Transform viewRoot;
        BallView ball;
        float elapsed; // seconds of play time since frame 0

        public void Configure(PlayerView player, BallView ballView)
        {
            playerPrefab = player;
            ballPrefab = ballView;
        }

        public void SetPlay(PlayData play)
        {
            Timeline = new PlayTimeline(play);
            SpawnViews();
            ball.ResetHeading();
            Restart();
            IsPlaying = true;
            PlayChanged?.Invoke(Timeline);
        }

        public void TogglePlay()
        {
            if (Timeline == null) return;
            if (!IsPlaying && CurrentFrame >= Timeline.LastFrame) Restart();
            IsPlaying = !IsPlaying;
        }

        /// <summary>Pause and move to the next/previous whole frame.</summary>
        public void Step(int direction)
        {
            if (Timeline == null) return;
            IsPlaying = false;
            float frame = CurrentFrame;
            float target = direction > 0
                ? Mathf.Floor(frame + 0.001f) + 1f
                : Mathf.Ceil(frame - 0.001f) - 1f;
            SeekFrame(target);
        }

        public void Restart()
        {
            if (Timeline == null) return;
            float start = leadInSeconds < 0f ? 0f : Timeline.SnapIndex - leadInSeconds * Timeline.FrameRate;
            SeekFrame(start);
        }

        public void SetSpeedIndex(int index) => SpeedIndex = Mathf.Clamp(index, 0, Speeds.Length - 1);

        public void SeekFrame(float frame)
        {
            if (Timeline == null) return;
            elapsed = Mathf.Clamp(frame, 0f, Timeline.LastFrame) / Timeline.FrameRate;
            Apply();
        }

        void Update()
        {
            if (Timeline == null) return;
            if (IsPlaying)
            {
                elapsed += Time.deltaTime * Speed;
                if (elapsed >= Timeline.Duration)
                {
                    elapsed = Timeline.Duration;
                    IsPlaying = false;
                }
            }
            Apply();
        }

        void Apply()
        {
            float frame = CurrentFrame;
            for (int p = 0; p < views.Count; p++)
            {
                Vector2 xy = Timeline.PlayerPosition(p, frame);
                views[p].SetPose(new Vector3(xy.x, 0f, xy.y), Timeline.PlayerOrientation(p, frame));
            }

            Vector2 b = Timeline.BallPosition(frame);
            ball.SetPosition(new Vector3(b.x, BallHeight(frame), b.y));
        }

        float BallHeight(float frame)
        {
            // Rises from the turf into the QB's hands over the first few frames after the snap.
            float lift = Mathf.Clamp01((frame - Timeline.SnapIndex) / 3f);
            float baseHeight = Mathf.Lerp(ballGroundHeight, ballCarryHeight, lift);
            return baseHeight + ThrowArc.HeightAt(Timeline.Throw, frame);
        }

        void SpawnViews()
        {
            if (viewRoot == null)
            {
                viewRoot = new GameObject("Players").transform;
                viewRoot.SetParent(transform, false);
            }
            if (ball == null) ball = Instantiate(ballPrefab, transform);

            PlayData play = Timeline.Play;
            PlayerInfo[] players = Timeline.Players;
            while (views.Count < players.Length)
                views.Add(Instantiate(playerPrefab, viewRoot));
            for (int i = players.Length; i < views.Count; i++)
                Destroy(views[i].gameObject);
            if (views.Count > players.Length)
                views.RemoveRange(players.Length, views.Count - players.Length);

            for (int p = 0; p < players.Length; p++)
            {
                bool isHome = players[p].club == play.homeTeam;
                views[p].Setup(players[p], TeamColors.For(players[p].club, isHome));
            }
        }
    }
}
