using System.Collections.Generic;
using Hindsight.Data;
using UnityEngine;

namespace Hindsight.Playback
{
    /// <summary>
    /// PlayData reorganized for fast sampling: flat per-player arrays indexed by frame,
    /// plus derived info (snap frame, throw window). Frame indices here are 0-based
    /// positions in play.frames, not the 1-based frameId from the data.
    /// </summary>
    public class PlayTimeline
    {
        public readonly PlayData Play;
        public readonly PlayerInfo[] Players;
        public readonly int FrameCount;
        public readonly float FrameRate;
        public readonly int SnapIndex;
        public readonly ThrowArc.Window Throw;

        readonly Vector2[] positions;   // [player * FrameCount + frame]
        readonly float[] orientations;  // degrees, 0 = +y, clockwise
        readonly Vector2[] ballPositions;
        readonly string[] latestEvent;  // most recent event at or before each frame

        public float LastFrame => FrameCount - 1;
        public float Duration => LastFrame / FrameRate;

        public PlayTimeline(PlayData play)
        {
            Play = play;
            Players = play.players ?? new PlayerInfo[0];
            FrameCount = play.frames.Length;
            FrameRate = play.frameRate > 0 ? play.frameRate : 10;

            int playerCount = Players.Length;
            positions = new Vector2[playerCount * FrameCount];
            orientations = new float[playerCount * FrameCount];
            ballPositions = new Vector2[FrameCount];
            latestEvent = new string[FrameCount];

            var slot = new Dictionary<long, int>(playerCount);
            for (int p = 0; p < playerCount; p++)
                slot[Players[p].nflId] = p;

            var seen = new bool[playerCount * FrameCount];
            string lastEvent = null;
            SnapIndex = 0;

            for (int f = 0; f < FrameCount; f++)
            {
                FrameData frame = play.frames[f];
                if (!string.IsNullOrEmpty(frame.@event)) lastEvent = frame.@event;
                latestEvent[f] = lastEvent;
                if (frame.frameId == play.snapFrameId) SnapIndex = f;

                ballPositions[f] = frame.ball != null
                    ? new Vector2(frame.ball.x, frame.ball.y)
                    : (f > 0 ? ballPositions[f - 1] : Vector2.zero);

                if (frame.players == null) continue;
                foreach (PlayerState s in frame.players)
                {
                    if (!slot.TryGetValue(s.nflId, out int p)) continue;
                    int k = p * FrameCount + f;
                    positions[k] = new Vector2(s.x, s.y);
                    orientations[k] = s.o;
                    seen[k] = true;
                }
            }

            FillGaps(seen);
            Throw = ThrowArc.FindWindow(play.frames, ballPositions);
        }

        /// <summary>Interpolated field position (data x, y) of a player at a fractional frame.</summary>
        public Vector2 PlayerPosition(int player, float frame)
        {
            Split(frame, out int a, out int b, out float t);
            int row = player * FrameCount;
            return Vector2.Lerp(positions[row + a], positions[row + b], t);
        }

        /// <summary>Interpolated orientation in degrees (0 = +y, clockwise), shortest path.</summary>
        public float PlayerOrientation(int player, float frame)
        {
            Split(frame, out int a, out int b, out float t);
            int row = player * FrameCount;
            return Mathf.LerpAngle(orientations[row + a], orientations[row + b], t);
        }

        public Vector2 BallPosition(float frame)
        {
            Split(frame, out int a, out int b, out float t);
            return Vector2.Lerp(ballPositions[a], ballPositions[b], t);
        }

        /// <summary>Most recent event at or before the given frame, or null.</summary>
        public string EventAt(float frame) => latestEvent[Mathf.Clamp(Mathf.FloorToInt(frame), 0, FrameCount - 1)];

        /// <summary>The data's frameId for a fractional frame index.</summary>
        public int FrameIdAt(float frame) => Play.frames[Mathf.Clamp(Mathf.FloorToInt(frame), 0, FrameCount - 1)].frameId;

        void Split(float frame, out int a, out int b, out float t)
        {
            frame = Mathf.Clamp(frame, 0f, LastFrame);
            a = Mathf.FloorToInt(frame);
            b = Mathf.Min(a + 1, FrameCount - 1);
            t = frame - a;
        }

        // A player missing from some frames holds their nearest known sample.
        void FillGaps(bool[] seen)
        {
            for (int p = 0; p < Players.Length; p++)
            {
                int row = p * FrameCount;
                int firstSeen = -1;
                for (int f = 0; f < FrameCount; f++)
                {
                    if (seen[row + f]) { if (firstSeen < 0) firstSeen = f; continue; }
                    if (f > 0 && firstSeen >= 0)
                    {
                        positions[row + f] = positions[row + f - 1];
                        orientations[row + f] = orientations[row + f - 1];
                    }
                }
                for (int f = 0; f < firstSeen; f++)
                {
                    positions[row + f] = positions[row + firstSeen];
                    orientations[row + f] = orientations[row + firstSeen];
                }
            }
        }
    }
}
