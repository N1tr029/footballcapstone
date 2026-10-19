using Hindsight.Data;
using UnityEngine;

namespace Hindsight.Playback
{
    /// <summary>
    /// Tracking data has no ball height, so passes get a synthetic parabola between the
    /// release (pass_forward) and the first pass outcome event. Ball x/y still come from data.
    /// </summary>
    public static class ThrowArc
    {
        /// <summary>Peak height in yards per yard of throw distance.</summary>
        const float PeakPerYard = 0.15f;
        const float MinPeak = 1f;
        const float MaxPeak = 12f;

        static readonly string[] ReleaseEvents = { "pass_forward", "pass_shovel" };
        static readonly string[] ArrivalEvents =
        {
            "pass_arrived", "pass_outcome_caught", "pass_outcome_incomplete",
            "pass_outcome_interception", "pass_outcome_touchdown", "pass_tipped",
        };

        public readonly struct Window
        {
            public readonly int Start;   // frame index of release
            public readonly int End;     // frame index of arrival
            public readonly float Peak;  // apex height above the release height, yards

            public Window(int start, int end, float peak) { Start = start; End = end; Peak = peak; }
            public bool IsValid => End > Start;
        }

        public static Window FindWindow(FrameData[] frames, Vector2[] ballPositions)
        {
            int start = -1;
            for (int f = 0; f < frames.Length; f++)
            {
                string e = frames[f].@event;
                if (string.IsNullOrEmpty(e)) continue;
                if (start < 0)
                {
                    if (System.Array.IndexOf(ReleaseEvents, e) >= 0) start = f;
                }
                else if (System.Array.IndexOf(ArrivalEvents, e) >= 0)
                {
                    float distance = Vector2.Distance(ballPositions[start], ballPositions[f]);
                    float peak = Mathf.Clamp(distance * PeakPerYard, MinPeak, MaxPeak);
                    return new Window(start, f, peak);
                }
            }
            return default;
        }

        /// <summary>Extra height above the normal ball height at a fractional frame.</summary>
        public static float HeightAt(in Window window, float frame)
        {
            if (!window.IsValid || frame <= window.Start || frame >= window.End) return 0f;
            float t = (frame - window.Start) / (window.End - window.Start);
            return 4f * window.Peak * t * (1f - t);
        }
    }
}
