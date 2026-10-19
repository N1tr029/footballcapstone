using System;
using System.Collections.Generic;
using UnityEngine;

namespace Hindsight.Data
{
    /// <summary>
    /// Owns the play list and loads plays by position in it. Raises PlayLoaded when a
    /// play is ready. Plays currently come from StreamingAssets; set source to Remote and
    /// point remoteBaseUrl at the backend once it serves the same JSON.
    /// </summary>
    public class PlayLoader : MonoBehaviour
    {
        public enum SourceKind { StreamingAssets, Remote }

        [SerializeField] SourceKind source = SourceKind.StreamingAssets;
        [Tooltip("Used when source is Remote. Expects {url}/index.json and {url}/{id}.json.")]
        [SerializeField] string remoteBaseUrl = "http://localhost:8000/plays";

        public event Action<PlayData> PlayLoaded;
        public event Action<string> LoadFailed;

        public PlayIndex Index => index;
        public IReadOnlyList<PlayIndexEntry> Plays => index?.plays ?? Array.Empty<PlayIndexEntry>();
        public int CurrentIndex { get; private set; } = -1;
        public bool IsLoading { get; private set; }

        IPlaySource playSource;
        PlayIndex index;
        readonly Dictionary<string, PlayData> cache = new();
        int requestVersion;

        public IPlaySource Source => playSource ??= source == SourceKind.Remote
            ? new WebPlaySource(remoteBaseUrl)
            : WebPlaySource.StreamingAssets();

        /// <summary>Fetch the play list, then load the first play.</summary>
        public async Awaitable InitializeAsync()
        {
            try
            {
                index = await Source.LoadIndexAsync();
            }
            catch (Exception e)
            {
                Fail($"Could not load play index: {e.Message}");
                return;
            }

            if (Plays.Count == 0)
            {
                Fail("Play index is empty.");
                return;
            }
            await LoadAtAsync(0);
        }

        public void Next() => _ = LoadAtAsync(Wrap(CurrentIndex + 1));
        public void Previous() => _ = LoadAtAsync(Wrap(CurrentIndex - 1));

        public async Awaitable LoadAtAsync(int i)
        {
            if (Plays.Count == 0) return;
            i = Wrap(i);
            string id = Plays[i].id;
            int version = ++requestVersion;
            CurrentIndex = i;

            if (!cache.TryGetValue(id, out var play))
            {
                IsLoading = true;
                try
                {
                    play = await Source.LoadPlayAsync(id);
                }
                catch (Exception e)
                {
                    if (version == requestVersion) IsLoading = false;
                    Fail($"Could not load play {id}: {e.Message}");
                    return;
                }
                cache[id] = play;
            }

            // A newer request started while this one was in flight; drop this result.
            if (version != requestVersion) return;
            IsLoading = false;

            if (play.frames == null || play.frames.Length == 0)
            {
                Fail($"Play {id} has no frames.");
                return;
            }
            PlayLoaded?.Invoke(play);
        }

        int Wrap(int i) => Plays.Count == 0 ? 0 : ((i % Plays.Count) + Plays.Count) % Plays.Count;

        void Fail(string message)
        {
            Debug.LogError($"[PlayLoader] {message}");
            LoadFailed?.Invoke(message);
        }
    }
}
