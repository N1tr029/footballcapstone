using Hindsight.Cameras;
using Hindsight.Data;
using Hindsight.Field;
using Hindsight.Playback;
using Hindsight.UI;
using UnityEngine;

namespace Hindsight
{
    /// <summary>
    /// Composition root. Put this on one GameObject in the scene (Hindsight > Setup Scene
    /// does it for you); it builds the field, cameras, HUD, and wires playback to the loader.
    /// </summary>
    [RequireComponent(typeof(PlayLoader), typeof(PlaybackController), typeof(FieldBuilder))]
    [RequireComponent(typeof(CameraRig), typeof(PlaybackInput), typeof(PlaybackHud))]
    public class HindsightApp : MonoBehaviour
    {
        [Tooltip("URP Unlit material; field colors are instanced from it.")]
        [SerializeField] Material unlitMaterial;
        [SerializeField] PlayerView playerPrefab;
        [SerializeField] BallView ballPrefab;

        PlayLoader loader;
        PlaybackController playback;
        FieldBuilder field;
        CameraRig cameras;

        void Awake()
        {
            if (unlitMaterial == null || playerPrefab == null || ballPrefab == null)
            {
                Debug.LogError("[Hindsight] Missing material or prefabs. Run Hindsight > Setup Scene in the editor.");
                enabled = false;
                return;
            }

            loader = GetComponent<PlayLoader>();
            playback = GetComponent<PlaybackController>();
            field = GetComponent<FieldBuilder>();
            cameras = GetComponent<CameraRig>();

            field.Build(unlitMaterial);
            cameras.Build();
            playback.Configure(playerPrefab, ballPrefab);
            GetComponent<PlaybackInput>().Init(playback, loader, cameras);
            GetComponent<PlaybackHud>().Init(playback, loader);

            loader.PlayLoaded += OnPlayLoaded;
        }

        async void Start()
        {
            if (!enabled) return;
            await loader.InitializeAsync();
        }

        void OnDestroy()
        {
            if (loader != null) loader.PlayLoaded -= OnPlayLoaded;
        }

        void OnPlayLoaded(PlayData play)
        {
            playback.SetPlay(play);
            field.SetMarkers(play.lineOfScrimmage, play.firstDownLine);
            cameras.SetTarget(playback.Ball.transform);
        }
    }
}
