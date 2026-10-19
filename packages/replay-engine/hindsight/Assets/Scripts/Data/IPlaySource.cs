using UnityEngine;

namespace Hindsight.Data
{
    /// <summary>
    /// Where plays come from. Today: JSON files in StreamingAssets. Later: the backend.
    /// Both serve the same schema, so only the source changes.
    /// </summary>
    public interface IPlaySource
    {
        Awaitable<PlayIndex> LoadIndexAsync();
        Awaitable<PlayData> LoadPlayAsync(string playId);
    }
}
