// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).

namespace SpawnDev.Codecs.Audio.Flac;

/// <summary>
/// FLAC frame blocking strategy. Fixed-block-size streams have constant block
/// size across every frame (frame number encoded); variable-block-size streams
/// may change block size per frame (sample number encoded instead).
/// </summary>
public enum FlacBlockingStrategy
{
    /// <summary>All frames have the same block size; header carries a frame number.</summary>
    Fixed = 0,

    /// <summary>Frames may vary in block size; header carries a sample number.</summary>
    Variable = 1,
}
