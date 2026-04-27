// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 decoded frame buffer (planar 8-bit YUV). Holds the output of the
// keyframe decode walker for downstream consumers.

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>Planar 8-bit YUV frame produced by the AV1 keyframe walker.</summary>
public sealed record Av1FrameBuffer
{
    /// <summary>Luma plane bytes, row-major, length = LumaWidth * LumaHeight.</summary>
    public required byte[] Y { get; init; }
    /// <summary>U chroma plane.</summary>
    public required byte[] U { get; init; }
    /// <summary>V chroma plane.</summary>
    public required byte[] V { get; init; }
    /// <summary>Luma width in pixels.</summary>
    public required int LumaWidth { get; init; }
    /// <summary>Luma height in pixels.</summary>
    public required int LumaHeight { get; init; }
    /// <summary>Chroma width.</summary>
    public required int ChromaWidth { get; init; }
    /// <summary>Chroma height.</summary>
    public required int ChromaHeight { get; init; }
}
