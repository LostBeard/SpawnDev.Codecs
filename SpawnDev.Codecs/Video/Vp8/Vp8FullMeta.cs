// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Shared constants of the full-syntax VP8 keyframe decoder (Vp8FullKeyframeDecoderGpu): the layout of the small "meta"
// int block that carries the parsed frame header and the error state between kernels, the macroblock record layout,
// and the error codes. A kernel and the host cannot disagree about a slot because both use these names.

namespace SpawnDev.Codecs.Video.Vp8;

/// <summary>Why a VP8 keyframe could not be decoded. The value is also what the kernels store in the meta block.</summary>
public enum Vp8DecodeError
{
    /// <summary>No error.</summary>
    None = 0,
    /// <summary>Fewer than the 10 bytes of the uncompressed keyframe header.</summary>
    TooShort = 1,
    /// <summary>The frame tag says inter frame; only keyframes decode here.</summary>
    NotKeyFrame = 2,
    /// <summary>The 9D 01 2A start code is missing.</summary>
    BadStartCode = 3,
    /// <summary>The picture width or height is zero.</summary>
    ZeroSize = 4,
    /// <summary>The frame tag version is above 3 (libwebp rejects it too).</summary>
    UnsupportedVersion = 5,
    /// <summary>The first partition size in the frame tag exceeds the frame.</summary>
    FirstPartitionExceedsFrame = 6,
    /// <summary>The first partition (frame header + macroblock modes) ended before the data the macroblocks need.</summary>
    FirstPartitionTruncated = 7,
    /// <summary>The token partition size table does not fit in the data after the first partition.</summary>
    PartitionTableTruncated = 8,
    /// <summary>A token partition declares a size that runs past the end of the frame.</summary>
    TokenPartitionExceedsFrame = 9,
    /// <summary>A token partition ended before the coefficients of its macroblock rows.</summary>
    TokenPartitionTruncated = 10,
}

/// <summary>A VP8 keyframe is malformed or truncated; <see cref="Error"/> says exactly how.</summary>
public sealed class Vp8DecodeException : Exception
{
    /// <summary>The specific failure.</summary>
    public Vp8DecodeError Error { get; }

    /// <summary>Create the exception.</summary>
    public Vp8DecodeException(Vp8DecodeError error, string message) : base(message) { Error = error; }

    internal static Vp8DecodeException From(Vp8DecodeError error) => new(error, error switch
    {
        Vp8DecodeError.TooShort => "VP8 keyframe must be at least 10 bytes (uncompressed header).",
        Vp8DecodeError.NotKeyFrame => "Not a VP8 key frame.",
        Vp8DecodeError.BadStartCode => "VP8 key-frame start code (9D 01 2A) missing.",
        Vp8DecodeError.ZeroSize => "VP8 keyframe signals a zero-sized picture.",
        Vp8DecodeError.UnsupportedVersion => "VP8 frame version above 3.",
        Vp8DecodeError.FirstPartitionExceedsFrame => "VP8 first partition size exceeds the frame length.",
        Vp8DecodeError.FirstPartitionTruncated => "VP8 first partition (header + macroblock modes) is truncated.",
        Vp8DecodeError.PartitionTableTruncated => "VP8 token partition size table is truncated.",
        Vp8DecodeError.TokenPartitionExceedsFrame => "VP8 token partition size exceeds the frame length.",
        Vp8DecodeError.TokenPartitionTruncated => "VP8 token partition is truncated (coefficient data ends early).",
        _ => "VP8 decode error " + (int)error,
    });
}

/// <summary>Slots of the meta int block of the full keyframe decoder.</summary>
internal static class Vp8FullMeta
{
    public const int Size = 128;

    /// <summary>Wavefront schedule: macroblock (row, col) runs on diagonal col + WaveRowWeight * row. 2 is the smallest weight that keeps the above-right neighbour (row-1, col+1) on an earlier diagonal.</summary>
    public const int WaveRowWeight = 2;

    public const int Error = 0;            // Vp8DecodeError, written by the kernels (first error wins)
    public const int Width = 1;
    public const int Height = 2;
    public const int MbCols = 3;
    public const int MbRows = 4;
    public const int FrameLength = 5;      // valid bytes in the encoded buffer
    public const int FirstPartLen = 6;
    public const int SegEnabled = 7;
    public const int SegUpdateMap = 8;
    public const int SegAbs = 9;
    public const int SegQ = 10;            // 4
    public const int SegLf = 14;           // 4
    public const int SegTreeProb = 18;     // 3
    public const int FilterType = 21;      // 0 normal, 1 simple
    public const int FilterLevel = 22;
    public const int Sharpness = 23;
    public const int LfDeltaEnabled = 24;
    public const int RefLfDelta = 25;      // 4
    public const int ModeLfDelta = 29;     // 4
    public const int Log2Parts = 33;
    public const int NumParts = 34;
    public const int MbNoSkipCoeff = 35;
    public const int ProbSkipFalse = 36;
    public const int Dequant = 40;         // 4 segments x 6: y1dc y1ac y2dc y2ac uvdc uvac  (40..63)

    // The bool decoder states (partition 0, then up to 8 token partitions) live in their own int buffer so that a
    // kernel that handles one macroblock row can pick its decoder up and put it back (long loops stay chunked).
    public const int BdStateInts = 8;      // Pos End Value Range BitCount Shifts Len Err
    public const int BdCount = 9;
}

/// <summary>Per-macroblock record (bytes) written by the mode kernel and read by the later kernels.</summary>
internal static class Vp8MbRecord
{
    public const int Size = 24;
    public const int YMode = 0;            // 0 DC, 1 V, 2 H, 3 TM, 4 B_PRED
    public const int UvMode = 1;           // 0 DC, 1 V, 2 H, 3 TM
    public const int Segment = 2;          // 0..3
    public const int Skip = 3;             // mb_skip_coeff as signalled
    public const int BModes = 4;           // 16 sub-block modes (implied ones for non-B_PRED macroblocks)
    public const int HasCoef = 20;         // any block of the macroblock has a non-zero end-of-block position
}
