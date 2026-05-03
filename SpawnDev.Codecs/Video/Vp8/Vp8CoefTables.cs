// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP8 entropy-coding constant tables (libvpx kBands, kZigzag, kCat3..kCat6).
// Defined here in main library so the GPU integration classes don't depend
// on the CPU reference classes (which live in SpawnDev.Codecs.References
// per the 2026-05-03 architectural directive). The CPU reference encoders
// + decoders read these back from `Vp8CoefTables.X`.

namespace SpawnDev.Codecs.Video.Vp8;

/// <summary>
/// VP8 entropy-coding constant tables. These are spec values from RFC 6386
/// section 13 (token probability tables) shared by every VP8
/// encoder + decoder primitive (CPU + GPU).
/// </summary>
public static class Vp8CoefTables
{
    /// <summary>Scan-position-to-band mapping (libvpx kBands).</summary>
    public static readonly byte[] CoefBands = new byte[]
    {
        0, 1, 2, 3, 6, 4, 5, 6, 6,
        6, 6, 6, 6, 6, 6, 7,
        0, // sentinel - probs lookup at coeff 16 reads kBands[16] which maps to band 0 unused.
    };

    /// <summary>4x4 zigzag scan order (libvpx kZigzag).</summary>
    public static readonly byte[] ZigzagScan = new byte[]
    {
        0, 1,  4,  8,  5, 2,  3,  6,
        9, 12, 13, 10, 7, 11, 14, 15,
    };

    /// <summary>Cat3 extra-bit probabilities (libvpx kCat3 without trailing 0).</summary>
    public static readonly byte[] Cat3Probs = new byte[] { 173, 148, 140 };

    /// <summary>Cat4 extra-bit probabilities (libvpx kCat4 without trailing 0).</summary>
    public static readonly byte[] Cat4Probs = new byte[] { 176, 155, 140, 135 };

    /// <summary>Cat5 extra-bit probabilities (libvpx kCat5 without trailing 0).</summary>
    public static readonly byte[] Cat5Probs = new byte[] { 180, 157, 141, 134, 130 };

    /// <summary>Cat6 extra-bit probabilities (libvpx kCat6 without trailing 0).</summary>
    public static readonly byte[] Cat6Probs = new byte[]
    {
        254, 254, 243, 230, 196, 177,
        153, 140, 133, 130, 129,
    };
}
