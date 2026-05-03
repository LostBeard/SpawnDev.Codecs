// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 reference frame info parser - the per-reference fields in
// the uncompressed header for non-keyframes. Mirror of the
// REFS_PER_FRAME loop in libvpx vp9/decoder/vp9_decodeframe.c
// read_uncompressed_header().
//
// Bitstream layout per reference (LAST, GOLDEN, ALTREF):
//   ref_frame_idx     f(REF_FRAMES_LOG2 = 3) -> 0..7 pool index
//   ref_frame_sign_bias f(1)
//
// Total: REFS_PER_FRAME * 4 = 12 bits.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 inter reference frame slot (libvpx LAST_FRAME / GOLDEN_FRAME / ALTREF_FRAME).</summary>
public enum Vp9ReferenceSlot : byte
{
    /// <summary>Most recent prior frame.</summary>
    Last = 0,
    /// <summary>Long-term reference (golden frame).</summary>
    Golden = 1,
    /// <summary>Alternate reference (typically future-frame in encoding order).</summary>
    AltRef = 2,
}

/// <summary>
/// Per-reference fields parsed from the inter-frame uncompressed
/// header. Bit-exact against libvpx <c>read_uncompressed_header</c>
/// per-ref loop.
/// </summary>
public sealed record Vp9ReferenceFrameInfo
{
    /// <summary>libvpx <c>REFS_PER_FRAME</c>.</summary>
    public const int RefsPerFrame = 3;

    /// <summary>libvpx <c>REF_FRAMES_LOG2</c> (8-entry pool).</summary>
    public const int RefFramesLog2 = 3;

    /// <summary>libvpx <c>REF_FRAMES</c> (size of the reference pool).</summary>
    public const int RefFramesPoolSize = 1 << RefFramesLog2;

    /// <summary>
    /// Pool index 0..7 for each of the 3 reference slots
    /// (Last / Golden / AltRef).
    /// </summary>
    public required int[] RefFrameIdx { get; init; } = Array.Empty<int>();

    /// <summary>
    /// Sign bias bit per reference slot. true means MVs to this
    /// reference get sign-biased.
    /// </summary>
    public required bool[] RefFrameSignBias { get; init; } = Array.Empty<bool>();
}

/// <summary>Parser for VP9 reference frame info in the uncompressed header.</summary>
public static class Vp9ReferenceFrameInfoParser
{
    /// <summary>
    /// Parse the per-reference fields. Reads
    /// <c>REFS_PER_FRAME * (REF_FRAMES_LOG2 + 1) = 12</c> bits.
    /// </summary>
    internal static Vp9ReferenceFrameInfo Parse(ref Vp9BitReader reader)
    {
        var refIdx = new int[Vp9ReferenceFrameInfo.RefsPerFrame];
        var refBias = new bool[Vp9ReferenceFrameInfo.RefsPerFrame];

        for (int i = 0; i < Vp9ReferenceFrameInfo.RefsPerFrame; i++)
        {
            refIdx[i] = (int)reader.ReadBits(Vp9ReferenceFrameInfo.RefFramesLog2);
            refBias[i] = reader.ReadFlag();
        }

        return new Vp9ReferenceFrameInfo
        {
            RefFrameIdx = refIdx,
            RefFrameSignBias = refBias,
        };
    }

    /// <summary>Convenience overload for unit tests.</summary>
    public static Vp9ReferenceFrameInfo Parse(ReadOnlySpan<byte> data)
    {
        var r = new Vp9BitReader(data);
        return Parse(ref r);
    }
}
