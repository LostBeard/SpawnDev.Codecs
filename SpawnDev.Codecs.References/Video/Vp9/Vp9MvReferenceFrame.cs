// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 MB reference frame enum. Mirror of libvpx
// vp9/common/vp9_blockd.h MV_REFERENCE_FRAME (excluding the NONE
// = -1 sentinel; callers use null in places where libvpx uses
// NONE).
//
// This is the "block-level" reference frame value: 0 = the block
// is intra-coded, 1..3 = the block is inter-coded against the
// matching slot in <see cref="Vp9ReferenceSlot"/>. Used by:
//
//   - MB_MODE_INFO.ref_frame[2] (single-ref or compound-ref pair)
//   - vp9_loopfilter.c ref_deltas indexing (MAX_REF_LF_DELTAS = 4)
//   - inter-prediction reference selection
//
// Distinct from <see cref="Vp9ReferenceSlot"/>, which is the
// 0..2 inter-only slot used by the per-frame
// <see cref="Vp9ReferenceFrameInfo"/> bitstream parser.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// VP9 MB-level reference frame value (libvpx <c>MV_REFERENCE_FRAME</c>
/// without the <c>NONE = -1</c> sentinel).
/// </summary>
public enum Vp9MvReferenceFrame : byte
{
    /// <summary>
    /// Block is intra-coded. No reference frame; the block is
    /// reconstructed entirely from intra prediction + residual.
    /// </summary>
    Intra = 0,

    /// <summary>Block is inter-coded against the LAST reference slot.</summary>
    Last = 1,

    /// <summary>Block is inter-coded against the GOLDEN reference slot.</summary>
    Golden = 2,

    /// <summary>Block is inter-coded against the ALTREF reference slot.</summary>
    AltRef = 3,
}

/// <summary>VP9 MV reference frame helpers.</summary>
public static class Vp9MvReferenceFrames
{
    /// <summary>libvpx <c>MAX_REF_FRAMES</c>.</summary>
    public const int MaxRefFrames = 4;

    /// <summary>True for inter-coded blocks (any non-Intra ref).</summary>
    public static bool IsInter(Vp9MvReferenceFrame frame) => frame != Vp9MvReferenceFrame.Intra;

    /// <summary>
    /// Convert a non-Intra MV reference frame to its
    /// <see cref="Vp9ReferenceSlot"/>. Throws when called on
    /// <see cref="Vp9MvReferenceFrame.Intra"/>.
    /// </summary>
    public static Vp9ReferenceSlot ToReferenceSlot(Vp9MvReferenceFrame frame)
    {
        if (frame == Vp9MvReferenceFrame.Intra)
            throw new ArgumentOutOfRangeException(nameof(frame), frame,
                "Intra has no Vp9ReferenceSlot - check IsInter first.");
        // Last=1 -> 0, Golden=2 -> 1, AltRef=3 -> 2
        return (Vp9ReferenceSlot)((int)frame - 1);
    }
}
