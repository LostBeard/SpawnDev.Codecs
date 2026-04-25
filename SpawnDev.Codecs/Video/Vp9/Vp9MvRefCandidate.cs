// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 motion vector reference candidate. Used by the MV reference
// candidate generator (libvpx vp9_find_mv_refs_idx) to carry an
// (mv, refFrame) pair from a neighboring block as a candidate
// reference for the current block's NewMV / NearestMV / NearMV
// modes.
//
// libvpx reference: vp9/common/vp9_mvref_common.c. The full
// candidate generator runs above + left + above-right + above-left
// + temporal scans, populates up to 8 candidates, and dedupes them.
// This slice ships only the per-candidate data type; the generator
// itself is a downstream slice.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// VP9 motion vector reference candidate (libvpx <c>int_mv</c>
/// + <c>MV_REFERENCE_FRAME</c> tuple).
/// </summary>
public readonly record struct Vp9MvRefCandidate(Vp9Mv Mv, Vp9MvReferenceFrame ReferenceFrame)
{
    /// <summary>The "no candidate" sentinel - zero MV against intra ref.</summary>
    public static readonly Vp9MvRefCandidate None =
        new Vp9MvRefCandidate(Vp9Mv.Zero, Vp9MvReferenceFrame.Intra);

    /// <summary>
    /// True for intra-coded candidates (which are not usable as a
    /// reference MV - the candidate generator filters these out).
    /// </summary>
    public bool IsIntra => ReferenceFrame == Vp9MvReferenceFrame.Intra;

    /// <summary>True when the candidate carries a nonzero MV.</summary>
    public bool HasMotion => !Mv.IsZero;
}
