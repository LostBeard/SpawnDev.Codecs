// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 per-reference-frame MV candidates store. The MV reference
// candidate generator populates a separate list per reference
// frame slot (Intra / Last / Golden / AltRef) and the per-block
// inter-mode decoder picks from the list matching the block's
// chosen reference frame.
//
// libvpx reference: vp9/common/vp9_mvref_common.c
// vp9_find_mv_refs_idx populates int_mv
// mv_ref_list[MAX_REF_FRAMES][MAX_MV_REF_CANDIDATES] (4 x 2 of MVs).

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 per-reference-frame MV candidates store.</summary>
public sealed class Vp9MvRefCandidatesByRef
{
    private readonly Vp9MvCandidatesList[] _byRef;

    /// <summary>Construct with empty per-ref lists.</summary>
    public Vp9MvRefCandidatesByRef()
    {
        _byRef = new Vp9MvCandidatesList[Vp9MvReferenceFrames.MaxRefFrames];
        for (int i = 0; i < _byRef.Length; i++)
            _byRef[i] = new Vp9MvCandidatesList();
    }

    /// <summary>
    /// Get the candidates list for <paramref name="frame"/>. The
    /// returned list is mutable; call <see cref="Vp9MvCandidatesList.TryAdd"/>
    /// during scanning.
    /// </summary>
    public Vp9MvCandidatesList ForRef(Vp9MvReferenceFrame frame)
    {
        int idx = (int)frame;
        if ((uint)idx >= (uint)Vp9MvReferenceFrames.MaxRefFrames)
            throw new ArgumentOutOfRangeException(nameof(frame), frame,
                "frame index out of range.");
        return _byRef[idx];
    }

    /// <summary>Reset all per-ref lists to empty.</summary>
    public void Clear()
    {
        for (int i = 0; i < _byRef.Length; i++)
            _byRef[i].Clear();
    }
}
