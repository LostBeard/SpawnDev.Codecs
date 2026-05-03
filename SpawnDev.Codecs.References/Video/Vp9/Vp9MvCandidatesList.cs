// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 MV reference candidates list. Fixed-capacity (2) deduplicating
// container that the MV reference candidate generator (libvpx
// vp9_find_mv_refs_idx) populates as it scans neighboring blocks.
//
// libvpx convention:
//   #define MAX_MV_REF_CANDIDATES 2
//   int_mv mv_ref_list[MAX_MV_REF_CANDIDATES];
//
// Candidates are added in scan order (above-row first, then left-
// column, then far-row, etc.) with deduplication against earlier
// entries. Once the list reaches capacity (2 entries) further
// candidates are silently dropped.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 MV reference candidates list (fixed capacity = 2).</summary>
public sealed class Vp9MvCandidatesList
{
    /// <summary>libvpx <c>MAX_MV_REF_CANDIDATES</c>.</summary>
    public const int MaxCandidates = 2;

    private readonly Vp9Mv[] _candidates = new Vp9Mv[MaxCandidates];
    private int _count;

    /// <summary>Number of candidates currently in the list.</summary>
    public int Count => _count;

    /// <summary>True when the list has reached its capacity.</summary>
    public bool IsFull => _count >= MaxCandidates;

    /// <summary>Read-only view over the active candidates.</summary>
    public ReadOnlySpan<Vp9Mv> AsSpan() =>
        new ReadOnlySpan<Vp9Mv>(_candidates, 0, _count);

    /// <summary>
    /// Try to add <paramref name="mv"/> to the list. Returns false if
    /// the MV duplicates an existing candidate or the list is already
    /// full.
    /// </summary>
    public bool TryAdd(Vp9Mv mv)
    {
        for (int i = 0; i < _count; i++)
            if (_candidates[i] == mv) return false;
        if (_count >= MaxCandidates) return false;
        _candidates[_count++] = mv;
        return true;
    }

    /// <summary>
    /// Indexer over the active candidates; throws on out-of-range.
    /// </summary>
    public Vp9Mv this[int index]
    {
        get
        {
            if ((uint)index >= (uint)_count)
                throw new ArgumentOutOfRangeException(nameof(index), index,
                    $"index must be in [0, {_count}).");
            return _candidates[index];
        }
    }

    /// <summary>Reset the list to empty.</summary>
    public void Clear()
    {
        Array.Clear(_candidates, 0, MaxCandidates);
        _count = 0;
    }
}
