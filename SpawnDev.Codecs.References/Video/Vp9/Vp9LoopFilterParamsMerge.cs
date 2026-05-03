// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 loop filter delta merger. The Vp9LoopFilterParams parser
// produces nullable RefDeltas[] / ModeDeltas[] where null means
// "no update from prior frame". The decoder maintains persistent
// effective delta state across frames; this helper merges the
// parsed null-or-update values with the persistent state to
// produce the effective int[] deltas consumed by
// Vp9LoopFilterLookup.ResolveBlockLevel.
//
// libvpx reference: vp9/decoder/vp9_decodeframe.c read_loopfilter
// (the "if (update_ref_delta) lf->ref_deltas[i] = signed_8;" path).

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 loop filter delta merger.</summary>
public static class Vp9LoopFilterParamsMerge
{
    /// <summary>
    /// Merge parsed RefDeltas (nullable, length 0 or 4) with the
    /// persistent decoder state (length 4) to produce the effective
    /// 4-entry RefDeltas array. A null parsed entry inherits the
    /// persistent value; a non-null entry overrides it.
    /// </summary>
    public static int[] MergeRefDeltas(int?[]? parsed, int[] persistent)
    {
        ArgumentNullException.ThrowIfNull(persistent);
        if (persistent.Length != Vp9LoopFilterParams.MaxRefDeltas)
            throw new ArgumentException(
                $"persistent must hold exactly {Vp9LoopFilterParams.MaxRefDeltas} entries.",
                nameof(persistent));
        return Merge(parsed, persistent, Vp9LoopFilterParams.MaxRefDeltas);
    }

    /// <summary>
    /// Merge parsed ModeDeltas (nullable, length 0 or 2) with the
    /// persistent decoder state (length 2). Same null-inherits
    /// semantics as <see cref="MergeRefDeltas"/>.
    /// </summary>
    public static int[] MergeModeDeltas(int?[]? parsed, int[] persistent)
    {
        ArgumentNullException.ThrowIfNull(persistent);
        if (persistent.Length != Vp9LoopFilterParams.MaxModeDeltas)
            throw new ArgumentException(
                $"persistent must hold exactly {Vp9LoopFilterParams.MaxModeDeltas} entries.",
                nameof(persistent));
        return Merge(parsed, persistent, Vp9LoopFilterParams.MaxModeDeltas);
    }

    private static int[] Merge(int?[]? parsed, int[] persistent, int expectedLength)
    {
        var result = new int[expectedLength];
        for (int i = 0; i < expectedLength; i++)
        {
            if (parsed is not null && i < parsed.Length && parsed[i].HasValue)
                result[i] = parsed[i]!.Value;
            else
                result[i] = persistent[i];
        }
        return result;
    }
}
