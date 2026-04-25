// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 switchable interpolation filter probability storage + parser.
// Per-context tree of 3 filters (EIGHTTAP / EIGHTTAP_SMOOTH /
// EIGHTTAP_SHARP); 2 binary tree leaves per context, 4 contexts.
//
// Layout: byte[SWITCHABLE_FILTER_CONTEXTS=4][SWITCHABLE_FILTERS-1=2].
// Mirror of libvpx fc->switchable_interp_prob.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 per-context switchable interpolation filter probabilities.</summary>
public sealed class Vp9SwitchableInterpProbs
{
    /// <summary>libvpx <c>SWITCHABLE_FILTER_CONTEXTS</c>.</summary>
    public const int SwitchableFilterContexts = 4;

    /// <summary>libvpx <c>SWITCHABLE_FILTERS</c>.</summary>
    public const int SwitchableFilters = 3;

    /// <summary>8 prob bytes: [4 contexts][2 binary tree leaves].</summary>
    public byte[,] Probs { get; } = new byte[SwitchableFilterContexts, SwitchableFilters - 1];
}

/// <summary>Parser for the read_switchable_interp_probs section of the compressed header.</summary>
public static class Vp9SwitchableInterpProbsParser
{
    /// <summary>
    /// Apply diff_update_prob to every entry of the switchable interp
    /// filter prob table. Mirror of libvpx
    /// <c>read_switchable_interp_probs</c>.
    /// </summary>
    public static void Read(Vp9SwitchableInterpProbs probs, Vp9BoolDecoder reader)
    {
        ArgumentNullException.ThrowIfNull(probs);
        ArgumentNullException.ThrowIfNull(reader);
        for (int j = 0; j < Vp9SwitchableInterpProbs.SwitchableFilterContexts; j++)
            for (int i = 0; i < Vp9SwitchableInterpProbs.SwitchableFilters - 1; i++)
                probs.Probs[j, i] = Vp9DiffUpdateProb.Read(reader, probs.Probs[j, i]);
    }
}
