// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 inter mode probability storage + parser. Per-context tree
// of 4 inter modes (NEARESTMV / NEARMV / ZEROMV / NEWMV); 3 binary
// tree leaves per context, 7 contexts.
//
// Layout: byte[INTER_MODE_CONTEXTS=7][INTER_MODES-1=3]. Mirror of
// libvpx fc->inter_mode_probs.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 per-context inter mode probabilities.</summary>
public sealed class Vp9InterModeProbsTable
{
    /// <summary>libvpx <c>INTER_MODE_CONTEXTS</c>.</summary>
    public const int InterModeContexts = 7;

    /// <summary>libvpx <c>INTER_MODES</c>.</summary>
    public const int InterModes = 4;

    /// <summary>21 prob bytes: [7 contexts][3 binary tree leaves].</summary>
    public byte[,] Probs { get; } = new byte[InterModeContexts, InterModes - 1];
}

/// <summary>Parser for the read_inter_mode_probs section of the compressed header.</summary>
public static class Vp9InterModeProbsParser
{
    /// <summary>
    /// Apply diff_update_prob to every entry of the
    /// <c>inter_mode_probs</c> table. Mirror of libvpx
    /// <c>read_inter_mode_probs</c>.
    /// </summary>
    public static void Read(Vp9InterModeProbsTable probs, Vp9BoolDecoder reader)
    {
        ArgumentNullException.ThrowIfNull(probs);
        ArgumentNullException.ThrowIfNull(reader);
        for (int i = 0; i < Vp9InterModeProbsTable.InterModeContexts; i++)
            for (int j = 0; j < Vp9InterModeProbsTable.InterModes - 1; j++)
                probs.Probs[i, j] = Vp9DiffUpdateProb.Read(reader, probs.Probs[i, j]);
    }
}
