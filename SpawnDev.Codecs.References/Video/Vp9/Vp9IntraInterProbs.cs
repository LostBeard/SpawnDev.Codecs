// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 intra-vs-inter probability storage + parser. 4 per-context
// probabilities for the binary "is this block intra-coded?" flag.
//
// Layout: byte[INTRA_INTER_CONTEXTS=4]. Mirror of libvpx
// fc->intra_inter_prob.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 intra-vs-inter flag probabilities.</summary>
public sealed class Vp9IntraInterProbs
{
    /// <summary>libvpx <c>INTRA_INTER_CONTEXTS</c>.</summary>
    public const int IntraInterContexts = 4;

    /// <summary>Per-context intra-vs-inter probability.</summary>
    public byte[] Probs { get; } = new byte[IntraInterContexts];
}

/// <summary>Parser for the read_intra_inter_probs section of the compressed header.</summary>
public static class Vp9IntraInterProbsParser
{
    /// <summary>
    /// Apply diff_update_prob to each of the 4 intra-vs-inter
    /// contexts. Mirror of the libvpx loop in
    /// <c>read_compressed_header</c>.
    /// </summary>
    public static void Read(Vp9IntraInterProbs probs, Vp9BoolDecoder reader)
    {
        ArgumentNullException.ThrowIfNull(probs);
        ArgumentNullException.ThrowIfNull(reader);
        for (int i = 0; i < Vp9IntraInterProbs.IntraInterContexts; i++)
            probs.Probs[i] = Vp9DiffUpdateProb.Read(reader, probs.Probs[i]);
    }
}
