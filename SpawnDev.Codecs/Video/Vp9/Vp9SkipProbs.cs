// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 skip probability storage + parser. The compressed header
// updates 3 per-context probabilities for the skip flag.
//
// Layout: byte[SKIP_CONTEXTS=3]. Mirror of libvpx
// fc->skip_probs[SKIP_CONTEXTS].

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 skip-flag probabilities (libvpx fc->skip_probs).</summary>
public sealed class Vp9SkipProbs
{
    /// <summary>libvpx <c>SKIP_CONTEXTS</c>.</summary>
    public const int SkipContexts = 3;

    /// <summary>Per-context skip-flag probability.</summary>
    public byte[] Probs { get; } = new byte[SkipContexts];
}

/// <summary>Parser for the read_skip_probs section of the compressed header.</summary>
public static class Vp9SkipProbsParser
{
    /// <summary>
    /// Apply diff_update_prob to each of the 3 skip-flag contexts.
    /// Mirror of the libvpx <c>read_skip_probs</c> loop.
    /// </summary>
    public static void Read(Vp9SkipProbs probs, Vp9BoolDecoder reader)
    {
        ArgumentNullException.ThrowIfNull(probs);
        ArgumentNullException.ThrowIfNull(reader);
        for (int k = 0; k < Vp9SkipProbs.SkipContexts; k++)
            probs.Probs[k] = Vp9DiffUpdateProb.Read(reader, probs.Probs[k]);
    }
}
