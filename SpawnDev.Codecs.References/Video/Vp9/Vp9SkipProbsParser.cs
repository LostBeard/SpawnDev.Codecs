// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).

namespace SpawnDev.Codecs.Video.Vp9;

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
