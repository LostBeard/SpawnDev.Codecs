// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 partition probability runtime updates. Walks the
// PARTITION_CONTEXTS=16 contexts x (PARTITION_TYPES-1=3) tree
// leaves and applies vp9_diff_update_prob (slice 210) per entry.
//
// Layout matches Vp9PartitionProbs.KfPartitionProbs / DefaultPartitionProbs:
// flat byte[48] in row-major order. Caller passes the working
// state buffer (typically a clone of the defaults) and this
// parser mutates it in place per the compressed header.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>Parser for the read_partition_probs section of the compressed header.</summary>
public static class Vp9PartitionProbsParser
{
    /// <summary>
    /// Apply diff_update_prob to every entry of a partition prob
    /// table. Mirror of the libvpx loop in
    /// <c>read_compressed_header</c>.
    /// </summary>
    /// <param name="probs">
    /// Flat byte[48] partition probability state ([16 contexts][3 leaves]).
    /// Modified in place.
    /// </param>
    /// <param name="reader">Compressed-header arithmetic reader.</param>
    public static void Read(byte[] probs, Vp9BoolDecoder reader)
    {
        ArgumentNullException.ThrowIfNull(probs);
        ArgumentNullException.ThrowIfNull(reader);
        int expectedSize = Vp9PartitionProbs.PartitionContexts * Vp9PartitionProbs.ProbsPerContext;
        if (probs.Length < expectedSize)
            throw new ArgumentException(
                $"probs must hold at least {expectedSize} bytes (got {probs.Length})",
                nameof(probs));

        for (int j = 0; j < Vp9PartitionProbs.PartitionContexts; j++)
            for (int i = 0; i < Vp9PartitionProbs.ProbsPerContext; i++)
            {
                int idx = j * Vp9PartitionProbs.ProbsPerContext + i;
                probs[idx] = Vp9DiffUpdateProb.Read(reader, probs[idx]);
            }
    }
}
