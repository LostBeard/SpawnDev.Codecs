// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 inter-frame Y intra mode probability runtime updates. Walks
// the [BLOCK_SIZE_GROUPS=4][INTRA_MODES-1=9] = 36 probability bytes
// through vp9_diff_update_prob (slice 210).
//
// Layout matches Vp9IntraModeProbs.DefaultIfYProbs: flat byte[36] in
// row-major order. Caller passes the working state buffer (clone of
// the defaults at frame start) and this parser mutates it in place
// per the compressed header.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>Parser for the read_y_mode_probs section of the compressed header.</summary>
public static class Vp9YModeProbsParser
{
    /// <summary>
    /// Apply diff_update_prob to every entry of the inter-frame Y
    /// intra mode probability table. Mirror of the libvpx loop in
    /// <c>read_compressed_header</c>.
    /// </summary>
    /// <param name="probs">
    /// Flat byte[36] = [4 block size groups][9 binary tree leaves].
    /// Modified in place.
    /// </param>
    /// <param name="reader">Compressed-header arithmetic reader.</param>
    public static void Read(byte[] probs, Vp9BoolDecoder reader)
    {
        ArgumentNullException.ThrowIfNull(probs);
        ArgumentNullException.ThrowIfNull(reader);
        int expectedSize = Vp9IntraModeProbs.BlockSizeGroups * Vp9IntraModeProbs.ProbsPerMode;
        if (probs.Length < expectedSize)
            throw new ArgumentException(
                $"probs must hold at least {expectedSize} bytes (got {probs.Length})",
                nameof(probs));

        for (int j = 0; j < Vp9IntraModeProbs.BlockSizeGroups; j++)
            for (int i = 0; i < Vp9IntraModeProbs.ProbsPerMode; i++)
            {
                int idx = j * Vp9IntraModeProbs.ProbsPerMode + i;
                probs[idx] = Vp9DiffUpdateProb.Read(reader, probs[idx]);
            }
    }
}
