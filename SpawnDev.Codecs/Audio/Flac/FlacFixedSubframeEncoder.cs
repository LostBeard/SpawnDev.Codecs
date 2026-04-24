// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// FIXED subframe encoder with automatic order selection and Rice parameter
// selection. Tries orders 1..4, computes forward-difference residuals, picks
// the (order, Rice-k) pair that minimizes the total encoded bit count, and
// emits the subframe. Returns the estimated bit count so the caller can
// compare against VERBATIM or CONSTANT and pick the cheapest overall subframe.

namespace SpawnDev.Codecs.Audio.Flac;

internal static class FlacFixedSubframeEncoder
{
    /// <summary>Max Rice parameter we attempt (4-bit Rice param field bounds it at 14).</summary>
    private const int MaxRiceParam = 14;

    /// <summary>
    /// Try all FIXED orders 1..4 on <paramref name="samples"/> and return the
    /// (order, Rice-k) pair with the smallest encoded-bit estimate.
    /// Returns <c>null</c> if no FIXED order beats the VERBATIM baseline
    /// <c>samples.Length * bps</c>.
    /// </summary>
    internal static FlacFixedChoice? PickBest(ReadOnlySpan<int> samples, int bps)
    {
        if (samples.Length < 5) return null; // Need at least (order + 1) samples to compute order-4 residuals.

        int verbatimBits = samples.Length * bps;
        FlacFixedChoice? best = null;
        int bestBits = verbatimBits;

        for (int order = 1; order <= 4; order++)
        {
            // Compute residuals (length = samples.Length - order).
            var residual = new int[samples.Length - order];
            ComputeResidual(samples, order, residual);

            // Find optimal Rice parameter in [0, MaxRiceParam].
            int bestK = 0;
            long bestPartitionBits = long.MaxValue;
            for (int k = 0; k <= MaxRiceParam; k++)
            {
                long bits = EstimateRicePartitionBits(residual, k);
                if (bits < bestPartitionBits)
                {
                    bestPartitionBits = bits;
                    bestK = k;
                }
            }

            // Total subframe bits (excluding 1-bit reserved + 6-bit type + 1-bit wasted flag =
            // constant 8 across all choices):
            //   warmup * bps + 2 (coding method) + 4 (partition order) + 4 (Rice param) + rice bits
            long subframeBits = 8 + (long)order * bps + 2 + 4 + 4 + bestPartitionBits;
            if (subframeBits < bestBits)
            {
                bestBits = (int)subframeBits;
                best = new FlacFixedChoice(order, bestK, residual, bestPartitionBits);
            }
        }
        return best;
    }

    /// <summary>
    /// Emit a FIXED subframe into <paramref name="w"/> using the chosen order and Rice parameter.
    /// </summary>
    internal static void Emit(FlacBitWriter w, ReadOnlySpan<int> samples, int bps, FlacFixedChoice choice)
    {
        // Subframe header: reserved 0, type 0b001000 | order, wasted flag 0.
        w.Write(0, 1);
        w.Write((uint)(0b001000 | choice.Order), 6);
        w.Write(0, 1);
        // Warm-up: first `order` samples at bps.
        for (int i = 0; i < choice.Order; i++)
            w.WriteSigned(samples[i], bps);
        // Residual coding method 0 (4-bit Rice), partition order 0.
        w.Write(0, 2);
        w.Write(0, 4);
        // Rice parameter.
        w.Write((uint)choice.RiceParam, 4);
        // Rice-coded residuals.
        foreach (int r in choice.Residual)
        {
            uint u = r >= 0 ? (uint)(r << 1) : (uint)((-r << 1) - 1);
            int q = (int)(u >> choice.RiceParam);
            uint rem = u & ((1u << choice.RiceParam) - 1);
            w.WriteUnary(q);
            if (choice.RiceParam > 0) w.Write(rem, choice.RiceParam);
        }
    }

    /// <summary>
    /// Compute the k-th forward difference of <paramref name="samples"/> into <paramref name="dest"/>.
    /// Matches libFLAC's fixed_compute_residual_ (residual of FIXED coder).
    /// </summary>
    private static void ComputeResidual(ReadOnlySpan<int> samples, int order, Span<int> dest)
    {
        // residual[n] = samples[n + order] - predictor_k(samples[n + order - 1 ... n])
        // Equivalent to k-th forward difference, computed directly via the predictor coefficients:
        //   order 1: r = s[n] - s[n-1]
        //   order 2: r = s[n] - 2*s[n-1] + s[n-2]
        //   order 3: r = s[n] - 3*s[n-1] + 3*s[n-2] - s[n-3]
        //   order 4: r = s[n] - 4*s[n-1] + 6*s[n-2] - 4*s[n-3] + s[n-4]
        switch (order)
        {
            case 1:
                for (int n = 1; n < samples.Length; n++) dest[n - 1] = samples[n] - samples[n - 1];
                break;
            case 2:
                for (int n = 2; n < samples.Length; n++) dest[n - 2] = samples[n] - 2 * samples[n - 1] + samples[n - 2];
                break;
            case 3:
                for (int n = 3; n < samples.Length; n++) dest[n - 3] = samples[n] - 3 * samples[n - 1] + 3 * samples[n - 2] - samples[n - 3];
                break;
            case 4:
                for (int n = 4; n < samples.Length; n++) dest[n - 4] = samples[n] - 4 * samples[n - 1] + 6 * samples[n - 2] - 4 * samples[n - 3] + samples[n - 4];
                break;
            default: throw new ArgumentOutOfRangeException(nameof(order));
        }
    }

    /// <summary>
    /// Estimate the total bits needed to Rice-code a partition at parameter <paramref name="k"/>.
    /// Formula: sum over residuals of (quotient + 1 + k). Uses the zigzag unsigned
    /// mapping to handle negative residuals.
    /// </summary>
    private static long EstimateRicePartitionBits(ReadOnlySpan<int> residual, int k)
    {
        long bits = (long)residual.Length * (1 + k);
        for (int i = 0; i < residual.Length; i++)
        {
            uint u = residual[i] >= 0 ? (uint)(residual[i] << 1) : (uint)((-residual[i] << 1) - 1);
            bits += u >> k;
        }
        return bits;
    }
}

/// <summary>
/// Result of FIXED order + Rice parameter selection: the chosen order, parameter,
/// and the residual values (pre-computed for reuse at emit time).
/// </summary>
internal sealed record FlacFixedChoice(int Order, int RiceParam, int[] Residual, long EstimatedRiceBits);
