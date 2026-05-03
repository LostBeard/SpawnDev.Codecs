// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Structural port of libopus silk/NLSF_stabilize.c to clean C#.
//
// Upstream Copyright (c) 2006-2011 Skype Limited. BSD 3-Clause. See NOTICE.md.
//
// NLSF stabilizer: ensures Normalized Line Spectral Frequencies stay ordered and
// separated by at least a minimum distance. Used by the NLSF decoder to recover
// from unstable reconstructed vectors. Iterates up to MAX_LOOPS times trying to
// preserve Euclidean distance to the input, then falls back to an insertion sort
// + clamping pass if convergence fails.

using static SpawnDev.Codecs.Audio.Opus.Silk.SilkMacros;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// NLSF stabilization: enforces ordering and minimum spacing between NLSF values.
/// </summary>
internal static class SilkNlsfStabilize
{
    /// <summary>Maximum number of iterative correction passes before falling back to sort.</summary>
    private const int MAX_LOOPS = 20;

    /// <summary>
    /// Stabilize a single NLSF vector in-place. After return, <paramref name="nlsfQ15"/> is
    /// sorted ascending, each adjacent pair differs by at least <paramref name="nDeltaMinQ15"/>,
    /// and the first/last elements respect the min-distance boundary conditions.
    /// </summary>
    /// <param name="nlsfQ15">In/out: NLSF values in Q15. Length = L.</param>
    /// <param name="nDeltaMinQ15">
    /// Per-position minimum distance array, length L+1. Index 0 is the lower bound,
    /// indices 1..L-1 are the pairwise minimum deltas, index L is the upper bound.
    /// The last element (<c>nDeltaMinQ15[L]</c>) must be >= 1.
    /// </param>
    internal static void Stabilize(Span<short> nlsfQ15, ReadOnlySpan<short> nDeltaMinQ15)
    {
        int L = nlsfQ15.Length;
        if (nDeltaMinQ15.Length < L + 1)
            throw new ArgumentException($"nDeltaMinQ15 must have length at least {L + 1} (was {nDeltaMinQ15.Length}).", nameof(nDeltaMinQ15));
        if (nDeltaMinQ15[L] < 1)
            throw new ArgumentException("nDeltaMinQ15[L] must be >= 1 to keep output within int16 range.", nameof(nDeltaMinQ15));

        int loops;
        int I = 0;
        for (loops = 0; loops < MAX_LOOPS; loops++)
        {
            // Find the smallest distance in the current NLSF vector.
            int minDiffQ15 = nlsfQ15[0] - nDeltaMinQ15[0];
            I = 0;

            for (int i = 1; i <= L - 1; i++)
            {
                int diffQ15 = nlsfQ15[i] - (nlsfQ15[i - 1] + nDeltaMinQ15[i]);
                if (diffQ15 < minDiffQ15)
                {
                    minDiffQ15 = diffQ15;
                    I = i;
                }
            }

            // Upper boundary distance.
            int upperDiffQ15 = (1 << 15) - (nlsfQ15[L - 1] + nDeltaMinQ15[L]);
            if (upperDiffQ15 < minDiffQ15)
            {
                minDiffQ15 = upperDiffQ15;
                I = L;
            }

            // Converged if all distances are non-negative.
            if (minDiffQ15 >= 0) return;

            if (I == 0)
            {
                nlsfQ15[0] = nDeltaMinQ15[0];
            }
            else if (I == L)
            {
                nlsfQ15[L - 1] = (short)((1 << 15) - nDeltaMinQ15[L]);
            }
            else
            {
                // Find lower extreme for current center frequency.
                int minCenterQ15 = 0;
                for (int k = 0; k < I; k++) minCenterQ15 += nDeltaMinQ15[k];
                minCenterQ15 += silk_RSHIFT(nDeltaMinQ15[I], 1);

                // Find upper extreme for current center frequency.
                int maxCenterQ15 = 1 << 15;
                for (int k = L; k > I; k--) maxCenterQ15 -= nDeltaMinQ15[k];
                maxCenterQ15 -= silk_RSHIFT(nDeltaMinQ15[I], 1);

                // Move apart, sorted by value, preserving center frequency.
                int sum = nlsfQ15[I - 1] + nlsfQ15[I];
                short centerFreqQ15 = (short)silk_LIMIT_32(
                    silk_RSHIFT_ROUND(sum, 1),
                    minCenterQ15,
                    maxCenterQ15);
                nlsfQ15[I - 1] = (short)(centerFreqQ15 - silk_RSHIFT(nDeltaMinQ15[I], 1));
                nlsfQ15[I] = (short)(nlsfQ15[I - 1] + nDeltaMinQ15[I]);
            }
        }

        // Fallback: didn't converge within MAX_LOOPS. Use insertion sort + clamp pass.
        if (loops == MAX_LOOPS)
        {
            silk_insertion_sort_increasing_all_values_int16(nlsfQ15);

            // First NLSF should be no less than nDeltaMinQ15[0].
            nlsfQ15[0] = (short)silk_max_int(nlsfQ15[0], nDeltaMinQ15[0]);

            // Keep min distance between adjacent NLSFs (forward pass).
            for (int i = 1; i < L; i++)
            {
                nlsfQ15[i] = (short)silk_max_int(nlsfQ15[i], silk_ADD_SAT16(nlsfQ15[i - 1], nDeltaMinQ15[i]));
            }

            // Last NLSF should be no higher than 1 - nDeltaMinQ15[L].
            nlsfQ15[L - 1] = (short)silk_min_int(nlsfQ15[L - 1], (1 << 15) - nDeltaMinQ15[L]);

            // Keep min distance (backward pass).
            for (int i = L - 2; i >= 0; i--)
            {
                nlsfQ15[i] = (short)silk_min_int(nlsfQ15[i], nlsfQ15[i + 1] - nDeltaMinQ15[i + 1]);
            }
        }
    }
}
