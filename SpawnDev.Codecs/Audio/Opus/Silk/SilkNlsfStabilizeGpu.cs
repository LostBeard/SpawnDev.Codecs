// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-callable SILK NLSF stabilizer. Mirror of
// SilkNlsfStabilize.Stabilize (libopus silk/NLSF_stabilize.c). Runs
// the iterative min-distance-bump loop + fallback insertion sort
// pass to ensure NLSF values are spec-compliant (sorted ascending
// with minimum spacing).
//
// Single-thread dispatch (the algorithm is inherently sequential).

using ILGPU;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// GPU-callable SILK NLSF stabilizer. Mirror of
/// <see cref="SilkNlsfStabilize"/>.Stabilize.
/// </summary>
public static class SilkNlsfStabilizeGpu
{
    private const int MaxLoops = 20;

    /// <summary>
    /// Stabilize an NLSF vector (Q15) in place against the per-position
    /// minimum spacing in <paramref name="nDeltaMinQ15"/>. Mirrors
    /// <see cref="SilkNlsfStabilize"/>.Stabilize bit-for-bit.
    /// </summary>
    /// <param name="nlsfQ15">Input/output NLSF vector (length L).</param>
    /// <param name="nlsfBase">Base offset.</param>
    /// <param name="L">Vector length.</param>
    /// <param name="nDeltaMinQ15">Per-position minimum spacing (length L+1).</param>
    /// <param name="nDeltaBase">Base offset.</param>
    public static void Stabilize(
        ArrayView<short> nlsfQ15, long nlsfBase, int L,
        ArrayView<short> nDeltaMinQ15, long nDeltaBase)
    {
        int loops = 0;
        for (loops = 0; loops < MaxLoops; loops++)
        {
            int minDiffQ15 = nlsfQ15[nlsfBase + 0] - nDeltaMinQ15[nDeltaBase + 0];
            int I = 0;

            for (int i = 1; i <= L - 1; i++)
            {
                int diffQ15 = nlsfQ15[nlsfBase + i] -
                    (nlsfQ15[nlsfBase + i - 1] + nDeltaMinQ15[nDeltaBase + i]);
                if (diffQ15 < minDiffQ15)
                {
                    minDiffQ15 = diffQ15;
                    I = i;
                }
            }

            int upperDiffQ15 = (1 << 15) -
                (nlsfQ15[nlsfBase + L - 1] + nDeltaMinQ15[nDeltaBase + L]);
            if (upperDiffQ15 < minDiffQ15)
            {
                minDiffQ15 = upperDiffQ15;
                I = L;
            }

            if (minDiffQ15 >= 0) return;

            if (I == 0)
            {
                nlsfQ15[nlsfBase + 0] = nDeltaMinQ15[nDeltaBase + 0];
            }
            else if (I == L)
            {
                nlsfQ15[nlsfBase + L - 1] = (short)((1 << 15) - nDeltaMinQ15[nDeltaBase + L]);
            }
            else
            {
                int minCenterQ15 = 0;
                for (int k = 0; k < I; k++) minCenterQ15 += nDeltaMinQ15[nDeltaBase + k];
                minCenterQ15 += nDeltaMinQ15[nDeltaBase + I] >> 1;

                int maxCenterQ15 = 1 << 15;
                for (int k = L; k > I; k--) maxCenterQ15 -= nDeltaMinQ15[nDeltaBase + k];
                maxCenterQ15 -= nDeltaMinQ15[nDeltaBase + I] >> 1;

                int sum = nlsfQ15[nlsfBase + I - 1] + nlsfQ15[nlsfBase + I];
                int centerRound = (sum + 1) >> 1;
                int centerFreqQ15 = centerRound < minCenterQ15 ? minCenterQ15
                    : centerRound > maxCenterQ15 ? maxCenterQ15
                    : centerRound;
                nlsfQ15[nlsfBase + I - 1] = (short)(centerFreqQ15 - (nDeltaMinQ15[nDeltaBase + I] >> 1));
                nlsfQ15[nlsfBase + I] = (short)(nlsfQ15[nlsfBase + I - 1] + nDeltaMinQ15[nDeltaBase + I]);
            }
        }

        // Fallback: insertion sort + clamp pass.
        InsertionSortInt16(nlsfQ15, nlsfBase, L);

        // First NLSF should be no less than nDeltaMinQ15[0].
        if (nlsfQ15[nlsfBase + 0] < nDeltaMinQ15[nDeltaBase + 0])
            nlsfQ15[nlsfBase + 0] = nDeltaMinQ15[nDeltaBase + 0];

        // Forward pass.
        for (int i = 1; i < L; i++)
        {
            int sum = nlsfQ15[nlsfBase + i - 1] + nDeltaMinQ15[nDeltaBase + i];
            int saturated = sum > short.MaxValue ? short.MaxValue
                : sum < short.MinValue ? short.MinValue : sum;
            int v = nlsfQ15[nlsfBase + i];
            if (saturated > v) v = saturated;
            nlsfQ15[nlsfBase + i] = (short)v;
        }

        // Last NLSF clamp.
        int upper = (1 << 15) - nDeltaMinQ15[nDeltaBase + L];
        if (nlsfQ15[nlsfBase + L - 1] > upper)
            nlsfQ15[nlsfBase + L - 1] = (short)upper;

        // Backward pass.
        for (int i = L - 2; i >= 0; i--)
        {
            int target = nlsfQ15[nlsfBase + i + 1] - nDeltaMinQ15[nDeltaBase + i + 1];
            if (nlsfQ15[nlsfBase + i] > target)
                nlsfQ15[nlsfBase + i] = (short)target;
        }
    }

    /// <summary>
    /// Simple insertion sort on the int16 array slice [nlsfBase, nlsfBase + L).
    /// Mirrors silk_insertion_sort_increasing_all_values_int16 used by the
    /// CPU stabilizer fallback path.
    /// </summary>
    private static void InsertionSortInt16(ArrayView<short> arr, long arrBase, int L)
    {
        for (int i = 1; i < L; i++)
        {
            short cur = arr[arrBase + i];
            int j = i - 1;
            while (j >= 0 && arr[arrBase + j] > cur)
            {
                arr[arrBase + j + 1] = arr[arrBase + j];
                j--;
            }
            arr[arrBase + j + 1] = cur;
        }
    }
}
