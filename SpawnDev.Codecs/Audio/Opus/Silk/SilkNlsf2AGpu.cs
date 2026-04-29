// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-callable SILK NLSF -> LPC coefficient conversion. Mirror of
// SilkNlsf2A.Compute (libopus silk/NLSF2A.c). Converts normalized
// line spectral frequencies (Q15) into Q12 monic whitening filter
// coefficients with iterative bandwidth expansion to guarantee
// stability.
//
// Composes existing GPU primitives:
//   - lookup into the LSF cosine table (caller-provided ArrayView)
//   - SilkLpcFitGpu.FitAt for the int32 -> Q12 quantization
//   - SilkLpcInvPredGainGpu.Compute for stability check
//   - SilkBwexpanderGpu.Expand32 for the iterative chirp expansion
//
// Sequential per-stream because each stage depends on prior output:
// FindPoly is sequential, the stability loop is sequential. Single-
// thread per stream on the GPU; multi-channel decode parallelizes
// across threads.

using ILGPU;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// GPU-callable SILK NLSF -&gt; LPC coefficient conversion. Mirror of
/// <see cref="SilkNlsf2A"/>.Compute.
/// </summary>
public static class SilkNlsf2AGpu
{
    private const int QA = 16;
    private const int MAX_LPC_STABILIZE_ITERATIONS = 16;
    private const int MAX_LPC_ORDER = 16;

    /// <summary>
    /// Convert NLSFs in Q15 to Q12 LPC coefficients with stability
    /// guarantee. Bit-exact vs the CPU SilkNlsf2A.Compute. The scratch
    /// buffer must be at least <c>3 * MAX_LPC_ORDER + MAX_LPC_ORDER/2 + 1</c>
    /// = 57 ints (cosLsfQA[16] + p[9] + q[9] + a32QA1[16] + invPredGainScratch[16]).
    /// </summary>
    /// <param name="aQ12">Output Q12 LPC coefs (length d).</param>
    /// <param name="aOutBase">Base offset.</param>
    /// <param name="nlsf">Input Q15 NLSFs (length d).</param>
    /// <param name="nlsfBase">Base offset.</param>
    /// <param name="lsfCosTabQ12">SilkLsfCosTab.Q12 table (length 129).</param>
    /// <param name="lsfCosBase">Base offset.</param>
    /// <param name="scratch">Per-call scratch (length &gt;= 65 ints). Contents replaced.</param>
    /// <param name="scratchBase">Base offset.</param>
    /// <param name="d">Filter order; must be 10 or 16.</param>
    public static void ComputeAt(
        ArrayView<short> aQ12, long aOutBase,
        ArrayView<short> nlsf, long nlsfBase,
        ArrayView<short> lsfCosTabQ12, long lsfCosBase,
        ArrayView<int> scratch, long scratchBase,
        int d)
    {
        long cosLsfQABase = scratchBase;             // length 16
        long pBase = scratchBase + MAX_LPC_ORDER;    // length 9
        long qBase = pBase + MAX_LPC_ORDER / 2 + 1;  // length 9
        long a32Q1Base = qBase + MAX_LPC_ORDER / 2 + 1;  // length 16
        long invPredScratchBase = a32Q1Base + MAX_LPC_ORDER; // length 16

        // Step 1: NLSF -> 2*cos(LSF) lookup with reordering.
        for (int k = 0; k < d; k++)
        {
            int nlsfVal = nlsf[nlsfBase + k];
            int fInt = nlsfVal >> 8;
            int fFrac = nlsfVal - (fInt << 8);

            int cosVal = lsfCosTabQ12[lsfCosBase + fInt];
            int delta = lsfCosTabQ12[lsfCosBase + fInt + 1] - cosVal;

            int interpolated = (cosVal << 8) + delta * fFrac;
            int rounded = RShiftRound(interpolated, 20 - QA);

            int orderingIdx = OrderingAt(d, k);
            scratch[cosLsfQABase + orderingIdx] = rounded;
        }

        int dd = d >> 1;

        // Step 2: FindPoly for even (P) and odd (Q) polynomials.
        FindPoly(scratch, pBase, scratch, cosLsfQABase, 2, dd);
        FindPoly(scratch, qBase, scratch, cosLsfQABase + 1, 2, dd);

        // Step 3: Fold P + Q into a32Q(QA+1).
        for (int k = 0; k < dd; k++)
        {
            int pTmp = scratch[pBase + k + 1] + scratch[pBase + k];
            int qTmp = scratch[qBase + k + 1] - scratch[qBase + k];
            scratch[a32Q1Base + k] = -qTmp - pTmp;
            scratch[a32Q1Base + d - k - 1] = qTmp - pTmp;
        }

        // Step 4: LPC fit (Q(QA+1) -> Q12).
        SilkLpcFitGpu.FitAt(aQ12, aOutBase, scratch, a32Q1Base, 12, QA + 1, d);

        // Step 5: stability loop.
        for (int i = 0; i < MAX_LPC_STABILIZE_ITERATIONS; i++)
        {
            int invGain = SilkLpcInvPredGainGpu.Compute(aQ12, aOutBase, d, scratch, invPredScratchBase);
            if (invGain != 0) break;

            // chirp = 65536 - (2 << i)
            int chirp = 65536 - (2 << i);
            SilkBwexpanderGpu.Expand32(scratch, a32Q1Base, d, chirp);

            for (int k = 0; k < d; k++)
            {
                aQ12[aOutBase + k] = (short)RShiftRound(scratch[a32Q1Base + k], QA + 1 - 12);
            }
        }
    }

    /// <summary>silk_NLSF2A_find_poly. Sequential polynomial construction.</summary>
    private static void FindPoly(
        ArrayView<int> outPoly, long outBase,
        ArrayView<int> cLSF, long cLSFBase, int cLSFStride, int dd)
    {
        outPoly[outBase + 0] = 1 << QA;
        outPoly[outBase + 1] = -cLSF[cLSFBase + 0];

        for (int k = 1; k < dd; k++)
        {
            int ftmp = cLSF[cLSFBase + cLSFStride * k];

            // outPoly[k+1] = (outPoly[k-1] << 1) - RSHIFT_ROUND64(SMULL(ftmp, outPoly[k]), QA)
            long mulOut = (long)ftmp * outPoly[outBase + k];
            int reduced = (int)RShiftRound64(mulOut, QA);
            outPoly[outBase + k + 1] = (outPoly[outBase + k - 1] << 1) - reduced;

            for (int n = k; n > 1; n--)
            {
                long mulInner = (long)ftmp * outPoly[outBase + n - 1];
                int reducedInner = (int)RShiftRound64(mulInner, QA);
                outPoly[outBase + n] = outPoly[outBase + n] + outPoly[outBase + n - 2] - reducedInner;
            }

            outPoly[outBase + 1] = outPoly[outBase + 1] - ftmp;
        }
    }

    /// <summary>Index into the order-d ordering permutation.</summary>
    private static int OrderingAt(int d, int k)
    {
        if (d == 16)
        {
            // { 0, 15, 8, 7, 4, 11, 12, 3, 2, 13, 10, 5, 6, 9, 14, 1 }
            switch (k)
            {
                case 0: return 0;
                case 1: return 15;
                case 2: return 8;
                case 3: return 7;
                case 4: return 4;
                case 5: return 11;
                case 6: return 12;
                case 7: return 3;
                case 8: return 2;
                case 9: return 13;
                case 10: return 10;
                case 11: return 5;
                case 12: return 6;
                case 13: return 9;
                case 14: return 14;
                default: return 1; // case 15
            }
        }
        // Order 10: { 0, 9, 6, 3, 4, 5, 8, 1, 2, 7 }
        switch (k)
        {
            case 0: return 0;
            case 1: return 9;
            case 2: return 6;
            case 3: return 3;
            case 4: return 4;
            case 5: return 5;
            case 6: return 8;
            case 7: return 1;
            case 8: return 2;
            default: return 7; // case 9
        }
    }

    /// <summary>silk_RSHIFT_ROUND for shift &gt;= 1.</summary>
    private static int RShiftRound(int a, int shift) =>
        ((a >> (shift - 1)) + 1) >> 1;

    /// <summary>silk_RSHIFT_ROUND64.</summary>
    private static long RShiftRound64(long a, int shift)
    {
        if (shift <= 0) return a;
        return ((a >> (shift - 1)) + 1) >> 1;
    }
}
