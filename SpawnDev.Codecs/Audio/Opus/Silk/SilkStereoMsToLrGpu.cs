// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-callable SILK stereo mid/side -> left/right conversion. Mirror of
// SilkStereoMsToLr.Apply (libopus silk/stereo_MS_to_LR.c). Used in every
// stereo Opus SILK frame to recover L/R PCM from independently decoded
// mid + side streams.
//
// Two primitives, both per-sample parallel:
//   - ApplySideAt - reconstruct side channel using mid + side predictors.
//                   The CPU's iterative pred += delta is rewritten as the
//                   closed-form pred = predPrev + (n+1)*delta for n < interpLen
//                   so each output sample is independent.
//   - ApplyMixAt  - convert (mid, side) -> (left, right) via L=M+S, R=M-S.
//
// Caller dispatches ApplySideAt for [0, frameLength), then ApplyMixAt for
// [0, frameLength). Host handles state I/O (read prefix, persist trailing,
// pre-compute deltas).
//
// All silk macros (LSHIFT, ADD_LSHIFT32, SMLAWB, RSHIFT_ROUND, SAT16) inlined.

using ILGPU;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// GPU-callable SILK mid/side -&gt; left/right conversion. Mirror of
/// <see cref="SilkStereoMsToLr"/>.Apply.
/// </summary>
public static class SilkStereoMsToLrGpu
{
    /// <summary>
    /// Compute one side-channel output sample at index <paramref name="n"/>.
    /// Writes <c>x2[n + 1]</c> using the standard SILK side-reconstruction
    /// formula. For n &lt; interpLen, the predictors interpolate from
    /// predPrev to predFinal in (n+1)*delta steps (closed form).
    /// For n &gt;= interpLen, the predictors are predFinal.
    /// </summary>
    public static void ApplySideAt(
        ArrayView<short> x1, long x1Base,
        ArrayView<short> x2, long x2Base,
        int predPrev0Q13, int delta0Q13, int pred0FinalQ13,
        int predPrev1Q13, int delta1Q13, int pred1FinalQ13,
        int interpLen, int n)
    {
        int pred0Q13;
        int pred1Q13;
        if (n < interpLen)
        {
            pred0Q13 = predPrev0Q13 + (n + 1) * delta0Q13;
            pred1Q13 = predPrev1Q13 + (n + 1) * delta1Q13;
        }
        else
        {
            pred0Q13 = pred0FinalQ13;
            pred1Q13 = pred1FinalQ13;
        }

        // sum = LSHIFT(ADD_LSHIFT32((int)x1[n] + (int)x1[n+2], (int)x1[n+1], 1), 9)
        int xSum = (int)x1[x1Base + n] + (int)x1[x1Base + n + 2];
        int addLshift = xSum + ((int)x1[x1Base + n + 1] << 1);
        int sum = addLshift << 9;

        // sum = SMLAWB(LSHIFT(x2[n+1], 8), sum, pred0Q13)
        int x2Center = (int)x2[x2Base + n + 1];
        int baseAcc = x2Center << 8;
        sum = baseAcc + (int)((long)sum * (short)pred0Q13 >> 16);

        // sum = SMLAWB(sum, LSHIFT(x1[n+1], 11), pred1Q13)
        int x1ShiftedQ11 = (int)x1[x1Base + n + 1] << 11;
        sum = sum + (int)((long)x1ShiftedQ11 * (short)pred1Q13 >> 16);

        // x2[n+1] = SAT16(RSHIFT_ROUND(sum, 8))
        int rounded = (sum + (1 << 7)) >> 8;
        if (rounded > short.MaxValue) rounded = short.MaxValue;
        else if (rounded < short.MinValue) rounded = short.MinValue;
        x2[x2Base + n + 1] = (short)rounded;
    }

    /// <summary>
    /// Convert one (mid, side) sample pair at index <paramref name="n"/> to
    /// (left, right): writes <c>x1[n+1] = SAT16(M+S)</c>, <c>x2[n+1] = SAT16(M-S)</c>.
    /// </summary>
    public static void ApplyMixAt(
        ArrayView<short> x1, long x1Base,
        ArrayView<short> x2, long x2Base,
        int n)
    {
        int mid = x1[x1Base + n + 1];
        int side = x2[x2Base + n + 1];
        int sum = mid + side;
        int diff = mid - side;
        x1[x1Base + n + 1] = Sat16(sum);
        x2[x2Base + n + 1] = Sat16(diff);
    }

    /// <summary>silk_SAT16: saturate int to int16.</summary>
    private static short Sat16(int v)
    {
        if (v > short.MaxValue) return short.MaxValue;
        if (v < short.MinValue) return short.MinValue;
        return (short)v;
    }
}
