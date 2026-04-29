// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-callable SILK LPC analysis filter. Mirror of
// SilkLpcAnalysisFilter.Apply (libopus silk/LPC_analysis_filter.c).
// Applies the MA prediction-error filter (inverse of LPC synthesis)
// to one input sample.
//
// Per-sample independent: out[n] depends on in[n], in[n-1], ..., in[n-d]
// but not on prior output samples. One thread per output sample maps
// cleanly across all backends. Caller pre-zeros output[0..d).

using ILGPU;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// GPU-callable SILK LPC analysis filter. Mirror of
/// <see cref="SilkLpcAnalysisFilter"/>.Apply.
/// </summary>
public static class SilkLpcAnalysisFilterGpu
{
    /// <summary>
    /// Compute one output sample at index <paramref name="ix"/>:
    /// <c>output[outBase + ix] = SAT16(RSHIFT_ROUND(LSHIFT(in[ix], 12) - sum_{k=0..d-1} bQ12[k] * in[ix - 1 - k], 12))</c>.
    /// Caller dispatches this for each ix in [d, len). Output samples
    /// at [0, d) are pre-zeroed by the caller.
    /// </summary>
    public static void ApplyAt(
        ArrayView<short> inSignal, long inBase,
        ArrayView<short> bQ12, long bBase,
        ArrayView<short> outSignal, long outBase,
        int d, int ix)
    {
        int baseIdx = ix - 1;

        // Accumulate prediction in Q12 with overflow-wrapping arithmetic.
        // SMULBB(a, b) = (short)a * (short)b
        // SMLABB_ovflw(c, a, b) = c + (short)a * (short)b (wrapping)
        int out32Q12 = (int)inSignal[inBase + baseIdx] * (int)bQ12[bBase + 0];
        out32Q12 = unchecked(out32Q12 + (int)inSignal[inBase + baseIdx - 1] * (int)bQ12[bBase + 1]);
        out32Q12 = unchecked(out32Q12 + (int)inSignal[inBase + baseIdx - 2] * (int)bQ12[bBase + 2]);
        out32Q12 = unchecked(out32Q12 + (int)inSignal[inBase + baseIdx - 3] * (int)bQ12[bBase + 3]);
        out32Q12 = unchecked(out32Q12 + (int)inSignal[inBase + baseIdx - 4] * (int)bQ12[bBase + 4]);
        out32Q12 = unchecked(out32Q12 + (int)inSignal[inBase + baseIdx - 5] * (int)bQ12[bBase + 5]);
        for (int j = 6; j < d; j += 2)
        {
            out32Q12 = unchecked(out32Q12 + (int)inSignal[inBase + baseIdx - j] * (int)bQ12[bBase + j]);
            out32Q12 = unchecked(out32Q12 + (int)inSignal[inBase + baseIdx - j - 1] * (int)bQ12[bBase + j + 1]);
        }

        // Subtract accumulated prediction from the delayed input in Q12.
        out32Q12 = unchecked(((int)inSignal[inBase + ix] << 12) - out32Q12);

        // RSHIFT_ROUND by 12 + saturate to int16.
        int rounded = (out32Q12 + (1 << 11)) >> 12;
        if (rounded > short.MaxValue) rounded = short.MaxValue;
        if (rounded < short.MinValue) rounded = short.MinValue;
        outSignal[outBase + ix] = (short)rounded;
    }
}
