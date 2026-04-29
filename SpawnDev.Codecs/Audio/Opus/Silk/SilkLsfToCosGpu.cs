// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-callable LSF -> 2*cos(LSF) lookup with linear interpolation.
// Mirror of the per-k inner loop in SilkNlsf2A (libopus
// silk/NLSF2A.c). Extracts one normalized LSF value (Q15) into a
// Q-scaled 2*cos(LSF) value via the 129-entry cosine lookup table.

using ILGPU;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// GPU-callable SILK LSF-to-cos converter. Pairs with
/// <see cref="SilkLsfCosTab"/> uploaded as a 129-entry short array.
/// </summary>
public static class SilkLsfToCosGpu
{
    /// <summary>
    /// Convert one normalized LSF value (Q15, range [0, 1)) to
    /// 2*cos(pi*LSF) in Q(QA) precision, using piecewise linear
    /// interpolation on the 129-entry cosine lookup.
    ///
    /// Mirrors the per-k loop in SilkNlsf2A.Convert (lines 85-99):
    ///   fInt = nlsfQ15 &gt;&gt; 8           // index into 129-entry table
    ///   fFrac = nlsfQ15 - (fInt &lt;&lt; 8)
    ///   cosVal = table[fInt]            // Q12
    ///   delta = table[fInt+1] - cosVal
    ///   result = RoundShiftRight((cosVal &lt;&lt; 8) + delta * fFrac, 20 - QA)
    /// </summary>
    /// <param name="nlsfQ15">Normalized LSF in Q15.</param>
    /// <param name="cosTabQ12">129-entry cosine table (SilkLsfCosTab.Q12).</param>
    /// <param name="cosTabBase">Base offset.</param>
    /// <param name="qa">Output Q-scale (typically QA = 16 in libopus).</param>
    /// <returns>2*cos(pi * nlsfQ15 / 2^15) in Q(qa).</returns>
    public static int Convert(int nlsfQ15,
        ArrayView<short> cosTabQ12, long cosTabBase, int qa)
    {
        int fInt = nlsfQ15 >> 8;
        int fFrac = nlsfQ15 - (fInt << 8);

        int cosVal = cosTabQ12[cosTabBase + fInt];           // Q12
        int delta = cosTabQ12[cosTabBase + fInt + 1] - cosVal;

        // (cosVal << 8) + delta * fFrac is in Q20.
        // Right-shift by (20 - qa) with rounding to land in Q(qa).
        int shift = 20 - qa;
        long sum = ((long)cosVal << 8) + (long)delta * fFrac;
        if (shift > 0)
            return (int)((sum + (1L << (shift - 1))) >> shift);
        if (shift < 0)
            return (int)(sum << -shift);
        return (int)sum;
    }
}
