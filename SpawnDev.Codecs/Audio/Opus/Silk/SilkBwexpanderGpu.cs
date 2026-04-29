// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// SILK chirp (bandwidth expansion) for AR filter coefficients,
// GPU-callable form. Bit-exact mirror of SilkBwexpander.Expand16 +
// .Expand32. Inlines silk_RSHIFT_ROUND, silk_MUL, silk_SMULWW,
// silk_SMULWB, silk_MLA as scalar math.
//
// Used on LPC and whitening filters in SILK prediction paths +
// shared with CELT LPC code (per upstream comment any fix needs to
// be mirrored).
//
// Sequential per-coefficient (chirpQ16 update depends on previous),
// so this runs as a single-thread kernel call. The math is per-
// coefficient cheap; throughput is bounded by AR filter length d
// (typically 16 for SILK, up to 32 for CELT).

using ILGPU;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// GPU-callable SILK bandwidth-expansion (chirp) helpers. Static
/// methods that mutate AR filter coefficients in place via
/// caller-supplied ArrayView buffers.
/// </summary>
public static class SilkBwexpanderGpu
{
    /// <summary>
    /// Chirp (bandwidth expand) a 16-bit AR filter in place.
    /// Reads + writes <paramref name="d"/> shorts from
    /// <paramref name="ar"/> starting at <paramref name="arBase"/>.
    /// Mirrors libopus <c>silk_bwexpander</c>.
    /// </summary>
    public static void Expand16(
        ArrayView<short> ar, long arBase, int d,
        int chirpQ16)
    {
        int chirpMinusOneQ16 = chirpQ16 - 65536;

        for (int i = 0; i < d - 1; i++)
        {
            int prod = chirpQ16 * ar[arBase + i];
            ar[arBase + i] = (short)RShiftRound(prod, 16);
            int prodChirp = chirpQ16 * chirpMinusOneQ16;
            chirpQ16 += RShiftRound(prodChirp, 16);
        }
        if (d > 0)
        {
            int prod = chirpQ16 * ar[arBase + (d - 1)];
            ar[arBase + (d - 1)] = (short)RShiftRound(prod, 16);
        }
    }

    /// <summary>
    /// Chirp (bandwidth expand) a 32-bit AR filter in place. Mirrors
    /// libopus <c>silk_bwexpander_32</c>.
    /// </summary>
    public static void Expand32(
        ArrayView<int> ar, long arBase, int d,
        int chirpQ16)
    {
        int chirpMinusOneQ16 = chirpQ16 - 65536;

        for (int i = 0; i < d - 1; i++)
        {
            ar[arBase + i] = SmulWW(chirpQ16, ar[arBase + i]);
            int prodChirp = chirpQ16 * chirpMinusOneQ16;
            chirpQ16 += RShiftRound(prodChirp, 16);
        }
        if (d > 0)
        {
            ar[arBase + (d - 1)] = SmulWW(chirpQ16, ar[arBase + (d - 1)]);
        }
    }

    /// <summary>
    /// libopus <c>silk_RSHIFT_ROUND</c>. Rounds half-away-from-zero
    /// for positive input.
    /// </summary>
    private static int RShiftRound(int a, int shift)
    {
        if (shift == 1) return (a >> 1) + (a & 1);
        return ((a >> (shift - 1)) + 1) >> 1;
    }

    /// <summary>
    /// libopus <c>silk_SMULWW</c> = silk_MLA(silk_SMULWB(a, b), a,
    /// silk_RSHIFT_ROUND(b, 16)) = (int)((long)a * (short)b >> 16) +
    /// a * RShiftRound(b, 16).
    /// </summary>
    private static int SmulWW(int a32, int b32)
    {
        int smulwb = (int)((long)a32 * (short)b32 >> 16);
        int rshift = RShiftRound(b32, 16);
        return smulwb + a32 * rshift;
    }
}
