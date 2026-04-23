// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Structural port of libopus silk/bwexpander.c and silk/bwexpander_32.c to clean C#.
//
// Upstream Copyright (c) 2006-2011 Skype Limited. BSD 3-Clause. See NOTICE.md.
//
// Chirp (bandwidth expansion) for AR filter coefficients. Multiplies the i-th
// coefficient by chirp^(i+1), effectively shrinking the filter's poles toward
// the origin. Used on LPC and whitening filters in SILK prediction paths.

using static SpawnDev.Codecs.Audio.Opus.Silk.SilkMacros;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// SILK chirp-expand helpers for AR filter coefficients. Two variants: 16-bit
/// coefficients (<see cref="Expand16"/>) and 32-bit coefficients (<see cref="Expand32"/>).
/// Both mirror libopus semantics bit-exactly.
/// </summary>
internal static class SilkBwexpander
{
    /// <summary>
    /// Chirp (bandwidth expand) a 16-bit AR filter in-place.
    /// <para>
    /// NB (from libopus): do NOT rewrite this using <see cref="silk_SMULWB"/>; the bias
    /// inherent in silk_SMULWB leads to unstable filters. The explicit
    /// <c>silk_RSHIFT_ROUND(silk_MUL(...), 16)</c> form is intentional and required.
    /// </para>
    /// </summary>
    /// <param name="ar">In/out: AR filter coefficients (without leading 1). Length <c>d</c>.</param>
    /// <param name="chirpQ16">Chirp factor in Q16, typically in <c>[0, 1)</c> (<c>65536</c> == 1.0).</param>
    internal static void Expand16(Span<short> ar, int chirpQ16)
    {
        int d = ar.Length;
        int chirpMinusOneQ16 = chirpQ16 - 65536;

        for (int i = 0; i < d - 1; i++)
        {
            ar[i] = (short)silk_RSHIFT_ROUND(silk_MUL(chirpQ16, ar[i]), 16);
            chirpQ16 += silk_RSHIFT_ROUND(silk_MUL(chirpQ16, chirpMinusOneQ16), 16);
        }
        if (d > 0)
        {
            ar[d - 1] = (short)silk_RSHIFT_ROUND(silk_MUL(chirpQ16, ar[d - 1]), 16);
        }
    }

    /// <summary>
    /// Chirp (bandwidth expand) a 32-bit AR filter in-place.
    /// Uses <see cref="silk_SMULWW"/> for the per-coefficient multiplication. This path
    /// is shared with the CELT LPC code per upstream comment - any fix needs to be
    /// mirrored in <c>_celt_lpc()</c> when we port CELT.
    /// </summary>
    /// <param name="ar">In/out: AR filter coefficients (without leading 1). Length <c>d</c>.</param>
    /// <param name="chirpQ16">Chirp factor in Q16.</param>
    internal static void Expand32(Span<int> ar, int chirpQ16)
    {
        int d = ar.Length;
        int chirpMinusOneQ16 = chirpQ16 - 65536;

        for (int i = 0; i < d - 1; i++)
        {
            ar[i] = silk_SMULWW(chirpQ16, ar[i]);
            chirpQ16 += silk_RSHIFT_ROUND(silk_MUL(chirpQ16, chirpMinusOneQ16), 16);
        }
        if (d > 0)
        {
            ar[d - 1] = silk_SMULWW(chirpQ16, ar[d - 1]);
        }
    }
}
