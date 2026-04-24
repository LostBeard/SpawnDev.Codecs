// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Vorbis I utility functions from specification Section 9.2.

namespace SpawnDev.Codecs.Audio.Vorbis;

internal static class VorbisMath
{
    /// <summary>
    /// ilog(x) per Vorbis I Section 9.2.1: the position of the highest set bit,
    /// counted starting from 1. ilog(0) = 0, ilog(1) = 1, ilog(2) = 2,
    /// ilog(3) = 2, ilog(4) = 3, etc.
    /// </summary>
    internal static int Ilog(int x)
    {
        int r = 0;
        while (x > 0) { r++; x >>= 1; }
        return r;
    }

    /// <summary>
    /// float32_unpack per Vorbis I Section 9.2.2: Vorbis's custom 32-bit float
    /// format (NOT IEEE 754). Packs as [sign:1][exponent:10][mantissa:21] with
    /// the representation
    ///   value = mantissa * 2^(exponent - 788), signed.
    /// </summary>
    internal static double Float32Unpack(uint x)
    {
        long mantissa = (long)(x & 0x1FFFFF);
        bool negative = (x & 0x80000000u) != 0;
        int exponent = (int)((x & 0x7FE00000u) >> 21);
        if (negative) mantissa = -mantissa;
        return mantissa * Math.Pow(2.0, exponent - 788);
    }

    /// <summary>
    /// lookup1_values per Vorbis I Section 9.2.4: the greatest integer n such
    /// that n^dimensions &lt;= entries. Used to size the multiplicand table for
    /// codebook lookup type 1.
    /// </summary>
    internal static int Lookup1Values(int entries, int dimensions)
    {
        if (dimensions == 0) return 0;
        int n = 0;
        while (true)
        {
            long product = 1;
            for (int i = 0; i < dimensions; i++)
            {
                product *= (n + 1);
                if (product > entries) return n;
            }
            n++;
        }
    }
}
