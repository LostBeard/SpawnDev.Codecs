// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt). See NOTICE.md for upstream attributions.
//
// libopus 1.5.2 fixed-point arithmetic (celt/fixed_generic.h, celt/arch.h, celt/mathops.h) with its C semantics, for code
// ported from the 1.5.2 C sources rather than from Concentus: opus_val16 is 16 bits, so every macro truncates where its C
// casts do (MULT16_16 and MAC16_16 their operands, ADD16 / SHL16 / EXTRACT16 their result, SUB16 its operands only), and
// the MULT32_32 / MULT16_32 forms are the exact 64-bit ones a 64-bit build uses (OPUS_FAST_INT64). Plain integer arithmetic:
// identical on every ILGPU backend.

using System.Runtime.CompilerServices;

namespace SpawnDev.Codecs.Audio.Opus.Celt;

/// <summary>libopus 1.5.2 fixed-point macros with their C truncations (GPU-callable).</summary>
internal static class CeltOps
{
    public const int DB_SHIFT = 10, SIG_SHIFT = 12, BITRES = 3, EPSILON = 1, SIG_SAT = 300000000, Q15ONE = 32767;

    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int S16(int x) => (short)x;                           // EXTRACT16
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int ADD16(int a, int b) => (short)((short)a + (short)b);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int SUB16(int a, int b) => (short)a - (short)b;
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int SHL16(int a, int shift) => (short)(ushort)((ushort)a << shift);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int SHR16(int a, int shift) => a >> shift;
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int SHL32(int a, int shift) => (int)((uint)a << shift);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int PSHR32(int a, int shift) => (a + ((1 << shift) >> 1)) >> shift;
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int VSHR32(int a, int shift) => shift > 0 ? a >> shift : SHL32(a, -shift);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int ROUND16(int x, int a) => (short)PSHR32(x, a);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int SATURATE(int x, int a) => x > a ? a : x < -a ? -a : x;
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int SATURATE16(int x) => (short)(x > 32767 ? 32767 : x < -32768 ? -32768 : x);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int SROUND16(int x, int a) => (short)SATURATE(PSHR32(x, a), 32767);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int MULT16_16(int a, int b) => (short)a * (short)b;
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int MULT16_16SU(int a, int b) => (short)a * (int)(ushort)b;
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int MAC16_16(int c, int a, int b) => c + (short)a * (short)b;
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int MULT16_16_Q13(int a, int b) => MULT16_16(a, b) >> 13;
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int MULT16_16_Q14(int a, int b) => MULT16_16(a, b) >> 14;
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int MULT16_16_Q15(int a, int b) => MULT16_16(a, b) >> 15;
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int MULT16_16_P14(int a, int b) => (8192 + MULT16_16(a, b)) >> 14;
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int MULT16_16_P15(int a, int b) => (16384 + MULT16_16(a, b)) >> 15;
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int MULT16_32_Q15(int a, int b) => (int)(((long)(short)a * b) >> 15);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int MULT16_32_Q16(int a, int b) => (int)(((long)(short)a * b) >> 16);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int MULT16_32_P16(int a, int b) => (int)(((long)(short)a * b + 32768) >> 16);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int MULT32_32_Q16(int a, int b) => (int)(((long)a * b) >> 16);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int MULT32_32_Q31(int a, int b) => (int)(((long)a * b) >> 31);
    /// <summary>libopus MAC16_32_Q15: no 64-bit form; the split one truncates (b &gt;&gt; 15) to 16 bits.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int MAC16_32_Q15(int c, int a, int b) => c + (MULT16_16(a, b >> 15) + (MULT16_16(a, b & 0x7fff) >> 15));
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int DIV32_16(int a, int b) => (short)(a / (short)b);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int HALF16(int x) => x >> 1;
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int HALF32(int x) => x >> 1;
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int ABS32(int x) => x < 0 ? -x : x;
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int IMIN(int a, int b) => a < b ? a : b;
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int IMAX(int a, int b) => a > b ? a : b;

    /// <summary>libopus celt_ilog2 (EC_ILOG(x) - 1, x &gt; 0).</summary>
    public static int celt_ilog2(int x) => CeltFixed.EC_ILOG((uint)x) - 1;

    /// <summary>libopus celt_zlog2: 0 for x &lt;= 0.</summary>
    public static int celt_zlog2(int x) => x <= 0 ? 0 : celt_ilog2(x);

    /// <summary>libopus 1.5.2 celt_log2 (mathops.h): Q14 input, Q10 (DB_SHIFT) output.</summary>
    public static int celt_log2(int x)
    {
        if (x == 0) return -32767;
        int i = celt_ilog2(x);
        int n = S16(VSHR32(x, i - 15) - 32768 - 16384);
        // C[] = {-6801 + (1 << (13 - DB_SHIFT)), 15746, -5217, 2545, -1401}
        int frac = ADD16(-6793, MULT16_16_Q15(n, ADD16(15746, MULT16_16_Q15(n, ADD16(-5217, MULT16_16_Q15(n, ADD16(2545,
            MULT16_16_Q15(n, -1401))))))));
        return S16(SHL16(i - 13, DB_SHIFT) + SHR16(frac, 14 - DB_SHIFT));
    }

    /// <summary>libopus 1.5.2 celt_rcp (mathops.c): the Q15 reciprocal of x &gt; 0 (x in Q(n), result Q(31 - n)); n and r are opus_val16.</summary>
    public static int celt_rcp(int x)
    {
        int i = celt_ilog2(x);
        int n = S16(VSHR32(x, i - 15) - 32768);
        int r = ADD16(30840, MULT16_16_Q15(-15420, n));
        r = S16(SUB16(r, MULT16_16_Q15(r, ADD16(MULT16_16_Q15(r, n), ADD16(r, -32768)))));
        r = S16(SUB16(r, ADD16(1, MULT16_16_Q15(r, ADD16(MULT16_16_Q15(r, n), ADD16(r, -32768))))));
        return VSHR32(r, i - 16);
    }

    /// <summary>libopus 1.5.2 frac_div32 (mathops.c): a / b in Q31, for |a| &lt;= |b|.</summary>
    public static int frac_div32(int a, int b)
    {
        int shift = celt_ilog2(b) - 29;
        a = VSHR32(a, shift);
        b = VSHR32(b, shift);
        int rcp = ROUND16(celt_rcp(ROUND16(b, 16)), 3);
        int result = MULT16_32_Q15(rcp, a);
        int rem = PSHR32(a, 2) - MULT32_32_Q31(result, b);
        result = result + SHL32(MULT16_32_Q15(rcp, rem), 2);
        if (result >= 536870912) return 2147483647;
        if (result <= -536870912) return -2147483647;
        return SHL32(result, 2);
    }

    /// <summary>libopus 1.5.2 celt_rsqrt_norm (mathops.c): the Q14 reciprocal square root of the Q16 x in [0.25, 1).</summary>
    public static int celt_rsqrt_norm(int x)
    {
        int n = S16(x - 32768);
        int r = ADD16(23557, MULT16_16_Q15(n, ADD16(-13490, MULT16_16_Q15(n, 6713))));
        int r2 = S16(MULT16_16_Q15(r, r));
        int y = SHL16(SUB16(ADD16(MULT16_16_Q15(r2, n), r2), 16384), 1);
        return ADD16(r, MULT16_16_Q15(r, MULT16_16_Q15(y, SUB16(MULT16_16_Q15(y, 12288), 16384))));
    }

    /// <summary>libopus celt_maxabs16 / celt_maxabs32 over <paramref name="len"/> values at <paramref name="at"/>.</summary>
    public static int MaxAbs(global::ILGPU.ArrayView<int> v, int at, int len)
    {
        int maxval = 0, minval = 0;
        for (int i = 0; i < len; i++)
        {
            int x = v[at + i];
            maxval = x > maxval ? x : maxval;
            minval = x < minval ? x : minval;
        }
        return maxval > -minval ? maxval : -minval;
    }
}
