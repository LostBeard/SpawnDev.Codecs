// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt). See NOTICE.md for upstream attributions.
//
// GENERATED (scratchpad celt/extract_inlines.py) from Concentus 2.2.2 Concentus.Common.Inlines (BSD-3, Logan Stromberg +
// the libopus authors): the libopus fixed-point helpers, EVERY overload of each, so CELT code ported from Concentus
// resolves each call to the overload Concentus uses. celt_sqrt's sqrt_C table is inlined as (short) literals (a short[]
// element is a short: an int literal would select the int overloads). Plain integer arithmetic: identical on every ILGPU backend. MULT32_32_Q31 follows libopus 1.5.2's 64-bit
// form (OPUS_FAST_INT64), not Concentus' 32-bit split.

using System.Runtime.CompilerServices;

namespace SpawnDev.Codecs.Audio.Opus.Celt;

/// <summary>libopus / Concentus fixed-point helpers for the CELT GPU primitives (generated).</summary>
internal static class CeltFixed
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static int ABS16(int x)
    {
        if (x >= 0)
        {
            return x;
        }
        return -x;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float ABS16(float x)
    {
        if (!(x < 0f))
        {
            return x;
        }
        return 0f - x;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static short ABS16(short x)
    {
        return (short)((x < 0) ? (-x) : x);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static short ADD16(short a, short b)
    {
        return (short)(a + b);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static int ADD16(int a, int b)
    {
        return a + b;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static int ADD32(int a, int b)
    {
        return a + b;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static short DIV32_16(int a, short b)
    {
        return (short)(a / b);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static int DIV32_16(int a, int b)
    {
        return a / b;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static int EC_ILOG(uint x)
    {
        if (x == 0)
        {
            return 1;
        }
        x |= x >> 1;
        x |= x >> 2;
        x |= x >> 4;
        x |= x >> 8;
        x |= x >> 16;
        uint num = x - ((x >> 1) & 0x55555555);
        num = ((num >> 2) & 0x33333333) + (num & 0x33333333);
        num = ((num >> 4) + num) & 0xF0F0F0F;
        num += num >> 8;
        num += num >> 16;
        return (int)(num & 0x3F);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static int EXTEND32(short x)
    {
        return x;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static int EXTEND32(int x)
    {
        return x;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static short EXTRACT16(int x)
    {
        return (short)x;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static int FRAC_MUL16(int a, int b)
    {
        return 16384 + (short)a * (short)b >> 15;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static short HALF16(short x)
    {
        return SHR16(x, 1);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static int HALF16(int x)
    {
        return SHR32(x, 1);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static int HALF32(int x)
    {
        return SHR32(x, 1);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static int IMAX(int a, int b)
    {
        if (a <= b)
        {
            return b;
        }
        return a;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static int IMIN(int a, int b)
    {
        if (a >= b)
        {
            return b;
        }
        return a;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static uint IMIN(uint a, uint b)
    {
        if (a >= b)
        {
            return b;
        }
        return a;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static int MAC16_16(short c, short a, short b)
    {
        return c + a * b;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static int MAC16_16(int c, short a, short b)
    {
        return c + a * b;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static int MAC16_16(int c, int a, int b)
    {
        return c + a * b;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static short MAX16(short a, short b)
    {
        if (a <= b)
        {
            return b;
        }
        return a;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static int MAX16(int a, int b)
    {
        if (a <= b)
        {
            return b;
        }
        return a;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float MAX16(float a, float b)
    {
        if (!(a > b))
        {
            return b;
        }
        return a;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static int MAX32(int a, int b)
    {
        if (a <= b)
        {
            return b;
        }
        return a;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float MAX32(float a, float b)
    {
        if (!(a > b))
        {
            return b;
        }
        return a;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static short MIN16(short a, short b)
    {
        if (a >= b)
        {
            return b;
        }
        return a;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static int MIN16(int a, int b)
    {
        if (a >= b)
        {
            return b;
        }
        return a;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float MIN16(float a, float b)
    {
        if (!(a < b))
        {
            return b;
        }
        return a;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static int MIN32(int a, int b)
    {
        if (a >= b)
        {
            return b;
        }
        return a;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float MIN32(float a, float b)
    {
        if (!(a < b))
        {
            return b;
        }
        return a;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static int MULT16_16(int a, int b)
    {
        return a * b;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static int MULT16_16(short a, short b)
    {
        return a * b;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static int MULT16_16SU(int a, int b)
    {
        return (short)a * (ushort)b;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static int MULT16_16SU(short a, ushort b)
    {
        return a * b;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static int MULT16_16SU(int a, uint b)
    {
        return a * (int)b;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static short MULT16_16_P15(short a, short b)
    {
        return (short)SHR(ADD32(16384, MULT16_16(a, b)), 15);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static int MULT16_16_P15(int a, int b)
    {
        return SHR(ADD32(16384, MULT16_16(a, b)), 15);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static short MULT16_16_Q14(short a, short b)
    {
        return (short)SHR(MULT16_16(a, b), 14);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static int MULT16_16_Q14(int a, int b)
    {
        return SHR(MULT16_16(a, b), 14);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static short MULT16_16_Q15(short a, short b)
    {
        return (short)SHR(MULT16_16(a, b), 15);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static int MULT16_16_Q15(int a, int b)
    {
        return SHR(MULT16_16(a, b), 15);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static int MULT16_32_Q15(short a, int b)
    {
        return (a * (b >> 16) << 1) + (a * (b & 0xFFFF) >> 15);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static int MULT16_32_Q15(int a, int b)
    {
        return (a * (b >> 16) << 1) + (a * (b & 0xFFFF) >> 15);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static int MULT16_32_Q16(short a, int b)
    {
        return ADD32(MULT16_16(a, SHR(b, 16)), SHR(MULT16_16SU(a, b & 0xFFFF), 16));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static int MULT16_32_Q16(int a, int b)
    {
        return ADD32(MULT16_16(a, SHR(b, 16)), SHR(MULT16_16SU(a, b & 0xFFFF), 16));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static int MULT32_32_Q31(int a, int b)
    {
        // libopus 1.5.2 with OPUS_FAST_INT64 (every 64-bit build): exact; Concentus / libopus 1.1's 32-bit split form drops
        // the low x low product and comes out 1-3 LSB lower
        return (int)(((long)a * b) >> 31);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static short NEG16(short x)
    {
        return (short)(-x);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static int NEG16(int x)
    {
        return -x;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static int NEG32(int x)
    {
        return -x;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static int PSHR32(int a, int shift)
    {
        return SHR32(a + (EXTEND32(1) << shift >> 1), shift);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static short SAT16(int x)
    {
        return (x > 32767) ? short.MaxValue : ((x < -32768) ? short.MinValue : ((short)x));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int SHL(int a, int shift)
    {
        return SHL32(a, shift);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int SHL(short a, int shift)
    {
        return SHL32(a, shift);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static short SHL16(short a, int shift)
    {
        return (short)((ushort)a << shift);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static int SHL16(int a, int shift)
    {
        return a << shift;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static int SHL32(int a, int shift)
    {
        return a << shift;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int SHR(int a, int shift)
    {
        return a >> shift;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int SHR(short a, int shift)
    {
        return a >> shift;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static short SHR16(short a, int shift)
    {
        return (short)(a >> shift);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static int SHR16(int a, int shift)
    {
        return a >> shift;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static int SHR32(int a, int shift)
    {
        return a >> shift;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static short SIG2WORD16(int x)
    {
        x = PSHR32(x, 12);
        x = MAX32(x, -32768);
        x = MIN32(x, 32767);
        return EXTRACT16(x);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static short SUB16(short a, short b)
    {
        return (short)(a - b);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static int SUB16(int a, int b)
    {
        return a - b;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static int SUB32(int a, int b)
    {
        return a - b;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static int VSHR32(int a, int shift)
    {
        if (shift <= 0)
        {
            return SHL32(a, -shift);
        }
        return SHR32(a, shift);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static int _celt_cos_pi_2(int x)
    {
        int num = MULT16_16_P15(x, x);
        return ADD32(1, MIN32(32766, ADD32(SUB16(32767, num), MULT16_16_P15(num, ADD32(-7651, MULT16_16_P15(num, ADD32(8277, MULT16_16_P15(-626, num))))))));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static int abs(int a)
    {
        if (a < 0)
        {
            return -a;
        }
        return a;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static int celt_atan01(int x)
    {
        return MULT16_16_P15(x, ADD32(32767, MULT16_16_P15(x, ADD32(-21, MULT16_16_P15(x, ADD32(-11943, MULT16_16_P15(4936, x)))))));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static int celt_atan2p(int y, int x)
    {
        if (y < x)
        {
            int num = celt_div(SHL32(EXTEND32(y), 15), x);
            if (num >= 32767)
            {
                num = 32767;
            }
            return SHR32(celt_atan01(EXTRACT16(num)), 1);
        }
        int num2 = celt_div(SHL32(EXTEND32(x), 15), y);
        if (num2 >= 32767)
        {
            num2 = 32767;
        }
        return 25736 - SHR16(celt_atan01(EXTRACT16(num2)), 1);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static int celt_cos_norm(int x)
    {
        x &= 0x1FFFF;
        if (x > SHL32(EXTEND32(1), 16))
        {
            x = SUB32(SHL32(EXTEND32(1), 17), x);
        }
        if ((x & 0x7FFF) != 0)
        {
            if (x < SHL32(EXTEND32(1), 15))
            {
                return _celt_cos_pi_2(EXTRACT16(x));
            }
            return NEG32(_celt_cos_pi_2(EXTRACT16(65536 - x)));
        }
        if ((x & 0xFFFF) != 0)
        {
            return 0;
        }
        if ((x & 0x1FFFF) != 0)
        {
            return -32767;
        }
        return 32767;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static int celt_div(int a, int b)
    {
        return MULT32_32_Q31(a, celt_rcp(b));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static int celt_exp2(int x)
    {
        int num = SHR16(x, 10);
        if (num > 14)
        {
            return 2130706432;
        }
        if (num < -15)
        {
            return 0;
        }
        return VSHR32(EXTEND32((int)(short)celt_exp2_frac((short)(x - SHL16((short)num, 10)))), -num - 2);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static int celt_exp2_frac(int x)
    {
        int num = SHL16(x, 4);
        return ADD16(16383, MULT16_16_Q15(num, ADD16(22804, MULT16_16_Q15(num, ADD16(14819, MULT16_16_Q15(10204, num))))));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static int celt_ilog2(int x)
    {
        return EC_ILOG((uint)x) - 1;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static int celt_rcp(int x)
    {
        int num = celt_ilog2(x);
        int b = VSHR32(x, num - 15) - 32768;
        int a = ADD16(30840, MULT16_16_Q15(-15420, b));
        a = SUB16(a, MULT16_16_Q15(a, ADD16(MULT16_16_Q15(a, b), ADD16(a, -32768))));
        a = SUB16(a, ADD16(1, MULT16_16_Q15(a, ADD16(MULT16_16_Q15(a, b), ADD16(a, -32768)))));
        return VSHR32(EXTEND32(a), num - 16);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static int celt_rsqrt_norm(int x)
    {
        int num = x - 32768;
        int num2 = ADD16(23557, MULT16_16_Q15(num, ADD16(-13490, MULT16_16_Q15(num, 6713))));
        int num3 = MULT16_16_Q15(num2, num2);
        int a = SHL16(SUB16(ADD16(MULT16_16_Q15(num3, num), num3), 16384), 1);
        return ADD16(num2, MULT16_16_Q15(num2, MULT16_16_Q15(a, SUB16(MULT16_16_Q15(a, 12288), 16384))));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static int celt_sqrt(int x)
    {
        if (x == 0)
        {
            return 0;
        }
        if (x >= 1073741824)
        {
            return 32767;
        }
        int num = (celt_ilog2(x) >> 1) - 7;
        x = VSHR32(x, 2 * num);
        short a = (short)(x - 32768);
        return VSHR32(ADD16((short)23175, MULT16_16_Q15(a, ADD16((short)11561, MULT16_16_Q15(a, ADD16((short)(-3011), MULT16_16_Q15(a, ADD16((short)1699, MULT16_16_Q15(a, (short)(-664))))))))), 7 - num);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static int celt_sudiv(int n, int d)
    {
        return n / d;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static uint celt_udiv(uint n, uint d)
    {
        return n / d;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static int celt_udiv(int n, int d)
    {
        return n / d;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static int celt_zlog2(int x)
    {
        if (x > 0)
        {
            return celt_ilog2(x);
        }
        return 0;
    }

}
