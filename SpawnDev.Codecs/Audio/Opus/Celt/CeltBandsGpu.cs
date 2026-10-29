// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt). See NOTICE.md for upstream attributions.
//
// CELT band-level operations on the accelerator (libopus celt/bands.c), fixed-point build: anti-collapse (noise fill for
// short blocks that got no pulses), the stereo transforms (intensity stereo, mid/side split and merge), the Haar step and
// the (de)interleave of short blocks used by the TF resolution changes, and the split helpers (compute_qn, bitexact_cos,
// bitexact_log2tan, the LCG). Ported statement for statement from Concentus 2.2.2 (BSD-3) so the results are bit-exact with it.
//
// GPU shape: libopus' per-channel arrays (X_[c], bandE[c]) are one view with a channel stride; the collapse masks are an int
// view (one mask per band and channel, libopus' bytes); (de)interleave_hadamard's stack array is a caller scratch view.

using ILGPU;
using static SpawnDev.Codecs.Audio.Opus.Celt.CeltFixed;

namespace SpawnDev.Codecs.Audio.Opus.Celt;

/// <summary>CELT band operations (bands.c): anti-collapse, stereo, Haar / Hadamard interleave, split helpers (GPU-callable).</summary>
public static class CeltBandsGpu
{
    /// <summary>The LCG CELT uses for noise (libopus celt_lcg_rand).</summary>
    public static uint CeltLcgRand(uint seed)
    {
        return 1664525 * seed + 1013904223;
    }

    /// <summary>cos(pi/2 * x / 16384) in Q15 with a fixed-point polynomial, the same on every platform (libopus bitexact_cos).</summary>
    public static int BitexactCos(int x)
    {
        int num = 4096 + x * x >> 13;
        num = 32767 - num + FRAC_MUL16(num, -7651 + FRAC_MUL16(num, 8277 + FRAC_MUL16(-626, num)));
        return 1 + num;
    }

    /// <summary>log2(isin / icos) in Q11 (libopus bitexact_log2tan).</summary>
    public static int BitexactLog2tan(int isin, int icos)
    {
        int num = EC_ILOG((uint)icos);
        int num2 = EC_ILOG((uint)isin);
        icos <<= 15 - num;
        isin <<= 15 - num2;
        return (num2 - num) * 2048 + FRAC_MUL16(isin, FRAC_MUL16(isin, -2597) + 7932) - FRAC_MUL16(icos, FRAC_MUL16(icos, -2597) + 7932);
    }

    /// <summary>Quantization steps of the split angle for a band of N with b (1/8) bits (libopus compute_qn).</summary>
    public static int ComputeQn(int N, int b, int offset, int pulse_cap, int stereo)
    {
        int num = 2 * N - 1;
        if (stereo != 0 && N == 2)
        {
            num--;
        }
        int b2 = celt_sudiv(b + num * offset, num);
        b2 = IMIN(b - pulse_cap - 32, b2);
        b2 = IMIN(64, b2);
        if (b2 < 4)
        {
            return 1;
        }
        int num2 = ExpTable(b2 & 7) >> 14 - (b2 >> 3);
        return num2 + 1 >> 1 << 1;
    }

    /// <summary>compute_qn's exp2 table: 2^(i/8) in Q14, as a short (the array is short[] in libopus / Concentus).</summary>
    private static short ExpTable(int i)
    {
        switch (i)
        {
            case 0: return 16384;
            case 1: return 17866;
            case 2: return 19483;
            case 3: return 21247;
            case 4: return 23170;
            case 5: return 25267;
            case 6: return 27554;
            default: return 30048;
        }
    }

    /// <summary>Fold the side channel into the mid by the bands' energies (libopus intensity_stereo).
    /// <paramref name="bandE"/> holds channel 1 at <paramref name="bandEAt"/> + <paramref name="bandEStride"/>.</summary>
    public static void IntensityStereo(ArrayView<int> x, int xAt, ArrayView<int> y, int yAt,
        ArrayView<int> bandE, int bandEAt, int bandEStride, int bandID, int N)
    {
        int shift = celt_zlog2(MAX32(bandE[bandEAt + bandID], bandE[bandEAt + bandEStride + bandID])) - 13;
        int num = VSHR32(bandE[bandEAt + bandID], shift);
        int num2 = VSHR32(bandE[bandEAt + bandEStride + bandID], shift);
        int b = 1 + celt_sqrt(1 + MULT16_16(num, num) + MULT16_16(num2, num2));
        int a = DIV32_16(SHL32(num, 14), b);
        int a2 = DIV32_16(SHL32(num2, 14), b);
        for (int i = 0; i < N; i++)
        {
            int b2 = x[xAt + i];
            int b3 = y[yAt + i];
            x[xAt + i] = EXTRACT16(SHR32(MAC16_16(MULT16_16(a, b2), a2, b3), 14));
        }
    }

    /// <summary>L/R to M/S scaled by 1/sqrt(2) (libopus stereo_split).</summary>
    public static void StereoSplit(ArrayView<int> x, int xAt, ArrayView<int> y, int yAt, int N)
    {
        for (int i = 0; i < N; i++)
        {
            int num = MULT16_16(23170, x[xAt + i]);
            int num2 = MULT16_16(23170, y[yAt + i]);
            x[xAt + i] = EXTRACT16(SHR32(ADD32(num, num2), 15));
            y[yAt + i] = EXTRACT16(SHR32(SUB32(num2, num), 15));
        }
    }

    /// <summary>M/S back to L/R, renormalised, given the mid gain <paramref name="mid"/> (libopus stereo_merge).</summary>
    public static void StereoMerge(ArrayView<int> x, int xAt, ArrayView<int> y, int yAt, int mid, int N)
    {
        int xy = 0;
        int xy2 = 0;
        for (int i = 0; i < N; i++)   // Kernels.dual_inner_prod(Y, X, Y)
        {
            xy = MAC16_16(xy, y[yAt + i], x[xAt + i]);
            xy2 = MAC16_16(xy2, y[yAt + i], y[yAt + i]);
        }
        xy = MULT16_32_Q15(mid, xy);
        int num = SHR16(mid, 1);
        int num2 = MULT16_16(num, num) + xy2 - 2 * xy;
        int num3 = MULT16_16(num, num) + xy2 + 2 * xy;
        if (num3 < 161061 || num2 < 161061)
        {
            for (int i = 0; i < N; i++) y[yAt + i] = x[xAt + i];
            return;
        }
        int num4 = celt_ilog2(num2) >> 1;
        int num5 = celt_ilog2(num3) >> 1;
        int a = celt_rsqrt_norm(VSHR32(num2, num4 - 7 << 1));
        int a2 = celt_rsqrt_norm(VSHR32(num3, num5 - 7 << 1));
        if (num4 < 7)
        {
            num4 = 7;
        }
        if (num5 < 7)
        {
            num5 = 7;
        }
        for (int i = 0; i < N; i++)
        {
            int a3 = MULT16_16_P15(mid, x[xAt + i]);
            int b = y[yAt + i];
            x[xAt + i] = EXTRACT16(PSHR32(MULT16_16(a, SUB16(a3, b)), num4 + 1));
            y[yAt + i] = EXTRACT16(PSHR32(MULT16_16(a2, ADD16(a3, b)), num5 + 1));
        }
    }

    /// <summary>One Haar step over <paramref name="N0"/> values of each of <paramref name="stride"/> interleaved blocks (libopus haar1).</summary>
    public static void Haar1(ArrayView<int> x, int xAt, int N0, int stride)
    {
        N0 >>= 1;
        for (int i = 0; i < stride; i++)
        {
            for (int j = 0; j < N0; j++)
            {
                int num = xAt + i + stride * 2 * j;
                int a = MULT16_16(23170, x[num]);
                int b = MULT16_16(23170, x[num + stride]);
                x[num] = EXTRACT16(PSHR32(ADD32(a, b), 15));
                x[num + stride] = EXTRACT16(PSHR32(SUB32(a, b), 15));
            }
        }
    }

    /// <summary>Interleaved short blocks to block-contiguous order, Hadamard order when <paramref name="hadamard"/> (libopus
    /// deinterleave_hadamard). <paramref name="scratch"/> holds N0 * stride ints.</summary>
    public static void DeinterleaveHadamard(CeltModeGpu m, ArrayView<int> x, int xAt, int N0, int stride, int hadamard,
        ArrayView<int> scratch, int scratchAt)
    {
        int num = N0 * stride;
        if (hadamard != 0)
        {
            int num2 = stride - 2;
            for (int i = 0; i < stride; i++)
            {
                for (int j = 0; j < N0; j++)
                {
                    scratch[scratchAt + m.OrderyTableOf(num2 + i) * N0 + j] = x[j * stride + i + xAt];
                }
            }
        }
        else
        {
            for (int i = 0; i < stride; i++)
            {
                for (int j = 0; j < N0; j++)
                {
                    scratch[scratchAt + i * N0 + j] = x[j * stride + i + xAt];
                }
            }
        }
        for (int i = 0; i < num; i++) x[xAt + i] = scratch[scratchAt + i];
    }

    /// <summary>The inverse of <see cref="DeinterleaveHadamard"/> (libopus interleave_hadamard).</summary>
    public static void InterleaveHadamard(CeltModeGpu m, ArrayView<int> x, int xAt, int N0, int stride, int hadamard,
        ArrayView<int> scratch, int scratchAt)
    {
        int num = N0 * stride;
        if (hadamard != 0)
        {
            int num2 = stride - 2;
            for (int i = 0; i < stride; i++)
            {
                for (int j = 0; j < N0; j++)
                {
                    scratch[scratchAt + j * stride + i] = x[m.OrderyTableOf(num2 + i) * N0 + j + xAt];
                }
            }
        }
        else
        {
            for (int i = 0; i < stride; i++)
            {
                for (int j = 0; j < N0; j++)
                {
                    scratch[scratchAt + j * stride + i] = x[i * N0 + j + xAt];
                }
            }
        }
        for (int i = 0; i < num; i++) x[xAt + i] = scratch[scratchAt + i];
    }

    /// <summary>
    /// Anti-collapse (libopus anti_collapse): fills the short blocks of each band that got no pulses (collapse mask bit clear) with
    /// signed noise at a level from the band's energy history, then renormalises the band. <paramref name="x"/> holds channel c at
    /// <paramref name="xAt"/> + c * <paramref name="xStride"/> (the frame's normalised spectrum); <paramref name="collapseMasks"/>
    /// one mask per (band, channel) at <c>i * C + c</c>; the energy arrays hold channel c at c * NbEBands from their offsets.
    /// </summary>
    public static void AntiCollapse(CeltModeGpu m, ArrayView<int> x, int xAt, int xStride, ArrayView<int> collapseMasks, int masksAt,
        int LM, int C, int start, int end,
        ArrayView<int> logE, int logEAt, ArrayView<int> prev1logE, int prev1At, ArrayView<int> prev2logE, int prev2At,
        ArrayView<int> pulses, int pulsesAt, uint seed)
    {
        int nbEBands = CeltModeTables.NbEBands;
        for (int i = start; i < end; i++)
        {
            int num = m.EBandsOf(i + 1) - m.EBandsOf(i);
            int a = celt_udiv(1 + pulses[pulsesAt + i], m.EBandsOf(i + 1) - m.EBandsOf(i)) >> LM;
            int b = SHR32(celt_exp2(-SHL16(a, 7)), 1);
            int a2 = MULT16_32_Q15((short)16384, MIN32(32767, b));
            int num2 = num << LM;
            int num3 = celt_ilog2(num2) >> 1;
            int a3 = celt_rsqrt_norm(SHL32(num2, 7 - num3 << 1));
            int num4 = 0;
            do
            {
                int num5 = 0;
                int a4 = prev1logE[prev1At + num4 * nbEBands + i];
                int num6 = prev2logE[prev2At + num4 * nbEBands + i];
                if (C == 1)
                {
                    a4 = MAX16(a4, prev1logE[prev1At + nbEBands + i]);
                    num6 = MAX16(num6, prev2logE[prev2At + nbEBands + i]);
                }
                int b2 = EXTEND32(logE[logEAt + num4 * nbEBands + i]) - EXTEND32(MIN16(a4, num6));
                b2 = MAX32(0, b2);
                int b4;
                if (b2 < 16384)
                {
                    int b3 = SHR32(celt_exp2((short)(-EXTRACT16(b2))), 1);
                    b4 = 2 * MIN16(16383, b3);
                }
                else
                {
                    b4 = 0;
                }
                if (LM == 3)
                {
                    b4 = MULT16_16_Q14(23170, MIN32(23169, b4));
                }
                b4 = SHR16(MIN16(a2, b4), 1);
                b4 = SHR32(MULT16_16_Q15(a3, b4), num3);
                int num7 = m.EBandsOf(i) << LM;
                int chan = xAt + num4 * xStride;
                for (int j = 0; j < 1 << LM; j++)
                {
                    if ((collapseMasks[masksAt + i * C + num4] & (1 << j)) == 0)
                    {
                        int num8 = num7 + j;
                        for (int k = 0; k < num; k++)
                        {
                            seed = CeltLcgRand(seed);
                            x[chan + num8 + (k << LM)] = (((seed & 0x8000) != 0) ? b4 : (-b4));
                        }
                        num5 = 1;
                    }
                }
                if (num5 != 0)
                {
                    CeltPvqGpu.RenormaliseVector(x, chan + num7, num << LM, 32767);
                }
            }
            while (++num4 < C);
        }
    }
}
