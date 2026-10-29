// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt). See NOTICE.md for upstream attributions.
//
// CELT decoder stages around the band decode, on the accelerator (libopus celt/celt.c tf_decode, celt/bands.c
// denormalise_bands, celt/celt_decoder.c deemphasis), fixed-point build. Ported statement for statement from Concentus 2.2.2
// (BSD-3), with one deliberate difference noted at DenormaliseBands.

using ILGPU;
using static SpawnDev.Codecs.Audio.Opus.Celt.CeltFixed;

namespace SpawnDev.Codecs.Audio.Opus.Celt;

/// <summary>CELT decoder stages: TF decode, band denormalisation, de-emphasis (GPU-callable).</summary>
public static class CeltSynthesisGpu
{
    /// <summary>
    /// Per-band TF resolution changes (libopus tf_decode): one bit per band (differential), then the tf_select bit, mapped
    /// through tf_select_table into <paramref name="tfRes"/>.
    /// </summary>
    public static void TfDecode(CeltModeGpu m, int start, int end, int isTransient, ArrayView<int> tfRes, int tfAt, int LM,
        ref OpusRangeDecoderGpuState dec, ArrayView<byte> pkt, int pktAt, uint storage)
    {
        uint num = storage * 8;
        uint num2 = (uint)CeltEntropyGpu.Tell(ref dec);
        int num3 = ((isTransient != 0) ? 2 : 4);
        int num4 = ((LM > 0 && num2 + num3 + 1 <= num) ? 1 : 0);
        num -= (uint)num4;
        int num6;
        int num5 = (num6 = 0);
        for (int i = start; i < end; i++)
        {
            if (num2 + num3 <= num)
            {
                num6 ^= OpusRangeDecoderGpu.DecodeBitLogP(ref dec, pkt, pktAt, storage, num3);
                num2 = (uint)CeltEntropyGpu.Tell(ref dec);
                num5 |= num6;
            }
            tfRes[tfAt + i] = num6;
            num3 = ((isTransient != 0) ? 4 : 5);
        }
        int num7 = 0;
        if (num4 != 0 && m.TfSelectTableOf(LM * 8 + 4 * isTransient + num5) != m.TfSelectTableOf(LM * 8 + 4 * isTransient + 2 + num5))
        {
            num7 = OpusRangeDecoderGpu.DecodeBitLogP(ref dec, pkt, pktAt, storage, 1);
        }
        for (int i = start; i < end; i++)
        {
            tfRes[tfAt + i] = m.TfSelectTableOf(LM * 8 + 4 * isTransient + 2 * num7 + tfRes[tfAt + i]);
        }
    }

    /// <summary>
    /// Scale the normalised spectrum by the band energies into MDCT coefficients (libopus denormalise_bands): X at
    /// <paramref name="xAt"/>, energies (log2, DB_SHIFT) at <paramref name="eAt"/>, M * 120 coefficients to
    /// <paramref name="freqAt"/>.
    /// </summary>
    /// <remarks>
    /// One deliberate difference from Concentus 2.2.2: when a band's gain shift comes out negative (band energy above 2^17,
    /// far outside any real signal), Concentus writes the same output element for every bin of the band and never advances
    /// its input; libopus advances both (`*f++ = SHL32(MULT16_16(*x++, g), -shift)`). This follows libopus.
    /// </remarks>
    public static void DenormaliseBands(CeltModeGpu m, ArrayView<int> x, int xAt, ArrayView<int> freq, int freqAt,
        ArrayView<int> bandLogE, int eAt, int start, int end, int M, int downsample, int silence)
    {
        int num = M * 120;
        int num2 = M * m.EBandsOf(end);
        if (downsample != 1)
        {
            num2 = IMIN(num2, num / downsample);
        }
        if (silence != 0)
        {
            num2 = 0;
            start = (end = 0);
        }
        int index = freqAt;
        int num3 = xAt + M * m.EBandsOf(start);
        for (int i = 0; i < M * m.EBandsOf(start); i++)
        {
            freq[index++] = 0;
        }
        for (int i = start; i < end; i++)
        {
            int num4 = M * m.EBandsOf(i);
            int num5 = M * m.EBandsOf(i + 1);
            int num6 = ADD16(bandLogE[eAt + i], SHL16(m.EMeansOf(i), 6));
            int num7 = 16 - (num6 >> 10);
            int b;
            if (num7 > 31)
            {
                num7 = 0;
                b = 0;
            }
            else
            {
                b = celt_exp2_frac(num6 & 0x3FF);
            }
            if (num7 < 0)
            {
                if (num7 < -2)
                {
                    b = 32767;
                    num7 = -2;
                }
                do
                {
                    freq[index++] = SHL32(MULT16_16(x[num3++], b), -num7);
                }
                while (++num4 < num5);
            }
            else
            {
                do
                {
                    freq[index++] = SHR32(MULT16_16(x[num3++], b), num7);
                }
                while (++num4 < num5);
            }
        }
        for (int i = num2; i < num; i++) freq[freqAt + i] = 0;
    }

    /// <summary>
    /// De-emphasis into int16 PCM (libopus / Concentus deemphasis): channel c's N samples at <paramref name="inAt"/> +
    /// c * <paramref name="inStride"/> to <paramref name="pcm"/> interleaved by C, with the per-channel filter memory at
    /// <paramref name="memAt"/>. <paramref name="accum"/> adds into the PCM already there (hybrid: CELT on top of SILK).
    /// <paramref name="scratch"/> holds N ints when <paramref name="downsample"/> &gt; 1.
    /// </summary>
    public static void Deemphasis(ArrayView<int> input, int inAt, int inStride, ArrayView<short> pcm, int pcmAt, int N, int C,
        int downsample, int coef0, ArrayView<int> mem, int memAt, int accum, ArrayView<int> scratch, int scratchAt)
    {
        int num = 0;
        if (downsample == 1 && C == 2 && accum == 0)
        {
            // deemphasis_stereo_simple
            int a0 = inAt, a1 = inAt + inStride;
            int m0 = mem[memAt], m1 = mem[memAt + 1];
            for (int i = 0; i < N; i++)
            {
                int num5 = input[a0 + i] + m0;
                int num6 = input[a1 + i] + m1;
                m0 = MULT16_32_Q15(coef0, num5);
                m1 = MULT16_32_Q15(coef0, num6);
                pcm[pcmAt + 2 * i] = SIG2WORD16(num5);
                pcm[pcmAt + 2 * i + 1] = SIG2WORD16(num6);
            }
            mem[memAt] = m0;
            mem[memAt + 1] = m1;
            return;
        }
        int a = coef0;
        int num2 = N / downsample;
        int num3 = 0;
        do
        {
            int num4 = mem[memAt + num3];
            int num5 = inAt + num3 * inStride;
            int num6 = pcmAt + num3;
            if (downsample > 1)
            {
                for (int i = 0; i < N; i++)
                {
                    int num7 = input[num5 + i] + num4;
                    num4 = MULT16_32_Q15(a, num7);
                    scratch[scratchAt + i] = num7;
                }
                num = 1;
            }
            else if (accum != 0)
            {
                for (int i = 0; i < N; i++)
                {
                    int num8 = input[num5 + i] + num4;
                    num4 = MULT16_32_Q15(a, num8);
                    pcm[num6 + i * C] = SAT16(ADD32(pcm[num6 + i * C], SIG2WORD16(num8)));
                }
            }
            else
            {
                for (int i = 0; i < N; i++)
                {
                    int num9 = input[num5 + i] + num4;
                    if (input[num5 + i] > 0 && num4 > 0 && num9 < 0)
                    {
                        num9 = int.MaxValue;
                        num4 = int.MaxValue;
                    }
                    else
                    {
                        num4 = MULT16_32_Q15(a, num9);
                    }
                    pcm[num6 + i * C] = SIG2WORD16(num9);
                }
            }
            mem[memAt + num3] = num4;
            if (num != 0)
            {
                for (int i = 0; i < num2; i++)
                {
                    pcm[num6 + i * C] = SIG2WORD16(scratch[scratchAt + i * downsample]);
                }
            }
        }
        while (++num3 < C);
    }
}
