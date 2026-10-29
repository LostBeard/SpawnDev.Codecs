// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt). See NOTICE.md for upstream attributions.
//
// The CELT MDCT on the accelerator (libopus celt/mdct.c clt_mdct_forward / clt_mdct_backward and celt/kiss_fft.c, fixed-point
// build) for the static 48 kHz mode: N = 1920 >> shift, the FFT of N / 4 points (480, 240, 120, 60) with radix 2, 3, 4 and 5
// butterflies. Ported statement for statement from Concentus 2.2.2 (BSD-3); the twiddle multiplies of Concentus' short and int
// overloads are the same arithmetic, so every value is an int here.

using ILGPU;
using static SpawnDev.Codecs.Audio.Opus.Celt.CeltFixed;

namespace SpawnDev.Codecs.Audio.Opus.Celt;

/// <summary>The CELT MDCT and its KissFFT (GPU-callable).</summary>
public static class CeltMdctGpu
{
    /// <summary>MDCT size of the static mode at shift 0 (libopus mdct.n).</summary>
    public const int MdctN = 1920;
    /// <summary>The FFTs' input scale, 1 / nfft in Q(15 + scale_shift) (libopus fft_state.scale).</summary>
    private const int FftScale = 17476;

    private static int S_MUL(int a, int b) => MULT16_32_Q15(b, a);

    /// <summary>The output order of the FFT at MDCT <paramref name="shift"/> (libopus fft_state.bitrev).</summary>
    private static int Bitrev(CeltModeGpu m, int shift, int i)
    {
        if (shift == 0) return m.FftBitrev480Of(i);
        if (shift == 1) return m.FftBitrev240Of(i);
        if (shift == 2) return m.FftBitrev120Of(i);
        return m.FftBitrev60Of(i);
    }

    private static void Bfly2(ArrayView<int> f, int at, int N)
    {
        const int b = 23170;
        for (int i = 0; i < N; i++)
        {
            int num = at + 8;
            int num2 = f[num];
            int num3 = f[num + 1];
            f[num] = f[at] - num2;
            f[num + 1] = f[at + 1] - num3;
            f[at] += num2;
            f[at + 1] += num3;
            num2 = S_MUL(f[num + 2] + f[num + 3], b);
            num3 = S_MUL(f[num + 3] - f[num + 2], b);
            f[num + 2] = f[at + 2] - num2;
            f[num + 3] = f[at + 3] - num3;
            f[at + 2] += num2;
            f[at + 3] += num3;
            num2 = f[num + 5];
            num3 = -f[num + 4];
            f[num + 4] = f[at + 4] - num2;
            f[num + 5] = f[at + 5] - num3;
            f[at + 4] += num2;
            f[at + 5] += num3;
            num2 = S_MUL(f[num + 7] - f[num + 6], b);
            num3 = S_MUL(-f[num + 7] - f[num + 6], b);
            f[num + 6] = f[at + 6] - num2;
            f[num + 7] = f[at + 7] - num3;
            f[at + 6] += num2;
            f[at + 7] += num3;
            at += 16;
        }
    }

    private static void Bfly4(CeltModeGpu st, ArrayView<int> f, int at, int fstride, int m, int N, int mm)
    {
        if (m == 1)
        {
            for (int i = 0; i < N; i++)
            {
                int num = f[at] - f[at + 4];
                int num2 = f[at + 1] - f[at + 5];
                f[at] += f[at + 4];
                f[at + 1] += f[at + 5];
                int num3 = f[at + 2] + f[at + 6];
                int num4 = f[at + 3] + f[at + 7];
                f[at + 4] = f[at] - num3;
                f[at + 5] = f[at + 1] - num4;
                f[at] += num3;
                f[at + 1] += num4;
                num3 = f[at + 2] - f[at + 6];
                num4 = f[at + 3] - f[at + 7];
                f[at + 2] = num + num4;
                f[at + 3] = num2 - num3;
                f[at + 6] = num - num4;
                f[at + 7] = num2 + num3;
                at += 8;
            }
            return;
        }
        int num5 = at;
        for (int i = 0; i < N; i++)
        {
            at = num5 + 2 * i * mm;
            int num6 = at + 2 * m;
            int num7 = at + 4 * m;
            int num8 = at + 6 * m;
            int num9 = 0, num10 = 0, num11 = 0;
            for (int j = 0; j < m; j++)
            {
                int num12 = S_MUL(f[num6], st.FftTwiddlesOf(num11)) - S_MUL(f[num6 + 1], st.FftTwiddlesOf(num11 + 1));
                int num13 = S_MUL(f[num6], st.FftTwiddlesOf(num11 + 1)) + S_MUL(f[num6 + 1], st.FftTwiddlesOf(num11));
                int num14 = S_MUL(f[num7], st.FftTwiddlesOf(num10)) - S_MUL(f[num7 + 1], st.FftTwiddlesOf(num10 + 1));
                int num15 = S_MUL(f[num7], st.FftTwiddlesOf(num10 + 1)) + S_MUL(f[num7 + 1], st.FftTwiddlesOf(num10));
                int num16 = S_MUL(f[num8], st.FftTwiddlesOf(num9)) - S_MUL(f[num8 + 1], st.FftTwiddlesOf(num9 + 1));
                int num17 = S_MUL(f[num8], st.FftTwiddlesOf(num9 + 1)) + S_MUL(f[num8 + 1], st.FftTwiddlesOf(num9));
                int num18 = f[at] - num14;
                int num19 = f[at + 1] - num15;
                f[at] += num14;
                f[at + 1] += num15;
                int num20 = num12 + num16;
                int num21 = num13 + num17;
                int num22 = num12 - num16;
                int num23 = num13 - num17;
                f[num7] = f[at] - num20;
                f[num7 + 1] = f[at + 1] - num21;
                num11 += fstride * 2;
                num10 += fstride * 4;
                num9 += fstride * 6;
                f[at] += num20;
                f[at + 1] += num21;
                f[num6] = num18 + num23;
                f[num6 + 1] = num19 - num22;
                f[num8] = num18 - num23;
                f[num8 + 1] = num19 + num22;
                at += 2;
                num6 += 2;
                num7 += 2;
                num8 += 2;
            }
        }
    }

    private static void Bfly3(CeltModeGpu st, ArrayView<int> f, int at, int fstride, int m, int N, int mm)
    {
        int num = 2 * m;
        int num2 = 4 * m;
        int num3 = at;
        for (int i = 0; i < N; i++)
        {
            at = num3 + 2 * i * mm;
            int num4 = 0, num5 = 0;
            int num6 = m;
            do
            {
                int num7 = S_MUL(f[at + num], st.FftTwiddlesOf(num4)) - S_MUL(f[at + num + 1], st.FftTwiddlesOf(num4 + 1));
                int num8 = S_MUL(f[at + num], st.FftTwiddlesOf(num4 + 1)) + S_MUL(f[at + num + 1], st.FftTwiddlesOf(num4));
                int num9 = S_MUL(f[at + num2], st.FftTwiddlesOf(num5)) - S_MUL(f[at + num2 + 1], st.FftTwiddlesOf(num5 + 1));
                int num10 = S_MUL(f[at + num2], st.FftTwiddlesOf(num5 + 1)) + S_MUL(f[at + num2 + 1], st.FftTwiddlesOf(num5));
                int num11 = num7 + num9;
                int num12 = num8 + num10;
                int a = num7 - num9;
                int a2 = num8 - num10;
                num4 += fstride * 2;
                num5 += fstride * 4;
                f[at + num] = f[at] - (num11 >> 1);
                f[at + num + 1] = f[at + 1] - (num12 >> 1);
                a = S_MUL(a, -28378);
                a2 = S_MUL(a2, -28378);
                f[at] += num11;
                f[at + 1] += num12;
                f[at + num2] = f[at + num] + a2;
                f[at + num2 + 1] = f[at + num + 1] - a;
                f[at + num] -= a2;
                f[at + num + 1] += a;
                at += 2;
            }
            while (--num6 != 0);
        }
    }

    private static void Bfly5(CeltModeGpu st, ArrayView<int> f, int at, int fstride, int m, int N, int mm)
    {
        int num = at;
        const int b = 10126, b2 = -31164, b3 = -26510, b4 = -19261;
        for (int i = 0; i < N; i++)
        {
            int num2 = 0, num3 = 0, num4 = 0, num5 = 0;
            at = num + 2 * i * mm;
            int num6 = at;
            int num7 = at + 2 * m;
            int num8 = at + 4 * m;
            int num9 = at + 6 * m;
            int num10 = at + 8 * m;
            for (int j = 0; j < m; j++)
            {
                int num11 = f[num6];
                int num12 = f[num6 + 1];
                int num13 = S_MUL(f[num7], st.FftTwiddlesOf(num2)) - S_MUL(f[num7 + 1], st.FftTwiddlesOf(num2 + 1));
                int num14 = S_MUL(f[num7], st.FftTwiddlesOf(num2 + 1)) + S_MUL(f[num7 + 1], st.FftTwiddlesOf(num2));
                int num15 = S_MUL(f[num8], st.FftTwiddlesOf(num3)) - S_MUL(f[num8 + 1], st.FftTwiddlesOf(num3 + 1));
                int num16 = S_MUL(f[num8], st.FftTwiddlesOf(num3 + 1)) + S_MUL(f[num8 + 1], st.FftTwiddlesOf(num3));
                int num17 = S_MUL(f[num9], st.FftTwiddlesOf(num4)) - S_MUL(f[num9 + 1], st.FftTwiddlesOf(num4 + 1));
                int num18 = S_MUL(f[num9], st.FftTwiddlesOf(num4 + 1)) + S_MUL(f[num9 + 1], st.FftTwiddlesOf(num4));
                int num19 = S_MUL(f[num10], st.FftTwiddlesOf(num5)) - S_MUL(f[num10 + 1], st.FftTwiddlesOf(num5 + 1));
                int num20 = S_MUL(f[num10], st.FftTwiddlesOf(num5 + 1)) + S_MUL(f[num10 + 1], st.FftTwiddlesOf(num5));
                num2 += 2 * fstride;
                num3 += 4 * fstride;
                num4 += 6 * fstride;
                num5 += 8 * fstride;
                int num21 = num13 + num19;
                int num22 = num14 + num20;
                int a = num13 - num19;
                int a2 = num14 - num20;
                int num23 = num15 + num17;
                int num24 = num16 + num18;
                int a3 = num15 - num17;
                int a4 = num16 - num18;
                f[num6] += num21 + num23;
                f[num6 + 1] += num22 + num24;
                int num25 = num11 + S_MUL(num21, b) + S_MUL(num23, b3);
                int num26 = num12 + S_MUL(num22, b) + S_MUL(num24, b3);
                int num27 = S_MUL(a2, b2) + S_MUL(a4, b4);
                int num28 = -S_MUL(a, b2) - S_MUL(a3, b4);
                f[num7] = num25 - num27;
                f[num7 + 1] = num26 - num28;
                f[num10] = num25 + num27;
                f[num10 + 1] = num26 + num28;
                int num29 = num11 + S_MUL(num21, b3) + S_MUL(num23, b);
                int num30 = num12 + S_MUL(num22, b3) + S_MUL(num24, b);
                int num31 = -S_MUL(a2, b4) + S_MUL(a4, b2);
                int num32 = S_MUL(a, b4) - S_MUL(a3, b2);
                f[num8] = num29 + num31;
                f[num8 + 1] = num30 + num32;
                f[num9] = num29 - num31;
                f[num9 + 1] = num30 - num32;
                num6 += 2;
                num7 += 2;
                num8 += 2;
                num9 += 2;
                num10 += 2;
            }
        }
    }

    /// <summary>
    /// The in-place FFT of the N / 4 = 480 >> <paramref name="shift"/> complex values (re, im interleaved) at
    /// <paramref name="at"/>, already in bit-reversed order (libopus opus_fft_impl).
    /// </summary>
    public static void FftImpl(CeltModeGpu st, int shift, ArrayView<int> f, int at)
    {
        int fa = shift * 16;
        int fshift = shift > 0 ? shift : 0;   // libopus fft_state.shift is -1 for the base FFT
        // libopus keeps fstride[i] = the product of the first i radices in a local array; the stages run in reverse, so each
        // product is recomputed here
        int stages = 0;
        while (st.FftFactorsOf(fa + 2 * stages + 1) != 1) stages++;
        stages++;
        int m = st.FftFactorsOf(fa + 2 * stages - 1);
        for (int s = stages - 1; s >= 0; s--)
        {
            int mm = s == 0 ? 1 : st.FftFactorsOf(fa + 2 * s - 1);
            int fstride = 1;
            for (int j = 0; j < s; j++) fstride *= st.FftFactorsOf(fa + 2 * j);
            int p = st.FftFactorsOf(fa + 2 * s);
            if (p == 2) Bfly2(f, at, fstride);
            else if (p == 4) Bfly4(st, f, at, fstride << fshift, m, fstride, mm);
            else if (p == 3) Bfly3(st, f, at, fstride << fshift, m, fstride, mm);
            else if (p == 5) Bfly5(st, f, at, fstride << fshift, m, fstride, mm);
            m = mm;
        }
    }

    /// <summary>
    /// The inverse MDCT with its windowed overlap (libopus clt_mdct_backward): N / 2 = 960 >> <paramref name="shift"/> coefficients
    /// at <paramref name="inAt"/>, every <paramref name="stride"/>-th (interleaved short blocks), to N / 2 + overlap samples at
    /// <paramref name="outAt"/>. The first and last <paramref name="overlap"/> samples are windowed for the overlap-add with the
    /// neighbouring frames, which the caller does.
    /// </summary>
    public static void Backward(CeltModeGpu m, ArrayView<int> input, int inAt, ArrayView<int> output, int outAt, int overlap,
        int shift, int stride)
    {
        int num = 0;
        int num2 = MdctN;
        for (int i = 0; i < shift; i++)
        {
            num2 >>= 1;
            num += num2;
        }
        int num3 = num2 >> 1;
        int num4 = num2 >> 2;
        int num5 = inAt + stride * (num3 - 1);
        int num6 = outAt + (overlap >> 1);
        for (int i = 0; i < num4; i++)
        {
            int num9 = num6 + 2 * Bitrev(m, shift, i);
            output[num9 + 1] = S_MUL(input[num5], m.MdctTrigOf(num + i)) + S_MUL(input[inAt], m.MdctTrigOf(num + num4 + i));
            output[num9] = S_MUL(input[inAt], m.MdctTrigOf(num + i)) - S_MUL(input[num5], m.MdctTrigOf(num + num4 + i));
            inAt += 2 * stride;
            num5 -= 2 * stride;
        }
        FftImpl(m, shift, output, outAt + (overlap >> 1));
        int num10 = outAt + (overlap >> 1);
        int num11 = outAt + (overlap >> 1) + num3 - 2;
        int num12 = num;
        int num13 = num12 + num4 - 1;
        int num14 = num12 + num3 - 1;
        for (int i = 0; i < (num4 + 1) >> 1; i++)
        {
            int a = output[num10 + 1];
            int a2 = output[num10];
            int b = m.MdctTrigOf(num12 + i);
            int b2 = m.MdctTrigOf(num12 + num4 + i);
            int num15 = S_MUL(a, b) + S_MUL(a2, b2);
            int num16 = S_MUL(a, b2) - S_MUL(a2, b);
            int a3 = output[num11 + 1];
            a2 = output[num11];
            output[num10] = num15;
            output[num11 + 1] = num16;
            b = m.MdctTrigOf(num13 - i);
            b2 = m.MdctTrigOf(num14 - i);
            num15 = S_MUL(a3, b) + S_MUL(a2, b2);
            num16 = S_MUL(a3, b2) - S_MUL(a2, b);
            output[num11] = num15;
            output[num10 + 1] = num16;
            num10 += 2;
            num11 -= 2;
        }
        int index = outAt + overlap - 1;
        num11 = outAt;
        int num17 = 0;
        int num18 = overlap - 1;
        for (int i = 0; i < overlap / 2; i++)
        {
            int b3 = output[index];
            int b4 = output[num11];
            output[num11++] = MULT16_32_Q15(m.WindowOf(num18), b4) - MULT16_32_Q15(m.WindowOf(num17), b3);
            output[index--] = MULT16_32_Q15(m.WindowOf(num17), b4) + MULT16_32_Q15(m.WindowOf(num18), b3);
            num17++;
            num18--;
        }
    }

    /// <summary>
    /// The forward MDCT (libopus clt_mdct_forward): N / 2 + overlap windowed samples at <paramref name="inAt"/> to N / 2 = 960 >>
    /// <paramref name="shift"/> coefficients at <paramref name="outAt"/>, every <paramref name="stride"/>-th.
    /// <paramref name="scratch"/> holds N / 2 + N / 2 ints at <paramref name="scratchAt"/> (libopus' f and f2).
    /// </summary>
    public static void Forward(CeltModeGpu m, ArrayView<int> input, int inAt, ArrayView<int> output, int outAt, int overlap,
        int shift, int stride, ArrayView<int> scratch, int scratchAt)
    {
        int num = 0;
        int shift2 = 8 - shift - 1;   // libopus fft_state.scale_shift - 1
        int num2 = MdctN;
        int i;
        for (i = 0; i < shift; i++)
        {
            num2 >>= 1;
            num += num2;
        }
        int num3 = num2 >> 1;
        int num4 = num2 >> 2;
        int fAt = scratchAt, f2At = scratchAt + num3;
        int num5 = inAt + (overlap >> 1);
        int num6 = inAt + num3 - 1 + (overlap >> 1);
        int num7 = fAt;
        int num8 = overlap >> 1;
        int num9 = (overlap >> 1) - 1;
        for (i = 0; i < (overlap + 3) >> 2; i++)
        {
            scratch[num7++] = MULT16_32_Q15(m.WindowOf(num9), input[num5 + num3]) + MULT16_32_Q15(m.WindowOf(num8), input[num6]);
            scratch[num7++] = MULT16_32_Q15(m.WindowOf(num8), input[num5]) - MULT16_32_Q15(m.WindowOf(num9), input[num6 - num3]);
            num5 += 2;
            num6 -= 2;
            num8 += 2;
            num9 -= 2;
        }
        num8 = 0;
        num9 = overlap - 1;
        for (; i < num4 - ((overlap + 3) >> 2); i++)
        {
            scratch[num7++] = input[num6];
            scratch[num7++] = input[num5];
            num5 += 2;
            num6 -= 2;
        }
        for (; i < num4; i++)
        {
            scratch[num7++] = MULT16_32_Q15(m.WindowOf(num9), input[num6]) - MULT16_32_Q15(m.WindowOf(num8), input[num5 - num3]);
            scratch[num7++] = MULT16_32_Q15(m.WindowOf(num9), input[num5]) + MULT16_32_Q15(m.WindowOf(num8), input[num6 + num3]);
            num5 += 2;
            num6 -= 2;
            num8 += 2;
            num9 -= 2;
        }
        int num10 = fAt;
        int num11 = num;
        for (i = 0; i < num4; i++)
        {
            int b = m.MdctTrigOf(num11 + i);
            int b2 = m.MdctTrigOf(num11 + num4 + i);
            int a = scratch[num10++];
            int a2 = scratch[num10++];
            int b3 = S_MUL(a, b) - S_MUL(a2, b2);
            int b4 = S_MUL(a2, b) + S_MUL(a, b2);
            int r = Bitrev(m, shift, i);
            scratch[f2At + 2 * r] = PSHR32(MULT16_32_Q16(FftScale, b3), shift2);
            scratch[f2At + 2 * r + 1] = PSHR32(MULT16_32_Q16(FftScale, b4), shift2);
        }
        FftImpl(m, shift, scratch, f2At);
        int num12 = f2At;
        int num13 = outAt;
        int num14 = outAt + stride * (num3 - 1);
        int num15 = num;
        for (i = 0; i < num4; i++)
        {
            int num16 = S_MUL(scratch[num12 + 1], m.MdctTrigOf(num15 + num4 + i)) - S_MUL(scratch[num12], m.MdctTrigOf(num15 + i));
            int num17 = S_MUL(scratch[num12], m.MdctTrigOf(num15 + num4 + i)) + S_MUL(scratch[num12 + 1], m.MdctTrigOf(num15 + i));
            output[num13] = num16;
            output[num14] = num17;
            num12 += 2;
            num13 += 2 * stride;
            num14 -= 2 * stride;
        }
    }
}
