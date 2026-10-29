// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt). See NOTICE.md for upstream attributions.
//
// CELT pyramid vector quantization (PVQ) on the accelerator: the codebook index <-> pulse vector maps (libopus celt/cwrs.c)
// and the band shape quantizer / dequantizer with its spreading rotation (celt/vq.c), fixed-point build. Ported statement for
// statement from Concentus 2.2.2 (BSD-3, a C# port of libopus) so the results are bit-exact with it: every helper call goes
// through CeltFixed, which carries all of Concentus' overloads, so each call resolves to the overload Concentus uses.
//
// GPU shape of the port: arrays become ArrayView + start offset; the per-call scratch arrays libopus allocates on the stack
// (the pulse vector, the search state) are caller-provided views; the codebook-size table U(N, K) is a uint view of
// CeltPvqTables.Data. The range coder is OpusRangeDecoderGpu. alg_quant's final enc_uint is left to the caller
// (AlgQuantSearch returns the pulses; the index is Icwrs(n, pulses) coded with ft = PvqV(n, k)).

using ILGPU;
using static SpawnDev.Codecs.Audio.Opus.Celt.CeltFixed;

namespace SpawnDev.Codecs.Audio.Opus.Celt;

/// <summary>CELT PVQ: codebook index maps, band shape decode / encode search, spreading rotation (GPU-callable).</summary>
public static class CeltPvqGpu
{
    /// <summary>libopus CELT_PVQ_U_ROW: start of row <paramref name="r"/> of the U(N, K) table.</summary>
    private static uint RowStart(int r)
    {
        switch (r)
        {
            case 0: return 0u;
            case 1: return 176u;
            case 2: return 351u;
            case 3: return 525u;
            case 4: return 698u;
            case 5: return 870u;
            case 6: return 1041u;
            case 7: return 1131u;
            case 8: return 1178u;
            case 9: return 1207u;
            case 10: return 1226u;
            case 11: return 1240u;
            case 12: return 1248u;
            case 13: return 1254u;
            default: return 1257u;
        }
    }

    /// <summary>U(N, K): the number of N-dimensional vectors of K pulses with a positive first non-zero entry.</summary>
    public static uint PvqU(ArrayView<uint> u, int uAt, int n, int k) => u[uAt + RowStart(IMIN(n, k)) + IMAX(n, k)];

    /// <summary>V(N, K): the size of the PVQ codebook for N dimensions and K pulses (libopus CELT_PVQ_V).</summary>
    public static uint PvqV(ArrayView<uint> u, int uAt, int n, int k) => PvqU(u, uAt, n, k) + PvqU(u, uAt, n, k + 1);

    /// <summary>Codebook index of the pulse vector <paramref name="y"/> (libopus icwrs).</summary>
    public static uint Icwrs(int n, ArrayView<int> y, int yAt, ArrayView<uint> u, int uAt)
    {
        int num = n - 1;
        uint num2 = ((y[yAt + num] < 0) ? 1u : 0u);
        int num3 = abs(y[yAt + num]);
        do
        {
            num--;
            num2 += PvqU(u, uAt, n - num, num3);
            num3 += abs(y[yAt + num]);
            if (y[yAt + num] < 0)
            {
                num2 += PvqU(u, uAt, n - num, num3 + 1);
            }
        }
        while (num > 0);
        return num2;
    }

    /// <summary>Pulse vector of codebook index <paramref name="i"/> into <paramref name="y"/>; returns its squared norm (libopus cwrsi).</summary>
    public static int Cwrsi(int n, int k, uint i, ArrayView<int> y, int yAt, ArrayView<uint> u, int uAt)
    {
        int c = 0;
        int num = 0;
        uint num3;
        int num5;
        int num4;
        short num7;
        while (n > 2)
        {
            if (k >= n)
            {
                uint num2 = RowStart(n);
                num3 = u[uAt + num2 + k + 1];
                num4 = 0 - ((i >= num3) ? 1 : 0);
                i -= num3 & (uint)num4;
                num5 = k;
                uint num6 = u[uAt + num2 + n];
                if (num6 > i)
                {
                    k = n;
                    do
                    {
                        num3 = u[uAt + RowStart(--k) + n];
                    }
                    while (num3 > i);
                }
                else
                {
                    for (num3 = u[uAt + num2 + k]; num3 > i; num3 = u[uAt + num2 + k])
                    {
                        k--;
                    }
                }
                i -= num3;
                num7 = (short)((num5 - k + num4) ^ num4);
                y[yAt + num++] = num7;
                c = MAC16_16(c, num7, num7);
            }
            else
            {
                num3 = u[uAt + RowStart(k) + n];
                uint num6 = u[uAt + RowStart(k + 1) + n];
                if (num3 <= i && i < num6)
                {
                    i -= num3;
                    y[yAt + num++] = 0;
                }
                else
                {
                    num4 = 0 - ((i >= num6) ? 1 : 0);
                    i -= num6 & (uint)num4;
                    num5 = k;
                    do
                    {
                        num3 = u[uAt + RowStart(--k) + n];
                    }
                    while (num3 > i);
                    i -= num3;
                    num7 = (short)((num5 - k + num4) ^ num4);
                    y[yAt + num++] = num7;
                    c = MAC16_16(c, num7, num7);
                }
            }
            n--;
        }
        num3 = (uint)(2 * k + 1);
        num4 = 0 - ((i >= num3) ? 1 : 0);
        i -= num3 & (uint)num4;
        num5 = k;
        k = (int)(i + 1 >> 1);
        if (k != 0)
        {
            i -= (uint)(2 * k - 1);
        }
        num7 = (short)((num5 - k + num4) ^ num4);
        y[yAt + num++] = num7;
        c = MAC16_16(c, num7, num7);
        num4 = (int)(0 - i);
        num7 = (short)(y[yAt + num] = (short)((k + num4) ^ num4));
        return MAC16_16(c, num7, num7);
    }

    /// <summary>Decode a pulse vector (libopus decode_pulses): index with <c>ec_dec_uint(V(n, k))</c>, then <see cref="Cwrsi"/>.</summary>
    public static int DecodePulses(
        ArrayView<int> y, int yAt, int n, int k,
        ref OpusRangeDecoderGpuState dec, ArrayView<byte> buf, int bufStart, uint storage,
        ArrayView<uint> u, int uAt)
    {
        uint index = OpusRangeDecoderGpu.DecodeUint(ref dec, buf, bufStart, storage, PvqV(u, uAt, n, k));
        return Cwrsi(n, k, index, y, yAt, u, uAt);
    }

    /// <summary>One Givens-rotation pass of the spreading rotation (libopus exp_rotation1).</summary>
    public static void ExpRotation1(ArrayView<int> x, int xAt, int len, int stride, int c, int s)
    {
        int num = xAt;
        int a = NEG16(s);
        for (int i = 0; i < len - stride; i++)
        {
            int b = x[num];
            int b2 = x[num + stride];
            x[num + stride] = EXTRACT16(PSHR32(MAC16_16(MULT16_16(c, b2), s, b), 15));
            x[num] = EXTRACT16(PSHR32(MAC16_16(MULT16_16(c, b), a, b2), 15));
            num++;
        }
        num = xAt + (len - 2 * stride - 1);
        for (int i = len - 2 * stride - 1; i >= 0; i--)
        {
            int b3 = x[num];
            int b4 = x[num + stride];
            x[num + stride] = EXTRACT16(PSHR32(MAC16_16(MULT16_16(c, b4), s, b3), 15));
            x[num] = EXTRACT16(PSHR32(MAC16_16(MULT16_16(c, b3), a, b4), 15));
            num--;
        }
    }

    /// <summary>
    /// The spreading rotation (libopus exp_rotation): <paramref name="dir"/> 1 before quantization, -1 after dequantization.
    /// <paramref name="spread"/> is the frame's spread decision (0 none .. 3 aggressive); nothing happens when
    /// <c>2 * K &gt;= len</c> or <c>spread == 0</c>.
    /// </summary>
    public static void ExpRotation(ArrayView<int> x, int xAt, int len, int dir, int stride, int K, int spread)
    {
        int i = 0;
        if (2 * K >= len || spread == 0)
        {
            return;
        }
        int num = spread == 1 ? 15 : spread == 2 ? 10 : 5;   // SPREAD_FACTOR[spread - 1]
        int num2 = celt_div(MULT16_16(32767, len), len + num * K);
        int num3 = HALF16(MULT16_16_Q15(num2, num2));
        int num4 = celt_cos_norm(EXTEND32(num3));
        int num5 = celt_cos_norm(EXTEND32(SUB16(32767, num3)));
        if (len >= 8 * stride)
        {
            for (i = 1; (i * i + i) * stride + (stride >> 2) < len; i++)
            {
            }
        }
        len = celt_udiv(len, stride);
        for (int j = 0; j < stride; j++)
        {
            if (dir < 0)
            {
                if (i != 0)
                {
                    ExpRotation1(x, xAt + j * len, len, i, num5, num4);
                }
                ExpRotation1(x, xAt + j * len, len, 1, num4, num5);
            }
            else
            {
                ExpRotation1(x, xAt + j * len, len, 1, num4, (short)(-num5));
                if (i != 0)
                {
                    ExpRotation1(x, xAt + j * len, len, i, num5, (short)(-num4));
                }
            }
        }
    }

    /// <summary>Scale the decoded pulses <paramref name="iy"/> to a unit-norm band shape times <paramref name="gain"/> (libopus normalise_residual).</summary>
    public static void NormaliseResidual(ArrayView<int> iy, int iyAt, ArrayView<int> x, int xAt, int n, int ryy, int gain)
    {
        int num = celt_ilog2(ryy) >> 1;
        int a = MULT16_16_P15(celt_rsqrt_norm(VSHR32(ryy, 2 * (num - 7))), gain);
        int num2 = 0;
        do
        {
            x[xAt + num2] = EXTRACT16(PSHR32(MULT16_16(a, iy[iyAt + num2]), num + 1));
        }
        while (++num2 < n);
    }

    /// <summary>Bit b set when sub-block b of the <paramref name="b"/> interleaved blocks got a pulse (libopus extract_collapse_mask).</summary>
    public static uint ExtractCollapseMask(ArrayView<int> iy, int iyAt, int n, int b)
    {
        if (b <= 1)
        {
            return 1u;
        }
        int num = celt_udiv(n, b);
        uint num2 = 0u;
        int num3 = 0;
        do
        {
            uint num4 = 0u;
            int num5 = 0;
            do
            {
                num4 |= (uint)iy[iyAt + num3 * num + num5];
            }
            while (++num5 < num);
            num2 |= ((num4 != 0) ? 1u : 0u) << num3;
        }
        while (++num3 < b);
        return num2;
    }

    /// <summary>
    /// Decode a band shape (libopus alg_unquant): the pulse vector from the range coder, scaled to <paramref name="gain"/> and
    /// rotated back. Writes <paramref name="n"/> values at <paramref name="xAt"/>; <paramref name="iy"/> is scratch of
    /// <paramref name="n"/> ints. Returns the collapse mask.
    /// </summary>
    public static uint AlgUnquant(
        ArrayView<int> x, int xAt, int n, int k, int spread, int b,
        ref OpusRangeDecoderGpuState dec, ArrayView<byte> buf, int bufStart, uint storage,
        int gain, ArrayView<int> iy, int iyAt, ArrayView<uint> u, int uAt)
    {
        int ryy = DecodePulses(iy, iyAt, n, k, ref dec, buf, bufStart, storage, u, uAt);
        NormaliseResidual(iy, iyAt, x, xAt, n, ryy, gain);
        ExpRotation(x, xAt, n, -1, b, k, spread);
        return ExtractCollapseMask(iy, iyAt, n, b);
    }

    /// <summary>
    /// The PVQ search of libopus alg_quant: rotates the band shape at <paramref name="xAt"/>, finds the <paramref name="k"/>-pulse
    /// vector closest to it and writes it (signed) to <paramref name="iy"/>; leaves |x| * sign in x as libopus does. Returns the
    /// collapse mask. The caller codes the result: <c>ec_enc_uint(Icwrs(n, iy), PvqV(n, k))</c>.
    /// Scratch: <paramref name="y"/> and <paramref name="signx"/>, <paramref name="n"/> ints each.
    /// </summary>
    public static uint AlgQuantSearch(
        ArrayView<int> x, int xAt, int n, int k, int spread, int b,
        ArrayView<int> iy, int iyAt, ArrayView<int> y, int yAt, ArrayView<int> signx, int signxAt)
    {
        ExpRotation(x, xAt, n, 1, b, k, spread);
        int num = 0;
        int num2 = 0;
        do
        {
            int index = xAt + num2;
            signx[signxAt + num2] = ((x[index] > 0) ? 1 : (-1));
            x[index] = ABS16(x[index]);
            iy[iyAt + num2] = 0;
            y[yAt + num2] = 0;
        }
        while (++num2 < n);
        int num4;
        int num3 = (num4 = 0);
        int num5 = k;
        if (k > n >> 1)
        {
            num2 = 0;
            do
            {
                num += x[xAt + num2];
            }
            while (++num2 < n);
            if (num <= k)
            {
                x[xAt] = 16384;
                num2 = xAt + 1;
                do
                {
                    x[num2] = 0;
                }
                while (++num2 < n + xAt);
                num = 16384;
            }
            int b0 = EXTRACT16(MULT16_32_Q16(k - 1, celt_rcp(num)));
            num2 = 0;
            do
            {
                iy[iyAt + num2] = MULT16_16_Q15(x[xAt + num2], b0);
                y[yAt + num2] = iy[iyAt + num2];
                num4 = MAC16_16(num4, y[yAt + num2], y[yAt + num2]);
                num3 = MAC16_16(num3, x[xAt + num2], y[yAt + num2]);
                y[yAt + num2] *= 2;
                num5 -= iy[iyAt + num2];
            }
            while (++num2 < n);
        }
        if (num5 > n + 3)
        {
            int num6 = num5;
            num4 = MAC16_16(num4, num6, num6);
            num4 = MAC16_16(num4, num6, y[yAt]);
            iy[iyAt] += num5;
            num5 = 0;
        }
        int num7 = 1;
        for (int i = 0; i < num5; i++)
        {
            int b2 = -32767;
            int a = 0;
            int shift = 1 + celt_ilog2(k - num5 + i + 1);
            int num8 = 0;
            num4 = ADD16(num4, 1);
            num2 = 0;
            do
            {
                int num9 = EXTRACT16(SHR32(ADD32(num3, EXTEND32(x[xAt + num2])), shift));
                int num10 = ADD16(num4, y[yAt + num2]);
                num9 = MULT16_16_Q15(num9, num9);
                if (MULT16_16(a, num9) > MULT16_16(num10, b2))
                {
                    a = num10;
                    b2 = num9;
                    num8 = num2;
                }
            }
            while (++num2 < n);
            num3 = ADD32(num3, EXTEND32(x[xAt + num8]));
            num4 = ADD16(num4, y[yAt + num8]);
            y[yAt + num8] += 2 * num7;
            iy[iyAt + num8]++;
        }
        num2 = 0;
        do
        {
            x[xAt + num2] = MULT16_16(signx[signxAt + num2], x[xAt + num2]);
            iy[iyAt + num2] = ((signx[signxAt + num2] < 0) ? (-iy[iyAt + num2]) : iy[iyAt + num2]);
        }
        while (++num2 < n);
        return ExtractCollapseMask(iy, iyAt, n, b);
    }

    /// <summary>Rescale <paramref name="n"/> values at <paramref name="xAt"/> to unit norm times <paramref name="gain"/> (libopus renormalise_vector).</summary>
    public static void RenormaliseVector(ArrayView<int> x, int xAt, int n, int gain)
    {
        int num = 1 + InnerProduct(x, xAt, x, xAt, n);
        int num2 = celt_ilog2(num) >> 1;
        int a = MULT16_16_P15(celt_rsqrt_norm(VSHR32(num, 2 * (num2 - 7))), gain);
        int num3 = xAt;
        for (int i = 0; i < n; i++)
        {
            x[num3] = EXTRACT16(PSHR32(MULT16_16(a, x[num3]), num2 + 1));
            num3++;
        }
    }

    /// <summary>The stereo angle of a band pair, Q14 of pi/2 (libopus stereo_itheta).</summary>
    public static int StereoItheta(ArrayView<int> x, int xAt, ArrayView<int> y, int yAt, int stereo, int n)
    {
        int num2;
        int num = (num2 = 1);
        if (stereo != 0)
        {
            for (int i = 0; i < n; i++)
            {
                int num3 = ADD16(SHR16(x[xAt + i], 1), SHR16(y[yAt + i], 1));
                int num4 = SUB16(SHR16(x[xAt + i], 1), SHR16(y[yAt + i], 1));
                num = MAC16_16(num, num3, num3);
                num2 = MAC16_16(num2, num4, num4);
            }
        }
        else
        {
            num += InnerProduct(x, xAt, x, xAt, n);
            num2 += InnerProduct(y, yAt, y, yAt, n);
        }
        int x2 = celt_sqrt(num);
        int y2 = celt_sqrt(num2);
        return MULT16_16_Q15(20861, celt_atan2p(y2, x2));
    }

    /// <summary>libopus celt_inner_prod (the int-norm overload): sum of x[i] * y[i].</summary>
    public static int InnerProduct(ArrayView<int> x, int xAt, ArrayView<int> y, int yAt, int n)
    {
        int num = 0;
        for (int i = 0; i < n; i++)
        {
            num = MAC16_16(num, x[xAt + i], y[yAt + i]);
        }
        return num;
    }
}
