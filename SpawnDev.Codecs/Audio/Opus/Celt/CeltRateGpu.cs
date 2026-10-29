// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt). See NOTICE.md for upstream attributions.
//
// CELT bit allocation on the accelerator (libopus celt/rate.c + init_caps from celt_decoder.c / celt_encoder.c), fixed-point
// build, decode side: how many bits each band's shape (PVQ) and fine energy get, plus the skip / intensity / dual-stereo
// decisions read from the range coder. Ported statement for statement from Concentus 2.2.2 (BSD-3) so the results are bit-exact
// with it; the mode tables come from CeltModeGpu (the static 48 kHz mode).
//
// GPU shape: the per-band arrays (offsets, cap, pulses, ebits, fine_priority) are views that share one start offset `at`
// (CeltModeTables.NbEBands ints each); libopus' four stack arrays are one caller scratch view of 4 * NbEBands ints. The
// encoder variant (the same decisions written with enc_bit_logp / enc_uint) follows with the GPU range encoder.

using ILGPU;
using static SpawnDev.Codecs.Audio.Opus.Celt.CeltFixed;

namespace SpawnDev.Codecs.Audio.Opus.Celt;

/// <summary>CELT bit allocation (rate.c), decode side (GPU-callable).</summary>
public static class CeltRateGpu
{
    /// <summary>Pulse count of pulse-cache entry <paramref name="i"/> (libopus get_pulses).</summary>
    public static int GetPulses(int i)
    {
        if (i >= 8)
        {
            return 8 + (i & 7) << (i >> 3) - 1;
        }
        return i;
    }

    /// <summary>The pulse-cache entry that spends closest to <paramref name="bits"/> (1/8 bit) on band <paramref name="band"/> (libopus bits2pulses).</summary>
    public static int Bits2Pulses(CeltModeGpu m, int band, int LM, int bits)
    {
        LM++;
        int num = m.CacheIndexOf(LM * CeltModeTables.NbEBands + band);
        int num2 = 0;
        int num3 = m.CacheBitsOf(num);
        bits--;
        for (int i = 0; i < 6; i++)
        {
            int num4 = num2 + num3 + 1 >> 1;
            if (m.CacheBitsOf(num + num4) >= bits)
            {
                num3 = num4;
            }
            else
            {
                num2 = num4;
            }
        }
        if (bits - ((num2 == 0) ? (-1) : m.CacheBitsOf(num + num2)) <= m.CacheBitsOf(num + num3) - bits)
        {
            return num2;
        }
        return num3;
    }

    /// <summary>Bits (1/8 bit) pulse-cache entry <paramref name="pulses"/> spends on band <paramref name="band"/> (libopus pulses2bits).</summary>
    public static int Pulses2Bits(CeltModeGpu m, int band, int LM, int pulses)
    {
        LM++;
        if (pulses != 0)
        {
            return m.CacheBitsOf(m.CacheIndexOf(LM * CeltModeTables.NbEBands + band) + pulses) + 1;
        }
        return 0;
    }

    /// <summary>Per-band bit caps for a frame of 120 &lt;&lt; LM samples and <paramref name="C"/> channels (libopus init_caps).</summary>
    public static void InitCaps(CeltModeGpu m, ArrayView<int> cap, int at, int LM, int C)
    {
        for (int i = 0; i < CeltModeTables.NbEBands; i++)
        {
            int num = m.EBandsOf(i + 1) - m.EBandsOf(i) << LM;
            cap[at + i] = (m.CacheCapsOf(CeltModeTables.NbEBands * (2 * LM + C - 1) + i) + 64) * C * num >> 2;
        }
    }

    /// <summary>
    /// libopus interp_bits2pulses (decoder): interpolates between two allocation vectors, reads the skip / intensity / dual-stereo
    /// decisions, and splits each band's bits into shape (<paramref name="bits"/>) and fine energy (<paramref name="ebits"/>).
    /// <paramref name="bits1"/> / <paramref name="bits2"/> / <paramref name="thresh"/> live in scratch at
    /// <paramref name="scratchAt"/>, +NbEBands and +2 * NbEBands. Returns the coded band count.
    /// </summary>
    public static int InterpBits2PulsesDecode(
        CeltModeGpu m, int start, int end, int skip_start,
        ArrayView<int> scratch, int scratchAt,
        ArrayView<int> cap, int total, out int _balance, int skip_rsv, ref int intensity, int intensity_rsv,
        ref int dual_stereo, int dual_stereo_rsv,
        ArrayView<int> bits, ArrayView<int> ebits, ArrayView<int> fine_priority, int at,
        int C, int LM,
        ref OpusRangeDecoderGpuState ec, ArrayView<byte> buf, int bufStart, uint storage,
        int prev, int signalBandwidth)
    {
        int bits1At = scratchAt, bits2At = scratchAt + CeltModeTables.NbEBands, threshAt = scratchAt + 2 * CeltModeTables.NbEBands;
        int num = -1;
        int num2 = C << 3;
        int num3 = ((C > 1) ? 1 : 0);
        int num4 = LM << 3;
        int num5 = 0;
        int num6 = 64;
        int num9;
        int num8;
        int num10;
        for (int i = 0; i < 6; i++)
        {
            int num7 = num5 + num6 >> 1;
            num8 = 0;
            num9 = 0;
            num10 = end;
            while (num10-- > start)
            {
                int num11 = scratch[bits1At + num10] + (num7 * scratch[bits2At + num10] >> 6);
                if (num11 >= scratch[threshAt + num10] || num9 != 0)
                {
                    num9 = 1;
                    num8 += IMIN(num11, cap[at + num10]);
                }
                else if (num11 >= num2)
                {
                    num8 += num2;
                }
            }
            if (num8 > total)
            {
                num6 = num7;
            }
            else
            {
                num5 = num7;
            }
        }
        num8 = 0;
        num9 = 0;
        num10 = end;
        while (num10-- > start)
        {
            int num12 = scratch[bits1At + num10] + (num5 * scratch[bits2At + num10] >> 6);
            if (num12 < scratch[threshAt + num10] && num9 == 0)
            {
                num12 = ((num12 >= num2) ? num2 : 0);
            }
            else
            {
                num9 = 1;
            }
            num8 += (bits[at + num10] = IMIN(num12, cap[at + num10]));
        }
        num = end;
        int num14;
        int num13;
        while (true)
        {
            num10 = num - 1;
            if (num10 <= skip_start)
            {
                total += skip_rsv;
                break;
            }
            num13 = total - num8;
            num14 = celt_udiv(num13, m.EBandsOf(num) - m.EBandsOf(start));
            num13 -= (m.EBandsOf(num) - m.EBandsOf(start)) * num14;
            int num15 = IMAX(num13 - (m.EBandsOf(num10) - m.EBandsOf(start)), 0);
            int num16 = m.EBandsOf(num) - m.EBandsOf(num10);
            int num17 = bits[at + num10] + num14 * num16 + num15;
            if (num17 >= IMAX(scratch[threshAt + num10], num2 + 8))
            {
                if (OpusRangeDecoderGpu.DecodeBitLogP(ref ec, buf, bufStart, storage, 1) != 0)
                {
                    break;
                }
                num8 += 8;
                num17 -= 8;
            }
            num8 -= bits[at + num10] + intensity_rsv;
            if (intensity_rsv > 0)
            {
                intensity_rsv = m.Log2FracTableOf(num10 - start);
            }
            num8 += intensity_rsv;
            if (num17 >= num2)
            {
                num8 += num2;
                bits[at + num10] = num2;
            }
            else
            {
                bits[at + num10] = 0;
            }
            num--;
        }
        if (intensity_rsv > 0)
        {
            intensity = start + (int)OpusRangeDecoderGpu.DecodeUint(ref ec, buf, bufStart, storage, (uint)(num + 1 - start));
        }
        else
        {
            intensity = 0;
        }
        if (intensity <= start)
        {
            total += dual_stereo_rsv;
            dual_stereo_rsv = 0;
        }
        if (dual_stereo_rsv > 0)
        {
            dual_stereo = OpusRangeDecoderGpu.DecodeBitLogP(ref ec, buf, bufStart, storage, 1);
        }
        else
        {
            dual_stereo = 0;
        }
        num13 = total - num8;
        num14 = celt_udiv(num13, m.EBandsOf(num) - m.EBandsOf(start));
        num13 -= (m.EBandsOf(num) - m.EBandsOf(start)) * num14;
        for (num10 = start; num10 < num; num10++)
        {
            bits[at + num10] += num14 * (m.EBandsOf(num10 + 1) - m.EBandsOf(num10));
        }
        for (num10 = start; num10 < num; num10++)
        {
            int num18 = IMIN(num13, m.EBandsOf(num10 + 1) - m.EBandsOf(num10));
            bits[at + num10] += num18;
            num13 -= num18;
        }
        int num19 = 0;
        for (num10 = start; num10 < num; num10++)
        {
            int num20 = m.EBandsOf(num10 + 1) - m.EBandsOf(num10) << LM;
            int num21 = bits[at + num10] + num19;
            int num22;
            if (num20 > 1)
            {
                num22 = MAX32(num21 - cap[at + num10], 0);
                bits[at + num10] = num21 - num22;
                int num23 = C * num20 + ((C == 2 && num20 > 2 && dual_stereo == 0 && num10 < intensity) ? 1 : 0);
                int num24 = num23 * (m.LogNOf(num10) + num4);
                int num25 = (num24 >> 1) - num23 * 21;
                if (num20 == 2)
                {
                    num25 += num23 << 3 >> 2;
                }
                if (bits[at + num10] + num25 < num23 * 2 << 3)
                {
                    num25 += num24 >> 2;
                }
                else if (bits[at + num10] + num25 < num23 * 3 << 3)
                {
                    num25 += num24 >> 3;
                }
                ebits[at + num10] = IMAX(0, bits[at + num10] + num25 + (num23 << 2));
                ebits[at + num10] = celt_udiv(ebits[at + num10], num23) >> 3;
                if (C * ebits[at + num10] > bits[at + num10] >> 3)
                {
                    ebits[at + num10] = bits[at + num10] >> num3 >> 3;
                }
                ebits[at + num10] = IMIN(ebits[at + num10], 8);
                fine_priority[at + num10] = ((ebits[at + num10] * (num23 << 3) >= bits[at + num10] + num25) ? 1 : 0);
                bits[at + num10] -= C * ebits[at + num10] << 3;
            }
            else
            {
                num22 = MAX32(0, num21 - (C << 3));
                bits[at + num10] = num21 - num22;
                ebits[at + num10] = 0;
                fine_priority[at + num10] = 1;
            }
            if (num22 > 0)
            {
                int num26 = IMIN(num22 >> num3 + 3, 8 - ebits[at + num10]);
                ebits[at + num10] += num26;
                int num27 = num26 * C << 3;
                fine_priority[at + num10] = ((num27 >= num22 - num19) ? 1 : 0);
                num22 -= num27;
            }
            num19 = num22;
        }
        _balance = num19;
        for (; num10 < end; num10++)
        {
            ebits[at + num10] = bits[at + num10] >> num3 >> 3;
            bits[at + num10] = 0;
            fine_priority[at + num10] = ((ebits[at + num10] < 1) ? 1 : 0);
        }
        return num;
    }

    /// <summary>
    /// libopus compute_allocation (decoder): the per-band shape bits (<paramref name="pulses"/>, 1/8 bit), fine-energy bits
    /// (<paramref name="ebits"/>) and fine priorities, reading the allocation decisions from the range coder. All per-band
    /// views share <paramref name="at"/>; <paramref name="scratch"/> holds 4 * NbEBands ints at <paramref name="scratchAt"/>.
    /// Returns the coded band count; <paramref name="balance"/>, <paramref name="intensity"/> and
    /// <paramref name="dual_stereo"/> come back as libopus returns them.
    /// </summary>
    public static int ComputeAllocationDecode(
        CeltModeGpu m, int start, int end, ArrayView<int> offsets, ArrayView<int> cap, int alloc_trim,
        ref int intensity, ref int dual_stereo, int total, out int balance,
        ArrayView<int> pulses, ArrayView<int> ebits, ArrayView<int> fine_priority, int at,
        ArrayView<int> scratch, int scratchAt,
        int C, int LM,
        ref OpusRangeDecoderGpuState ec, ArrayView<byte> buf, int bufStart, uint storage,
        int prev, int signalBandwidth)
    {
        int nbEBands = CeltModeTables.NbEBands;
        int bits1At = scratchAt, bits2At = scratchAt + nbEBands, threshAt = scratchAt + 2 * nbEBands, trimAt = scratchAt + 3 * nbEBands;
        total = IMAX(total, 0);
        int skip_start = start;
        int num = ((total >= 8) ? 8 : 0);
        total -= num;
        int num3;
        int num2 = (num3 = 0);
        if (C == 2)
        {
            num2 = m.Log2FracTableOf(end - start);
            if (num2 > total)
            {
                num2 = 0;
            }
            else
            {
                total -= num2;
                num3 = ((total >= 8) ? 8 : 0);
                total -= num3;
            }
        }
        for (int i = 0; i < nbEBands; i++)
        {
            scratch[bits1At + i] = 0;
            scratch[bits2At + i] = 0;
            scratch[threshAt + i] = 0;
            scratch[trimAt + i] = 0;
        }
        for (int i = start; i < end; i++)
        {
            scratch[threshAt + i] = IMAX(C << 3, 3 * (m.EBandsOf(i + 1) - m.EBandsOf(i)) << LM << 3 >> 4);
            scratch[trimAt + i] = C * (m.EBandsOf(i + 1) - m.EBandsOf(i)) * (alloc_trim - 5 - LM) * (end - i - 1) * (1 << LM + 3) >> 6;
            if (m.EBandsOf(i + 1) - m.EBandsOf(i) << LM == 1)
            {
                scratch[trimAt + i] -= C << 3;
            }
        }
        int num4 = 1;
        int num5 = CeltModeTables.NbAllocVectors - 1;
        do
        {
            int num6 = 0;
            int num7 = 0;
            int num8 = num4 + num5 >> 1;
            int i = end;
            while (i-- > start)
            {
                int num9 = m.EBandsOf(i + 1) - m.EBandsOf(i);
                int num10 = C * num9 * m.AllocVectorsOf(num8 * nbEBands + i) << LM >> 2;
                if (num10 > 0)
                {
                    num10 = IMAX(0, num10 + scratch[trimAt + i]);
                }
                num10 += offsets[at + i];
                if (num10 >= scratch[threshAt + i] || num6 != 0)
                {
                    num6 = 1;
                    num7 += IMIN(num10, cap[at + i]);
                }
                else if (num10 >= C << 3)
                {
                    num7 += C << 3;
                }
            }
            if (num7 > total)
            {
                num5 = num8 - 1;
            }
            else
            {
                num4 = num8 + 1;
            }
        }
        while (num4 <= num5);
        num5 = num4--;
        for (int i = start; i < end; i++)
        {
            int num11 = m.EBandsOf(i + 1) - m.EBandsOf(i);
            int num12 = C * num11 * m.AllocVectorsOf(num4 * nbEBands + i) << LM >> 2;
            int num13 = ((num5 >= CeltModeTables.NbAllocVectors) ? cap[at + i] : (C * num11 * m.AllocVectorsOf(num5 * nbEBands + i) << LM >> 2));
            if (num12 > 0)
            {
                num12 = IMAX(0, num12 + scratch[trimAt + i]);
            }
            if (num13 > 0)
            {
                num13 = IMAX(0, num13 + scratch[trimAt + i]);
            }
            if (num4 > 0)
            {
                num12 += offsets[at + i];
            }
            num13 += offsets[at + i];
            if (offsets[at + i] > 0)
            {
                skip_start = i;
            }
            num13 = IMAX(0, num13 - num12);
            scratch[bits1At + i] = num12;
            scratch[bits2At + i] = num13;
        }
        return InterpBits2PulsesDecode(m, start, end, skip_start, scratch, scratchAt, cap, total, out balance, num, ref intensity,
            num2, ref dual_stereo, num3, pulses, ebits, fine_priority, at, C, LM, ref ec, buf, bufStart, storage, prev, signalBandwidth);
    }
}
