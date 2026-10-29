// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt). See NOTICE.md for upstream attributions.
//
// The CELT band recursion on the accelerator, decode side (libopus celt/bands.c quant_all_bands, quant_band,
// quant_band_stereo, quant_band_n1, quant_partition, compute_theta), fixed-point build: per band, the shape bits are split
// recursively (an angle theta decides how they divide between the halves / between mid and side) down to PVQ codes, with
// folding from lower bands where a band gets no pulses. Ported statement for statement from Concentus 2.2.2 (BSD-3, libopus
// ~1.1) and brought to libopus 1.5.2: disable_inv, the hybrid second-band folding (special_hybrid_folding, the lowband_offset
// and fold_end changes) and, through CeltFixed, 1.5.2's 64-bit MULT32_32_Q31 in exp_rotation's gain.
//
// GPU shape:
// - X_, Y_ and libopus' normalised-band history `_norm` live in ONE int view (`buf`) at caller offsets, so every libopus
//   Span (X, Y, lowband, lowband_out, lowband_scratch, which may be null) is an offset into `buf`, -1 = null. libopus'
//   lowband_scratch points into X_ past the last band's start, exactly as in Concentus.
// - quant_partition recurses; ILGPU's GPU backends do not, so it runs on an explicit stack of frames (CeltBandFrame layout,
//   at most PartitionDepth deep: every split halves N and lowers LM) in caller scratch; each frame resumes after its
//   first and after its second child, combining their collapse masks in libopus order.
// - The range decoder is OpusRangeDecoderGpu + CeltEntropyGpu (ec_decode, ec_dec_update, ec_tell_frac).

using ILGPU;
using static SpawnDev.Codecs.Audio.Opus.Celt.CeltFixed;

namespace SpawnDev.Codecs.Audio.Opus.Celt;

/// <summary>libopus band_ctx (decode): the band being coded and the bits / noise state shared down its recursion.</summary>
public struct CeltBandCtx
{
    /// <summary>Band index.</summary>
    public int I;
    /// <summary>First intensity-stereo band.</summary>
    public int Intensity;
    /// <summary>Spread decision.</summary>
    public int Spread;
    /// <summary>TF change of the band.</summary>
    public int TfChange;
    /// <summary>Bits left in the frame, 1/8 bit.</summary>
    public int RemainingBits;
    /// <summary>The folding noise seed.</summary>
    public uint Seed;
    /// <summary>1 forces the stereo inversion flag off (libopus disable_inv: a mono decoder of a stereo stream).</summary>
    public int DisableInv;
}

/// <summary>The CELT band recursion (bands.c quant_all_bands and its callees), decode side (GPU-callable).</summary>
public static class CeltQuantBandsGpu
{
    /// <summary>Maximum quant_partition depth (each split halves N and lowers LM; LM starts at most at 3).</summary>
    public const int PartitionDepth = 8;
    /// <summary>Ints per quant_partition frame.</summary>
    public const int FrameInts = 20;
    /// <summary>Largest band size N (8 short blocks x the widest 2.5 ms band, 22 bins).</summary>
    public const int MaxBandN = 176;
    /// <summary>Scratch ints one QuantAllBandsDecode call needs: the partition stack, the PVQ pulses, the interleave buffer.</summary>
    public const int ScratchInts = PartitionDepth * FrameInts + 2 * MaxBandN;

    // frame fields
    private const int FX = 0, FN = 1, FBits = 2, FB = 3, FLow = 4, FLM = 5, FGain = 6, FFill = 7, FStage = 8, FMidFirst = 9,
        FB0 = 10, FXSide = 11, FLow2 = 12, FRem = 13, FItheta = 14, FImid = 15, FIside = 16, FMbits = 17, FSbits = 18, FResult = 19;

    private static int BitInterleave(int i)
    {
        switch (i)
        {
            case 0: return 0;
            case 1: case 2: case 3: return i == 1 ? 1 : 1;
            case 4: return 2;
            case 5: case 6: case 7: return 3;
            case 8: return 2;
            case 9: case 10: case 11: return 3;
            case 12: return 2;
            default: return 3;
        }
    }

    private static uint BitDeinterleave(uint i)
    {
        switch (i)
        {
            case 0: return 0;
            case 1: return 3;
            case 2: return 12;
            case 3: return 15;
            case 4: return 48;
            case 5: return 51;
            case 6: return 60;
            case 7: return 63;
            case 8: return 192;
            case 9: return 195;
            case 10: return 204;
            case 11: return 207;
            case 12: return 240;
            case 13: return 243;
            case 14: return 252;
            default: return 255;
        }
    }

    /// <summary>libopus compute_theta (decoder): the split angle of a band (or a mid / side pair) and how many bits it took.</summary>
    public static void ComputeThetaDecode(CeltModeGpu m, ref CeltBandCtx ctx,
        ref OpusRangeDecoderGpuState ec, ArrayView<byte> pkt, int pktAt, uint storage,
        int N, ref int b, int B, int B0, int LM, int stereo, ref int fill,
        out int sInv, out int sImid, out int sIside, out int sDelta, out int sItheta, out int sQalloc)
    {
        int num = 0;
        int inv = 0;
        int i = ctx.I;
        int intensity = ctx.Intensity;
        int num2 = m.LogNOf(i) + LM * 8;
        int offset = (num2 >> 1) - ((stereo != 0 && N == 2) ? 16 : 4);
        int num3 = CeltBandsGpu.ComputeQn(N, b, offset, num2, stereo);
        if (stereo != 0 && i >= intensity)
        {
            num3 = 1;
        }
        int num4 = (int)CeltEntropyGpu.TellFrac(ref ec);
        if (num3 != 1)
        {
            if (stereo != 0 && N > 2)
            {
                int num5 = 3;
                int num6 = num;
                int num7 = num3 / 2;
                uint ft = (uint)(num5 * (num7 + 1) + num7);
                int num8 = (int)CeltEntropyGpu.Decode(ref ec, ft);
                num6 = ((num8 >= (num7 + 1) * num5) ? (num7 + 1 + (num8 - (num7 + 1) * num5)) : (num8 / num5));
                CeltEntropyGpu.DecUpdate(ref ec, pkt, pktAt, storage,
                    (uint)((num6 <= num7) ? (num5 * num6) : (num6 - 1 - num7 + (num7 + 1) * num5)),
                    (uint)((num6 <= num7) ? (num5 * (num6 + 1)) : (num6 - num7 + (num7 + 1) * num5)), ft);
                num = num6;
            }
            else if (B0 > 1 || stereo != 0)
            {
                num = (int)OpusRangeDecoderGpu.DecodeUint(ref ec, pkt, pktAt, storage, (uint)(num3 + 1));
            }
            else
            {
                int num9 = 1;
                int num10 = ((num3 >> 1) + 1) * ((num3 >> 1) + 1);
                int num11 = 0;
                int num12 = (int)CeltEntropyGpu.Decode(ref ec, (uint)num10);
                if (num12 < (num3 >> 1) * ((num3 >> 1) + 1) >> 1)
                {
                    num = (int)(CeltEntropyGpu.Isqrt32((uint)(8 * num12 + 1)) - 1) >> 1;
                    num9 = num + 1;
                    num11 = num * (num + 1) >> 1;
                }
                else
                {
                    num = (int)(2 * (num3 + 1) - CeltEntropyGpu.Isqrt32((uint)(8 * (num10 - num12 - 1) + 1))) >> 1;
                    num9 = num3 + 1 - num;
                    num11 = num10 - ((num3 + 1 - num) * (num3 + 2 - num) >> 1);
                }
                CeltEntropyGpu.DecUpdate(ref ec, pkt, pktAt, storage, (uint)num11, (uint)(num11 + num9), (uint)num10);
            }
            num = celt_udiv(num * 16384, num3);
        }
        else if (stereo != 0)
        {
            inv = ((b > 16 && ctx.RemainingBits > 16) ? OpusRangeDecoderGpu.DecodeBitLogP(ref ec, pkt, pktAt, storage, 2) : 0);
            // inv flag override to avoid problems with downmixing (libopus 1.5.2)
            if (ctx.DisableInv != 0) inv = 0;
            num = 0;
        }
        int num13 = (int)CeltEntropyGpu.TellFrac(ref ec) - num4;
        b -= num13;
        int num14;
        int num15;
        int delta;
        if (num == 0)
        {
            num14 = 32767;
            num15 = 0;
            fill &= (1 << B) - 1;
            delta = -16384;
        }
        else if (num == 16384)
        {
            num14 = 0;
            num15 = 32767;
            fill &= (1 << B) - 1 << B;
            delta = 16384;
        }
        else
        {
            num14 = CeltBandsGpu.BitexactCos((short)num);
            num15 = CeltBandsGpu.BitexactCos((short)(16384 - num));
            delta = FRAC_MUL16(N - 1 << 7, CeltBandsGpu.BitexactLog2tan(num15, num14));
        }
        sInv = inv;
        sImid = num14;
        sIside = num15;
        sDelta = delta;
        sItheta = num;
        sQalloc = num13;
    }

    /// <summary>libopus quant_band_n1 (decoder): a band of one bin per channel is a sign bit.</summary>
    public static uint QuantBandN1Decode(ref CeltBandCtx ctx, ref OpusRangeDecoderGpuState ec, ArrayView<byte> pkt, int pktAt, uint storage,
        ArrayView<int> buf, int xAt, int yAt, int b, int lowbandOutAt)
    {
        int num = 1;
        int index = xAt;
        int num2 = ((yAt >= 0) ? 1 : 0);
        int num3 = 0;
        do
        {
            int num4 = 0;
            if (ctx.RemainingBits >= 8)
            {
                num4 = (int)OpusRangeDecoderGpu.DecodeBits(ref ec, pkt, pktAt, storage, 1);
                ctx.RemainingBits -= 8;
                b -= 8;
            }
            if (num != 0)
            {
                buf[index] = ((num4 != 0) ? (-16384) : 16384);
            }
            index = yAt;
        }
        while (++num3 < 1 + num2);
        if (lowbandOutAt >= 0)
        {
            buf[lowbandOutAt] = SHR16(buf[xAt], 4);
        }
        return 1u;
    }

    /// <summary>
    /// libopus quant_partition (decoder), iterative: decodes N values at <paramref name="xAt"/> with b bits, splitting
    /// recursively. <paramref name="lowbandAt"/> = the folding source (-1 none). Uses the frame stack and the pulse area of
    /// <paramref name="scratch"/> at <paramref name="scratchAt"/>.
    /// </summary>
    public static uint QuantPartitionDecode(CeltModeGpu m, ref CeltBandCtx ctx,
        ref OpusRangeDecoderGpuState ec, ArrayView<byte> pkt, int pktAt, uint storage,
        ArrayView<int> buf, int xAt, int N, int b, int B, int lowbandAt, int LM, int gain, int fill,
        ArrayView<int> scratch, int scratchAt, ArrayView<uint> pvqU, int pvqUAt)
    {
        int iyAt = scratchAt + PartitionDepth * FrameInts;
        int sp = 0;
        uint ret = 0u;
        // push the root
        int f0 = scratchAt;
        scratch[f0 + FX] = xAt; scratch[f0 + FN] = N; scratch[f0 + FBits] = b; scratch[f0 + FB] = B; scratch[f0 + FLow] = lowbandAt;
        scratch[f0 + FLM] = LM; scratch[f0 + FGain] = gain; scratch[f0 + FFill] = fill; scratch[f0 + FStage] = 0;
        sp = 1;
        while (true)
        {
            int f = scratchAt + (sp - 1) * FrameInts;
            int stage = scratch[f + FStage];
            if (stage == 0)
            {
                int fx = scratch[f + FX];
                int fN = scratch[f + FN];
                int fb = scratch[f + FBits];
                int fB = scratch[f + FB];
                int fLow = scratch[f + FLow];
                int fLM = scratch[f + FLM];
                int fGain = scratch[f + FGain];
                int fFill = scratch[f + FFill];
                int num2 = fB;
                int i = ctx.I;
                int num7 = m.CacheIndexOf((fLM + 1) * CeltModeTables.NbEBands + i);
                if (fLM != -1 && fb > m.CacheBitsOf(num7 + m.CacheBitsOf(num7)) + 12 && fN > 2)
                {
                    int lowband_ptr2 = 0;
                    fN >>= 1;
                    int num6 = fx + fN;
                    fLM--;
                    if (fB == 1)
                    {
                        fFill = (fFill & 1) | (fFill << 1);
                    }
                    fB = fB + 1 >> 1;
                    ComputeThetaDecode(m, ref ctx, ref ec, pkt, pktAt, storage, fN, ref fb, fB, num2, fLM, 0, ref fFill,
                        out _, out int imid, out int iside, out int num8, out int itheta, out int qalloc);
                    if (num2 > 1 && (itheta & 0x3FFF) != 0)
                    {
                        num8 = ((itheta <= 8192) ? IMIN(0, num8 + (fN << 3 >> 5 - fLM)) : (num8 - (num8 >> 4 - fLM)));
                    }
                    int num9 = IMAX(0, IMIN(fb, (fb - num8) / 2));
                    int num10 = fb - num9;
                    ctx.RemainingBits -= qalloc;
                    lowband_ptr2 = fLow >= 0 ? fLow + fN : -1;
                    // the frame becomes the split's continuation
                    scratch[f + FN] = fN; scratch[f + FB] = fB; scratch[f + FLM] = fLM; scratch[f + FFill] = fFill;
                    scratch[f + FB0] = num2; scratch[f + FXSide] = num6; scratch[f + FLow2] = lowband_ptr2;
                    scratch[f + FRem] = ctx.RemainingBits; scratch[f + FItheta] = itheta; scratch[f + FImid] = imid;
                    scratch[f + FIside] = iside; scratch[f + FMbits] = num9; scratch[f + FSbits] = num10;
                    scratch[f + FMidFirst] = num9 >= num10 ? 1 : 0;
                    scratch[f + FStage] = 1;
                    int c = f + FrameInts;
                    if (num9 >= num10)
                    {
                        scratch[c + FX] = fx; scratch[c + FN] = fN; scratch[c + FBits] = num9; scratch[c + FB] = fB; scratch[c + FLow] = fLow;
                        scratch[c + FLM] = fLM; scratch[c + FGain] = MULT16_16_P15(fGain, imid); scratch[c + FFill] = fFill;
                    }
                    else
                    {
                        scratch[c + FX] = num6; scratch[c + FN] = fN; scratch[c + FBits] = num10; scratch[c + FB] = fB; scratch[c + FLow] = lowband_ptr2;
                        scratch[c + FLM] = fLM; scratch[c + FGain] = MULT16_16_P15(fGain, iside); scratch[c + FFill] = fFill >> fB;
                    }
                    scratch[c + FStage] = 0;
                    sp++;
                    continue;
                }
                // leaf
                uint result = 0u;
                int num11 = CeltRateGpu.Bits2Pulses(m, i, fLM, fb);
                int num12 = CeltRateGpu.Pulses2Bits(m, i, fLM, num11);
                ctx.RemainingBits -= num12;
                while (ctx.RemainingBits < 0 && num11 > 0)
                {
                    ctx.RemainingBits += num12;
                    num11--;
                    num12 = CeltRateGpu.Pulses2Bits(m, i, fLM, num11);
                    ctx.RemainingBits -= num12;
                }
                if (num11 != 0)
                {
                    int k = CeltRateGpu.GetPulses(num11);
                    result = CeltPvqGpu.AlgUnquant(buf, fx, fN, k, ctx.Spread, fB, ref ec, pkt, pktAt, storage, fGain, scratch, iyAt, pvqU, pvqUAt);
                }
                else
                {
                    uint num13 = (uint)((int)(1L << fB) - 1);
                    fFill &= (int)num13;
                    if (fFill == 0)
                    {
                        for (int j = 0; j < fN; j++) buf[fx + j] = 0;
                    }
                    else
                    {
                        if (fLow < 0)
                        {
                            for (int j = 0; j < fN; j++)
                            {
                                ctx.Seed = CeltBandsGpu.CeltLcgRand(ctx.Seed);
                                buf[fx + j] = (int)ctx.Seed >> 20;
                            }
                            result = num13;
                        }
                        else
                        {
                            for (int j = 0; j < fN; j++)
                            {
                                ctx.Seed = CeltBandsGpu.CeltLcgRand(ctx.Seed);
                                int num14 = 4;
                                num14 = (((ctx.Seed & 0x8000) != 0) ? num14 : (-num14));
                                buf[fx + j] = buf[fLow + j] + num14;
                            }
                            result = (uint)fFill;
                        }
                        CeltPvqGpu.RenormaliseVector(buf, fx, fN, fGain);
                    }
                }
                ret = result;
                sp--;
                if (sp == 0) break;
                continue;
            }
            if (stage == 1)
            {
                // the first child finished with `ret`
                int fN = scratch[f + FN], fB = scratch[f + FB], fLM = scratch[f + FLM], fFill = scratch[f + FFill], fGain = scratch[f + FGain];
                int num2 = scratch[f + FB0], itheta = scratch[f + FItheta];
                int num9 = scratch[f + FMbits], num10 = scratch[f + FSbits];
                int remaining_bits = scratch[f + FRem];
                int c = f + FrameInts;
                if (scratch[f + FMidFirst] != 0)
                {
                    scratch[f + FResult] = (int)ret;
                    remaining_bits = num9 - (remaining_bits - ctx.RemainingBits);
                    if (remaining_bits > 24 && itheta != 0)
                    {
                        num10 += remaining_bits - 24;
                    }
                    scratch[c + FX] = scratch[f + FXSide]; scratch[c + FN] = fN; scratch[c + FBits] = num10; scratch[c + FB] = fB;
                    scratch[c + FLow] = scratch[f + FLow2]; scratch[c + FLM] = fLM; scratch[c + FGain] = MULT16_16_P15(fGain, scratch[f + FIside]);
                    scratch[c + FFill] = fFill >> fB;
                }
                else
                {
                    scratch[f + FResult] = (int)(ret << (num2 >> 1));
                    remaining_bits = num10 - (remaining_bits - ctx.RemainingBits);
                    if (remaining_bits > 24 && itheta != 16384)
                    {
                        num9 += remaining_bits - 24;
                    }
                    scratch[c + FX] = scratch[f + FX]; scratch[c + FN] = fN; scratch[c + FBits] = num9; scratch[c + FB] = fB;
                    scratch[c + FLow] = scratch[f + FLow]; scratch[c + FLM] = fLM; scratch[c + FGain] = MULT16_16_P15(fGain, scratch[f + FImid]);
                    scratch[c + FFill] = fFill;
                }
                scratch[c + FStage] = 0;
                scratch[f + FStage] = 2;
                sp++;
                continue;
            }
            // stage 2: the second child finished with `ret`
            {
                uint result = (uint)scratch[f + FResult];
                if (scratch[f + FMidFirst] != 0) result |= ret << (scratch[f + FB0] >> 1);
                else result |= ret;
                ret = result;
                sp--;
                if (sp == 0) break;
            }
        }
        return ret;
    }

    /// <summary>libopus quant_band (decoder): one band of one channel, with the TF resolution changes around quant_partition.</summary>
    public static uint QuantBandDecode(CeltModeGpu m, ref CeltBandCtx ctx,
        ref OpusRangeDecoderGpuState ec, ArrayView<byte> pkt, int pktAt, uint storage,
        ArrayView<int> buf, int xAt, int N, int b, int B, int lowbandAt, int LM, int lowbandOutAt, int gain, int lowbandScratchAt, int fill,
        ArrayView<int> scratch, int scratchAt, ArrayView<uint> pvqU, int pvqUAt)
    {
        int tmpAt = scratchAt + PartitionDepth * FrameInts + MaxBandN;
        int n = N;
        int num = B;
        int num2 = 0;
        int num3 = 0;
        uint num4 = 0u;
        int num5 = 1;
        int num6 = ctx.TfChange;
        int hadamard = ((num == 1) ? 1 : 0);
        n = celt_udiv(n, B);
        if (N == 1)
        {
            return QuantBandN1Decode(ref ctx, ref ec, pkt, pktAt, storage, buf, xAt, -1, b, lowbandOutAt);
        }
        if (num6 > 0)
        {
            num3 = num6;
        }
        if (lowbandScratchAt >= 0 && lowbandAt >= 0 && (num3 != 0 || ((n & 1) == 0 && num6 < 0) || num > 1))
        {
            for (int j = 0; j < N; j++) buf[lowbandScratchAt + j] = buf[lowbandAt + j];
            lowbandAt = lowbandScratchAt;
        }
        for (int i = 0; i < num3; i++)
        {
            if (lowbandAt >= 0)
            {
                CeltBandsGpu.Haar1(buf, lowbandAt, N >> i, 1 << i);
            }
            fill = BitInterleave(fill & 0xF) | (BitInterleave(fill >> 4) << 2);
        }
        B >>= num3;
        n <<= num3;
        while ((n & 1) == 0 && num6 < 0)
        {
            if (lowbandAt >= 0)
            {
                CeltBandsGpu.Haar1(buf, lowbandAt, n, B);
            }
            fill |= fill << B;
            B <<= 1;
            n >>= 1;
            num2++;
            num6++;
        }
        num = B;
        int num7 = n;
        if (num > 1 && lowbandAt >= 0)
        {
            CeltBandsGpu.DeinterleaveHadamard(m, buf, lowbandAt, n >> num3, num << num3, hadamard, scratch, tmpAt);
        }
        num4 = QuantPartitionDecode(m, ref ctx, ref ec, pkt, pktAt, storage, buf, xAt, N, b, B, lowbandAt, LM, gain, fill,
            scratch, scratchAt, pvqU, pvqUAt);
        if (num5 != 0)
        {
            if (num > 1)
            {
                CeltBandsGpu.InterleaveHadamard(m, buf, xAt, n >> num3, num << num3, hadamard, scratch, tmpAt);
            }
            n = num7;
            B = num;
            for (int i = 0; i < num2; i++)
            {
                B >>= 1;
                n <<= 1;
                num4 |= num4 >> B;
                CeltBandsGpu.Haar1(buf, xAt, n, B);
            }
            for (int i = 0; i < num3; i++)
            {
                num4 = BitDeinterleave(num4);
                CeltBandsGpu.Haar1(buf, xAt, N >> i, 1 << i);
            }
            B <<= num3;
            if (lowbandOutAt >= 0)
            {
                int a = celt_sqrt(SHL32(N, 22));
                for (int j = 0; j < N; j++)
                {
                    buf[lowbandOutAt + j] = MULT16_16_Q15(a, buf[xAt + j]);
                }
            }
            num4 &= (uint)((1 << B) - 1);
        }
        return num4;
    }

    /// <summary>
    /// libopus 1.5.2 special_hybrid_folding: duplicate enough of the first band's folding data (at <paramref name="normAt"/>, and
    /// at <paramref name="norm2At"/> with dual stereo) to fold the second band. Copies nothing when the start band is 0.
    /// </summary>
    private static void SpecialHybridFolding(CeltModeGpu m, ArrayView<int> buf, int normAt, int norm2At, int start, int M, int dual_stereo)
    {
        int n1 = M * (m.EBandsOf(start + 1) - m.EBandsOf(start));
        int n2 = M * (m.EBandsOf(start + 2) - m.EBandsOf(start + 1));
        for (int j = 0; j < n2 - n1; j++) buf[normAt + n1 + j] = buf[normAt + 2 * n1 - n2 + j];
        if (dual_stereo != 0)
            for (int j = 0; j < n2 - n1; j++) buf[norm2At + n1 + j] = buf[norm2At + 2 * n1 - n2 + j];
    }

    /// <summary>
    /// libopus quant_all_bands (decoder): every band's shape, in order, with folding and the collapse masks.
    /// <paramref name="buf"/> holds X_ at <paramref name="xAt"/> (C * (120 &lt;&lt; LM) is libopus' size; at least
    /// M * eBands[NbEBands]), Y_ at <paramref name="yAt"/> (-1 = mono) and the folding history `_norm` at
    /// <paramref name="normAt"/> (2 * M * (eBands[NbEBands - 1] - eBands[start]) ints). Collapse masks (one per band and
    /// channel, i * C + c) go to <paramref name="masks"/>; <paramref name="seed"/> comes back advanced.
    /// </summary>
    public static void QuantAllBandsDecode(CeltModeGpu m, int start, int end,
        ArrayView<int> buf, int xAt, int yAt, int normAt,
        ArrayView<int> masks, int masksAt, ArrayView<int> pulses, int pulsesAt,
        int shortBlocks, int spread, int dual_stereo, int intensity, ArrayView<int> tfRes, int tfAt,
        int total_bits, int balance,
        ref OpusRangeDecoderGpuState ec, ArrayView<byte> pkt, int pktAt, uint storage,
        int LM, int codedBands, ref uint seed,
        ArrayView<int> scratch, int scratchAt, ArrayView<uint> pvqU, int pvqUAt, int disableInv)
    {
        int nb = CeltModeTables.NbEBands;
        int num = 1;
        int num2 = ((yAt < 0) ? 1 : 2);
        int num3 = 1;   // decode
        var band_ctx2 = new CeltBandCtx();
        int num4 = 1 << LM;
        int num5 = ((shortBlocks == 0) ? 1 : num4);
        int num6 = num4 * m.EBandsOf(start);
        int num7 = num4 * m.EBandsOf(nb - 1) - num6;
        int array2 = xAt;   // lowband_scratch = X_ (array2 null => -1)
        int lowband_scratch_ptr = num4 * m.EBandsOf(nb - 1);
        int num8 = 0;
        band_ctx2.Intensity = intensity;
        band_ctx2.Seed = seed;
        band_ctx2.Spread = spread;
        band_ctx2.DisableInv = disableInv;
        for (int i = start; i < end; i++)
        {
            int num9 = -1;
            int num10 = 0;
            int num11 = 0;
            band_ctx2.I = i;
            int num12 = ((i == end - 1) ? 1 : 0);
            int array3 = xAt;
            int x_ptr = num4 * m.EBandsOf(i);
            int array4;
            if (yAt >= 0)
            {
                array4 = yAt;
                num10 = num4 * m.EBandsOf(i);
            }
            else
            {
                array4 = -1;
            }
            int num13 = num4 * m.EBandsOf(i + 1) - num4 * m.EBandsOf(i);
            int num14 = (int)CeltEntropyGpu.TellFrac(ref ec);
            if (i != start)
            {
                balance -= num14;
            }
            int num15 = (band_ctx2.RemainingBits = total_bits - num14 - 1);
            int num17;
            if (i <= codedBands - 1)
            {
                int num16 = celt_sudiv(balance, IMIN(3, codedBands - i));
                num17 = IMAX(0, IMIN(16383, IMIN(num15 + 1, pulses[pulsesAt + i] + num16)));
            }
            else
            {
                num17 = 0;
            }
            // libopus 1.5.2 (DISABLE_UPDATE_DRAFT off): the second band of a hybrid frame folds too, from a copy of the first's data
            if (num3 != 0 && (num4 * m.EBandsOf(i) - num13 >= num4 * m.EBandsOf(start) || i == start + 1) && (num != 0 || num8 == 0))
            {
                num8 = i;
            }
            if (i == start + 1)
                SpecialHybridFolding(m, buf, normAt, normAt + num7, start, num4, dual_stereo);
            num11 = (band_ctx2.TfChange = tfRes[tfAt + i]);
            if (i >= nb)   // m.effEBands == nbEBands in the static mode
            {
                array3 = normAt;
                x_ptr = 0;
                if (yAt >= 0)
                {
                    array4 = normAt;
                    num10 = 0;
                }
                array2 = -1;
            }
            if (i == end - 1)
            {
                array2 = -1;
            }
            uint num20;
            uint num21;
            if (num8 != 0 && (spread != 3 || num5 > 1 || num11 < 0))
            {
                num9 = IMAX(0, num4 * m.EBandsOf(num8) - num6 - num13);
                int num18 = num8;
                while (num4 * m.EBandsOf(--num18) > num9 + num6)
                {
                }
                int num19 = num8 - 1;
                while (++num19 < i && num4 * m.EBandsOf(num19) < num9 + num6 + num13)   // libopus 1.5.2 stops at i
                {
                }
                num20 = (num21 = 0u);
                int num22 = num18;
                do
                {
                    num20 |= (uint)masks[masksAt + num22 * num2];
                    num21 |= (uint)masks[masksAt + num22 * num2 + num2 - 1];
                }
                while (++num22 < num19);
            }
            else
            {
                num20 = (num21 = (uint)((1 << num5) - 1));
            }
            if (dual_stereo != 0 && i == intensity)
            {
                dual_stereo = 0;
                if (num3 != 0)
                {
                    for (int j = 0; j < num4 * m.EBandsOf(i) - num6; j++)
                    {
                        buf[normAt + j] = HALF32(buf[normAt + j] + buf[normAt + num7 + j]);
                    }
                }
            }
            int scr = array2 >= 0 ? array2 + lowband_scratch_ptr : -1;
            int low = num9 != -1 ? normAt + num9 : -1;
            int lowOut = num12 != 0 ? -1 : normAt + (num4 * m.EBandsOf(i) - num6);
            // The band's QuantBandDecode calls, as up to two jobs through ONE call site: every call site inlines the whole
            // partition recursion (WGSL helpers cannot take views, so WebGPU inlines every view-taking helper), and libopus'
            // quant_all_bands + quant_band_stereo have eight. Jobs, in libopus order: mono X; dual stereo X then Y; stereo
            // (quant_band_stereo) the N == 2 mid, or mid and side in the order of their bit budgets, rebalanced in between.
            int bandMode;   // 0 mono, 1 dual stereo, 2 stereo N == 2, 3 stereo split, 4 stereo N == 1 (no job)
            int njobs;
            int xBand = array3 + x_ptr, yBand = array4 + num10;
            int sFill = (int)(num20 | num21), sInv = 0, sImid = 0, sIside = 0, sItheta = 0, mbits = 0, sbits = 0, rebalance = 0;
            int n2Sign = 1, n2X = 0, n2Y = 0;
            uint cmX = 0u, cmY = 0u;
            if (dual_stereo != 0) { bandMode = 1; njobs = 2; }
            else if (array4 < 0) { bandMode = 0; njobs = 1; }
            else if (num13 == 1)
            {
                bandMode = 4; njobs = 0;
                cmX = QuantBandN1Decode(ref band_ctx2, ref ec, pkt, pktAt, storage, buf, xBand, yBand, num17, lowOut);
            }
            else
            {
                int sb = num17;
                int fill2 = sFill;
                ComputeThetaDecode(m, ref band_ctx2, ref ec, pkt, pktAt, storage, num13, ref sb, num5, num5, LM, 1, ref sFill,
                    out sInv, out sImid, out sIside, out int sDelta, out sItheta, out int sQalloc);
                if (num13 == 2)
                {
                    bandMode = 2; njobs = 1;
                    sbits = sItheta != 0 && sItheta != 16384 ? 8 : 0;
                    mbits = sb - sbits;
                    band_ctx2.RemainingBits -= sQalloc + sbits;
                    bool c = sItheta > 8192;
                    n2X = c ? yBand : xBand;   // libopus x2 / y2
                    n2Y = c ? xBand : yBand;
                    int sign = 0;
                    if (sbits != 0) sign = (int)OpusRangeDecoderGpu.DecodeBits(ref ec, pkt, pktAt, storage, 1);
                    n2Sign = 1 - 2 * sign;
                    sFill = fill2;
                }
                else
                {
                    bandMode = 3; njobs = 2;
                    mbits = IMAX(0, IMIN(sb, (sb - sDelta) / 2));
                    sbits = sb - mbits;
                    band_ctx2.RemainingBits -= sQalloc;
                    rebalance = band_ctx2.RemainingBits;
                }
            }
            for (int job = 0; job < njobs; job++)
            {
                int jX, jB, jLow, jLowOut, jGain, jScr, jFill;
                if (bandMode == 0)
                {
                    jX = xBand; jB = num17; jLow = low; jLowOut = lowOut; jGain = 32767; jScr = scr; jFill = sFill;
                }
                else if (bandMode == 1)
                {
                    jB = num17 / 2; jGain = 32767; jScr = scr;
                    if (job == 0) { jX = xBand; jLow = low; jLowOut = lowOut; jFill = (int)num20; }
                    else
                    {
                        jX = yBand;
                        jLow = num9 != -1 ? normAt + num7 + num9 : -1;
                        jLowOut = num12 != 0 ? -1 : normAt + num7 + (num4 * m.EBandsOf(i) - num6);
                        jFill = (int)num21;
                    }
                }
                else if (bandMode == 2)
                {
                    jX = n2X; jB = mbits; jLow = low; jLowOut = lowOut; jGain = 32767; jScr = scr; jFill = sFill;
                }
                else
                {
                    bool midFirst = mbits >= sbits;
                    if (job == 1)
                    {
                        // Rebalance what the first half did not use into the second.
                        int used = (midFirst ? mbits : sbits) - (rebalance - band_ctx2.RemainingBits);
                        if (used > 24 && sItheta != (midFirst ? 0 : 16384))
                        {
                            if (midFirst) sbits += used - 24;
                            else mbits += used - 24;
                        }
                    }
                    if ((job == 0) == midFirst)
                    {
                        jX = xBand; jB = mbits; jLow = low; jLowOut = lowOut; jGain = 32767; jScr = scr; jFill = sFill;
                    }
                    else
                    {
                        jX = yBand; jB = sbits; jLow = -1; jLowOut = -1; jGain = sIside; jScr = -1; jFill = sFill >> num5;
                    }
                }
                uint r = QuantBandDecode(m, ref band_ctx2, ref ec, pkt, pktAt, storage, buf, jX, num13, jB, num5, jLow, LM, jLowOut,
                    jGain, jScr, jFill, scratch, scratchAt, pvqU, pvqUAt);
                if (bandMode == 1 && job == 1) cmY = r;
                else cmX |= r;
            }
            if (bandMode == 2)
            {
                // The side of a 2-bin band from the mid's bins and the sign, then the mid / side mix (libopus quant_band_stereo).
                buf[n2Y] = -n2Sign * buf[n2X + 1];
                buf[n2Y + 1] = n2Sign * buf[n2X];
                buf[xBand] = MULT16_16_Q15(sImid, buf[xBand]);
                buf[xBand + 1] = MULT16_16_Q15(sImid, buf[xBand + 1]);
                buf[yBand] = MULT16_16_Q15(sIside, buf[yBand]);
                buf[yBand + 1] = MULT16_16_Q15(sIside, buf[yBand + 1]);
                int a = buf[xBand];
                buf[xBand] = SUB16(a, buf[yBand]);
                buf[yBand] = ADD16(a, buf[yBand]);
                a = buf[xBand + 1];
                buf[xBand + 1] = SUB16(a, buf[yBand + 1]);
                buf[yBand + 1] = ADD16(a, buf[yBand + 1]);
            }
            if (bandMode == 3)
                CeltBandsGpu.StereoMerge(buf, xBand, buf, yBand, sImid, num13);
            if ((bandMode == 2 || bandMode == 3) && sInv != 0)
                for (int k = yBand; k < num13 + yBand; k++)
                    buf[k] = (short)(-buf[k]);
            num20 = cmX;
            num21 = bandMode == 1 ? cmY : cmX;
            masks[masksAt + i * num2] = (int)(num20 & 0xFF);
            masks[masksAt + i * num2 + num2 - 1] = (int)(num21 & 0xFF);
            balance += pulses[pulsesAt + i] + num14;
            num = ((num17 > num13 << 3) ? 1 : 0);
        }
        seed = band_ctx2.Seed;
    }
}
