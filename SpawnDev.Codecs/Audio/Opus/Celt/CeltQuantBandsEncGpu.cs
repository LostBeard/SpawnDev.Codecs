// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt). See NOTICE.md for upstream attributions.
//
// The CELT band recursion on the accelerator, encode side without resynthesis, ported from libopus 1.5.2 (fixed point;
// celt/bands.c quant_all_bands, quant_band, quant_band_stereo, quant_band_n1, quant_partition, compute_theta with
// encode = 1). libopus' encoder resynthesises only for theta RDO (stereo, no dual stereo, complexity >= 8): without it
// lowband_offset never leaves 0, so no band folds and the coded bits depend on X / Y and the bit budget alone. The split
// recursion runs on an explicit frame stack as in the decoder (CeltQuantBandsGpu); the leaves are CeltPvqEncGpu.AlgQuant.

using ILGPU;
using static SpawnDev.Codecs.Audio.Opus.Celt.CeltOps;

namespace SpawnDev.Codecs.Audio.Opus.Celt;

/// <summary>libopus band_ctx (encode, no resynthesis): the band being coded and the bit state shared down its recursion.</summary>
public struct CeltBandEncCtx
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
    /// <summary>1 forces the stereo inversion flag off (libopus disable_inv).</summary>
    public int DisableInv;
    /// <summary>1 while coding the first band of a frame with short blocks (libopus avoid_split_noise).</summary>
    public int AvoidSplitNoise;
    /// <summary>Theta rounding of a stereo split: 0 nearest, -1 down, 1 up (libopus theta_round, theta RDO).</summary>
    public int ThetaRound;
    /// <summary>The folding noise seed (resynthesis only).</summary>
    public uint Seed;
}

/// <summary>The CELT band recursion, encode side without resynthesis (libopus 1.5.2, GPU-callable).</summary>
public static class CeltQuantBandsEncGpu
{
    private const int Nb = CeltModeTables.NbEBands;
    /// <summary>Maximum quant_partition depth.</summary>
    public const int PartitionDepth = CeltQuantBandsGpu.PartitionDepth;
    /// <summary>Ints per quant_partition frame.</summary>
    public const int FrameInts = 12;
    /// <summary>Largest band size N.</summary>
    public const int MaxBandN = CeltQuantBandsGpu.MaxBandN;
    /// <summary>Scratch ints one <see cref="QuantAllBandsEncode"/> call needs: the partition stack, alg_quant's three
    /// vectors and the interleave buffer.</summary>
    public const int ScratchInts = PartitionDepth * FrameInts + 4 * MaxBandN;

    // frame fields
    private const int FX = 0, FN = 1, FBits = 2, FB = 3, FLM = 4, FFill = 5, FStage = 6, FMidFirst = 7, FXSide = 8, FRem = 9,
        FItheta = 10, FSbits = 11;

    private static int FRAC_MUL16(int a, int b) => (16384 + (short)a * (short)b) >> 15;

    /// <summary>libopus compute_theta (encoder): quantizes (nearest, or for a stereo split of theta RDO down / up by
    /// <see cref="CeltBandEncCtx.ThetaRound"/>) and codes the split angle of a band (or of a mid / side pair, which it mixes
    /// in place), and how many bits that took.</summary>
    public static void ComputeThetaEncode(CeltModeGpu m, ref CeltBandEncCtx ctx, ref OpusRangeEncoderGpuState ec,
        ArrayView<byte> buf, int bufAt, uint storage, ArrayView<int> x, int xAt, int yAt, ArrayView<int> bandE, int bandEAt,
        int N, ref int b, int B, int B0, int LM, int stereo, ref int fill,
        out int sInv, out int sImid, out int sIside, out int sDelta, out int sItheta, out int sQalloc)
    {
        int i = ctx.I;
        int inv = 0;
        // Decide on the resolution to give to the split parameter theta
        int pulse_cap = m.LogNOf(i) + LM * (1 << BITRES);
        int offset = (pulse_cap >> 1) - (stereo != 0 && N == 2 ? 16 : 4);   // QTHETA_OFFSET_TWOPHASE / QTHETA_OFFSET
        int qn = CeltBandsGpu.ComputeQn(N, b, offset, pulse_cap, stereo);
        if (stereo != 0 && i >= ctx.Intensity) qn = 1;
        int itheta = CeltPvqGpu.StereoItheta(x, xAt, x, yAt, stereo, N);
        int tell = (int)OpusRangeEncoderGpu.TellFrac(ref ec);
        if (qn != 1)
        {
            if (stereo != 0 && ctx.ThetaRound != 0)
            {
                // Bias quantization towards itheta=0 and itheta=16384.
                int bias = itheta > 8192 ? 32767 / qn : -32767 / qn;
                int down = IMIN(qn - 1, IMAX(0, (itheta * qn + bias) >> 14));
                itheta = ctx.ThetaRound < 0 ? down : down + 1;
            }
            else itheta = (itheta * qn + 8192) >> 14;
            if (stereo == 0 && ctx.AvoidSplitNoise != 0 && itheta > 0 && itheta < qn)
            {
                // Check if the selected value of theta will cause the bit allocation to inject noise on one side. If so,
                // make sure the energy of that side is zero.
                int unquantized = itheta * 16384 / qn;
                int imid0 = CeltBandsGpu.BitexactCos((short)unquantized);
                int iside0 = CeltBandsGpu.BitexactCos((short)(16384 - unquantized));
                int delta0 = FRAC_MUL16((N - 1) << 7, CeltBandsGpu.BitexactLog2tan(iside0, imid0));
                if (delta0 > b) itheta = qn;
                else if (delta0 < -b) itheta = 0;
            }
            // Entropy coding of the angle: a uniform pdf for the time split, a step for stereo, a triangular one for the rest.
            if (stereo != 0 && N > 2)
            {
                const int p0 = 3;
                int xv = itheta, x0 = qn / 2;
                uint ft = (uint)(p0 * (x0 + 1) + x0);
                OpusRangeEncoderGpu.Encode(ref ec, buf, bufAt, storage,
                    (uint)(xv <= x0 ? p0 * xv : (xv - 1 - x0) + (x0 + 1) * p0),
                    (uint)(xv <= x0 ? p0 * (xv + 1) : (xv - x0) + (x0 + 1) * p0), ft);
            }
            else if (B0 > 1 || stereo != 0)
            {
                OpusRangeEncoderGpu.EncodeUint(ref ec, buf, bufAt, storage, (uint)itheta, (uint)(qn + 1));
            }
            else
            {
                int ft = ((qn >> 1) + 1) * ((qn >> 1) + 1);
                int fs = itheta <= (qn >> 1) ? itheta + 1 : qn + 1 - itheta;
                int fl = itheta <= (qn >> 1) ? itheta * (itheta + 1) >> 1 : ft - ((qn + 1 - itheta) * (qn + 2 - itheta) >> 1);
                OpusRangeEncoderGpu.Encode(ref ec, buf, bufAt, storage, (uint)fl, (uint)(fl + fs), (uint)ft);
            }
            itheta = itheta * 16384 / qn;
            if (stereo != 0)
            {
                if (itheta == 0) CeltBandsGpu.IntensityStereo(x, xAt, x, yAt, bandE, bandEAt, Nb, i, N);
                else CeltBandsGpu.StereoSplit(x, xAt, x, yAt, N);
            }
        }
        else if (stereo != 0)
        {
            inv = itheta > 8192 && ctx.DisableInv == 0 ? 1 : 0;
            if (inv != 0)
                for (int j = 0; j < N; j++) x[yAt + j] = -x[yAt + j];
            CeltBandsGpu.IntensityStereo(x, xAt, x, yAt, bandE, bandEAt, Nb, i, N);
            if (b > 2 << BITRES && ctx.RemainingBits > 2 << BITRES)
                OpusRangeEncoderGpu.EncodeBitLogP(ref ec, buf, bufAt, storage, inv, 2);
            else inv = 0;
            // inv flag override to avoid problems with downmixing
            if (ctx.DisableInv != 0) inv = 0;
            itheta = 0;
        }
        int qalloc = (int)OpusRangeEncoderGpu.TellFrac(ref ec) - tell;
        b -= qalloc;
        int imid, iside, delta;
        if (itheta == 0)
        {
            imid = 32767;
            iside = 0;
            fill &= (1 << B) - 1;
            delta = -16384;
        }
        else if (itheta == 16384)
        {
            imid = 0;
            iside = 32767;
            fill &= ((1 << B) - 1) << B;
            delta = 16384;
        }
        else
        {
            imid = CeltBandsGpu.BitexactCos((short)itheta);
            iside = CeltBandsGpu.BitexactCos((short)(16384 - itheta));
            // The mid vs side allocation that minimizes squared error in that band
            delta = FRAC_MUL16((N - 1) << 7, CeltBandsGpu.BitexactLog2tan(iside, imid));
        }
        sInv = inv;
        sImid = imid;
        sIside = iside;
        sDelta = delta;
        sItheta = itheta;
        sQalloc = qalloc;
    }

    /// <summary>libopus quant_band_n1 (encoder): a band of one bin per channel is a sign bit.</summary>
    public static uint QuantBandN1Encode(ref CeltBandEncCtx ctx, ref OpusRangeEncoderGpuState ec, ArrayView<byte> buf, int bufAt,
        uint storage, ArrayView<int> x, int xAt, int yAt)
    {
        int at = xAt;
        int channels = yAt >= 0 ? 2 : 1;
        for (int c = 0; c < channels; c++)
        {
            if (ctx.RemainingBits >= 1 << BITRES)
            {
                OpusRangeEncoderGpu.EncodeBits(ref ec, buf, bufAt, storage, x[at] < 0 ? 1u : 0u, 1);
                ctx.RemainingBits -= 1 << BITRES;
            }
            at = yAt;
        }
        return 1;
    }

    /// <summary>
    /// libopus quant_partition (encoder, no resynthesis), iterative: codes the N values at <paramref name="xAt"/> with b bits,
    /// splitting recursively. Uses the frame stack and alg_quant's vectors in <paramref name="scratch"/> at
    /// <paramref name="scratchAt"/>.
    /// </summary>
    public static uint QuantPartitionEncode(CeltModeGpu m, ref CeltBandEncCtx ctx, ref OpusRangeEncoderGpuState ec,
        ArrayView<byte> buf, int bufAt, uint storage, ArrayView<int> x, int xAt, int N, int b, int B, int LM, int fill,
        ArrayView<int> scratch, int scratchAt, ArrayView<uint> pvqU, int pvqUAt)
    {
        int vecAt = scratchAt + PartitionDepth * FrameInts;
        uint ret = 0;
        int f0 = scratchAt;
        scratch[f0 + FX] = xAt; scratch[f0 + FN] = N; scratch[f0 + FBits] = b; scratch[f0 + FB] = B; scratch[f0 + FLM] = LM;
        scratch[f0 + FFill] = fill; scratch[f0 + FStage] = 0;
        int sp = 1;
        while (true)
        {
            int f = scratchAt + (sp - 1) * FrameInts;
            int stage = scratch[f + FStage];
            if (stage == 0)
            {
                int fx = scratch[f + FX], fN = scratch[f + FN], fb = scratch[f + FBits], fB = scratch[f + FB], fLM = scratch[f + FLM];
                int fFill = scratch[f + FFill];
                int B0 = fB;
                int i = ctx.I;
                // If we need 1.5 more bit than we can produce, split the band in two.
                int cache = m.CacheIndexOf((fLM + 1) * Nb + i);
                if (fLM != -1 && fb > m.CacheBitsOf(cache + m.CacheBitsOf(cache)) + 12 && fN > 2)
                {
                    fN >>= 1;
                    int fy = fx + fN;
                    fLM -= 1;
                    if (fB == 1) fFill = (fFill & 1) | (fFill << 1);
                    fB = (fB + 1) >> 1;
                    ComputeThetaEncode(m, ref ctx, ref ec, buf, bufAt, storage, x, fx, fy, x, 0, fN, ref fb, fB, B0, fLM, 0, ref fFill,
                        out _, out _, out _, out int delta, out int itheta, out int qalloc);
                    // Give more bits to low-energy MDCTs than they would otherwise deserve
                    if (B0 > 1 && (itheta & 0x3fff) != 0)
                    {
                        if (itheta > 8192) delta -= delta >> (4 - fLM);   // Rough approximation for pre-echo masking
                        else delta = IMIN(0, delta + (fN << BITRES >> (5 - fLM)));   // a forward-masking slope of 1.5 dB per 10 ms
                    }
                    int mbits = IMAX(0, IMIN(fb, (fb - delta) / 2));
                    int sbits = fb - mbits;
                    ctx.RemainingBits -= qalloc;
                    // the frame becomes the split's continuation
                    scratch[f + FN] = fN; scratch[f + FB] = fB; scratch[f + FLM] = fLM; scratch[f + FFill] = fFill;
                    scratch[f + FXSide] = fy; scratch[f + FRem] = ctx.RemainingBits; scratch[f + FItheta] = itheta;
                    scratch[f + FMidFirst] = mbits >= sbits ? 1 : 0;
                    scratch[f + FBits] = mbits; scratch[f + FSbits] = sbits;
                    scratch[f + FStage] = 1;
                    int c = f + FrameInts;
                    if (mbits >= sbits)
                    {
                        scratch[c + FX] = fx; scratch[c + FBits] = mbits; scratch[c + FFill] = fFill;
                    }
                    else
                    {
                        scratch[c + FX] = fy; scratch[c + FBits] = sbits; scratch[c + FFill] = fFill >> fB;
                    }
                    scratch[c + FN] = fN; scratch[c + FB] = fB; scratch[c + FLM] = fLM; scratch[c + FStage] = 0;
                    sp++;
                    continue;
                }
                // The basic no-split case
                uint cm = 0;
                int q = CeltRateGpu.Bits2Pulses(m, i, fLM, fb);
                int curr_bits = CeltRateGpu.Pulses2Bits(m, i, fLM, q);
                ctx.RemainingBits -= curr_bits;
                // Ensures we can never bust the budget
                while (ctx.RemainingBits < 0 && q > 0)
                {
                    ctx.RemainingBits += curr_bits;
                    q--;
                    curr_bits = CeltRateGpu.Pulses2Bits(m, i, fLM, q);
                    ctx.RemainingBits -= curr_bits;
                }
                if (q != 0)
                {
                    int K = CeltRateGpu.GetPulses(q);
                    cm = CeltPvqEncGpu.AlgQuant(x, fx, fN, K, ctx.Spread, fB, ref ec, buf, bufAt, storage, Q15ONE, 0, scratch, vecAt,
                        vecAt + MaxBandN, vecAt + 2 * MaxBandN, pvqU, pvqUAt);
                }
                ret = cm;
                sp--;
                if (sp == 0) break;
                continue;
            }
            if (stage == 1)
            {
                // the first child finished with `ret`
                int fB = scratch[f + FB], itheta = scratch[f + FItheta];
                int mbits = scratch[f + FBits], sbits = scratch[f + FSbits];
                int rebalance = scratch[f + FRem];
                int c = f + FrameInts;
                if (scratch[f + FMidFirst] != 0)
                {
                    rebalance = mbits - (rebalance - ctx.RemainingBits);
                    if (rebalance > 3 << BITRES && itheta != 0) sbits += rebalance - (3 << BITRES);
                    scratch[c + FX] = scratch[f + FXSide]; scratch[c + FBits] = sbits; scratch[c + FFill] = scratch[f + FFill] >> fB;
                }
                else
                {
                    rebalance = sbits - (rebalance - ctx.RemainingBits);
                    if (rebalance > 3 << BITRES && itheta != 16384) mbits += rebalance - (3 << BITRES);
                    scratch[c + FX] = scratch[f + FX]; scratch[c + FBits] = mbits; scratch[c + FFill] = scratch[f + FFill];
                }
                scratch[c + FN] = scratch[f + FN]; scratch[c + FB] = fB; scratch[c + FLM] = scratch[f + FLM]; scratch[c + FStage] = 0;
                scratch[f + FStage] = 2;
                sp++;
                continue;
            }
            // stage 2: the second child finished; the encoder's collapse masks are not used without resynthesis
            ret = 0;
            sp--;
            if (sp == 0) break;
        }
        return ret;
    }

    /// <summary>libopus quant_band (encoder, no resynthesis): one band of one channel, with the TF resolution changes before
    /// quant_partition.</summary>
    public static uint QuantBandEncode(CeltModeGpu m, ref CeltBandEncCtx ctx, ref OpusRangeEncoderGpuState ec,
        ArrayView<byte> buf, int bufAt, uint storage, ArrayView<int> x, int xAt, int N, int b, int B, int LM, int fill,
        ArrayView<int> scratch, int scratchAt, ArrayView<uint> pvqU, int pvqUAt)
    {
        int tmpAt = scratchAt + PartitionDepth * FrameInts + 3 * MaxBandN;
        int B0 = B;
        int time_divide = 0, recombine = 0;
        int longBlocks = B0 == 1 ? 1 : 0;
        int tf_change = ctx.TfChange;
        int N_B = N / B;
        // Special case for one sample
        if (N == 1) return QuantBandN1Encode(ref ctx, ref ec, buf, bufAt, storage, x, xAt, -1);
        if (tf_change > 0) recombine = tf_change;
        // Band recombining to increase frequency resolution
        for (int k = 0; k < recombine; k++)
        {
            CeltBandsGpu.Haar1(x, xAt, N >> k, 1 << k);
            fill = CeltQuantBandsGpu.BitInterleave(fill & 0xF) | CeltQuantBandsGpu.BitInterleave(fill >> 4) << 2;
        }
        B >>= recombine;
        N_B <<= recombine;
        // Increasing the time resolution
        while ((N_B & 1) == 0 && tf_change < 0)
        {
            CeltBandsGpu.Haar1(x, xAt, N_B, B);
            fill |= fill << B;
            B <<= 1;
            N_B >>= 1;
            time_divide++;
            tf_change++;
        }
        B0 = B;
        // Reorganize the samples in time order instead of frequency order
        if (B0 > 1) CeltBandsGpu.DeinterleaveHadamard(m, x, xAt, N_B >> recombine, B0 << recombine, longBlocks, scratch, tmpAt);
        return QuantPartitionEncode(m, ref ctx, ref ec, buf, bufAt, storage, x, xAt, N, b, B, LM, fill, scratch, scratchAt, pvqU, pvqUAt);
    }

    /// <summary>libopus quant_band_stereo (encoder, no resynthesis): one band of a mid / side or intensity pair.</summary>
    public static uint QuantBandStereoEncode(CeltModeGpu m, ref CeltBandEncCtx ctx, ref OpusRangeEncoderGpuState ec,
        ArrayView<byte> buf, int bufAt, uint storage, ArrayView<int> x, int xAt, int yAt, ArrayView<int> bandE, int bandEAt,
        int N, int b, int B, int LM, int fill, ArrayView<int> scratch, int scratchAt, ArrayView<uint> pvqU, int pvqUAt)
    {
        // Special case for one sample
        if (N == 1) return QuantBandN1Encode(ref ctx, ref ec, buf, bufAt, storage, x, xAt, yAt);
        int orig_fill = fill;
        ComputeThetaEncode(m, ref ctx, ref ec, buf, bufAt, storage, x, xAt, yAt, bandE, bandEAt, N, ref b, B, B, LM, 1, ref fill,
            out _, out _, out _, out int delta, out int itheta, out int qalloc);
        uint cm;
        if (N == 2)
        {
            // N = 2 codes the side with one sign bit (mid and side are orthogonal)
            int mbits = b, sbits = 0;
            if (itheta != 0 && itheta != 16384) sbits = 1 << BITRES;
            mbits -= sbits;
            int c = itheta > 8192 ? 1 : 0;
            ctx.RemainingBits -= qalloc + sbits;
            int x2 = c != 0 ? yAt : xAt, y2 = c != 0 ? xAt : yAt;
            int sign = 0;
            if (sbits != 0)
            {
                sign = x[x2] * x[y2 + 1] - x[x2 + 1] * x[y2] < 0 ? 1 : 0;
                OpusRangeEncoderGpu.EncodeBits(ref ec, buf, bufAt, storage, (uint)sign, 1);
            }
            sign = 1 - 2 * sign;
            // orig_fill: we want to fold the side, but itheta == 16384 cleared the low bits of fill
            cm = QuantBandEncode(m, ref ctx, ref ec, buf, bufAt, storage, x, x2, N, mbits, B, LM, orig_fill, scratch, scratchAt, pvqU, pvqUAt);
            x[y2] = -sign * x[x2 + 1];
            x[y2 + 1] = sign * x[x2];
        }
        else
        {
            // "Normal" split code
            int mbits = IMAX(0, IMIN(b, (b - delta) / 2));
            int sbits = b - mbits;
            ctx.RemainingBits -= qalloc;
            int rebalance = ctx.RemainingBits;
            if (mbits >= sbits)
            {
                cm = QuantBandEncode(m, ref ctx, ref ec, buf, bufAt, storage, x, xAt, N, mbits, B, LM, fill, scratch, scratchAt, pvqU, pvqUAt);
                rebalance = mbits - (rebalance - ctx.RemainingBits);
                if (rebalance > 3 << BITRES && itheta != 0) sbits += rebalance - (3 << BITRES);
                // For a stereo split, the high bits of fill are always zero, so no folding will be done to the side.
                cm |= QuantBandEncode(m, ref ctx, ref ec, buf, bufAt, storage, x, yAt, N, sbits, B, LM, fill >> B, scratch, scratchAt, pvqU, pvqUAt);
            }
            else
            {
                cm = QuantBandEncode(m, ref ctx, ref ec, buf, bufAt, storage, x, yAt, N, sbits, B, LM, fill >> B, scratch, scratchAt, pvqU, pvqUAt);
                rebalance = sbits - (rebalance - ctx.RemainingBits);
                if (rebalance > 3 << BITRES && itheta != 16384) mbits += rebalance - (3 << BITRES);
                cm |= QuantBandEncode(m, ref ctx, ref ec, buf, bufAt, storage, x, xAt, N, mbits, B, LM, fill, scratch, scratchAt, pvqU, pvqUAt);
            }
        }
        return cm;
    }

    /// <summary>
    /// libopus quant_all_bands (encode, no resynthesis: mono, dual stereo, or stereo below complexity 8): codes the shapes of
    /// bands start..end-1 of the normalised spectrum (channel 0 at <paramref name="xAt"/>, channel 1 at <paramref name="yAt"/>,
    /// -1 for mono; consumed in place) with the allocation's per-band <paramref name="pulses"/> and the balance.
    /// <paramref name="bandE"/> holds the linear band energies (channel c at bandEAt + c * NbEBands), used by intensity stereo.
    /// <paramref name="scratch"/> holds <see cref="ScratchInts"/> ints at <paramref name="scratchAt"/>.
    /// </summary>
    public static void QuantAllBandsEncode(CeltModeGpu m, int start, int end, ArrayView<int> x, int xAt, int yAt,
        ArrayView<int> bandE, int bandEAt, ArrayView<int> pulses, int pulsesAt, int shortBlocks, int spread, int dual_stereo,
        int intensity, ArrayView<int> tfRes, int tfAt, int total_bits, int balance, ref OpusRangeEncoderGpuState ec,
        ArrayView<byte> buf, int bufAt, uint storage, int LM, int codedBands, int disableInv,
        ArrayView<int> scratch, int scratchAt, ArrayView<uint> pvqU, int pvqUAt)
    {
        int M = 1 << LM;
        int B = shortBlocks != 0 ? M : 1;
        var ctx = new CeltBandEncCtx { Intensity = intensity, Spread = spread, DisableInv = disableInv };
        // Avoid injecting noise in the first band on transients.
        ctx.AvoidSplitNoise = B > 1 ? 1 : 0;
        for (int i = start; i < end; i++)
        {
            ctx.I = i;
            int X = xAt + M * m.EBandsOf(i);
            int Y = yAt >= 0 ? yAt + M * m.EBandsOf(i) : -1;
            int N = M * m.EBandsOf(i + 1) - M * m.EBandsOf(i);
            int tell = (int)OpusRangeEncoderGpu.TellFrac(ref ec);
            // Compute how many bits we want to allocate to this band
            if (i != start) balance -= tell;
            int remaining_bits = total_bits - tell - 1;
            ctx.RemainingBits = remaining_bits;
            int b;
            if (i <= codedBands - 1)
            {
                int curr_balance = balance / IMIN(3, codedBands - i);
                b = IMAX(0, IMIN(16383, IMIN(remaining_bits + 1, pulses[pulsesAt + i] + curr_balance)));
            }
            else b = 0;
            ctx.TfChange = tfRes[tfAt + i];
            // no resynthesis: nothing folds, every block of the band may get pulses
            int fill = (1 << B) - 1;
            if (dual_stereo != 0 && i == intensity) dual_stereo = 0;   // Switch off dual stereo to do intensity.
            if (dual_stereo != 0)
            {
                QuantBandEncode(m, ref ctx, ref ec, buf, bufAt, storage, x, X, N, b / 2, B, LM, fill, scratch, scratchAt, pvqU, pvqUAt);
                QuantBandEncode(m, ref ctx, ref ec, buf, bufAt, storage, x, Y, N, b / 2, B, LM, fill, scratch, scratchAt, pvqU, pvqUAt);
            }
            else if (Y >= 0)
            {
                QuantBandStereoEncode(m, ref ctx, ref ec, buf, bufAt, storage, x, X, Y, bandE, bandEAt, N, b, B, LM, fill,
                    scratch, scratchAt, pvqU, pvqUAt);
            }
            else
            {
                QuantBandEncode(m, ref ctx, ref ec, buf, bufAt, storage, x, X, N, b, B, LM, fill, scratch, scratchAt, pvqU, pvqUAt);
            }
            balance += pulses[pulsesAt + i] + tell;
            // We only need to avoid noise on a split for the first band. After that, we have folding.
            ctx.AvoidSplitNoise = 0;
        }
    }
}
