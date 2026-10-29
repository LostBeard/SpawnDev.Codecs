// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt). See NOTICE.md for upstream attributions.
//
// The CELT encoder's band context and split angle, ported from libopus 1.5.2 (fixed point; celt/bands.c band_ctx and
// compute_theta with encode = 1), used by the band recursion (CeltBandsEncGpu).

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

/// <summary>The CELT encoder's split angle coding (libopus 1.5.2 compute_theta, GPU-callable).</summary>
public static class CeltQuantBandsEncGpu
{
    private const int Nb = CeltModeTables.NbEBands;
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
}
