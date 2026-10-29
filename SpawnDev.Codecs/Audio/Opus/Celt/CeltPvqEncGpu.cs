// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt). See NOTICE.md for upstream attributions.
//
// The CELT band shape quantizer on the accelerator, ported from libopus 1.5.2 (fixed point; celt/vq.c alg_quant and
// op_pvq_search_c, celt/cwrs.c encode_pulses) with 1.5.2's C arithmetic (CeltOps). Not Concentus' alg_quant (libopus ~1.1):
// 1.5.2 projects on the pyramid with K pulses, not K - 1, keeps the sign as 0 / 1 and stores yy and the scores as
// opus_val16. The spreading rotation, the residual normalisation, the collapse mask and the codebook index are the decoder's
// (CeltPvqGpu), unchanged since 1.1.

using ILGPU;
using static SpawnDev.Codecs.Audio.Opus.Celt.CeltOps;

namespace SpawnDev.Codecs.Audio.Opus.Celt;

/// <summary>CELT band shape quantization (libopus 1.5.2 alg_quant, GPU-callable).</summary>
public static class CeltPvqEncGpu
{
    /// <summary>
    /// libopus op_pvq_search_c: the <paramref name="K"/>-pulse vector closest in direction to the <paramref name="N"/> values at
    /// <paramref name="xAt"/>, signed, to <paramref name="iyAt"/>; leaves |X| in x. Returns yy (the squared norm of the pulses,
    /// 16 bits). <paramref name="yAt"/> and <paramref name="signAt"/> hold N ints of scratch each, in <paramref name="w"/>.
    /// </summary>
    public static int OpPvqSearch(ArrayView<int> x, int xAt, ArrayView<int> iy, int iyAt, int K, int N, ArrayView<int> w,
        int yAt, int signAt)
    {
        // Get rid of the sign
        int sum = 0;
        for (int j = 0; j < N; j++)
        {
            w[signAt + j] = x[xAt + j] < 0 ? 1 : 0;
            x[xAt + j] = S16(ABS32(x[xAt + j]));
            iy[iyAt + j] = 0;
            w[yAt + j] = 0;
        }
        int xy = 0, yy = 0;
        int pulsesLeft = K;
        // Do a pre-search by projecting on the pyramid
        if (K > (N >> 1))
        {
            for (int j = 0; j < N; j++) sum += x[xAt + j];
            // If X is too small, just replace it with a pulse at 0
            if (sum <= K)
            {
                x[xAt] = 16384;   // QCONST16(1.f, 14)
                for (int j = 1; j < N; j++) x[xAt + j] = 0;
                sum = 16384;
            }
            int rcp = S16(MULT16_32_Q16(K, celt_rcp(sum)));
            for (int j = 0; j < N; j++)
            {
                // It's really important to round *towards zero* here
                iy[iyAt + j] = MULT16_16_Q15(x[xAt + j], rcp);
                int yj = S16(iy[iyAt + j]);
                yy = S16(MAC16_16(yy, yj, yj));
                xy = MAC16_16(xy, x[xAt + j], yj);
                w[yAt + j] = S16(yj * 2);
                pulsesLeft -= iy[iyAt + j];
            }
        }
        // This should never happen, but just in case it does (e.g. on silence) we fill the first bin with pulses.
        if (pulsesLeft > N + 3)
        {
            int tmp = S16(pulsesLeft);
            yy = S16(MAC16_16(yy, tmp, tmp));
            yy = S16(MAC16_16(yy, tmp, w[yAt]));
            iy[iyAt] += pulsesLeft;
            pulsesLeft = 0;
        }
        for (int i = 0; i < pulsesLeft; i++)
        {
            int rshift = 1 + celt_ilog2(K - pulsesLeft + i + 1);
            int best_id = 0;
            // The squared magnitude term gets added anyway, so we might as well add it outside the loop
            yy = ADD16(yy, 1);
            // Position 0 out of the loop
            int Rxy = S16((xy + x[xAt]) >> rshift);
            int Ryy = ADD16(yy, w[yAt]);
            Rxy = S16(MULT16_16_Q15(Rxy, Rxy));
            int best_den = Ryy, best_num = Rxy;
            for (int j = 1; j < N; j++)
            {
                Rxy = S16((xy + x[xAt + j]) >> rshift);
                Ryy = ADD16(yy, w[yAt + j]);
                Rxy = S16(MULT16_16_Q15(Rxy, Rxy));
                // num / den >= best_num / best_den, without a division
                if (MULT16_16(best_den, Rxy) > MULT16_16(Ryy, best_num))
                {
                    best_den = Ryy;
                    best_num = Rxy;
                    best_id = j;
                }
            }
            // Updating the sums of the new pulse(s)
            xy = xy + x[xAt + best_id];
            yy = ADD16(yy, w[yAt + best_id]);
            // Only now that we've made the final choice, update y / iy
            w[yAt + best_id] = S16(w[yAt + best_id] + 2);
            iy[iyAt + best_id]++;
        }
        // Put the original sign back
        for (int j = 0; j < N; j++)
        {
            int sx = w[signAt + j];
            iy[iyAt + j] = (iy[iyAt + j] ^ -sx) + sx;
        }
        return yy;
    }

    /// <summary>
    /// libopus alg_quant: rotates the band shape at <paramref name="xAt"/> (spreading), searches its <paramref name="K"/>-pulse
    /// vector and codes it (encode_pulses: ec_enc_uint(icwrs(N, iy), V(N, K))); with <paramref name="resynth"/> writes back the
    /// decoder's view of the band (normalised to <paramref name="gain"/>, rotated back). Returns the collapse mask. Scratch in
    /// <paramref name="w"/>: <paramref name="iyAt"/>, <paramref name="yAt"/> and <paramref name="signAt"/>, N ints each.
    /// </summary>
    public static uint AlgQuant(ArrayView<int> x, int xAt, int N, int K, int spread, int B, ref OpusRangeEncoderGpuState enc,
        ArrayView<byte> buf, int bufStart, uint storage, int gain, int resynth, ArrayView<int> w, int iyAt, int yAt, int signAt,
        ArrayView<uint> u, int uAt)
    {
        CeltPvqGpu.ExpRotation(x, xAt, N, 1, B, K, spread);
        int yy = OpPvqSearch(x, xAt, w, iyAt, K, N, w, yAt, signAt);
        OpusRangeEncoderGpu.EncodeUint(ref enc, buf, bufStart, storage, CeltPvqGpu.Icwrs(N, w, iyAt, u, uAt), CeltPvqGpu.PvqV(u, uAt, N, K));
        if (resynth != 0)
        {
            CeltPvqGpu.NormaliseResidual(w, iyAt, x, xAt, N, yy, gain);
            CeltPvqGpu.ExpRotation(x, xAt, N, -1, B, K, spread);
        }
        return CeltPvqGpu.ExtractCollapseMask(w, iyAt, N, B);
    }
}
