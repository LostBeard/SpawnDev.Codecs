// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// LPC (Linear Predictive Coding) subframe encoder. The approach matches
// libFLAC's flac_encoder.c / lpc.c:
//   1. Compute autocorrelation of the input samples.
//   2. Solve the Yule-Walker equations via Levinson-Durbin recursion to obtain
//      real-valued LPC coefficients.
//   3. Quantize coefficients to a signed integer representation at a fixed
//      precision (we use 12 bits here; libFLAC's default is 14).
//   4. Compute residuals against the QUANTIZED predictor so the encoder and
//      decoder agree bit-exactly.
//   5. Search Rice parameters, pick the smallest-bit-cost option, compare
//      against VERBATIM and the FIXED baseline from FlacFixedSubframeEncoder.

namespace SpawnDev.Codecs.Audio.Flac;

internal static class FlacLpcSubframeEncoder
{
    /// <summary>LPC orders we attempt. Keeping the search small bounds encoder runtime.</summary>
    private static readonly int[] CandidateOrders = { 4, 6, 8, 10, 12 };

    /// <summary>Quantization precision in bits for the LPC coefficients.</summary>
    private const int CoefficientPrecision = 12;

    private const int MaxRiceParam = 14;

    /// <summary>
    /// Try several LPC orders on <paramref name="samples"/> and return the best option
    /// or <c>null</c> if nothing beats <paramref name="verbatimBits"/>. The chosen
    /// option is a complete encode plan (order, quant level, quantized coefs,
    /// residuals, Rice parameter).
    /// </summary>
    internal static FlacLpcChoice? PickBest(ReadOnlySpan<int> samples, int bps, long verbatimBits)
    {
        if (samples.Length < 32) return null; // LPC needs enough samples for stable autocorrelation.

        double[] autocorr = ComputeAutocorrelation(samples, CandidateOrders[^1]);
        FlacLpcChoice? best = null;
        long bestBits = verbatimBits;

        foreach (int order in CandidateOrders)
        {
            if (order >= samples.Length) continue;
            if (autocorr[0] <= 0) continue; // silent or constant block - not a valid LPC target.

            double[] realCoefs = LevinsonDurbin(autocorr, order);
            if (realCoefs == null) continue;

            int[] qCoefs = QuantizeCoefficients(realCoefs, CoefficientPrecision, out int quantLevel);
            if (quantLevel < 0) continue; // libFLAC forbids negative quantization level on decode.

            int[] residual = ComputeResidualWithQuantizedCoefs(samples, qCoefs, quantLevel);

            int bestK = 0;
            long bestPartitionBits = long.MaxValue;
            for (int k = 0; k <= MaxRiceParam; k++)
            {
                long bits = EstimateRicePartitionBits(residual, k);
                if (bits < bestPartitionBits)
                {
                    bestPartitionBits = bits;
                    bestK = k;
                }
            }

            // Subframe bit count = 8 header + order*bps warm-up + 4 precision + 5 quant
            //   + order * precision coef bits + 10 residual-section + rice bits.
            long subframeBits = 8 + (long)order * bps + 4 + 5 + (long)order * CoefficientPrecision
                              + 2 + 4 + 4 + bestPartitionBits;
            if (subframeBits < bestBits)
            {
                bestBits = subframeBits;
                best = new FlacLpcChoice(order, quantLevel, qCoefs, residual, bestK, subframeBits);
            }
        }
        return best;
    }

    /// <summary>Emit the chosen LPC subframe into <paramref name="w"/>.</summary>
    internal static void Emit(FlacBitWriter w, ReadOnlySpan<int> samples, int bps, FlacLpcChoice choice)
    {
        int order = choice.Order;
        // Subframe header: reserved 0, type 0b1xxxxx (xxxxx = order - 1), wasted flag 0.
        w.Write(0, 1);
        w.Write((uint)(0b100000 | (order - 1)), 6);
        w.Write(0, 1);
        // Warm-up samples.
        for (int i = 0; i < order; i++) w.WriteSigned(samples[i], bps);
        // 4-bit (precision - 1), 5-bit signed quant level, then order * precision signed coefficient bits.
        w.Write((uint)(CoefficientPrecision - 1), 4);
        w.WriteSigned(choice.QuantLevel, 5);
        for (int i = 0; i < order; i++) w.WriteSigned(choice.Coefficients[i], CoefficientPrecision);
        // Residual section: method 0 (4-bit Rice), partition order 0.
        w.Write(0, 2);
        w.Write(0, 4);
        w.Write((uint)choice.RiceParam, 4);
        foreach (int r in choice.Residual)
        {
            uint u = r >= 0 ? (uint)(r << 1) : (uint)((-r << 1) - 1);
            int q = (int)(u >> choice.RiceParam);
            uint rem = u & ((1u << choice.RiceParam) - 1);
            w.WriteUnary(q);
            if (choice.RiceParam > 0) w.Write(rem, choice.RiceParam);
        }
    }

    private static double[] ComputeAutocorrelation(ReadOnlySpan<int> samples, int maxLag)
    {
        var r = new double[maxLag + 1];
        for (int lag = 0; lag <= maxLag; lag++)
        {
            double sum = 0;
            for (int i = 0; i < samples.Length - lag; i++)
                sum += (double)samples[i] * samples[i + lag];
            r[lag] = sum;
        }
        return r;
    }

    private static double[] LevinsonDurbin(double[] r, int order)
    {
        // Standard Yule-Walker solver. Produces coefs a[1..order] such that
        // predicted[n] = -sum(a[i] * samples[n-i]). We negate at the end to match
        // FLAC's convention (predictor uses positive coefs, so FLAC's a_flac[i] = -a[i]).
        var a = new double[order + 1];
        var aPrev = new double[order + 1];
        a[0] = 1.0;
        double err = r[0];
        for (int i = 1; i <= order; i++)
        {
            double acc = r[i];
            for (int j = 1; j < i; j++) acc += a[j] * r[i - j];
            if (err == 0) return null!;
            double k = -acc / err;
            Array.Copy(a, aPrev, i);
            a[i] = k;
            for (int j = 1; j < i; j++) a[j] = aPrev[j] + k * aPrev[i - j];
            err *= (1.0 - k * k);
            if (err <= 0) return null!;
        }
        // FLAC predictor convention: sample[n] = sum(c[i] * sample[n-1-i]) >> quant,
        // so c[i] = -a[i+1] (shifted and negated from Levinson convention).
        var flacCoefs = new double[order];
        for (int i = 0; i < order; i++) flacCoefs[i] = -a[i + 1];
        return flacCoefs;
    }

    private static int[] QuantizeCoefficients(double[] coefs, int precision, out int quantLevel)
    {
        // Pick quant level so max |coef * 2^quant| fits in `precision - 1` magnitude bits.
        double maxAbs = 0;
        for (int i = 0; i < coefs.Length; i++) maxAbs = Math.Max(maxAbs, Math.Abs(coefs[i]));
        if (maxAbs == 0)
        {
            quantLevel = 0;
            return new int[coefs.Length];
        }
        int maxSigned = (1 << (precision - 1)) - 1;
        double log2 = Math.Log2(maxSigned / maxAbs);
        quantLevel = (int)Math.Floor(log2);
        if (quantLevel < 0) quantLevel = 0;
        if (quantLevel > 30) quantLevel = 30;
        int[] q = new int[coefs.Length];
        int limit = (1 << (precision - 1)) - 1;
        double scale = 1L << quantLevel;
        for (int i = 0; i < coefs.Length; i++)
        {
            long v = (long)Math.Round(coefs[i] * scale);
            if (v > limit) v = limit;
            if (v < -limit - 1) v = -limit - 1;
            q[i] = (int)v;
        }
        return q;
    }

    private static int[] ComputeResidualWithQuantizedCoefs(ReadOnlySpan<int> samples, int[] coefs, int quantLevel)
    {
        int order = coefs.Length;
        var residual = new int[samples.Length - order];
        for (int n = order; n < samples.Length; n++)
        {
            long pred = 0;
            for (int i = 0; i < order; i++) pred += (long)coefs[i] * samples[n - 1 - i];
            residual[n - order] = samples[n] - (int)(pred >> quantLevel);
        }
        return residual;
    }

    private static long EstimateRicePartitionBits(ReadOnlySpan<int> residual, int k)
    {
        long bits = (long)residual.Length * (1 + k);
        for (int i = 0; i < residual.Length; i++)
        {
            uint u = residual[i] >= 0 ? (uint)(residual[i] << 1) : (uint)((-residual[i] << 1) - 1);
            bits += u >> k;
        }
        return bits;
    }
}

/// <summary>
/// Chosen LPC encode plan: order, quantization level, quantized coefficients,
/// precomputed residuals, Rice parameter, and total subframe bit count.
/// </summary>
internal sealed record FlacLpcChoice(
    int Order,
    int QuantLevel,
    int[] Coefficients,
    int[] Residual,
    int RiceParam,
    long TotalSubframeBits);
