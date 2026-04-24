// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// FLAC subframe body decoder. Matches libFLAC stream_decoder.c
// ::read_subframe_constant_ / read_subframe_verbatim_ /
// read_subframe_fixed_ / read_subframe_lpc_.
//
// Each subframe carries a single channel at a given bit depth. After decoding,
// the decoded samples have any "wasted" bits re-inserted on the low end via
// a left-shift - this reverses the encoder's lossless bit-depth reduction.

namespace SpawnDev.Codecs.Audio.Flac;

internal static class FlacSubframeDecoder
{
    /// <summary>
    /// Fixed-predictor coefficients (applied with no shift). Index is predictor order.
    /// Predictor for order k: sum(FixedCoefs[k][i] * samples[n-1-i] for i in 0..k-1).
    /// </summary>
    private static readonly int[][] FixedCoefs = new[]
    {
        Array.Empty<int>(),
        new[] { 1 },
        new[] { 2, -1 },
        new[] { 3, -3, 1 },
        new[] { 4, -6, 4, -1 },
    };

    /// <summary>
    /// Decode one subframe into <paramref name="samplesOut"/>. The output span
    /// length must equal the frame's block size.
    /// </summary>
    /// <param name="reader">Bit reader positioned at the start of this subframe.</param>
    /// <param name="samplesOut">Destination span, length = blockSize.</param>
    /// <param name="subframeBitsPerSample">
    /// Bit depth for this subframe's samples. Note: for L-side, R-side, or M-side
    /// stereo, the side channel is one bit wider than the frame's declared bit depth.
    /// </param>
    internal static void Decode(
        ref FlacBitReader reader,
        Span<int> samplesOut,
        int subframeBitsPerSample)
    {
        var hdr = FlacSubframeHeaderParser.Parse(ref reader);
        int effectiveBps = subframeBitsPerSample - hdr.WastedBitsPerSample;
        if (effectiveBps <= 0 || effectiveBps > 32)
            throw new InvalidDataException(
                $"Invalid effective bit depth {effectiveBps} = {subframeBitsPerSample} - {hdr.WastedBitsPerSample} wasted.");

        switch (hdr.Kind)
        {
            case FlacSubframeKind.Constant:
                DecodeConstant(ref reader, samplesOut, effectiveBps);
                break;
            case FlacSubframeKind.Verbatim:
                DecodeVerbatim(ref reader, samplesOut, effectiveBps);
                break;
            case FlacSubframeKind.Fixed:
                DecodeFixed(ref reader, samplesOut, effectiveBps, hdr.Order);
                break;
            case FlacSubframeKind.Lpc:
                DecodeLpc(ref reader, samplesOut, effectiveBps, hdr.Order);
                break;
            default:
                throw new InvalidDataException($"Unsupported subframe kind: {hdr.Kind}.");
        }

        if (hdr.WastedBitsPerSample > 0)
        {
            for (int i = 0; i < samplesOut.Length; i++)
                samplesOut[i] = samplesOut[i] << hdr.WastedBitsPerSample;
        }
    }

    private static void DecodeConstant(ref FlacBitReader reader, Span<int> samples, int bps)
    {
        int value = reader.ReadBitsSigned(bps);
        samples.Fill(value);
    }

    private static void DecodeVerbatim(ref FlacBitReader reader, Span<int> samples, int bps)
    {
        for (int i = 0; i < samples.Length; i++)
            samples[i] = reader.ReadBitsSigned(bps);
    }

    private static void DecodeFixed(ref FlacBitReader reader, Span<int> samples, int bps, int order)
    {
        if (order < 0 || order > FlacConstants.MaxFixedOrder)
            throw new InvalidDataException($"FIXED order {order} out of range.");
        // Warm-up samples.
        for (int i = 0; i < order; i++)
            samples[i] = reader.ReadBitsSigned(bps);
        // Residual into samples[order..].
        FlacResidualDecoder.Decode(ref reader, samples.Slice(order), samples.Length, order);
        // Reconstruct.
        var coefs = FixedCoefs[order];
        for (int n = order; n < samples.Length; n++)
        {
            long pred = 0;
            for (int i = 0; i < order; i++)
                pred += (long)coefs[i] * samples[n - 1 - i];
            samples[n] = (int)(samples[n] + pred);
        }
    }

    private static void DecodeLpc(ref FlacBitReader reader, Span<int> samples, int bps, int order)
    {
        if (order < 1 || order > FlacConstants.MaxLpcOrder)
            throw new InvalidDataException($"LPC order {order} out of range.");
        // Warm-up samples at subframe bps, signed.
        for (int i = 0; i < order; i++)
            samples[i] = reader.ReadBitsSigned(bps);
        // 4-bit QLP coefficient precision - 1; value 0b1111 is invalid per RFC 9639.
        int precMinusOne = (int)reader.ReadBits(4);
        if (precMinusOne == 0b1111)
            throw new InvalidDataException("LPC QLP coefficient precision 0b1111 is reserved.");
        int precision = precMinusOne + 1;
        // 5-bit signed quantization level (right-shift amount after the integer multiply).
        int quantLevel = reader.ReadBitsSigned(5);
        if (quantLevel < 0)
            throw new InvalidDataException($"LPC quantization level {quantLevel} is negative (libFLAC forbids).");
        // QLP coefficients (signed), MSB-first in the order they're applied.
        Span<int> coefs = order <= 64 ? stackalloc int[order] : new int[order];
        for (int i = 0; i < order; i++)
            coefs[i] = reader.ReadBitsSigned(precision);
        // Residual.
        FlacResidualDecoder.Decode(ref reader, samples.Slice(order), samples.Length, order);
        // Reconstruct.
        for (int n = order; n < samples.Length; n++)
        {
            long pred = 0;
            for (int i = 0; i < order; i++)
                pred += (long)coefs[i] * samples[n - 1 - i];
            samples[n] = (int)(samples[n] + (pred >> quantLevel));
        }
    }
}
