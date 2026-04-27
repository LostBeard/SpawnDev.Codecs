// End-to-end VP9 block encode pipeline tests.
//
// Pixels -> residual -> forward DCT -> quantize -> coef encode ->
// bit stream -> coef decode -> dequantize -> inverse DCT + add ->
// reconstructed pixels.
//
// This is the smallest unit that proves the entire VP9 encoder
// data path is wired end-to-end. The reconstructed pixels are
// compared against the original with a tolerance proportional to
// the quantizer index used for the test.

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    /// <summary>
    /// Drive a 4x4 block through the full VP9 encode + decode pipeline
    /// and return (reconstructed, encodedByteCount).
    /// </summary>
    private static (byte[] Reconstructed, int EncodedBytes) RoundTripVp9Block4x4(
        ReadOnlySpan<byte> pixels, int qIndex)
    {
        // Predictor = constant 128 (the simplest baseline; real codecs
        // would compute V/H/DC/TM/etc per-mode, but this isolates the
        // transform + quantize + entropy path from intra prediction).
        const byte predictorValue = 128;

        // Residual = pixels - predictor, packed row-major.
        Span<short> residual = stackalloc short[16];
        for (int i = 0; i < 16; i++) residual[i] = (short)(pixels[i] - predictorValue);

        // Forward DCT 4x4.
        Span<int> coefs = stackalloc int[16];
        Vp9ForwardDct4x4.Transform(residual, 4, coefs);

        // Quantize with the requested Q index.
        var planeQ = Vp9Dequantizer.PlaneQuantizer(qIndex, dcDelta: 0, acDelta: 0);
        Vp9ForwardQuantizer.QuantizeBlock(coefs, planeQ.Dc, planeQ.Ac);

        // Cast to short for the encoder (and the decoder's storage type).
        Span<short> coefsShort = stackalloc short[16];
        for (int i = 0; i < 16; i++)
        {
            int v = coefs[i];
            if (v > short.MaxValue || v < short.MinValue)
                throw new InvalidOperationException($"quantized coef {v} doesn't fit in short");
            coefsShort[i] = (short)v;
        }

        // Encode.
        var enc = new Vp9BoolEncoder();
        var coefsArray = coefsShort.ToArray();
        Vp9BlockCoefEncoder.EncodeBlockCoefficients(
            (prob, bit) => enc.Write(bit, prob),
            Vp9TxSize.Tx4x4, Vp9ScanType.Default,
            Vp9BlockCoefDecoder.PlaneType.Y, Vp9BlockCoefDecoder.RefType.Intra,
            coefsArray);
        byte[] encoded = enc.Stop();

        // Decode bits back to coefficients.
        var dec = new Vp9BoolDecoder(encoded, 0, encoded.Length);
        Span<short> decodedCoefs = stackalloc short[16];
        int eob = Vp9BlockCoefDecoder.DecodeBlockCoefficients(
            prob => dec.Read(prob),
            Vp9TxSize.Tx4x4, Vp9ScanType.Default,
            Vp9BlockCoefDecoder.PlaneType.Y, Vp9BlockCoefDecoder.RefType.Intra,
            decodedCoefs);

        // Verify the entropy round-trip is lossless.
        for (int i = 0; i < 16; i++)
        {
            if (coefsShort[i] != decodedCoefs[i])
                throw new Exception($"entropy round-trip mismatch at raster {i}: " +
                                    $"encoded {coefsShort[i]} decoded {decodedCoefs[i]}, eob={eob}");
        }

        // Dequantize.
        Vp9Dequantizer.DequantizeInPlace(decodedCoefs, planeQ);

        // Inverse DCT into a fresh predictor=128 buffer.
        var reconstructed = new byte[16];
        for (int i = 0; i < 16; i++) reconstructed[i] = predictorValue;
        Vp9Idct4x4Reference.Idct4x4_16_Add(decodedCoefs, reconstructed, stride: 4);

        return (reconstructed, encoded.Length);
    }

    [TestMethod]
    public void Vp9BlockEncodePipeline_FlatBlock_LowQ_ReconstructsExactly()
    {
        // Flat 128 block -> zero residual -> all-zero coefs -> 1-byte
        // bitstream (just the EOB bit + 32 trailing zeros). Reconstruction
        // is bit-exact (no quantization loss possible for zero coefs).
        var pixels = new byte[16];
        for (int i = 0; i < 16; i++) pixels[i] = 128;

        var (recon, _) = RoundTripVp9Block4x4(pixels, qIndex: 8);
        for (int i = 0; i < 16; i++) Equal((byte)128, recon[i]);
    }

    [TestMethod]
    public void Vp9BlockEncodePipeline_DcOffset_LowQ_LowError()
    {
        // Constant block != predictor -> single DC coefficient; round-trip
        // through quantize + entropy + dequantize + inverse DCT gives
        // back the same constant within a few units.
        var pixels = new byte[16];
        for (int i = 0; i < 16; i++) pixels[i] = 144;  // +16 above predictor

        var (recon, _) = RoundTripVp9Block4x4(pixels, qIndex: 8);

        int maxErr = 0;
        for (int i = 0; i < 16; i++)
            maxErr = Math.Max(maxErr, Math.Abs(recon[i] - pixels[i]));
        True(maxErr <= 2, $"DC-only block round-trip error = {maxErr}, expected <= 2");
    }

    [TestMethod]
    public void Vp9BlockEncodePipeline_Gradient_LowQ_LowError()
    {
        // Gradient block exercises every basis function. Low Q -> low
        // quantization error.
        var pixels = new byte[16];
        for (int r = 0; r < 4; r++)
            for (int c = 0; c < 4; c++)
                pixels[r * 4 + c] = (byte)(120 + r * 4 + c * 2);

        var (recon, _) = RoundTripVp9Block4x4(pixels, qIndex: 16);

        int maxErr = 0;
        for (int i = 0; i < 16; i++)
            maxErr = Math.Max(maxErr, Math.Abs(recon[i] - pixels[i]));
        True(maxErr <= 4, $"gradient round-trip error = {maxErr}, expected <= 4");
    }

    [TestMethod]
    public void Vp9BlockEncodePipeline_RandomBlocks_HighQ_BoundedError()
    {
        // High Q -> larger error per pixel. Bound at +/- 16 which is
        // generous for Q=80 on natural-ish data.
        var rng = new Random(0xB10C);
        var pixels = new byte[16];
        for (int trial = 0; trial < 4; trial++)
        {
            for (int i = 0; i < 16; i++) pixels[i] = (byte)rng.Next(80, 180);

            var (recon, _) = RoundTripVp9Block4x4(pixels, qIndex: 80);

            int maxErr = 0;
            for (int i = 0; i < 16; i++)
                maxErr = Math.Max(maxErr, Math.Abs(recon[i] - pixels[i]));
            True(maxErr <= 16, $"trial {trial}: random block error = {maxErr}, expected <= 16");
        }
    }

    [TestMethod]
    public void Vp9BlockEncodePipeline_AllZero_ProducesSingleEobByte()
    {
        // The all-zero block emits only the leading marker bit + the
        // single EOB bit + 32 zero flush bits. Stream should fit in a
        // few bytes (libvpx vpx_stop_encode emits up to 4 trailing
        // zero bytes after the flush).
        var pixels = new byte[16];
        for (int i = 0; i < 16; i++) pixels[i] = 128;

        var (_, encodedBytes) = RoundTripVp9Block4x4(pixels, qIndex: 8);
        True(encodedBytes <= 6,
            $"all-zero coef block encoded to {encodedBytes} bytes, expected <= 6");
    }

    /// <summary>
    /// 8x8 variant of the 4x4 pipeline: forward DCT 8x8 -> quantize ->
    /// coef encode -> bool stream -> coef decode -> dequantize ->
    /// inverse DCT 8x8 + add into fresh predictor=128 buffer.
    /// </summary>
    private static (byte[] Reconstructed, int EncodedBytes) RoundTripVp9Block8x8(
        ReadOnlySpan<byte> pixels, int qIndex)
    {
        const byte predictorValue = 128;

        Span<short> residual = stackalloc short[64];
        for (int i = 0; i < 64; i++) residual[i] = (short)(pixels[i] - predictorValue);

        Span<int> coefs = stackalloc int[64];
        Vp9ForwardDct8x8.Transform(residual, 8, coefs);

        var planeQ = Vp9Dequantizer.PlaneQuantizer(qIndex, dcDelta: 0, acDelta: 0);
        Vp9ForwardQuantizer.QuantizeBlock(coefs, planeQ.Dc, planeQ.Ac);

        var coefsShort = new short[64];
        for (int i = 0; i < 64; i++)
        {
            int v = coefs[i];
            if (v > short.MaxValue || v < short.MinValue)
                throw new InvalidOperationException($"quantized coef {v} doesn't fit in short");
            coefsShort[i] = (short)v;
        }

        var enc = new Vp9BoolEncoder();
        Vp9BlockCoefEncoder.EncodeBlockCoefficients(
            (prob, bit) => enc.Write(bit, prob),
            Vp9TxSize.Tx8x8, Vp9ScanType.Default,
            Vp9BlockCoefDecoder.PlaneType.Y, Vp9BlockCoefDecoder.RefType.Intra,
            coefsShort);
        byte[] encoded = enc.Stop();

        var dec = new Vp9BoolDecoder(encoded, 0, encoded.Length);
        Span<short> decoded = stackalloc short[64];
        Vp9BlockCoefDecoder.DecodeBlockCoefficients(
            prob => dec.Read(prob),
            Vp9TxSize.Tx8x8, Vp9ScanType.Default,
            Vp9BlockCoefDecoder.PlaneType.Y, Vp9BlockCoefDecoder.RefType.Intra,
            decoded);

        for (int i = 0; i < 64; i++)
            if (coefsShort[i] != decoded[i])
                throw new Exception($"8x8 entropy round-trip mismatch at {i}: enc {coefsShort[i]} dec {decoded[i]}");

        Vp9Dequantizer.DequantizeInPlace(decoded, planeQ);

        var reconstructed = new byte[64];
        for (int i = 0; i < 64; i++) reconstructed[i] = predictorValue;
        Vp9Idct8x8Reference.Idct8x8_64_Add(decoded, reconstructed, stride: 8);

        return (reconstructed, encoded.Length);
    }

    [TestMethod]
    public void Vp9BlockEncodePipeline_8x8_FlatBlock_LowQ_ReconstructsExactly()
    {
        var pixels = new byte[64];
        for (int i = 0; i < 64; i++) pixels[i] = 128;

        var (recon, _) = RoundTripVp9Block8x8(pixels, qIndex: 8);
        for (int i = 0; i < 64; i++) Equal((byte)128, recon[i]);
    }

    [TestMethod]
    public void Vp9BlockEncodePipeline_8x8_Gradient_LowQ_LowError()
    {
        var pixels = new byte[64];
        for (int r = 0; r < 8; r++)
            for (int c = 0; c < 8; c++)
                pixels[r * 8 + c] = (byte)(110 + r * 4 + c * 2);

        var (recon, _) = RoundTripVp9Block8x8(pixels, qIndex: 16);

        int maxErr = 0;
        for (int i = 0; i < 64; i++)
            maxErr = Math.Max(maxErr, Math.Abs(recon[i] - pixels[i]));
        True(maxErr <= 4, $"8x8 gradient round-trip error = {maxErr}, expected <= 4");
    }

    /// <summary>16x16 pipeline variant.</summary>
    private static (byte[] Reconstructed, int EncodedBytes) RoundTripVp9Block16x16(
        ReadOnlySpan<byte> pixels, int qIndex)
    {
        const byte predictorValue = 128;

        var residual = new short[256];
        for (int i = 0; i < 256; i++) residual[i] = (short)(pixels[i] - predictorValue);

        var coefs = new int[256];
        Vp9ForwardDct16x16.Transform(residual, 16, coefs);

        var planeQ = Vp9Dequantizer.PlaneQuantizer(qIndex, dcDelta: 0, acDelta: 0);
        Vp9ForwardQuantizer.QuantizeBlock(coefs, planeQ.Dc, planeQ.Ac);

        var coefsShort = new short[256];
        for (int i = 0; i < 256; i++)
        {
            int v = coefs[i];
            if (v > short.MaxValue || v < short.MinValue)
                throw new InvalidOperationException($"quantized coef {v} doesn't fit in short");
            coefsShort[i] = (short)v;
        }

        var enc = new Vp9BoolEncoder();
        Vp9BlockCoefEncoder.EncodeBlockCoefficients(
            (prob, bit) => enc.Write(bit, prob),
            Vp9TxSize.Tx16x16, Vp9ScanType.Default,
            Vp9BlockCoefDecoder.PlaneType.Y, Vp9BlockCoefDecoder.RefType.Intra,
            coefsShort);
        byte[] encoded = enc.Stop();

        var dec = new Vp9BoolDecoder(encoded, 0, encoded.Length);
        var decoded = new short[256];
        Vp9BlockCoefDecoder.DecodeBlockCoefficients(
            prob => dec.Read(prob),
            Vp9TxSize.Tx16x16, Vp9ScanType.Default,
            Vp9BlockCoefDecoder.PlaneType.Y, Vp9BlockCoefDecoder.RefType.Intra,
            decoded);

        for (int i = 0; i < 256; i++)
            if (coefsShort[i] != decoded[i])
                throw new Exception($"16x16 entropy round-trip mismatch at {i}: enc {coefsShort[i]} dec {decoded[i]}");

        Vp9Dequantizer.DequantizeInPlace(decoded, planeQ);

        var reconstructed = new byte[256];
        for (int i = 0; i < 256; i++) reconstructed[i] = predictorValue;
        Vp9Idct16x16Reference.Idct16x16_256_Add(decoded, reconstructed, stride: 16);

        return (reconstructed, encoded.Length);
    }

    [TestMethod]
    public void Vp9BlockEncodePipeline_16x16_FlatBlock_LowQ_ReconstructsExactly()
    {
        var pixels = new byte[256];
        for (int i = 0; i < 256; i++) pixels[i] = 128;

        var (recon, _) = RoundTripVp9Block16x16(pixels, qIndex: 8);
        for (int i = 0; i < 256; i++) Equal((byte)128, recon[i]);
    }

    [TestMethod]
    public void Vp9BlockEncodePipeline_16x16_Gradient_LowQ_LowError()
    {
        var pixels = new byte[256];
        for (int r = 0; r < 16; r++)
            for (int c = 0; c < 16; c++)
                pixels[r * 16 + c] = (byte)(100 + r * 2 + c);

        var (recon, _) = RoundTripVp9Block16x16(pixels, qIndex: 16);

        int maxErr = 0;
        for (int i = 0; i < 256; i++)
            maxErr = Math.Max(maxErr, Math.Abs(recon[i] - pixels[i]));
        True(maxErr <= 4, $"16x16 gradient round-trip error = {maxErr}, expected <= 4");
    }
}
