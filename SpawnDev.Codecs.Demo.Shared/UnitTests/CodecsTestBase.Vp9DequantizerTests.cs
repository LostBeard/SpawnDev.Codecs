// Tests for Vp9Dequantizer. The reference tables and helper math must
// match libvpx vp9_dc_quant / vp9_ac_quant / dequantize_b_q1 bit-exactly
// per the VP9 normative bitstream.

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9Dequantizer_LookupTables_HaveExactly256Entries()
    {
        Equal(256, Vp9Dequantizer.DcQLookup8.Length);
        Equal(256, Vp9Dequantizer.AcQLookup8.Length);
    }

    [TestMethod]
    public void Vp9Dequantizer_DcLookup_KnownEndpointsMatchSpec()
    {
        // Pinned values straight out of libvpx vp9_quant_common.c
        // dc_qlookup table - hand-verified against the C source.
        Equal((short)4,    Vp9Dequantizer.DcQLookup8[0]);
        Equal((short)8,    Vp9Dequantizer.DcQLookup8[1]);
        Equal((short)19,   Vp9Dequantizer.DcQLookup8[15]);
        Equal((short)85,   Vp9Dequantizer.DcQLookup8[95]);
        Equal((short)93,   Vp9Dequantizer.DcQLookup8[100]);
        Equal((short)1336, Vp9Dequantizer.DcQLookup8[255]);
    }

    [TestMethod]
    public void Vp9Dequantizer_AcLookup_KnownEndpointsMatchSpec()
    {
        Equal((short)4,    Vp9Dequantizer.AcQLookup8[0]);
        Equal((short)8,    Vp9Dequantizer.AcQLookup8[1]);
        Equal((short)22,   Vp9Dequantizer.AcQLookup8[15]);
        Equal((short)102,  Vp9Dequantizer.AcQLookup8[95]);
        Equal((short)112,  Vp9Dequantizer.AcQLookup8[100]);
        Equal((short)1828, Vp9Dequantizer.AcQLookup8[255]);
    }

    [TestMethod]
    public void Vp9Dequantizer_DcQuant_AppliesDeltaThenClamps()
    {
        // No delta - direct lookup.
        Equal((short)Vp9Dequantizer.DcQLookup8[42], Vp9Dequantizer.DcQuant(42, 0));
        // Positive delta within range - delta is added before lookup.
        Equal((short)Vp9Dequantizer.DcQLookup8[50], Vp9Dequantizer.DcQuant(42, 8));
        // Negative delta within range.
        Equal((short)Vp9Dequantizer.DcQLookup8[34], Vp9Dequantizer.DcQuant(42, -8));
        // Clamp low: q+delta < 0 -> index 0.
        Equal((short)Vp9Dequantizer.DcQLookup8[0], Vp9Dequantizer.DcQuant(0, -100));
        // Clamp high: q+delta > 255 -> index 255.
        Equal((short)Vp9Dequantizer.DcQLookup8[255], Vp9Dequantizer.DcQuant(200, 100));
    }

    [TestMethod]
    public void Vp9Dequantizer_AcQuant_AppliesDeltaThenClamps()
    {
        Equal((short)Vp9Dequantizer.AcQLookup8[42], Vp9Dequantizer.AcQuant(42, 0));
        Equal((short)Vp9Dequantizer.AcQLookup8[50], Vp9Dequantizer.AcQuant(42, 8));
        Equal((short)Vp9Dequantizer.AcQLookup8[34], Vp9Dequantizer.AcQuant(42, -8));
        Equal((short)Vp9Dequantizer.AcQLookup8[0], Vp9Dequantizer.AcQuant(0, -100));
        Equal((short)Vp9Dequantizer.AcQLookup8[255], Vp9Dequantizer.AcQuant(200, 100));
    }

    [TestMethod]
    public void Vp9Dequantizer_PlaneQuantizer_BuildsBothFromBaseAndDeltas()
    {
        var pq = Vp9Dequantizer.PlaneQuantizer(qIndex: 60, dcDelta: -2, acDelta: 3);
        Equal(Vp9Dequantizer.DcQuant(60, -2), pq.Dc);
        Equal(Vp9Dequantizer.AcQuant(60, 3), pq.Ac);
    }

    [TestMethod]
    public void Vp9Dequantizer_DequantizeInPlace_ZeroCoefficientsStayZero()
    {
        var coeffs = new short[16];
        var quant = new Vp9PlaneQuantizer(Dc: 17, Ac: 23);
        Vp9Dequantizer.DequantizeInPlace(coeffs, quant);
        for (int i = 0; i < 16; i++)
            Equal((short)0, coeffs[i]);
    }

    [TestMethod]
    public void Vp9Dequantizer_DequantizeInPlace_AppliesDcThenAc()
    {
        // First coefficient uses Dc, every subsequent coefficient uses Ac.
        var coeffs = new short[] { 3, 5, 7, 11 };
        var quant = new Vp9PlaneQuantizer(Dc: 100, Ac: 50);
        Vp9Dequantizer.DequantizeInPlace(coeffs, quant);
        Equal((short)(3 * 100), coeffs[0]);
        Equal((short)(5 * 50),  coeffs[1]);
        Equal((short)(7 * 50),  coeffs[2]);
        Equal((short)(11 * 50), coeffs[3]);
    }

    [TestMethod]
    public void Vp9Dequantizer_DequantizeInPlace_NegativeCoefficientsPreserveSign()
    {
        var coeffs = new short[] { -10, -20, 30 };
        var quant = new Vp9PlaneQuantizer(Dc: 7, Ac: 11);
        Vp9Dequantizer.DequantizeInPlace(coeffs, quant);
        Equal((short)(-10 * 7),  coeffs[0]);
        Equal((short)(-20 * 11), coeffs[1]);
        Equal((short)( 30 * 11), coeffs[2]);
    }

    [TestMethod]
    public void Vp9Dequantizer_DequantizeInPlace_SaturatesAtInt16Bounds()
    {
        // 4096 * 16 = 65536 > short.MaxValue (32767) -> clamps high.
        // -4096 * 16 = -65536 < short.MinValue (-32768) -> clamps low.
        var coeffs = new short[] { 4096, -4096, 100, -100 };
        var quant = new Vp9PlaneQuantizer(Dc: 16, Ac: 16);
        Vp9Dequantizer.DequantizeInPlace(coeffs, quant);
        Equal(short.MaxValue, coeffs[0]);
        Equal(short.MinValue, coeffs[1]);
        Equal((short)(100 * 16), coeffs[2]);
        Equal((short)(-100 * 16), coeffs[3]);
    }

    [TestMethod]
    public void Vp9Dequantizer_DequantizeInPlace_EmptySpanIsNoOp()
    {
        Span<short> empty = stackalloc short[0];
        var quant = new Vp9PlaneQuantizer(Dc: 10, Ac: 20);
        Vp9Dequantizer.DequantizeInPlace(empty, quant);
        // No assertions - the test passes if no exception is thrown.
    }

    [TestMethod]
    public void Vp9Dequantizer_DequantizeInPlace_FollowedByIdct4x4_RoundTripsThroughPipeline()
    {
        // End-to-end: build a quantized 4x4 block, dequantize, run iDCT 4x4
        // residual-add against a flat predictor. Verifies the dequant value
        // domain (int16) feeds the existing iDCT path correctly.
        //
        // Setup mirrors slice 117's DC-only fixture: a quantized DC of 16
        // with dc_quant=64 dequantizes to 1024, the same DC value that
        // Vp9Idct4x4_DcOnlyCoefficient_AppliesUniformResidual covers - and
        // produces a uniform +32 residual per pixel.
        var coeffs = new short[16];
        coeffs[0] = 16;
        var quant = new Vp9PlaneQuantizer(Dc: 64, Ac: 99);
        Vp9Dequantizer.DequantizeInPlace(coeffs, quant);
        Equal((short)1024, coeffs[0]);
        // Pure-AC values stayed zero (multiplied by 99 but were 0).
        for (int i = 1; i < 16; i++) Equal((short)0, coeffs[i]);

        var dest = new byte[16];
        for (int i = 0; i < 16; i++) dest[i] = 100;
        Vp9Idct4x4Reference.Idct4x4_16_Add(coeffs, dest, 4);
        for (int i = 0; i < 16; i++)
            Equal((byte)132, dest[i]); // 100 + 32 per slice 117 math
    }
}
