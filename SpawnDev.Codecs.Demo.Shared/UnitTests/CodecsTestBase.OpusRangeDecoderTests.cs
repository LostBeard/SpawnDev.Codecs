using SpawnDev.Codecs.EntropyCoders;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Smoke tests for <see cref="OpusRangeDecoder"/>. Full bit-exact RFC 6716 conformance
/// lives in separate vector-driven tests; this set validates constructor argument handling,
/// initial state, determinism, and per-operation in-range behavior.
/// </summary>
public abstract partial class CodecsTestBase
{
    [TestMethod(Category = "CpuOnly")]
    public void Decoder_Ctor_NullBuffer_Throws()
    {
        CpuOnlyTest();
        Throws<ArgumentNullException>(() => new OpusRangeDecoder(null!));
    }

    [TestMethod(Category = "CpuOnly")]
    public void Decoder_Ctor_NegativeOffset_Throws()
    {
        CpuOnlyTest();
        var buf = new byte[4];
        Throws<ArgumentOutOfRangeException>(() => new OpusRangeDecoder(buf, -1, 4));
    }

    [TestMethod(Category = "CpuOnly")]
    public void Decoder_Ctor_OffsetPastEnd_Throws()
    {
        CpuOnlyTest();
        var buf = new byte[4];
        Throws<ArgumentOutOfRangeException>(() => new OpusRangeDecoder(buf, 5, 0));
    }

    [TestMethod(Category = "CpuOnly")]
    public void Decoder_Ctor_NegativeLength_Throws()
    {
        CpuOnlyTest();
        var buf = new byte[4];
        Throws<ArgumentOutOfRangeException>(() => new OpusRangeDecoder(buf, 0, -1));
    }

    [TestMethod(Category = "CpuOnly")]
    public void Decoder_Ctor_LengthPastEnd_Throws()
    {
        CpuOnlyTest();
        var buf = new byte[4];
        Throws<ArgumentOutOfRangeException>(() => new OpusRangeDecoder(buf, 2, 5));
    }

    [TestMethod(Category = "CpuOnly")]
    public void Decoder_Ctor_EmptyBuffer_ConstructsWithoutThrow()
    {
        CpuOnlyTest();
        var dec = new OpusRangeDecoder(Array.Empty<byte>());
        Equal(0, dec.Error, "Error after empty-buffer init");
    }

    [TestMethod(Category = "CpuOnly")]
    public void Decoder_Ctor_SingleByteBuffer_ConstructsWithoutThrow()
    {
        CpuOnlyTest();
        var dec = new OpusRangeDecoder(new byte[] { 0x55 });
        Equal(0, dec.Error);
    }

    [TestMethod(Category = "CpuOnly")]
    public void Decoder_InitialState_ErrorIsZero()
    {
        CpuOnlyTest();
        var dec = new OpusRangeDecoder(new byte[] { 0x00, 0xFF, 0x55, 0xAA });
        Equal(0, dec.Error);
    }

    [TestMethod(Category = "CpuOnly")]
    public void Decoder_InitialState_TellIsNonNegative()
    {
        CpuOnlyTest();
        var dec = new OpusRangeDecoder(new byte[] { 0x12, 0x34, 0x56, 0x78 });
        True(dec.Tell >= 0, $"Tell={dec.Tell} should be non-negative");
    }

    [TestMethod(Category = "CpuOnly")]
    public void Decoder_InitialState_TellFracIsPositive()
    {
        CpuOnlyTest();
        var dec = new OpusRangeDecoder(new byte[] { 0x12, 0x34, 0x56, 0x78 });
        True(dec.TellFrac > 0, $"TellFrac={dec.TellFrac} should be positive");
    }

    [TestMethod(Category = "CpuOnly")]
    public void Decoder_TwoInstances_SameInput_ProduceSameInitialState()
    {
        CpuOnlyTest();
        var buf = new byte[] { 0xDE, 0xAD, 0xBE, 0xEF, 0x42, 0x17, 0x88, 0x99 };
        var a = new OpusRangeDecoder(buf);
        var b = new OpusRangeDecoder(buf);
        Equal(a.Tell, b.Tell, "Tell");
        Equal(a.TellFrac, b.TellFrac, "TellFrac");
        Equal(a.RangeBytes, b.RangeBytes, "RangeBytes");
        Equal(a.Error, b.Error, "Error");
    }

    [TestMethod(Category = "CpuOnly")]
    public void Decoder_TwoInstances_SameSequence_ProduceSameResults()
    {
        CpuOnlyTest();
        var buf = new byte[] { 0xDE, 0xAD, 0xBE, 0xEF, 0x42, 0x17, 0x88, 0x99 };
        var a = new OpusRangeDecoder(buf);
        var b = new OpusRangeDecoder(buf);

        var icdf = new byte[] { 200, 100, 0 };
        Equal(a.DecodeIcdf(icdf, 8), b.DecodeIcdf(icdf, 8), "icdf step");
        Equal(a.DecodeBitLogP(2), b.DecodeBitLogP(2), "bitLogP step");
        Equal(a.Tell, b.Tell, "Tell after 2 steps");
        Equal(a.RangeBytes, b.RangeBytes, "RangeBytes after 2 steps");
    }

    [TestMethod(Category = "CpuOnly")]
    public void Decoder_DecodeBitLogP_ReturnsZeroOrOne()
    {
        CpuOnlyTest();
        var dec = new OpusRangeDecoder(new byte[] { 0x12, 0x34, 0x56, 0x78 });
        int result = dec.DecodeBitLogP(4);
        InRange(result, 0, 1);
    }

    [TestMethod(Category = "CpuOnly")]
    public void Decoder_DecodeBitLogP_MultipleCalls_ProduceValidBits()
    {
        CpuOnlyTest();
        var dec = new OpusRangeDecoder(new byte[] { 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08 });
        for (int i = 0; i < 10; i++)
        {
            int bit = dec.DecodeBitLogP(3);
            InRange(bit, 0, 1);
        }
        Equal(0, dec.Error, "Error after 10 decode-bit calls");
    }

    [TestMethod(Category = "CpuOnly")]
    public void Decoder_DecodeIcdf_TwoSymbolCdf_ReturnsValidSymbol()
    {
        CpuOnlyTest();
        var dec = new OpusRangeDecoder(new byte[] { 0x80, 0x00, 0x00, 0x00 });
        var icdf = new byte[] { 128, 0 };
        InRange(dec.DecodeIcdf(icdf, 8), 0, 1);
    }

    [TestMethod(Category = "CpuOnly")]
    public void Decoder_DecodeIcdf_ThreeSymbolCdf_ReturnsValidSymbol()
    {
        CpuOnlyTest();
        var dec = new OpusRangeDecoder(new byte[] { 0xAA, 0xBB, 0xCC, 0xDD });
        var icdf = new byte[] { 200, 100, 0 };
        InRange(dec.DecodeIcdf(icdf, 8), 0, 2);
    }

    [TestMethod(Category = "CpuOnly")]
    public void Decoder_DecodeIcdf16_SmokeTest()
    {
        CpuOnlyTest();
        var dec = new OpusRangeDecoder(new byte[] { 0x12, 0x34, 0x56, 0x78, 0x9A });
        var icdf = new ushort[] { 32768, 16384, 0 };
        InRange(dec.DecodeIcdf16(icdf, 15), 0, 2);
    }

    [TestMethod(Category = "CpuOnly")]
    public void Decoder_DecodeBits_ReturnsValueWithinRange()
    {
        CpuOnlyTest();
        var dec = new OpusRangeDecoder(new byte[] { 0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC });
        uint value = dec.DecodeBits(8);
        InRange(value, 0u, 0xFFu);
    }

    [TestMethod(Category = "CpuOnly")]
    public void Decoder_DecodeBits_ZeroBits_ReturnsZero()
    {
        CpuOnlyTest();
        var dec = new OpusRangeDecoder(new byte[] { 0x12, 0x34, 0x56, 0x78 });
        Equal(0u, dec.DecodeBits(0));
    }

    [TestMethod(Category = "CpuOnly")]
    public void Decoder_DecodeUint_FtOne_Throws()
    {
        CpuOnlyTest();
        var dec = new OpusRangeDecoder(new byte[] { 0x12, 0x34, 0x56, 0x78 });
        Throws<ArgumentOutOfRangeException>(() => dec.DecodeUint(1));
    }

    [TestMethod(Category = "CpuOnly")]
    public void Decoder_DecodeUint_FtZero_Throws()
    {
        CpuOnlyTest();
        var dec = new OpusRangeDecoder(new byte[] { 0x12, 0x34, 0x56, 0x78 });
        Throws<ArgumentOutOfRangeException>(() => dec.DecodeUint(0));
    }

    [TestMethod(Category = "CpuOnly")]
    public void Decoder_DecodeUint_SmallFt_ReturnsValueInRange()
    {
        CpuOnlyTest();
        var dec = new OpusRangeDecoder(new byte[] { 0x12, 0x34, 0x56, 0x78 });
        InRange(dec.DecodeUint(10), 0u, 9u);
    }

    [TestMethod(Category = "CpuOnly")]
    public void Decoder_DecodeUint_LargeFt_ReturnsValueInRange()
    {
        CpuOnlyTest();
        var dec = new OpusRangeDecoder(new byte[] { 0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0 });
        uint ft = 1u << 16;
        InRange(dec.DecodeUint(ft), 0u, ft - 1u);
    }

    [TestMethod(Category = "CpuOnly")]
    public void Decoder_DecodeUpdate_SmokeTest()
    {
        CpuOnlyTest();
        var dec = new OpusRangeDecoder(new byte[] { 0x12, 0x34, 0x56, 0x78 });
        uint ft = 16;
        uint cumFreq = dec.Decode(ft);
        InRange(cumFreq, 0u, ft - 1u);
        dec.Update(cumFreq, cumFreq + 1u, ft);
        Equal(0, dec.Error);
    }

    [TestMethod(Category = "CpuOnly")]
    public void Decoder_DecodeBin_SmokeTest()
    {
        CpuOnlyTest();
        var dec = new OpusRangeDecoder(new byte[] { 0x12, 0x34, 0x56, 0x78 });
        InRange(dec.DecodeBin(4), 0u, 15u);
    }
}
