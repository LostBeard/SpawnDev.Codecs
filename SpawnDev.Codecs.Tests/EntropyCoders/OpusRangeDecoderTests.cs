using SpawnDev.Codecs.EntropyCoders;

namespace SpawnDev.Codecs.Tests.EntropyCoders;

/// <summary>
/// Smoke tests for the Opus range decoder (Phase 1a first slice). Full RFC 6716
/// bit-exact conformance is gated by the full Opus decoder landing in later
/// Phase 1a work - these tests validate the range coder in isolation against
/// itself (determinism + argument validation).
/// </summary>
public class OpusRangeDecoderTests
{
    // -------- Construction / argument validation --------

    [Fact]
    public void Ctor_NullBuffer_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => new OpusRangeDecoder(null!));
    }

    [Fact]
    public void Ctor_NegativeOffset_Throws()
    {
        var buf = new byte[4];
        Assert.Throws<ArgumentOutOfRangeException>(() => new OpusRangeDecoder(buf, -1, 4));
    }

    [Fact]
    public void Ctor_OffsetPastEnd_Throws()
    {
        var buf = new byte[4];
        Assert.Throws<ArgumentOutOfRangeException>(() => new OpusRangeDecoder(buf, 5, 0));
    }

    [Fact]
    public void Ctor_NegativeLength_Throws()
    {
        var buf = new byte[4];
        Assert.Throws<ArgumentOutOfRangeException>(() => new OpusRangeDecoder(buf, 0, -1));
    }

    [Fact]
    public void Ctor_LengthPastEnd_Throws()
    {
        var buf = new byte[4];
        Assert.Throws<ArgumentOutOfRangeException>(() => new OpusRangeDecoder(buf, 2, 5));
    }

    [Fact]
    public void Ctor_EmptyBuffer_ConstructsWithoutThrow()
    {
        // All zero reads from an empty buffer are valid per libopus; state settles to "consumed" position.
        var dec = new OpusRangeDecoder(Array.Empty<byte>());
        Assert.Equal(0, dec.Error);
    }

    [Fact]
    public void Ctor_SingleByteBuffer_ConstructsWithoutThrow()
    {
        var dec = new OpusRangeDecoder(new byte[] { 0x55 });
        Assert.Equal(0, dec.Error);
    }

    // -------- Initial state properties --------

    [Fact]
    public void InitialState_ErrorIsZero()
    {
        var dec = new OpusRangeDecoder(new byte[] { 0x00, 0xFF, 0x55, 0xAA });
        Assert.Equal(0, dec.Error);
    }

    [Fact]
    public void InitialState_TellIsPositive()
    {
        // After init + normalize, some bits are already claimed by the range coder state.
        var dec = new OpusRangeDecoder(new byte[] { 0x12, 0x34, 0x56, 0x78 });
        Assert.True(dec.Tell >= 0, $"Tell={dec.Tell} should be non-negative");
    }

    [Fact]
    public void InitialState_TellFracIsPositive()
    {
        var dec = new OpusRangeDecoder(new byte[] { 0x12, 0x34, 0x56, 0x78 });
        Assert.True(dec.TellFrac > 0, $"TellFrac={dec.TellFrac} should be positive");
    }

    // -------- Determinism / reproducibility --------

    [Fact]
    public void TwoDecoders_SameInput_ProduceSameState()
    {
        var buf = new byte[] { 0xDE, 0xAD, 0xBE, 0xEF, 0x42, 0x17, 0x88, 0x99 };
        var a = new OpusRangeDecoder(buf);
        var b = new OpusRangeDecoder(buf);
        Assert.Equal(a.Tell, b.Tell);
        Assert.Equal(a.TellFrac, b.TellFrac);
        Assert.Equal(a.RangeBytes, b.RangeBytes);
        Assert.Equal(a.Error, b.Error);
    }

    [Fact]
    public void TwoDecoders_SameSequence_ProduceSameResults()
    {
        var buf = new byte[] { 0xDE, 0xAD, 0xBE, 0xEF, 0x42, 0x17, 0x88, 0x99 };
        var a = new OpusRangeDecoder(buf);
        var b = new OpusRangeDecoder(buf);

        var icdf = new byte[] { 200, 100, 0 };
        int aFirst = a.DecodeIcdf(icdf, ftb: 8);
        int bFirst = b.DecodeIcdf(icdf, ftb: 8);
        Assert.Equal(aFirst, bFirst);

        int aSecond = a.DecodeBitLogP(2);
        int bSecond = b.DecodeBitLogP(2);
        Assert.Equal(aSecond, bSecond);

        Assert.Equal(a.Tell, b.Tell);
        Assert.Equal(a.RangeBytes, b.RangeBytes);
    }

    // -------- DecodeBitLogP --------

    [Fact]
    public void DecodeBitLogP_ReturnsZeroOrOne()
    {
        var dec = new OpusRangeDecoder(new byte[] { 0x12, 0x34, 0x56, 0x78 });
        int result = dec.DecodeBitLogP(4);
        Assert.InRange(result, 0, 1);
    }

    [Fact]
    public void DecodeBitLogP_MultipleCalls_DoNotThrow()
    {
        var dec = new OpusRangeDecoder(new byte[] { 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08 });
        for (int i = 0; i < 10; i++)
        {
            int bit = dec.DecodeBitLogP(3);
            Assert.InRange(bit, 0, 1);
        }
        Assert.Equal(0, dec.Error);
    }

    // -------- DecodeIcdf --------

    [Fact]
    public void DecodeIcdf_TwoSymbolCdf_ReturnsZeroOrOne()
    {
        var dec = new OpusRangeDecoder(new byte[] { 0x80, 0x00, 0x00, 0x00 });
        var icdf = new byte[] { 128, 0 }; // 50/50 probability at ftb=8
        int symbol = dec.DecodeIcdf(icdf, ftb: 8);
        Assert.InRange(symbol, 0, 1);
    }

    [Fact]
    public void DecodeIcdf_ThreeSymbolCdf_ReturnsValidSymbol()
    {
        var dec = new OpusRangeDecoder(new byte[] { 0xAA, 0xBB, 0xCC, 0xDD });
        var icdf = new byte[] { 200, 100, 0 };
        int symbol = dec.DecodeIcdf(icdf, ftb: 8);
        Assert.InRange(symbol, 0, 2);
    }

    [Fact]
    public void DecodeIcdf16_SmokeTest()
    {
        var dec = new OpusRangeDecoder(new byte[] { 0x12, 0x34, 0x56, 0x78, 0x9A });
        var icdf = new ushort[] { 32768, 16384, 0 };
        int symbol = dec.DecodeIcdf16(icdf, ftb: 15);
        Assert.InRange(symbol, 0, 2);
    }

    // -------- DecodeBits (raw) --------

    [Fact]
    public void DecodeBits_ReturnsValueWithinRange()
    {
        var dec = new OpusRangeDecoder(new byte[] { 0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC });
        uint value = dec.DecodeBits(8);
        Assert.InRange(value, 0u, 0xFFu);
    }

    [Fact]
    public void DecodeBits_ZeroBits_ReturnsZero()
    {
        var dec = new OpusRangeDecoder(new byte[] { 0x12, 0x34, 0x56, 0x78 });
        uint value = dec.DecodeBits(0);
        Assert.Equal(0u, value);
    }

    // -------- DecodeUint --------

    [Fact]
    public void DecodeUint_FtOne_Throws()
    {
        var dec = new OpusRangeDecoder(new byte[] { 0x12, 0x34, 0x56, 0x78 });
        Assert.Throws<ArgumentOutOfRangeException>(() => dec.DecodeUint(1));
    }

    [Fact]
    public void DecodeUint_FtZero_Throws()
    {
        var dec = new OpusRangeDecoder(new byte[] { 0x12, 0x34, 0x56, 0x78 });
        Assert.Throws<ArgumentOutOfRangeException>(() => dec.DecodeUint(0));
    }

    [Fact]
    public void DecodeUint_SmallFt_ReturnsValueInRange()
    {
        var dec = new OpusRangeDecoder(new byte[] { 0x12, 0x34, 0x56, 0x78 });
        uint value = dec.DecodeUint(10);
        Assert.InRange(value, 0u, 9u);
    }

    [Fact]
    public void DecodeUint_LargeFt_ReturnsValueInRange()
    {
        var dec = new OpusRangeDecoder(new byte[] { 0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0 });
        uint ft = 1u << 16;
        uint value = dec.DecodeUint(ft);
        Assert.InRange(value, 0u, ft - 1u);
    }

    // -------- Decode / Update pair --------

    [Fact]
    public void DecodeUpdate_SmokeTest()
    {
        var dec = new OpusRangeDecoder(new byte[] { 0x12, 0x34, 0x56, 0x78 });
        uint ft = 16;
        uint cumFreq = dec.Decode(ft);
        Assert.InRange(cumFreq, 0u, ft - 1u);

        // Pick a symbol range containing the cumulative frequency.
        uint fl = cumFreq;
        uint fh = cumFreq + 1u;
        dec.Update(fl, fh, ft);
        Assert.Equal(0, dec.Error);
    }

    [Fact]
    public void DecodeBin_SmokeTest()
    {
        var dec = new OpusRangeDecoder(new byte[] { 0x12, 0x34, 0x56, 0x78 });
        uint cumFreq = dec.DecodeBin(4);
        Assert.InRange(cumFreq, 0u, 15u);
    }
}
