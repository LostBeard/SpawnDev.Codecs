using SpawnDev.Codecs.EntropyCoders;

namespace SpawnDev.Codecs.Tests.EntropyCoders;

/// <summary>
/// Tests for the Opus range encoder. Includes round-trip tests (encode then decode)
/// that serve as the primary correctness gate for both encoder and decoder in
/// Phase 1a slice 2. If a round-trip fails, at least one of the two has a bug.
/// </summary>
public class OpusRangeEncoderTests
{
    // -------- Construction / argument validation --------

    [Fact]
    public void Ctor_Capacity_AllocatesBuffer()
    {
        var enc = new OpusRangeEncoder(64);
        Assert.Equal(0u, enc.RangeBytes);
        Assert.Equal(0, enc.Error);
    }

    [Fact]
    public void Ctor_NegativeCapacity_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new OpusRangeEncoder(-1));
    }

    [Fact]
    public void Ctor_NullBuffer_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => new OpusRangeEncoder(null!, 0, 0));
    }

    [Fact]
    public void Ctor_OffsetOutOfRange_Throws()
    {
        var buf = new byte[4];
        Assert.Throws<ArgumentOutOfRangeException>(() => new OpusRangeEncoder(buf, 5, 0));
    }

    // -------- Done() guards --------

    [Fact]
    public void Encode_AfterDone_Throws()
    {
        var enc = new OpusRangeEncoder(64);
        enc.EncodeBitLogP(1, 2);
        enc.Done();
        Assert.Throws<InvalidOperationException>(() => enc.EncodeBitLogP(1, 2));
    }

    [Fact]
    public void ToArray_BeforeDone_Throws()
    {
        var enc = new OpusRangeEncoder(64);
        enc.EncodeBitLogP(1, 2);
        Assert.Throws<InvalidOperationException>(() => enc.ToArray());
    }

    [Fact]
    public void Done_TwiceIsNoOp()
    {
        var enc = new OpusRangeEncoder(64);
        enc.EncodeBitLogP(1, 2);
        enc.Done();
        enc.Done(); // should not throw
        Assert.True(enc.IsDone);
    }

    // -------- Round-trip: the real correctness gate --------

    [Fact]
    public void RoundTrip_SingleBitLogP()
    {
        var enc = new OpusRangeEncoder(64);
        enc.EncodeBitLogP(1, 3);
        enc.Done();

        var dec = new OpusRangeDecoder(enc.ToArray());
        int decoded = dec.DecodeBitLogP(3);
        Assert.Equal(1, decoded);
    }

    [Fact]
    public void RoundTrip_MultipleBitLogP()
    {
        var enc = new OpusRangeEncoder(128);
        int[] bits = { 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0 };
        int[] logps = { 2, 3, 1, 4, 2, 5, 3, 2, 1, 3, 4, 2, 3, 1, 2, 5 };
        for (int i = 0; i < bits.Length; i++) enc.EncodeBitLogP(bits[i], logps[i]);
        enc.Done();

        var dec = new OpusRangeDecoder(enc.ToArray());
        for (int i = 0; i < bits.Length; i++)
        {
            int decoded = dec.DecodeBitLogP(logps[i]);
            Assert.Equal(expected: bits[i], actual: decoded);
        }
        Assert.Equal(0, dec.Error);
    }

    [Fact]
    public void RoundTrip_IcdfSmallTable()
    {
        var enc = new OpusRangeEncoder(128);
        var icdf = new byte[] { 200, 100, 50, 0 }; // 4-symbol table at ftb=8
        int[] symbols = { 0, 1, 2, 3, 1, 2, 0, 3, 2, 1 };
        foreach (int s in symbols) enc.EncodeIcdf(s, icdf, 8);
        enc.Done();

        var dec = new OpusRangeDecoder(enc.ToArray());
        foreach (int s in symbols)
        {
            int decoded = dec.DecodeIcdf(icdf, 8);
            Assert.Equal(s, decoded);
        }
        Assert.Equal(0, dec.Error);
    }

    [Fact]
    public void RoundTrip_Icdf16_Wide()
    {
        var enc = new OpusRangeEncoder(256);
        var icdf = new ushort[] { 60000, 50000, 40000, 30000, 20000, 10000, 0 };
        int[] symbols = { 0, 6, 3, 1, 5, 2, 4, 0, 6, 3, 5, 1, 2, 4 };
        foreach (int s in symbols) enc.EncodeIcdf16(s, icdf, 16);
        enc.Done();

        var dec = new OpusRangeDecoder(enc.ToArray());
        foreach (int s in symbols)
        {
            int decoded = dec.DecodeIcdf16(icdf, 16);
            Assert.Equal(s, decoded);
        }
    }

    [Fact]
    public void RoundTrip_EncodeDecodePair_PowerOfTwoFt()
    {
        var enc = new OpusRangeEncoder(128);
        uint ft = 16;
        uint[] symbols = { 0, 5, 10, 15, 7, 3, 11, 2, 14, 8, 1, 9 };
        foreach (uint s in symbols) enc.Encode(s, s + 1u, ft);
        enc.Done();

        var dec = new OpusRangeDecoder(enc.ToArray());
        foreach (uint s in symbols)
        {
            uint cum = dec.Decode(ft);
            Assert.InRange(cum, s, s);
            dec.Update(s, s + 1u, ft);
        }
        Assert.Equal(0, dec.Error);
    }

    [Fact]
    public void RoundTrip_EncodeBin_DecodeBin()
    {
        var enc = new OpusRangeEncoder(128);
        int bits = 5; // ft = 32
        uint[] symbols = { 0, 31, 15, 1, 30, 7, 22, 13 };
        foreach (uint s in symbols) enc.EncodeBin(s, s + 1u, bits);
        enc.Done();

        var dec = new OpusRangeDecoder(enc.ToArray());
        foreach (uint s in symbols)
        {
            uint cum = dec.DecodeBin(bits);
            Assert.Equal(s, cum);
            dec.Update(s, s + 1u, 1u << bits);
        }
    }

    [Fact]
    public void RoundTrip_RawBits_Small()
    {
        var enc = new OpusRangeEncoder(128);
        uint[] values = { 5, 0, 15, 7, 1, 8, 14, 3 };
        foreach (uint v in values) enc.EncodeBits(v, 4);
        enc.Done();

        var dec = new OpusRangeDecoder(enc.ToArray());
        foreach (uint v in values)
        {
            uint decoded = dec.DecodeBits(4);
            Assert.Equal(v, decoded);
        }
    }

    [Fact]
    public void RoundTrip_RawBits_Large()
    {
        var enc = new OpusRangeEncoder(256);
        uint[] values = { 0x12345, 0xABCDE, 0x1FFFFFF, 0x0, 0x55555 };
        foreach (uint v in values) enc.EncodeBits(v, 25);
        enc.Done();

        var dec = new OpusRangeDecoder(enc.ToArray());
        foreach (uint v in values)
        {
            uint decoded = dec.DecodeBits(25);
            Assert.Equal(v, decoded);
        }
    }

    [Fact]
    public void RoundTrip_Uint_Small()
    {
        var enc = new OpusRangeEncoder(64);
        uint ft = 100;
        uint[] values = { 0, 50, 99, 17, 82, 3 };
        foreach (uint v in values) enc.EncodeUint(v, ft);
        enc.Done();

        var dec = new OpusRangeDecoder(enc.ToArray());
        foreach (uint v in values)
        {
            uint decoded = dec.DecodeUint(ft);
            Assert.Equal(v, decoded);
        }
    }

    [Fact]
    public void RoundTrip_Uint_Large()
    {
        var enc = new OpusRangeEncoder(256);
        uint ft = 1u << 20;
        uint[] values = { 0, ft - 1u, 0x12345, 0xABCDE, 0x7FFFF };
        foreach (uint v in values) enc.EncodeUint(v, ft);
        enc.Done();

        var dec = new OpusRangeDecoder(enc.ToArray());
        foreach (uint v in values)
        {
            uint decoded = dec.DecodeUint(ft);
            Assert.Equal(v, decoded);
        }
    }

    [Fact]
    public void RoundTrip_MixedOperations()
    {
        var enc = new OpusRangeEncoder(512);
        var icdf = new byte[] { 180, 100, 30, 0 };

        // Arbitrary mixed sequence.
        enc.EncodeBitLogP(1, 2);
        enc.EncodeIcdf(2, icdf, 8);
        enc.EncodeBitLogP(0, 4);
        enc.Encode(7, 8, 16);
        enc.EncodeUint(42, 100);
        enc.EncodeIcdf(0, icdf, 8);
        enc.EncodeBin(5, 6, 3); // ft = 8, symbol 5
        enc.EncodeBits(0xABC, 12);
        enc.EncodeBitLogP(1, 1);
        enc.EncodeIcdf(3, icdf, 8);
        enc.Done();

        var dec = new OpusRangeDecoder(enc.ToArray());
        Assert.Equal(1, dec.DecodeBitLogP(2));
        Assert.Equal(2, dec.DecodeIcdf(icdf, 8));
        Assert.Equal(0, dec.DecodeBitLogP(4));

        uint cum = dec.Decode(16);
        Assert.Equal(7u, cum);
        dec.Update(7, 8, 16);

        Assert.Equal(42u, dec.DecodeUint(100));
        Assert.Equal(0, dec.DecodeIcdf(icdf, 8));

        uint cumBin = dec.DecodeBin(3);
        Assert.Equal(5u, cumBin);
        dec.Update(5, 6, 8);

        Assert.Equal(0xABCu, dec.DecodeBits(12));
        Assert.Equal(1, dec.DecodeBitLogP(1));
        Assert.Equal(3, dec.DecodeIcdf(icdf, 8));

        Assert.Equal(0, dec.Error);
    }

    [Fact]
    public void RoundTrip_LongSymbolStream_DeterministicBitExact()
    {
        var enc = new OpusRangeEncoder(1024);
        var icdf = new byte[] { 240, 200, 150, 100, 60, 30, 10, 0 };
        var rng = new Random(42);
        int[] symbols = new int[200];
        for (int i = 0; i < symbols.Length; i++) symbols[i] = rng.Next(0, icdf.Length);

        foreach (int s in symbols) enc.EncodeIcdf(s, icdf, 8);
        enc.Done();

        byte[] encoded = enc.ToArray();

        // First decode pass.
        var dec1 = new OpusRangeDecoder(encoded);
        int[] decoded1 = new int[symbols.Length];
        for (int i = 0; i < symbols.Length; i++) decoded1[i] = dec1.DecodeIcdf(icdf, 8);
        Assert.Equal(symbols, decoded1);

        // Second decode pass on the same bytes must match the first.
        var dec2 = new OpusRangeDecoder(encoded);
        int[] decoded2 = new int[symbols.Length];
        for (int i = 0; i < symbols.Length; i++) decoded2[i] = dec2.DecodeIcdf(icdf, 8);
        Assert.Equal(decoded1, decoded2);
    }

    // -------- Shrink + PatchInitialBits --------

    [Fact]
    public void Shrink_ValidSize_Succeeds()
    {
        var enc = new OpusRangeEncoder(256);
        enc.EncodeBits(0x5A, 8);
        enc.Shrink(128);
        enc.Done();

        var dec = new OpusRangeDecoder(enc.ToArray());
        Assert.Equal(0x5Au, dec.DecodeBits(8));
    }

    [Fact]
    public void Shrink_TooSmall_Throws()
    {
        var enc = new OpusRangeEncoder(128);
        // 3 calls of 24 bits each flush enough bytes to push _endOffs to 6.
        enc.EncodeBits(0xABCDEF, 24);
        enc.EncodeBits(0x123456, 24);
        enc.EncodeBits(0x789ABC, 24);
        Assert.Throws<ArgumentOutOfRangeException>(() => enc.Shrink(5));
    }

    [Fact]
    public void PatchInitialBits_BeforeAnyOutput_UpdatesFirstByte()
    {
        var enc = new OpusRangeEncoder(64);
        enc.EncodeBitLogP(0, 2);
        enc.EncodeBitLogP(1, 2);
        enc.PatchInitialBits(0b101, 3); // overwrite the top 3 bits of the first output byte
        enc.Done();

        // Patched bits are decoded first in the bitstream.
        var dec = new OpusRangeDecoder(enc.ToArray());
        Assert.Equal(1, dec.DecodeBitLogP(1));
        Assert.Equal(0, dec.DecodeBitLogP(1));
        Assert.Equal(1, dec.DecodeBitLogP(1));
    }
}
