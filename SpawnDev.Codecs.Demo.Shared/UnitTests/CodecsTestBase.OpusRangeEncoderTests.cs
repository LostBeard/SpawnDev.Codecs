using SpawnDev.Codecs.EntropyCoders;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for <see cref="OpusRangeEncoder"/>, including round-trip tests that serve as
/// the primary correctness gate for both encoder and decoder. If a round-trip fails,
/// at least one of the two has a bug.
/// </summary>
public abstract partial class CodecsTestBase
{
    // -------- Argument / state guards --------

    [TestMethod]
    public void Encoder_Ctor_Capacity_AllocatesBuffer()
    {
        var enc = new OpusRangeEncoder(64);
        Equal(0u, enc.RangeBytes);
        Equal(0, enc.Error);
    }

    [TestMethod]
    public void Encoder_Ctor_NegativeCapacity_Throws()
    {
        Throws<ArgumentOutOfRangeException>(() => new OpusRangeEncoder(-1));
    }

    [TestMethod]
    public void Encoder_Ctor_NullBuffer_Throws()
    {
        Throws<ArgumentNullException>(() => new OpusRangeEncoder(null!, 0, 0));
    }

    [TestMethod]
    public void Encoder_Ctor_OffsetOutOfRange_Throws()
    {
        var buf = new byte[4];
        Throws<ArgumentOutOfRangeException>(() => new OpusRangeEncoder(buf, 5, 0));
    }

    [TestMethod]
    public void Encoder_Encode_AfterDone_Throws()
    {
        var enc = new OpusRangeEncoder(64);
        enc.EncodeBitLogP(1, 2);
        enc.Done();
        Throws<InvalidOperationException>(() => enc.EncodeBitLogP(1, 2));
    }

    [TestMethod]
    public void Encoder_ToArray_BeforeDone_Throws()
    {
        var enc = new OpusRangeEncoder(64);
        enc.EncodeBitLogP(1, 2);
        Throws<InvalidOperationException>(() => enc.ToArray());
    }

    [TestMethod]
    public void Encoder_Done_Twice_IsNoOp()
    {
        var enc = new OpusRangeEncoder(64);
        enc.EncodeBitLogP(1, 2);
        enc.Done();
        enc.Done();
        True(enc.IsDone);
    }

    // -------- Round-trip: the real correctness gate --------

    [TestMethod]
    public void Encoder_RoundTrip_SingleBitLogP()
    {
        var enc = new OpusRangeEncoder(64);
        enc.EncodeBitLogP(1, 3);
        enc.Done();

        var dec = new OpusRangeDecoder(enc.ToArray());
        Equal(1, dec.DecodeBitLogP(3));
    }

    [TestMethod]
    public void Encoder_RoundTrip_MultipleBitLogP()
    {
        var enc = new OpusRangeEncoder(128);
        int[] bits = { 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0 };
        int[] logps = { 2, 3, 1, 4, 2, 5, 3, 2, 1, 3, 4, 2, 3, 1, 2, 5 };
        for (int i = 0; i < bits.Length; i++) enc.EncodeBitLogP(bits[i], logps[i]);
        enc.Done();

        var dec = new OpusRangeDecoder(enc.ToArray());
        for (int i = 0; i < bits.Length; i++)
        {
            Equal(bits[i], dec.DecodeBitLogP(logps[i]), $"bit {i}");
        }
        Equal(0, dec.Error);
    }

    [TestMethod]
    public void Encoder_RoundTrip_IcdfSmallTable()
    {
        var enc = new OpusRangeEncoder(128);
        var icdf = new byte[] { 200, 100, 50, 0 };
        int[] symbols = { 0, 1, 2, 3, 1, 2, 0, 3, 2, 1 };
        foreach (int s in symbols) enc.EncodeIcdf(s, icdf, 8);
        enc.Done();

        var dec = new OpusRangeDecoder(enc.ToArray());
        foreach (int s in symbols) Equal(s, dec.DecodeIcdf(icdf, 8));
        Equal(0, dec.Error);
    }

    [TestMethod]
    public void Encoder_RoundTrip_Icdf16Wide()
    {
        var enc = new OpusRangeEncoder(256);
        var icdf = new ushort[] { 60000, 50000, 40000, 30000, 20000, 10000, 0 };
        int[] symbols = { 0, 6, 3, 1, 5, 2, 4, 0, 6, 3, 5, 1, 2, 4 };
        foreach (int s in symbols) enc.EncodeIcdf16(s, icdf, 16);
        enc.Done();

        var dec = new OpusRangeDecoder(enc.ToArray());
        foreach (int s in symbols) Equal(s, dec.DecodeIcdf16(icdf, 16));
    }

    [TestMethod]
    public void Encoder_RoundTrip_EncodeUpdatePair()
    {
        var enc = new OpusRangeEncoder(128);
        uint ft = 16;
        uint[] symbols = { 0, 5, 10, 15, 7, 3, 11, 2, 14, 8, 1, 9 };
        foreach (uint s in symbols) enc.Encode(s, s + 1u, ft);
        enc.Done();

        var dec = new OpusRangeDecoder(enc.ToArray());
        foreach (uint s in symbols)
        {
            Equal(s, dec.Decode(ft));
            dec.Update(s, s + 1u, ft);
        }
        Equal(0, dec.Error);
    }

    [TestMethod]
    public void Encoder_RoundTrip_EncodeBinDecodeBin()
    {
        var enc = new OpusRangeEncoder(128);
        int bits = 5;
        uint[] symbols = { 0, 31, 15, 1, 30, 7, 22, 13 };
        foreach (uint s in symbols) enc.EncodeBin(s, s + 1u, bits);
        enc.Done();

        var dec = new OpusRangeDecoder(enc.ToArray());
        foreach (uint s in symbols)
        {
            Equal(s, dec.DecodeBin(bits));
            dec.Update(s, s + 1u, 1u << bits);
        }
    }

    [TestMethod]
    public void Encoder_RoundTrip_RawBitsSmall()
    {
        var enc = new OpusRangeEncoder(128);
        uint[] values = { 5, 0, 15, 7, 1, 8, 14, 3 };
        foreach (uint v in values) enc.EncodeBits(v, 4);
        enc.Done();

        var dec = new OpusRangeDecoder(enc.ToArray());
        foreach (uint v in values) Equal(v, dec.DecodeBits(4));
    }

    [TestMethod]
    public void Encoder_RoundTrip_RawBitsLarge()
    {
        var enc = new OpusRangeEncoder(256);
        uint[] values = { 0x12345, 0xABCDE, 0x1FFFFFF, 0x0, 0x55555 };
        foreach (uint v in values) enc.EncodeBits(v, 25);
        enc.Done();

        var dec = new OpusRangeDecoder(enc.ToArray());
        foreach (uint v in values) Equal(v, dec.DecodeBits(25));
    }

    [TestMethod]
    public void Encoder_RoundTrip_UintSmall()
    {
        var enc = new OpusRangeEncoder(64);
        uint ft = 100;
        uint[] values = { 0, 50, 99, 17, 82, 3 };
        foreach (uint v in values) enc.EncodeUint(v, ft);
        enc.Done();

        var dec = new OpusRangeDecoder(enc.ToArray());
        foreach (uint v in values) Equal(v, dec.DecodeUint(ft));
    }

    [TestMethod]
    public void Encoder_RoundTrip_UintLarge()
    {
        var enc = new OpusRangeEncoder(256);
        uint ft = 1u << 20;
        uint[] values = { 0, ft - 1u, 0x12345, 0xABCDE, 0x7FFFF };
        foreach (uint v in values) enc.EncodeUint(v, ft);
        enc.Done();

        var dec = new OpusRangeDecoder(enc.ToArray());
        foreach (uint v in values) Equal(v, dec.DecodeUint(ft));
    }

    [TestMethod]
    public void Encoder_RoundTrip_MixedOperations()
    {
        var enc = new OpusRangeEncoder(512);
        var icdf = new byte[] { 180, 100, 30, 0 };

        enc.EncodeBitLogP(1, 2);
        enc.EncodeIcdf(2, icdf, 8);
        enc.EncodeBitLogP(0, 4);
        enc.Encode(7, 8, 16);
        enc.EncodeUint(42, 100);
        enc.EncodeIcdf(0, icdf, 8);
        enc.EncodeBin(5, 6, 3);
        enc.EncodeBits(0xABC, 12);
        enc.EncodeBitLogP(1, 1);
        enc.EncodeIcdf(3, icdf, 8);
        enc.Done();

        var dec = new OpusRangeDecoder(enc.ToArray());
        Equal(1, dec.DecodeBitLogP(2));
        Equal(2, dec.DecodeIcdf(icdf, 8));
        Equal(0, dec.DecodeBitLogP(4));
        Equal(7u, dec.Decode(16));
        dec.Update(7, 8, 16);
        Equal(42u, dec.DecodeUint(100));
        Equal(0, dec.DecodeIcdf(icdf, 8));
        Equal(5u, dec.DecodeBin(3));
        dec.Update(5, 6, 8);
        Equal(0xABCu, dec.DecodeBits(12));
        Equal(1, dec.DecodeBitLogP(1));
        Equal(3, dec.DecodeIcdf(icdf, 8));
        Equal(0, dec.Error);
    }

    [TestMethod]
    public void Encoder_RoundTrip_LongRandomSymbolStream()
    {
        var enc = new OpusRangeEncoder(1024);
        var icdf = new byte[] { 240, 200, 150, 100, 60, 30, 10, 0 };
        var rng = new Random(42);
        int[] symbols = new int[200];
        for (int i = 0; i < symbols.Length; i++) symbols[i] = rng.Next(0, icdf.Length);

        foreach (int s in symbols) enc.EncodeIcdf(s, icdf, 8);
        enc.Done();

        byte[] encoded = enc.ToArray();

        var dec1 = new OpusRangeDecoder(encoded);
        int[] decoded1 = new int[symbols.Length];
        for (int i = 0; i < symbols.Length; i++) decoded1[i] = dec1.DecodeIcdf(icdf, 8);
        EqualInts(symbols, decoded1, "first pass decode");

        var dec2 = new OpusRangeDecoder(encoded);
        int[] decoded2 = new int[symbols.Length];
        for (int i = 0; i < symbols.Length; i++) decoded2[i] = dec2.DecodeIcdf(icdf, 8);
        EqualInts(decoded1, decoded2, "second pass matches first");
    }

    [TestMethod]
    public void Encoder_Shrink_ValidSize_Succeeds()
    {
        var enc = new OpusRangeEncoder(256);
        enc.EncodeBits(0x5A, 8);
        enc.Shrink(128);
        enc.Done();

        var dec = new OpusRangeDecoder(enc.ToArray());
        Equal(0x5Au, dec.DecodeBits(8));
    }

    [TestMethod]
    public void Encoder_Shrink_TooSmall_Throws()
    {
        var enc = new OpusRangeEncoder(128);
        enc.EncodeBits(0xABCDEF, 24);
        enc.EncodeBits(0x123456, 24);
        enc.EncodeBits(0x789ABC, 24);
        Throws<ArgumentOutOfRangeException>(() => enc.Shrink(5));
    }

    [TestMethod]
    public void Encoder_PatchInitialBits_UpdatesFirstByte()
    {
        var enc = new OpusRangeEncoder(64);
        enc.EncodeBitLogP(0, 2);
        enc.EncodeBitLogP(1, 2);
        enc.PatchInitialBits(0b101, 3);
        enc.Done();

        var dec = new OpusRangeDecoder(enc.ToArray());
        Equal(1, dec.DecodeBitLogP(1));
        Equal(0, dec.DecodeBitLogP(1));
        Equal(1, dec.DecodeBitLogP(1));
    }
}
