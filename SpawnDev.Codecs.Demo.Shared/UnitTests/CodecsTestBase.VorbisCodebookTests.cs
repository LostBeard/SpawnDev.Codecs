using SpawnDev.Codecs.Audio.Vorbis;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for <see cref="VorbisMath"/> and <see cref="VorbisCodebookParser"/>.
/// Each codebook test hand-packs a valid bit pattern using a private
/// LSB-first writer and parses it back through the production reader.
/// </summary>
public abstract partial class CodecsTestBase
{
    // -------- VorbisMath --------

    [TestMethod]
    public void VorbisMath_Ilog_KnownValues()
    {
        Equal(0, VorbisMath.Ilog(0));
        Equal(1, VorbisMath.Ilog(1));
        Equal(2, VorbisMath.Ilog(2));
        Equal(2, VorbisMath.Ilog(3));
        Equal(3, VorbisMath.Ilog(4));
        Equal(3, VorbisMath.Ilog(7));
        Equal(4, VorbisMath.Ilog(8));
        Equal(4, VorbisMath.Ilog(15));
        Equal(5, VorbisMath.Ilog(16));
    }

    [TestMethod]
    public void VorbisMath_Lookup1Values_KnownGeometries()
    {
        // lookup1_values(64, 2) = 8 because 8^2 = 64.
        Equal(8, VorbisMath.Lookup1Values(64, 2));
        // lookup1_values(27, 3) = 3 because 3^3 = 27.
        Equal(3, VorbisMath.Lookup1Values(27, 3));
        // lookup1_values(10, 2) = 3 because 3^2 = 9 <= 10 < 16 = 4^2.
        Equal(3, VorbisMath.Lookup1Values(10, 2));
        // Single-dimension: trivially equals entries.
        Equal(100, VorbisMath.Lookup1Values(100, 1));
    }

    [TestMethod]
    public void VorbisMath_Float32Unpack_RoundNumbers()
    {
        // 0 is trivially 0.
        Equal(0.0, VorbisMath.Float32Unpack(0));
        // mantissa=1, exponent=788 -> 1 * 2^0 = 1.0
        // packing: sign 0 | exponent 788 << 21 | mantissa 1
        uint packedOne = (788u << 21) | 1u;
        Equal(1.0, VorbisMath.Float32Unpack(packedOne));
        // Negative: sign 1 | same fields -> -1.0
        Equal(-1.0, VorbisMath.Float32Unpack(0x80000000u | packedOne));
    }

    // -------- Codebook parser --------

    /// <summary>Minimal LSB-first bit writer for building Vorbis test fixtures.</summary>
    private sealed class VorbisTestWriter
    {
        private readonly List<byte> _bytes = new();
        private int _currentByte;
        private int _bitPos;

        public void Write(uint value, int bits)
        {
            for (int i = 0; i < bits; i++)
            {
                uint bit = (value >> i) & 1u;
                _currentByte |= (int)(bit << _bitPos);
                _bitPos++;
                if (_bitPos == 8)
                {
                    _bytes.Add((byte)_currentByte);
                    _currentByte = 0;
                    _bitPos = 0;
                }
            }
        }

        public byte[] ToArray()
        {
            if (_bitPos > 0) _bytes.Add((byte)_currentByte);
            return _bytes.ToArray();
        }
    }

    private static VorbisCodebook ParseCodebook(byte[] data)
    {
        var r = new VorbisBitReader(data);
        return VorbisCodebookParser.Parse(ref r);
    }

    [TestMethod]
    public void VorbisCodebook_Unordered_Dense_4Entries_2BitEach()
    {
        // dimensions=1, entries=4, ordered=0, sparse=0, each length=2 (write 1),
        // lookup_type=0.
        var w = new VorbisTestWriter();
        w.Write(0x564342, 24);  // sync
        w.Write(1, 16);          // dimensions
        w.Write(4, 24);          // entries
        w.Write(0, 1);           // ordered
        w.Write(0, 1);           // sparse
        for (int i = 0; i < 4; i++) w.Write(1, 5); // length-1 = 1, so length = 2
        w.Write(0, 4);           // lookup_type = 0
        var cb = ParseCodebook(w.ToArray());
        Equal(1, cb.Dimensions);
        Equal(4, cb.Entries);
        False(cb.Ordered);
        False(cb.Sparse);
        EqualInts(new[] { 2, 2, 2, 2 }, cb.Lengths);
        Equal(0, cb.LookupType);
    }

    [TestMethod]
    public void VorbisCodebook_Sparse_SomeEntriesUnused()
    {
        // dimensions=1, entries=3, ordered=0, sparse=1
        // entry 0: used, length 3 -> flag=1, length-1=2
        // entry 1: unused        -> flag=0
        // entry 2: used, length 5 -> flag=1, length-1=4
        var w = new VorbisTestWriter();
        w.Write(0x564342, 24);
        w.Write(1, 16);
        w.Write(3, 24);
        w.Write(0, 1);
        w.Write(1, 1); // sparse
        w.Write(1, 1); w.Write(2, 5); // entry 0 used, length 3
        w.Write(0, 1);                 // entry 1 unused
        w.Write(1, 1); w.Write(4, 5); // entry 2 used, length 5
        w.Write(0, 4);
        var cb = ParseCodebook(w.ToArray());
        Equal(3, cb.Entries);
        True(cb.Sparse);
        EqualInts(new[] { 3, 0, 5 }, cb.Lengths);
    }

    [TestMethod]
    public void VorbisCodebook_Ordered_MonotonicLengths()
    {
        // dimensions=1, entries=5, ordered=1.
        // initial length = 2 (write 1), run lengths: 2 entries of length 2, 3 entries of length 3.
        // ilog(5 - 0) = 3 bits to encode "2" (first run count)
        // ilog(5 - 2) = 2 bits to encode "3" (second run count)
        var w = new VorbisTestWriter();
        w.Write(0x564342, 24);
        w.Write(1, 16);
        w.Write(5, 24);
        w.Write(1, 1);       // ordered
        w.Write(1, 5);       // initial length - 1 = 1, so length = 2
        w.Write(2, 3);       // 2 entries at length 2 (ilog(5)=3)
        w.Write(3, 2);       // 3 entries at length 3 (ilog(5-2)=ilog(3)=2)
        w.Write(0, 4);       // lookup_type = 0
        var cb = ParseCodebook(w.ToArray());
        True(cb.Ordered);
        EqualInts(new[] { 2, 2, 3, 3, 3 }, cb.Lengths);
    }

    [TestMethod]
    public void VorbisCodebook_LookupType2_MultiplicandsParsed()
    {
        // dimensions=2, entries=3, ordered=0, sparse=0, lengths=[1,2,2]
        // lookup_type=2 -> multiplicand count = 3*2 = 6
        // min_value and delta_value packed as zero floats.
        // value_bits = 4 (write 3), sequence_p = 0
        // multiplicands: 1, 2, 3, 4, 5, 6 (each 4 bits)
        var w = new VorbisTestWriter();
        w.Write(0x564342, 24);
        w.Write(2, 16);
        w.Write(3, 24);
        w.Write(0, 1); // ordered
        w.Write(0, 1); // sparse
        w.Write(0, 5); // length-1=0 for entry 0 -> length 1
        w.Write(1, 5); // length-1=1 for entry 1 -> length 2
        w.Write(1, 5); // length-1=1 for entry 2 -> length 2
        w.Write(2, 4); // lookup_type
        w.Write(0, 32);            // min_value (0.0)
        w.Write(0, 32);            // delta_value (0.0)
        w.Write(3, 4);             // value_bits - 1 = 3 -> 4 bits
        w.Write(0, 1);             // sequence_p = 0
        for (int i = 1; i <= 6; i++) w.Write((uint)i, 4);
        var cb = ParseCodebook(w.ToArray());
        Equal(2, cb.LookupType);
        Equal(4, cb.ValueBits);
        False(cb.SequenceP);
        EqualInts(new[] { 1, 2, 3, 4, 5, 6 }, cb.Multiplicands);
    }

    [TestMethod]
    public void VorbisCodebook_BadSync_Throws()
    {
        var w = new VorbisTestWriter();
        w.Write(0xABCDEF, 24);
        bool threw = false;
        try { ParseCodebook(w.ToArray()); } catch (InvalidDataException) { threw = true; }
        True(threw);
    }

    [TestMethod]
    public void VorbisCodebook_ReservedLookupType_Throws()
    {
        var w = new VorbisTestWriter();
        w.Write(0x564342, 24);
        w.Write(1, 16);
        w.Write(1, 24);
        w.Write(0, 1);       // ordered
        w.Write(0, 1);       // sparse
        w.Write(0, 5);       // length
        w.Write(3, 4);       // reserved lookup type (3..15 invalid)
        bool threw = false;
        try { ParseCodebook(w.ToArray()); } catch (InvalidDataException) { threw = true; }
        True(threw);
    }
}
