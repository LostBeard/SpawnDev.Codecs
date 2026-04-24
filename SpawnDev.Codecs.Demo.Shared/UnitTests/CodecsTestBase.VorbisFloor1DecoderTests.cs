using SpawnDev.Codecs.Audio.Vorbis;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for <see cref="VorbisFloor1Decoder"/>. Each test builds a minimal
/// floor 1 config + codebooks, hand-packs an audio-packet floor-decode
/// snippet, runs <see cref="VorbisFloor1Decoder.Decode"/> and asserts the
/// returned Y array matches.
/// </summary>
public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void VorbisFloor1Decoder_NonzeroFlagClear_ReturnsNull()
    {
        // "Nonzero" bit 0 -> decoder returns null (silent floor), no further reads.
        var cfg = new VorbisFloor1Config
        {
            Partitions = 0,
            PartitionClassList = Array.Empty<int>(),
            ClassDimensions = Array.Empty<int>(),
            ClassSubclasses = Array.Empty<int>(),
            ClassMasterbooks = Array.Empty<int>(),
            ClassSubclassBooks = Array.Empty<int[]>(),
            Multiplier = 1,
            RangeBits = 4,
            XList = new[] { 0, 16 },
        };
        var w = new VorbisTestWriter();
        w.Write(0, 1);            // nonzero = 0
        var reader = new VorbisBitReader(w.ToArray());
        var y = VorbisFloor1Decoder.Decode(ref reader, cfg, Array.Empty<VorbisHuffmanDecoder>());
        True(y is null, "Silent-floor packet should return null.");
    }

    [TestMethod]
    public void VorbisFloor1Decoder_SinglePartition_NoMasterbook_ReadsYValues()
    {
        // 1 partition, 1 class, class dimensions 1, subclasses 0.
        // Subclass book 0 = codebook 0 (4-entry len-2 codebook, codes 00/01/10/11 -> entries 0/1/2/3).
        var cfg = new VorbisFloor1Config
        {
            Partitions = 1,
            PartitionClassList = new[] { 0 },
            ClassDimensions = new[] { 1 },
            ClassSubclasses = new[] { 0 },
            ClassMasterbooks = new[] { -1 },
            ClassSubclassBooks = new[] { new[] { 0 } },
            Multiplier = 1,
            RangeBits = 4,
            XList = new[] { 0, 16, 7 },
        };
        var cb = VorbisHuffman.Build(new[] { 2, 2, 2, 2 });
        var decoders = new[] { new VorbisHuffmanDecoder(cb) };

        // Packet bits: nonzero=1, y[0]=100 (8 bits), y[1]=50 (8 bits),
        // Huffman code for entry 2 at 2 bits = code value 2 = MSB-first "10"
        // (on wire: first bit 1, second bit 0).
        var w = new VorbisTestWriter();
        w.Write(1, 1);            // nonzero
        w.Write(100, 8);          // y[0]
        w.Write(50, 8);           // y[1]
        WriteHuffmanCodeword(w, cb.Codewords[2], cb.EntryLengths[2]);
        var reader = new VorbisBitReader(w.ToArray());
        var y = VorbisFloor1Decoder.Decode(ref reader, cfg, decoders);
        NotNull(y);
        EqualInts(new[] { 100, 50, 2 }, y!);
    }

    [TestMethod]
    public void VorbisFloor1Decoder_ClassSubclasses_ReadsMasterbookFirst()
    {
        // Class with subclasses=1 (2 subclass slots). Masterbook returns a 2-bit
        // cval; low bit indexes the subclass table.
        //   subclass[0] = codebook index 1 (4-entry 2-bit codebook)
        //   subclass[1] = -1 (force Y = 0)
        // We expect: masterbook Huffman -> cval; then per-dim:
        //   dim 0: book = subclass[cval & 1]; shift cval; read Huffman if book != -1, else Y = 0.
        //   dim 1: book = subclass[cval & 1]; shift cval; same.
        // Let class dimensions = 2, subclasses = 1 (so csub = 0b01 = 1, cbits = 1).
        var cfg = new VorbisFloor1Config
        {
            Partitions = 1,
            PartitionClassList = new[] { 0 },
            ClassDimensions = new[] { 2 },
            ClassSubclasses = new[] { 1 },
            ClassMasterbooks = new[] { 0 }, // codebook 0
            ClassSubclassBooks = new[] { new[] { 1, -1 } },
            Multiplier = 1,
            RangeBits = 4,
            XList = new[] { 0, 16, 3, 5 },
        };
        // codebook 0: masterbook, 2-entry length-1 codebook -> codes 0 and 1 -> entries 0 and 1.
        var masterCb = VorbisHuffman.Build(new[] { 1, 1 });
        // codebook 1: a 4-entry len-2 codebook for subclass 0.
        var valueCb = VorbisHuffman.Build(new[] { 2, 2, 2, 2 });
        var decoders = new[]
        {
            new VorbisHuffmanDecoder(masterCb),
            new VorbisHuffmanDecoder(valueCb),
        };

        // Choose: masterbook returns entry 1 -> cval = 1 (binary 01).
        // After dim 0: cval & 1 = 1 -> subclass index 1 -> book = -1 -> y[2] = 0. Shift: cval = 0.
        // After dim 1: cval & 1 = 0 -> subclass index 0 -> book = 1 -> decode Huffman -> y[3].
        var w = new VorbisTestWriter();
        w.Write(1, 1);           // nonzero
        w.Write(10, 8);          // y[0]
        w.Write(20, 8);          // y[1]
        WriteHuffmanCodeword(w, masterCb.Codewords[1], masterCb.EntryLengths[1]); // cval = 1
        // dim 0 uses -1 book: no Huffman read.
        // dim 1 uses book 1 (codebook 1). Let's decode entry 3.
        WriteHuffmanCodeword(w, valueCb.Codewords[3], valueCb.EntryLengths[3]);

        var reader = new VorbisBitReader(w.ToArray());
        var y = VorbisFloor1Decoder.Decode(ref reader, cfg, decoders);
        NotNull(y);
        // Y layout: y[0] and y[1] endpoints; y[2] and y[3] from the single partition's two dimensions.
        EqualInts(new[] { 10, 20, 0, 3 }, y!);
    }

    [TestMethod]
    public void VorbisFloor1Decoder_MultiplierEndpointBits_PicksRightWidth()
    {
        // Multiplier 4 -> endpointBits = 6. Values fit in 6 bits (0..63).
        var cfg = new VorbisFloor1Config
        {
            Partitions = 0,
            PartitionClassList = Array.Empty<int>(),
            ClassDimensions = Array.Empty<int>(),
            ClassSubclasses = Array.Empty<int>(),
            ClassMasterbooks = Array.Empty<int>(),
            ClassSubclassBooks = Array.Empty<int[]>(),
            Multiplier = 4,
            RangeBits = 4,
            XList = new[] { 0, 16 },
        };
        var w = new VorbisTestWriter();
        w.Write(1, 1);           // nonzero
        w.Write(63, 6);          // y[0] at 6 bits
        w.Write(0, 6);           // y[1] at 6 bits
        var reader = new VorbisBitReader(w.ToArray());
        var y = VorbisFloor1Decoder.Decode(ref reader, cfg, Array.Empty<VorbisHuffmanDecoder>());
        NotNull(y);
        EqualInts(new[] { 63, 0 }, y!);
    }

    [TestMethod]
    public void VorbisFloor1Decoder_BadMasterbookIndex_Throws()
    {
        var cfg = new VorbisFloor1Config
        {
            Partitions = 1,
            PartitionClassList = new[] { 0 },
            ClassDimensions = new[] { 1 },
            ClassSubclasses = new[] { 1 },
            ClassMasterbooks = new[] { 5 }, // out of range
            ClassSubclassBooks = new[] { new[] { 0, -1 } },
            Multiplier = 1,
            RangeBits = 4,
            XList = new[] { 0, 16, 0 },
        };
        var cb = VorbisHuffman.Build(new[] { 2, 2, 2, 2 });
        var decoders = new[] { new VorbisHuffmanDecoder(cb) };

        var w = new VorbisTestWriter();
        w.Write(1, 1); w.Write(0, 8); w.Write(0, 8);
        bool threw = false;
        try
        {
            var reader = new VorbisBitReader(w.ToArray());
            _ = VorbisFloor1Decoder.Decode(ref reader, cfg, decoders);
        }
        catch (InvalidDataException) { threw = true; }
        True(threw);
    }
}
