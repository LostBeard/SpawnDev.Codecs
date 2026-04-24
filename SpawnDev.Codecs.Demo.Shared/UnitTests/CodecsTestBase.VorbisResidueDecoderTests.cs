using SpawnDev.Codecs.Audio.Vorbis;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Sanity tests for <see cref="VorbisResidueDecoder"/>. Full-correctness
/// validation against real Vorbis audio packets will land alongside the
/// audio-packet integration slice; these tests exercise the no-op paths
/// (all do-not-decode, out-of-range) plus classification framing on a
/// minimal handcrafted codebook + config.
/// </summary>
public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void VorbisResidue_AllChannelsDoNotDecode_LeavesOutputUnchanged()
    {
        // Minimal config; actual bit reader never consumed because all
        // channels are skipped.
        var cfg = new VorbisResidueConfig
        {
            Type = VorbisResidueType.Type0,
            Begin = 0, End = 32, PartitionSize = 4,
            Classifications = 1, Classbook = 0,
            Cascade = new[] { 0 },
            Books = new[] { new[] { -1, -1, -1, -1, -1, -1, -1, -1 } },
        };
        var ch0 = new float[32];
        var ch1 = new float[32];
        // Pre-fill with sentinel values so we can detect any writes.
        Array.Fill(ch0, 1.5f);
        Array.Fill(ch1, -2.5f);

        // Empty reader - all do-not-decode means no reads.
        var reader = new VorbisBitReader(Array.Empty<byte>());
        var outBuf = new[] { ch0, ch1 };
        VorbisResidueDecoder.Decode(
            ref reader, cfg,
            decoders: Array.Empty<VorbisHuffmanDecoder>(),
            codebooks: Array.Empty<VorbisCodebook>(),
            residueOut: outBuf,
            doNotDecode: new[] { true, true },
            n: 32);
        // Output should be untouched.
        for (int i = 0; i < 32; i++)
        {
            Equal(1.5f, ch0[i]);
            Equal(-2.5f, ch1[i]);
        }
    }

    [TestMethod]
    public void VorbisResidue_NToReadIsZero_NoReads()
    {
        // actual range clamps to 0 when End <= Begin or n is below Begin.
        var cfg = new VorbisResidueConfig
        {
            Type = VorbisResidueType.Type1,
            Begin = 100, End = 100,
            PartitionSize = 4, Classifications = 1, Classbook = 0,
            Cascade = new[] { 0 },
            Books = new[] { new[] { -1, -1, -1, -1, -1, -1, -1, -1 } },
        };
        var reader = new VorbisBitReader(Array.Empty<byte>());
        var ch0 = new float[32];
        VorbisResidueDecoder.Decode(
            ref reader, cfg,
            decoders: Array.Empty<VorbisHuffmanDecoder>(),
            codebooks: Array.Empty<VorbisCodebook>(),
            residueOut: new[] { ch0 },
            doNotDecode: new[] { false },
            n: 32);
        // All output should remain zero.
        for (int i = 0; i < 32; i++) Equal(0.0f, ch0[i]);
    }

    [TestMethod]
    public void VorbisResidue_RangeNotAlignedToPartition_Throws()
    {
        var cfg = new VorbisResidueConfig
        {
            Type = VorbisResidueType.Type0,
            Begin = 0, End = 10, PartitionSize = 4,
            Classifications = 1, Classbook = 0,
            Cascade = new[] { 0 },
            Books = new[] { new[] { -1, -1, -1, -1, -1, -1, -1, -1 } },
        };
        var reader = new VorbisBitReader(Array.Empty<byte>());
        var ch0 = new float[16];
        bool threw = false;
        try
        {
            VorbisResidueDecoder.Decode(
                ref reader, cfg,
                decoders: Array.Empty<VorbisHuffmanDecoder>(),
                codebooks: Array.Empty<VorbisCodebook>(),
                residueOut: new[] { ch0 },
                doNotDecode: new[] { false },
                n: 16);
        }
        catch (InvalidDataException) { threw = true; }
        True(threw);
    }

    [TestMethod]
    public void VorbisResidue_CascadeAllZero_NoVectorBooks_OutputRemainsZero()
    {
        // Classbook decodes classifications but since every pass's book index
        // is -1, no residue vectors are added. Output stays at the initial
        // zero state.
        // Use an absolute-minimum classbook: 1-dim, 1-entry, length 1.
        // Actually with 1 entry the Huffman decoder short-circuits on "single
        // used entry". Let's use a 2-entry len-1 codebook with dims 1 so the
        // decoder reads bits and returns either 0 or 1.
        var classCb = new VorbisCodebook
        {
            Dimensions = 1, Entries = 2, Ordered = false, Sparse = false,
            Lengths = new[] { 1, 1 },
            LookupType = 0,
            Multiplicands = Array.Empty<int>(),
        };
        var classDec = new VorbisHuffmanDecoder(VorbisHuffman.Build(classCb.Lengths));

        var cfg = new VorbisResidueConfig
        {
            Type = VorbisResidueType.Type0,
            Begin = 0, End = 4, PartitionSize = 4,
            Classifications = 1, Classbook = 0,
            Cascade = new[] { 0 },   // no passes active
            Books = new[] { new[] { -1, -1, -1, -1, -1, -1, -1, -1 } },
        };
        // Bit stream supplies 1 classbook codeword (for pass 0 only; subsequent
        // passes have no book so don't read). Write code = 0 (bit '0').
        var w = new VorbisTestWriter();
        w.Write(0, 1);
        var reader = new VorbisBitReader(w.ToArray());

        var ch0 = new float[4];
        VorbisResidueDecoder.Decode(
            ref reader, cfg,
            decoders: new[] { classDec },
            codebooks: new[] { classCb },
            residueOut: new[] { ch0 },
            doNotDecode: new[] { false },
            n: 4);
        // All zero since no vector book added anything.
        for (int i = 0; i < 4; i++) Equal(0.0f, ch0[i]);
    }
}
