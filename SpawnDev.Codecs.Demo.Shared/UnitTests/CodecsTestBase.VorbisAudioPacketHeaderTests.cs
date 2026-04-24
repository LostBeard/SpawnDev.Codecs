using SpawnDev.Codecs.Audio.Vorbis;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for <see cref="VorbisAudioPacketHeaderParser"/>. Each test builds a
/// tiny setup header / ident header plus a packet-header bit pattern and
/// asserts the parse matches expectations.
/// </summary>
public abstract partial class CodecsTestBase
{
    private static VorbisIdentificationHeader MakeIdent(int blockSize0 = 256, int blockSize1 = 2048)
    {
        return new VorbisIdentificationHeader
        {
            VorbisVersion = 0,
            AudioChannels = 1,
            SampleRateHz = 48000,
            BitrateMaximum = 0,
            BitrateNominal = 128000,
            BitrateMinimum = 0,
            BlockSize0 = blockSize0,
            BlockSize1 = blockSize1,
        };
    }

    private static VorbisSetupHeader MakeSetup(params VorbisModeConfig[] modes)
    {
        return new VorbisSetupHeader
        {
            Codebooks = Array.Empty<VorbisCodebook>(),
            Floors = Array.Empty<VorbisFloor1Config>(),
            Residues = Array.Empty<VorbisResidueConfig>(),
            Mappings = Array.Empty<VorbisMappingConfig>(),
            Modes = modes,
        };
    }

    [TestMethod]
    public void VorbisAudio_SingleMode_ShortBlock_NoModeBits()
    {
        // Only 1 mode -> ilog(0) = 0 bits for mode selection. Short block -> no window flags.
        var modes = new[] { new VorbisModeConfig { BlockFlag = false, Mapping = 0 } };
        var setup = MakeSetup(modes);
        var ident = MakeIdent();
        var w = new VorbisTestWriter();
        w.Write(0, 1);           // packet type = audio
        // No mode bits (1 mode means 0 bits).
        // No window flags (short block).
        var packet = w.ToArray();
        var hdr = VorbisAudioPacketHeaderParser.Parse(packet, setup, ident);
        Equal(0, hdr.ModeNumber);
        Equal(256, hdr.BlockSize);
        False(hdr.IsLongBlock);
        False(hdr.PreviousWindowLong);
        False(hdr.NextWindowLong);
    }

    [TestMethod]
    public void VorbisAudio_TwoModes_SelectsShortBlock()
    {
        // Two modes: index 0 = short, index 1 = long. ilog(1) = 1 bit.
        var modes = new[]
        {
            new VorbisModeConfig { BlockFlag = false, Mapping = 0 },
            new VorbisModeConfig { BlockFlag = true, Mapping = 0 },
        };
        var setup = MakeSetup(modes);
        var ident = MakeIdent();
        var w = new VorbisTestWriter();
        w.Write(0, 1);           // packet type
        w.Write(0, 1);           // mode 0
        var hdr = VorbisAudioPacketHeaderParser.Parse(w.ToArray(), setup, ident);
        Equal(0, hdr.ModeNumber);
        Equal(256, hdr.BlockSize);
        False(hdr.IsLongBlock);
    }

    [TestMethod]
    public void VorbisAudio_TwoModes_SelectsLongBlock_ReadsWindowFlags()
    {
        var modes = new[]
        {
            new VorbisModeConfig { BlockFlag = false, Mapping = 0 },
            new VorbisModeConfig { BlockFlag = true, Mapping = 0 },
        };
        var setup = MakeSetup(modes);
        var ident = MakeIdent();
        var w = new VorbisTestWriter();
        w.Write(0, 1);           // packet type
        w.Write(1, 1);           // mode 1 (long)
        w.Write(1, 1);           // prev window long = true
        w.Write(0, 1);           // next window long = false
        var hdr = VorbisAudioPacketHeaderParser.Parse(w.ToArray(), setup, ident);
        Equal(1, hdr.ModeNumber);
        Equal(2048, hdr.BlockSize);
        True(hdr.IsLongBlock);
        True(hdr.PreviousWindowLong);
        False(hdr.NextWindowLong);
    }

    [TestMethod]
    public void VorbisAudio_FourModes_UsesTwoModeBits()
    {
        // 4 modes -> ilog(3) = 2 bits of mode selection.
        var modes = new[]
        {
            new VorbisModeConfig { BlockFlag = false, Mapping = 0 },
            new VorbisModeConfig { BlockFlag = true, Mapping = 0 },
            new VorbisModeConfig { BlockFlag = false, Mapping = 1 },
            new VorbisModeConfig { BlockFlag = true, Mapping = 1 },
        };
        var setup = MakeSetup(modes);
        var ident = MakeIdent();
        var w = new VorbisTestWriter();
        w.Write(0, 1);           // packet type
        w.Write(3, 2);           // mode 3 (long, mapping 1)
        w.Write(0, 1); w.Write(1, 1); // prev false, next true
        var hdr = VorbisAudioPacketHeaderParser.Parse(w.ToArray(), setup, ident);
        Equal(3, hdr.ModeNumber);
        Equal(2048, hdr.BlockSize);
        True(hdr.IsLongBlock);
        False(hdr.PreviousWindowLong);
        True(hdr.NextWindowLong);
    }

    [TestMethod]
    public void VorbisAudio_PacketTypeOne_Throws()
    {
        // Packet type = 1 indicates a header packet, not an audio packet.
        var modes = new[] { new VorbisModeConfig { BlockFlag = false, Mapping = 0 } };
        var setup = MakeSetup(modes);
        var ident = MakeIdent();
        var w = new VorbisTestWriter();
        w.Write(1, 1);           // header, not audio
        bool threw = false;
        try { _ = VorbisAudioPacketHeaderParser.Parse(w.ToArray(), setup, ident); }
        catch (InvalidDataException) { threw = true; }
        True(threw);
    }

    [TestMethod]
    public void VorbisAudio_CustomBlockSizes_ReflectedInResult()
    {
        var modes = new[]
        {
            new VorbisModeConfig { BlockFlag = false, Mapping = 0 },
            new VorbisModeConfig { BlockFlag = true, Mapping = 0 },
        };
        var setup = MakeSetup(modes);
        var ident = MakeIdent(blockSize0: 64, blockSize1: 8192);
        var w = new VorbisTestWriter();
        w.Write(0, 1);
        w.Write(1, 1);           // long block
        w.Write(0, 1); w.Write(0, 1);
        var hdr = VorbisAudioPacketHeaderParser.Parse(w.ToArray(), setup, ident);
        Equal(8192, hdr.BlockSize);
    }
}
