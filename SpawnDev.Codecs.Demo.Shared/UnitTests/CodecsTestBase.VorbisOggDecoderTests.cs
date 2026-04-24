using SpawnDev.Codecs.Audio.Vorbis;
using SpawnDev.Codecs.Container.Ogg;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Structural tests for <see cref="VorbisOggDecoder"/>. Hand-builds a minimal
/// Ogg-Vorbis stream from the existing ident + comment + setup helpers,
/// feeds it through the decoder, and verifies the three headers parse back
/// correctly. Bit-accuracy against real libvorbis audio packets is pending
/// test-vector bundling.
/// </summary>
public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void VorbisOggDecoder_HeadersOnlyStream_ParsesThreeHeaders()
    {
        byte[] identPacket = BuildIdentHeader(
            version: 0, channels: 1, sampleRate: 48000,
            bitrateMax: 0, bitrateNom: 128000, bitrateMin: 0,
            bs0Log: 8, bs1Log: 11);
        byte[] commentPacket = BuildCommentHeader(
            "SpawnDev.Codecs test",
            new[] { "ARTIST=TJ", "TITLE=Ogg-Vorbis Round-Trip" },
            framingFlag: true);
        byte[] setupPacket = BuildMinimalSetupPacket(audioChannels: 1);

        var outgoing = new List<OggOutgoingPacket>
        {
            new OggOutgoingPacket { Data = identPacket, GranulePosition = 0 },
            new OggOutgoingPacket { Data = commentPacket, GranulePosition = 0 },
            new OggOutgoingPacket { Data = setupPacket, GranulePosition = 0 },
        };
        byte[] ogg = OggPageWriter.WriteStream(bitstreamSerial: 0x42, outgoing);

        var decoded = VorbisOggDecoder.Decode(ogg);
        Equal(48000, decoded.Identification.SampleRateHz);
        Equal(1, decoded.Identification.AudioChannels);
        Equal("SpawnDev.Codecs test", decoded.Comments.Vendor);
        Equal(2, decoded.Comments.UserComments.Count);
        Equal("ARTIST=TJ", decoded.Comments.UserComments[0]);
        Equal(1, decoded.Setup.Codebooks.Length);
        Equal(1, decoded.Setup.Floors.Length);
        Equal(1, decoded.Setup.Residues.Length);
        Equal(1, decoded.Setup.Mappings.Length);
        Equal(1, decoded.Setup.Modes.Length);
        // Headers-only stream has no audio packets.
        Equal(0, decoded.TotalSamplesPerChannel);
        Equal(0, decoded.InterleavedSamples.Length);
    }

    [TestMethod]
    public void VorbisOggDecoder_MissingHeaderPacket_Throws()
    {
        byte[] identPacket = BuildIdentHeader(0, 1, 48000, 0, 0, 0, 8, 11);
        byte[] commentPacket = BuildCommentHeader("V", Array.Empty<string>(), framingFlag: true);
        var outgoing = new List<OggOutgoingPacket>
        {
            new OggOutgoingPacket { Data = identPacket, GranulePosition = 0 },
            new OggOutgoingPacket { Data = commentPacket, GranulePosition = 0 },
        };
        byte[] ogg = OggPageWriter.WriteStream(1, outgoing);
        bool threw = false;
        try { _ = VorbisOggDecoder.Decode(ogg); } catch (InvalidDataException) { threw = true; }
        True(threw, "Stream with < 3 header packets must throw.");
    }

    [TestMethod]
    public void VorbisOggDecoder_EmptyStream_Throws()
    {
        bool threw = false;
        try { _ = VorbisOggDecoder.Decode(Array.Empty<byte>()); } catch (InvalidDataException) { threw = true; }
        True(threw);
    }

    [TestMethod]
    public void VorbisOggDecoder_FirstPageNotBos_Throws()
    {
        // Manually build a page with BOS flag cleared.
        byte[] identPacket = BuildIdentHeader(0, 1, 48000, 0, 0, 0, 8, 11);
        byte[] page = OggPageWriter.WriteSinglePacketPage(
            headerType: 0, // no BOS flag
            granulePosition: 0, bitstreamSerial: 1, pageSequence: 0,
            packet: identPacket);
        bool threw = false;
        try { _ = VorbisOggDecoder.Decode(page); } catch (InvalidDataException) { threw = true; }
        True(threw);
    }
}
