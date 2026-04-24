using SpawnDev.Codecs.Audio.Flac;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for the metadata-aware <see cref="FlacEncoder"/> overload. Encodes
/// with <see cref="FlacEncoderOptions"/> (vendor + user comments), decodes
/// via <c>ReadAllBlocks</c>, and verifies the embedded VORBIS_COMMENT block.
/// </summary>
public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void FlacEncoderOptions_VorbisComments_InjectedAndDecoded()
    {
        var input = GenerateSineInt(256, 1, 44100, 16);
        var opts = new FlacEncoderOptions
        {
            BlockSize = 256,
            Vendor = "SpawnDev.Codecs 0.1.0",
            VorbisComments = new[] { "ARTIST=TJ Tanner", "TITLE=Codecs Demo", "ALBUM=Phase 1a" },
        };
        byte[] flac = FlacEncoder.EncodeStream(input, 44100, 1, 16, opts);
        var blocks = FlacMetadataParser.ReadAllBlocks(flac);
        NotNull(blocks.VorbisComment);
        Equal("SpawnDev.Codecs 0.1.0", blocks.VorbisComment!.Vendor);
        Equal(3, blocks.VorbisComment.UserComments.Count);
        Equal("ARTIST=TJ Tanner", blocks.VorbisComment.UserComments[0]);
        Equal("TITLE=Codecs Demo", blocks.VorbisComment.UserComments[1]);
        Equal("ALBUM=Phase 1a", blocks.VorbisComment.UserComments[2]);

        // Audio still round-trips losslessly.
        var decoded = FlacDecoder.Decode(flac);
        EqualInts(input, decoded.InterleavedSamples);
    }

    [TestMethod]
    public void FlacEncoderOptions_EmptyCommentList_OmitsBlock()
    {
        var input = GenerateSineInt(64, 1, 44100, 16);
        var opts = new FlacEncoderOptions
        {
            BlockSize = 64,
            Vendor = "V",
            VorbisComments = Array.Empty<string>(),
        };
        byte[] flac = FlacEncoder.EncodeStream(input, 44100, 1, 16, opts);
        var blocks = FlacMetadataParser.ReadAllBlocks(flac);
        True(blocks.VorbisComment is null, "Empty comment list should omit the VORBIS_COMMENT block.");
    }

    [TestMethod]
    public void FlacEncoderOptions_NullCommentList_OmitsBlock()
    {
        var input = GenerateSineInt(64, 1, 44100, 16);
        var opts = new FlacEncoderOptions { BlockSize = 64 };
        byte[] flac = FlacEncoder.EncodeStream(input, 44100, 1, 16, opts);
        var blocks = FlacMetadataParser.ReadAllBlocks(flac);
        True(blocks.VorbisComment is null);
    }

    [TestMethod]
    public void FlacEncoderOptions_Utf8Content_RoundtripsExactly()
    {
        var input = GenerateSineInt(64, 1, 44100, 16);
        var opts = new FlacEncoderOptions
        {
            BlockSize = 64,
            Vendor = "VendΘr",
            VorbisComments = new[] { "TITLE=Μπουζούκι", "ARTIST=漢字" },
        };
        byte[] flac = FlacEncoder.EncodeStream(input, 44100, 1, 16, opts);
        var blocks = FlacMetadataParser.ReadAllBlocks(flac);
        NotNull(blocks.VorbisComment);
        Equal("VendΘr", blocks.VorbisComment!.Vendor);
        Equal("TITLE=Μπουζούκι", blocks.VorbisComment.UserComments[0]);
        Equal("ARTIST=漢字", blocks.VorbisComment.UserComments[1]);
    }

    [TestMethod]
    public void FlacEncoderOptions_NullOptions_Throws()
    {
        var input = GenerateSineInt(64, 1, 44100, 16);
        bool threw = false;
        try { _ = FlacEncoder.EncodeStream(input, 44100, 1, 16, (FlacEncoderOptions)null!); }
        catch (ArgumentNullException) { threw = true; }
        True(threw);
    }
}
