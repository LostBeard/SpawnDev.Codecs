using SpawnDev.Codecs.Audio.Flac;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for <see cref="FlacMetadataParser"/>.<c>ReadAllBlocks</c>. Builds
/// synthetic FLAC preludes with various combinations of optional metadata
/// blocks (VORBIS_COMMENT, SEEKTABLE) after the mandatory STREAMINFO and
/// verifies the aggregate parse.
/// </summary>
public abstract partial class CodecsTestBase
{
    /// <summary>Build a VORBIS_COMMENT block body (no 0x03 + "vorbis" packet prefix, no framing byte).</summary>
    private static byte[] BuildFlacVorbisCommentBody(string vendor, string[] comments)
    {
        byte[] vb = System.Text.Encoding.UTF8.GetBytes(vendor);
        byte[][] cb = comments.Select(c => System.Text.Encoding.UTF8.GetBytes(c)).ToArray();
        int size = 4 + vb.Length + 4;
        foreach (var b in cb) size += 4 + b.Length;
        var bytes = new byte[size];
        int pos = 0;
        WriteUInt32Le(bytes, pos, (uint)vb.Length); pos += 4;
        Array.Copy(vb, 0, bytes, pos, vb.Length); pos += vb.Length;
        WriteUInt32Le(bytes, pos, (uint)cb.Length); pos += 4;
        foreach (var b in cb)
        {
            WriteUInt32Le(bytes, pos, (uint)b.Length); pos += 4;
            Array.Copy(b, 0, bytes, pos, b.Length); pos += b.Length;
        }
        return bytes;
    }

    private static byte[] BuildMetadataBlockHeader(bool isLast, int blockType, int length)
    {
        return new byte[]
        {
            (byte)((isLast ? 0x80 : 0x00) | (blockType & 0x7F)),
            (byte)(length >> 16),
            (byte)(length >> 8),
            (byte)length,
        };
    }

    [TestMethod]
    public void FlacReadAllBlocks_StreamInfoOnly_ReturnsNullOptionals()
    {
        // Use the existing encoder's output; it emits STREAMINFO as the only metadata block.
        var input = GenerateSineInt(64, 1, 44100, 16);
        byte[] flac = FlacEncoder.EncodeStream(input, 44100, 1, 16, blockSize: 64);
        var blocks = FlacMetadataParser.ReadAllBlocks(flac);
        Equal(44100, blocks.StreamInfo.SampleRateHz);
        True(blocks.VorbisComment is null, "No VORBIS_COMMENT expected.");
        True(blocks.SeekTable is null, "No SEEKTABLE expected.");
        // Audio start offset matches the stream prelude end.
        Equal(4 + 4 + 34, blocks.AudioStartOffset);
    }

    [TestMethod]
    public void FlacReadAllBlocks_StreamInfoPlusVorbisComment_Parses()
    {
        // Build a synthetic FLAC prelude: fLaC + STREAMINFO (not last) + VORBIS_COMMENT (last).
        var input = GenerateSineInt(64, 1, 44100, 16);
        byte[] nativeFlac = FlacEncoder.EncodeStream(input, 44100, 1, 16, blockSize: 64);
        // Extract the STREAMINFO payload (bytes [8..42)) from the encoded stream.
        var streamInfoPayload = new byte[34];
        Array.Copy(nativeFlac, 8, streamInfoPayload, 0, 34);

        byte[] commentBody = BuildFlacVorbisCommentBody(
            vendor: "SpawnDev.Codecs test",
            comments: new[] { "ARTIST=TJ", "TITLE=Final Frontier" });

        var bytes = new List<byte>();
        bytes.AddRange(new byte[] { (byte)'f', (byte)'L', (byte)'a', (byte)'C' });
        bytes.AddRange(BuildMetadataBlockHeader(isLast: false, blockType: FlacConstants.MetadataStreamInfo, length: 34));
        bytes.AddRange(streamInfoPayload);
        bytes.AddRange(BuildMetadataBlockHeader(isLast: true, blockType: FlacConstants.MetadataVorbisComment, length: commentBody.Length));
        bytes.AddRange(commentBody);

        var blocks = FlacMetadataParser.ReadAllBlocks(bytes.ToArray());
        NotNull(blocks.VorbisComment);
        Equal("SpawnDev.Codecs test", blocks.VorbisComment!.Vendor);
        Equal(2, blocks.VorbisComment.UserComments.Count);
        Equal("ARTIST=TJ", blocks.VorbisComment.UserComments[0]);
        Equal("TITLE=Final Frontier", blocks.VorbisComment.UserComments[1]);
    }

    [TestMethod]
    public void FlacReadAllBlocks_StreamInfoPlusSeekTable_Parses()
    {
        var input = GenerateSineInt(64, 1, 44100, 16);
        byte[] nativeFlac = FlacEncoder.EncodeStream(input, 44100, 1, 16, blockSize: 64);
        var streamInfoPayload = new byte[34];
        Array.Copy(nativeFlac, 8, streamInfoPayload, 0, 34);

        var seekPoints = new[]
        {
            new FlacSeekPoint(0, 0, 64),
            new FlacSeekPoint(64, 100, 64),
        };
        var seekBody = new byte[seekPoints.Length * 18];
        int sp = 0;
        foreach (var p in seekPoints)
        {
            for (int i = 0; i < 8; i++) seekBody[sp + i] = (byte)(p.SampleNumber >> (56 - 8 * i));
            for (int i = 0; i < 8; i++) seekBody[sp + 8 + i] = (byte)(p.StreamOffset >> (56 - 8 * i));
            seekBody[sp + 16] = (byte)(p.FrameSamples >> 8);
            seekBody[sp + 17] = (byte)p.FrameSamples;
            sp += 18;
        }

        var bytes = new List<byte>();
        bytes.AddRange(new byte[] { (byte)'f', (byte)'L', (byte)'a', (byte)'C' });
        bytes.AddRange(BuildMetadataBlockHeader(false, FlacConstants.MetadataStreamInfo, 34));
        bytes.AddRange(streamInfoPayload);
        bytes.AddRange(BuildMetadataBlockHeader(true, FlacConstants.MetadataSeekTable, seekBody.Length));
        bytes.AddRange(seekBody);

        var blocks = FlacMetadataParser.ReadAllBlocks(bytes.ToArray());
        NotNull(blocks.SeekTable);
        Equal(2, blocks.SeekTable!.Points.Length);
        Equal(0UL, blocks.SeekTable.Points[0].SampleNumber);
        Equal(64UL, blocks.SeekTable.Points[1].SampleNumber);
    }

    [TestMethod]
    public void FlacReadAllBlocks_StreamInfoPlusCommentAndSeek_Both()
    {
        var input = GenerateSineInt(64, 1, 44100, 16);
        byte[] nativeFlac = FlacEncoder.EncodeStream(input, 44100, 1, 16, blockSize: 64);
        var streamInfoPayload = new byte[34];
        Array.Copy(nativeFlac, 8, streamInfoPayload, 0, 34);

        byte[] commentBody = BuildFlacVorbisCommentBody("V", new[] { "A=1" });
        byte[] seekBody = new byte[18];

        var bytes = new List<byte>();
        bytes.AddRange(new byte[] { (byte)'f', (byte)'L', (byte)'a', (byte)'C' });
        bytes.AddRange(BuildMetadataBlockHeader(false, FlacConstants.MetadataStreamInfo, 34));
        bytes.AddRange(streamInfoPayload);
        bytes.AddRange(BuildMetadataBlockHeader(false, FlacConstants.MetadataVorbisComment, commentBody.Length));
        bytes.AddRange(commentBody);
        bytes.AddRange(BuildMetadataBlockHeader(true, FlacConstants.MetadataSeekTable, seekBody.Length));
        bytes.AddRange(seekBody);

        var blocks = FlacMetadataParser.ReadAllBlocks(bytes.ToArray());
        NotNull(blocks.VorbisComment);
        NotNull(blocks.SeekTable);
        Equal("V", blocks.VorbisComment!.Vendor);
        Equal(1, blocks.SeekTable!.Points.Length);
    }
}
