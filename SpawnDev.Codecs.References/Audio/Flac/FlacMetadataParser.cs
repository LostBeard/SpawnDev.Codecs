// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Parser for the FLAC stream prelude: "fLaC" marker + metadata block chain.
// Matches libFLAC stream_decoder.c::read_metadata_ for the blocks we actively
// consume in the decoder. Non-STREAMINFO blocks are currently surfaced only
// via their header so callers can skip them.

namespace SpawnDev.Codecs.Audio.Flac;

/// <summary>
/// Parses the FLAC stream prelude. The prelude consists of the 4-byte "fLaC"
/// stream marker followed by a chain of metadata blocks, the first of which
/// must be STREAMINFO.
/// </summary>
public static class FlacMetadataParser
{
    /// <summary>
    /// Verify the 4-byte "fLaC" stream marker at the start of a FLAC file.
    /// Throws <see cref="InvalidDataException"/> if the marker is missing.
    /// </summary>
    public static void ReadStreamMarker(ReadOnlySpan<byte> data, out int bytesRead)
    {
        if (data.Length < 4)
            throw new InvalidDataException("FLAC stream marker requires at least 4 bytes.");
        if (data[0] != (byte)'f' || data[1] != (byte)'L' || data[2] != (byte)'a' || data[3] != (byte)'C')
            throw new InvalidDataException(
                $"FLAC stream marker mismatch: expected 'fLaC' (66 4C 61 43), got " +
                $"{data[0]:X2} {data[1]:X2} {data[2]:X2} {data[3]:X2}.");
        bytesRead = 4;
    }

    /// <summary>
    /// Read a 4-byte metadata block header.
    /// </summary>
    public static FlacMetadataBlockHeader ReadBlockHeader(ReadOnlySpan<byte> data, out int bytesRead)
    {
        if (data.Length < 4)
            throw new InvalidDataException("FLAC metadata block header requires at least 4 bytes.");
        var r = new FlacBitReader(data[..4]);
        bool isLast = r.ReadBit() != 0;
        int blockType = (int)r.ReadBits(7);
        int length = (int)r.ReadBits(24);
        bytesRead = 4;
        return new FlacMetadataBlockHeader(isLast, blockType, length);
    }

    /// <summary>
    /// Read the 34-byte STREAMINFO payload (after the 4-byte header).
    /// </summary>
    public static FlacStreamInfo ReadStreamInfo(ReadOnlySpan<byte> payload)
    {
        // STREAMINFO is exactly 34 bytes per RFC 9639 Section 8.1.
        const int StreamInfoBytes = 34;
        if (payload.Length < StreamInfoBytes)
            throw new InvalidDataException(
                $"STREAMINFO payload too short: {payload.Length} bytes, need {StreamInfoBytes}.");

        var r = new FlacBitReader(payload[..StreamInfoBytes]);
        int minBlock = (int)r.ReadBits(16);
        int maxBlock = (int)r.ReadBits(16);
        int minFrame = (int)r.ReadBits(24);
        int maxFrame = (int)r.ReadBits(24);
        int sampleRate = (int)r.ReadBits(20);
        int channels = (int)r.ReadBits(3) + 1;
        int bitsPerSample = (int)r.ReadBits(5) + 1;
        // TotalSamples is a 36-bit field; read as two pieces because ReadBits supports up to 32.
        ulong totalHi = r.ReadBits(4);
        ulong totalLo = r.ReadBits(32);
        ulong totalSamples = (totalHi << 32) | totalLo;

        // Remaining 128 bits are the MD5 signature; read as 16 bytes (now byte-aligned).
        byte[] md5 = new byte[16];
        for (int i = 0; i < 16; i++)
        {
            md5[i] = (byte)r.ReadBits(8);
        }

        if (sampleRate is < 1 or > 655350)
            throw new InvalidDataException($"STREAMINFO sample rate out of range: {sampleRate} Hz.");
        if (channels is < 1 or > FlacConstants.MaxChannels)
            throw new InvalidDataException($"STREAMINFO channel count out of range: {channels}.");
        if (bitsPerSample is < 4 or > 32)
            throw new InvalidDataException($"STREAMINFO bits-per-sample out of range: {bitsPerSample}.");
        if (minBlock > maxBlock && maxBlock != 0)
            throw new InvalidDataException(
                $"STREAMINFO min-block ({minBlock}) > max-block ({maxBlock}).");

        return new FlacStreamInfo
        {
            MinBlockSize = minBlock,
            MaxBlockSize = maxBlock,
            MinFrameSize = minFrame,
            MaxFrameSize = maxFrame,
            SampleRateHz = sampleRate,
            Channels = channels,
            BitsPerSample = bitsPerSample,
            TotalSamples = totalSamples,
            Md5Signature = md5,
        };
    }

    /// <summary>
    /// Aggregated metadata surface returned by <see cref="ReadAllBlocks"/>. All
    /// block fields are optional except STREAMINFO which the format requires.
    /// </summary>
    public sealed record FlacMetadataBlocks
    {
        /// <summary>STREAMINFO block (always present in a valid stream).</summary>
        public required FlacStreamInfo StreamInfo { get; init; }

        /// <summary>Vorbis-format comment block if present.</summary>
        public SpawnDev.Codecs.Audio.Vorbis.VorbisCommentHeader? VorbisComment { get; init; }

        /// <summary>Seek table block if present.</summary>
        public FlacSeekTable? SeekTable { get; init; }

        /// <summary>Byte offset of the first audio frame.</summary>
        public int AudioStartOffset { get; init; }
    }

    /// <summary>
    /// Walk the full metadata chain and parse STREAMINFO, VORBIS_COMMENT, and
    /// SEEKTABLE if present. Other block types are skipped.
    /// </summary>
    public static FlacMetadataBlocks ReadAllBlocks(ReadOnlySpan<byte> data)
    {
        int pos = 0;
        ReadStreamMarker(data[pos..], out int markerSize);
        pos += markerSize;

        var firstHeader = ReadBlockHeader(data[pos..], out int hdrSize);
        pos += hdrSize;
        if (firstHeader.BlockType != FlacConstants.MetadataStreamInfo)
            throw new InvalidDataException(
                $"FLAC first metadata block must be STREAMINFO, got type {firstHeader.BlockType}.");
        if (data.Length < pos + firstHeader.LengthBytes)
            throw new InvalidDataException("FLAC STREAMINFO block truncated.");
        var streamInfo = ReadStreamInfo(data.Slice(pos, firstHeader.LengthBytes));
        pos += firstHeader.LengthBytes;

        SpawnDev.Codecs.Audio.Vorbis.VorbisCommentHeader? comment = null;
        FlacSeekTable? seekTable = null;

        bool isLast = firstHeader.IsLast;
        while (!isLast)
        {
            if (data.Length < pos + 4)
                throw new InvalidDataException("FLAC metadata chain truncated mid-header.");
            var hdr = ReadBlockHeader(data[pos..], out int hs);
            pos += hs;
            if (data.Length < pos + hdr.LengthBytes)
                throw new InvalidDataException($"FLAC metadata block (type={hdr.BlockType}) truncated.");
            var payload = data.Slice(pos, hdr.LengthBytes);
            switch (hdr.BlockType)
            {
                case FlacConstants.MetadataVorbisComment:
                    comment = SpawnDev.Codecs.Audio.Vorbis.VorbisCommentHeaderParser.Parse(BuildVorbisCommentSyntheticPacket(payload));
                    break;
                case FlacConstants.MetadataSeekTable:
                    seekTable = FlacSeekTableParser.Parse(payload);
                    break;
                default:
                    // STREAMINFO (handled above), PADDING, APPLICATION, CUESHEET, PICTURE: skipped.
                    break;
            }
            pos += hdr.LengthBytes;
            isLast = hdr.IsLast;
        }

        return new FlacMetadataBlocks
        {
            StreamInfo = streamInfo,
            VorbisComment = comment,
            SeekTable = seekTable,
            AudioStartOffset = pos,
        };
    }

    /// <summary>
    /// FLAC's VORBIS_COMMENT block reuses the Vorbis comment body (vendor + user
    /// comments) but without the Vorbis packet header. Our
    /// <see cref="SpawnDev.Codecs.Audio.Vorbis.VorbisCommentHeaderParser"/> expects
    /// the full packet shape (0x03 + "vorbis"), so we prepend those 7 bytes
    /// and let the parser skip the framing-flag byte (it's optional).
    /// </summary>
    private static byte[] BuildVorbisCommentSyntheticPacket(ReadOnlySpan<byte> commentBody)
    {
        byte[] prefix = { 0x03, (byte)'v', (byte)'o', (byte)'r', (byte)'b', (byte)'i', (byte)'s' };
        var buf = new byte[prefix.Length + commentBody.Length];
        Array.Copy(prefix, buf, prefix.Length);
        commentBody.CopyTo(buf.AsSpan(prefix.Length));
        return buf;
    }

    /// <summary>
    /// Walk the full metadata chain starting at offset 0 (including the "fLaC" marker),
    /// parse STREAMINFO (which must be the first block), and return the byte offset of
    /// the first audio frame. Non-STREAMINFO blocks are skipped.
    /// </summary>
    public static (FlacStreamInfo StreamInfo, int AudioStartOffset) ReadStreamPrelude(ReadOnlySpan<byte> data)
    {
        int pos = 0;
        ReadStreamMarker(data[pos..], out int markerSize);
        pos += markerSize;

        var firstHeader = ReadBlockHeader(data[pos..], out int hdrSize);
        pos += hdrSize;
        if (firstHeader.BlockType != FlacConstants.MetadataStreamInfo)
            throw new InvalidDataException(
                $"FLAC first metadata block must be STREAMINFO, got type {firstHeader.BlockType}.");
        if (data.Length < pos + firstHeader.LengthBytes)
            throw new InvalidDataException("FLAC STREAMINFO block truncated.");
        var streamInfo = ReadStreamInfo(data.Slice(pos, firstHeader.LengthBytes));
        pos += firstHeader.LengthBytes;

        bool isLast = firstHeader.IsLast;
        while (!isLast)
        {
            if (data.Length < pos + 4)
                throw new InvalidDataException("FLAC metadata chain truncated mid-header.");
            var hdr = ReadBlockHeader(data[pos..], out int hs);
            pos += hs;
            if (data.Length < pos + hdr.LengthBytes)
                throw new InvalidDataException($"FLAC metadata block (type={hdr.BlockType}) truncated.");
            pos += hdr.LengthBytes;
            isLast = hdr.IsLast;
        }
        return (streamInfo, pos);
    }
}
