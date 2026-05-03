// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Decode a FLAC-in-Ogg stream per the Xiph FLAC-to-Ogg mapping. The first
// Ogg packet in a logical FLAC stream carries the FLAC-in-Ogg mapping header
// plus the native STREAMINFO metadata block; subsequent packets each contain
// a single FLAC frame.
//
// Native FLAC (.flac) is the common case and is handled by FlacDecoder;
// Ogg-wrapped FLAC (.oga or .ogg with FLAC content) is less common but
// legitimate and part of a complete FLAC implementation.

using SpawnDev.Codecs.Container.Ogg;

namespace SpawnDev.Codecs.Audio.Flac;

/// <summary>Decode a FLAC-in-Ogg byte stream to interleaved PCM samples.</summary>
public static class FlacOggDecoder
{
    /// <summary>
    /// Parse + decode a FLAC-in-Ogg byte stream. Returns the same
    /// <see cref="FlacStreamDecodeResult"/> shape as native-format
    /// <see cref="FlacDecoder.Decode(ReadOnlyMemory{byte})"/>.
    /// </summary>
    public static FlacStreamDecodeResult Decode(ReadOnlyMemory<byte> data)
    {
        var pages = OggPageReader.EnumeratePages(data.ToArray()).ToArray();
        if (pages.Length < 1)
            throw new InvalidDataException("FLAC-in-Ogg stream has no pages.");
        if (!pages[0].IsBeginningOfStream)
            throw new InvalidDataException("FLAC-in-Ogg: first page must be BOS.");
        uint serial = pages[0].BitstreamSerial;
        var ourPages = pages.Where(p => p.BitstreamSerial == serial).ToArray();
        var packets = OggPacketReader.AssemblePackets(ourPages).ToArray();
        if (packets.Length < 1)
            throw new InvalidDataException("FLAC-in-Ogg stream has no packets.");

        // Packet 0 layout per FLAC-to-Ogg mapping:
        //   1 byte  : 0x7F marker
        //   4 bytes : "FLAC" magic
        //   1 byte  : major version (1)
        //   1 byte  : minor version (0)
        //   2 bytes : header packets count (big endian)
        //   4 bytes : "fLaC" stream marker (native FLAC prelude)
        //   4 bytes : metadata block header with STREAMINFO type (last flag ignored here)
        //   34 bytes: STREAMINFO payload
        var first = packets[0].Data;
        if (first.Length < 1 + 4 + 1 + 1 + 2 + 4 + 4 + 34)
            throw new InvalidDataException("FLAC-in-Ogg mapping header too short.");
        if (first[0] != 0x7F)
            throw new InvalidDataException(
                $"FLAC-in-Ogg mapping byte 0 must be 0x7F, got 0x{first[0]:X2}.");
        if (first[1] != 'F' || first[2] != 'L' || first[3] != 'A' || first[4] != 'C')
            throw new InvalidDataException("FLAC-in-Ogg missing 'FLAC' magic.");
        int majorVersion = first[5];
        if (majorVersion != 1)
            throw new InvalidDataException($"FLAC-in-Ogg major version {majorVersion} not supported.");
        // byte 6 = minor version (ignored), bytes 7..8 = header packet count (ignored for decode).
        if (first[9] != (byte)'f' || first[10] != (byte)'L' || first[11] != (byte)'a' || first[12] != (byte)'C')
            throw new InvalidDataException("FLAC-in-Ogg missing native 'fLaC' marker.");
        // Bytes 13..16 are a native FLAC metadata block header for STREAMINFO (type = 0).
        int blockType = first[13] & 0x7F;
        if (blockType != FlacConstants.MetadataStreamInfo)
            throw new InvalidDataException($"FLAC-in-Ogg first block must be STREAMINFO, got {blockType}.");
        // Bytes 17..50 = 34-byte STREAMINFO payload.
        var streamInfo = FlacMetadataParser.ReadStreamInfo(first.AsSpan(17, 34));

        // Remaining packets = one FLAC frame each.
        var totalBuffer = new List<int>();
        int totalPerChannel = 0;
        for (int i = 1; i < packets.Length; i++)
        {
            byte[] framePkt = packets[i].Data;
            if (framePkt.Length == 0) continue;
            var frame = FlacFrameDecoder.Decode(framePkt, streamInfo);
            int block = frame.Header.BlockSize;
            int channels = frame.Header.Channels;
            for (int n = 0; n < block; n++)
            {
                for (int ch = 0; ch < channels; ch++)
                    totalBuffer.Add(frame.Samples[ch * block + n]);
            }
            totalPerChannel += block;
        }

        return new FlacStreamDecodeResult
        {
            StreamInfo = streamInfo,
            InterleavedSamples = totalBuffer.ToArray(),
            TotalSamplesPerChannel = totalPerChannel,
        };
    }
}
