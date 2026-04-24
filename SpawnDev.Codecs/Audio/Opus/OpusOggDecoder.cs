// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Glue that reads an Ogg-encapsulated Opus stream (the .opus file format per
// RFC 7845) and decodes it into PCM using the existing OpusDecoder.
//
// Layout of a minimal Opus-in-Ogg stream:
//   Page 0 (BOS): single packet = OpusHead ("OpusHead" magic + stream geometry).
//   Page 1:       single packet = OpusTags ("OpusTags" magic + vendor + user comments).
//   Pages 2..N:   one or more Opus audio packets per page, each a full Opus TOC packet
//                 decodable with OpusDecoder.DecodePacketAsync.
//
// Opus itself always outputs 48 kHz internally; callers can downsample afterward.

using SpawnDev.Codecs.Container.Ogg;

namespace SpawnDev.Codecs.Audio.Opus;

/// <summary>
/// Result of decoding a whole Opus-in-Ogg stream: the OpusHead + OpusTags
/// metadata plus interleaved 48 kHz PCM samples concatenated across every
/// audio packet.
/// </summary>
public sealed record OpusOggDecodeResult
{
    /// <summary>Identification header parsed from the BOS page.</summary>
    public required OpusHead Head { get; init; }

    /// <summary>Comment header parsed from page 1.</summary>
    public required OpusTags Tags { get; init; }

    /// <summary>Sample-interleaved 48 kHz float PCM output, channel-interleaved.</summary>
    public required float[] InterleavedSamples48kHz { get; init; }

    /// <summary>Total samples per channel after pre-skip trimming.</summary>
    public int TotalSamplesPerChannel { get; init; }
}

/// <summary>
/// Decode an Opus-in-Ogg byte stream into PCM samples at 48 kHz.
/// Single-stream only (the first BOS page's bitstream serial is used).
/// </summary>
public static class OpusOggDecoder
{
    /// <summary>
    /// Parse + decode an Opus-in-Ogg byte stream. Applies <c>OpusHead.PreSkip</c>
    /// to trim the encoder's lookahead samples from the start of the output.
    /// </summary>
    public static async Task<OpusOggDecodeResult> DecodeAsync(ReadOnlyMemory<byte> data, CancellationToken ct = default)
    {
        var pagesArr = OggPageReader.EnumeratePages(data.ToArray()).ToArray();
        if (pagesArr.Length < 2)
            throw new InvalidDataException("Opus-in-Ogg stream needs at least 2 pages (OpusHead + OpusTags).");
        var firstPage = pagesArr[0];
        if (!firstPage.IsBeginningOfStream)
            throw new InvalidDataException("First Ogg page must be BOS.");
        uint serial = firstPage.BitstreamSerial;

        // Keep only the pages for THIS logical bitstream (Opus may be multiplexed with others).
        var ourPages = pagesArr.Where(p => p.BitstreamSerial == serial).ToArray();
        var packets = OggPacketReader.AssemblePackets(ourPages).ToArray();
        if (packets.Length < 2)
            throw new InvalidDataException("Opus-in-Ogg stream needs at least 2 packets for Head + Tags.");

        var head = OpusHeadParser.Parse(packets[0].Data);
        var tags = OpusTagsParser.Parse(packets[1].Data);

        int channels = head.OutputChannels;
        if (head.ChannelMappingFamily > 1 || channels > 2)
            throw new NotSupportedException(
                "Opus-in-Ogg multi-stream (mapping family >= 1, channels > 2) is not yet supported.");

        var dec = OpusCodec.CreateDecoder(new OpusDecoderConfig
        {
            SampleRateHz = 48000,
            ChannelCount = channels,
        });

        // Max decoded samples per Opus packet at 48 kHz is 5760 (120 ms).
        float[] perPacketBuffer = new float[5760 * channels];
        var all = new List<float>();
        try
        {
            for (int i = 2; i < packets.Length; i++)
            {
                int samples = await dec.DecodePacketAsync(packets[i].Data, perPacketBuffer, ct);
                for (int n = 0; n < samples * channels; n++) all.Add(perPacketBuffer[n]);
            }
        }
        finally
        {
            await dec.DisposeAsync();
        }

        // Apply pre-skip: drop the first `PreSkip` samples per channel from the head.
        int skipInterleaved = Math.Min(head.PreSkip * channels, all.Count);
        float[] trimmed = new float[all.Count - skipInterleaved];
        for (int i = 0; i < trimmed.Length; i++) trimmed[i] = all[skipInterleaved + i];

        return new OpusOggDecodeResult
        {
            Head = head,
            Tags = tags,
            InterleavedSamples48kHz = trimmed,
            TotalSamplesPerChannel = trimmed.Length / Math.Max(1, channels),
        };
    }
}
