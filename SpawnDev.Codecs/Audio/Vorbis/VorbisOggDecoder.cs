// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Decode an Ogg-Vorbis byte stream into PCM samples. Mirrors the shape of
// OpusOggDecoder: parse the three Vorbis header packets from the BOS pages,
// then feed every subsequent audio packet through VorbisAudioDecoder.
//
// This is the structural glue. Bit-accuracy against real libvorbis-encoded
// Ogg-Vorbis files requires test vectors that are not yet bundled; the
// decoder runs end-to-end but downstream validation work is pending.

using SpawnDev.Codecs.Container.Ogg;

namespace SpawnDev.Codecs.Audio.Vorbis;

/// <summary>Whole-stream Ogg-Vorbis decode result.</summary>
public sealed record VorbisOggDecodeResult
{
    /// <summary>Identification header from header packet 1.</summary>
    public required VorbisIdentificationHeader Identification { get; init; }

    /// <summary>Comment header from header packet 2.</summary>
    public required VorbisCommentHeader Comments { get; init; }

    /// <summary>Setup header from header packet 3.</summary>
    public required VorbisSetupHeader Setup { get; init; }

    /// <summary>
    /// Interleaved float PCM samples produced by the audio-packet decoder,
    /// concatenated across every decoded audio packet.
    /// </summary>
    public required float[] InterleavedSamples { get; init; }

    /// <summary>Total sample frames per channel.</summary>
    public int TotalSamplesPerChannel { get; init; }
}

/// <summary>Decode an Ogg-Vorbis byte stream end to end.</summary>
public static class VorbisOggDecoder
{
    /// <summary>
    /// Parse and decode an Ogg-Vorbis stream. Reads the three header packets
    /// (identification, comment, setup) from the Ogg logical bitstream and
    /// then runs each audio packet through <see cref="VorbisAudioDecoder"/>.
    /// </summary>
    public static VorbisOggDecodeResult Decode(ReadOnlyMemory<byte> data)
    {
        var pages = OggPageReader.EnumeratePages(data.ToArray()).ToArray();
        if (pages.Length < 1)
            throw new InvalidDataException("Ogg-Vorbis stream has no pages.");
        if (!pages[0].IsBeginningOfStream)
            throw new InvalidDataException("Ogg-Vorbis first page must be BOS.");
        uint serial = pages[0].BitstreamSerial;
        var ourPages = pages.Where(p => p.BitstreamSerial == serial).ToArray();
        var packets = OggPacketReader.AssemblePackets(ourPages).ToArray();
        if (packets.Length < 3)
            throw new InvalidDataException(
                "Ogg-Vorbis stream needs at least 3 header packets (identification, comment, setup).");

        var ident = VorbisIdentificationHeaderParser.Parse(packets[0].Data);
        var comments = VorbisCommentHeaderParser.Parse(packets[1].Data);
        var setup = VorbisSetupHeaderParser.Parse(packets[2].Data, ident.AudioChannels);

        var audioDecoder = new VorbisAudioDecoder(ident, setup);
        int channels = ident.AudioChannels;
        int maxPerPacketFrames = ident.BlockSize1 / 2; // upper bound per packet
        float[] perPacket = new float[maxPerPacketFrames * channels];
        var all = new List<float>();
        int totalPerChannel = 0;

        for (int i = 3; i < packets.Length; i++)
        {
            int frames = audioDecoder.DecodePacket(packets[i].Data, perPacket);
            for (int n = 0; n < frames * channels; n++) all.Add(perPacket[n]);
            totalPerChannel += frames;
        }

        return new VorbisOggDecodeResult
        {
            Identification = ident,
            Comments = comments,
            Setup = setup,
            InterleavedSamples = all.ToArray(),
            TotalSamplesPerChannel = totalPerChannel,
        };
    }
}
