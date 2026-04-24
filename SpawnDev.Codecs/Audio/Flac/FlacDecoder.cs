// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Public FLAC decoder. Wraps the metadata parser + frame decoder into a
// stream-oriented decode loop. Given a complete FLAC byte sequence (the
// "fLaC" marker, metadata chain, and audio frames), produces a sequence of
// decoded frames or a flat interleaved PCM buffer.

namespace SpawnDev.Codecs.Audio.Flac;

/// <summary>
/// Streaming FLAC decoder. Open with the full FLAC byte buffer; the STREAMINFO
/// metadata is parsed immediately so <see cref="StreamInfo"/> is available
/// before any frames are read.
/// </summary>
public sealed class FlacDecoder
{
    private readonly ReadOnlyMemory<byte> _data;
    private int _pos;

    /// <summary>Parsed STREAMINFO metadata.</summary>
    public FlacStreamInfo StreamInfo { get; }

    /// <summary>True once the decoder has consumed all input bytes.</summary>
    public bool IsAtEnd => _pos >= _data.Length;

    private FlacDecoder(ReadOnlyMemory<byte> data, FlacStreamInfo info, int audioOffset)
    {
        _data = data;
        StreamInfo = info;
        _pos = audioOffset;
    }

    /// <summary>
    /// Open a FLAC byte stream. Parses the "fLaC" marker and metadata chain
    /// immediately; the decoder is positioned at the first audio frame.
    /// </summary>
    public static FlacDecoder Open(ReadOnlyMemory<byte> data)
    {
        var (info, audioOffset) = FlacMetadataParser.ReadStreamPrelude(data.Span);
        return new FlacDecoder(data, info, audioOffset);
    }

    /// <summary>
    /// Decode the next audio frame. Returns <c>null</c> if no more bytes remain.
    /// </summary>
    public FlacFrame? ReadNextFrame()
    {
        if (IsAtEnd) return null;
        var frame = FlacFrameDecoder.Decode(_data.Span.Slice(_pos), StreamInfo);
        _pos += frame.FrameBytesConsumed;
        return frame;
    }

    /// <summary>
    /// Decode every remaining frame and concatenate the samples into a single
    /// interleaved PCM buffer (channel-interleaved per sample, not per frame).
    /// </summary>
    public FlacStreamDecodeResult DecodeAll()
    {
        var frames = new List<FlacFrame>();
        while (ReadNextFrame() is { } f) frames.Add(f);

        int channels = StreamInfo.Channels;
        int totalPerChannel = 0;
        for (int i = 0; i < frames.Count; i++) totalPerChannel += frames[i].Header.BlockSize;

        int[] interleaved = new int[totalPerChannel * channels];
        int writeIndex = 0;
        foreach (var f in frames)
        {
            int frameBlock = f.Header.BlockSize;
            for (int n = 0; n < frameBlock; n++)
            {
                for (int ch = 0; ch < channels; ch++)
                    interleaved[writeIndex++] = f.Samples[ch * frameBlock + n];
            }
        }

        return new FlacStreamDecodeResult
        {
            StreamInfo = StreamInfo,
            InterleavedSamples = interleaved,
            TotalSamplesPerChannel = totalPerChannel,
        };
    }

    /// <summary>Shorthand for <c>Open(data).DecodeAll()</c>.</summary>
    public static FlacStreamDecodeResult Decode(ReadOnlyMemory<byte> data) => Open(data).DecodeAll();
}
