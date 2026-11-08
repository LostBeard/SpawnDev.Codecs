// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Ogg Opus writer: frames already-encoded Opus packets into an Ogg logical bitstream (RFC 3533 pages and lacing, RFC 7845
// OpusHead / OpusTags headers and granule positions, RFC 6716 packets). It runs on the host because it only frames bytes
// that are already encoded into the caller's Stream (the same role as IvfWriter); it never reads or changes a sample.
//
// Page layout (chosen to match the files FFmpeg 8.1's ogg muxer writes, worked out by running the ffmpeg binary and
// reading its output pages; see Plans/PLAN-CleanRoom-Rewrite.md, Component A):
//   * OpusHead alone on the first page (beginning-of-stream flag), OpusTags on the page(s) after it, both granule 0. A
//     header packet larger than a page continues on the next page; a page on which no packet ends has granule -1.
//   * Audio packets are packed into pages. Before a packet is added, the open page is written out when
//       - the page's body bytes plus the packet's bytes exceed 65025 (255 segments of 255 bytes), or
//       - PageDurationUs > 0 and the packet starts at least PageDurationUs after the page's start, measured in
//         microseconds rounded to nearest (halves up). A page starts at the pts of the first packet with data on it (a
//         continued packet included); the first audio page starts PreSkip samples before the first packet's pts.
//   * A packet that does not fit in the open page's 255 lacing values continues on the next page (continued-packet flag).
//   * A page's granule position is pts + duration + PreSkip of the last packet that ends on it (RFC 7845 section 4).
//   * Finish() writes the open page with the end-of-stream flag; a stream with no audio packets ends after OpusTags.
//   * CRC-32 per RFC 3533 (polynomial 0x04C11DB7, no reflection, initial value 0).

using System.Buffers.Binary;
using System.Text;

namespace SpawnDev.Codecs.Container.Ogg;

/// <summary>Settings of an <see cref="OggOpusWriter"/> (the OpusHead and OpusTags fields and the page layout).</summary>
public sealed record OggOpusWriterOptions
{
    /// <summary>Output channels, 1 or 2 (channel mapping family 0).</summary>
    public required int Channels { get; init; }

    /// <summary>Samples at 48 kHz the decoder drops from the start (the encoder's lookahead, RFC 7845 section 4.2).</summary>
    public required int PreSkip { get; init; }

    /// <summary>The original input sample rate, informational (OpusHead).</summary>
    public int InputSampleRate { get; init; } = 48000;

    /// <summary>Output gain in Q7.8 dB (OpusHead).</summary>
    public short OutputGain { get; init; }

    /// <summary>The OpusTags vendor string.</summary>
    public string Vendor { get; init; } = "ffmpeg";

    /// <summary>User comments, written in order as "KEY=value"; null for none.</summary>
    public IReadOnlyList<KeyValuePair<string, string>>? Tags { get; init; }

    /// <summary>The logical bitstream serial number.</summary>
    public uint Serial { get; init; }

    /// <summary>Preferred page duration in microseconds (0 = no duration limit, pages close only when full).</summary>
    public long PageDurationUs { get; init; } = 1_000_000;
}

/// <summary>
/// Writes an Ogg Opus file (RFC 7845) to a <see cref="Stream"/> from already-encoded Opus packets. The header pages are
/// written by the constructor; call <see cref="WritePacket"/> per audio packet, then <see cref="Finish"/> (or dispose).
/// </summary>
public sealed class OggOpusWriter : IDisposable
{
    private const int MaxSegments = 255;
    private const int MaxBody = 255 * 255;

    private readonly Stream _stream;
    private readonly bool _leaveOpen;
    private readonly OggOpusWriterOptions _options;

    // The open page.
    private readonly byte[] _page = new byte[27 + MaxSegments + MaxBody];
    private readonly byte[] _segments = new byte[MaxSegments];
    private int _segmentCount, _bodyLength;
    private bool _pageOpen, _pageContinued;
    private long _pageGranule = -1, _pageStart;

    private uint _sequence;
    private bool _bos = true, _audioStarted, _finished, _disposed;

    /// <summary>Start an Ogg Opus stream on <paramref name="stream"/> and write its OpusHead and OpusTags pages.</summary>
    public OggOpusWriter(Stream stream, OggOpusWriterOptions options, bool leaveOpen = false)
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentNullException.ThrowIfNull(options);
        if (!stream.CanWrite) throw new ArgumentException("The stream is not writable.", nameof(stream));
        if (options.Channels is not (1 or 2)) throw new ArgumentOutOfRangeException(nameof(options), "Channels must be 1 or 2.");
        if (options.PreSkip is < 0 or > ushort.MaxValue) throw new ArgumentOutOfRangeException(nameof(options), "PreSkip must be 0..65535.");
        if (options.InputSampleRate < 0) throw new ArgumentOutOfRangeException(nameof(options), "InputSampleRate must not be negative.");
        if (options.PageDurationUs < 0) throw new ArgumentOutOfRangeException(nameof(options), "PageDurationUs must not be negative.");
        ArgumentNullException.ThrowIfNull(options.Vendor);
        _stream = stream;
        _leaveOpen = leaveOpen;
        _options = options;

        WriteHeaderPacket(BuildOpusHead(options));
        WriteHeaderPacket(BuildOpusTags(options));
    }

    /// <summary>The OpusHead packet (RFC 7845 section 5.1), channel mapping family 0.</summary>
    internal static byte[] BuildOpusHead(OggOpusWriterOptions o)
    {
        var head = new byte[19];
        "OpusHead"u8.CopyTo(head);
        head[8] = 1;                                                            // version
        head[9] = (byte)o.Channels;
        BinaryPrimitives.WriteUInt16LittleEndian(head.AsSpan(10), (ushort)o.PreSkip);
        BinaryPrimitives.WriteUInt32LittleEndian(head.AsSpan(12), (uint)o.InputSampleRate);
        BinaryPrimitives.WriteInt16LittleEndian(head.AsSpan(16), o.OutputGain);
        head[18] = 0;                                                           // mapping family 0
        return head;
    }

    /// <summary>The OpusTags packet (RFC 7845 section 5.2): vendor, then each comment as "KEY=value".</summary>
    internal static byte[] BuildOpusTags(OggOpusWriterOptions o)
    {
        var vendor = Encoding.UTF8.GetBytes(o.Vendor);
        var tags = o.Tags ?? Array.Empty<KeyValuePair<string, string>>();
        var fields = new byte[tags.Count][];
        long length = 8 + 4 + vendor.Length + 4;
        for (int i = 0; i < tags.Count; i++)
        {
            fields[i] = Encoding.UTF8.GetBytes(tags[i].Key + "=" + tags[i].Value);
            length += 4 + fields[i].Length;
        }
        if (length > int.MaxValue) throw new ArgumentException("OpusTags is too large.");
        var t = new byte[length];
        "OpusTags"u8.CopyTo(t);
        int p = 8;
        BinaryPrimitives.WriteUInt32LittleEndian(t.AsSpan(p), (uint)vendor.Length); p += 4;
        vendor.CopyTo(t, p); p += vendor.Length;
        BinaryPrimitives.WriteUInt32LittleEndian(t.AsSpan(p), (uint)fields.Length); p += 4;
        foreach (var f in fields)
        {
            BinaryPrimitives.WriteUInt32LittleEndian(t.AsSpan(p), (uint)f.Length); p += 4;
            f.CopyTo(t, p); p += f.Length;
        }
        return t;
    }

    // A header packet: on its own page(s), granule 0, written out at once.
    private void WriteHeaderPacket(ReadOnlySpan<byte> packet)
    {
        OpenPage(continued: false, start: 0);
        AppendPacket(packet, granule: 0, start: 0);
        WritePage(eos: false);
    }

    /// <summary>
    /// Add one audio packet. <paramref name="pts"/> is its first sample in 48 kHz samples (the first packet at -PreSkip);
    /// <paramref name="duration"/> is how many 48 kHz samples it covers (the last packet: up to the stream's end).
    /// </summary>
    public void WritePacket(ReadOnlySpan<byte> packet, long pts, int duration)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_finished) throw new InvalidOperationException("The stream is finished.");
        if (duration < 0) throw new ArgumentOutOfRangeException(nameof(duration));
        if (!_audioStarted)
        {
            _audioStarted = true;
            OpenPage(continued: false, start: pts - _options.PreSkip);
        }
        else if (_pageOpen && _segmentCount > 0 && (_bodyLength + (long)packet.Length > MaxBody || PageDurationReached(pts)))
            WritePage(eos: false);
        if (!_pageOpen) OpenPage(continued: false, start: pts);
        AppendPacket(packet, pts + duration + _options.PreSkip, pts);
    }

    private bool PageDurationReached(long pts)
    {
        long limit = _options.PageDurationUs;
        if (limit <= 0) return false;
        // (pts - start) 48 kHz samples in microseconds, rounded to nearest with halves away from zero.
        Int128 num = (Int128)(pts - _pageStart) * 1_000_000;
        Int128 us = num >= 0 ? (num + 24000) / 48000 : -((-num + 24000) / 48000);
        return us >= limit;
    }

    private void OpenPage(bool continued, long start)
    {
        _pageOpen = true;
        _pageContinued = continued;
        _pageStart = start;
        _pageGranule = -1;
        _segmentCount = 0;
        _bodyLength = 0;
    }

    // Lace one packet into the open page, continuing on new pages while it does not fit.
    private void AppendPacket(ReadOnlySpan<byte> packet, long granule, long start)
    {
        int at = 0;
        while (true)
        {
            if (_segmentCount == MaxSegments)
            {
                // The page is full: write it out. The next page continues this packet unless none of it is laced yet.
                WritePage(eos: false);
                OpenPage(continued: at > 0, start);
            }
            int lace = Math.Min(255, packet.Length - at);
            packet.Slice(at, lace).CopyTo(_page.AsSpan(27 + MaxSegments + _bodyLength));
            _segments[_segmentCount++] = (byte)lace;
            _bodyLength += lace;
            at += lace;
            if (lace < 255)
            {
                _pageGranule = granule;                                         // the packet ends on this page
                return;
            }
        }
    }

    // Write the open page: header, lacing values, body, CRC.
    private void WritePage(bool eos)
    {
        int headerLength = 27 + _segmentCount;
        var page = _page.AsSpan();
        // The body was laced in after a full-size segment table; move it up behind the actual one.
        if (_segmentCount < MaxSegments)
            page.Slice(27 + MaxSegments, _bodyLength).CopyTo(page.Slice(headerLength));
        "OggS"u8.CopyTo(page);
        page[4] = 0;                                                            // stream structure version
        page[5] = (byte)((_pageContinued ? 1 : 0) | (_bos ? 2 : 0) | (eos ? 4 : 0));
        BinaryPrimitives.WriteInt64LittleEndian(page.Slice(6), _pageGranule);
        BinaryPrimitives.WriteUInt32LittleEndian(page.Slice(14), _options.Serial);
        BinaryPrimitives.WriteUInt32LittleEndian(page.Slice(18), _sequence);
        BinaryPrimitives.WriteUInt32LittleEndian(page.Slice(22), 0);
        page[26] = (byte)_segmentCount;
        _segments.AsSpan(0, _segmentCount).CopyTo(page.Slice(27));
        var bytes = page.Slice(0, headerLength + _bodyLength);
        BinaryPrimitives.WriteUInt32LittleEndian(page.Slice(22), OggCrc32.Compute(bytes));
        _stream.Write(bytes);
        _sequence++;
        _bos = false;
        _pageOpen = false;
    }

    /// <summary>Write the last page (end-of-stream flag) and flush the stream. Further packets are refused.</summary>
    public void Finish()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_finished) return;
        _finished = true;
        if (_pageOpen) WritePage(eos: true);
        _stream.Flush();
    }

    /// <summary>Finish the stream (if not finished) and close it unless leaveOpen was set.</summary>
    public void Dispose()
    {
        if (_disposed) return;
        try { Finish(); }
        finally
        {
            _disposed = true;
            if (!_leaveOpen) _stream.Dispose();
        }
    }
}
