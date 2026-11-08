// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Ogg Opus file writer (RFC 7845): a port of FFmpeg 8.1's libavformat/oggenc.c for one Opus stream, the container
// PyAV / ffmpeg write (ComfyUI's Save Audio (Opus)). Given the same packets and timestamps, the file is the one
// `ffmpeg -fflags +bitexact` writes:
//   * OpusHead alone on the first (beginning-of-stream) page, OpusTags (vendor "ffmpeg" in bit-exact mode) on the next;
//   * audio packets packed into pages, a page closed at 255 segments, before a packet that would not fit (65025 bytes),
//     or once it spans the preferred page duration (1 s of granule time);
//   * each packet's granule = pts + duration + pre-skip (48 kHz); a page carries the granule of its last whole packet;
//   * pages are written one behind (the newest is held back) so the last page gets the end-of-stream flag;
//   * the Ogg CRC-32 of every page; serial number = the stream index (0) in bit-exact mode.
//
// A container writer on the host: the packets come already encoded and go to the caller's stream (like WebMWriter).

using System.Text;

namespace SpawnDev.Codecs.Container.Ogg;

/// <summary>Settings of an <see cref="OggOpusWriter"/>.</summary>
public sealed record OggOpusWriterOptions
{
    /// <summary>Channel count of the Opus stream (1 or 2: mapping family 0).</summary>
    public required int Channels { get; init; }

    /// <summary>Pre-skip in 48 kHz samples (the encoder's lookahead; libopus at 48 kHz: 312).</summary>
    public required int PreSkip { get; init; }

    /// <summary>OpusHead input sample rate (informational: the rate the audio had before encoding).</summary>
    public int InputSampleRate { get; init; } = 48000;

    /// <summary>OpusHead output gain (Q7.8 dB, default 0).</summary>
    public short OutputGain { get; init; }

    /// <summary>OpusTags vendor string (FFmpeg: "ffmpeg" in bit-exact mode, else "Lavf62.x.100").</summary>
    public string Vendor { get; init; } = "ffmpeg";

    /// <summary>OpusTags comment fields, in order, written as "KEY=value".</summary>
    public IReadOnlyList<KeyValuePair<string, string>>? Tags { get; init; }

    /// <summary>Ogg serial number (FFmpeg: the stream index, 0, in bit-exact mode; random otherwise).</summary>
    public uint Serial { get; init; }

    /// <summary>Preferred page duration in microseconds (FFmpeg's page_duration, default 1 s; 0 = fill pages).</summary>
    public long PageDurationUs { get; init; } = 1_000_000;
}

/// <summary>Ogg Opus writer, a port of FFmpeg's ogg muxer (see the file header). Write packets, then Finish.</summary>
public sealed class OggOpusWriter : IDisposable
{
    private const int MaxPageSize = 255 * 255;

    private sealed class Page
    {
        public long Granule = -1;
        public int Flags;
        public readonly byte[] Segments = new byte[255];
        public int SegmentCount;
        public readonly MemoryStream Data = new();
        public long StartGranule;
    }

    private readonly Stream _out;
    private readonly bool _leaveOpen;
    private readonly OggOpusWriterOptions _o;
    private readonly Queue<Page> _buffered = new();
    private Page _page = new();
    private int _pageCounter;
    private bool _finished;
    private const long NoPts = long.MinValue;

    private static readonly uint[] CrcTable = BuildCrcTable();

    /// <summary>Write the stream headers (OpusHead, OpusTags) to <paramref name="stream"/>.</summary>
    public OggOpusWriter(Stream stream, OggOpusWriterOptions options, bool leaveOpen = false)
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentNullException.ThrowIfNull(options);
        if (options.Channels < 1 || options.Channels > 2)
            throw new NotSupportedException("Ogg Opus mapping family 0 carries 1 or 2 channels.");
        if (options.PreSkip < 0 || options.PreSkip > 65535) throw new ArgumentOutOfRangeException(nameof(options), "PreSkip must be 0..65535.");
        _out = stream;
        _leaveOpen = leaveOpen;
        _o = options;

        // first packet: OpusHead (libopusenc's extradata), alone on the BOS page
        var head = new byte[19];
        Encoding.ASCII.GetBytes("OpusHead").CopyTo(head, 0);
        head[8] = 1;
        head[9] = (byte)options.Channels;
        head[10] = (byte)options.PreSkip; head[11] = (byte)(options.PreSkip >> 8);
        uint rate = (uint)options.InputSampleRate;
        head[12] = (byte)rate; head[13] = (byte)(rate >> 8); head[14] = (byte)(rate >> 16); head[15] = (byte)(rate >> 24);
        head[16] = (byte)options.OutputGain; head[17] = (byte)(options.OutputGain >> 8);
        head[18] = 0;   // mapping family 0
        BufferData(head, 0, header: true);
        _page.Flags |= 2;   // beginning of stream
        BufferPage();

        // second packet: OpusTags + Vorbis comment
        var tags = new MemoryStream();
        tags.Write(Encoding.ASCII.GetBytes("OpusTags"));
        void Le32(int v) { tags.WriteByte((byte)v); tags.WriteByte((byte)(v >> 8)); tags.WriteByte((byte)(v >> 16)); tags.WriteByte((byte)(v >> 24)); }
        var vendor = Encoding.UTF8.GetBytes(options.Vendor ?? "");
        Le32(vendor.Length);
        tags.Write(vendor);
        var fields = options.Tags ?? System.Array.Empty<KeyValuePair<string, string>>();
        Le32(fields.Count);
        foreach (var kv in fields)
        {
            var f = Encoding.UTF8.GetBytes(kv.Key + "=" + kv.Value);
            Le32(f.Length);
            tags.Write(f);
        }
        BufferData(tags.ToArray(), 0, header: true);
        BufferPage();
        _page.StartGranule = NoPts;
        WritePages(flush: 2);
    }

    /// <summary>
    /// Write one Opus packet whose first sample is at <paramref name="pts"/> (48 kHz, FFmpeg's convention: the first
    /// packet at -pre-skip) and that holds <paramref name="duration"/> samples of real audio.
    /// </summary>
    public void WritePacket(ReadOnlySpan<byte> packet, long pts, int duration)
    {
        if (_finished) throw new InvalidOperationException("The writer is finished.");
        long granule = pts + duration + _o.PreSkip;
        if (_page.StartGranule == NoPts) _page.StartGranule = pts;
        BufferData(packet.ToArray(), granule, header: false);
        WritePages(flush: 0);
    }

    /// <summary>Close the last page (end-of-stream flag) and flush.</summary>
    public void Finish()
    {
        if (_finished) return;
        if (_page.SegmentCount > 0) BufferPage();
        WritePages(flush: 1);
        _finished = true;
        _out.Flush();
    }

    /// <summary>Finish (if not yet) and release the stream unless leaveOpen.</summary>
    public void Dispose()
    {
        if (!_finished) Finish();
        if (!_leaveOpen) _out.Dispose();
    }

    private static long ToMicros(long granule) =>
        granule >= 0 ? (granule * 1_000_000 + 24_000) / 48_000 : -((-granule * 1_000_000 + 24_000) / 48_000);

    /// <summary>ogg_buffer_data for an Opus stream.</summary>
    private void BufferData(byte[] data, long granule, bool header)
    {
        int size = data.Length;
        int totalSegments = size / 255 + 1;
        int p = 0;
        // avoid a continued page
        if (!header && _page.Data.Length > 0 && MaxPageSize - _page.Data.Length < size) BufferPage();
        for (int i = 0; i < totalSegments;)
        {
            var page = _page;
            int segments = Math.Min(totalSegments - i, 255 - page.SegmentCount);
            if (i > 0 && page.SegmentCount == 0) page.Flags |= 1;   // continued packet
            for (int k = 0; k < segments - 1; k++) page.Segments[page.SegmentCount + k] = 255;
            page.SegmentCount += segments - 1;
            int len = Math.Min(size, segments * 255);
            page.Segments[page.SegmentCount++] = (byte)(len - (segments - 1) * 255);
            if (len > 0) page.Data.Write(data, p, len);
            p += len;
            size -= len;
            i += segments;
            if (i == totalSegments) page.Granule = granule;
            long start = page.StartGranule == NoPts ? 0 : ToMicros(page.StartGranule);
            long next = ToMicros(page.Granule);
            if (page.SegmentCount == 255) BufferPage();
            else if (!header && _o.PageDurationUs > 0 && page.StartGranule != NoPts && next - start >= _o.PageDurationUs) BufferPage();
        }
    }

    /// <summary>ogg_buffer_page: queue the current page; the next one starts at its granule.</summary>
    private void BufferPage()
    {
        var done = _page;
        _buffered.Enqueue(done);
        _page = new Page { StartGranule = done.Granule };
    }

    /// <summary>ogg_write_pages: flush 0 = all but the newest page, 1 = all (the last gets end-of-stream), 2 = all.</summary>
    private void WritePages(int flush)
    {
        while (_buffered.Count > 0)
        {
            if (_buffered.Count < 2 && flush == 0) break;
            var page = _buffered.Dequeue();
            WritePage(page, flush == 1 && _buffered.Count == 0 ? 4 : 0);
        }
    }

    private void WritePage(Page page, int extraFlags)
    {
        var h = new byte[27 + page.SegmentCount];
        h[0] = (byte)'O'; h[1] = (byte)'g'; h[2] = (byte)'g'; h[3] = (byte)'S';
        h[4] = 0;
        h[5] = (byte)(page.Flags | extraFlags);
        ulong g = unchecked((ulong)page.Granule);
        for (int i = 0; i < 8; i++) h[6 + i] = (byte)(g >> (8 * i));
        for (int i = 0; i < 4; i++) h[14 + i] = (byte)(_o.Serial >> (8 * i));
        uint seq = (uint)_pageCounter++;
        for (int i = 0; i < 4; i++) h[18 + i] = (byte)(seq >> (8 * i));
        h[26] = (byte)page.SegmentCount;
        System.Array.Copy(page.Segments, 0, h, 27, page.SegmentCount);
        var body = page.Data.ToArray();
        uint crc = Crc(0, h);
        crc = Crc(crc, body);
        for (int i = 0; i < 4; i++) h[22 + i] = (byte)(crc >> (8 * i));
        _out.Write(h);
        _out.Write(body);
    }

    /// <summary>The Ogg CRC-32 (polynomial 0x04C11DB7, initial 0, no reflection) as libogg computes it.</summary>
    private static uint Crc(uint crc, byte[] data)
    {
        foreach (byte b in data) crc = (crc << 8) ^ CrcTable[((crc >> 24) & 0xFF) ^ b];
        return crc;
    }

    private static uint[] BuildCrcTable()
    {
        var t = new uint[256];
        for (uint i = 0; i < 256; i++)
        {
            uint r = i << 24;
            for (int k = 0; k < 8; k++) r = (r & 0x80000000) != 0 ? (r << 1) ^ 0x04C11DB7 : r << 1;
            t[i] = r;
        }
        return t;
    }
}
