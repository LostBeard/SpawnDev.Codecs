// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// MP4 (ISO BMFF) muxer for AV1 video and Opus audio: frames packets an encoder already produced into a caller's Stream.
// Written from the public specifications (ISO/IEC 14496-12, the AV1 ISOBMFF binding, Opus in ISOBMFF) and the layout the
// ffmpeg command line tool writes with -fflags +bitexact, observed as a black box on the fixtures in
// SpawnDev.Codecs.Demo.Shared/TestData/Mp4Mux (scripts/video/make_mp4_mux_fixtures.py). Host-side container work only.
//
// Layout: ftyp (isom, minor 0x200; isom, av01 when a track is AV1, iso2, mp41), free (8), mdat, then moov: mvhd (ms),
// (the av01 sample entry carries av1C, pasp and btrt)
// one trak per track (tkhd, edts/elst, mdia: mdhd, hdlr, minf: vmhd / smhd, dinf/dref, stbl: stsd, stts, stss when not
// every sample is a sync sample, stsc, stsz, stco (co64 past 4 GiB), and for Opus sgpd + sbgp 'roll'), udta/meta.
// Packets go into mdat in decode-time order across tracks (ties: the lower track); each run of one track's packets is a
// chunk. btrt: buffer 0, max = avg = floor(bits * timescale / media duration). Opus: timescale 48000, the pre-skip as
// negative decode times and the edit list's media time; roll distance -(80 ms / packet duration), the first packets
// that cannot pre-roll in sample group 0.

using System.Buffers.Binary;
using System.Text;

namespace SpawnDev.Codecs.Container.Mp4;

/// <summary>An AV1 video track for <see cref="Mp4Writer"/>.</summary>
public sealed record Mp4Av1Track
{
    /// <summary>The AV1CodecConfigurationRecord (av1C body: marker / version byte onwards, config OBUs included).</summary>
    public required byte[] Av1C { get; init; }
    /// <summary>Picture width.</summary>
    public required int Width { get; init; }
    /// <summary>Picture height.</summary>
    public required int Height { get; init; }
    /// <summary>Media timescale (ticks per second); <see cref="Mp4Writer.VideoTimescale"/> gives ffmpeg's choice.</summary>
    public required int Timescale { get; init; }
    /// <summary>Pixel aspect ratio (pasp: horizontal spacing, vertical spacing).</summary>
    public (int H, int V) PixelAspect { get; init; } = (1, 1);
}

/// <summary>An Opus audio track for <see cref="Mp4Writer"/> (timescale 48000).</summary>
public sealed record Mp4OpusTrack
{
    /// <summary>The OpusHead (RFC 7845 identification header, 19+ bytes): channels, pre-skip, input rate, gain, mapping.</summary>
    public required byte[] OpusHead { get; init; }
}

/// <summary>Writes AV1 + Opus into an MP4 file (see the file header for the layout).</summary>
public sealed class Mp4Writer : IDisposable
{
    private sealed class Track
    {
        public Mp4Av1Track? Video;
        public Mp4OpusTrack? Audio;
        public int Timescale;
        public readonly Queue<(byte[] Data, long Dts, long Duration, bool Key)> Pending = new();
        public readonly List<int> Sizes = new();
        public readonly List<long> Durations = new();
        public readonly List<int> Sync = new();          // 1-based sample numbers of sync samples
        public readonly List<long> ChunkOffsets = new();
        public readonly List<int> ChunkSamples = new();
        public long FirstDts = long.MinValue;
        public long Bytes;
        public bool AllSync = true;
    }

    private readonly Stream _out;
    private readonly bool _leaveOpen;
    private readonly List<Track> _tracks = new();
    private readonly MemoryStream? _mdatBuffer;   // a stream that cannot seek: mdat is buffered
    private bool _started, _finished;
    private long _mdatStart, _mdatLength;
    private int _lastTrack = -1;

    /// <summary>A writer onto <paramref name="output"/> (any stream; a non-seekable one buffers the media data).</summary>
    public Mp4Writer(Stream output, bool leaveOpen = false)
    {
        _out = output ?? throw new ArgumentNullException(nameof(output));
        _leaveOpen = leaveOpen;
        if (!output.CanSeek) _mdatBuffer = new MemoryStream();
    }

    /// <summary>
    /// The video timescale ffmpeg picks for a frame rate num / den: den-based ticks (num when the rate is whole)... in
    /// ffmpeg's terms the stream time base 1 / rateNum scaled by the smallest power of two that reaches 10000
    /// (10 fps: 10240, 24: 12288, 16: 16384, 30000/1001: 30000).
    /// </summary>
    public static int VideoTimescale(int rateNum, int rateDen)
    {
        if (rateNum <= 0 || rateDen <= 0) throw new ArgumentOutOfRangeException(nameof(rateNum));
        long ts = rateNum;
        while (ts < 10000) ts <<= 1;
        return (int)ts;
    }

    /// <summary>Add an AV1 track; returns its track number (1-based, file order).</summary>
    public int AddVideoTrack(Mp4Av1Track track)
    {
        ArgumentNullException.ThrowIfNull(track);
        if (_started) throw new InvalidOperationException("Add every track before the first frame.");
        if (track.Width <= 0 || track.Height <= 0 || track.Width > 65535 || track.Height > 65535) throw new ArgumentOutOfRangeException(nameof(track));
        if (track.Timescale <= 0) throw new ArgumentOutOfRangeException(nameof(track), "Timescale must be positive.");
        _tracks.Add(new Track { Video = track, Timescale = track.Timescale });
        return _tracks.Count;
    }

    /// <summary>Add an Opus track; returns its track number (1-based, file order).</summary>
    public int AddAudioTrack(Mp4OpusTrack track)
    {
        ArgumentNullException.ThrowIfNull(track);
        if (_started) throw new InvalidOperationException("Add every track before the first frame.");
        var h = track.OpusHead;
        if (h == null || h.Length < 19 || Encoding.ASCII.GetString(h, 0, 8) != "OpusHead") throw new ArgumentException("Not an OpusHead.", nameof(track));
        if (h[18] != 0 && h.Length < 21 + h[9]) throw new ArgumentException("OpusHead channel mapping table missing.", nameof(track));
        _tracks.Add(new Track { Audio = track, Timescale = 48000 });
        return _tracks.Count;
    }

    /// <summary>
    /// Queue one packet of <paramref name="track"/>: decode time and duration in the track's timescale (Opus: 48 kHz
    /// samples, the first packet at -pre-skip), <paramref name="key"/> when it is a sync sample. Packets of one track
    /// must come in decode order; tracks may be interleaved any way (the writer orders them by decode time).
    /// </summary>
    public void WriteFrame(int track, ReadOnlySpan<byte> data, long dts, long duration, bool key)
    {
        if (_finished) throw new InvalidOperationException("Finish was called.");
        if (track < 1 || track > _tracks.Count) throw new ArgumentOutOfRangeException(nameof(track));
        if (duration < 0) throw new ArgumentOutOfRangeException(nameof(duration));
        if (!_started) Start();
        _tracks[track - 1].Pending.Enqueue((data.ToArray(), dts, duration, key));
        while (_tracks.All(t => t.Pending.Count > 0)) Emit(Next());
    }

    private int Next()
    {
        int best = -1;
        for (int i = 0; i < _tracks.Count; i++)
        {
            if (_tracks[i].Pending.Count == 0) continue;
            if (best < 0 || Earlier(i, best)) best = i;
        }
        return best;
    }

    /// <summary>Track a's next packet decodes strictly earlier than track b's (exact, across timescales).</summary>
    private bool Earlier(int a, int b)
    {
        var pa = _tracks[a].Pending.Peek(); var pb = _tracks[b].Pending.Peek();
        return (Int128)pa.Dts * _tracks[b].Timescale < (Int128)pb.Dts * _tracks[a].Timescale;
    }

    private void Start()
    {
        if (_tracks.Count == 0) throw new InvalidOperationException("Add a track first.");
        _started = true;
        var head = new MemoryStream();
        // ftyp
        var brands = new List<string> { "isom" };
        if (_tracks.Any(t => t.Video != null)) brands.Add("av01");
        brands.Add("iso2"); brands.Add("mp41");
        Box(head, "ftyp", b => { Str(b, "isom"); U32(b, 0x200); foreach (var s in brands) Str(b, s); });
        Box(head, "free", _ => { });
        _out.Write(head.GetBuffer(), 0, (int)head.Length);
        _mdatStart = _mdatBuffer == null ? _out.Position : head.Length;
        var mdatHead = new byte[8];
        Encoding.ASCII.GetBytes("mdat").CopyTo(mdatHead, 4);
        if (_mdatBuffer == null) _out.Write(mdatHead);
        _mdatLength = 8;
    }

    private void Emit(int index)
    {
        var t = _tracks[index];
        var p = t.Pending.Dequeue();
        long offset = _mdatStart + _mdatLength;
        if (index != _lastTrack)
        {
            t.ChunkOffsets.Add(offset);
            t.ChunkSamples.Add(0);
            _lastTrack = index;
        }
        t.ChunkSamples[^1]++;
        if (t.FirstDts == long.MinValue) t.FirstDts = p.Dts;
        t.Sizes.Add(p.Data.Length);
        t.Durations.Add(p.Duration);
        t.Bytes += p.Data.Length;
        if (p.Key) t.Sync.Add(t.Sizes.Count); else t.AllSync = false;
        if (_mdatBuffer != null) _mdatBuffer.Write(p.Data); else _out.Write(p.Data);
        _mdatLength += p.Data.Length;
    }

    /// <summary>Write every queued packet, the mdat size and the moov.</summary>
    public void Finish()
    {
        if (_finished) return;
        if (!_started) Start();
        while (_tracks.Any(t => t.Pending.Count > 0)) Emit(Next());
        _finished = true;
        bool wide = _mdatLength > uint.MaxValue;
        if (_mdatBuffer == null)
        {
            long end = _out.Position;
            if (wide)
            {
                // the 8-byte free box before mdat becomes room for a 64-bit size
                _out.Position = _mdatStart - 8;
                var h = new byte[16];
                BinaryPrimitives.WriteUInt32BigEndian(h, 1);
                Encoding.ASCII.GetBytes("mdat").CopyTo(h, 4);
                BinaryPrimitives.WriteUInt64BigEndian(h.AsSpan(8), (ulong)(_mdatLength + 8));
                _out.Write(h);
            }
            else
            {
                _out.Position = _mdatStart;
                var h = new byte[4];
                BinaryPrimitives.WriteUInt32BigEndian(h, (uint)_mdatLength);
                _out.Write(h);
            }
            _out.Position = end;
        }
        else
        {
            var h = new byte[wide ? 16 : 8];
            if (wide)
            {
                // replaces the free box already written: not possible on a pipe, so a wide mdat follows the free box
                BinaryPrimitives.WriteUInt32BigEndian(h, 1);
                Encoding.ASCII.GetBytes("mdat").CopyTo(h, 4);
                BinaryPrimitives.WriteUInt64BigEndian(h.AsSpan(8), (ulong)(_mdatLength + 8));
            }
            else
            {
                BinaryPrimitives.WriteUInt32BigEndian(h, (uint)_mdatLength);
                Encoding.ASCII.GetBytes("mdat").CopyTo(h, 4);
            }
            _out.Write(h);
            _mdatBuffer.Position = 0;
            _mdatBuffer.CopyTo(_out);
        }
        var moov = new MemoryStream();
        WriteMoov(moov);
        _out.Write(moov.GetBuffer(), 0, (int)moov.Length);
        _out.Flush();
    }

    private static long Rescale(long value, long from, long to) =>
        (long)(((Int128)value * to + (from / 2)) / from);   // round half up (values are non-negative)

    private void WriteMoov(Stream s)
    {
        var durationsMs = new long[_tracks.Count];
        for (int i = 0; i < _tracks.Count; i++)
        {
            var t = _tracks[i];
            if (t.FirstDts > 0) throw new NotSupportedException("A track starting after time 0 (an empty edit) is not supported.");
            long media = t.Durations.Sum();
            durationsMs[i] = Rescale(media + Math.Min(t.FirstDts, 0), t.Timescale, 1000);
        }
        long movieMs = durationsMs.Length == 0 ? 0 : durationsMs.Max();
        Box(s, "moov", m =>
        {
            Full(m, "mvhd", 0, 0, b =>
            {
                U32(b, 0); U32(b, 0); U32(b, 1000); U32(b, (uint)movieMs);
                U32(b, 0x00010000); U16(b, 0x0100); Zero(b, 10); Matrix(b); Zero(b, 24);
                U32(b, (uint)_tracks.Count + 1);
            });
            for (int i = 0; i < _tracks.Count; i++) WriteTrak(m, i, durationsMs[i]);
            Box(m, "udta", u => Full(u, "meta", 0, 0, b =>
            {
                Full(b, "hdlr", 0, 0, h => { U32(h, 0); Str(h, "mdir"); Str(h, "appl"); U32(h, 0); U32(h, 0); h.WriteByte(0); });
                Box(b, "ilst", _ => { });
            }));
        });
    }

    private void WriteTrak(Stream m, int index, long durationMs)
    {
        var t = _tracks[index];
        bool video = t.Video != null;
        long media = t.Durations.Sum();
        long mediaTime = Math.Max(0, -t.FirstDts);
        Box(m, "trak", k =>
        {
            Full(k, "tkhd", 0, 3, b =>
            {
                U32(b, 0); U32(b, 0); U32(b, (uint)(index + 1)); U32(b, 0); U32(b, (uint)durationMs);
                Zero(b, 8); U16(b, 0); U16(b, video ? (ushort)0 : (ushort)1); U16(b, video ? (ushort)0 : (ushort)0x0100); U16(b, 0);
                Matrix(b);
                U32(b, video ? (uint)t.Video!.Width << 16 : 0); U32(b, video ? (uint)t.Video!.Height << 16 : 0);
            });
            Box(k, "edts", e => Full(e, "elst", 0, 0, b =>
            {
                U32(b, 1); U32(b, (uint)durationMs); U32(b, (uint)mediaTime); U32(b, 0x00010000);
            }));
            Box(k, "mdia", md =>
            {
                Full(md, "mdhd", 0, 0, b => { U32(b, 0); U32(b, 0); U32(b, (uint)t.Timescale); U32(b, (uint)media); U16(b, 0x55C4); U16(b, 0); });
                Full(md, "hdlr", 0, 0, b =>
                {
                    U32(b, 0); Str(b, video ? "vide" : "soun"); Zero(b, 12);
                    var name = Encoding.ASCII.GetBytes(video ? "VideoHandler" : "SoundHandler");
                    b.Write(name); b.WriteByte(0);
                });
                Box(md, "minf", mi =>
                {
                    if (video) Full(mi, "vmhd", 0, 1, b => Zero(b, 8));
                    else Full(mi, "smhd", 0, 0, b => Zero(b, 4));
                    Box(mi, "dinf", d => Full(d, "dref", 0, 0, b => { U32(b, 1); Full(b, "url ", 0, 1, _ => { }); }));
                    Box(mi, "stbl", st => WriteStbl(st, t, media));
                });
            });
        });
    }

    private void WriteStbl(Stream st, Track t, long media)
    {
        long bitrate = media == 0 ? 0 : (long)((Int128)t.Bytes * 8 * t.Timescale / media);   // truncated
        Full(st, "stsd", 0, 0, b =>
        {
            U32(b, 1);
            if (t.Video is { } v)
            {
                Box(b, "av01", e =>
                {
                    Zero(e, 6); U16(e, 1); Zero(e, 16);
                    U16(e, (ushort)v.Width); U16(e, (ushort)v.Height);
                    U32(e, 0x00480000); U32(e, 0x00480000); U32(e, 0); U16(e, 1);
                    Zero(e, 32); U16(e, 0x0018); U16(e, 0xFFFF);
                    Box(e, "av1C", c => c.Write(v.Av1C));
                    Box(e, "pasp", c => { U32(c, (uint)v.PixelAspect.H); U32(c, (uint)v.PixelAspect.V); });
                    Btrt(e, bitrate);
                });
            }
            else
            {
                var h = t.Audio!.OpusHead;
                int channels = h[9];
                Box(b, "Opus", e =>
                {
                    Zero(e, 6); U16(e, 1); Zero(e, 8);
                    U16(e, (ushort)channels); U16(e, 16); U16(e, 0); U16(e, 0); U32(e, 48000u << 16);
                    Box(e, "dOps", c =>
                    {
                        c.WriteByte(0); c.WriteByte((byte)channels);
                        U16(c, BinaryPrimitives.ReadUInt16LittleEndian(h.AsSpan(10)));
                        U32(c, BinaryPrimitives.ReadUInt32LittleEndian(h.AsSpan(12)));
                        U16(c, BinaryPrimitives.ReadUInt16LittleEndian(h.AsSpan(16)));
                        c.WriteByte(h[18]);
                        if (h[18] != 0) c.Write(h, 19, 2 + channels);
                    });
                    Btrt(e, bitrate);
                });
            }
        });
        // stts
        var runs = new List<(int Count, long Delta)>();
        foreach (long d in t.Durations)
            if (runs.Count > 0 && runs[^1].Delta == d) runs[^1] = (runs[^1].Count + 1, d); else runs.Add((1, d));
        Full(st, "stts", 0, 0, b => { U32(b, (uint)runs.Count); foreach (var r in runs) { U32(b, (uint)r.Count); U32(b, (uint)r.Delta); } });
        if (t.Video != null && !t.AllSync)
            Full(st, "stss", 0, 0, b => { U32(b, (uint)t.Sync.Count); foreach (int n in t.Sync) U32(b, (uint)n); });
        // stsc: a new entry when the samples per chunk change
        var stsc = new List<(int First, int Samples)>();
        for (int c = 0; c < t.ChunkSamples.Count; c++)
            if (stsc.Count == 0 || stsc[^1].Samples != t.ChunkSamples[c]) stsc.Add((c + 1, t.ChunkSamples[c]));
        Full(st, "stsc", 0, 0, b => { U32(b, (uint)stsc.Count); foreach (var e in stsc) { U32(b, (uint)e.First); U32(b, (uint)e.Samples); U32(b, 1); } });
        Full(st, "stsz", 0, 0, b => { U32(b, 0); U32(b, (uint)t.Sizes.Count); foreach (int z in t.Sizes) U32(b, (uint)z); });
        if (t.ChunkOffsets.Any(o => o > uint.MaxValue))
            Full(st, "co64", 0, 0, b => { U32(b, (uint)t.ChunkOffsets.Count); foreach (long o in t.ChunkOffsets) U64(b, (ulong)o); });
        else
            Full(st, "stco", 0, 0, b => { U32(b, (uint)t.ChunkOffsets.Count); foreach (long o in t.ChunkOffsets) U32(b, (uint)o); });
        if (t.Audio != null && t.Durations.Count > 0)
        {
            long frame = t.Durations[0];
            int roll = frame > 0 ? (int)((3840 + frame - 1) / frame) : 0;   // 80 ms of pre-roll, in packets
            Full(st, "sgpd", 1, 0, b => { Str(b, "roll"); U32(b, 2); U32(b, 1); U16(b, (ushort)(short)-roll); });
            int head = Math.Min(roll, t.Sizes.Count), rest = t.Sizes.Count - head;
            Full(st, "sbgp", 0, 0, b =>
            {
                Str(b, "roll");
                U32(b, rest > 0 ? 2u : 1u);
                U32(b, (uint)head); U32(b, 0);
                if (rest > 0) { U32(b, (uint)rest); U32(b, 1); }
            });
        }
    }

    private static void Btrt(Stream s, long bitrate) => Box(s, "btrt", b => { U32(b, 0); U32(b, (uint)bitrate); U32(b, (uint)bitrate); });

    private static void Matrix(Stream b)
    {
        U32(b, 0x00010000); U32(b, 0); U32(b, 0);
        U32(b, 0); U32(b, 0x00010000); U32(b, 0);
        U32(b, 0); U32(b, 0); U32(b, 0x40000000);
    }

    private static void Box(Stream s, string type, Action<Stream> body)
    {
        var inner = new MemoryStream();
        body(inner);
        U32(s, (uint)(inner.Length + 8));
        Str(s, type);
        s.Write(inner.GetBuffer(), 0, (int)inner.Length);
    }

    private static void Full(Stream s, string type, byte version, uint flags, Action<Stream> body) =>
        Box(s, type, b => { U32(b, ((uint)version << 24) | (flags & 0xFFFFFF)); body(b); });

    private static void U16(Stream s, ushort v) { Span<byte> b = stackalloc byte[2]; BinaryPrimitives.WriteUInt16BigEndian(b, v); s.Write(b); }
    private static void U32(Stream s, uint v) { Span<byte> b = stackalloc byte[4]; BinaryPrimitives.WriteUInt32BigEndian(b, v); s.Write(b); }
    private static void U64(Stream s, ulong v) { Span<byte> b = stackalloc byte[8]; BinaryPrimitives.WriteUInt64BigEndian(b, v); s.Write(b); }
    private static void Str(Stream s, string four) => s.Write(Encoding.ASCII.GetBytes(four));
    private static void Zero(Stream s, int n) { for (int i = 0; i < n; i++) s.WriteByte(0); }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (!_finished && _started) Finish();
        _mdatBuffer?.Dispose();
        if (!_leaveOpen) _out.Dispose();
    }
}
