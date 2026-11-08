// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// WebM (Matroska, DocType "webm") muxer: a port of FFmpeg 8.1's libavformat/matroskaenc.c in WebM mode, which is what
// PyAV / ffmpeg (and so ComfyUI's Save WEBM / Save Video) write. Given the same tracks and packets, the file is the one
// `ffmpeg -fflags +bitexact -c copy` writes (bit-exact mode: "Lavf" as muxing / writing app, track UIDs 1, 2, ...;
// set MuxingApp / WritingApp / track Uid to match a non-bitexact producer).
//
// What it writes, as matroskaenc does: EBML header (DocTypeVersion 4 with Opus or alpha / stereo metadata, else 2);
// Segment (size patched at the end on a seekable stream); a reserved SeekHead; Info (TimestampScale 1 ms, Duration patched
// at the end); Tracks; Tags (global tags, then one Tag per track whose DURATION is patched at the end); Clusters (new one
// after 5 s or 5 MB, or 4 KB before a video key frame, or when the 16-bit block timestamp would overflow); SimpleBlocks,
// or BlockGroups when a block needs BlockDuration / DiscardPadding / BlockAdditions (VP8 / VP9 alpha); Cues (video key
// frames, or the first key frame of each cluster without video). Packets of all tracks are interleaved by timestamp
// (ties by track order) like libavformat's interleaver, and an audio packet is held back one packet so the one that
// shares a video key frame's timestamp lands in that frame's cluster. A non-seekable stream gets matroskaenc's live-less
// streaming layout (SeekHead after the Tags, no Duration, no Cues, 1 s / 32 KB clusters).
//
// Timestamps are Matroska ticks of 1 ms (matroskaenc's stream time base). Convert with round-half-away-from-zero, as
// av_rescale_q does: ms = round(pts * 1000 * tb.num / tb.den).
//
// This is a container writer on the host: the frame payloads are already encoded (on the accelerator) and arrive here
// for a stream, a terminal file sink; nothing here touches codec data beyond copying it (and removing AV1 temporal
// delimiter / padding OBUs, as matroskaenc's AV1 reformat does).

using System.Buffers.Binary;
using System.Globalization;
using System.Text;

namespace SpawnDev.Codecs.Container.Matroska;

/// <summary>Colour metadata of a video track (Matroska Colour element; ISO/IEC 23091-4 code points, null = not written).</summary>
public sealed record WebMColour
{
    /// <summary>MatrixCoefficients (AVColorSpace).</summary>
    public int? MatrixCoefficients { get; init; }
    /// <summary>Range: 1 = broadcast (tv / limited), 2 = full (pc).</summary>
    public int? Range { get; init; }
    /// <summary>TransferCharacteristics (AVColorTransferCharacteristic).</summary>
    public int? TransferCharacteristics { get; init; }
    /// <summary>Primaries (AVColorPrimaries).</summary>
    public int? Primaries { get; init; }
    /// <summary>ChromaSitingHorz (1 = left / co-sited, 2 = half).</summary>
    public int? ChromaSitingHorz { get; init; }
    /// <summary>ChromaSitingVert (1 = top / co-sited, 2 = half).</summary>
    public int? ChromaSitingVert { get; init; }
}

/// <summary>A track of a <see cref="WebMWriter"/>.</summary>
public abstract record WebMTrack
{
    /// <summary>Matroska CodecID: "V_VP8", "V_VP9", "V_AV1", "A_OPUS", "A_VORBIS".</summary>
    public required string CodecId { get; init; }
    /// <summary>CodecPrivate payload (OpusHead, av1C, Xiph-laced Vorbis headers); null / empty = none.</summary>
    public byte[]? CodecPrivate { get; init; }
    /// <summary>
    /// Room reserved for CodecPrivate in bytes of payload (0 = its own size). ffmpeg reserves room when the codec
    /// configuration is not known at header time (AV1 from libaom: 100 + the largest sequence header) and pads with a
    /// Void; give the same value to write the same layout.
    /// </summary>
    public int CodecPrivateReserve { get; init; }
    /// <summary>TrackUID (null = the track's position, 1-based, as ffmpeg's bit-exact mode).</summary>
    public ulong? Uid { get; init; }
    /// <summary>TrackName (null = none).</summary>
    public string? Name { get; init; }
    /// <summary>Language (null / empty = "und").</summary>
    public string? Language { get; init; }
    /// <summary>FlagDefault (true = not written, its default value).</summary>
    public bool IsDefault { get; init; } = true;
    /// <summary>FlagForced.</summary>
    public bool IsForced { get; init; }
    /// <summary>DefaultDuration in ns (null = none; ffmpeg: 1e9 / frame rate for video, frame samples / rate for audio).</summary>
    public long? DefaultDurationNs { get; init; }
    /// <summary>
    /// Per-track tags, in order (ffmpeg stream metadata). Written as SimpleTags with upper-cased names; "title",
    /// "language", "duration", "creation_time", "encoding_tool", "stereo_mode" are not tags (as in matroskaenc).
    /// </summary>
    public IReadOnlyList<KeyValuePair<string, string>>? Tags { get; init; }
}

/// <summary>A video track (V_VP8 / V_VP9 / V_AV1).</summary>
public sealed record WebMVideoTrack : WebMTrack
{
    /// <summary>PixelWidth.</summary>
    public required int Width { get; init; }
    /// <summary>PixelHeight.</summary>
    public required int Height { get; init; }
    /// <summary>FlagInterlaced: null = not written, false = progressive (2), true = interlaced (1).</summary>
    public bool? Interlaced { get; init; }
    /// <summary>AlphaMode 1: the frames carry an alpha plane in BlockAdditional 1 (VP8 / VP9 with alpha).</summary>
    public bool AlphaMode { get; init; }
    /// <summary>DisplayWidth / DisplayHeight (null = none).</summary>
    public (int Width, int Height)? DisplaySize { get; init; }
    /// <summary>Colour element (null = none).</summary>
    public WebMColour? Colour { get; init; }
}

/// <summary>An audio track (A_OPUS / A_VORBIS).</summary>
public sealed record WebMAudioTrack : WebMTrack
{
    /// <summary>SamplingFrequency in Hz.</summary>
    public required double SampleRate { get; init; }
    /// <summary>Channels.</summary>
    public required int Channels { get; init; }
    /// <summary>BitDepth (null = none).</summary>
    public int? BitDepth { get; init; }
    /// <summary>
    /// Encoder delay in samples (Opus pre-skip): written as CodecDelay (ns) and added to every timestamp of the track,
    /// rounded to ms, as matroskaenc's initial_padding.
    /// </summary>
    public int InitialPaddingSamples { get; init; }
}

/// <summary>Settings of a <see cref="WebMWriter"/>.</summary>
public sealed record WebMWriterOptions
{
    /// <summary>MuxingApp (ffmpeg: "Lavf" in bit-exact mode, else "Lavf62.x.100").</summary>
    public string MuxingApp { get; init; } = "Lavf";
    /// <summary>WritingApp (ffmpeg: as MuxingApp unless the encoding_tool metadata is set).</summary>
    public string WritingApp { get; init; } = "Lavf";
    /// <summary>Segment Title (null = none).</summary>
    public string? Title { get; init; }
    /// <summary>Global tags, in order (ffmpeg format metadata), written as a Tag without targets.</summary>
    public IReadOnlyList<KeyValuePair<string, string>>? Tags { get; init; }
    /// <summary>Cluster time limit in ms (null = ffmpeg's: 5000 seekable, 1000 not).</summary>
    public long? ClusterTimeLimitMs { get; init; }
    /// <summary>Cluster size limit in bytes (null = ffmpeg's: 5 MB seekable, 32 KB not).</summary>
    public int? ClusterSizeLimit { get; init; }
    /// <summary>libavformat max_interleave_delta in ms (default 10 s): a track this far behind no longer holds the others.</summary>
    public long MaxInterleaveDeltaMs { get; init; } = 10_000;
}

/// <summary>WebM muxer, a port of FFmpeg's matroskaenc (see the file header). Add every track, write frames, then Finish.</summary>
public sealed class WebMWriter : IDisposable
{
    // EBML / Matroska element IDs
    private const uint IdEbml = 0x1A45DFA3, IdEbmlVersion = 0x4286, IdEbmlReadVersion = 0x42F7, IdEbmlMaxIdLength = 0x42F2,
        IdEbmlMaxSizeLength = 0x42F3, IdDocType = 0x4282, IdDocTypeVersion = 0x4287, IdDocTypeReadVersion = 0x4285, IdVoid = 0xEC;
    private const uint IdSegment = 0x18538067, IdSeekHead = 0x114D9B74, IdSeek = 0x4DBB, IdSeekId = 0x53AB, IdSeekPosition = 0x53AC,
        IdInfo = 0x1549A966, IdTimestampScale = 0x2AD7B1, IdDuration = 0x4489, IdTitle = 0x7BA9, IdMuxingApp = 0x4D80, IdWritingApp = 0x5741,
        IdTracks = 0x1654AE6B, IdTrackEntry = 0xAE, IdTrackNumber = 0xD7, IdTrackUid = 0x73C5, IdFlagLacing = 0x9C, IdName = 0x536E,
        IdLanguage = 0x22B59C, IdFlagDefault = 0x88, IdFlagForced = 0x55AA, IdCodecId = 0x86, IdTrackType = 0x83,
        IdDefaultDuration = 0x23E383, IdCodecDelay = 0x56AA, IdSeekPreRoll = 0x56BB, IdCodecPrivate = 0x63A2,
        IdVideo = 0xE0, IdPixelWidth = 0xB0, IdPixelHeight = 0xBA, IdFlagInterlaced = 0x9A, IdAlphaMode = 0x53C0,
        IdDisplayWidth = 0x54B0, IdDisplayHeight = 0x54BA, IdColour = 0x55B0, IdMatrixCoefficients = 0x55B1, IdChromaSitingHorz = 0x55B7,
        IdChromaSitingVert = 0x55B8, IdRange = 0x55B9, IdTransferCharacteristics = 0x55BA, IdPrimaries = 0x55BB,
        IdAudio = 0xE1, IdChannels = 0x9F, IdSamplingFrequency = 0xB5, IdBitDepth = 0x6264,
        IdTags = 0x1254C367, IdTag = 0x7373, IdTargets = 0x63C0, IdTagTrackUid = 0x63C5, IdSimpleTag = 0x67C8, IdTagName = 0x45A3,
        IdTagString = 0x4487, IdCluster = 0x1F43B675, IdTimestamp = 0xE7, IdSimpleBlock = 0xA3, IdBlockGroup = 0xA0, IdBlock = 0xA1,
        IdBlockDuration = 0x9B, IdBlockReference = 0xFB, IdDiscardPadding = 0x75A2, IdBlockAdditions = 0x75A1, IdBlockMore = 0xA6,
        IdBlockAddId = 0xEE, IdBlockAdditional = 0xA5, IdCues = 0x1C53BB6B, IdCuePoint = 0xBB, IdCueTime = 0xB3,
        IdCueTrackPositions = 0xB7, IdCueTrack = 0xF7, IdCueClusterPosition = 0xF1, IdCueRelativePosition = 0xF0, IdCueDuration = 0xB2;

    private const int MaxSeekHeadEntries = 7, MaxSeekEntrySize = 21, MaxCueTrackPosSize = 40;
    private const int DurationStringLength = 19;
    private const int DurationSimpleTagSize = 2 + 1 + (2 + 1 + 8) + (2 + 1 + DurationStringLength);
    private const ulong OpusSeekPreRollNs = 80_000_000;

    private static readonly int[] OpusFrameDuration =
    {
        480, 960, 1920, 2880, 480, 960, 1920, 2880, 480, 960, 1920, 2880, 480, 960, 480, 960,
        120, 240, 480, 960, 120, 240, 480, 960, 120, 240, 480, 960, 120, 240, 480, 960,
    };

    private sealed class Track
    {
        public WebMTrack Def = null!;
        public int Index, Number, NumberSize;
        public ulong Uid;
        public bool IsVideo, IsOpus, IsAv1, HasCue;
        public long TsOffset, LastTimestamp, Duration, DurationTagOffset = -1;
        public ulong DefaultDurationLow, DefaultDurationHigh;
    }

    private sealed record Packet(int Track, byte[] Data, long Timestamp, bool Keyframe, long Duration, byte[]? BlockAdditional,
        long DiscardPaddingNs, long Order);

    private readonly Stream _out;
    private readonly bool _leaveOpen, _seekable;
    private readonly WebMWriterOptions _o;
    private readonly List<Track> _tracks = new();
    private bool _started, _finished, _haveVideo;

    private long _segmentOffset;
    private long _seekHeadPos;
    private const int SeekHeadReserved = MaxSeekHeadEntries * MaxSeekEntrySize + 14;
    private readonly List<(uint Id, long SegmentPos)> _seekEntries = new();
    private byte[]? _info, _tracksBuf;
    private MemoryStream? _tags;
    private long _infoPos, _tracksPos, _tagsPos, _durationOffset;
    private long _duration;

    private MemoryStream? _cluster;
    private long _clusterPos = -1, _clusterPts;
    private long _clusterTimeLimit, _clusterSizeLimit;
    private Packet? _cachedAudio;
    private readonly List<(long Pts, int Track, long ClusterPos, long RelativePos, long Duration)> _cues = new();

    private readonly List<Packet> _queue = new();   // libavformat interleaving buffer, sorted
    private readonly int[] _queued = new int[64];
    private long _order;

    /// <summary>A writer onto <paramref name="stream"/> (seekable for Duration, Cues and the SeekHead; see the file header).</summary>
    public WebMWriter(Stream stream, WebMWriterOptions? options = null, bool leaveOpen = false)
    {
        ArgumentNullException.ThrowIfNull(stream);
        if (!stream.CanWrite) throw new ArgumentException("The stream must be writable.", nameof(stream));
        _out = stream;
        _leaveOpen = leaveOpen;
        _seekable = stream.CanSeek;
        _o = options ?? new WebMWriterOptions();
    }

    /// <summary>Add a video track; returns its number (1-based, Matroska TrackNumber). All tracks before the first frame.</summary>
    public int AddVideoTrack(WebMVideoTrack track) => AddTrack(track, true);

    /// <summary>Add an audio track; returns its number (1-based). All tracks before the first frame.</summary>
    public int AddAudioTrack(WebMAudioTrack track) => AddTrack(track, false);

    private int AddTrack(WebMTrack def, bool video)
    {
        ArgumentNullException.ThrowIfNull(def);
        if (_started) throw new InvalidOperationException("Tracks must be added before the first frame.");
        if (_tracks.Count == _queued.Length) throw new InvalidOperationException("Too many tracks.");
        string id = def.CodecId;
        bool ok = video ? id is "V_VP8" or "V_VP9" or "V_AV1" : id is "A_OPUS" or "A_VORBIS";
        if (!ok) throw new NotSupportedException($"WebM carries VP8 / VP9 / AV1 video and Opus / Vorbis audio; got {id}.");
        if (def is WebMVideoTrack v && (v.Width <= 0 || v.Height <= 0)) throw new ArgumentException("Video size must be positive.");
        if (def is WebMAudioTrack a && (a.SampleRate <= 0 || a.Channels <= 0)) throw new ArgumentException("Audio rate and channels must be positive.");
        var t = new Track
        {
            Def = def, Index = _tracks.Count, Number = _tracks.Count + 1, IsVideo = video, IsOpus = id == "A_OPUS", IsAv1 = id == "V_AV1",
        };
        t.NumberSize = NumSize((ulong)t.Number);
        t.Uid = def.Uid ?? (ulong)(t.Index + 1);
        if (def.DefaultDurationNs is long dd)
        {
            t.DefaultDurationLow = (ulong)(dd / 1_000_000);
            t.DefaultDurationHigh = t.DefaultDurationLow + (dd % 1_000_000 != 0 ? 1UL : 0UL);
        }
        if (def is WebMAudioTrack au && au.InitialPaddingSamples > 0)
            t.TsOffset = RescaleRound(au.InitialPaddingSamples, 1000, (long)au.SampleRate);
        _tracks.Add(t);
        return t.Number;
    }

    /// <summary>
    /// Write one frame of track <paramref name="track"/>. <paramref name="timestampMs"/> is in ms ticks (see the file
    /// header); <paramref name="durationMs"/> 0 = unknown. <paramref name="alpha"/> = the frame's alpha plane bitstream
    /// (VP8 / VP9 BlockAdditional 1). <paramref name="discardPaddingNs"/> = samples to drop at the end (Opus), in ns.
    /// </summary>
    public void WriteFrame(int track, ReadOnlySpan<byte> data, long timestampMs, bool keyframe, long durationMs = 0,
        ReadOnlySpan<byte> alpha = default, long discardPaddingNs = 0)
    {
        if (_finished) throw new InvalidOperationException("The writer is finished.");
        if (track < 1 || track > _tracks.Count) throw new ArgumentOutOfRangeException(nameof(track));
        if (_tracks.Count == 0) throw new InvalidOperationException("No tracks.");
        if (!_started) WriteHeader();
        var t = _tracks[track - 1];
        byte[] payload = t.IsAv1 ? FilterAv1Obus(data) : data.ToArray();
        var p = new Packet(t.Index, payload, timestampMs, keyframe, Math.Max(0, durationMs), alpha.IsEmpty ? null : alpha.ToArray(),
            discardPaddingNs, _order++);
        Interleave(p, false);
    }

    /// <summary>Flush, write the index and patch the header fields (seekable streams). The stream stays at the end.</summary>
    public void Finish()
    {
        if (_finished) return;
        if (!_started) WriteHeader();
        Interleave(null, true);
        WriteTrailer();
        _finished = true;
        _out.Flush();
    }

    /// <summary>Finish (if not yet) and release the stream unless leaveOpen.</summary>
    public void Dispose()
    {
        if (!_finished && _started) Finish();
        if (!_leaveOpen) _out.Dispose();
    }

    // ------------------------------------------------------------------ interleaving (libavformat ff_interleave_packet_per_dts)

    private void Interleave(Packet? p, bool flush)
    {
        if (p != null)
        {
            // keep sorted by timestamp, then track; equal keys keep arrival order
            int i = _queue.Count;
            while (i > 0 && Compare(_queue[i - 1], p) > 0) i--;
            _queue.Insert(i, p);
            _queued[p.Track]++;
        }
        while (_queue.Count > 0)
        {
            int streamsWithPackets = 0;
            for (int k = 0; k < _tracks.Count; k++) if (_queued[k] > 0) streamsWithPackets++;
            bool release = flush || streamsWithPackets >= _tracks.Count;
            if (!release && _o.MaxInterleaveDeltaMs > 0)
            {
                // a track that has fallen this far behind no longer holds the others back
                long first = _queue[0].Timestamp;
                long last = long.MinValue;
                foreach (var q in _queue) last = Math.Max(last, q.Timestamp);
                if (last - first > _o.MaxInterleaveDeltaMs) release = true;
            }
            if (!release) break;
            var head = _queue[0];
            _queue.RemoveAt(0);
            _queued[head.Track]--;
            WritePacket(head);
        }
    }

    /// <summary>libavformat's order: the packet timestamps as given (matroskaenc adds the codec-delay offset later), then track.</summary>
    private static int Compare(Packet a, Packet b)
    {
        long ta = a.Timestamp, tb = b.Timestamp;
        if (ta != tb) return ta < tb ? -1 : 1;
        if (a.Track != b.Track) return a.Track < b.Track ? -1 : 1;
        return a.Order < b.Order ? -1 : 1;
    }

    // ------------------------------------------------------------------ EBML primitives (matroskaenc.c)

    private static int IdSize(uint id) => id > 0xFFFFFF ? 4 : id > 0xFFFF ? 3 : id > 0xFF ? 2 : 1;

    private static void PutId(Stream s, uint id)
    {
        for (int i = IdSize(id) - 1; i >= 0; i--) s.WriteByte((byte)(id >> (i * 8)));
    }

    private static int NumSize(ulong num)
    {
        int bytes = 0;
        do { bytes++; } while ((num >>= 7) != 0);
        return bytes;
    }

    private static int LengthSize(ulong length) => NumSize(length + 1);

    private static void PutNum(Stream s, ulong num, int bytes)
    {
        num |= 1UL << (bytes * 7);
        for (int i = bytes - 1; i >= 0; i--) s.WriteByte((byte)(num >> (i * 8)));
    }

    private static void PutLength(Stream s, ulong length, int bytes)
    {
        int needed = LengthSize(length);
        if (bytes == 0) bytes = needed;
        if (bytes < needed) throw new InvalidOperationException("EBML length field too small.");
        PutNum(s, length, bytes);
    }

    private static void PutSizeUnknown(Stream s, int bytes)
    {
        s.WriteByte((byte)(0x1FF >> bytes));
        for (int i = 1; i < bytes; i++) s.WriteByte(0xFF);
    }

    private static int UintSize(ulong v)
    {
        int bytes = 0;
        do { bytes++; } while ((v >>= 8) != 0);
        return bytes;
    }

    private static int SintSize(long v) => UintSize(2 * (ulong)(v < 0 ? v ^ -1 : v));

    private static void PutUint(Stream s, uint id, ulong v)
    {
        int bytes = UintSize(v);
        PutId(s, id);
        PutLength(s, (ulong)bytes, 0);
        for (int i = bytes - 1; i >= 0; i--) s.WriteByte((byte)(v >> (i * 8)));
    }

    private static void PutSint(Stream s, uint id, long v)
    {
        int bytes = SintSize(v);
        PutId(s, id);
        PutLength(s, (ulong)bytes, 0);
        for (int i = bytes - 1; i >= 0; i--) s.WriteByte((byte)((ulong)v >> (i * 8)));
    }

    private static void PutBe64(Stream s, ulong v)
    {
        Span<byte> b = stackalloc byte[8];
        BinaryPrimitives.WriteUInt64BigEndian(b, v);
        s.Write(b);
    }

    private static void PutUid(Stream s, uint id, ulong uid)
    {
        PutId(s, id);
        PutLength(s, 8, 0);
        PutBe64(s, uid);
    }

    private static void PutFloat(Stream s, uint id, double v)
    {
        PutId(s, id);
        PutLength(s, 8, 0);
        PutBe64(s, (ulong)BitConverter.DoubleToInt64Bits(v));
    }

    private static void PutBinary(Stream s, uint id, ReadOnlySpan<byte> data)
    {
        PutId(s, id);
        PutLength(s, (ulong)data.Length, 0);
        s.Write(data);
    }

    private static void PutString(Stream s, uint id, string str) => PutBinary(s, id, Encoding.UTF8.GetBytes(str));

    private static void PutVoid(Stream s, int size)
    {
        if (size < 2) throw new InvalidOperationException("EBML Void needs at least 2 bytes.");
        PutId(s, IdVoid);
        if (size < 10) { size -= 2; PutLength(s, (ulong)size, 0); }
        else { size -= 9; PutLength(s, (ulong)size, 8); }
        for (int i = 0; i < size; i++) s.WriteByte(0);
    }

    /// <summary>start_ebml_master: id + an unknown-size length field of the width the expected size needs (8 when 0).</summary>
    private static (long Pos, int Bytes) StartMaster(Stream s, uint id, ulong expectedSize)
    {
        int bytes = expectedSize != 0 ? LengthSize(expectedSize) : 8;
        PutId(s, id);
        PutSizeUnknown(s, bytes);
        return (s.Position, bytes);
    }

    private static void EndMaster(Stream s, (long Pos, int Bytes) m)
    {
        long pos = s.Position;
        s.Position = m.Pos - m.Bytes;
        PutLength(s, (ulong)(pos - m.Pos), m.Bytes);
        s.Position = pos;
    }

    /// <summary>A master element whose length is the minimal one for its children (matroskaenc's EbmlWriter).</summary>
    private static void PutMaster(Stream s, uint id, ReadOnlySpan<byte> children) => PutBinary(s, id, children);

    private static long RescaleRound(long a, long b, long c)
    {
        // av_rescale_rnd(a, b, c, AV_ROUND_NEAR_INF): round half away from zero
        Int128 r = (Int128)a * b;
        Int128 half = c / 2;
        return (long)(r >= 0 ? (r + half) / c : -((-r + half) / c));
    }

    // ------------------------------------------------------------------ header (mkv_write_header)

    private void WriteHeader()
    {
        _started = true;
        // The header is built in memory (matroskaenc writes it into avio's buffer, so even a non-seekable output gets
        // its SeekHead in the reserved room), then emitted in one write.
        _base = _seekable ? _out.Position : 0;
        _hdr = new MemoryStream();
        _haveVideo = _tracks.Any(t => t.IsVideo);
        int version = 2;
        bool HasMeta(IReadOnlyList<KeyValuePair<string, string>>? tags) =>
            tags != null && tags.Any(k => k.Key.Equals("stereo_mode", StringComparison.OrdinalIgnoreCase)
                                       || k.Key.Equals("alpha_mode", StringComparison.OrdinalIgnoreCase));
        if (HasMeta(_o.Tags)) version = 4;
        foreach (var t in _tracks) if (t.IsOpus || HasMeta(t.Def.Tags)) version = 4;

        var h = new MemoryStream();
        PutUint(h, IdEbmlVersion, 1);
        PutUint(h, IdEbmlReadVersion, 1);
        PutUint(h, IdEbmlMaxIdLength, 4);
        PutUint(h, IdEbmlMaxSizeLength, 8);
        PutString(h, IdDocType, "webm");
        PutUint(h, IdDocTypeVersion, (ulong)version);
        PutUint(h, IdDocTypeReadVersion, 2);
        PutMaster(_hdr, IdEbml, h.ToArray());

        PutId(_hdr, IdSegment);
        PutSizeUnknown(_hdr, 8);
        _segmentOffset = Pos;

        _seekHeadPos = Pos;
        PutVoid(_hdr, SeekHeadReserved);

        WriteInfo();
        WriteTracks();
        WriteTags();
        if (!_seekable)
        {
            _hdr.Position = _seekHeadPos - _base;
            WriteSeekHeadInto(_hdr);
            _hdr.Position = _hdr.Length;
        }
        var header = _hdr.ToArray();
        _hdr = null;
        Emit(header);

        _clusterPos = -1;
        _clusterTimeLimit = _o.ClusterTimeLimitMs ?? (_seekable ? 5000 : 1000);
        _clusterSizeLimit = _o.ClusterSizeLimit ?? (_seekable ? 5 * 1024 * 1024 : 32 * 1024);
    }

    /// <summary>Position relative to where the writer started (matroskaenc's avio_tell).</summary>
    private long Pos => _hdr != null ? _base + _hdr.Position : (_seekable ? _out.Position : _base + _written);
    private long _written, _base;
    private MemoryStream? _hdr;

    private void Emit(ReadOnlySpan<byte> bytes)
    {
        if (_hdr != null) { _hdr.Write(bytes); return; }
        _out.Write(bytes);
        _written += bytes.Length;
    }

    private void EmitElement(uint id, ReadOnlySpan<byte> payload, int lengthSize = 0)
    {
        var h = new MemoryStream(16);
        PutId(h, id);
        PutLength(h, (ulong)payload.Length, lengthSize);
        Emit(h.ToArray());
        Emit(payload);
    }

    private void AddSeekEntry(uint id, long filePos) => _seekEntries.Add((id, filePos - _segmentOffset));

    /// <summary>end_ebml_master_crc32_tentatively (no CRC in WebM): write the element, remember where for a rewrite.</summary>
    private long WriteTentatively(uint id, byte[] payload)
    {
        long pos = Pos;
        AddSeekEntry(id, pos);
        EmitElement(id, payload);
        return pos;
    }

    private void WriteInfo()
    {
        var b = new MemoryStream();
        PutUint(b, IdTimestampScale, 1_000_000);
        if (_o.Title != null) PutString(b, IdTitle, _o.Title);
        PutString(b, IdMuxingApp, _o.MuxingApp);
        PutString(b, IdWritingApp, _o.WritingApp);
        _durationOffset = b.Position;
        if (_seekable) PutVoid(b, 11);   // room for the Duration float, written at the end
        _info = b.ToArray();
        _infoPos = WriteTentatively(IdInfo, _info);
    }

    private void WriteTracks()
    {
        var b = new MemoryStream();
        foreach (var t in _tracks) WriteTrack(b, t);
        _tracksBuf = b.ToArray();
        _tracksPos = WriteTentatively(IdTracks, _tracksBuf);
    }

    private void WriteTrack(MemoryStream pb, Track t)
    {
        var d = t.Def;
        var master = StartMaster(pb, IdTrackEntry, 0);
        PutUint(pb, IdTrackNumber, (ulong)t.Number);
        PutUid(pb, IdTrackUid, t.Uid);
        PutUint(pb, IdFlagLacing, 0);
        if (d.Name != null) PutString(pb, IdName, d.Name);
        PutString(pb, IdLanguage, string.IsNullOrEmpty(d.Language) ? "und" : d.Language);
        if (!d.IsDefault) PutUint(pb, IdFlagDefault, 0);
        if (d.IsForced) PutUint(pb, IdFlagForced, 1);
        PutString(pb, IdCodecId, d.CodecId);
        if (d is WebMVideoTrack v)
        {
            PutUint(pb, IdTrackType, 1);
            if (d.DefaultDurationNs is long dd) PutUint(pb, IdDefaultDuration, (ulong)dd);
            var vb = new MemoryStream();
            PutUint(vb, IdPixelWidth, (ulong)v.Width);
            PutUint(vb, IdPixelHeight, (ulong)v.Height);
            if (v.Interlaced is bool il) PutUint(vb, IdFlagInterlaced, il ? 1UL : 2UL);
            if (v.AlphaMode || AlphaModeTag(v)) PutUint(vb, IdAlphaMode, 1);
            if (v.DisplaySize is { } ds)
            {
                PutUint(vb, IdDisplayWidth, (ulong)ds.Width);
                PutUint(vb, IdDisplayHeight, (ulong)ds.Height);
            }
            if (v.Colour is { } c)
            {
                var cb = new MemoryStream();
                if (c.TransferCharacteristics is int tc) PutUint(cb, IdTransferCharacteristics, (ulong)tc);
                if (c.MatrixCoefficients is int mc) PutUint(cb, IdMatrixCoefficients, (ulong)mc);
                if (c.Primaries is int pr) PutUint(cb, IdPrimaries, (ulong)pr);
                if (c.Range is int rg) PutUint(cb, IdRange, (ulong)rg);
                if (c.ChromaSitingHorz is int sh && c.ChromaSitingVert is int sv)
                {
                    PutUint(cb, IdChromaSitingHorz, (ulong)sh);
                    PutUint(cb, IdChromaSitingVert, (ulong)sv);
                }
                if (cb.Length > 0) PutMaster(vb, IdColour, cb.ToArray());   // an empty Colour is discarded
            }
            PutMaster(pb, IdVideo, vb.ToArray());
        }
        else
        {
            var a = (WebMAudioTrack)d;
            if (a.InitialPaddingSamples > 0)
                PutUint(pb, IdCodecDelay, (ulong)RescaleRound(a.InitialPaddingSamples, 1_000_000_000, (long)a.SampleRate));
            if (t.IsOpus) PutUint(pb, IdSeekPreRoll, OpusSeekPreRollNs);
            PutUint(pb, IdTrackType, 2);
            if (d.DefaultDurationNs is long dd) PutUint(pb, IdDefaultDuration, (ulong)dd);
            var sub = StartMaster(pb, IdAudio, 6 + 4 * 9);
            PutUint(pb, IdChannels, (ulong)a.Channels);
            PutFloat(pb, IdSamplingFrequency, a.SampleRate);
            if (a.BitDepth is int bd) PutUint(pb, IdBitDepth, (ulong)bd);
            EndMaster(pb, sub);
        }
        PutCodecPrivate(pb, Math.Max(d.CodecPrivateReserve, d.CodecPrivate?.Length ?? 0), d.CodecPrivate ?? System.Array.Empty<byte>());
        EndMaster(pb, master);
    }

    /// <summary>mkv_put_codecprivate: the CodecPrivate element, then a Void up to the reserved room.</summary>
    private static void PutCodecPrivate(Stream pb, int maxPayloadSize, byte[] codecPrivate)
    {
        if (maxPayloadSize == 0) return;
        int totalSize = 2 + LengthSize((ulong)maxPayloadSize) + maxPayloadSize;
        int totalCodecPrivate = 0;
        if (codecPrivate.Length > 0)
        {
            int lengthSize = LengthSize((ulong)codecPrivate.Length);
            totalCodecPrivate = 2 + lengthSize + codecPrivate.Length;
            if (totalCodecPrivate + 1 == totalSize)
            {
                lengthSize++;   // a 1-byte Void does not exist: widen the length field instead
                totalCodecPrivate++;
            }
            PutId(pb, IdCodecPrivate);
            PutLength(pb, (ulong)codecPrivate.Length, lengthSize);
            pb.Write(codecPrivate);
        }
        if (totalCodecPrivate < totalSize) PutVoid(pb, totalSize - totalCodecPrivate);
    }

    private static bool AlphaModeTag(WebMTrack t) =>
        t.Tags != null && t.Tags.Any(k => k.Key.Equals("alpha_mode", StringComparison.OrdinalIgnoreCase)
            && long.TryParse(k.Value, NumberStyles.Integer, CultureInfo.InvariantCulture, out long n) && n != 0);

    private static bool IsTagName(string name, bool track) =>
        !name.Equals("title", StringComparison.OrdinalIgnoreCase) && !name.Equals("stereo_mode", StringComparison.OrdinalIgnoreCase)
        && !name.Equals("creation_time", StringComparison.OrdinalIgnoreCase) && !name.Equals("encoding_tool", StringComparison.OrdinalIgnoreCase)
        && !name.Equals("duration", StringComparison.OrdinalIgnoreCase)
        && !(track && name.Equals("language", StringComparison.OrdinalIgnoreCase));

    /// <summary>mkv_write_simpletag: name upper-cased, spaces to underscores (ffmpeg's generic-to-Matroska names applied).</summary>
    private static void PutSimpleTag(Stream s, string key, string value)
    {
        string mapped = key.ToLowerInvariant() switch { "performer" => "LEAD_PERFORMER", "track" => "PART_NUMBER", _ => key };
        var name = new StringBuilder(mapped.Length);
        foreach (char ch in mapped) name.Append(ch == ' ' ? '_' : ch is >= 'a' and <= 'z' ? (char)(ch - 32) : ch);
        var b = new MemoryStream();
        PutString(b, IdTagName, name.ToString());
        PutString(b, IdTagString, value);
        PutMaster(s, IdSimpleTag, b.ToArray());
    }

    /// <summary>mkv_write_tag: Targets (+ TrackUID), the tags, the reserved DURATION room; null when nothing to write.</summary>
    private byte[]? BuildTag(IReadOnlyList<KeyValuePair<string, string>>? tags, Track? track, int reserved, out long voidOffset)
    {
        var b = new MemoryStream();
        var targets = StartMaster(b, IdTargets, 4 + 1 + 8);
        if (track != null) PutUid(b, IdTagTrackUid, track.Uid);
        EndMaster(b, targets);
        bool written = false;
        if (tags != null)
            foreach (var kv in tags)
                if (IsTagName(kv.Key, track != null))
                {
                    PutSimpleTag(b, kv.Key, kv.Value);
                    written = true;
                }
        voidOffset = b.Position;
        if (reserved > 0) PutVoid(b, reserved);
        else if (!written) return null;
        return b.ToArray();
    }

    private void WriteTags()
    {
        _tags = new MemoryStream();
        bool any = false;
        var global = BuildTag(_o.Tags, null, 0, out _);
        if (global != null) { PutBinary(_tags, IdTag, global); any = true; }
        foreach (var t in _tracks)
        {
            var tag = BuildTag(t.Def.Tags, t, _seekable ? DurationSimpleTagSize : 0, out long voidOffset);
            if (tag == null) continue;
            PutBinary(_tags, IdTag, tag);
            any = true;
            if (_seekable) t.DurationTagOffset = _tags.Position - DurationSimpleTagSize;
            _ = voidOffset;
        }
        if (!any) { _tags = null; return; }
        _tagsPos = WriteTentatively(IdTags, _tags.ToArray());
    }

    /// <summary>The SeekHead (Seek entries, then a Void filling the reserved room).</summary>
    private byte[] BuildSeekHead()
    {
        var b = new MemoryStream();
        foreach (var (id, segPos) in _seekEntries)
        {
            var entry = StartMaster(b, IdSeek, MaxSeekEntrySize);
            PutId(b, IdSeekId);
            PutLength(b, (ulong)IdSize(id), 0);
            PutId(b, id);
            PutUint(b, IdSeekPosition, (ulong)segPos);
            EndMaster(b, entry);
        }
        var o = new MemoryStream();
        PutBinary(o, IdSeekHead, b.ToArray());
        return o.ToArray();
    }

    /// <summary>The SeekHead and the Void after it, written over the reserved room at the stream's position.</summary>
    private void WriteSeekHeadInto(Stream s)
    {
        long start = s.Position;
        var sh = BuildSeekHead();
        s.Write(sh);
        PutVoid(s, (int)(SeekHeadReserved - (s.Position - start)));
    }

    // ------------------------------------------------------------------ packets (mkv_write_packet, mkv_write_packet_internal, mkv_write_block)

    private void WritePacket(Packet p)
    {
        var t = _tracks[p.Track];
        bool keyframe = p.Keyframe;
        if (_clusterPos != -1)
        {
            long clusterTime = p.Timestamp + t.TsOffset - _clusterPts;
            long clusterSize = _cluster!.Length;
            if (clusterSize > _clusterSizeLimit || clusterTime > _clusterTimeLimit || (t.IsVideo && keyframe && clusterSize > 4 * 1024))
                EndCluster();
        }
        if (_cachedAudio != null)
        {
            var c = _cachedAudio;
            _cachedAudio = null;
            WritePacketInternal(c);
        }
        if (!t.IsVideo) _cachedAudio = p.Data.Length > 0 ? p : null;
        else WritePacketInternal(p);
    }

    private void WritePacketInternal(Packet p)
    {
        var t = _tracks[p.Track];
        long ts = p.Timestamp + t.TsOffset;
        long duration = Math.Max(p.Duration, 0);
        if (_clusterPos != -1)
        {
            long ct = ts - _clusterPts;
            if ((short)ct != ct) EndCluster();
        }
        if (_clusterPos == -1)
        {
            _cluster = new MemoryStream();
            _clusterPos = Pos;
            PutUint(_cluster, IdTimestamp, (ulong)Math.Max(0, ts));
            _clusterPts = Math.Max(0, ts);
        }
        long relativePos = _cluster!.Length;
        WriteBlock(_cluster, t, p, ts, duration);
        if (p.Keyframe && _seekable && (t.IsVideo || (!_haveVideo && !t.HasCue)))
        {
            AddCue(ts, p.Track, _clusterPos, relativePos);
            t.HasCue = true;
        }
        t.LastTimestamp = ts;
        _duration = Math.Max(_duration, ts + duration);
        t.Duration = Math.Max(t.Duration, ts + duration);
    }

    private void AddCue(long ts, int track, long clusterPos, long relativePos)
    {
        if (ts < 0) return;
        int idx = _cues.Count;
        while (idx > 0 && _cues[idx - 1].Pts > ts) idx--;
        _cues.Insert(idx, (ts, track, clusterPos - _segmentOffset, relativePos, 0));
    }

    private static int OpusPacketSamples(byte[] b)
    {
        if (b.Length == 0) return -1;
        int code = b[0] & 3, config = b[0] >> 3, frames;
        if (code == 0) frames = 1;
        else if (code < 3) frames = 2;
        else
        {
            if (b.Length <= 1) return -1;
            frames = b[1] & 0x3F;
        }
        return frames * OpusFrameDuration[config];
    }

    private void WriteBlock(MemoryStream pb, Track t, Packet p, long ts, long duration)
    {
        short relTs = (short)(ts - _clusterPts);
        bool needDuration = false;
        if (duration > 0 && t.DefaultDurationHigh > 0 && (ulong)duration != t.DefaultDurationHigh && (ulong)duration != t.DefaultDurationLow)
            needDuration = true;
        else if (t.IsOpus)
        {
            int samples = OpusPacketSamples(p.Data);
            if (samples >= 0 && RescaleRound(samples, 1000, 48000) != duration) needDuration = true;
        }
        long discard = p.DiscardPaddingNs;
        bool additions = p.BlockAdditional != null;

        // the block itself: track number, 16-bit relative timestamp, flags, payload
        var block = new MemoryStream(p.Data.Length + 8);
        PutNum(block, (ulong)t.Number, t.NumberSize);
        block.WriteByte((byte)((ushort)relTs >> 8));
        block.WriteByte((byte)relTs);
        bool simple = !needDuration && discard == 0 && !additions;
        block.WriteByte((byte)(simple && p.Keyframe ? 0x80 : 0));
        block.Write(p.Data);

        if (simple)
        {
            PutBinary(pb, IdSimpleBlock, block.ToArray());
            return;
        }
        var g = new MemoryStream();
        PutBinary(g, IdBlock, block.ToArray());
        if (needDuration) PutUint(g, IdBlockDuration, (ulong)duration);
        if (discard != 0) PutSint(g, IdDiscardPadding, discard);
        if (additions)
        {
            var more = new MemoryStream();
            PutUint(more, IdBlockAddId, 1);
            PutBinary(more, IdBlockAdditional, p.BlockAdditional);
            var adds = new MemoryStream();
            PutMaster(adds, IdBlockMore, more.ToArray());
            PutMaster(g, IdBlockAdditions, adds.ToArray());
        }
        if (!p.Keyframe) PutSint(g, IdBlockReference, t.LastTimestamp - ts);
        PutMaster(pb, IdBlockGroup, g.ToArray());
    }

    private void EndCluster()
    {
        if (!_haveVideo) foreach (var t in _tracks) t.HasCue = false;
        _clusterPos = -1;
        EmitElement(IdCluster, _cluster!.ToArray());
        _cluster = null;
    }

    // ------------------------------------------------------------------ trailer (mkv_write_trailer)

    private void WriteTrailer()
    {
        if (_cachedAudio != null)
        {
            var c = _cachedAudio;
            _cachedAudio = null;
            WritePacketInternal(c);
        }
        if (_clusterPos != -1)
        {
            _clusterPos = -1;
            EmitElement(IdCluster, _cluster!.ToArray());
            _cluster = null;
        }
        if (!_seekable) return;

        long endPos = Pos;
        if (_cues.Count > 0)
        {
            var cues = new MemoryStream();
            for (int i = 0; i < _cues.Count;)
            {
                long pts = _cues[i].Pts;
                var point = new MemoryStream();
                PutUint(point, IdCueTime, (ulong)pts);
                foreach (var tr in _tracks) tr.HasCue = false;
                do
                {
                    var e = _cues[i];
                    var tr = _tracks[e.Track];
                    if (tr.HasCue) continue;
                    tr.HasCue = true;
                    var pos = StartMaster(point, IdCueTrackPositions, MaxCueTrackPosSize);
                    PutUint(point, IdCueTrack, (ulong)tr.Number);
                    PutUint(point, IdCueClusterPosition, (ulong)e.ClusterPos);
                    PutUint(point, IdCueRelativePosition, (ulong)e.RelativePos);
                    if (e.Duration > 0) PutUint(point, IdCueDuration, (ulong)e.Duration);
                    EndMaster(point, pos);
                } while (++i < _cues.Count && _cues[i].Pts == pts);
                PutBinary(cues, IdCuePoint, point.ToArray());
            }
            AddSeekEntry(IdCues, Pos);
            EmitElement(IdCues, cues.ToArray());
            endPos = Pos;
        }

        // Segment size
        if (endPos - _segmentOffset < (long)((1UL << 56) - 1))
        {
            _out.Position = _segmentOffset - 8;
            var sz = new MemoryStream(8);
            PutLength(sz, (ulong)(endPos - _segmentOffset), 8);
            _out.Write(sz.ToArray());
        }

        // SeekHead in its reserved room, the rest a Void
        _out.Position = _seekHeadPos;
        var shBuf = new MemoryStream();
        WriteSeekHeadInto(shBuf);
        _out.Write(shBuf.ToArray());

        // Info with the Duration in place of its Void
        var info = new MemoryStream(_info!);
        info.Position = _durationOffset;
        PutFloat(info, IdDuration, _duration);
        _out.Position = _infoPos;
        WriteElementAt(IdInfo, info.ToArray());

        // Tracks (unchanged for WebM)
        _out.Position = _tracksPos;
        WriteElementAt(IdTracks, _tracksBuf!);

        // Tags with each track's DURATION
        if (_tags != null)
        {
            foreach (var t in _tracks)
            {
                if (t.DurationTagOffset <= 0) continue;
                double sec = t.Duration * 0.001;
                string str = string.Format(CultureInfo.InvariantCulture, "{0:00}:{1:00}:{2}", (int)sec / 3600, ((int)sec / 60) % 60,
                    FormatSeconds(sec % 60));
                var bytes = new byte[DurationStringLength];   // snprintf into 20 bytes, 19 written: the NUL is kept
                Encoding.ASCII.GetBytes(str, 0, Math.Min(str.Length, DurationStringLength), bytes, 0);
                _tags.Position = t.DurationTagOffset;
                var st = StartMaster(_tags, IdSimpleTag, 2 + 1 + 8 + 23);
                PutString(_tags, IdTagName, "DURATION");
                PutBinary(_tags, IdTagString, bytes);
                EndMaster(_tags, st);
            }
            _out.Position = _tagsPos;
            WriteElementAt(IdTags, _tags.ToArray());
        }
        _out.Position = endPos;
    }

    /// <summary>printf "%012.9f": nine decimals, zero-padded to 12 characters.</summary>
    private static string FormatSeconds(double s)
    {
        string f = s.ToString("F9", CultureInfo.InvariantCulture);
        return f.Length >= 12 ? f : new string('0', 12 - f.Length) + f;
    }

    private void WriteElementAt(uint id, byte[] payload)
    {
        var h = new MemoryStream(16);
        PutId(h, id);
        PutLength(h, (ulong)payload.Length, 0);
        _out.Write(h.ToArray());
        _out.Write(payload);
    }

    // ------------------------------------------------------------------ AV1 (mkv_reformat_av1 -> ff_av1_filter_obus)

    /// <summary>Drop temporal delimiter, redundant frame header, tile list and padding OBUs, as ffmpeg's AV1 reformat.</summary>
    private static byte[] FilterAv1Obus(ReadOnlySpan<byte> data)
    {
        var o = new MemoryStream(data.Length);
        int p = 0;
        while (p < data.Length)
        {
            int start = p;
            byte header = data[p++];
            int type = (header >> 3) & 0xF;
            bool extension = (header & 4) != 0, hasSize = (header & 2) != 0;
            if (extension) p++;
            long size;
            if (hasSize)
            {
                size = 0;
                for (int i = 0; i < 8; i++)
                {
                    if (p >= data.Length) throw new InvalidDataException("Truncated AV1 OBU size.");
                    byte b = data[p++];
                    size |= (long)(b & 0x7F) << (7 * i);
                    if ((b & 0x80) == 0) break;
                }
            }
            else size = data.Length - p;
            if (p + size > data.Length) throw new InvalidDataException("AV1 OBU runs past the packet.");
            p += (int)size;
            bool drop = type is 2 or 7 or 8 or 15;   // TD, redundant frame header, tile list, padding
            if (!drop) o.Write(data.Slice(start, p - start));
        }
        return o.ToArray();
    }
}
