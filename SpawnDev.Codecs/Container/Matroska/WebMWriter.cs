// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// WebM (Matroska subset) muxer: frames already-encoded VP8 / VP9 / AV1 video and Opus / Vorbis audio packets into a
// caller's Stream. Written clean-room from the public specifications: RFC 8794 (EBML), RFC 9559 (Matroska) and the WebM
// container guidelines (webmproject.org). It is a host-side container writer (like IvfWriter): it only frames packets that
// an encoder already produced; it does no codec work.
//
// Layout it produces (matching what the ffmpeg command line tool writes with -fflags +bitexact, observed as a black box on
// the fixtures in SpawnDev.Codecs.Demo.Shared/TestData/WebMMux):
//   EBML header (DocType "webm", DocTypeVersion 4 when a track carries CodecDelay / SeekPreRoll, else 2)
//   Segment (8-byte size; "unknown" when the output cannot seek)
//     161 bytes reserved for the SeekHead (Info, Tracks, Tags, Cues), the rest filled by a Void
//     Info (TimestampScale 1 ms, Title, MuxingApp, WritingApp, Duration when seekable)
//     Tracks (each TrackEntry with an 8-byte size field; CodecPrivate last, optionally inside reserved room + Void)
//     Tags (global tags first with empty Targets, then one Tag per track; per-track DURATION when seekable)
//     Clusters (whole clusters buffered, so every size field is the shortest one)
//     Cues (seekable output only)
// Packet scheduling:
//   - Packets are interleaved by timestamp across tracks (ties go to the lower track number), so callers may hand them in
//     any per-track order; a packet leaves the interleaver once every track has one waiting (or at Finish).
//   - Audio packets wait one step before they are written, so that when the next packet opens a new cluster (a video key
//     frame, or the time limit) the waiting audio packet starts that cluster.
//   - A new cluster starts when: the cluster holds 5 MiB (32 KiB when the output cannot seek); the next packet lies more
//     than 5 s (1 s) after the cluster timestamp; or the next packet is a video key frame and the cluster holds 4 KiB.
//   - Opus (any track with initial padding) blocks are shifted later by the CodecDelay, rounded to the millisecond.
//   - SimpleBlock unless the packet carries BlockAdditional data (alpha), DiscardPadding, or a duration other than the
//     track's DefaultDuration; then a BlockGroup (Block, BlockAdditions, BlockDuration, ReferenceBlock, DiscardPadding).
//   - Cues: one per video key frame; without video tracks, one per track per cluster (its first block).

using System.Buffers;
using System.Text;

namespace SpawnDev.Codecs.Container.Matroska;

/// <summary>Segment-level options for <see cref="WebMWriter"/>.</summary>
public sealed record WebMWriterOptions
{
    /// <summary>Info/MuxingApp.</summary>
    public string MuxingApp { get; init; } = "SpawnDev.Codecs";
    /// <summary>Info/WritingApp.</summary>
    public string WritingApp { get; init; } = "SpawnDev.Codecs";
    /// <summary>Info/Title, omitted when null.</summary>
    public string? Title { get; init; }
    /// <summary>Global tags (a Tag with empty Targets), written in order as SimpleTag name / string pairs.</summary>
    public IReadOnlyList<KeyValuePair<string, string>>? Tags { get; init; }
    /// <summary>Cluster duration limit in ms. Null: 5000 for seekable output, 1000 otherwise.</summary>
    public int? ClusterTimeLimitMs { get; init; }
    /// <summary>Cluster size limit in bytes. Null: 5 MiB for seekable output, 32 KiB otherwise.</summary>
    public int? ClusterSizeLimit { get; init; }
    /// <summary>
    /// EBML DocType: "webm" (default) or "matroska" (a .mkv file: the same element layout, which every Matroska reader
    /// accepts; not byte-identical with ffmpeg's matroska muxer, which writes some extra elements).
    /// </summary>
    public string DocType { get; init; } = "webm";
}

/// <summary>Fields shared by WebM video and audio tracks.</summary>
public abstract record WebMTrack
{
    /// <summary>Matroska CodecID, e.g. "V_VP9", "V_AV1", "A_OPUS".</summary>
    public required string CodecId { get; init; }
    /// <summary>CodecPrivate bytes (av1C, OpusHead, ...), omitted when null or empty.</summary>
    public byte[]? CodecPrivate { get; init; }
    /// <summary>
    /// When greater than zero, room for a CodecPrivate element of this many data bytes is reserved at the end of the
    /// TrackEntry; the part the actual CodecPrivate does not use is filled with a Void element.
    /// </summary>
    public int CodecPrivateReserve { get; init; }
    /// <summary>TrackUID (written as 8 bytes). 0 picks the track number.</summary>
    public ulong Uid { get; init; }
    /// <summary>Track Name, omitted when null.</summary>
    public string? Name { get; init; }
    /// <summary>Track Language; null writes "und".</summary>
    public string? Language { get; init; }
    /// <summary>FlagDefault (only written when false, its default being 1).</summary>
    public bool IsDefault { get; init; } = true;
    /// <summary>FlagForced (only written when true).</summary>
    public bool IsForced { get; init; }
    /// <summary>DefaultDuration in nanoseconds, omitted when null.</summary>
    public long? DefaultDurationNs { get; init; }
    /// <summary>Track tags, written in order (a DURATION tag is appended for seekable output).</summary>
    public IReadOnlyList<KeyValuePair<string, string>>? Tags { get; init; }
}

/// <summary>Colour element fields (values as in ISO/IEC 23091-4 / ITU-T H.273); null fields are omitted.</summary>
public sealed record WebMColour
{
    /// <summary>MatrixCoefficients.</summary>
    public int? MatrixCoefficients { get; init; }
    /// <summary>Range (1 = broadcast / limited, 2 = full).</summary>
    public int? Range { get; init; }
    /// <summary>TransferCharacteristics.</summary>
    public int? TransferCharacteristics { get; init; }
    /// <summary>Primaries.</summary>
    public int? Primaries { get; init; }
    /// <summary>ChromaSitingHorz (1 = left collocated, 2 = half).</summary>
    public int? ChromaSitingHorz { get; init; }
    /// <summary>ChromaSitingVert (1 = top collocated, 2 = half).</summary>
    public int? ChromaSitingVert { get; init; }
}

/// <summary>A WebM video track (TrackType 1).</summary>
public sealed record WebMVideoTrack : WebMTrack
{
    /// <summary>PixelWidth.</summary>
    public required int Width { get; init; }
    /// <summary>PixelHeight.</summary>
    public required int Height { get; init; }
    /// <summary>FlagInterlaced: true writes 1 (interlaced), false writes 2 (progressive), null omits it.</summary>
    public bool? Interlaced { get; init; }
    /// <summary>AlphaMode 1: alpha is carried in BlockAdditional (BlockAddID 1).</summary>
    public bool AlphaMode { get; init; }
    /// <summary>DisplayWidth / DisplayHeight, omitted when null.</summary>
    public (int Width, int Height)? DisplaySize { get; init; }
    /// <summary>Colour element, omitted when null or empty.</summary>
    public WebMColour? Colour { get; init; }
}

/// <summary>A WebM audio track (TrackType 2).</summary>
public sealed record WebMAudioTrack : WebMTrack
{
    /// <summary>SamplingFrequency (48000 for Opus whatever the source rate was).</summary>
    public required double SampleRate { get; init; }
    /// <summary>Channels.</summary>
    public required int Channels { get; init; }
    /// <summary>BitDepth, omitted when null.</summary>
    public int? BitDepth { get; init; }
    /// <summary>
    /// Samples (at <see cref="SampleRate"/>) the decoder discards at the start (Opus pre-skip). Written as CodecDelay and
    /// applied to block timestamps (shifted later by the delay rounded to the millisecond).
    /// </summary>
    public int InitialPaddingSamples { get; init; }
}

/// <summary>
/// Writes a WebM file from already-encoded packets. Add every track, then hand packets to <see cref="WriteFrame"/>
/// (timestamps in milliseconds), then call <see cref="Finish"/>. Seekable streams get Duration, per-track DURATION
/// tags, Cues and a SeekHead; non-seekable streams get a live layout (unknown Segment size, 1 s clusters).
/// </summary>
public sealed class WebMWriter : IDisposable
{
    private const int SeekHeadRoom = 161;
    private const int KeyFrameClusterThreshold = 4096;
    private const int DurationTagLength = 19;

    private sealed class TrackState
    {
        public required WebMTrack Track;
        public required int Number;
        public required bool IsVideo;
        public ulong Uid;
        public long DelayMs;
        public long DefaultMs = -1;
        public long MaxEndMs;
        public long LastTs;
        public bool HasLast;
        public int DurationTagOffset = -1;   // segment-relative offset of the DURATION TagString data
        public readonly Queue<Packet> Queue = new();
    }

    private sealed class Packet
    {
        public required TrackState Track;
        public long InputTs, FileTs, Duration, Discard;
        public bool Key;
        public byte[] Data = Array.Empty<byte>();
        public int Length;
        public byte[]? Add;
        public int AddLength;
    }

    private readonly record struct Cue(long Time, int Track, long ClusterPos, int RelativePos);

    private readonly Stream _stream;
    private readonly WebMWriterOptions _options;
    private readonly bool _leaveOpen;
    private readonly bool _seekable;
    private readonly long _start;
    private readonly int _timeLimit;
    private readonly int _sizeLimit;
    private readonly List<TrackState> _tracks = new();
    private bool _hasVideo;

    private bool _headerWritten, _finished, _disposed;
    private int _docTypeVersionOffset;   // file-relative (from _start)
    private long _segmentSizeOffset;     // file-relative
    private long _segmentDataStart;      // file-relative
    private long _segPos;                // bytes written after the Segment header
    private int _infoPos, _tracksPos, _tagsPos = -1, _durationOffset = -1;
    private bool _docTypeIsV4, _wroteDiscardPadding;

    private Packet? _held;
    private readonly EbmlBuffer _cluster = new(64 * 1024);
    private bool _clusterOpen;
    private long _clusterTs;
    private readonly List<Cue> _cues = new();
    private readonly List<Cue> _clusterCues = new();
    private readonly HashSet<int> _cuedThisCluster = new();
    private readonly EbmlBuffer _scratch = new(64);

    /// <summary>Create a writer over <paramref name="stream"/>. Seekability decides the layout (see the class summary).</summary>
    public WebMWriter(Stream stream, WebMWriterOptions? options = null, bool leaveOpen = false)
    {
        _stream = stream ?? throw new ArgumentNullException(nameof(stream));
        if (!stream.CanWrite) throw new ArgumentException("Stream is not writable.", nameof(stream));
        _options = options ?? new WebMWriterOptions();
        _leaveOpen = leaveOpen;
        _seekable = stream.CanSeek;
        _start = _seekable ? stream.Position : 0;
        _timeLimit = _options.ClusterTimeLimitMs ?? (_seekable ? 5000 : 1000);
        _sizeLimit = _options.ClusterSizeLimit ?? (_seekable ? 5 * 1024 * 1024 : 32 * 1024);
        if (_timeLimit <= 0) throw new ArgumentOutOfRangeException(nameof(options), "ClusterTimeLimitMs must be positive.");
        if (_sizeLimit <= 0) throw new ArgumentOutOfRangeException(nameof(options), "ClusterSizeLimit must be positive.");
        if (_options.DocType is not ("webm" or "matroska")) throw new ArgumentException("DocType must be \"webm\" or \"matroska\".", nameof(options));
    }

    /// <summary>Add a video track. Returns its track number (1-based).</summary>
    public int AddVideoTrack(WebMVideoTrack track)
    {
        ArgumentNullException.ThrowIfNull(track);
        if (track.Width <= 0 || track.Height <= 0) throw new ArgumentOutOfRangeException(nameof(track), "Width and Height must be positive.");
        return AddTrack(track, isVideo: true);
    }

    /// <summary>Add an audio track. Returns its track number (1-based).</summary>
    public int AddAudioTrack(WebMAudioTrack track)
    {
        ArgumentNullException.ThrowIfNull(track);
        if (track.Channels <= 0) throw new ArgumentOutOfRangeException(nameof(track), "Channels must be positive.");
        if (!(track.SampleRate > 0)) throw new ArgumentOutOfRangeException(nameof(track), "SampleRate must be positive.");
        if (track.InitialPaddingSamples < 0) throw new ArgumentOutOfRangeException(nameof(track), "InitialPaddingSamples must not be negative.");
        return AddTrack(track, isVideo: false);
    }

    private int AddTrack(WebMTrack track, bool isVideo)
    {
        ThrowIfDone();
        if (_headerWritten) throw new InvalidOperationException("Tracks must be added before the first frame is written.");
        if (string.IsNullOrEmpty(track.CodecId)) throw new ArgumentException("CodecId is required.", nameof(track));
        if (track.CodecPrivateReserve < 0) throw new ArgumentOutOfRangeException(nameof(track), "CodecPrivateReserve must not be negative.");
        if (_tracks.Count >= 126) throw new InvalidOperationException("At most 126 tracks are supported.");
        int number = _tracks.Count + 1;
        var st = new TrackState { Track = track, Number = number, IsVideo = isVideo, Uid = track.Uid != 0 ? track.Uid : (ulong)number };
        if (track.DefaultDurationNs is long dd && dd > 0) st.DefaultMs = (dd + 500_000) / 1_000_000;
        if (track is WebMAudioTrack a && a.InitialPaddingSamples > 0)
        {
            long rate = (long)a.SampleRate;
            st.DelayMs = (a.InitialPaddingSamples * 1000L * 2 + rate) / (2 * rate);
        }
        _hasVideo |= isVideo;
        _tracks.Add(st);
        return number;
    }

    /// <summary>
    /// Queue one encoded packet. <paramref name="timestampMs"/> is the presentation time in milliseconds (before any
    /// codec delay); <paramref name="durationMs"/> is the packet duration (0 = unknown);
    /// <paramref name="blockAdditional"/> carries BlockAddID 1 data (VP8 / VP9 alpha); <paramref name="discardPaddingNs"/>
    /// is written as DiscardPadding when non-zero (Opus end trimming).
    /// </summary>
    public void WriteFrame(int trackNumber, ReadOnlySpan<byte> data, long timestampMs, bool keyFrame, long durationMs = 0,
        ReadOnlySpan<byte> blockAdditional = default, long discardPaddingNs = 0)
    {
        ThrowIfDone();
        if (trackNumber < 1 || trackNumber > _tracks.Count) throw new ArgumentOutOfRangeException(nameof(trackNumber));
        if (durationMs < 0) throw new ArgumentOutOfRangeException(nameof(durationMs));
        if (!_headerWritten) WriteHeader();
        var t = _tracks[trackNumber - 1];
        var p = new Packet
        {
            Track = t, InputTs = timestampMs, FileTs = timestampMs + t.DelayMs, Duration = durationMs, Discard = discardPaddingNs,
            Key = keyFrame, Length = data.Length,
        };
        if (data.Length > 0)
        {
            p.Data = ArrayPool<byte>.Shared.Rent(data.Length);
            data.CopyTo(p.Data);
        }
        if (!blockAdditional.IsEmpty)
        {
            p.Add = ArrayPool<byte>.Shared.Rent(blockAdditional.Length);
            blockAdditional.CopyTo(p.Add);
            p.AddLength = blockAdditional.Length;
        }
        t.Queue.Enqueue(p);
        Interleave(flushAll: false);
    }

    /// <summary>Write everything still queued, the Cues and (seekable output) the header fields known only at the end.</summary>
    public void Finish()
    {
        if (_finished) return;
        ThrowIfDisposed();
        if (!_headerWritten) WriteHeader();
        Interleave(flushAll: true);
        if (_held is { } h) { _held = null; WriteBlock(h); }
        CloseCluster();
        _finished = true;
        if (_seekable) FinishSeekable();
        _stream.Flush();
    }

    /// <summary>Finishes the file if <see cref="Finish"/> was not called, then closes the stream unless leaveOpen.</summary>
    public void Dispose()
    {
        if (_disposed) return;
        try
        {
            if (!_finished) Finish();
        }
        finally
        {
            _disposed = true;
            if (!_leaveOpen) _stream.Dispose();
        }
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, this);

    private void ThrowIfDone()
    {
        ThrowIfDisposed();
        if (_finished) throw new InvalidOperationException("The writer is finished.");
    }

    // ---- interleaving

    private void Interleave(bool flushAll)
    {
        while (true)
        {
            TrackState? best = null;
            foreach (var t in _tracks)
            {
                if (t.Queue.Count == 0)
                {
                    if (!flushAll) return;   // wait until every track has a packet
                    continue;
                }
                if (best is null || t.Queue.Peek().InputTs < best.Queue.Peek().InputTs) best = t;
            }
            if (best is null) return;
            Schedule(best.Queue.Dequeue());
        }
    }

    // ---- cluster scheduling

    private void Schedule(Packet p)
    {
        if (_clusterOpen)
        {
            bool split = _cluster.Length >= _sizeLimit
                || p.FileTs - _clusterTs > _timeLimit
                || (p.Track.IsVideo && p.Key && _cluster.Length >= KeyFrameClusterThreshold);
            if (split) CloseCluster();
        }
        if (_held is { } h) { _held = null; WriteBlock(h); }
        if (p.Track.IsVideo) WriteBlock(p);
        else _held = p;
    }

    private void OpenCluster(long ts)
    {
        _cluster.Clear();
        _cluster.WriteUInt(MatroskaIds.Timestamp, (ulong)Math.Max(0, ts));
        _clusterTs = Math.Max(0, ts);
        _clusterOpen = true;
        _cuedThisCluster.Clear();
    }

    private void CloseCluster()
    {
        if (!_clusterOpen) return;
        long clusterPos = _segPos;
        _scratch.Clear();
        _scratch.WriteHeader(MatroskaIds.Cluster, _cluster.Length);
        Emit(_scratch.Span);
        Emit(_cluster.Span);
        foreach (var c in _clusterCues) _cues.Add(c with { ClusterPos = clusterPos });
        _clusterCues.Clear();
        _clusterOpen = false;
    }

    private void WriteBlock(Packet p)
    {
        try
        {
            if (_clusterOpen && (p.FileTs - _clusterTs < short.MinValue || p.FileTs - _clusterTs > short.MaxValue)) CloseCluster();
            if (!_clusterOpen) OpenCluster(p.FileTs);
            var t = p.Track;
            long rel = p.FileTs - _clusterTs;
            if (rel < short.MinValue || rel > short.MaxValue)
                throw new InvalidOperationException($"Track {t.Number}: timestamp {p.FileTs} ms cannot be stored in a cluster at {_clusterTs} ms.");
            int blockPos = _cluster.Length;
            bool durationDiffers = p.Duration > 0 && (t.DefaultMs < 0 || p.Duration != t.DefaultMs);
            bool group = p.AddLength > 0 || p.Discard != 0 || (t.DefaultMs >= 0 && durationDiffers);
            int blockData = 1 + 2 + 1 + p.Length;   // track number VINT (1 byte for 1..126), timestamp, flags, payload
            if (!group)
            {
                _cluster.WriteHeader(MatroskaIds.SimpleBlock, blockData);
                WriteBlockHead(t.Number, (short)rel, p.Key ? (byte)0x80 : (byte)0x00);
                _cluster.WriteBytes(p.Data.AsSpan(0, p.Length));
            }
            else
            {
                var g = _scratch;
                g.Clear();
                g.WriteHeader(MatroskaIds.Block, blockData);
                g.WriteByte((byte)(0x80 | t.Number));
                g.WriteByte((byte)(rel >> 8));
                g.WriteByte((byte)rel);
                g.WriteByte(0x00);
                g.WriteBytes(p.Data.AsSpan(0, p.Length));
                if (p.AddLength > 0)
                {
                    int moreLen = EbmlBuffer.ElementLength(MatroskaIds.BlockAddId, 1) + EbmlBuffer.ElementLength(MatroskaIds.BlockAdditional, p.AddLength);
                    g.WriteHeader(MatroskaIds.BlockAdditions, EbmlBuffer.ElementLength(MatroskaIds.BlockMore, moreLen));
                    g.WriteHeader(MatroskaIds.BlockMore, moreLen);
                    g.WriteUInt(MatroskaIds.BlockAddId, 1);
                    g.WriteBinary(MatroskaIds.BlockAdditional, p.Add.AsSpan(0, p.AddLength));
                }
                if (durationDiffers) g.WriteUInt(MatroskaIds.BlockDuration, (ulong)p.Duration);
                if (!p.Key && t.HasLast) g.WriteInt(MatroskaIds.ReferenceBlock, t.LastTs - p.FileTs);
                if (p.Discard != 0)
                {
                    g.WriteInt(MatroskaIds.DiscardPadding, p.Discard);
                    _wroteDiscardPadding = true;
                }
                _cluster.WriteMaster(MatroskaIds.BlockGroup, g);
            }

            if (_seekable)
            {
                bool cue = _hasVideo ? (t.IsVideo && p.Key) : _cuedThisCluster.Add(t.Number);
                if (cue) _clusterCues.Add(new Cue(p.FileTs, t.Number, 0, blockPos));
            }
            t.LastTs = p.FileTs;
            t.HasLast = true;
            t.MaxEndMs = Math.Max(t.MaxEndMs, p.FileTs + p.Duration);
        }
        finally
        {
            if (p.Length > 0) ArrayPool<byte>.Shared.Return(p.Data);
            if (p.Add is not null) ArrayPool<byte>.Shared.Return(p.Add);
        }
    }

    private void WriteBlockHead(int track, short rel, byte flags)
    {
        _cluster.WriteByte((byte)(0x80 | track));
        _cluster.WriteByte((byte)(rel >> 8));
        _cluster.WriteByte((byte)rel);
        _cluster.WriteByte(flags);
    }

    // ---- output

    private void Emit(ReadOnlySpan<byte> bytes)
    {
        _stream.Write(bytes);
        _segPos += bytes.Length;
    }

    private void WriteHeader()
    {
        _headerWritten = true;
        var file = new EbmlBuffer(1024);

        _docTypeIsV4 = _tracks.Any(t => t.Track is WebMAudioTrack a && (a.InitialPaddingSamples > 0 || IsOpus(a)));
        var ebml = new EbmlBuffer(64);
        ebml.WriteUInt(MatroskaIds.EbmlVersion, 1);
        ebml.WriteUInt(MatroskaIds.EbmlReadVersion, 1);
        ebml.WriteUInt(MatroskaIds.EbmlMaxIdLength, 4);
        ebml.WriteUInt(MatroskaIds.EbmlMaxSizeLength, 8);
        ebml.WriteString(MatroskaIds.DocType, _options.DocType);
        ebml.WriteUInt(MatroskaIds.DocTypeVersion, _docTypeIsV4 ? 4UL : 2UL);
        _docTypeVersionOffset = ebml.Length - 1;
        ebml.WriteUInt(MatroskaIds.DocTypeReadVersion, 2);
        file.WriteMaster(MatroskaIds.Ebml, ebml);
        _docTypeVersionOffset += file.Length - ebml.Length;

        file.WriteId(MatroskaIds.Segment);
        _segmentSizeOffset = file.Length;
        if (_seekable) file.WriteSize(0, 8);
        else file.WriteUnknownSize();
        _segmentDataStart = file.Length;
        int segBase = file.Length;

        // Info, Tracks and Tags are built first so the SeekHead (live output) can point at them.
        var info = new EbmlBuffer(128);
        info.WriteUInt(MatroskaIds.TimestampScale, 1_000_000);
        if (_options.Title is { } title) info.WriteString(MatroskaIds.Title, title);
        info.WriteString(MatroskaIds.MuxingApp, _options.MuxingApp);
        info.WriteString(MatroskaIds.WritingApp, _options.WritingApp);
        int durationInInfo = _seekable ? info.WriteDouble(MatroskaIds.Duration, 0) : -1;

        var tracks = new EbmlBuffer(512);
        foreach (var t in _tracks) WriteTrackEntry(tracks, t);

        var tags = new EbmlBuffer(256);
        var tagDurationOffsets = new List<(TrackState Track, int Offset)>();
        if (_options.Tags is { Count: > 0 } global)
        {
            var tag = new EbmlBuffer(128);
            tag.WriteHeader(MatroskaIds.Targets, 0);
            foreach (var kv in global) WriteSimpleTag(tag, kv.Key, kv.Value);
            tags.WriteMaster(MatroskaIds.Tag, tag);
        }
        foreach (var t in _tracks)
        {
            bool any = t.Track.Tags is { Count: > 0 };
            if (!any && !_seekable) continue;
            var tag = new EbmlBuffer(128);
            var targets = new EbmlBuffer(16);
            targets.WriteUIntFixed(MatroskaIds.TagTrackUid, t.Uid, 8);
            tag.WriteMaster(MatroskaIds.Targets, targets);
            if (t.Track.Tags is not null)
                foreach (var kv in t.Track.Tags) WriteSimpleTag(tag, kv.Key, kv.Value);
            int durationAt = -1;
            if (_seekable)
            {
                var st = new EbmlBuffer(48);
                st.WriteString(MatroskaIds.TagName, "DURATION");
                st.WriteHeader(MatroskaIds.TagString, DurationTagLength);
                int inSimple = st.Length;
                st.WriteZeros(DurationTagLength);
                tag.WriteHeader(MatroskaIds.SimpleTag, st.Length);
                durationAt = tag.Length + inSimple;
                tag.WriteBytes(st.Span);
            }
            int tagHeader = EbmlBuffer.IdLength(MatroskaIds.Tag) + EbmlBuffer.SizeLength((ulong)tag.Length);
            if (durationAt >= 0) tagDurationOffsets.Add((t, tags.Length + tagHeader + durationAt));
            tags.WriteMaster(MatroskaIds.Tag, tag);
        }

        _infoPos = SeekHeadRoom;
        int infoHeader = EbmlBuffer.IdLength(MatroskaIds.Info) + EbmlBuffer.SizeLength((ulong)info.Length);
        if (durationInInfo >= 0) _durationOffset = _infoPos + infoHeader + durationInInfo;
        _tracksPos = _infoPos + infoHeader + info.Length;
        int tracksLen = EbmlBuffer.ElementLength(MatroskaIds.Tracks, tracks.Length);
        if (tags.Length > 0)
        {
            _tagsPos = _tracksPos + tracksLen;
            int tagsHeader = EbmlBuffer.IdLength(MatroskaIds.Tags) + EbmlBuffer.SizeLength((ulong)tags.Length);
            foreach (var (t, off) in tagDurationOffsets) t.DurationTagOffset = _tagsPos + tagsHeader + off;
        }

        if (_seekable) file.WriteVoid(SeekHeadRoom);   // replaced by the SeekHead at Finish
        else WriteSeekHead(file, cuesPos: -1);
        if (file.Length - segBase != SeekHeadRoom) throw new InvalidOperationException("SeekHead room mismatch.");
        file.WriteMaster(MatroskaIds.Info, info);
        file.WriteMaster(MatroskaIds.Tracks, tracks);
        if (tags.Length > 0) file.WriteMaster(MatroskaIds.Tags, tags);

        _stream.Write(file.Span);
        _segPos = file.Length - segBase;
    }

    private static bool IsOpus(WebMTrack t) => t.CodecId == "A_OPUS";

    /// <summary>Writes the SeekHead plus the Void that fills the rest of the reserved room.</summary>
    private void WriteSeekHead(EbmlBuffer dst, long cuesPos)
    {
        var sh = new EbmlBuffer(96);
        void Entry(uint id, long pos)
        {
            var seek = new EbmlBuffer(16);
            Span<byte> idBytes = stackalloc byte[4];
            int n = EbmlBuffer.IdLength(id);
            for (int i = 0; i < n; i++) idBytes[i] = (byte)(id >> (8 * (n - 1 - i)));
            seek.WriteBinary(MatroskaIds.SeekId, idBytes[..n]);
            seek.WriteUInt(MatroskaIds.SeekPosition, (ulong)pos);
            sh.WriteMaster(MatroskaIds.Seek, seek);
        }
        Entry(MatroskaIds.Info, _infoPos);
        Entry(MatroskaIds.Tracks, _tracksPos);
        if (_tagsPos >= 0) Entry(MatroskaIds.Tags, _tagsPos);
        if (cuesPos >= 0) Entry(MatroskaIds.Cues, cuesPos);
        int start = dst.Length;
        dst.WriteMaster(MatroskaIds.SeekHead, sh);
        int rest = SeekHeadRoom - (dst.Length - start);
        if (rest < 2) throw new InvalidOperationException("SeekHead does not fit its reserved room.");
        dst.WriteVoid(rest);
    }

    private static void WriteSimpleTag(EbmlBuffer tag, string name, string value)
    {
        var st = new EbmlBuffer(64);
        st.WriteString(MatroskaIds.TagName, name);
        st.WriteString(MatroskaIds.TagString, value);
        tag.WriteMaster(MatroskaIds.SimpleTag, st);
    }

    private void WriteTrackEntry(EbmlBuffer tracks, TrackState t)
    {
        var tr = t.Track;
        var e = new EbmlBuffer(128);
        e.WriteUInt(MatroskaIds.TrackNumber, (ulong)t.Number);
        e.WriteUIntFixed(MatroskaIds.TrackUid, t.Uid, 8);
        e.WriteUInt(MatroskaIds.FlagLacing, 0);
        if (tr.Name is { } name) e.WriteString(MatroskaIds.Name, name);
        e.WriteString(MatroskaIds.Language, tr.Language ?? "und");
        if (!tr.IsDefault) e.WriteUInt(MatroskaIds.FlagDefault, 0);
        if (tr.IsForced) e.WriteUInt(MatroskaIds.FlagForced, 1);
        e.WriteString(MatroskaIds.CodecId, tr.CodecId);
        if (tr is WebMAudioTrack a)
        {
            if (a.InitialPaddingSamples > 0)
            {
                long rate = (long)a.SampleRate;
                long ns = (a.InitialPaddingSamples * 1_000_000_000L * 2 + rate) / (2 * rate);
                e.WriteUInt(MatroskaIds.CodecDelay, (ulong)ns);
            }
            if (IsOpus(a)) e.WriteUInt(MatroskaIds.SeekPreRoll, 80_000_000);   // RFC 7845 / WebM: 80 ms Opus pre-roll
        }
        e.WriteUInt(MatroskaIds.TrackType, t.IsVideo ? 1UL : 2UL);
        if (tr.DefaultDurationNs is long dd && dd > 0) e.WriteUInt(MatroskaIds.DefaultDuration, (ulong)dd);
        if (tr is WebMVideoTrack v)
        {
            var vb = new EbmlBuffer(64);
            vb.WriteUInt(MatroskaIds.PixelWidth, (ulong)v.Width);
            vb.WriteUInt(MatroskaIds.PixelHeight, (ulong)v.Height);
            if (v.Interlaced is bool il) vb.WriteUInt(MatroskaIds.FlagInterlaced, il ? 1UL : 2UL);
            if (v.AlphaMode) vb.WriteUInt(MatroskaIds.AlphaMode, 1);
            if (v.DisplaySize is var (dw, dh))
            {
                vb.WriteUInt(MatroskaIds.DisplayWidth, (ulong)dw);
                vb.WriteUInt(MatroskaIds.DisplayHeight, (ulong)dh);
            }
            if (v.Colour is { } c)
            {
                var cb = new EbmlBuffer(32);
                void Put(uint id, int? val) { if (val is int x) cb.WriteUInt(id, (ulong)x); }
                Put(MatroskaIds.TransferCharacteristics, c.TransferCharacteristics);
                Put(MatroskaIds.MatrixCoefficients, c.MatrixCoefficients);
                Put(MatroskaIds.Primaries, c.Primaries);
                Put(MatroskaIds.Range, c.Range);
                Put(MatroskaIds.ChromaSitingHorz, c.ChromaSitingHorz);
                Put(MatroskaIds.ChromaSitingVert, c.ChromaSitingVert);
                if (cb.Length > 0) vb.WriteMaster(MatroskaIds.Colour, cb);
            }
            e.WriteMaster(MatroskaIds.Video, vb);
        }
        else if (tr is WebMAudioTrack au)
        {
            var ab = new EbmlBuffer(32);
            ab.WriteUInt(MatroskaIds.Channels, (ulong)au.Channels);
            ab.WriteDouble(MatroskaIds.SamplingFrequency, au.SampleRate);
            if (au.BitDepth is int bd) ab.WriteUInt(MatroskaIds.BitDepth, (ulong)bd);
            e.WriteMaster(MatroskaIds.Audio, ab);
        }
        WriteCodecPrivate(e, tr);
        tracks.WriteMaster(MatroskaIds.TrackEntry, e, sizeWidth: 8);
    }

    private static void WriteCodecPrivate(EbmlBuffer e, WebMTrack tr)
    {
        var cp = tr.CodecPrivate is { Length: > 0 } x ? x : null;
        if (tr.CodecPrivateReserve <= 0)
        {
            if (cp is not null) e.WriteBinary(MatroskaIds.CodecPrivate, cp);
            return;
        }
        int room = EbmlBuffer.ElementLength(MatroskaIds.CodecPrivate, tr.CodecPrivateReserve);
        if (cp is null)
        {
            e.WriteVoid(room);
            return;
        }
        int used = EbmlBuffer.ElementLength(MatroskaIds.CodecPrivate, cp.Length);
        if (used > room)
            throw new ArgumentException($"CodecPrivate ({cp.Length} bytes) does not fit CodecPrivateReserve ({tr.CodecPrivateReserve}).");
        int rest = room - used;
        if (rest == 1)
        {
            // A Void cannot be 1 byte: widen the CodecPrivate size field by one byte instead.
            e.WriteHeader(MatroskaIds.CodecPrivate, cp.Length, EbmlBuffer.SizeLength((ulong)cp.Length) + 1);
            e.WriteBytes(cp);
            return;
        }
        e.WriteBinary(MatroskaIds.CodecPrivate, cp);
        if (rest > 0) e.WriteVoid(rest);
    }

    // ---- end of a seekable file

    private void FinishSeekable()
    {
        long cuesPos = -1;
        if (_cues.Count > 0)
        {
            cuesPos = _segPos;
            var cues = new EbmlBuffer(64 + _cues.Count * 20);
            int i = 0;
            while (i < _cues.Count)
            {
                long time = _cues[i].Time;
                var point = new EbmlBuffer(32);
                point.WriteUInt(MatroskaIds.CueTime, (ulong)Math.Max(0, time));
                for (; i < _cues.Count && _cues[i].Time == time; i++)
                {
                    var pos = new EbmlBuffer(24);
                    pos.WriteUInt(MatroskaIds.CueTrack, (ulong)_cues[i].Track);
                    pos.WriteUInt(MatroskaIds.CueClusterPosition, (ulong)_cues[i].ClusterPos);
                    pos.WriteUInt(MatroskaIds.CueRelativePosition, (ulong)_cues[i].RelativePos);
                    point.WriteMaster(MatroskaIds.CueTrackPositions, pos);
                }
                cues.WriteMaster(MatroskaIds.CuePoint, point);
            }
            var whole = new EbmlBuffer(cues.Length + 16);
            whole.WriteMaster(MatroskaIds.Cues, cues);
            Emit(whole.Span);
        }
        long end = _start + _segmentDataStart + _segPos;

        Span<byte> b8 = stackalloc byte[8];
        EbmlBuffer.WriteSizeTo(b8, (ulong)_segPos, 8);
        Patch(_segmentSizeOffset, b8);

        var sh = new EbmlBuffer(SeekHeadRoom);
        WriteSeekHead(sh, cuesPos);
        Patch(_segmentDataStart, sh.Span);

        long durationMs = 0;
        foreach (var t in _tracks) durationMs = Math.Max(durationMs, t.MaxEndMs);
        if (_durationOffset >= 0)
        {
            System.Buffers.Binary.BinaryPrimitives.WriteDoubleBigEndian(b8, durationMs);
            Patch(_segmentDataStart + _durationOffset, b8);
        }
        Span<byte> tagBytes = stackalloc byte[DurationTagLength];
        foreach (var t in _tracks)
        {
            if (t.DurationTagOffset < 0) continue;
            FormatDuration(t.MaxEndMs, tagBytes);
            Patch(_segmentDataStart + t.DurationTagOffset, tagBytes);
        }
        if (_wroteDiscardPadding && !_docTypeIsV4) Patch(_docTypeVersionOffset, stackalloc byte[] { 4 });   // DiscardPadding: Matroska v4
        _stream.Position = end;
    }

    private void Patch(long fileOffset, ReadOnlySpan<byte> bytes)
    {
        _stream.Position = _start + fileOffset;
        _stream.Write(bytes);
    }

    /// <summary>"HH:MM:SS.nnnnnnnnn" followed by zero bytes, 19 bytes in all.</summary>
    private static void FormatDuration(long ms, Span<byte> dst)
    {
        long h = ms / 3_600_000, m = ms / 60_000 % 60, s = ms / 1000 % 60, ns = ms % 1000 * 1_000_000;
        string text = $"{h:00}:{m:00}:{s:00}.{ns:000000000}";
        if (text.Length > dst.Length) throw new InvalidOperationException($"Duration {ms} ms is too long for the DURATION tag.");
        dst.Clear();
        Encoding.ASCII.GetBytes(text, dst);
    }
}
