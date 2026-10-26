// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Snappy decompressor on the accelerator for both container formats, reading any valid stream (every literal length
// form, copy-1 / copy-2 / copy-4, overlapping copies, offsets of any distance in a raw stream; chunks of any size,
// uncompressed / padding / skippable chunks and repeated stream identifiers in a framed stream). Every check fails with
// the reason: truncation, a bad length varint, a copy offset of 0 or before the output, a stream whose elements do not
// add up to its declared length, a missing or damaged stream identifier, a reserved chunk type, a chunk over 64 KB,
// a chunk's CRC-32C.
//
// The work is cut into SEGMENTS, each decoded by one thread, DecodeBudget output bytes per launch (a literal or copy
// resumes mid-run):
//   Framed: one segment per data chunk (each chunk is a self-contained raw stream, or stored bytes, of at most 64 KB).
//     A walk (one thread, WalkHops chunks per launch; run twice: count, then record) reads the chunk headers and each
//     compressed chunk's length varint, so it lays the output out as it goes. A scan (one thread per chunk) validates
//     the chunk's elements; after decoding, one thread per chunk checks its CRC-32C.
//   Raw: one stream, whose copies may reach anywhere back. A scan (one thread, ScanHops elements per launch) validates
//     every element and starts a new segment at the first element at or past each 64 KB of output, marking a segment
//     DEPENDENT when one of its copies reaches before the segment. Streams cut into 64 KB blocks (snap, the C++
//     library) have none. Independent segments decode one thread each; then one thread decodes the dependent ones in
//     order.
// The host reads back a handful of ints (counts, the total length, one error word) and nothing else.
//
// API shape: the same three overloads as the codecs (ArrayView1D / managed / browser TypedArray).

using System.Runtime.Versioning;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.SpawnJS.JSObjects;

namespace SpawnDev.Codecs.Compression.Snappy;

/// <summary>Snappy (raw or framed) decompressor on the accelerator.</summary>
public sealed class SnappyDecompressorGpu : IDisposable
{
    /// <summary>Chunk headers one framed walk launch reads.</summary>
    public const int WalkHops = 4096;
    /// <summary>Elements one raw scan launch reads.</summary>
    public const int ScanHops = 65536;
    /// <summary>Output bytes one decode launch writes per segment.</summary>
    public const int DecodeBudget = 65536;
    /// <summary>Output bytes after which a raw stream starts a new segment (at the next element).</summary>
    public const int SegmentBytes = 65536;
    /// <summary>The framing format's largest chunk (uncompressed bytes).</summary>
    public const int MaxChunk = 65536;

    // Segment table (ints per segment). GIp / GOp / GCopyLeft / GCopyOff / GLitLeft: where the decoder stopped.
    internal const int GSrc = 0, GEnd = 1, GDst = 2, GSize = 3, GFlags = 4, GCrc = 5, GIp = 6, GOp = 7, GCopyLeft = 8, GCopyOff = 9,
        GLitLeft = 10, GPhase = 11;
    internal const int SegInts = 12;
    internal const int FStored = 1, FDependent = 2, FCompressed = 4;

    // prm ints.
    internal const int PLength = 0, PPos = 1, PState = 2, PSegs = 3, PRecord = 4, PError = 5, PErrorAt = 6, PTotal = 7, POut = 8,
        PCapSegs = 9, PMaxSeg = 10, PAnyDependent = 11, PDependentNext = 12, PSegDst = 13, PNextMark = 14, PFormat = 15;
    internal const int PrmInts = 16;
    // Walk / scan states.
    internal const int WStart = 0, WChunks = 1, WDone = 2;

    /// <summary>Why a stream was refused (the error word the kernels write).</summary>
    public enum SnappyError
    {
        None = 0, Truncated = 1, BadVarint = 2, BadOffset = 3, LengthMismatch = 4, BadStreamIdentifier = 5, ReservedChunk = 6,
        ChunkTooLarge = 7, ChunkChecksum = 8, BadChunkLength = 9, TooLarge = 10, OutputTooSmall = 11,
    }

    /// <summary>Thrown for a stream the decoder refuses; <see cref="Reason"/> says why, <see cref="Offset"/> where (input byte, or chunk index for checksums).</summary>
    public sealed class SnappyFormatException : FormatException
    {
        public SnappyError Reason { get; }
        public long Offset { get; }
        public SnappyFormatException(SnappyError reason, long offset) : base($"Snappy: {reason} at {offset}.")
        {
            Reason = reason;
            Offset = offset;
        }
    }

    private readonly Accelerator _accelerator;
    private readonly LaunchPacer _pacer;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>> _rawHeader;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>> _rawScan;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>> _walk;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>> _chunkScan;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, int> _decode;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<uint>> _chunkCheck;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _prm;
    private MemoryBuffer1D<int, Stride1D.Dense>? _segs;
    // Stand-in table of the counting walk (never written; WebGPU refuses one buffer bound to two read_write slots).
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _noSegs;
    // Stand-in output of a stream of empty chunks.
    private readonly MemoryBuffer1D<byte, Stride1D.Dense> _noOut;
    private readonly MemoryBuffer1D<uint, Stride1D.Dense> _crcTable;
    private readonly StageBuffer<byte> _stageIn, _stageOut;

    /// <summary>Compiles the kernels.</summary>
    public SnappyDecompressorGpu(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _pacer = new LaunchPacer(accelerator);
        _rawHeader = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<int>>(RawHeaderKernel);
        _rawScan = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>>(RawScanKernel);
        _walk = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>>(WalkKernel);
        _chunkScan = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>>(ChunkScanKernel);
        _decode = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, int>(DecodeKernel);
        _chunkCheck = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<uint>>(ChunkCheckKernel);
        _prm = accelerator.Allocate1D<int>(PrmInts);
        _noSegs = accelerator.Allocate1D<int>(1);
        _noOut = accelerator.Allocate1D<byte>(1);
        _crcTable = accelerator.Allocate1D(Crc32Gpu.BuildTable(Crc32Gpu.CastagnoliPolynomial));
        _stageIn = new StageBuffer<byte>(accelerator);
        _stageOut = new StageBuffer<byte>(accelerator);
    }

    // ------------------------------------------------------------------
    // Overloads
    // ------------------------------------------------------------------

    /// <summary>
    /// Decompresses the <paramref name="format"/> stream in the first <paramref name="length"/> bytes of
    /// <paramref name="input"/> into <paramref name="output"/>; returns the decompressed length. Throws
    /// <see cref="SnappyFormatException"/> for a bad stream (including output too small).
    /// </summary>
    public async Task<int> DecompressAsync(ArrayView1D<byte, Stride1D.Dense> input, int length, ArrayView1D<byte, Stride1D.Dense> output, SnappyFormat format = SnappyFormat.Raw)
    {
        int total = await PrepareAsync(input, length, format);
        if (total > output.Length) throw new SnappyFormatException(SnappyError.OutputTooSmall, total);
        await DecodeAsync(input, length, output, total, format);
        return total;
    }

    /// <summary>The decompressed length of the stream in <paramref name="input"/> (validates its structure; no output written).</summary>
    public Task<int> GetDecompressedLengthAsync(ArrayView1D<byte, Stride1D.Dense> input, int length, SnappyFormat format = SnappyFormat.Raw) =>
        PrepareAsync(input, length, format);

    /// <summary>Decompresses <paramref name="stream"/> into a new array.</summary>
    public async Task<byte[]> DecompressAsync(ReadOnlyMemory<byte> stream, SnappyFormat format = SnappyFormat.Raw)
    {
        var src = _stageIn.Ensure(stream.Length);
        if (stream.Length > 0) DeviceTransfer.Upload(src, stream.Span);
        int total = await PrepareAsync(src, stream.Length, format);
        var dst = _stageOut.Ensure(total);
        await DecodeAsync(src, stream.Length, dst, total, format);
        return await DeviceTransfer.DownloadAsync(dst, total);
    }

    /// <summary>Decompresses the bytes of <paramref name="stream"/> into a new Uint8Array (never on the .NET heap); the caller disposes it.</summary>
    [SupportedOSPlatform("browser")]
    public async Task<Uint8Array> DecompressAsync(TypedArray stream, SnappyFormat format = SnappyFormat.Raw)
    {
        ArgumentNullException.ThrowIfNull(stream);
        if (stream.ByteLength > int.MaxValue) throw new ArgumentOutOfRangeException(nameof(stream), "At most 2^31 - 1 bytes per call.");
        int length = (int)stream.ByteLength;
        var src = _stageIn.Ensure(length);
        if (length > 0) DeviceTransfer.Upload(src, stream);
        int total = await PrepareAsync(src, length, format);
        var dst = _stageOut.Ensure(total);
        await DecodeAsync(src, length, dst, total, format);
        return await DeviceTransfer.DownloadUint8ArrayAsync(dst, total);
    }

    // ------------------------------------------------------------------
    // Host coordination
    // ------------------------------------------------------------------

    private int _segCount, _maxSeg;
    private bool _anyDependent;

    /// <summary>Validates the stream and lays its segments out. Returns the total decompressed length.</summary>
    private async Task<int> PrepareAsync(ArrayView1D<byte, Stride1D.Dense> input, int length, SnappyFormat format)
    {
        if (length < 0 || length > input.Length) throw new ArgumentOutOfRangeException(nameof(length), $"{length} bytes; the input holds {input.Length}.");
        if (format != SnappyFormat.Raw && format != SnappyFormat.Framed) throw new ArgumentOutOfRangeException(nameof(format));
        _segCount = 0;
        _maxSeg = 0;
        _anyDependent = false;
        if (length == 0)
        {
            // An empty framed stream holds nothing; a raw stream needs at least its length varint.
            if (format == SnappyFormat.Raw) throw new SnappyFormatException(SnappyError.Truncated, 0);
            return 0;
        }
        var data = input.SubView(0, length);
        int[] end = format == SnappyFormat.Raw ? await PrepareRawAsync(data, length) : await PrepareFramedAsync(data, length);
        _segCount = end[PSegs];
        _maxSeg = end[PMaxSeg];
        _anyDependent = end[PAnyDependent] != 0;
        return end[PTotal];
    }

    private int[] NewPrm(int length, SnappyFormat format)
    {
        var prm = new int[PrmInts];
        prm[PLength] = length;
        prm[PFormat] = (int)format;
        return prm;
    }

    private async Task<int[]> PrepareRawAsync(ArrayView<byte> data, int length)
    {
        _prm.View.CopyFromCPU(NewPrm(length, SnappyFormat.Raw));
        _rawHeader(1, data, _prm.View);
        await _accelerator.SynchronizeAsync();
        var head = await _prm.View.CopyToHostAsync();
        ThrowIfError(head);
        int total = head[PTotal];
        // A segment starts only at the first element at or past each SegmentBytes of output.
        EnsureSegments(total / SegmentBytes + 2);
        _prm.View.SubView(PCapSegs, 1).CopyFromCPU(new[] { total / SegmentBytes + 2 });
        // Every element reads at least 2 input bytes, so length / 2 / ScanHops + 3 launches always reach the end;
        // check the state every 64 launches so short inputs stop early.
        int maxLaunches = length / 2 / ScanHops + 3;
        for (int l = 0; l < maxLaunches; l++)
        {
            _rawScan(1, data, _segs!.View, _prm.View);
            await _pacer.LaunchedAsync();
            if ((l & 63) == 63)
            {
                await _accelerator.SynchronizeAsync();
                var st = await _prm.View.SubView(PState, PError - PState + 1).CopyToHostAsync();
                if (st[0] == WDone || st[PError - PState] != 0) break;
            }
        }
        await _accelerator.SynchronizeAsync();
        var end = await _prm.View.CopyToHostAsync();
        ThrowIfError(end);
        if (end[PState] != WDone) throw new InvalidOperationException("Snappy raw scan did not reach the end of the stream in its launch bound (bug).");
        return end;
    }

    private async Task<int[]> PrepareFramedAsync(ArrayView<byte> data, int length)
    {
        // Pass 1: count data chunks.
        await RunWalkAsync(data, length, record: false);
        var counts = await _prm.View.CopyToHostAsync();
        ThrowIfError(counts);
        int segs = counts[PSegs];
        EnsureSegments(Math.Max(1, segs));
        // Pass 2: record (and lay out).
        await RunWalkAsync(data, length, record: true, cap: segs);
        var walked = await _prm.View.CopyToHostAsync();
        ThrowIfError(walked);
        if (segs > 0)
        {
            _chunkScan(segs, data, _segs!.View, _prm.View);
            await _accelerator.SynchronizeAsync();
            var scanned = await _prm.View.CopyToHostAsync();
            ThrowIfError(scanned);
            return scanned;
        }
        return walked;
    }

    private async Task RunWalkAsync(ArrayView<byte> data, int length, bool record, int cap = 0)
    {
        var prm = NewPrm(length, SnappyFormat.Framed);
        prm[PRecord] = record ? 1 : 0;
        prm[PCapSegs] = cap;
        _prm.View.CopyFromCPU(prm);
        var segs = !record || _segs is null ? _noSegs.View : _segs.View;
        // Every chunk is at least 4 bytes, so length / 4 / WalkHops + 3 launches always reach the end.
        int maxLaunches = length / 4 / WalkHops + 3;
        for (int l = 0; l < maxLaunches; l++)
        {
            _walk(1, data, segs, _prm.View);
            await _pacer.LaunchedAsync();
            if ((l & 63) == 63)
            {
                await _accelerator.SynchronizeAsync();
                var st = await _prm.View.SubView(PState, 1).CopyToHostAsync();
                if (st[0] == WDone) break;
            }
        }
        await _accelerator.SynchronizeAsync();
        var end = await _prm.View.CopyToHostAsync();
        ThrowIfError(end);
        if (end[PState] != WDone) throw new InvalidOperationException("Snappy walk did not reach the end of the stream in its launch bound (bug).");
    }

    private void EnsureSegments(int segs)
    {
        if (_segs is null || _segs.Length < (long)segs * SegInts)
        {
            _segs?.Dispose();
            _segs = _accelerator.Allocate1D<int>((long)segs * SegInts);
        }
    }

    private async Task DecodeAsync(ArrayView1D<byte, Stride1D.Dense> input, int length, ArrayView1D<byte, Stride1D.Dense> output, int total, SnappyFormat format)
    {
        if (total > output.Length) throw new SnappyFormatException(SnappyError.OutputTooSmall, total);
        if (_segCount == 0) return;
        var data = input.SubView(0, length);
        // Empty chunks still carry a CRC to check; their (empty) output is the stand-in byte.
        var outView = total > 0 ? output.SubView(0, total) : _noOut.View;
        // Independent segments: one thread each. Dependent segments (raw only): one thread walks them in order (mode 1)
        // after the independent pass skipped them.
        int steps = _maxSeg / DecodeBudget + 2;
        for (int s = 0; s < steps; s++)
        {
            _decode(_segCount, data, outView, _segs!.View, _prm.View, 0);
            await _pacer.LaunchedAsync();
        }
        if (_anyDependent)
        {
            long dependentSteps = (long)total / DecodeBudget + 2L * _segCount + 2;
            for (long s = 0; s < dependentSteps; s++)
            {
                _decode(1, data, outView, _segs!.View, _prm.View, 1);
                await _pacer.LaunchedAsync();
            }
        }
        if (format == SnappyFormat.Framed) _chunkCheck(_segCount, outView, _segs!.View, _prm.View, _crcTable.View);
        await _accelerator.SynchronizeAsync();
        var end = await _prm.View.CopyToHostAsync();
        ThrowIfError(end);
        if (end[POut] != _segCount) throw new InvalidOperationException($"Snappy decoder: {_segCount - end[POut]} segments did not finish in their launch budget (bug).");
    }

    private static void ThrowIfError(int[] prm)
    {
        if (prm[PError] != 0) throw new SnappyFormatException((SnappyError)prm[PError], prm[PErrorAt]);
    }

    // ------------------------------------------------------------------
    // Kernels
    // ------------------------------------------------------------------

    private static uint Read32(ArrayView<byte> d, int at) =>
        (uint)d[at] | ((uint)d[at + 1] << 8) | ((uint)d[at + 2] << 16) | ((uint)d[at + 3] << 24);

    /// <summary>First error wins.</summary>
    private static void Fail(ArrayView<int> prm, SnappyError e, int at)
    {
        if (prm[PError] == 0)
        {
            prm[PError] = (int)e;
            prm[PErrorAt] = at;
        }
    }

    /// <summary>
    /// Reads a length varint (at most 5 bytes, at most 2^31 - 1) at <paramref name="at"/> before <paramref name="end"/>:
    /// returns its byte count, 0 when it runs past the end, -1 when it is longer than 5 bytes or 32 bits, -2 when its value
    /// is 2^31 or more.
    /// </summary>
    private static int ReadVarint(ArrayView<byte> d, int at, int end, ref int value)
    {
        uint v = 0;
        int n = 0;
        int result = 0;
        bool go = true;
        while (go)
        {
            if (at + n >= end) { result = 0; go = false; }
            else if (n == 5) { result = -1; go = false; }
            else
            {
                uint b = d[at + n];
                // The 5th byte holds bits 28..31 only.
                if ((n == 4) & (b > 0x0F)) { result = -1; go = false; }
                else
                {
                    v |= (b & 0x7F) << (7 * n);
                    n++;
                    if (b < 0x80)
                    {
                        result = n;
                        go = false;
                    }
                }
            }
        }
        if ((result > 0) & (v > 0x7FFFFFFFu)) result = -2;
        value = (int)v;
        return result;
    }

    /// <summary>
    /// The element at <paramref name="ip"/> (before <paramref name="end"/>): returns its header length, or 0 when the
    /// header or a literal's bytes run past the end. <paramref name="isCopy"/>, <paramref name="len"/> and
    /// <paramref name="off"/> (copies) describe it.
    /// </summary>
    private static int ReadElement(ArrayView<byte> d, int ip, int end, ref bool isCopy, ref int len, ref uint off)
    {
        int tag = d[ip];
        int kind = tag & 3;
        int hdr = 0;
        if (kind == 0)
        {
            int m = tag >> 2;
            isCopy = false;
            if (m < 60)
            {
                hdr = 1;
                len = m + 1;
            }
            else
            {
                int nb = m - 59;
                hdr = 0;
                if (ip + 1 + nb <= end)
                {
                    uint v = 0;
                    for (int k = 0; k < nb; k++) v |= (uint)d[ip + 1 + k] << (8 * k);
                    // m + 1 bytes must follow the header: v < available keeps len in int range.
                    uint avail = (uint)(end - ip - 1 - nb);
                    if (v < avail)
                    {
                        hdr = 1 + nb;
                        len = (int)v + 1;
                    }
                }
            }
            if ((hdr > 0) & (ip + hdr + len > end)) hdr = 0;
        }
        else
        {
            isCopy = true;
            if (kind == 1)
            {
                hdr = 2;
                len = ((tag >> 2) & 7) + 4;
                if (ip + 2 <= end) off = (uint)(((tag >> 5) << 8) | d[ip + 1]);
            }
            else if (kind == 2)
            {
                hdr = 3;
                len = (tag >> 2) + 1;
                if (ip + 3 <= end) off = (uint)(d[ip + 1] | (d[ip + 2] << 8));
            }
            else
            {
                hdr = 5;
                len = (tag >> 2) + 1;
                if (ip + 5 <= end) off = Read32(d, ip + 1);
            }
            if (ip + hdr > end) hdr = 0;
        }
        return hdr;
    }

    /// <summary>One thread (raw): the length varint; the scan starts after it.</summary>
    private static void RawHeaderKernel(Index1D idx, ArrayView<byte> d, ArrayView<int> prm)
    {
        int length = prm[PLength];
        int total = 0;
        int n = ReadVarint(d, 0, length, ref total);
        if (n == 0) Fail(prm, SnappyError.Truncated, 0);
        else if (n == -1) Fail(prm, SnappyError.BadVarint, 0);
        else if (n == -2) Fail(prm, SnappyError.TooLarge, 0);
        else
        {
            prm[PTotal] = total;
            prm[PPos] = n;
            prm[PState] = WStart;
            prm[POut] = 0;
            prm[PSegs] = 0;
            prm[PSegDst] = 0;
            prm[PNextMark] = 0;
        }
    }

    /// <summary>
    /// One thread (raw), up to ScanHops elements per launch: validates each element (fits the stream, copy offset 1 ..
    /// output so far, output stays within the declared length) and records the segments: a new one at the first element
    /// at or past each SegmentBytes of output; FDependent when a copy reaches before its segment.
    /// </summary>
    private static void RawScanKernel(Index1D idx, ArrayView<byte> d, ArrayView<int> segs, ArrayView<int> prm)
    {
        int state = prm[PState];
        if ((state == WDone) | (prm[PError] != 0)) return;
        int length = prm[PLength];
        int total = prm[PTotal];
        int ip = prm[PPos];
        int op = prm[POut];
        int ns = prm[PSegs];
        int segDst = prm[PSegDst];
        int nextMark = prm[PNextMark];
        int maxSeg = prm[PMaxSeg];
        int any = prm[PAnyDependent];
        int cap = prm[PCapSegs];
        int hops = 0;
        bool go = true;
        while (go)
        {
            if (hops >= ScanHops) { go = false; continue; }
            hops++;
            bool atEnd = ip >= length;
            if ((op >= nextMark) | atEnd)
            {
                // Close the open segment (its source ends here) and, unless at the end, open the next.
                if (ns > 0)
                {
                    int la = (ns - 1) * SegInts;
                    segs[la + GEnd] = ip;
                    int size = op - segDst;
                    segs[la + GSize] = size;
                    if (size > maxSeg) maxSeg = size;
                }
                if (atEnd)
                {
                    if (op != total) Fail(prm, SnappyError.LengthMismatch, ip);
                    state = WDone;
                    go = false;
                    continue;
                }
                if (ns >= cap) { Fail(prm, SnappyError.TooLarge, ip); go = false; continue; }
                int sa = ns * SegInts;
                for (int k = 0; k < SegInts; k++) segs[sa + k] = 0;
                segs[sa + GSrc] = ip;
                segs[sa + GDst] = op;
                segs[sa + GIp] = ip;
                ns++;
                segDst = op;
                nextMark = (op / SegmentBytes + 1) * SegmentBytes;
            }
            bool isCopy = false;
            int len = 0;
            uint off = 0;
            int hdr = ReadElement(d, ip, length, ref isCopy, ref len, ref off);
            if (hdr == 0) { Fail(prm, SnappyError.Truncated, ip); go = false; continue; }
            if (len > total - op) { Fail(prm, SnappyError.LengthMismatch, ip); go = false; continue; }
            if (isCopy)
            {
                bool bad = (off == 0) | (off > (uint)op);
                if (bad) { Fail(prm, SnappyError.BadOffset, ip); go = false; continue; }
                if ((int)off > op - segDst)
                {
                    segs[(ns - 1) * SegInts + GFlags] = FDependent;
                    any = 1;
                }
                ip += hdr;
            }
            else ip += hdr + len;
            op += len;
        }
        prm[PState] = state;
        prm[PPos] = ip;
        prm[POut] = op;
        prm[PSegs] = ns;
        prm[PSegDst] = segDst;
        prm[PNextMark] = nextMark;
        prm[PMaxSeg] = maxSeg;
        prm[PAnyDependent] = any;
        if (state == WDone)
        {
            prm[POut] = 0;
            prm[PDependentNext] = 0;
        }
    }

    /// <summary>
    /// One thread (framed), up to WalkHops chunks per launch. Count pass (prm[PRecord] = 0): counts data chunks. Record
    /// pass: one segment per data chunk with its output offset (running sum), size, CRC; total and largest size in prm.
    /// </summary>
    private static void WalkKernel(Index1D idx, ArrayView<byte> d, ArrayView<int> segs, ArrayView<int> prm)
    {
        int state = prm[PState];
        if ((state == WDone) | (prm[PError] != 0)) return;
        int length = prm[PLength];
        int pos = prm[PPos];
        bool record = prm[PRecord] != 0;
        int ns = prm[PSegs];
        int total = prm[PTotal];
        int maxSeg = prm[PMaxSeg];
        int cap = prm[PCapSegs];
        int hops = 0;
        bool go = true;
        while (go)
        {
            if (hops >= WalkHops) { go = false; continue; }
            hops++;
            if (pos == length)
            {
                state = WDone;
                go = false;
                continue;
            }
            if (pos + 4 > length) { Fail(prm, SnappyError.Truncated, pos); go = false; continue; }
            int type = d[pos];
            int clen = d[pos + 1] | (d[pos + 2] << 8) | (d[pos + 3] << 16);
            int body = pos + 4;
            if (body + clen > length) { Fail(prm, SnappyError.Truncated, pos); go = false; continue; }
            if (type == 0xFF)
            {
                bool ok = clen == 6;
                if (ok) ok = (d[body] == (byte)'s') & (d[body + 1] == (byte)'N') & (d[body + 2] == (byte)'a') & (d[body + 3] == (byte)'P')
                    & (d[body + 4] == (byte)'p') & (d[body + 5] == (byte)'Y');
                if (!ok) { Fail(prm, SnappyError.BadStreamIdentifier, pos); go = false; continue; }
                state = WChunks;
            }
            else if (state == WStart) { Fail(prm, SnappyError.BadStreamIdentifier, pos); go = false; continue; }
            else if ((type == 0x00) | (type == 0x01))
            {
                if (clen < 4) { Fail(prm, SnappyError.BadChunkLength, pos); go = false; continue; }
                int src = body + 4;
                int end = body + clen;
                int size = clen - 4;
                int flags = FStored;
                if (type == 0x00)
                {
                    int n = ReadVarint(d, src, end, ref size);
                    if (n == 0) { Fail(prm, SnappyError.Truncated, src); go = false; continue; }
                    if (n < 0) { Fail(prm, SnappyError.BadVarint, src); go = false; continue; }
                    src += n;
                    flags = FCompressed;
                }
                if (size > MaxChunk) { Fail(prm, SnappyError.ChunkTooLarge, pos); go = false; continue; }
                if (record)
                {
                    if (ns >= cap) { Fail(prm, SnappyError.TooLarge, pos); go = false; continue; }
                    int sa = ns * SegInts;
                    for (int k = 0; k < SegInts; k++) segs[sa + k] = 0;
                    segs[sa + GSrc] = src;
                    segs[sa + GEnd] = end;
                    segs[sa + GDst] = total;
                    segs[sa + GSize] = size;
                    segs[sa + GFlags] = flags;
                    segs[sa + GCrc] = (int)Read32(d, body);
                    segs[sa + GIp] = src;
                    if (size > maxSeg) maxSeg = size;
                }
                long next = (long)total + size;
                if (next > int.MaxValue) { Fail(prm, SnappyError.TooLarge, pos); go = false; continue; }
                total = (int)next;
                ns++;
            }
            else if (type < 0x80) { Fail(prm, SnappyError.ReservedChunk, pos); go = false; continue; }
            // 0x80 .. 0xFE: skippable (0xFE = padding).
            pos = body + clen;
        }
        prm[PState] = state;
        prm[PPos] = pos;
        prm[PSegs] = ns;
        prm[PTotal] = total;
        prm[PMaxSeg] = maxSeg;
        if (state == WDone)
        {
            prm[POut] = 0;
            prm[PDependentNext] = 0;
        }
    }

    /// <summary>One thread per framed data chunk: a compressed chunk's elements fit it, copies stay inside it, and they add up to its varint.</summary>
    private static void ChunkScanKernel(Index1D idx, ArrayView<byte> d, ArrayView<int> segs, ArrayView<int> prm)
    {
        int sa = idx.X * SegInts;
        if ((segs[sa + GFlags] & FCompressed) == 0) return;
        int ip = segs[sa + GSrc];
        int end = segs[sa + GEnd];
        int size = segs[sa + GSize];
        int op = 0;
        bool go = ip < end;
        while (go)
        {
            bool isCopy = false;
            int len = 0;
            uint off = 0;
            int hdr = ReadElement(d, ip, end, ref isCopy, ref len, ref off);
            if (hdr == 0) { Fail(prm, SnappyError.Truncated, ip); go = false; continue; }
            if (len > size - op) { Fail(prm, SnappyError.LengthMismatch, ip); go = false; continue; }
            if (isCopy)
            {
                bool bad = (off == 0) | (off > (uint)op);
                if (bad) { Fail(prm, SnappyError.BadOffset, ip); go = false; continue; }
                ip += hdr;
            }
            else ip += hdr + len;
            op += len;
            go = ip < end;
        }
        if ((ip >= end) & (op != size)) Fail(prm, SnappyError.LengthMismatch, end);
    }

    /// <summary>
    /// Decode, DecodeBudget output bytes per launch and segment (a long literal or a copy resumes in the next launch).
    /// mode 0: one thread per segment, independent ones only. mode 1: one thread walks the dependent segments in order
    /// (prm[PDependentNext] = the next one). The structure was validated by the scans; prm[POut] counts finished segments.
    /// </summary>
    private static void DecodeKernel(Index1D idx, ArrayView<byte> d, ArrayView<byte> o, ArrayView<int> segs, ArrayView<int> prm, int mode)
    {
        if (prm[PError] != 0) return;
        int count = prm[PSegs];
        int g = idx.X;
        if (mode == 1)
        {
            g = prm[PDependentNext];
            bool find = true;
            while (find)
            {
                if (g >= count) find = false;
                else
                {
                    int sa0 = g * SegInts;
                    bool skip = ((segs[sa0 + GFlags] & FDependent) == 0) | (segs[sa0 + GPhase] == 2);
                    if (skip) g++;
                    else find = false;
                }
            }
            prm[PDependentNext] = g;
            if (g >= count) return;
        }
        int sa = g * SegInts;
        int flags = segs[sa + GFlags];
        if ((mode == 0) & ((flags & FDependent) != 0)) return;
        if (segs[sa + GPhase] == 2) return;
        int src = segs[sa + GSrc];
        int end = segs[sa + GEnd];
        int dst = segs[sa + GDst];
        int size = segs[sa + GSize];
        int ip = segs[sa + GIp];
        int op = segs[sa + GOp];
        int copyLeft = segs[sa + GCopyLeft];
        int copyOff = segs[sa + GCopyOff];
        int litLeft = segs[sa + GLitLeft];
        int budgetEnd = op + DecodeBudget;
        bool done = false;
        if ((flags & FStored) != 0)
        {
            int n = size - op;
            if (n > DecodeBudget) n = DecodeBudget;
            for (int k = 0; k < n; k++) o[dst + op + k] = d[src + op + k];
            op += n;
            done = op >= size;
        }
        else
        {
            bool go = true;
            while (go)
            {
                if (copyLeft > 0)
                {
                    int n = copyLeft;
                    if (n > budgetEnd - op) n = budgetEnd - op;
                    int from = dst + op - copyOff;
                    for (int k = 0; k < n; k++) o[dst + op + k] = o[from + k];
                    op += n;
                    copyLeft -= n;
                }
                if (litLeft > 0)
                {
                    int n = litLeft;
                    if (n > budgetEnd - op) n = budgetEnd - op;
                    for (int k = 0; k < n; k++) o[dst + op + k] = d[ip + k];
                    ip += n;
                    op += n;
                    litLeft -= n;
                }
                bool stop = (copyLeft > 0) | (litLeft > 0) | (op >= budgetEnd);
                if (stop) { go = false; continue; }
                if (ip >= end) { done = true; go = false; continue; }
                bool isCopy = false;
                int len = 0;
                uint off = 0;
                int hdr = ReadElement(d, ip, end, ref isCopy, ref len, ref off);
                ip += hdr;
                if (isCopy)
                {
                    copyLeft = len;
                    copyOff = (int)off;
                }
                else litLeft = len;
            }
        }
        segs[sa + GIp] = ip;
        segs[sa + GOp] = op;
        segs[sa + GCopyLeft] = copyLeft;
        segs[sa + GCopyOff] = copyOff;
        segs[sa + GLitLeft] = litLeft;
        if (done)
        {
            segs[sa + GPhase] = 2;
            Atomic.Add(ref prm[POut], 1);
        }
        else segs[sa + GPhase] = 1;
    }

    /// <summary>One thread per framed data chunk: the masked CRC-32C of its decoded bytes against the chunk's.</summary>
    private static void ChunkCheckKernel(Index1D idx, ArrayView<byte> o, ArrayView<int> segs, ArrayView<int> prm, ArrayView<uint> crcTable)
    {
        int sa = idx.X * SegInts;
        uint crc = Crc32Gpu.Finish(Crc32Gpu.Update(Crc32Gpu.Seed, o, segs[sa + GDst], segs[sa + GSize], crcTable));
        uint masked = ((crc >> 15) | (crc << 17)) + 0xA282EAD8u;
        if (masked != (uint)segs[sa + GCrc]) Fail(prm, SnappyError.ChunkChecksum, idx.X);
    }

    public void Dispose()
    {
        _prm.Dispose();
        _noSegs.Dispose();
        _noOut.Dispose();
        _segs?.Dispose();
        _crcTable.Dispose();
        _stageIn.Dispose();
        _stageOut.Dispose();
    }
}
