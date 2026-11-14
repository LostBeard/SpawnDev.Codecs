// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP8 / VP9 boolean encoder over a symbol stream, in parallel. The bytes are identical to the serial
// Vp8BoolEncoderGpu (libvpx vpx_write + vpx_stop_encode) fed the same (bit, probability) symbols.
//
// Why it can run in parallel. The serial coder's state is (range, lowvalue, count). The range sequence depends only
// on the symbols and the previous range (8 bits: 128..255 after every renormalization, 255 at the start), and the
// output is the big-endian big integer
//     N = sum over the symbols coded 1 of split_i << (TotalBits - 8 - P_i)
// where split_i = 1 + (((range_i - 1) * prob_i) >> 8) and P_i = the renormalization shifts before symbol i (the carry
// propagation of the serial coder is exactly the carries of this sum). Its length is (P_total - 24) / 8 + 1 bytes,
// P_total counted after the 32 zero bits at probability 128 that end the stream (vpx_stop_encode). Checked against
// the serial coder before any GPU code was written (CPU model, 1,152 random streams, every chunk size).
//
// The passes (every launch bounded: no thread walks more than a fixed number of items):
//   Begin   counts per segment (the caller's leaves, in stream order) -> exclusive offsets + total, the coder state
//           reset (range 255, P 0) and the byte accumulator cleared.
//   Waves   the caller writes the symbols with a stream index in [w * WindowSymbols, (w + 1) * WindowSymbols) to
//           the window, then CodeWaveAsync(w):
//             Spec      one thread per (chunk of ChunkSymbols symbols, start range 128..255): the chunk's end range
//                       and its shifts for that start range;
//             GroupSpec one thread per (group of GroupChunks chunks, start range): the same over the group;
//             WaveWalk  one thread: from the carried (range, P), each group's real start, then the carried state;
//             ChunkStart one thread per group: each chunk's real start;
//             Contribute one thread per chunk: every symbol coded 1 adds split at bit P to the accumulator (atomics:
//                       neighbouring chunks share bytes).
//   Finish  carry passes over the accumulator (each moves the bits above 8 to the byte before), then carry-lookahead
//           per block of bytes, one thread over the blocks, and the bytes written with VP9's superframe-marker padding
//           byte. Bound: between two renormalization shifts the range falls from at most 255 to at least 128, so the
//           splits coded 1 at one bit position P sum to at most 127 + 254 = 381; a byte collects the high parts of the
//           8 positions in it (split >> o: at most 2 * 381) and the low parts of the 8 before it ((split << (8 - o)) &
//           255: at most 381 * (128 + 64 + ... + 2)), below 2^17. One pass leaves digits up to 255 + 381, two up to
//           255 + 2, and the lookahead takes any digit up to 510 (a carry in of 0 or 1 gives a carry out of 0 or 1):
//           two passes.
//
// Symbol format: ushort (prob << 1) | bit, prob 1..255.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp8;

/// <summary>
/// Where a symbol pass's next symbol goes: its stream index (<see cref="Next"/>) and the coder window [Lo, Hi) that is
/// written; a count pass leaves the window empty (Lo = Hi = 0) and reads <see cref="Next"/> as the count.
/// </summary>
public struct Vp8BoolSymbolSink
{
    /// <summary>Stream index of the next symbol; window bounds.</summary>
    public int Next, Lo, Hi;

    /// <summary>One bool: written as (prob &lt;&lt; 1) | bit when its index is in the window.</summary>
    public static void Put(ref Vp8BoolSymbolSink s, ArrayView<ushort> window, int bit, int prob)
    {
        bool inWindow = s.Next >= s.Lo & s.Next < s.Hi;
        if (inWindow) window[s.Next - s.Lo] = (ushort)((prob << 1) | bit);
        s.Next++;
    }
}

/// <summary>
/// The VP8 / VP9 boolean encoder over a symbol stream, in parallel (bytes identical to <see cref="Vp8BoolEncoderGpu"/>).
/// One instance codes one stream at a time; buffers grow to the largest stream seen.
/// </summary>
public sealed class Vp8BoolEncoderParallelGpu : IDisposable
{
    /// <summary>Symbols per chunk of the range pass (one speculative walk per start range).</summary>
    public const int ChunkSymbols = 256;
    /// <summary>Chunks per group of the range pass's second level.</summary>
    public const int GroupChunks = 64;
    /// <summary>Output bytes per block of the carry-lookahead pass (the default; tests use tiny blocks so carries cross them).</summary>
    public const int DefaultBlockBytes = 1024;
    /// <summary>Segments per group of the offsets scan.</summary>
    public const int ScanGroup = 256;
    /// <summary>The 32 zero bits at probability 128 that end a stream (vpx_stop_encode).</summary>
    public const int StopSymbols = 32;

    // coder state (ints): the carried range and shifts, the stream's symbol total, its byte length
    private const int SRange = 0, SP = 1, STotal = 2, SLen = 3, StateInts = 4;

    private readonly Accelerator _acc;
    private readonly Action<Index1D, ArrayView<int>, int, ArrayView<int>> _scanSums;
    private readonly Action<Index1D, ArrayView<int>, int, ArrayView<int>> _scanGroups;
    private readonly Action<Index1D, ArrayView<int>, int, ArrayView<int>, ArrayView<int>> _scanWrite;
    private readonly Action<Index1D, ArrayView<ushort>, ArrayView<int>, ArrayView<int>, int> _spec;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, int> _groupSpec;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, int, int> _waveWalk;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int> _chunkStart;
    private readonly Action<Index1D, ArrayView<ushort>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int> _contribute;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>> _carry;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, int> _blockStatus;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, int> _blockWalk;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<byte>, ArrayView<long>, int, int> _apply;

    private MemoryBuffer1D<int, Stride1D.Dense>? _offsets, _groupSums, _state, _spec1, _spec2, _groupStart, _chunkStartBuf, _accA, _accB, _blocks;
    private MemoryBuffer1D<ushort, Stride1D.Dense>? _window;
    private int _segCap, _accCap, _accAllocated, _windowSymbols, _blockBytes;

    /// <summary>Compile the kernels.</summary>
    public Vp8BoolEncoderParallelGpu(Accelerator accelerator, int windowSymbols = 1 << 22, int blockBytes = DefaultBlockBytes)
    {
        _acc = accelerator ?? throw new ArgumentNullException(nameof(accelerator));
        if (windowSymbols <= 0 || windowSymbols % (ChunkSymbols * GroupChunks) != 0)
            throw new ArgumentOutOfRangeException(nameof(windowSymbols), $"a positive multiple of {ChunkSymbols * GroupChunks}");
        if (blockBytes <= 0) throw new ArgumentOutOfRangeException(nameof(blockBytes));
        _windowSymbols = windowSymbols;
        _blockBytes = blockBytes;
        _scanSums = _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, int, ArrayView<int>>(ScanSumsKernel);
        _scanGroups = _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, int, ArrayView<int>>(ScanGroupsKernel);
        _scanWrite = _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, int, ArrayView<int>, ArrayView<int>>(ScanWriteKernel);
        _spec = _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<ushort>, ArrayView<int>, ArrayView<int>, int>(SpecKernel);
        _groupSpec = _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, int>(GroupSpecKernel);
        _waveWalk = _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, int, int>(WaveWalkKernel);
        _chunkStart = _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int>(ChunkStartKernel);
        _contribute = _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<ushort>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int>(ContributeKernel);
        _carry = _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>>(CarryKernel);
        _blockStatus = _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, int>(BlockStatusKernel);
        _blockWalk = _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, int>(BlockWalkKernel);
        _apply = _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<byte>, ArrayView<long>, int, int>(ApplyKernel);
    }

    /// <summary>
    /// Whether the parallel passes beat a serial coder on <paramref name="accelerator"/>: on the GPUs (CUDA, OpenCL,
    /// WebGPU) yes; on the CPU accelerators (CPU, Velocity, Wasm) no: the 128-way range speculation, the count pass and
    /// the idle windows are cheap on thousands of GPU threads and pure extra work on a few cores. Measured 2026-11-13,
    /// VP9, 512x512 q40, ms per frame, parallel / serial: CUDA key 14.6 / 236, inter 15.0 / 112; CPU accelerator key
    /// 320 / 204, inter 189 / 91; Velocity key 279 / 97, inter 858 / 780. Wasm not measured: the AV1 encoder's parallel
    /// symbol pass made its Wasm frames slower (114 -> 123 ms), so it codes serially until measured.
    /// </summary>
    public static bool PaysOn(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        var t = accelerator.AcceleratorType;
        return t == AcceleratorType.Cuda || t == AcceleratorType.OpenCL || t == AcceleratorType.WebGPU;
    }

    /// <summary>Symbols per wave window (the caller writes a window's symbols at index - wave * WindowSymbols).</summary>
    public int WindowSymbols => _windowSymbols;

    /// <summary>The window the caller writes a wave's symbols to (valid after <see cref="Begin"/>).</summary>
    public ArrayView<ushort> Window => _window!.View;

    /// <summary>Each segment's first symbol index in the stream (valid after <see cref="Begin"/>).</summary>
    public ArrayView<int> Offsets => _offsets!.View;

    /// <summary>
    /// Allocate for streams of up to <paramref name="segmentCount"/> segments and outputs of up to
    /// <paramref name="outputCapacity"/> bytes, so <see cref="Window"/> exists before the caller's first kernel (a count
    /// pass) binds it. Growing REPLACES buffers: call it (and <see cref="Begin"/> with larger sizes) only when no queued
    /// work still uses this coder. Several streams in a row (the tiles of a frame): prepare once for the largest first.
    /// Found 2026-11-13: a frame's second, larger tile grew the offsets buffer while the first tile's launches were
    /// queued; on Wasm the queued work wrote into the freed buffer (RangeError), elsewhere it was a silent use-after-free.
    /// </summary>
    public void Prepare(int segmentCount, int outputCapacity)
    {
        if (segmentCount <= 0) throw new ArgumentOutOfRangeException(nameof(segmentCount));
        if (outputCapacity <= 0) throw new ArgumentOutOfRangeException(nameof(outputCapacity));
        Ensure(segmentCount, outputCapacity);
    }

    /// <summary>Waves that cover a stream of at most <paramref name="maxSymbols"/> symbols (the stop symbols included).</summary>
    public int WaveCount(long maxSymbols) => (int)((maxSymbols + StopSymbols + _windowSymbols - 1) / _windowSymbols);

    /// <summary>
    /// Start a stream: the exclusive offsets of <paramref name="counts"/> (symbols per segment, in stream order), the
    /// coder state, and the accumulator for an output of at most <paramref name="outputCapacity"/> bytes. Queued, no wait.
    /// </summary>
    public void Begin(ArrayView<int> counts, int segmentCount, int outputCapacity)
    {
        if (segmentCount <= 0) throw new ArgumentOutOfRangeException(nameof(segmentCount));
        if (outputCapacity <= 0) throw new ArgumentOutOfRangeException(nameof(outputCapacity));
        Ensure(segmentCount, outputCapacity);
        int groups = (segmentCount + ScanGroup - 1) / ScanGroup;
        _scanSums(groups, counts, segmentCount, _groupSums!.View);
        _scanGroups(1, _groupSums.View, groups, _state!.View);
        _scanWrite(groups, counts, segmentCount, _groupSums.View, _offsets!.View);
        _accA!.View.SubView(0, _accCap).MemSetToZero();
    }

    /// <summary>Code wave <paramref name="wave"/> (the caller has written its symbols to <see cref="Window"/>). Queued, no wait.</summary>
    public void CodeWave(int wave)
    {
        int chunks = _windowSymbols / ChunkSymbols;
        int groups = chunks / GroupChunks;
        int baseIdx = checked(wave * _windowSymbols);
        _spec(chunks * 128, _window!.View, _state!.View, _spec1!.View, baseIdx);
        _groupSpec(groups * 128, _spec1.View, _state.View, _spec2!.View, baseIdx);
        _waveWalk(1, _spec2.View, _state.View, _groupStart!.View, baseIdx, groups);
        _chunkStart(groups, _spec1.View, _state.View, _groupStart.View, _chunkStartBuf!.View, baseIdx);
        _contribute(chunks, _window.View, _state.View, _chunkStartBuf.View, _accA!.View.SubView(0, _accCap), baseIdx);
    }

    /// <summary>
    /// Resolve the carries and write the stream's bytes to <paramref name="output"/> (bytes past its length are not
    /// written) and its length to <paramref name="outLen"/>[0] (the true length, larger than the output when it did
    /// not fit, as <see cref="Vp8BoolEncoderGpu"/> counts). <paramref name="vp9"/>: VP9's padding byte after a last byte
    /// that reads as a superframe marker (vpx_stop_encode). Queued, no wait.
    /// </summary>
    public void Finish(ArrayView<byte> output, ArrayView<long> outLen, bool vp9)
    {
        var a = _accA!.View.SubView(0, _accCap);
        var b = _accB!.View.SubView(0, _accCap);
        _carry(_accCap, a, b);
        _carry(_accCap, b, a);
        int blocks = (_accCap + _blockBytes - 1) / _blockBytes;
        _blockStatus(blocks, a, _state!.View, _blocks!.View, _blockBytes);
        _blockWalk(1, _blocks.View, _state.View, blocks);
        _apply(blocks, a, _state.View, _blocks.View, output, outLen, vp9 ? 1 : 0, _blockBytes);
    }

    private void Ensure(int segmentCount, int outputCapacity)
    {
        _state ??= _acc.Allocate1D<int>(StateInts);
        if (_window == null)
        {
            int chunks = _windowSymbols / ChunkSymbols;
            _window = _acc.Allocate1D<ushort>(_windowSymbols);
            _spec1 = _acc.Allocate1D<int>((long)chunks * 128);
            _spec2 = _acc.Allocate1D<int>((long)chunks / GroupChunks * 128);
            _groupStart = _acc.Allocate1D<int>((long)chunks / GroupChunks * 2);
            _chunkStartBuf = _acc.Allocate1D<int>((long)chunks * 2);
        }
        if (segmentCount > _segCap)
        {
            _offsets?.Dispose(); _groupSums?.Dispose();
            _segCap = segmentCount;
            _offsets = _acc.Allocate1D<int>(segmentCount);
            _groupSums = _acc.Allocate1D<int>((segmentCount + ScanGroup - 1) / ScanGroup);
        }
        // the accumulator: every output byte + 4 (contributions reach one byte past a short output)
        int accNeeded = checked(outputCapacity + 4);
        if (accNeeded > _accAllocated)
        {
            _accA?.Dispose(); _accB?.Dispose(); _blocks?.Dispose();
            _accA = _acc.Allocate1D<int>(accNeeded);
            _accB = _acc.Allocate1D<int>(accNeeded);
            _blocks = _acc.Allocate1D<int>((accNeeded + _blockBytes - 1) / _blockBytes);
            _accAllocated = accNeeded;
        }
        _accCap = accNeeded;
    }

    // ---- offsets scan ----

    private static void ScanSumsKernel(Index1D g, ArrayView<int> counts, int segCount, ArrayView<int> groupSums)
    {
        int start = g * ScanGroup;
        int end = start + ScanGroup;
        if (end > segCount) end = segCount;
        int sum = 0;
        for (int i = start; i < end; i++) sum += counts[i];
        groupSums[g] = sum;
    }

    // One thread: the groups' sums -> their exclusive offsets (in place), the total, the coder's start state.
    private static void ScanGroupsKernel(Index1D t, ArrayView<int> groupSums, int groups, ArrayView<int> state)
    {
        int run = 0;
        for (int g = 0; g < groups; g++)
        {
            int s = groupSums[g];
            groupSums[g] = run;
            run += s;
        }
        state[STotal] = run;
        state[SRange] = 255;
        state[SP] = 0;
        state[SLen] = 0;
    }

    private static void ScanWriteKernel(Index1D g, ArrayView<int> counts, int segCount, ArrayView<int> groupOffsets, ArrayView<int> offsets)
    {
        int start = g * ScanGroup;
        int end = start + ScanGroup;
        if (end > segCount) end = segCount;
        int run = groupOffsets[g];
        for (int i = start; i < end; i++)
        {
            offsets[i] = run;
            run += counts[i];
        }
    }

    // ---- range pass ----

    /// <summary>Symbol <paramref name="index"/> of the stream: from the window, or one of the stop symbols (bit 0, prob 128).</summary>
    private static int SymbolAt(ArrayView<ushort> window, int index, int baseIdx, int total)
    {
        int sym = 128 << 1;
        if (index < total) sym = window[index - baseIdx];
        return sym;
    }

    /// <summary>The range after coding <paramref name="sym"/> from <paramref name="range"/>, before renormalization.</summary>
    private static int Split(int range, int sym) => 1 + (((range - 1) * (sym >> 1)) >> 8);

    /// <summary>Renormalization shift of a range 1..255 (vpx_norm).</summary>
    private static int Norm(int r) => System.Numerics.BitOperations.LeadingZeroCount((uint)r) - 24;

    private static void SpecKernel(Index1D t, ArrayView<ushort> window, ArrayView<int> state, ArrayView<int> spec, int baseIdx)
    {
        int chunk = t >> 7;
        int first = baseIdx + chunk * ChunkSymbols;
        int n = state[STotal] + StopSymbols;
        if (first >= n) return;
        int end = first + ChunkSymbols;
        if (end > n) end = n;
        int total = state[STotal];
        int r = 128 + (t & 127);
        int shifts = 0;
        for (int i = first; i < end; i++)
        {
            int sym = SymbolAt(window, i, baseIdx, total);
            int split = Split(r, sym);
            int r2 = (sym & 1) != 0 ? r - split : split;
            int k = Norm(r2);
            r = r2 << k;
            shifts += k;
        }
        spec[t] = (shifts << 8) | r;
    }

    private static void GroupSpecKernel(Index1D t, ArrayView<int> spec, ArrayView<int> state, ArrayView<int> groupSpec, int baseIdx)
    {
        int group = t >> 7;
        int n = state[STotal] + StopSymbols;
        int firstChunk = group * GroupChunks;
        if (baseIdx + firstChunk * ChunkSymbols >= n) return;
        int r = 128 + (t & 127);
        int shifts = 0;
        for (int c = 0; c < GroupChunks; c++)
        {
            int chunk = firstChunk + c;
            bool live = baseIdx + chunk * ChunkSymbols < n;
            if (live)
            {
                int e = spec[chunk * 128 + r - 128];
                shifts += e >> 8;
                r = e & 255;
            }
        }
        groupSpec[t] = (shifts << 8) | r;
    }

    // One thread: each group's real start (range, P) from the state carried from the previous wave; the state after
    // this wave's symbols is carried to the next.
    private static void WaveWalkKernel(Index1D t, ArrayView<int> groupSpec, ArrayView<int> state, ArrayView<int> groupStart, int baseIdx, int groups)
    {
        int n = state[STotal] + StopSymbols;
        if (baseIdx >= n) return;
        int r = state[SRange];
        int p = state[SP];
        for (int g = 0; g < groups; g++)
        {
            bool live = baseIdx + g * GroupChunks * ChunkSymbols < n;
            if (live)
            {
                groupStart[g * 2] = r;
                groupStart[g * 2 + 1] = p;
                int e = groupSpec[g * 128 + r - 128];
                p += e >> 8;
                r = e & 255;
            }
        }
        state[SRange] = r;
        state[SP] = p;
    }

    private static void ChunkStartKernel(Index1D g, ArrayView<int> spec, ArrayView<int> state, ArrayView<int> groupStart, ArrayView<int> chunkStart, int baseIdx)
    {
        int n = state[STotal] + StopSymbols;
        int firstChunk = g * GroupChunks;
        if (baseIdx + firstChunk * ChunkSymbols >= n) return;
        int r = groupStart[g * 2];
        int p = groupStart[g * 2 + 1];
        for (int c = 0; c < GroupChunks; c++)
        {
            int chunk = firstChunk + c;
            bool live = baseIdx + chunk * ChunkSymbols < n;
            if (live)
            {
                chunkStart[chunk * 2] = r;
                chunkStart[chunk * 2 + 1] = p;
                int e = spec[chunk * 128 + r - 128];
                p += e >> 8;
                r = e & 255;
            }
        }
    }

    private static void ContributeKernel(Index1D chunk, ArrayView<ushort> window, ArrayView<int> state, ArrayView<int> chunkStart, ArrayView<int> acc, int baseIdx)
    {
        int first = baseIdx + chunk * ChunkSymbols;
        int total = state[STotal];
        // the stop symbols are zeros: they add nothing
        if (first >= total) return;
        int end = first + ChunkSymbols;
        if (end > total) end = total;
        int r = chunkStart[chunk * 2];
        int p = chunkStart[chunk * 2 + 1];
        int accLen = acc.IntLength;
        for (int i = first; i < end; i++)
        {
            int sym = window[i - baseIdx];
            int split = Split(r, sym);
            int r2 = split;
            if ((sym & 1) != 0)
            {
                r2 = r - split;
                // split's 8 bits at output bits [p, p + 8), most significant first
                int j = p >> 3;
                int v = split << (8 - (p & 7));
                int hi = v >> 8, lo = v & 255;
                if (hi != 0 & j < accLen) Atomic.Add(ref acc[j], hi);
                if (lo != 0 & j + 1 < accLen) Atomic.Add(ref acc[j + 1], lo);
            }
            int k = Norm(r2);
            r = r2 << k;
            p += k;
        }
    }

    // ---- carries ----

    // dst[j] = the low 8 bits of src[j] + the bits of src[j + 1] above 8 (a byte's carry goes to the byte before it)
    private static void CarryKernel(Index1D j, ArrayView<int> src, ArrayView<int> dst)
    {
        int len = src.IntLength;
        int v = src[j] & 255;
        if (j + 1 < len) v += src[j + 1] >> 8;
        dst[j] = v;
    }

    // Per block of digits 0..510: the carry out of its first byte for a carry 0 (bit 0) and 1 (bit 1) into its last.
    // Also the stream's byte length from its total shifts (thread 0).
    private static void BlockStatusKernel(Index1D b, ArrayView<int> digits, ArrayView<int> state, ArrayView<int> blocks, int blockBytes)
    {
        int len = digits.IntLength;
        int start = b * blockBytes;
        int end = start + blockBytes;
        if (end > len) end = len;
        int c0 = 0, c1 = 1;
        for (int j = end - 1; j >= start; j--)
        {
            int d = digits[j];
            c0 = (d + c0) >> 8;
            c1 = (d + c1) >> 8;
        }
        blocks[b] = c0 | (c1 << 1);
        if (b == 0)
        {
            int p = state[SP];
            state[SLen] = p >= 24 ? ((p - 24) >> 3) + 1 : 0;
        }
    }

    // One thread, last block to first: the carry into each block (bit 2 of its entry).
    private static void BlockWalkKernel(Index1D t, ArrayView<int> blocks, ArrayView<int> state, int blockCount)
    {
        int c = 0;
        for (int b = blockCount - 1; b >= 0; b--)
        {
            int s = blocks[b];
            blocks[b] = (s & 3) | (c << 2);
            c = c != 0 ? (s >> 1) & 1 : s & 1;
        }
    }

    private static void ApplyKernel(Index1D b, ArrayView<int> digits, ArrayView<int> state, ArrayView<int> blocks,
        ArrayView<byte> output, ArrayView<long> outLen, int vp9, int blockBytes)
    {
        int len = state[SLen];
        int start = b * blockBytes;
        int end = start + blockBytes;
        if (end > digits.IntLength) end = digits.IntLength;
        int c = (blocks[b] >> 2) & 1;
        int cap = output.IntLength;
        for (int j = end - 1; j >= start; j--)
        {
            int s = digits[j] + c;
            c = s >> 8;
            bool write = j < len & j < cap;
            if (write) output[j] = (byte)(s & 255);
            if (j == len - 1)
            {
                // vpx_stop_encode: a 0 byte after a last byte that reads as a superframe index marker (0b110xxxxx)
                int pad = vp9 != 0 & (s & 0xE0) == 0xC0 ? 1 : 0;
                if (pad != 0 & len < cap) output[len] = 0;
                outLen[0] = len + pad;
            }
        }
        if (b == 0 & len == 0) outLen[0] = 0;
    }

    /// <summary>Release the buffers.</summary>
    public void Dispose()
    {
        _offsets?.Dispose(); _groupSums?.Dispose(); _state?.Dispose(); _spec1?.Dispose(); _spec2?.Dispose();
        _groupStart?.Dispose(); _chunkStartBuf?.Dispose(); _accA?.Dispose(); _accB?.Dispose(); _blocks?.Dispose();
        _window?.Dispose();
    }
}
