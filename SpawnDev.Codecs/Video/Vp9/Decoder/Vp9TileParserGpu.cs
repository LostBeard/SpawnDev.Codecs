// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt). See NOTICE.md for upstream attributions.
//
// Host side of the VP9 tile parse (coordination only): buffer setup, the per-frame arguments, the tile offsets (the
// 4-byte big-endian tile sizes between tiles, container-level bookkeeping as in libvpx get_tile_buffers), the
// segmentation map ping-pong and the per-frame-buffer MV arrays (libvpx cur_frame->mvs / prev_frame).

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp9.Decoder;

/// <summary>The VP9 parse stage on the accelerator: mode info, MVs, coefficients and counts of a frame.</summary>
public sealed class Vp9TileParserGpu : IDisposable
{
    private readonly Accelerator _acc;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _tables, _args;
    private MemoryBuffer1D<int, Stride1D.Dense>? _mi, _sb, _dq, _work, _maps, _counts;
    private Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>,
        ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>> _parse = null!;
    private int _miCols, _miRows, _miCount, _maxMiCount, _sbCount, _tileColsAlloc, _dqStride, _dqU, _dqV;
    private int _lastSeg, _curSeg;   // which of the two seg map regions is last / current
    private int _alignedCols;

    /// <summary>A parser on <paramref name="accelerator"/>.</summary>
    public Vp9TileParserGpu(Accelerator accelerator)
    {
        _acc = accelerator ?? throw new ArgumentNullException(nameof(accelerator));
        _tables = _acc.Allocate1D(Vp9PScan.Tables);
        _args = _acc.Allocate1D<int>(Vp9PArg.Count);
        _lastSeg = 0;
        _curSeg = 1;
    }

    /// <summary>Load the parse kernel (awaited: the compile yields to the browser's event loop first).</summary>
    internal async Task LoadKernelsAsync()
    {
        _parse = await _acc.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>,
            ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>>(Vp9ParseKernels.ParseKernel);
    }

    /// <summary>Mode info records + grid (see <see cref="Vp9Mi"/>; the grid follows the records).</summary>
    internal ArrayView<int> ModeInfo => _mi!.View;
    internal int MiCols => _miCols;
    internal int MiRows => _miRows;
    internal int GridOffset => _miCount * Vp9Mi.Stride;
    /// <summary>Per superblock: partitions and eobs (<see cref="Vp9PLayout"/>).</summary>
    internal ArrayView<int> Superblocks => _sb!.View;
    /// <summary>Dequantized coefficients per superblock (Y, then U at <see cref="DqU"/>, V at <see cref="DqV"/>).</summary>
    internal ArrayView<int> Coefficients => _dq!.View;
    internal int DqStride => _dqStride;
    internal int DqU => _dqU;
    internal int DqV => _dqV;
    /// <summary>FRAME_COUNTS (<see cref="Vp9Cnt"/>) per tile column.</summary>
    internal ArrayView<int> Counts => _counts!.View;
    /// <summary>Tests: the first tile column's FRAME_COUNTS of the last parsed frame.</summary>
    internal async Task<int[]> ReadCountsAsync() => await _counts!.View.SubView(0, Vp9Cnt.Size).CopyToHostAsync();
    internal int TileCols { get; private set; }
    /// <summary>Segmentation maps + MV arrays.</summary>
    internal ArrayView<int> Maps => _maps!.View;
    internal ArrayView<int> Work => _work!.View;

    private int SegOffset(int which) => which * _maxMiCount;
    private int MvOffset(int buffer) => 2 * _maxMiCount + buffer * _maxMiCount * Vp9PLayout.MvStride;

    private void Ensure(Vp9FrameInfo h)
    {
        int miCount = h.MiCols * h.MiRows;
        int sbCols = (h.MiCols + 7) >> 3, sbRows = (h.MiRows + 7) >> 3;
        int tileCols = 1 << h.Log2TileCols;
        int uv = 4096 >> (h.SubsamplingX + h.SubsamplingY);
        _miCols = h.MiCols;
        _miRows = h.MiRows;
        _alignedCols = sbCols * 8;
        if (_mi == null || miCount > _maxMiCount || sbCols * sbRows > _sbCount || tileCols > _tileColsAlloc || 4096 + 2 * uv != _dqStride)
        {
            _mi?.Dispose(); _sb?.Dispose(); _dq?.Dispose(); _work?.Dispose(); _maps?.Dispose(); _counts?.Dispose();
            _maxMiCount = Math.Max(miCount, _maxMiCount);
            _sbCount = Math.Max(sbCols * sbRows, _sbCount);
            _tileColsAlloc = Math.Max(tileCols, _tileColsAlloc);
            _dqStride = 4096 + 2 * uv;
            _mi = _acc.Allocate1D<int>((long)_maxMiCount * (Vp9Mi.Stride + 1));
            _sb = _acc.Allocate1D<int>((long)_sbCount * Vp9PLayout.SbStride);
            _dq = _acc.Allocate1D<int>((long)_sbCount * _dqStride);
            _work = _acc.Allocate1D<int>(WorkSize(_tileColsAlloc));
            _maps = _acc.Allocate1D<int>(2L * _maxMiCount + (long)Vp9HeaderReader.FrameBuffers * _maxMiCount * Vp9PLayout.MvStride);
            _counts = _acc.Allocate1D<int>((long)_tileColsAlloc * Vp9Cnt.Size);
            _mi.MemSetToZero();
            _maps.MemSetToZero();
        }
        _miCount = miCount;
        _dqU = 4096;
        _dqV = 4096 + uv;
    }

    /// <summary>VP9's widest frame (65536) in aligned mi columns.</summary>
    private const int MaxAlignedCols = 65536 / 8;

    /// <summary>
    /// Above entropy contexts (3 planes x 2 per mi column) + status words + each parse thread's state
    /// (<see cref="Vp9PLocal.ThreadInts"/> per tile column).
    /// </summary>
    private static long WorkSize(int tileCols) => 3L * 2 * MaxAlignedCols + tileCols + (long)tileCols * Vp9PLocal.ThreadInts;

    /// <summary>
    /// libvpx vp9_setup_past_independence clears both segmentation maps (call after <see cref="Vp9HeaderReader.ReadFrame"/>).
    /// </summary>
    private void ClearSegMaps() => _maps!.View.SubView(0, 2L * _maxMiCount).MemSetToZero();

    /// <summary>
    /// Parse every tile of the frame <paramref name="h"/> (bytes in <paramref name="frame"/> from <paramref name="frameOffset"/>,
    /// <paramref name="frameBytes"/> on the host for the tile sizes) with the current context in <paramref name="contexts"/>.
    /// <paramref name="prevBuffer"/>: the frame buffer of the previous decoded frame (its MVs), or -1.
    /// </summary>
    public void Parse(ArrayView<byte> frame, int frameOffset, ReadOnlySpan<byte> frameBytes, Vp9FrameInfo h, Vp9FrameContextsGpu contexts,
        int prevBuffer)
    {
        Ensure(h);
        if (h.PastIndependence) ClearSegMaps();
        int tileCols = 1 << h.Log2TileCols, tileRows = 1 << h.Log2TileRows;
        TileCols = tileCols;
        var a = new int[Vp9PArg.Count];
        a[Vp9PArg.MiCols] = h.MiCols;
        a[Vp9PArg.MiRows] = h.MiRows;
        a[Vp9PArg.Log2TileCols] = h.Log2TileCols;
        a[Vp9PArg.Log2TileRows] = h.Log2TileRows;
        a[Vp9PArg.IntraOnly] = h.IsIntraOnlyFrame ? 1 : 0;
        a[Vp9PArg.CountsOn] = h.ErrorResilient || h.FrameParallel ? 0 : 1;
        a[Vp9PArg.Interp] = (int)h.InterpFilter;
        a[Vp9PArg.AllowHp] = h.AllowHighPrecisionMv ? 1 : 0;
        a[Vp9PArg.UsePrevMvs] = h.UsePrevFrameMvs && prevBuffer >= 0 ? 1 : 0;
        a[Vp9PArg.Lossless] = h.Lossless ? 1 : 0;
        var seg = h.Segmentation;
        a[Vp9PArg.SegEnabled] = seg.Enabled ? 1 : 0;
        a[Vp9PArg.SegUpdateMap] = seg.UpdateMap ? 1 : 0;
        a[Vp9PArg.SegTemporal] = seg.TemporalUpdate ? 1 : 0;
        for (int i = 0; i < 7; i++) a[Vp9PArg.SegTree + i] = seg.TreeProbs[i];
        for (int i = 0; i < 3; i++) a[Vp9PArg.SegPred + i] = seg.PredProbs[i];
        for (int s = 0; s < 8; s++)
        {
            a[Vp9PArg.SegMask + s] = seg.FeatureMask[s];
            for (int f = 0; f < 4; f++) a[Vp9PArg.SegData + s * 4 + f] = seg.FeatureData[s, f];
        }
        for (int i = 0; i < 4; i++) a[Vp9PArg.SignBias + i] = h.SignBias[i];
        for (int s = 0; s < 8; s++)
            for (int k = 0; k < 4; k++) a[Vp9PArg.Dequant + s * 4 + k] = h.Dequant[Math.Min(s, h.Dequant.GetLength(0) - 1), k];
        a[Vp9PArg.SbCols] = (h.MiCols + 7) >> 3;
        a[Vp9PArg.GridOffset] = GridOffset;
        a[Vp9PArg.LastSegOffset] = SegOffset(_lastSeg);
        a[Vp9PArg.CurSegOffset] = SegOffset(_curSeg);
        a[Vp9PArg.CurMvOffset] = MvOffset(h.NewBuffer);
        a[Vp9PArg.PrevMvOffset] = prevBuffer >= 0 ? MvOffset(prevBuffer) : 0;
        int aboveCtx = 3 * 2 * _alignedCols;   // above entropy contexts from 0, then the status words
        a[Vp9PArg.StatusOffset] = aboveCtx;
        a[Vp9PArg.DqStride] = _dqStride;
        a[Vp9PArg.DqU] = _dqU;
        a[Vp9PArg.DqV] = _dqV;
        a[Vp9PArg.SsX] = h.SubsamplingX;
        a[Vp9PArg.SsY] = h.SubsamplingY;
        // get_tile_buffers
        int at = h.UncompressedHeaderBytes + h.FirstPartitionSize, end = frameBytes.Length;
        for (int r = 0; r < tileRows; r++)
            for (int c = 0; c < tileCols; c++)
            {
                bool last = r == tileRows - 1 && c == tileCols - 1;
                int size;
                if (!last)
                {
                    if (at + 4 > end) throw new InvalidDataException("VP9: truncated packet or corrupt tile length.");
                    size = (frameBytes[at] << 24) | (frameBytes[at + 1] << 16) | (frameBytes[at + 2] << 8) | frameBytes[at + 3];
                    at += 4;
                    if (size < 0 || size > end - at) throw new InvalidDataException("VP9: truncated packet or corrupt tile size.");
                }
                else size = end - at;
                a[Vp9PArg.TileBuf + (r * tileCols + c) * 2] = frameOffset + at;
                a[Vp9PArg.TileBuf + (r * tileCols + c) * 2 + 1] = size;
                at += size;
            }
        _args.View.CopyFromCPU(a);   // frame metadata scalars
        _dq!.View.SubView(0, (long)((h.MiCols + 7) >> 3) * ((h.MiRows + 7) >> 3) * _dqStride).MemSetToZero();
        _counts!.View.SubView(0, (long)tileCols * Vp9Cnt.Size).MemSetToZero();
        // decode_tiles' memset of the above entropy contexts (the above partition contexts are cleared in the kernel,
        // in each thread's local memory)
        _work!.View.SubView(0, aboveCtx).MemSetToZero();
        _parse(tileCols, frame, _args.View, contexts.Contexts, _tables.View, _mi!.View, _sb!.View, _dq.View, _work!.View, _maps!.View,
            _counts.View, contexts.Results);
    }

    /// <summary>End of a frame (libvpx: swap the segmentation maps when segmentation was enabled).</summary>
    public void EndFrame(Vp9FrameInfo h)
    {
        if (!h.ShowExistingFrame && h.Segmentation.Enabled) (_lastSeg, _curSeg) = (_curSeg, _lastSeg);
    }

    /// <summary>The per tile column status words of the last parse (0 = ok; a tiny readback).</summary>
    internal async Task<int[]> ReadStatusAsync() => await _work!.View.SubView(3 * 2 * _alignedCols, TileCols).CopyToHostAsync();

    /// <inheritdoc/>
    public void Dispose()
    {
        _tables.Dispose(); _args.Dispose();
        _mi?.Dispose(); _sb?.Dispose(); _dq?.Dispose(); _work?.Dispose(); _maps?.Dispose(); _counts?.Dispose();
    }
}
