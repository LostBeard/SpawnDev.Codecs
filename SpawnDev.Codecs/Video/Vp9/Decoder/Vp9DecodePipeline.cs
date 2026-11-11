// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt). See NOTICE.md for upstream attributions.
//
// The VP9 decoder's per-frame pipeline (host = coordinator): uncompressed header (host metadata) -> frame contexts ->
// compressed header kernel -> tile parse kernel -> reconstruction kernel -> [loop filter] -> backward adaptation ->
// context store and reference bookkeeping. libvpx v1.17.0 vp9_receive_compressed_data / vp9_decode_frame order.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp9.Decoder;

/// <summary>Decodes VP9 frames into frame buffers on the accelerator.</summary>
internal sealed class Vp9DecodePipeline : IDisposable
{
    private readonly Accelerator _acc;
    private readonly Vp9HeaderReader _reader = new();
    private readonly Vp9FrameContextsGpu _contexts;
    private readonly Vp9TileParserGpu _parser;
    private readonly Vp9FrameBuffersGpu _frames;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _tables, _reconArgs, _lfHdr, _lfTable, _lfArgs;
    private Action<Index1D, ArrayView<int>, ArrayView<int>> _lfInit = null!;
    private Action<KernelConfig, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int> _lfWave = null!;
    private MemoryBuffer1D<int, Stride1D.Dense>? _scratch;
    private Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>> _interPredict = null!;
    private Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>> _interResidual = null!;
    private Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int> _reconIntra = null!;
    private int _prevBuffer = -1;
    private Task? _kernelsLoad;
    private Action<Index1D, ArrayView<byte>, ArrayView<byte>, int, int, int, int, int, int, long, long, long>? _packDiag;

    public Vp9DecodePipeline(Accelerator accelerator)
    {
        _acc = accelerator ?? throw new ArgumentNullException(nameof(accelerator));
        _contexts = new Vp9FrameContextsGpu(_acc);
        _parser = new Vp9TileParserGpu(_acc);
        _frames = new Vp9FrameBuffersGpu(_acc);
        _tables = _acc.Allocate1D(Vp9T.Packed);
        _reconArgs = _acc.Allocate1D<int>(Vp9RArg.Count);
        _lfHdr = _acc.Allocate1D<int>(Vp9LfHdr.Ints);
        _lfTable = _acc.Allocate1D<int>(Vp9LfTable.Ints);
        _lfArgs = _acc.Allocate1D<int>(Vp9LArg.Count);
    }

    /// <summary>Load every kernel once (awaited loads: each compile yields to the browser's event loop first).</summary>
    internal Task EnsureKernelsAsync()
    {
        if (_kernelsLoad == null || _kernelsLoad.IsFaulted || _kernelsLoad.IsCanceled) _kernelsLoad = LoadKernelsAsync();
        return _kernelsLoad;
    }

    private async Task LoadKernelsAsync()
    {
        await _contexts.LoadKernelsAsync();
        await _parser.LoadKernelsAsync();
        _interPredict = await _acc.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>>(
            Vp9ReconKernels.InterPredictKernel);
        _interResidual = await _acc.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>,
            ArrayView<int>, ArrayView<int>>(Vp9ReconKernels.InterResidualKernel);
        _reconIntra = await _acc.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>,
            ArrayView<int>, ArrayView<int>, int>(Vp9ReconKernels.IntraWaveKernel);
        _lfInit = await _acc.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<int>, ArrayView<int>>(Vp9LoopFilterKernels.FrameInitKernel);
        _lfWave = await _acc.LoadStreamKernelAsync<ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int>(
            Vp9LoopFilterKernels.WavefrontKernel);
    }

    /// <summary>libvpx vp9_loop_filter_frame (after the tiles): filter level table, then the superblock wavefront.</summary>
    private void LoopFilter(Vp9FrameInfo h)
    {
        if (h.LoopFilter.Level == 0) return;   // libvpx: cm->lf.filter_level == 0 -> no filtering
        var hdr = new int[Vp9LfHdr.Ints];
        Vp9LfHdr.Pack(hdr, h.LoopFilter, h.Segmentation);
        _lfHdr.View.CopyFromCPU(hdr);   // frame metadata scalars
        _lfInit(1, _lfHdr.View, _lfTable.View);
        var l = _frames.Layout(h.NewBuffer);
        long b = _frames.Offset(h.NewBuffer);
        var a = new int[Vp9LArg.Count];
        a[Vp9LArg.MiRows] = h.MiRows;
        a[Vp9LArg.MiCols] = h.MiCols;
        a[Vp9LArg.SsX] = h.SubsamplingX;
        a[Vp9LArg.SsY] = h.SubsamplingY;
        a[Vp9LArg.GridOffset] = _parser.GridOffset;
        a[Vp9LArg.Y] = checked((int)b);
        a[Vp9LArg.U] = checked((int)(b + l.UOffset));
        a[Vp9LArg.V] = checked((int)(b + l.VOffset));
        a[Vp9LArg.YStride] = l.YStride;
        a[Vp9LArg.UvStride] = l.UvStride;
        _lfArgs.View.CopyFromCPU(a);
        int sbRows = (h.MiRows + 7) >> 3, sbCols = (h.MiCols + 7) >> 3;
        for (int k = 0; k < 2 * (sbRows - 1) + sbCols; k++)
        {
            int rMax = Math.Min(sbRows - 1, k / 2);
            // one thread group per superblock row of the step
            _lfWave(new KernelConfig(rMax + 1, Vp9Lpf.GroupLanes), _frames.Pool, _lfArgs.View, _parser.ModeInfo, _lfTable.View, k);
        }
    }

    /// <summary>Diagnostics only: see <see cref="Vp9RArg.DebugFlags"/> (tests set it from VP9_DEBUG_FLAGS).</summary>
    internal int DebugFlags { get; set; }

    /// <summary>
    /// Diagnostics: when set, every frame waits for the accelerator after each stage and adds the stage's time (ms) here
    /// (total and worst frame), so a long single submission (a browser GPU watchdog) can be pinned to a stage. Null in
    /// normal decoding: no waits are added.
    /// </summary>
    internal Dictionary<string, (double TotalMs, double MaxMs, int Frames)>? StageTimes { get; set; }

    private async Task MarkStageAsync(System.Diagnostics.Stopwatch? timer, string stage)
    {
        if (timer == null || StageTimes == null) return;
        await _acc.SynchronizeAsync();
        double ms = timer.Elapsed.TotalMilliseconds;
        StageTimes.TryGetValue(stage, out var t);
        StageTimes[stage] = (t.TotalMs + ms, Math.Max(t.MaxMs, ms), t.Frames + 1);
        timer.Restart();
    }

    internal Vp9FrameContextsGpu Contexts => _contexts;
    internal Vp9TileParserGpu Parser => _parser;
    internal Vp9FrameBuffersGpu Frames => _frames;

    /// <summary>
    /// Decode one frame (<paramref name="frame"/> on the host for the headers, <paramref name="frameView"/> = the same bytes
    /// on the accelerator from <paramref name="frameOffset"/>). <paramref name="afterRecon"/> runs between the
    /// reconstruction and the loop filter (diagnostics). Returns the frame's header; the decoded picture is frame buffer
    /// <see cref="Vp9FrameInfo.NewBuffer"/>.
    /// </summary>
    public async Task<Vp9FrameInfo> DecodeFrameAsync(ReadOnlyMemory<byte> frame, ArrayView<byte> frameView, int frameOffset,
        Func<Vp9FrameInfo, Task>? afterRecon = null)
    {
        await EnsureKernelsAsync();
        var h = _reader.ReadFrame(frame.Span);
        if (h.ShowExistingFrame)
        {
            _reader.FinishFrame();
            return h;
        }
        _frames.Ensure(h.NewBuffer, h.Width, h.Height, h.SubsamplingX, h.SubsamplingY);
        var timer = StageTimes != null ? System.Diagnostics.Stopwatch.StartNew() : null;
        _contexts.BeginFrame(h);
        _contexts.ReadCompressedHeader(frameView, frameOffset, h);
        await MarkStageAsync(timer, "compressed header");
        _parser.Parse(frameView, frameOffset, frame.Span, h, _contexts, _prevBuffer);
        await MarkStageAsync(timer, "tile parse");
        Reconstruct(h);
        await MarkStageAsync(timer, "reconstruction");
        if (afterRecon != null) await afterRecon(h);
        timer?.Restart();
        LoopFilter(h);
        await MarkStageAsync(timer, "loop filter");
        _contexts.Adapt(h, _parser.Counts, _parser.TileCols);
        await MarkStageAsync(timer, "adaptation");
        _contexts.EndFrame(h);
        _parser.EndFrame(h);
        _reader.FinishFrame();
        _prevBuffer = h.NewBuffer;
        return h;
    }

    private void Reconstruct(Vp9FrameInfo h)
    {
        int sbRows = (h.MiRows + 7) >> 3, sbCols = (h.MiCols + 7) >> 3;
        // launch sizes in 4x4 units (Y, then U and V): inter prediction over the superblock grid, inter residual over
        // the mi grid
        int predUnits = sbCols * 16 * (sbRows * 16) + 2 * ((sbCols * 16) >> h.SubsamplingX) * ((sbRows * 16) >> h.SubsamplingY);
        int residUnits = h.MiCols * 2 * (h.MiRows * 2) + 2 * ((h.MiCols * 2) >> h.SubsamplingX) * ((h.MiRows * 2) >> h.SubsamplingY);
        // one scratch buffer for the launches, which run one after another: per prediction unit, per inter transform
        // block (Vp9RArg.ResidualScratchAt over the frame's coefficients), per intra superblock thread
        // (in 64 bits here: the kernels index it with 32-bit offsets, so it must stay below 2^31 ints)
        long scratchInts = Math.Max((long)sbRows * sbCols * Vp9RArg.Scratch,
            Math.Max((long)predUnits * Vp9InterPred.RegionScratchInts, ((long)sbRows * sbCols * _parser.DqStride >> 2) * 5));
        if (scratchInts > int.MaxValue)
            throw new NotSupportedException($"VP9 frames of {h.Width}x{h.Height} need {scratchInts:N0} scratch ints; the decoder addresses at most {int.MaxValue:N0}.");
        if (_scratch == null || _scratch.Length < scratchInts)
        {
            _scratch?.Dispose();
            _scratch = _acc.Allocate1D<int>(scratchInts);
        }
        var a = new int[Vp9RArg.Count];
        a[Vp9RArg.MiCols] = h.MiCols;
        a[Vp9RArg.MiRows] = h.MiRows;
        a[Vp9RArg.Log2TileCols] = h.Log2TileCols;
        a[Vp9RArg.Log2TileRows] = h.Log2TileRows;
        a[Vp9RArg.Lossless] = h.Lossless ? 1 : 0;
        a[Vp9RArg.SsX] = h.SubsamplingX;
        a[Vp9RArg.SsY] = h.SubsamplingY;
        a[Vp9RArg.SbCols] = (h.MiCols + 7) >> 3;
        a[Vp9RArg.GridOffset] = _parser.GridOffset;
        a[Vp9RArg.DqStride] = _parser.DqStride;
        a[Vp9RArg.DqU] = _parser.DqU;
        a[Vp9RArg.DqV] = _parser.DqV;
        a[Vp9RArg.DebugFlags] = DebugFlags;
        var cur = _frames.Layout(h.NewBuffer);
        long curBase = _frames.Offset(h.NewBuffer);
        for (int p = 0; p < 3; p++)
        {
            a[Vp9RArg.Cur + p * 4] = checked((int)(curBase + cur.PlaneOffset(p)));
            a[Vp9RArg.Cur + p * 4 + 1] = cur.Stride(p);
            a[Vp9RArg.Cur + p * 4 + 2] = p == 0 ? cur.AlignedWidth : cur.UvAlignedWidth;
            a[Vp9RArg.Cur + p * 4 + 3] = p == 0 ? cur.AlignedHeight : cur.UvAlignedHeight;
        }
        if (!h.IsIntraOnlyFrame)
            for (int r = 0; r < 3; r++)
            {
                int buf = h.RefBuffers[r];
                var rl = _frames.Layout(buf);
                long rBase = _frames.Offset(buf);
                for (int p = 0; p < 3; p++)
                {
                    int at = Vp9RArg.Ref + (r * 3 + p) * 4;
                    a[at] = checked((int)(rBase + rl.PlaneOffset(p)));
                    a[at + 1] = rl.Stride(p);
                    a[at + 2] = rl.CropWidth(p);
                    a[at + 3] = rl.CropHeight(p);
                }
                var sf = Vp9InterPred.SetupScaleFactors(rl.Width, rl.Height, h.Width, h.Height);
                a[Vp9RArg.Scale + r * 4] = sf.XScaleFp;
                a[Vp9RArg.Scale + r * 4 + 1] = sf.YScaleFp;
                a[Vp9RArg.Scale + r * 4 + 2] = sf.XStepQ4;
                a[Vp9RArg.Scale + r * 4 + 3] = sf.YStepQ4;
            }
        _reconArgs.View.CopyFromCPU(a);   // frame metadata scalars
        // inter prediction of every inter pixel, then the inter residual per transform block (after all prediction),
        // then the intra blocks on the superblock wavefront row + col
        if (!h.IsIntraOnlyFrame)
        {
            _interPredict(predUnits, _frames.Pool, _reconArgs.View, _parser.ModeInfo, _scratch.View);
            _interResidual(residUnits, _frames.Pool, _reconArgs.View, _tables.View, _parser.ModeInfo, _parser.Superblocks,
                _parser.Coefficients, _scratch.View);
        }
        for (int k = 0; k < sbRows + sbCols - 1; k++)
            _reconIntra(Math.Min(sbRows - 1, k) + 1, _frames.Pool, _reconArgs.View, _tables.View, _parser.ModeInfo, _parser.Superblocks,
                _parser.Coefficients, _scratch.View, k);
    }

    /// <summary>The cropped planes (Y, U, V, tight) of frame buffer <paramref name="buffer"/> (diagnostics / tests).</summary>
    internal async Task<byte[]> ReadPlanesAsync(int buffer)
    {
        var l = _frames.Layout(buffer);
        long b = _frames.Offset(buffer);
        int uvW = l.UvCropWidth, uvH = l.UvCropHeight;
        long total = (long)l.Width * l.Height + 2L * uvW * uvH;
        using var dst = _acc.Allocate1D<byte>(total);
        var pack = _packDiag ??= await _acc.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<byte>, ArrayView<byte>, int, int, int, int, int, int, long, long, long>(
            Vp9OutputKernels.PackPlanesKernel);
        pack((int)total, _frames.Pool, dst.View, l.Width, l.Height, uvW, uvH, l.YStride, l.UvStride, b, b + l.UOffset, b + l.VOffset);
        return await dst.View.CopyToHostAsync();
    }

    public void Dispose()
    {
        _contexts.Dispose();
        _parser.Dispose();
        _frames.Dispose();
        _tables.Dispose();
        _reconArgs.Dispose();
        _scratch?.Dispose();
        _lfHdr.Dispose();
        _lfTable.Dispose();
        _lfArgs.Dispose();
    }
}
