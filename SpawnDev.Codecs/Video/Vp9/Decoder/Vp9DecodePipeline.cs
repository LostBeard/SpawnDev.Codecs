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
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _tables, _reconArgs;
    private MemoryBuffer1D<int, Stride1D.Dense>? _scratch;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>> _recon;
    private int _prevBuffer = -1;

    public Vp9DecodePipeline(Accelerator accelerator)
    {
        _acc = accelerator ?? throw new ArgumentNullException(nameof(accelerator));
        _contexts = new Vp9FrameContextsGpu(_acc);
        _parser = new Vp9TileParserGpu(_acc);
        _frames = new Vp9FrameBuffersGpu(_acc);
        _tables = _acc.Allocate1D(Vp9T.Packed);
        _reconArgs = _acc.Allocate1D<int>(Vp9RArg.Count);
        _recon = _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>,
            ArrayView<int>, ArrayView<int>>(Vp9ReconKernels.ReconKernel);
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
        var h = _reader.ReadFrame(frame.Span);
        if (h.ShowExistingFrame)
        {
            _reader.FinishFrame();
            return h;
        }
        _frames.Ensure(h.NewBuffer, h.Width, h.Height, h.SubsamplingX, h.SubsamplingY);
        _contexts.BeginFrame(h);
        _contexts.ReadCompressedHeader(frameView, frameOffset, h);
        _parser.Parse(frameView, frameOffset, frame.Span, h, _contexts, _prevBuffer);
        Reconstruct(h);
        if (afterRecon != null) await afterRecon(h);
        // loop filter: next (Vp9Lpf)
        _contexts.Adapt(h, _parser.Counts, _parser.TileCols);
        _contexts.EndFrame(h);
        _parser.EndFrame(h);
        _reader.FinishFrame();
        _prevBuffer = h.NewBuffer;
        return h;
    }

    private void Reconstruct(Vp9FrameInfo h)
    {
        int tileCols = 1 << h.Log2TileCols;
        long scratchInts = (long)tileCols * (Vp9RArg.Scratch + 64);
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
        _recon(tileCols, _frames.Pool, _reconArgs.View, _tables.View, _parser.ModeInfo, _parser.Superblocks, _parser.Coefficients, _scratch.View);
    }

    /// <summary>The cropped planes (Y, U, V, tight) of frame buffer <paramref name="buffer"/> (diagnostics / tests).</summary>
    internal async Task<byte[]> ReadPlanesAsync(int buffer)
    {
        var l = _frames.Layout(buffer);
        long b = _frames.Offset(buffer);
        int uvW = l.UvCropWidth, uvH = l.UvCropHeight;
        long total = (long)l.Width * l.Height + 2L * uvW * uvH;
        using var dst = _acc.Allocate1D<byte>(total);
        var pack = _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<byte>, int, int, int, int, int, int, long, long, long>(
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
    }
}
