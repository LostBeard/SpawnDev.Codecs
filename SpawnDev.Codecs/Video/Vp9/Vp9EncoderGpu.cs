// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt). See NOTICE.md for upstream attributions.
//
// VP9 encoder with inter frames (Plans/PLAN-VP9-Encoder-Inter-Frames.md). The host is a pure coordinator: it keeps
// the reference frame on the accelerator and dispatches the kernel chain; nothing but the encoded length (a scalar)
// crosses to the host.
//
// Key frames go through Vp9KeyframeEncoderGpu, whose reconstruction is written straight into this encoder's frame
// buffer pool. Inter frames: the encoder writes the full decoder's own layouts and runs the decoder's reconstruction
// kernels on them, so the reference it keeps is what any decoder reconstructs (see Vp9InterEncodeKernels.cs):
//
//   PlanePadKernel                       tight source planes -> the source buffer (frame buffer layout)
//   Vp9MotionSearchKernels               the motion search: pyramid, full-pel MV per leaf (before the setup), then
//                                        sub-pel refinement and the merge with the neighbours' MVs (after it)
//   Vp9InterEncodeKernels.SetupKernel    mode info, grid, block table of every leaf
//   Vp9ReconKernels.InterPredictKernel   the prediction, into the new frame buffer         (the decoder's kernel)
//   Vp9InterEncodeKernels.ForwardKernel  residual, forward DCT, quantizer: levels, dq, eobs, skip
//   Vp9ReconKernels.InterResidualKernel  the residual added in place = the reconstruction   (the decoder's kernel)
//   Vp9InterHeaderKernels                compressed header, then (after the tile) the uncompressed header
//   Vp9InterEntropy.StepKernel           the tile, in bounded resumable launches
//   Vp9FrameAssembleKernel               the frame, into the caller's output

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Vp9.Decoder;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>One encoded VP9 frame: its length in the output and whether it is a key frame (a sync sample).</summary>
public readonly record struct Vp9EncodedFrame(int Length, bool IsKeyFrame);

/// <summary>
/// VP9 encoder for a sequence of frames: a key frame, then inter frames predicted from the previous frame, with a key
/// frame every <see cref="KeyFrameInterval"/> frames. Fixed quantizer. Profile 0, 4:2:0, 8 bits.
/// </summary>
/// <remarks>Not thread-safe: one in-flight encode per instance; the frames of one stream go to one instance, in order.</remarks>
public sealed class Vp9EncoderGpu : IDisposable
{
    private readonly Accelerator _acc;
    private readonly Vp9KeyframeEncoderGpu _key;
    private readonly PlanePadKernel _pad;
    private readonly Vp9FrameAssembleKernel _assemble;
    private readonly Vp9FrameBuffersGpu _frames;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _tables, _args, _entropyArgs, _state;
    /// <summary>The decoder's own frame contexts (four saved + the current one), its compressed header reader and its
    /// backward adaptation: the probabilities the encoder codes with are by construction the decoder's.</summary>
    private readonly Vp9FrameContextsGpu _contexts;
    /// <summary>MV maps of two frames (the decoder's prev_frame_mvs layout): the one being written, the previous one.</summary>
    private MemoryBuffer1D<int, Stride1D.Dense>? _maps;
    private int[] _entropyArgsHost = new int[Vp9EArg.Count];
    private int _miCount, _mapCur;
    /// <summary>The previous frame of the sequence: -1 none, 0 key, 1 inter.</summary>
    private int _lastFrameType = -1;
    private readonly MemoryBuffer1D<byte, Stride1D.Dense> _compressedHeader, _uncompressedHeader;
    private readonly MemoryBuffer1D<long, Stride1D.Dense> _compressedLen, _uncompressedLen, _tileLen, _frameLen;
    private readonly StageBuffer<byte> _stageY, _stageU, _stageV, _stageOut;

    // per picture size
    private int _width = -1, _height = -1;
    private MemoryBuffer1D<byte, Stride1D.Dense>? _src, _tile, _half, _quarter, _leafPred;
    private MemoryBuffer1D<int, Stride1D.Dense>? _mi, _sb, _dq, _scratch, _mvs, _mvsB;
    private bool _mvsSet;
    private MemoryBuffer1D<short, Stride1D.Dense>? _levels;
    private int _dqStride;

    private Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>> _setup = null!;
    private Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>> _predict = null!;
    private Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<short>,
        ArrayView<ushort>> _forward = null!;
    private Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>> _residual = null!;
    private Action<Index1D, ArrayView<byte>, ArrayView<long>, ArrayView<short>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>,
        ArrayView<byte>, ArrayView<ushort>, ArrayView<int>, ArrayView<int>, int> _entropy = null!;
    // (the entropy step's symbol counts are in its state buffer: Vp9InterEntropy.Counts)
    private Action<Index1D, ArrayView<byte>, ArrayView<byte>, int, int, int, int, int, int> _downsample = null!;
    private Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>> _fullPel = null!;
    private Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>,
        ArrayView<int>, int> _refine = null!;
    private Action<Index1D, ArrayView<byte>, ArrayView<long>> _compressed = null!;
    private Action<Index1D, ArrayView<byte>, ArrayView<long>, ArrayView<long>, int, int, int> _uncompressed = null!;
    private Action<Index1D, ArrayView<byte>, ArrayView<byte>, int, int, int, int, int, int, long, long, long>? _pack;
    private Task? _kernelsLoad;

    /// <summary>The frame buffer (0 or 1) holding the last encoded frame's reconstruction; -1 before the first frame.</summary>
    private int _refBuffer = -1;
    private int _sinceKey;

    /// <summary>
    /// A key frame at least every this many frames (the first frame always is one). 0 = only when asked for
    /// (<c>forceKeyFrame</c>) or when the picture size changes.
    /// </summary>
    public int KeyFrameInterval { get; set; } = 128;

    /// <summary>
    /// Inter frames search each block's motion in the previous frame (the default). Off: every block predicts from the
    /// same position (zero motion), which is all a still picture needs.
    /// </summary>
    public bool MotionSearch { get; set; } = true;

    /// <summary>
    /// Tests / diagnostics: the MV of every 16x16 position of the next inter frames instead of the motion search's,
    /// row-major over <c>((width + 15) / 16) x ((height + 15) / 16)</c> positions (libvpx int_mv: row in the low 16
    /// bits, col in the high, in 1/8 pixel, even components).
    /// </summary>
    internal int[]? DebugMvs { get; set; }

    /// <summary>
    /// Inter frames' quantizer, in 1/128 of a step: a coefficient below <see cref="InterZbinQ7"/> is dropped, the rest
    /// are rounded with <see cref="InterRoundQ7"/> (64, 64 = round to nearest; libvpx: 80 to 84, 48).
    /// 100 / 48 is measured (DemoConsole vp9-encode-bench, 2026-11-12, two ComfyUI clips, 48 frames, q 20 to 130):
    /// against round to nearest the same luma PSNR costs 33% (512x512) and 39% (832x480) fewer bytes, the same RGB
    /// PSNR 23% and 28% fewer. Round to nearest spends bits re-coding the previous frame's quantization noise. Larger
    /// dead zones (up to 120) gain nothing more in luma and lose in chroma; 86 / 42 gains 30% and 35%.
    /// </summary>
    internal int InterZbinQ7 { get; set; } = 100;
    /// <summary>See <see cref="InterZbinQ7"/>.</summary>
    internal int InterRoundQ7 { get; set; } = 48;

    /// <summary>
    /// Inter frames adapt their probabilities backward (the default): each frame starts from the context the frame
    /// before left (libvpx's adaptation of the probabilities to the symbols that frame coded) and its MV candidates
    /// include the previous frame's MVs. Off: error_resilient_mode, every frame coded with the default probabilities
    /// (a lost frame then costs no later frame its probabilities).
    /// </summary>
    internal bool BackwardAdaptation { get; set; } = true;

    /// <summary>An encoder on <paramref name="accelerator"/>.</summary>
    public Vp9EncoderGpu(Accelerator accelerator)
    {
        _acc = accelerator ?? throw new ArgumentNullException(nameof(accelerator));
        _key = new Vp9KeyframeEncoderGpu(_acc);
        _pad = new PlanePadKernel(_acc);
        _assemble = new Vp9FrameAssembleKernel(_acc);
        _frames = new Vp9FrameBuffersGpu(_acc);
        _tables = _acc.Allocate1D(Vp9T.Packed);
        // the frame contexts (the defaults until a frame adapts them) are the decoder's: _contexts
        _contexts = new Vp9FrameContextsGpu(_acc);
        _args = _acc.Allocate1D<int>(Vp9MArg.Count);
        _entropyArgs = _acc.Allocate1D<int>(Vp9EArg.Count);
        _state = _acc.Allocate1D<int>(Vp9InterEntropy.StateInts);
        _compressedHeader = _acc.Allocate1D<byte>(128);
        _uncompressedHeader = _acc.Allocate1D<byte>(32);
        _compressedLen = _acc.Allocate1D<long>(1);
        _uncompressedLen = _acc.Allocate1D<long>(1);
        _tileLen = _acc.Allocate1D<long>(1);
        _frameLen = _acc.Allocate1D<long>(1);
        _stageY = new StageBuffer<byte>(_acc);
        _stageU = new StageBuffer<byte>(_acc);
        _stageV = new StageBuffer<byte>(_acc);
        _stageOut = new StageBuffer<byte>(_acc);
    }

    /// <summary>The output bytes one encoded frame of a <paramref name="width"/> x <paramref name="height"/> picture can need.</summary>
    public static int GetMaxEncodedSize(int width, int height) => Vp9KeyframeEncoderGpu.GetMaxEncodedSize(width, height);

    /// <summary>The next frame starts a new sequence (a key frame).</summary>
    public void Reset() => _refBuffer = -1;

    private Task EnsureKernelsAsync()
    {
        if (_kernelsLoad == null || _kernelsLoad.IsFaulted || _kernelsLoad.IsCanceled) _kernelsLoad = LoadKernelsAsync();
        return _kernelsLoad;
    }

    private async Task LoadKernelsAsync()
    {
        _setup = await _acc.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>>(
            Vp9InterEncodeKernels.SetupKernel);
        _predict = await _acc.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>>(
            Vp9ReconKernels.InterPredictKernel);
        _forward = await _acc.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>,
            ArrayView<int>, ArrayView<short>, ArrayView<ushort>>(Vp9InterEncodeKernels.ForwardKernel);
        _residual = await _acc.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>,
            ArrayView<int>, ArrayView<int>>(Vp9ReconKernels.InterResidualKernel);
        _entropy = await _acc.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<byte>, ArrayView<long>, ArrayView<short>, ArrayView<int>, ArrayView<int>,
            ArrayView<int>, ArrayView<int>, ArrayView<byte>, ArrayView<ushort>, ArrayView<int>, ArrayView<int>, int>(Vp9InterEntropy.StepKernel);
        _downsample = await _acc.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<byte>, ArrayView<byte>, int, int, int, int, int, int>(
            Vp9MotionSearchKernels.DownsampleKernel);
        _fullPel = await _acc.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<int>,
            ArrayView<int>>(Vp9MotionSearchKernels.FullPelKernel);
        _refine = await _acc.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>,
            ArrayView<int>, ArrayView<int>, ArrayView<int>, int>(Vp9MotionSearchKernels.RefineKernel);
        _compressed = await _acc.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<byte>, ArrayView<long>>(Vp9InterHeaderKernels.CompressedHeaderKernel);
        await _contexts.LoadKernelsAsync();
        _uncompressed = await _acc.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<byte>, ArrayView<long>, ArrayView<long>, int, int, int>(
            Vp9InterHeaderKernels.UncompressedHeaderKernel);
    }

    /// <summary>
    /// The per-size buffers. Called at the start of an encode (every earlier one has ended with a wait, so replacing a
    /// buffer cannot free memory a pending launch reads). A new size starts a new sequence.
    /// </summary>
    private void EnsureBuffers(int width, int height)
    {
        if (width == _width && height == _height) return;
        if (((width + 7) >> 3) > Vp9InterEntropy.MaxMiCols)
            throw new NotSupportedException($"VP9 encode: pictures wider than {Vp9InterEntropy.MaxMiCols * 8} pixels are not supported.");
        _src?.Dispose(); _tile?.Dispose(); _mi?.Dispose(); _sb?.Dispose(); _dq?.Dispose(); _scratch?.Dispose(); _levels?.Dispose(); _mvs?.Dispose();
        _mvsB?.Dispose(); _half?.Dispose(); _quarter?.Dispose(); _leafPred?.Dispose();
        _width = width;
        _height = height;
        _refBuffer = -1;
        var layout = Vp9PlaneLayout.For(width, height, 1, 1);
        _frames.Ensure(0, width, height, 1, 1);
        _frames.Ensure(1, width, height, 1, 1);
        int miCols = (width + 7) >> 3, miRows = (height + 7) >> 3;
        int sbCols = (miCols + 7) >> 3, sbRows = (miRows + 7) >> 3, sbCount = sbCols * sbRows;
        int miCount = miCols * miRows;
        _miCount = miCount;
        _maps?.Dispose();
        _maps = _acc.Allocate1D<int>(2L * miCount * Vp9PLayout.MvStride);
        _maps.MemSetToZero();
        _mapCur = 0;
        _dqStride = 4096 + 2 * 1024;   // per superblock: 4096 luma, 1024 per chroma plane (4:2:0)
        _src = _acc.Allocate1D<byte>(layout.Bytes);
        _mi = _acc.Allocate1D<int>((long)miCount * (Vp9Mi.Stride + 1));
        _sb = _acc.Allocate1D<int>((long)sbCount * Vp9PLayout.SbStride);
        _dq = _acc.Allocate1D<int>((long)sbCount * _dqStride);
        _levels = _acc.Allocate1D<short>((long)sbCount * _dqStride);
        // the decoder kernels' scratch: per prediction unit, per inter transform block (Vp9RArg.ResidualScratchAt)
        long predUnits = PredUnits(sbCols, sbRows);
        _scratch = _acc.Allocate1D<int>(Math.Max(predUnits * Vp9InterPred.RegionScratchInts, ((long)sbCount * _dqStride >> 2) * 5));
        _tile = _acc.Allocate1D<byte>(GetMaxEncodedSize(width, height));
        int leafCount = ((miCols + 1) >> 1) * ((miRows + 1) >> 1);
        _mvs = _acc.Allocate1D<int>(leafCount);
        _mvs.MemSetToZero();
        _mvsSet = false;
        // the motion search: the luma pyramid (source then reference, tightly packed, per level), a second MV buffer
        // (each refinement pass reads one and writes the other), one 16x16 block of prediction per leaf
        int halfW = (width + 1) >> 1, halfH = (height + 1) >> 1;
        _mvsB = _acc.Allocate1D<int>(leafCount);
        _half = _acc.Allocate1D<byte>(2L * halfW * halfH);
        _quarter = _acc.Allocate1D<byte>(2L * ((halfW + 1) >> 1) * ((halfH + 1) >> 1));
        _leafPred = _acc.Allocate1D<byte>((long)leafCount * Vp9MotionSearchKernels.LeafBytes);

        // the entropy pass's arguments depend only on the size
        var e = _entropyArgsHost = new int[Vp9EArg.Count];
        e[Vp9PArg.MiCols] = miCols;
        e[Vp9PArg.MiRows] = miRows;
        e[Vp9PArg.SbCols] = sbCols;
        e[Vp9PArg.GridOffset] = miCount * Vp9Mi.Stride;
        e[Vp9PArg.DqStride] = _dqStride;
        e[Vp9PArg.DqU] = 4096;
        e[Vp9PArg.DqV] = 4096 + 1024;
        Vp9EArg.BuildClassTokens(e);
        _entropyArgs.View.CopyFromCPU(e);   // frame metadata scalars
    }

    private static int PredUnits(int sbCols, int sbRows) => sbCols * 16 * (sbRows * 16) + 2 * ((sbCols * 16) >> 1) * ((sbRows * 16) >> 1);

    /// <summary>
    /// Encode the next frame of the sequence from tightly packed 4:2:0 planes that are already on the accelerator:
    /// a key frame when it is the first, when <paramref name="forceKeyFrame"/> is set, when the size changed or after
    /// <see cref="KeyFrameInterval"/> frames; else an inter frame predicted from the previous frame. The frame is
    /// assembled straight into <paramref name="output"/> (at least <see cref="GetMaxEncodedSize"/> bytes); only its
    /// length crosses to the host.
    /// </summary>
    public async Task<Vp9EncodedFrame> EncodeFrameAsync(
        ArrayView1D<byte, Stride1D.Dense> yPlane, ArrayView1D<byte, Stride1D.Dense> uPlane, ArrayView1D<byte, Stride1D.Dense> vPlane,
        int width, int height, ArrayView1D<byte, Stride1D.Dense> output, int baseQIndex = 30, bool forceKeyFrame = false)
    {
        if (width <= 0) throw new ArgumentOutOfRangeException(nameof(width));
        if (height <= 0) throw new ArgumentOutOfRangeException(nameof(height));
        if (baseQIndex <= 0 || baseQIndex > 255) throw new ArgumentOutOfRangeException(nameof(baseQIndex), "baseQIndex must be in [1, 255].");
        int uvWidth = (width + 1) / 2, uvHeight = (height + 1) / 2;
        if (yPlane.Length < (long)width * height) throw new ArgumentException("yPlane too short.", nameof(yPlane));
        if (uPlane.Length < (long)uvWidth * uvHeight) throw new ArgumentException("uPlane too short.", nameof(uPlane));
        if (vPlane.Length < (long)uvWidth * uvHeight) throw new ArgumentException("vPlane too short.", nameof(vPlane));
        if (output.Length < GetMaxEncodedSize(width, height))
            throw new ArgumentException($"output must be at least GetMaxEncodedSize({width}, {height}) = {GetMaxEncodedSize(width, height)} bytes.", nameof(output));

        await EnsureKernelsAsync();
        EnsureBuffers(width, height);
        var layout = Vp9PlaneLayout.For(width, height, 1, 1);
        long yBytes = (long)layout.YStride * layout.YRows, uvBytes = (long)layout.UvStride * layout.UvRows;
        bool key = _refBuffer < 0 || forceKeyFrame || (KeyFrameInterval > 0 && _sinceKey >= KeyFrameInterval);
        int cur = _refBuffer < 0 ? 0 : 1 - _refBuffer;
        long curBase = _frames.Offset(cur);
        int length;
        if (key)
        {
            // the key frame's reconstruction goes straight into frame buffer cur (same plane geometry)
            length = await _key.EncodeKeyFrameIntoAsync(yPlane, uPlane, vPlane, width, height, output, baseQIndex,
                _frames.Pool.SubView(curBase, yBytes), _frames.Pool.SubView(curBase + layout.UOffset, uvBytes),
                _frames.Pool.SubView(curBase + layout.VOffset, uvBytes));
            _sinceKey = 1;
            // A key frame resets every saved context to the defaults. Its header has refresh_frame_context = 0, so
            // what a decoder adapts from the key frame's symbols is not saved: the next frame starts from the defaults.
            _contexts.BeginFrame(FrameInfo(key: true, adapt: false));
            _lastFrameType = 0;
        }
        else
        {
            length = await EncodeInterAsync(yPlane, uPlane, vPlane, width, height, output, baseQIndex, layout, cur);
            _sinceKey++;
        }
        _refBuffer = cur;
        return new Vp9EncodedFrame(length, key);
    }

    private async Task<int> EncodeInterAsync(
        ArrayView1D<byte, Stride1D.Dense> yPlane, ArrayView1D<byte, Stride1D.Dense> uPlane, ArrayView1D<byte, Stride1D.Dense> vPlane,
        int width, int height, ArrayView1D<byte, Stride1D.Dense> output, int baseQIndex, Vp9PlaneLayout layout, int cur)
    {
        int miCols = (width + 7) >> 3, miRows = (height + 7) >> 3;
        int sbCols = (miCols + 7) >> 3, sbRows = (miRows + 7) >> 3;
        int uvWidth = (width + 1) / 2, uvHeight = (height + 1) / 2;
        long yBytes = (long)layout.YStride * layout.YRows, uvBytes = (long)layout.UvStride * layout.UvRows;
        var src = _src!.View;

        // the bool encoders' carry reaches back into bytes already written: their targets start at zero
        _compressedHeader.View.MemSetToZero();
        _uncompressedHeader.View.MemSetToZero();
        _tile!.View.MemSetToZero();
        _compressedLen.View.MemSetToZero();
        _uncompressedLen.View.MemSetToZero();
        _tileLen.View.MemSetToZero();
        _frameLen.View.MemSetToZero();
        _state.View.MemSetToZero();   // the entropy walk and every context start at zero

        // source planes -> the source buffer (a frame buffer's layout), zero padded
        _pad.Run(yPlane, 0, width, width, height, src.SubView(0, yBytes), layout.YStride, layout.YRows);
        _pad.Run(uPlane, 0, uvWidth, uvWidth, uvHeight, src.SubView(layout.UOffset, uvBytes), layout.UvStride, layout.UvRows);
        _pad.Run(vPlane, 0, uvWidth, uvWidth, uvHeight, src.SubView(layout.VOffset, uvBytes), layout.UvStride, layout.UvRows);

        // the decoder's reconstruction arguments (Vp9DecodePipeline.Reconstruct) + the encoder's
        var a = new int[Vp9MArg.Count];
        a[Vp9RArg.MiCols] = miCols;
        a[Vp9RArg.MiRows] = miRows;
        a[Vp9RArg.SsX] = 1;
        a[Vp9RArg.SsY] = 1;
        a[Vp9RArg.SbCols] = sbCols;
        a[Vp9RArg.GridOffset] = miCols * miRows * Vp9Mi.Stride;
        a[Vp9RArg.DqStride] = _dqStride;
        a[Vp9RArg.DqU] = 4096;
        a[Vp9RArg.DqV] = 4096 + 1024;
        long curBase = _frames.Offset(cur), refBase = _frames.Offset(_refBuffer);
        var sf = Vp9InterPred.SetupScaleFactors(width, height, width, height);
        for (int p = 0; p < 3; p++)
        {
            a[Vp9RArg.Cur + p * 4] = checked((int)(curBase + layout.PlaneOffset(p)));
            a[Vp9RArg.Cur + p * 4 + 1] = layout.Stride(p);
            a[Vp9RArg.Cur + p * 4 + 2] = p == 0 ? layout.AlignedWidth : layout.UvAlignedWidth;
            a[Vp9RArg.Cur + p * 4 + 3] = p == 0 ? layout.AlignedHeight : layout.UvAlignedHeight;
            a[Vp9IArg.SrcY + p] = checked((int)layout.PlaneOffset(p));
            for (int r = 0; r < 3; r++)   // LAST, GOLDEN, ALTREF: all the previous frame
            {
                int at = Vp9RArg.Ref + (r * 3 + p) * 4;
                a[at] = checked((int)(refBase + layout.PlaneOffset(p)));
                a[at + 1] = layout.Stride(p);
                a[at + 2] = layout.CropWidth(p);
                a[at + 3] = layout.CropHeight(p);
            }
        }
        for (int r = 0; r < 3; r++)
        {
            a[Vp9RArg.Scale + r * 4] = sf.XScaleFp;
            a[Vp9RArg.Scale + r * 4 + 1] = sf.YScaleFp;
            a[Vp9RArg.Scale + r * 4 + 2] = sf.XStepQ4;
            a[Vp9RArg.Scale + r * 4 + 3] = sf.YStepQ4;
        }
        a[Vp9IArg.DqYDc] = Vp9Dequantizer.DcQuant(baseQIndex, 0);
        a[Vp9IArg.DqYAc] = Vp9Dequantizer.AcQuant(baseQIndex, 0);
        a[Vp9IArg.DqUvDc] = Vp9Dequantizer.DcQuant(baseQIndex, 0);
        a[Vp9IArg.DqUvAc] = Vp9Dequantizer.AcQuant(baseQIndex, 0);
        a[Vp9IArg.ZbinQ7] = InterZbinQ7;
        a[Vp9IArg.RoundQ7] = InterRoundQ7;
        int hw = (width + 1) >> 1, hh = (height + 1) >> 1, qw = (hw + 1) >> 1, qh = (hh + 1) >> 1;
        a[Vp9MArg.Width] = width;
        a[Vp9MArg.Height] = height;
        a[Vp9MArg.HalfW] = hw;
        a[Vp9MArg.HalfH] = hh;
        a[Vp9MArg.QuarterW] = qw;
        a[Vp9MArg.QuarterH] = qh;
        a[Vp9MArg.Stride] = layout.YStride;
        a[Vp9MArg.SrcAt] = 0;
        a[Vp9MArg.RefAt] = checked((int)refBase);
        // costs in SAD units of a 16x16 block, scaled with the quantizer step (a bit is worth about one step of error)
        int acStep = a[Vp9IArg.DqYAc];
        a[Vp9MArg.LengthCost] = 1 + acStep / 32;
        a[Vp9MArg.NewMvCost] = acStep * 3 / 2;
        _args.View.CopyFromCPU(a);   // frame metadata scalars

        int leaves = ((miCols + 1) >> 1) * ((miRows + 1) >> 1);
        if (DebugMvs != null)
        {
            if (DebugMvs.Length != leaves) throw new ArgumentException($"DebugMvs must hold {leaves} MVs.");
            _mvs!.View.CopyFromCPU(DebugMvs);   // tests only
            _mvsSet = true;
        }
        else if (_mvsSet && !MotionSearch)
        {
            _mvs!.MemSetToZero();
            _mvsSet = false;
        }
        int residUnits = miCols * 2 * (miRows * 2) + 2 * ((miCols * 2) >> 1) * ((miRows * 2) >> 1);
        var pool = _frames.Pool;
        bool search = MotionSearch && DebugMvs == null;
        if (search)
        {
            // the pyramid of the source and of the reference, then every leaf's full-pel MV
            _downsample(hw * hh, src, _half!.View, 0, layout.YStride, width, height, 0, hw);
            _downsample(hw * hh, pool, _half.View, checked((int)refBase), layout.YStride, width, height, hw * hh, hw);
            _downsample(qw * qh, _half.View, _quarter!.View, 0, hw, hw, hh, 0, qw);
            _downsample(qw * qh, _half.View, _quarter.View, hw * hh, hw, hw, hh, qw * qh, qw);
            _fullPel(leaves, _quarter.View, _half.View, src, pool, _args.View, _mvs!.View);
            _mvsSet = true;
        }
        _setup(leaves, _args.View, _mi!.View, _sb!.View, _mvs!.View);
        if (search)
        {
            // half-pel, quarter-pel, then two passes of merging with the neighbours' MVs; the last pass leaves the
            // MVs in the mode info
            _refine(leaves, pool, src, _leafPred!.View, _args.View, _mi.View, _mvs.View, _mvsB!.View, _scratch!.View, 0);
            _refine(leaves, pool, src, _leafPred.View, _args.View, _mi.View, _mvsB.View, _mvs.View, _scratch.View, 1);
            _refine(leaves, pool, src, _leafPred.View, _args.View, _mi.View, _mvs.View, _mvsB.View, _scratch.View, 2);
            _refine(leaves, pool, src, _leafPred.View, _args.View, _mi.View, _mvsB.View, _mvs.View, _scratch.View, 2);
        }
        _predict(PredUnits(sbCols, sbRows), pool, _args.View, _mi.View, _scratch!.View);
        _forward(leaves, pool, src, _args.View, _mi.View, _sb.View, _dq!.View, _levels!.View, _key.UshortConsts);
        _residual(residUnits, pool, _args.View, _tables.View, _mi.View, _sb.View, _dq.View, _scratch.View);

        // The frame's probabilities: the saved context (or the defaults), then whatever the compressed header changes,
        // read back by the decoder's own reader (it also leaves the tx mode the adaptation needs).
        bool adapt = BackwardAdaptation;
        var info = FrameInfo(key: false, adapt);
        _contexts.BeginFrame(info);
        _compressed(1, _compressedHeader.View, _compressedLen.View);
        _contexts.ReadCompressedHeader(_compressedHeader.View, 0, info);
        var frameContext = _contexts.Contexts.SubView((long)Vp9Ctx.Current * Vp9Ctx.Bytes, Vp9Ctx.Bytes);
        // the entropy pass's per-frame arguments: counting, the MV maps (this frame's and the previous frame's)
        var e = _entropyArgsHost;
        _mapCur ^= 1;
        e[Vp9PArg.CountsOn] = adapt ? 1 : 0;
        // vp9_decode_frame: the previous frame's MVs are candidates when it was an inter frame of the same size, shown,
        // and this frame is not error resilient (every frame here is shown, a new size starts with a key frame)
        e[Vp9PArg.UsePrevMvs] = adapt && _lastFrameType == 1 ? 1 : 0;
        e[Vp9PArg.CurMvOffset] = _mapCur * _miCount * Vp9PLayout.MvStride;
        e[Vp9PArg.PrevMvOffset] = (1 - _mapCur) * _miCount * Vp9PLayout.MvStride;
        _entropyArgs.View.CopyFromCPU(e);   // frame metadata scalars
        int slots = sbCols * sbRows * 16;
        int steps = (slots + Vp9InterEntropy.LeavesPerLaunch - 1) / Vp9InterEntropy.LeavesPerLaunch;
        var pacer = new LaunchPacer(_acc);
        for (int s = 0; s < steps; s++)
        {
            _entropy(1, _tile.View, _tileLen.View, _levels.View, _mi.View, _entropyArgs.View, frameContext, _tables.View,
                _key.ByteConsts, _key.UshortConsts, _state.View, _maps!.View, Vp9InterEntropy.LeavesPerLaunch);
            await pacer.LaunchedAsync();
        }
        // libvpx's backward adaptation from the frame's counts, then the save (both do nothing for an error resilient frame)
        _contexts.Adapt(info, _state.View.SubView(Vp9InterEntropy.Counts, Vp9Cnt.Size), 1);
        _contexts.EndFrame(info);
        _lastFrameType = 1;
        _uncompressed(1, _uncompressedHeader.View, _uncompressedLen.View, _compressedLen.View, width, baseQIndex, adapt ? 0 : 1);
        _assemble.Run(_uncompressedHeader.View, _compressedHeader.View, _tile.View, output, _frameLen.View,
            _uncompressedLen.View, _compressedLen.View, _tileLen.View);
        await _acc.SynchronizeAsync();
        long length = (await _frameLen.CopyToHostAsync())[0];
        if (length < 0) throw new InvalidOperationException("VP9 encode: the tile outgrew its buffer; the frame was not assembled.");
        return (int)length;
    }

    /// <summary>
    /// What the decoder's frame context class needs to know of a frame this encoder writes (the fields its
    /// BeginFrame / ReadCompressedHeader / Adapt / EndFrame read; the header kernels write the same choices).
    /// </summary>
    private Vp9FrameInfo FrameInfo(bool key, bool adapt) => new()
    {
        FrameType = key ? 0 : 1,
        LastFrameType = _lastFrameType == 0 ? 0 : 1,
        ShowFrame = true,
        ErrorResilient = !key && !adapt,
        FrameParallel = !key && !adapt,
        RefreshFrameContext = !key && adapt,
        FrameContextIdx = 0,
        // key frames and error resilient frames reset every saved context (vp9_setup_past_independence)
        ContextReset = key || !adapt ? Vp9ContextReset.All : Vp9ContextReset.None,
        InterpFilter = Vp9Interp.EightTap,
        AllowHighPrecisionMv = false,
        UncompressedHeaderBytes = 0,
        FirstPartitionSize = (int)_compressedHeader.Length,   // the reader's bound: the whole (zeroed) buffer
        LoopFilter = new Vp9LoopFilterInfo(0, 0, false, false, new int[4], new int[2]),
        Segmentation = new Vp9SegmentationInfo(false, false, false, false, false, new byte[7], new byte[3], new int[8], new int[8, 4]),
        Dequant = new short[8, 4],
    };

    /// <summary>
    /// Encode the next frame from managed, tightly packed 4:2:0 planes; returns the frame's bytes (a terminal CPU
    /// sink: desktop and small inputs).
    /// </summary>
    public async Task<(byte[] Bytes, bool IsKeyFrame)> EncodeFrameAsync(
        ReadOnlyMemory<byte> yPlane, ReadOnlyMemory<byte> uPlane, ReadOnlyMemory<byte> vPlane,
        int width, int height, int baseQIndex = 30, bool forceKeyFrame = false)
    {
        if (width <= 0) throw new ArgumentOutOfRangeException(nameof(width));
        if (height <= 0) throw new ArgumentOutOfRangeException(nameof(height));
        int uvWidth = (width + 1) / 2, uvHeight = (height + 1) / 2;
        DeviceTransfer.UploadPlane(_stageY, yPlane, width, width, height, nameof(yPlane));
        DeviceTransfer.UploadPlane(_stageU, uPlane, uvWidth, uvWidth, uvHeight, nameof(uPlane));
        DeviceTransfer.UploadPlane(_stageV, vPlane, uvWidth, uvWidth, uvHeight, nameof(vPlane));
        _stageOut.Ensure(GetMaxEncodedSize(width, height));
        var frame = await EncodeFrameAsync(_stageY.View, _stageU.View, _stageV.View, width, height, _stageOut.View, baseQIndex, forceKeyFrame);
        return (await DeviceTransfer.DownloadAsync(_stageOut.View, frame.Length), frame.IsKeyFrame);
    }

    /// <summary>
    /// Tests / diagnostics: the last encoded frame's reconstruction, the cropped planes (Y, U, V, tight): what a decoder
    /// shows for that frame.
    /// </summary>
    internal async Task<byte[]> ReadReconAsync()
    {
        if (_refBuffer < 0) throw new InvalidOperationException("No frame has been encoded.");
        var l = _frames.Layout(_refBuffer);
        long b = _frames.Offset(_refBuffer);
        int uvW = l.UvCropWidth, uvH = l.UvCropHeight;
        long total = (long)l.Width * l.Height + 2L * uvW * uvH;
        using var dst = _acc.Allocate1D<byte>(total);
        var pack = _pack ??= await _acc.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<byte>, ArrayView<byte>, int, int, int, int, int, int, long, long, long>(
            Vp9OutputKernels.PackPlanesKernel);
        pack((int)total, _frames.Pool, dst.View, l.Width, l.Height, uvW, uvH, l.YStride, l.UvStride, b, b + l.UOffset, b + l.VOffset);
        return await dst.View.CopyToHostAsync();
    }

    /// <summary>
    /// Tests / diagnostics: the MV of every 16x16 position of the last inter frame (the layout of
    /// <see cref="DebugMvs"/>): what the motion search chose. A few ints per frame, tests only.
    /// </summary>
    internal async Task<int[]> ReadMvsAsync()
    {
        if (_mvs == null) throw new InvalidOperationException("No frame has been encoded.");
        return await _mvs.View.CopyToHostAsync();
    }

    /// <summary>Tests: the last inter frame's symbol counts (the decoder's FRAME_COUNTS layout, <see cref="Vp9Cnt"/>).</summary>
    internal async Task<int[]> ReadCountsAsync() => await _state.View.SubView(Vp9InterEntropy.Counts, Vp9Cnt.Size).CopyToHostAsync();

    /// <summary>Tests: the frame contexts (four saved + the current one, <see cref="Vp9Ctx"/>).</summary>
    internal Task<int[]> ReadContextsAsync() => _contexts.ReadContextsAsync();

    /// <summary>Release every resource the encoder owns.</summary>
    public void Dispose()
    {
        _key.Dispose();
        _pad.Dispose();
        _assemble.Dispose();
        _frames.Dispose();
        _tables.Dispose(); _contexts.Dispose(); _args.Dispose(); _entropyArgs.Dispose(); _state.Dispose(); _maps?.Dispose();
        _compressedHeader.Dispose(); _uncompressedHeader.Dispose();
        _compressedLen.Dispose(); _uncompressedLen.Dispose(); _tileLen.Dispose(); _frameLen.Dispose();
        _src?.Dispose(); _tile?.Dispose(); _mi?.Dispose(); _sb?.Dispose(); _dq?.Dispose(); _scratch?.Dispose(); _levels?.Dispose(); _mvs?.Dispose();
        _mvsB?.Dispose(); _half?.Dispose(); _quarter?.Dispose(); _leafPred?.Dispose();
        _stageY.Dispose(); _stageU.Dispose(); _stageV.Dispose(); _stageOut.Dispose();
    }
}
