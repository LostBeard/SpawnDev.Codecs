// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Lossy WebP encoder on the accelerator, a port of libwebp 1.6.0's VP8 encoder (methods 3 and 4: RD_OPT_BASIC) whose output
// is BYTE-IDENTICAL to libwebp's for the same pixels and settings (checked against libwebp files and against the CPU port
// Vp8LossyEncoderReference in SpawnDev.Codecs.References). The host is a pure coordinator: it uploads the RGBA picture, a few
// hundred ints of config / layout and the constant tables, dispatches the kernel chain and reads back scalars:
//
//   WebPRgbaToYuvGpu          RGBA -> YUV 4:2:0 + alpha, libwebp's import + transparent-area cleanup (unless Exact)
//   WebPLossyAnalysisKernel   one thread per macroblock: susceptibility histogram, I16 / UV mode guesses
//   WebPLossySetupKernel      one thread: k-means segmentation, quantizers / matrices / lambdas, segment probabilities
//   WebPLossyEncodeRowKernel  one thread, launches of MbsPerLaunch macroblocks: the RD token loop (serial by libwebp's design)
//   WebPLossyFinishKernel     one thread, a chain of bounded launches: final probabilities, token partition, filter
//                             strength, partition 0
//
// No launch does work that grows with the picture (a long launch stalls the desktop and trips the OS GPU watchdog): the
// serial stages resume from state kept in meta[WebPLossyEncMeta.Step ..] and the device buffers.
//   AssembleKernel            one thread per output byte: RIFF / VP8 chunk / frame tag / partitions
//
// Not yet: methods 0..2 (distortion-only refinement) and 5..6 (trellis), pictures with transparency (ALPH chunk), multiple
// token partitions. Those throw NotSupportedException rather than produce a different file.

using System.Runtime.Versioning;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Images.Yuv;
using SpawnDev.ILGPU;
using SpawnDev.SpawnJS.JSObjects;
using M = SpawnDev.Codecs.Images.WebP.Lossy.WebPLossyEncMeta;

namespace SpawnDev.Codecs.Images.WebP.Lossy;

/// <summary>Settings of the lossy encoder (libwebp WebPConfig fields; the rest keep WebPConfigInit's defaults).</summary>
public sealed record WebPLossyEncodeOptions
{
    /// <summary>0..100 (libwebp's default 75; Pillow passes 80 unless told otherwise).</summary>
    public float Quality { get; init; } = 75f;

    /// <summary>libwebp method; 3 and 4 (the default) are supported.</summary>
    public int Method { get; init; } = 4;

    /// <summary>Keep the RGB under fully transparent pixels (libwebp `exact`); false runs libwebp's transparent-area cleanup.</summary>
    public bool Exact { get; init; }
}

/// <summary>Lossy (VP8) WebP encoder running on an ILGPU accelerator, byte-identical with libwebp.</summary>
/// <remarks>Not thread-safe: one in-flight encode per instance.</remarks>
public sealed class WebPLossyEncoderGpu : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly WebPRgbaToYuvGpu _toYuv;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<byte>, ArrayView<int>> _analysis;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>> _setup;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int, int, int> _row;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<byte>>
        _finishProbas, _finishTokens, _finishHeader, _finishModes, _finishEnd;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<byte>, ArrayView<byte>> _assemble;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _tables;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _meta;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _qTable;
    private readonly StageBuffer<byte> _yuv, _work, _scratch, _stageIn, _stageOut;
    private readonly StageBuffer<int> _mbRecs, _state, _tokens;
    private long _tokenCapacity;

    private const ulong Partition0SizeLimit = ((1UL << 19) - 2048UL) << 11;

    /// <summary>Compile the kernels and upload the constant tables.</summary>
    public WebPLossyEncoderGpu(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _toYuv = new WebPRgbaToYuvGpu(accelerator);
        _analysis = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<byte>, ArrayView<int>>(WebPLossyAnalysisKernel.AnalysisKernel);
        _setup = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>>(WebPLossySetupKernel.SetupKernel);
        _row = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int, int, int>(WebPLossyEncodeRowKernel.EncodeRowKernel);
        _finishProbas = LoadFinish(accelerator, WebPLossyFinishKernel.FinishProbasKernel);
        _finishTokens = LoadFinish(accelerator, WebPLossyFinishKernel.FinishTokensKernel);
        _finishHeader = LoadFinish(accelerator, WebPLossyFinishKernel.FinishHeaderKernel);
        _finishModes = LoadFinish(accelerator, WebPLossyFinishKernel.FinishModesKernel);
        _finishEnd = LoadFinish(accelerator, WebPLossyFinishKernel.FinishEndKernel);
        _assemble = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<byte>, ArrayView<byte>>(WebPLossyFinishKernel.AssembleKernel);
        _tables = accelerator.Allocate1D<int>(WebPLossyEncTables.Size);
        _tables.View.CopyFromCPU(WebPLossyEncTables.Build());
        _meta = accelerator.Allocate1D<int>(M.Size);
        _qTable = accelerator.Allocate1D<int>(255);
        _yuv = new StageBuffer<byte>(accelerator);
        _work = new StageBuffer<byte>(accelerator);
        _scratch = new StageBuffer<byte>(accelerator);
        _stageIn = new StageBuffer<byte>(accelerator);
        _stageOut = new StageBuffer<byte>(accelerator);
        _mbRecs = new StageBuffer<int>(accelerator);
        _state = new StageBuffer<int>(accelerator);
        _tokens = new StageBuffer<int>(accelerator);
    }

    private static Action<Index1D, ArrayView<int>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<byte>> LoadFinish(
        Accelerator accelerator, Action<Index1D, ArrayView<int>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<byte>> kernel)
        => accelerator.LoadAutoGroupedStreamKernel(kernel);

    private static int Align4(int v) => (v + 3) & ~3;

    private sealed class Layout
    {
        public int Width, Height, MbW, MbH, MbCount, PredsSize, Cw, Ch;
        public int YuvU, YuvV, YuvA, YuvSize;
        public int WorkYTop, WorkScratch, WorkSize;
        public int StNz, StTopDerr, StCoeffs, StStats, StLevelCost, StScratch, StateSize;
        public int P0Cap, T1Cap;
    }

    private static Layout MakeLayout(int width, int height, long tokenCapacity)
    {
        var l = new Layout { Width = width, Height = height, MbW = (width + 15) >> 4, MbH = (height + 15) >> 4 };
        l.MbCount = l.MbW * l.MbH;
        l.Cw = (width + 1) >> 1; l.Ch = (height + 1) >> 1;
        int n = width * height, c = l.Cw * l.Ch;
        l.YuvU = Align4(n); l.YuvV = Align4(l.YuvU + c); l.YuvA = Align4(l.YuvV + c); l.YuvSize = l.YuvA + n;
        int predsW = 4 * l.MbW + 1;
        l.PredsSize = predsW * (4 * l.MbH + 1);
        l.WorkYTop = Align4(l.PredsSize); l.WorkScratch = Align4(l.WorkYTop + 2 * l.MbW * 16);
        l.WorkSize = l.WorkScratch + WebPLossyEncodeRowKernel.ByteScratchSize;
        l.StNz = 0; l.StTopDerr = l.MbW + 1; l.StCoeffs = l.StTopDerr + l.MbW * 4;
        l.StStats = l.StCoeffs + WebPLossySetupKernel.ProbaCount; l.StLevelCost = l.StStats + WebPLossySetupKernel.ProbaCount;
        l.StScratch = l.StLevelCost + WebPLossySetupKernel.LevelCostCount;
        l.StateSize = l.StScratch + WebPLossyEncodeRowKernel.IntScratchSize;
        l.P0Cap = 4096 + l.MbCount * 64;
        // A token codes at most one bit; at worst a bit costs 8 bits (probability 1/256): tokens bytes + slack.
        l.T1Cap = (int)Math.Min(int.MaxValue / 2, tokenCapacity + 1024);
        return l;
    }

    private int[] HostMeta(Layout l, WebPLossyEncodeOptions o, long tokenCapacity, int maxI4HeaderBits)
    {
        var m = new int[M.Size];
        m[M.Width] = l.Width; m[M.Height] = l.Height; m[M.MbW] = l.MbW; m[M.MbH] = l.MbH;
        m[M.Method] = o.Method;
        m[M.QualityX100] = (int)(o.Quality * 100f);
        m[M.SnsStrength] = 50; m[M.FilterStrength] = 60; m[M.FilterSharpnessCfg] = 0; m[M.FilterType] = 1; m[M.SegmentsCfg] = 4;
        m[M.ErrorDiffusion] = o.Quality <= 98f ? 1 : 0;
        m[M.MaxI4HeaderBits] = 256 * 16 * 16;
        m[M.PassMaxI4HeaderBits] = maxI4HeaderBits;
        long mbHeaderLimit = (long)256 * 510 * 8 * 1024 / l.MbCount;
        m[M.MbHeaderLimitLo] = (int)mbHeaderLimit; m[M.MbHeaderLimitHi] = (int)(mbHeaderLimit >> 32);
        m[M.TokenCapacity] = (int)tokenCapacity;
        m[M.Partition0Capacity] = l.P0Cap;
        m[M.Token1Capacity] = l.T1Cap;
        m[M.Profile] = 0;   // filter_strength 60 > 0 and filter_type 1 -> profile 0
        int maxCount = l.MbCount >> 3;
        if (maxCount < 96) maxCount = 96;
        m[M.MaxCount] = maxCount;
        m[M.MbCountDown] = maxCount;
        m[M.YuvUOff] = l.YuvU; m[M.YuvVOff] = l.YuvV;
        m[M.WorkYTopOff] = l.WorkYTop; m[M.WorkScratchOff] = l.WorkScratch;
        m[M.StNzOff] = l.StNz; m[M.StTopDerrOff] = l.StTopDerr; m[M.StCoeffsOff] = l.StCoeffs;
        m[M.StStatsOff] = l.StStats; m[M.StLevelCostOff] = l.StLevelCost; m[M.StScratchOff] = l.StScratch;
        return m;
    }

    private static void Validate(int width, int height, WebPLossyEncodeOptions o)
    {
        if (width <= 0 || width > 16383) throw new ArgumentOutOfRangeException(nameof(width));
        if (height <= 0 || height > 16383) throw new ArgumentOutOfRangeException(nameof(height));
        if (o.Quality < 0 || o.Quality > 100) throw new ArgumentOutOfRangeException(nameof(o), "Quality must be 0..100.");
        if (o.Method is not (3 or 4))
            throw new NotSupportedException($"Method {o.Method}: only methods 3 and 4 are ported to the accelerator so far.");
    }

    /// <summary>
    /// Encode tightly packed RGBA8 that is on the accelerator; the file is written to <paramref name="output"/>. Returns its
    /// length. Throws <see cref="ArgumentException"/> naming the needed size when <paramref name="output"/> is too small.
    /// </summary>
    public async Task<int> EncodeAsync(ArrayView1D<byte, Stride1D.Dense> rgba, int width, int height,
        ArrayView1D<byte, Stride1D.Dense> output, WebPLossyEncodeOptions? options = null)
    {
        int length = await EncodeToScratchAsync(rgba, width, height, options ?? new WebPLossyEncodeOptions());
        if (output.Length < length) throw new ArgumentException($"output must hold {length} bytes.", nameof(output));
        _assemble(length, _meta.View, _scratch.View, output);
        await _accelerator.SynchronizeAsync();
        return length;
    }

    private Layout PrepareBuffers(int width, int height, WebPLossyEncodeOptions o)
    {
        Validate(width, height, o);
        if (_tokenCapacity == 0) _tokenCapacity = 1 << 16;
        var l0 = MakeLayout(width, height, _tokenCapacity);
        _tokenCapacity = Math.Max(_tokenCapacity, (long)l0.MbCount * 256);
        _yuv.Ensure(l0.YuvSize);
        _work.Ensure(l0.WorkSize);
        _mbRecs.Ensure((long)l0.MbCount * M.MbStride);
        _state.Ensure(l0.StateSize);
        _qTable.View.CopyFromCPU(WebPLossyEncTables.BuildQuantTable(o.Quality, 50));
        return l0;
    }

    /// <summary>Run every stage up to the partitions from RGBA8; returns the file length (the assemble step is the caller's).</summary>
    private async Task<int> EncodeToScratchAsync(ArrayView1D<byte, Stride1D.Dense> rgba, int width, int height, WebPLossyEncodeOptions o)
    {
        if (rgba.Length < (long)width * height * 4) throw new ArgumentException("rgba too short.", nameof(rgba));
        var l0 = PrepareBuffers(width, height, o);
        var yuv = _yuv.View;
        _toYuv.Convert(rgba, width, height, yuv.SubView(0, (long)width * height), yuv.SubView(l0.YuvU, (long)l0.Cw * l0.Ch),
            yuv.SubView(l0.YuvV, (long)l0.Cw * l0.Ch), yuv.SubView(l0.YuvA, (long)width * height), cleanupTransparent: !o.Exact);
        return await EncodePlanesToScratchAsync(width, height, o, alphaPlanePresent: true);
    }

    /// <summary>
    /// Encode YUV 4:2:0 planes (Y then U then V, tightly packed, as libwebp's WebPPicture holds them after its import) - the
    /// entry the byte-identity tests use with libwebp's own planes, and the one a video pipeline that already has YUV wants.
    /// </summary>
    internal async Task<byte[]> EncodeYuvAsync(ReadOnlyMemory<byte> yuvPlanes, int width, int height, WebPLossyEncodeOptions? options = null)
    {
        var o = options ?? new WebPLossyEncodeOptions();
        var l0 = PrepareBuffers(width, height, o);
        int n = width * height, c = l0.Cw * l0.Ch;
        if (yuvPlanes.Length < n + 2 * c) throw new ArgumentException("yuvPlanes too short.", nameof(yuvPlanes));
        DeviceTransfer.Upload(_yuv.View.SubView(0, n), yuvPlanes.Span.Slice(0, n));
        DeviceTransfer.Upload(_yuv.View.SubView(l0.YuvU, c), yuvPlanes.Span.Slice(n, c));
        DeviceTransfer.Upload(_yuv.View.SubView(l0.YuvV, c), yuvPlanes.Span.Slice(n + c, c));
        int length = await EncodePlanesToScratchAsync(width, height, o, alphaPlanePresent: false);
        _stageOut.Ensure(length);
        _assemble(length, _meta.View, _scratch.View, _stageOut.View);
        await _accelerator.SynchronizeAsync();
        return await DeviceTransfer.DownloadAsync(_stageOut.View, length);
    }

    private async Task<int> EncodePlanesToScratchAsync(int width, int height, WebPLossyEncodeOptions o, bool alphaPlanePresent)
    {
        var yuv = _yuv.View;
        int maxI4 = 256 * 16 * 16;
        while (true)
        {
            var l = MakeLayout(width, height, _tokenCapacity);
            _tokens.Ensure(_tokenCapacity);
            _scratch.Ensure((long)l.P0Cap + l.T1Cap);
            var hostMeta = HostMeta(l, o, _tokenCapacity, maxI4);
            hostMeta[M.AlphaPlanePresent] = alphaPlanePresent ? 1 : 0;
            _meta.View.CopyFromCPU(hostMeta);
            var state = _state.View;
            // ResetBoundaryPredictions: the border row / column of the 4x4 mode map is B_DC_PRED (0); the analysis writes the rest.
            _work.View.SubView(0, l.PredsSize).MemSetToZero();
            _analysis(l.MbCount, yuv, yuv.SubView(l.YuvU, (long)l.Cw * l.Ch), yuv.SubView(l.YuvV, (long)l.Cw * l.Ch),
                yuv.SubView(l.YuvA, (long)width * height), _mbRecs.View, _work.View, _meta.View);
            _setup(1, _mbRecs.View, _meta.View, _tables.View, _qTable.View,
                state.SubView(l.StCoeffs, WebPLossySetupKernel.ProbaCount), state.SubView(l.StStats, WebPLossySetupKernel.ProbaCount),
                state.SubView(l.StLevelCost, WebPLossySetupKernel.LevelCostCount));
            var pacer = new LaunchPacer(_accelerator);
            for (int row = 0; row < l.MbH; row++)
                for (int x = 0; x < l.MbW; x += WebPLossyEncodeRowKernel.MbsPerLaunch)
                {
                    _row(1, yuv, _mbRecs.View, _work.View, state, _meta.View, _tables.View, _tokens.View, row, x, x + WebPLossyEncodeRowKernel.MbsPerLaunch);
                    await pacer.LaunchedAsync();
                }
            await _accelerator.SynchronizeAsync();
            var m = await _meta.View.CopyToHostAsync();
            var err = (WebPLossyEncError)m[M.Error];
            if (err == WebPLossyEncError.TokenBufferTooSmall)
            {
                _tokenCapacity = (long)(m[M.TokenCount] * 1.25) + 1024;
                continue;
            }
            if (err != WebPLossyEncError.None) throw new InvalidOperationException($"lossy WebP encode failed: {err}");
            if (m[M.HasAlpha] != 0)
                throw new NotSupportedException("The picture has transparency: lossy WebP with an ALPH chunk is not ported yet (use WebPLosslessEncoderGpu).");
            ulong sizeP0 = ((ulong)(uint)m[M.SizeP0Hi] << 32) | (uint)m[M.SizeP0Lo];
            if (maxI4 > 0 && sizeP0 > Partition0SizeLimit)
            {
                maxI4 >>= 1;   // VP8EncTokenLoop: strengthen the header bit limitation and start over
                continue;
            }
            _finishProbas(1, _mbRecs.View, _work.View, state, _meta.View, _tables.View, _tokens.View, _scratch.View);
            int tokenSteps = Math.Max(1, (m[M.TokenCount] + WebPLossyFinishKernel.TokensPerLaunch - 1) / WebPLossyFinishKernel.TokensPerLaunch);
            for (int i = 0; i < tokenSteps; i++)
            {
                _finishTokens(1, _mbRecs.View, _work.View, state, _meta.View, _tables.View, _tokens.View, _scratch.View);
                await pacer.LaunchedAsync();
            }
            _finishHeader(1, _mbRecs.View, _work.View, state, _meta.View, _tables.View, _tokens.View, _scratch.View);
            int modeSteps = (l.MbCount + WebPLossyFinishKernel.MbsPerLaunch - 1) / WebPLossyFinishKernel.MbsPerLaunch;
            for (int i = 0; i < modeSteps; i++)
            {
                _finishModes(1, _mbRecs.View, _work.View, state, _meta.View, _tables.View, _tokens.View, _scratch.View);
                await pacer.LaunchedAsync();
            }
            _finishEnd(1, _mbRecs.View, _work.View, state, _meta.View, _tables.View, _tokens.View, _scratch.View);
            await _accelerator.SynchronizeAsync();
            var f = await _meta.View.SubView(0, M.Seg).CopyToHostAsync();
            var ferr = (WebPLossyEncError)f[M.Error];
            if (ferr != WebPLossyEncError.None) throw new InvalidOperationException($"lossy WebP encode failed: {ferr}");
            return f[M.OutLength];
        }
    }

    /// <summary>Encode managed RGBA8; returns the file (a terminal CPU sink; in a browser prefer the TypedArray overload).</summary>
    public async Task<byte[]> EncodeAsync(ReadOnlyMemory<byte> rgba, int width, int height, WebPLossyEncodeOptions? options = null)
    {
        _stageIn.Ensure(rgba.Length);
        DeviceTransfer.Upload(_stageIn.View, rgba.Span);
        int length = await EncodeToScratchAsync(_stageIn.View, width, height, options ?? new WebPLossyEncodeOptions());
        _stageOut.Ensure(length);
        _assemble(length, _meta.View, _scratch.View, _stageOut.View);
        await _accelerator.SynchronizeAsync();
        return await DeviceTransfer.DownloadAsync(_stageOut.View, length);
    }

    /// <summary>Encode RGBA8 held in a JS typed array (browser accelerators only); the file comes back as a JS Uint8Array.</summary>
    [SupportedOSPlatform("browser")]
    public async Task<Uint8Array> EncodeAsync(TypedArray rgba, int width, int height, WebPLossyEncodeOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(rgba);
        _stageIn.Ensure(rgba.ByteLength);
        DeviceTransfer.Upload(_stageIn.View, rgba);
        int length = await EncodeToScratchAsync(_stageIn.View, width, height, options ?? new WebPLossyEncodeOptions());
        _stageOut.Ensure(length);
        _assemble(length, _meta.View, _scratch.View, _stageOut.View);
        await _accelerator.SynchronizeAsync();
        return await DeviceTransfer.DownloadUint8ArrayAsync(_stageOut.View, length);
    }

    /// <summary>Release the kernels' buffers.</summary>
    public void Dispose()
    {
        _toYuv.Dispose();
        _tables.Dispose(); _meta.Dispose(); _qTable.Dispose();
        _yuv.Dispose(); _work.Dispose(); _scratch.Dispose(); _stageIn.Dispose(); _stageOut.Dispose();
        _mbRecs.Dispose(); _state.Dispose(); _tokens.Dispose();
    }
}
