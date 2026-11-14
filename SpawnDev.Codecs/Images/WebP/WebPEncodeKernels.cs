// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// The WebP lossless encode kernels. Every stage runs on the accelerator; the host dispatches them in a fixed order
// and reads back a few scalars (the output length, the cost of a colour-cache variant).
//
//   1. ToArgbKernel          (1 thread per pixel)   RGBA8 -> ARGB, optionally clearing RGB under alpha 0.
//   2. Analyze               (resumable, 1 thread per launch)  alpha, distinct colours (palette when <= 256, sorted), delta-coded palette.
//   3. PaletteIndexKernel    (1 thread per packed pixel)   colour -> palette index, 2/4/8 pixels bundled per word.
//   4. SubtractGreenKernel   (1 thread per pixel)   forward subtract green.
//   5. PredictorModeKernel   (1 thread per tile)    picks the predictor mode per tile by minimum residual magnitude.
//   6. PredictorResidualKernel (1 thread per pixel) residuals against the ORIGINAL neighbours (fully parallel).
//   7. CrossColorModeKernel / CrossColorKernel      per-tile multipliers and the forward cross-colour transform.
//   8. Tokenize              (resumable, 1 thread per launch)  LZ77 hash chain + colour cache -> tokens + histograms.
//   9. BuildCodesKernel      (1 thread per code)    length-limited Huffman codes of the five prefix codes.
//  10. Emit                  (small pieces in stream order, each continuing the bit writer)  header, transform bits,
//                            sub-images (histograms + codes + pixels, chunked), prefix code definitions, tokens (resumable), last byte.
//  11. FinalizeKernel + CopyBlobKernel              RIFF / VP8X / chunk headers, ICCP / EXIF / XMP payloads.
//  12. AlphaCost / AlphaPick / AlphaToArgb / AlphaRaw / FinalizeAlpha   ALPH chunk: filter choice, filtered plane, header byte.
//  13. FinalizeFrame / FinalizeAnim                 ANMF + VP8L chunk headers per frame; RIFF / VP8X / ICCP / ANIM / EXIF / XMP of an animation.
//
// The transforms read only ORIGINAL pixels, so they are parallel; the bitstream (variable-length codes in one
// stream) is a sequential dependency chain, hence the single-thread kernels - still accelerator-resident. Every serial
// stage whose work grows with the image (analysis, tokenisation, tile histograms, token emission) is RESUMABLE: it runs as
// a chain of launches of Vp8lEncLayout.StepChunk pixels / tokens each, continuing from state kept in the work buffer, so
// no launch holds the GPU long (a long launch stalls the desktop compositor and trips the OS GPU watchdog). The meta
// prefix clustering is parallel per tile / per (cluster, symbol).
// Stages that do not apply to the image (palette decided on the accelerator) return at once, so the host
// needs no mid-pipeline readback.

using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.Codecs.Images.WebP;

/// <summary>Loads and launches the WebP lossless encode kernels.</summary>
public sealed class WebPEncodeKernels : IDisposable
{
    private Action<Index1D, ArrayView<byte>, ArrayView<uint>, ArrayView<int>, int> _toArgb = null!;
    private Action<Index1D, ArrayView<uint>, ArrayView<int>, ArrayView<int>, int> _transparentFill = null!;
    private Action<Index1D, ArrayView<int>, ArrayView<int>> _analyzeInit = null!, _analyzeFinish = null!;
    private Action<Index1D, ArrayView<uint>, ArrayView<int>, ArrayView<int>> _analyze = null!;
    private Action<Index1D, ArrayView<uint>, ArrayView<uint>, ArrayView<int>, ArrayView<int>> _paletteIndex = null!;
    private Action<Index1D, ArrayView<uint>, ArrayView<int>> _subtractGreen = null!;
    private Action<Index1D, ArrayView<uint>, ArrayView<uint>, ArrayView<int>, ArrayView<int>, int> _predictorMode = null!;
    private Action<Index1D, ArrayView<uint>, ArrayView<uint>, ArrayView<int>, ArrayView<int>> _predictorResidual = null!;
    private Action<Index1D, ArrayView<uint>, ArrayView<int>, ArrayView<int>> _crossMode = null!;
    private Action<Index1D, ArrayView<uint>, ArrayView<int>, ArrayView<int>> _crossForward = null!;
    private Action<Index1D, ArrayView<int>, ArrayView<int>, int> _tokenizeInit = null!;
    private Action<Index1D, ArrayView<uint>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int> _tokenize = null!;
    private Action<Index1D, ArrayView<int>, ArrayView<int>> _buildCodes = null!;
    // the parallel tokeniser (Vp8lTokenizeParallel)
    private Action<Index1D, ArrayView<int>, int, int> _ptFill = null!;
    private Action<Index1D, ArrayView<uint>, ArrayView<int>, Vp8lTokenizeParallel.Args> _ptLinks = null!, _ptFixup = null!, _ptSearch = null!;
    private Action<Index1D, ArrayView<int>, Vp8lTokenizeParallel.Args> _ptCarry = null!, _ptCount = null!, _ptScan = null!;
    private Action<Index1D, ArrayView<int>, ArrayView<int>, Vp8lTokenizeParallel.Args> _ptParse = null!, _ptFinish = null!;
    private Action<Index1D, ArrayView<uint>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>,
        Vp8lTokenizeParallel.Args> _ptEmit = null!;
    private MemoryBuffer1D<int, Stride1D.Dense>? _ptScratch;
    // the parallel token writer (Vp8lEmitParallel): valid while the last TokenizeAsync was the parallel tokeniser
    private Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, Vp8lTokenizeParallel.Args> _peBits = null!;
    private Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, Vp8lTokenizeParallel.Args> _peScan = null!;
    private Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, Vp8lTokenizeParallel.Args> _peWrite = null!;
    private Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, Vp8lTokenizeParallel.Args> _peCopy = null!;
    private Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, Vp8lTokenizeParallel.Args> _peFinish = null!;
    private MemoryBuffer1D<int, Stride1D.Dense>? _peWords;
    private Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, Vp8lTokenizeParallel.Args> _peTileHist = null!;
    private Vp8lTokenizeParallel.Args _ptArgs;
    private bool _ptTokens;

    /// <summary>Tests only: write the tokens with the serial writer (EmitTokensKernel) even after the parallel tokeniser.</summary>
    internal static bool SerialEmit;

    /// <summary>
    /// Tests only: tokenise with the serial walk (Vp8lEncodeGpu.TokenizeStep, one thread) instead of the parallel tokeniser
    /// (Vp8lTokenizeParallel). Both produce the same tokens, histograms and extra bits; the serial walk is the oracle.
    /// </summary>
    internal static bool SerialTokenize;

    /// <summary>Tests only: off = transparent pixels stay black under Exact = false (the encoder before the fill), to measure it.</summary>
    internal static bool FillTransparent = true;

    /// <summary>Tests only: off = the predictor modes are scored on the cleared pixels (the encoder before 2026-11-14), to measure it.</summary>
    internal static bool PredictorModeSeesFill = true;
    private Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>> _emitHeader = null!, _emitTail = null!;
    private Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, int> _emitBits = null!;
    private Action<Index1D, ArrayView<int>, ArrayView<int>, int> _subImageBegin = null!, _subImageBuild = null!;
    private Action<Index1D, ArrayView<int>, ArrayView<int>, int, int> _subImageHist = null!;
    private Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, int, int> _subImageSymbols = null!;
    private Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, int, int, int> _emitCode = null!;
    private Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>> _emitTokens = null!;
    private Action<Index1D, ArrayView<byte>, ArrayView<int>> _finalize = null!;
    private Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, int, int> _copyBlob = null!;
    private Action<Index1D, ArrayView<int>, ArrayView<int>> _tileHistInit = null!;
    private Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>> _tileHist = null!;
    private Action<Index1D, ArrayView<int>, ArrayView<int>> _clusterSeedFirst = null!, _clusterTileMin = null!, _clusterSeedNext = null!,
        _clusterLengths = null!, _clusterAssign = null!, _clusterAccumulate = null!, _clusterFinish = null!;
    private Action<Index1D, ArrayView<int>, ArrayView<int>> _buildGroupCodes = null!;
    private Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>> _alphaCost = null!;
    private Action<Index1D, ArrayView<int>, ArrayView<int>> _alphaPick = null!;
    private Action<Index1D, ArrayView<byte>, ArrayView<uint>, ArrayView<int>> _alphaToArgb = null!;
    private Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>> _alphaRaw = null!;
    private Action<Index1D, ArrayView<byte>, ArrayView<int>, int> _finalizeAlpha = null!;
    private Action<Index1D, ArrayView<byte>, ArrayView<int>> _finalizeFrame = null!;
    private Action<Index1D, ArrayView<byte>, ArrayView<int>> _finalizeAnim = null!;

    private readonly LaunchPacer _pacer;
    private readonly Accelerator _accelerator;

    /// <summary>
    /// Diagnostics: when set, every emit launch is awaited on its own and then named here, so a launch that never returns is
    /// the one after the last name printed. Null = off (no extra synchronisation).
    /// </summary>
    internal static Action<string>? Trace;

    private ValueTask TraceAsync(string name) => Trace is null ? ValueTask.CompletedTask : TraceSlowAsync(name);

    private async ValueTask TraceSlowAsync(string name)
    {
        await _accelerator.SynchronizeAsync();
        Trace?.Invoke(name);
    }

    // The kernels above load lazily (EnsureKernelsAsync, awaited loads): the constructor compiles nothing.
    private Task? _kernelsLoad;

    /// <summary>Bind to <paramref name="accelerator"/>. Compiles nothing: await <see cref="EnsureKernelsAsync"/> before the first launch.</summary>
    public WebPEncodeKernels(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _pacer = new LaunchPacer(accelerator);
    }

    /// <summary>Load the kernels once (awaited loads: each compile yields to the browser's event loop first).</summary>
    public Task EnsureKernelsAsync()
    {
        if (_kernelsLoad == null || _kernelsLoad.IsFaulted || _kernelsLoad.IsCanceled) _kernelsLoad = LoadKernelsAsync();
        return _kernelsLoad;
    }

    private async Task LoadKernelsAsync()
    {
        var a = _accelerator;
        _toArgb = await a.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<byte>, ArrayView<uint>, ArrayView<int>, int>(ToArgbKernel);
        _transparentFill = await a.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<uint>, ArrayView<int>, ArrayView<int>, int>(TransparentFillKernel);
        _analyzeInit = await a.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<int>, ArrayView<int>>(AnalyzeInitKernel);
        _analyze = await a.LoadAutoGroupedStreamKernelAsync<
            Index1D, ArrayView<uint>, ArrayView<int>, ArrayView<int>>(AnalyzeKernel);
        _analyzeFinish = await a.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<int>, ArrayView<int>>(AnalyzeFinishKernel);
        _paletteIndex = await a.LoadAutoGroupedStreamKernelAsync<
            Index1D, ArrayView<uint>, ArrayView<uint>, ArrayView<int>, ArrayView<int>>(PaletteIndexKernel);
        _subtractGreen = await a.LoadAutoGroupedStreamKernelAsync<
            Index1D, ArrayView<uint>, ArrayView<int>>(SubtractGreenKernel);
        _predictorMode = await a.LoadAutoGroupedStreamKernelAsync<
            Index1D, ArrayView<uint>, ArrayView<uint>, ArrayView<int>, ArrayView<int>, int>(PredictorModeKernel);
        _predictorResidual = await a.LoadAutoGroupedStreamKernelAsync<
            Index1D, ArrayView<uint>, ArrayView<uint>, ArrayView<int>, ArrayView<int>>(PredictorResidualKernel);
        _crossMode = await a.LoadAutoGroupedStreamKernelAsync<
            Index1D, ArrayView<uint>, ArrayView<int>, ArrayView<int>>(CrossColorModeKernel);
        _crossForward = await a.LoadAutoGroupedStreamKernelAsync<
            Index1D, ArrayView<uint>, ArrayView<int>, ArrayView<int>>(CrossColorKernel);
        _tokenizeInit = await a.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<int>, ArrayView<int>, int>(TokenizeInitKernel);
        _tokenize = await a.LoadAutoGroupedStreamKernelAsync<
            Index1D, ArrayView<uint>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int>(TokenizeKernel);
        _buildCodes = await a.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<int>, ArrayView<int>>(BuildCodesKernel);
        _ptFill = await a.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<int>, int, int>(Vp8lTokenizeParallel.FillKernel);
        _ptLinks = await a.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<uint>, ArrayView<int>, Vp8lTokenizeParallel.Args>(Vp8lTokenizeParallel.LinksKernel);
        _ptFixup = await a.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<uint>, ArrayView<int>, Vp8lTokenizeParallel.Args>(Vp8lTokenizeParallel.FixupKernel);
        _ptSearch = await a.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<uint>, ArrayView<int>, Vp8lTokenizeParallel.Args>(Vp8lTokenizeParallel.SearchKernel);
        _ptCarry = await a.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<int>, Vp8lTokenizeParallel.Args>(Vp8lTokenizeParallel.CarryKernel);
        _ptCount = await a.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<int>, Vp8lTokenizeParallel.Args>(Vp8lTokenizeParallel.CountKernel);
        _ptScan = await a.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<int>, Vp8lTokenizeParallel.Args>(Vp8lTokenizeParallel.ScanKernel);
        _ptParse = await a.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<int>, ArrayView<int>, Vp8lTokenizeParallel.Args>(Vp8lTokenizeParallel.ParseKernel);
        _ptFinish = await a.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<int>, ArrayView<int>, Vp8lTokenizeParallel.Args>(Vp8lTokenizeParallel.FinishKernel);
        _ptEmit = await a.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<uint>, ArrayView<int>, ArrayView<int>, ArrayView<int>,
            ArrayView<int>, ArrayView<int>, Vp8lTokenizeParallel.Args>(Vp8lTokenizeParallel.EmitKernel);
        _peBits = await a.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>,
            ArrayView<int>, Vp8lTokenizeParallel.Args>(Vp8lEmitParallel.BitsKernel);
        _peScan = await a.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>,
            Vp8lTokenizeParallel.Args>(Vp8lEmitParallel.ScanKernel);
        _peWrite = await a.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>,
            ArrayView<int>, ArrayView<int>, Vp8lTokenizeParallel.Args>(Vp8lEmitParallel.WriteKernel);
        _peCopy = await a.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>,
            ArrayView<int>, Vp8lTokenizeParallel.Args>(Vp8lEmitParallel.CopyKernel);
        _peFinish = await a.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>,
            Vp8lTokenizeParallel.Args>(Vp8lEmitParallel.FinishKernel);
        _peTileHist = await a.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>,
            ArrayView<int>, Vp8lTokenizeParallel.Args>(Vp8lEmitParallel.TileHistKernel);
        _emitHeader = await a.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>>(EmitHeaderKernel);
        _emitBits = await a.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, int>(EmitBitsKernel);
        _subImageBegin = await a.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<int>, ArrayView<int>, int>(SubImageBeginKernel);
        _subImageHist = await a.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<int>, ArrayView<int>, int, int>(SubImageHistKernel);
        _subImageBuild = await a.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<int>, ArrayView<int>, int>(SubImageBuildKernel);
        _subImageSymbols = await a.LoadAutoGroupedStreamKernelAsync<
            Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, int, int>(SubImageSymbolsKernel);
        _emitCode = await a.LoadAutoGroupedStreamKernelAsync<
            Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, int, int, int>(EmitCodeKernel);
        _emitTokens = await a.LoadAutoGroupedStreamKernelAsync<
            Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>>(EmitTokensKernel);
        _emitTail = await a.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>>(EmitTailKernel);
        _finalize = await a.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<byte>, ArrayView<int>>(FinalizeKernel);
        _copyBlob = await a.LoadAutoGroupedStreamKernelAsync<
            Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, int, int>(CopyBlobKernel);
        _tileHistInit = await a.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<int>, ArrayView<int>>(TileHistInitKernel);
        _tileHist = await a.LoadAutoGroupedStreamKernelAsync<
            Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>>(TileHistKernel);
        _clusterSeedFirst = await a.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<int>, ArrayView<int>>(ClusterSeedFirstKernel);
        _clusterTileMin = await a.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<int>, ArrayView<int>>(ClusterTileMinKernel);
        _clusterSeedNext = await a.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<int>, ArrayView<int>>(ClusterSeedNextKernel);
        _clusterLengths = await a.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<int>, ArrayView<int>>(ClusterLengthsKernel);
        _clusterAssign = await a.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<int>, ArrayView<int>>(ClusterAssignKernel);
        _clusterAccumulate = await a.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<int>, ArrayView<int>>(ClusterAccumulateKernel);
        _clusterFinish = await a.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<int>, ArrayView<int>>(ClusterFinishKernel);
        _buildGroupCodes = await a.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<int>, ArrayView<int>>(BuildGroupCodesKernel);
        _alphaCost = await a.LoadAutoGroupedStreamKernelAsync<
            Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>>(AlphaCostKernel);
        _alphaPick = await a.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<int>, ArrayView<int>>(AlphaPickKernel);
        _alphaToArgb = await a.LoadAutoGroupedStreamKernelAsync<
            Index1D, ArrayView<byte>, ArrayView<uint>, ArrayView<int>>(AlphaToArgbKernel);
        _alphaRaw = await a.LoadAutoGroupedStreamKernelAsync<
            Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>>(AlphaRawKernel);
        _finalizeAlpha = await a.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<byte>, ArrayView<int>, int>(FinalizeAlphaKernel);
        _finalizeFrame = await a.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<byte>, ArrayView<int>>(FinalizeFrameKernel);
        _finalizeAnim = await a.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<byte>, ArrayView<int>>(FinalizeAnimKernel);
    }

    // ------------------------------------------------------------------
    // Launchers
    // ------------------------------------------------------------------

    /// <summary>RGBA8 to ARGB over <paramref name="pixels"/> pixels; <paramref name="exact"/> 0 clears RGB where alpha is 0.</summary>
    public void ToArgb(ArrayView<byte> rgba, ArrayView<uint> argb, ArrayView<int> meta, int pixels, int exact) => _toArgb(pixels, rgba, argb, meta, exact);

    /// <summary>Pixels per launch of <see cref="TransparentFill"/> (one thread walks them in raster order).</summary>
    public const int TransparentFillChunk = 1 << 16;

    /// <summary>
    /// libwebp's lossless GetResidual with exact = 0 (predictor_enc.c): every transparent pixel (alpha 0) of
    /// <paramref name="argb"/> becomes its own prediction with alpha 0, in raster order (each replaced pixel feeds the next
    /// predictions), so the residual kernel gives it an RGB residual of 0 where clearing to black cost a full residual
    /// next to opaque pixels. Run after the predictor modes are chosen, before the residuals; nothing to do (a one-load
    /// launch) when no pixel was cleared (<see cref="WebPEncMeta.Transparent"/>). Queued.
    /// </summary>
    public void TransparentFill(ArrayView<uint> argb, ArrayView<int> work, ArrayView<int> meta, int pixels)
    {
        if (!FillTransparent) return;
        for (int start = 0; start < pixels; start += TransparentFillChunk)
            _transparentFill(1, argb, work, meta, start);
    }

    /// <summary>Colour analysis (alpha, palette) of <paramref name="pixels"/> pixels: set-up, bounded resumable scan launches, finish.</summary>
    public async ValueTask AnalyzeAsync(ArrayView<uint> argb, ArrayView<int> work, ArrayView<int> meta, long pixels)
    {
        _analyzeInit(1024, work, meta);
        for (int i = Vp8lEncLayout.Steps(pixels); i > 0; i--)
        {
            _analyze(1, argb, work, meta);
            await _pacer.LaunchedAsync();
        }
        _analyzeFinish(1, work, meta);
    }

    /// <summary>Palette indices (bundled) from <paramref name="src"/> into <paramref name="dst"/>; extent = the pixel count (surplus threads return).</summary>
    public void PaletteIndex(ArrayView<uint> src, ArrayView<uint> dst, ArrayView<int> work, ArrayView<int> meta, int pixels)
        => _paletteIndex(pixels, src, dst, work, meta);

    /// <summary>Forward subtract green, in place.</summary>
    public void SubtractGreen(ArrayView<uint> argb, ArrayView<int> meta, int pixels) => _subtractGreen(pixels, argb, meta);

    /// <summary>
    /// Predictor mode per tile. <paramref name="scratch"/> (a buffer of at least the image's pixels, not
    /// <paramref name="argb"/>, free until the residual pass) holds each tile's transparent fill while a mode is scored
    /// (Exact = false; see PredictorModeKernel).
    /// </summary>
    public void PredictorMode(ArrayView<uint> argb, ArrayView<uint> scratch, ArrayView<int> work, ArrayView<int> meta, int tiles)
        => _predictorMode(tiles, argb, scratch, work, meta, FillTransparent && PredictorModeSeesFill ? 1 : 0);

    /// <summary>Predictor residuals of <paramref name="src"/> into <paramref name="dst"/>.</summary>
    public void PredictorResidual(ArrayView<uint> src, ArrayView<uint> dst, ArrayView<int> work, ArrayView<int> meta, int pixels)
        => _predictorResidual(pixels, src, dst, work, meta);

    /// <summary>Cross-colour multipliers per tile of the (residual) image.</summary>
    public void CrossColorMode(ArrayView<uint> argb, ArrayView<int> work, ArrayView<int> meta, int tiles)
        => _crossMode(tiles, argb, work, meta);

    /// <summary>Forward cross-colour transform, in place.</summary>
    public void CrossColor(ArrayView<uint> argb, ArrayView<int> work, ArrayView<int> meta, int pixels)
        => _crossForward(pixels, argb, work, meta);

    /// <summary>
    /// LZ77 + colour cache tokenisation of the main image of <paramref name="pixels"/> (packed) pixels: a parallel set-up, then
    /// bounded launches that each continue the serial walk where the previous one stopped.
    /// </summary>
    public async ValueTask TokenizeAsync(ArrayView<uint> main, ArrayView<int> work, ArrayView<int> tok, ArrayView<int> meta, ArrayView<int> tables,
        int cacheBits, long pixels, int xsize, int chainDepth)
    {
        _tokenizeInit(1 << Vp8lEncLayout.HashBits, work, meta, cacheBits);   // histograms, and the serial walk's state
        _ptTokens = false;
        if (SerialTokenize)
        {
            for (int i = Vp8lEncLayout.Steps(pixels, Vp8lEncLayout.TokenizeChunk); i > 0; i--)
            {
                _tokenize(1, main, work, tok, meta, tables, cacheBits);
                await _pacer.LaunchedAsync();
            }
            return;
        }
        // the parallel tokeniser (Vp8lTokenizeParallel): the same tokens, histograms and extra bits
        int n = checked((int)pixels);
        if (n == 0)
        {
            _tokenize(1, main, work, tok, meta, tables, cacheBits);   // writes the empty stream's counts
            return;
        }
        var l = new Vp8lTokenizeParallel.Layout(n);
        long words = Vp8lEmitParallel.Words(n);
        if (_ptScratch == null || _ptScratch.Length < l.Total || _peWords == null || _peWords.Length < words)
        {
            await _accelerator.SynchronizeAsync();   // queued work may still use the old buffers
            _ptScratch?.Dispose();
            _peWords?.Dispose();
            _ptScratch = _accelerator.Allocate1D<int>(l.Total);
            _peWords = _accelerator.Allocate1D<int>(words);
        }
        var s = _ptScratch.View;
        var args = Vp8lTokenizeParallel.MakeArgs(l, xsize, cacheBits, chainDepth);
        int tables0 = args.PairTable, tablesLen = checked((int)(l.BlockCount - l.PairTable));
        _ptFill(tablesLen, s, tables0, -1);
        _ptFill(n, s, args.Flag, 0);
        _ptFill(Vp8lTokenizeParallel.StateInts, s, args.State, 0);
        _ptLinks(l.Blocks, main, s, args);
        _ptCarry(Vp8lTokenizeParallel.PairKeys + Vp8lTokenizeParallel.CacheKeys, s, args);
        _ptFixup(n, main, s, args);
        await _pacer.LaunchedAsync();
        for (int b = 0; b < n; b += Vp8lTokenizeParallel.SearchChunk)
        {
            var sa = args;
            sa.Base = b;
            _ptSearch(Math.Min(Vp8lTokenizeParallel.SearchChunk, n - b), main, s, sa);
            await _pacer.LaunchedAsync();
        }
        // the parse: at most n tokens, ParseChunk per launch
        for (int i = (n + Vp8lTokenizeParallel.ParseChunk - 1) / Vp8lTokenizeParallel.ParseChunk; i > 0; i--)
        {
            _ptParse(1, s, tables, args);
            await _pacer.LaunchedAsync();
        }
        _ptCount(l.EmitBlocks, s, args);
        _ptScan(1, s, args);
        _ptEmit(l.EmitBlocks, main, s, work, tok, meta, tables, args);
        _ptFinish(1, s, meta, args);
        await _pacer.LaunchedAsync();
        _ptArgs = args;
        _ptTokens = true;
    }

    /// <summary>
    /// The main image's tokens with Vp8lEmitParallel, continuing the bit writer in the step state: the same bits as the
    /// EmitTokensKernel chain. Needs the parallel tokeniser's scratch of the LAST TokenizeAsync (the final tokenisation).
    /// </summary>
    private async ValueTask EmitTokensParallelAsync(ArrayView<byte> output, ArrayView<int> work, ArrayView<int> tok, ArrayView<int> meta,
        ArrayView<int> tables)
    {
        var a = _ptArgs;
        var s = _ptScratch!.View;
        var words = _peWords!.View;
        int wordCount = checked((int)Vp8lEmitParallel.Words(a.N));
        _ptFill(wordCount, words, 0, 0);
        _peBits(a.EmitBlocks, s, work, tok, meta, tables, a);
        _peScan(1, s, work, words, a);
        _peWrite(a.EmitBlocks, s, work, tok, meta, tables, words, a);
        await _pacer.LaunchedAsync();
        await TraceAsync("emit tokens: bits, scan, write");
        // at most 4 bytes per word; bytes past the output overflow (FinishKernel flags them)
        int copyThreads = (int)Math.Min(output.Length, 4L * wordCount);
        _peCopy(copyThreads, output, s, work, words, meta, a);
        _peFinish(1, s, work, words, meta, a);
        await _pacer.LaunchedAsync();
        await TraceAsync("emit tokens: copy");
    }

    /// <summary>Huffman codes of the five main prefix codes (one thread each).</summary>
    public void BuildCodes(ArrayView<int> work, ArrayView<int> meta) => _buildCodes(5, work, meta);

    /// <summary>Per-tile histograms of the main token stream (for the meta prefix codes); <paramref name="pixels"/> bounds the token count.</summary>
    public async ValueTask TileHistogramsAsync(ArrayView<int> work, ArrayView<int> tok, ArrayView<int> meta, ArrayView<int> tables, int tiles, long pixels)
    {
        _tileHistInit(tiles * Vp8lEncLayout.SetAlphaMax, work, meta);
        if (_ptTokens && !SerialEmit)
        {
            // the same counts from the parallel tokeniser's scratch, a thread per emit block (Vp8lEmitParallel.TileHistKernel)
            _peTileHist(_ptArgs.EmitBlocks, _ptScratch!.View, work, tok, meta, tables, _ptArgs);
            await _pacer.LaunchedAsync();
            return;
        }
        for (int i = Vp8lEncLayout.Steps(pixels); i > 0; i--)
        {
            _tileHist(1, work, tok, meta, tables);
            await _pacer.LaunchedAsync();
        }
    }

    /// <summary>Clusters the <paramref name="tiles"/> tiles into at most <paramref name="groups"/> prefix groups (see Vp8lEncodeGroups).</summary>
    public void Cluster(ArrayView<int> work, ArrayView<int> meta, int groups, int tiles)
    {
        _clusterSeedFirst(1, work, meta);
        _clusterTileMin(tiles, work, meta);
        for (int s = 1; s < groups; s++)
        {
            _clusterSeedNext(1, work, meta);
            _clusterTileMin(tiles, work, meta);
        }
        for (int round = 0; round < 4; round++)
        {
            _clusterLengths(Vp8lEncodeGroups.MaxGroups, work, meta);
            _clusterAssign(tiles, work, meta);
            _clusterAccumulate(Vp8lEncodeGroups.MaxGroups * Vp8lEncLayout.SetAlphaMax, work, meta);
        }
        _clusterFinish(1, work, meta);
    }

    /// <summary>Huffman codes of every code of every group (one thread each).</summary>
    public void BuildGroupCodes(ArrayView<int> work, ArrayView<int> meta) => _buildGroupCodes(Vp8lEncodeGroups.MaxGroups * 5, work, meta);

    /// <summary>
    /// Bitstream, in stream order: header and first transform bits, the palette image, predictor bits + image, cross-colour
    /// bits + image, cache / meta-code bits, the entropy image, the prefix codes (one launch per code), the tokens, the last
    /// byte. Every piece is its own small kernel continuing the bit writer kept in the step state (one kernel writing all
    /// of it was ~42k lines of WGSL that Dawn took minutes to compile, and walked every transform tile in one launch); a
    /// piece the stream does not have returns at once. <paramref name="pixels"/> bounds the tokens, <paramref name="subImageItems"/>
    /// the pixels of the predictor, cross-colour and entropy images.
    /// </summary>
    public async ValueTask EmitAsync(ArrayView<byte> output, ArrayView<int> work, ArrayView<int> tok, ArrayView<int> meta, ArrayView<int> tables,
        long pixels, long subImageItems)
    {
        _emitHeader(1, output, work, meta);
        await TraceAsync("emit header");
        await SubImageAsync(output, work, meta, SubPalette, 256);
        _emitBits(1, output, work, meta, 1);
        await TraceAsync("emit bits 1");
        await SubImageAsync(output, work, meta, SubPredictor, subImageItems);
        _emitBits(1, output, work, meta, 2);
        await TraceAsync("emit bits 2");
        await SubImageAsync(output, work, meta, SubCross, subImageItems);
        _emitBits(1, output, work, meta, 3);
        await TraceAsync("emit bits 3");
        await SubImageAsync(output, work, meta, SubEntropy, subImageItems);
        for (int g = 0; g < Vp8lEncodeGroups.MaxGroups; g++)
        {
            for (int k = 0; k < 5; k++)
            {
                _emitCode(1, output, work, meta, g, k, 0);
                await _pacer.LaunchedAsync();
                await TraceAsync($"emit code g{g} k{k}");
            }
        }
        if (_ptTokens && !SerialEmit)
        {
            await EmitTokensParallelAsync(output, work, tok, meta, tables);
        }
        else
        {
            for (int i = Vp8lEncLayout.Steps(pixels); i > 0; i--)
            {
                _emitTokens(1, output, work, tok, meta, tables);
                await _pacer.LaunchedAsync();
                await TraceAsync($"emit tokens, {i - 1} steps left");
            }
        }
        _emitTail(1, output, work, meta);
        await TraceAsync("emit tail");
    }

    /// <summary>
    /// Sub-image <paramref name="which"/> of at most <paramref name="items"/> pixels (a transform image or the entropy image):
    /// histograms, its five codes, their definitions and the pixels. Nothing is written when the stream does not have it.
    /// </summary>
    private async ValueTask SubImageAsync(ArrayView<byte> output, ArrayView<int> work, ArrayView<int> meta, int which, long items)
    {
        int steps = Vp8lEncLayout.Steps(items);
        _subImageBegin(Vp8lEncodeGpu.SetAlpha(0), work, meta, which);
        await TraceAsync($"sub-image {which} begin");
        for (int s = 0; s < steps; s++)
        {
            _subImageHist(4, work, meta, which, s);
            await _pacer.LaunchedAsync();
            await TraceAsync($"sub-image {which} histogram step {s}");
        }
        _subImageBuild(5, work, meta, which);
        await TraceAsync($"sub-image {which} build");
        for (int k = 0; k < 5; k++)
        {
            _emitCode(1, output, work, meta, -1, k, which);
            await _pacer.LaunchedAsync();
            await TraceAsync($"sub-image {which} code {k}");
        }
        for (int s = 0; s < steps; s++)
        {
            _subImageSymbols(1, output, work, meta, which, s);
            await _pacer.LaunchedAsync();
            await TraceAsync($"sub-image {which} symbols step {s}");
        }
    }

    /// <summary>RIFF / VP8X / chunk headers and the total length.</summary>
    public void FinalizeFile(ArrayView<byte> output, ArrayView<int> meta) => _finalize(1, output, meta);

    /// <summary>
    /// Cost estimate of every ALPH filter (one thread per filter and row, written to <paramref name="rowCost"/>, 4 x height ints),
    /// then the pick: the cheapest filter, or the requested one, lands in <see cref="WebPEncMeta.AlphaFilter"/>.
    /// </summary>
    public void AlphaPickFilter(ArrayView<byte> plane, ArrayView<int> rowCost, ArrayView<int> meta, int height)
    {
        _alphaCost(4 * height, plane, rowCost, meta);
        _alphaPick(1, rowCost, meta);
    }

    /// <summary>The filtered alpha plane as an ARGB working image (alpha 255, green = residual).</summary>
    public void AlphaToArgb(ArrayView<byte> plane, ArrayView<uint> argb, ArrayView<int> meta, int pixels)
        => _alphaToArgb(pixels, plane, argb, meta);

    /// <summary>The filtered alpha plane as raw bytes at <c>output[1 ..]</c> (ALPH compression method 0).</summary>
    public void AlphaRaw(ArrayView<byte> plane, ArrayView<byte> output, ArrayView<int> meta, int pixels)
        => _alphaRaw(pixels, plane, output, meta);

    /// <summary>ALPH header byte at <c>output[0]</c> and the total length; <paramref name="raw"/> 1 = method 0.</summary>
    public void FinalizeAlpha(ArrayView<byte> output, ArrayView<int> meta, int raw) => _finalizeAlpha(1, output, meta, raw);

    /// <summary>ANMF and VP8L chunk headers of the animation frame whose stream was just emitted.</summary>
    public void FinalizeFrame(ArrayView<byte> output, ArrayView<int> meta) => _finalizeFrame(1, output, meta);

    /// <summary>RIFF, VP8X, ICCP and ANIM headers, the EXIF / XMP chunk headers and the total length of an animation.</summary>
    public void FinalizeAnimation(ArrayView<byte> output, ArrayView<int> meta) => _finalizeAnim(1, output, meta);

    /// <summary>Copy <paramref name="count"/> blob bytes from <c>blobs[srcOffset ..]</c> to <c>output[meta[dstSlot] ..]</c>.</summary>
    public void CopyBlob(ArrayView<byte> blobs, ArrayView<byte> output, ArrayView<int> meta, int srcOffset, int count, int dstSlot)
    {
        if (count > 0) _copyBlob(count, blobs, output, meta, srcOffset, dstSlot);
    }

    // ------------------------------------------------------------------
    // 1. RGBA -> ARGB
    // ------------------------------------------------------------------

    private static void ToArgbKernel(Index1D index, ArrayView<byte> rgba, ArrayView<uint> argb, ArrayView<int> meta, int exact)
    {
        int o = index.X * 4;
        uint r = rgba[o + 0];
        uint g = rgba[o + 1];
        uint b = rgba[o + 2];
        uint a = rgba[o + 3];
        bool clear = (exact == 0) & (a == 0);
        if (clear)
        {
            r = 0;
            g = 0;
            b = 0;
            meta[WebPEncMeta.Transparent] = 1;   // every writer stores 1
        }
        argb[index.X] = (a << 24) | (r << 16) | (g << 8) | b;
    }

    // ------------------------------------------------------------------
    // 2. Analysis
    // ------------------------------------------------------------------

    private static uint SubPixels(uint a, uint b)
    {
        uint ag = 0x00ff00ffu + (a & 0xff00ff00u) - (b & 0xff00ff00u);
        uint rb = 0xff00ff00u + (a & 0x00ff00ffu) - (b & 0x00ff00ffu);
        return (ag & 0xff00ff00u) | (rb & 0x00ff00ffu);
    }

    /// <summary>1024 threads: clear the colour hash; thread 0 resets the resumable analysis state.</summary>
    private static void AnalyzeInitKernel(Index1D index, ArrayView<int> w, ArrayView<int> meta)
    {
        w[Vp8lEncLayout.Analysis + 1024 + index.X] = 0;
        if (index.X == 0)
        {
            int s = Vp8lEncLayout.StepState;
            w[s + Vp8lEncLayout.SAnalyzeI] = 0;
            w[s + Vp8lEncLayout.SAnalyzeCount] = 0;
            w[s + Vp8lEncLayout.SAnalyzeAlpha] = 0;
            w[s + Vp8lEncLayout.SAnalyzeCollecting] = meta[WebPEncMeta.AllowPalette] == 1 ? 1 : 0;
        }
    }

    /// <summary>The next <see cref="Vp8lEncLayout.StepChunk"/> pixels of the alpha / distinct-colour scan (resumable).</summary>
    private static void AnalyzeKernel(Index1D _, ArrayView<uint> argb, ArrayView<int> w, ArrayView<int> meta)
    {
        int width = meta[WebPEncMeta.Width];
        int height = meta[WebPEncMeta.Height];
        int n = width * height;
        int keys = Vp8lEncLayout.Analysis;
        int flags = Vp8lEncLayout.Analysis + 1024;
        int s = Vp8lEncLayout.StepState;

        int start = w[s + Vp8lEncLayout.SAnalyzeI];
        int count = w[s + Vp8lEncLayout.SAnalyzeCount];
        int hasAlpha = w[s + Vp8lEncLayout.SAnalyzeAlpha];
        bool collecting = w[s + Vp8lEncLayout.SAnalyzeCollecting] == 1;
        int end = start + Vp8lEncLayout.StepChunk;
        if (end > n) end = n;
        for (int i = start; i < end; i++)
        {
            uint px = argb[i];
            if ((px >> 24) != 255u) hasAlpha = 1;
            if (collecting)
            {
                int h = (int)((px * 0x9E3779B1u) >> 22);
                bool probing = true;
                while (probing)
                {
                    if (w[flags + h] == 0)
                    {
                        if (count == 256)
                        {
                            collecting = false;
                        }
                        else
                        {
                            w[flags + h] = 1;
                            w[keys + h] = (int)px;
                            count++;
                        }
                        probing = false;
                    }
                    else if ((uint)w[keys + h] == px)
                    {
                        probing = false;
                    }
                    else
                    {
                        h = (h + 1) & 1023;
                    }
                }
            }
        }
        if (start < n)
        {
            w[s + Vp8lEncLayout.SAnalyzeI] = end;
            w[s + Vp8lEncLayout.SAnalyzeCount] = count;
            w[s + Vp8lEncLayout.SAnalyzeAlpha] = hasAlpha;
            w[s + Vp8lEncLayout.SAnalyzeCollecting] = collecting ? 1 : 0;
        }
    }

    /// <summary>After the scan: HasAlpha, and the sorted (and delta-coded) palette when there are at most 256 colours.</summary>
    private static void AnalyzeFinishKernel(Index1D _, ArrayView<int> w, ArrayView<int> meta)
    {
        int keys = Vp8lEncLayout.Analysis;
        int flags = Vp8lEncLayout.Analysis + 1024;
        int s = Vp8lEncLayout.StepState;
        int count = w[s + Vp8lEncLayout.SAnalyzeCount];
        bool collecting = w[s + Vp8lEncLayout.SAnalyzeCollecting] == 1;
        meta[WebPEncMeta.HasAlpha] = w[s + Vp8lEncLayout.SAnalyzeAlpha];

        int palette = 0;
        if (collecting)
        {
            palette = count;
            int pal = Vp8lEncLayout.Palette;
            int k = 0;
            for (int i = 0; i < 1024; i++)
            {
                if (w[flags + i] == 1)
                {
                    // insertion sort by unsigned value
                    uint v = (uint)w[keys + i];
                    int j = k - 1;
                    bool shifting = j >= 0;
                    while (shifting)
                    {
                        bool greater = (uint)w[pal + j] > v;
                        if (greater)
                        {
                            w[pal + j + 1] = w[pal + j];
                            j--;
                        }
                        shifting = greater & (j >= 0);
                    }
                    w[pal + j + 1] = (int)v;
                    k++;
                }
            }
            int delta = Vp8lEncLayout.PaletteDelta;
            w[delta] = w[pal];
            for (int i = 1; i < count; i++)
                w[delta + i] = (int)SubPixels((uint)w[pal + i], (uint)w[pal + i - 1]);
        }
        meta[WebPEncMeta.PaletteAvailable] = palette;
    }

    // ------------------------------------------------------------------
    // 3. Palette indices
    // ------------------------------------------------------------------

    private static void PaletteIndexKernel(
        Index1D index, ArrayView<uint> src, ArrayView<uint> dst, ArrayView<int> w, ArrayView<int> meta)
    {
        int count = meta[WebPEncMeta.PaletteCount];
        int width = meta[WebPEncMeta.Width];
        int height = meta[WebPEncMeta.Height];
        int bits = meta[WebPEncMeta.PaletteBits];
        int packedWidth = meta[WebPEncMeta.PackedWidth];
        int i = index.X;
        bool active = (count > 0) & (i < packedWidth * height);
        if (active)
        {
            int y = i / packedWidth;
            int xp = i - y * packedWidth;
            int per = 1 << bits;
            int bpp = 8 >> bits;
            uint green = 0;
            for (int k = 0; k < per; k++)
            {
                int x = xp * per + k;
                if (x < width)
                {
                    uint px = src[y * width + x];
                    // binary search in the sorted palette
                    int lo = 0;
                    int hi = count - 1;
                    while (lo < hi)
                    {
                        int mid = (lo + hi) >> 1;
                        bool below = (uint)w[Vp8lEncLayout.Palette + mid] < px;
                        if (below) lo = mid + 1;
                        else hi = mid;
                    }
                    green |= (uint)lo << (k * bpp);
                }
            }
            dst[i] = 0xff000000u | (green << 8);
        }
    }

    // ------------------------------------------------------------------
    // 4-7. Transforms
    // ------------------------------------------------------------------

    private static void SubtractGreenKernel(Index1D index, ArrayView<uint> argb, ArrayView<int> meta)
    {
        bool active = meta[WebPEncMeta.UseSubtractGreen] == 1;
        if (active)
        {
            uint p = argb[index.X];
            uint g = (p >> 8) & 0xFFu;
            uint r = (((p >> 16) & 0xFFu) - g) & 0xFFu;
            uint b = ((p & 0xFFu) - g) & 0xFFu;
            argb[index.X] = (p & 0xff00ff00u) | (r << 16) | b;
        }
    }

    private static int AbsResidual(uint residual)
    {
        int cost = 0;
        for (int i = 0; i < 4; i++)
        {
            int v = (int)((residual >> (i * 8)) & 0xFFu);
            cost += v < 128 ? v : 256 - v;
        }
        return cost;
    }

    private static int ModeCandidate(int set, int i)
    {
        // set 0: a handful of cheap, usually good modes; set 1: all of them
        int m = i;
        if (set == 0)
        {
            if (i == 0) m = 1;
            else if (i == 1) m = 2;
            else if (i == 2) m = 11;
            else if (i == 3) m = 12;
            else m = 13;
        }
        return m;
    }

    /// <summary>
    /// A neighbour (index <paramref name="j"/>, column <paramref name="jx"/>, row <paramref name="jy"/>) of pixel (x, y) while
    /// a tile [x0, x1) x [y0, ...) is scored with the transparent fill: the tile's own pixels already walked (raster order)
    /// come from the fill scratch <paramref name="s"/>, everything else is the original (libwebp's tile rows: the row above
    /// the tile, the left column and the top-right past the tile are read from the picture).
    /// </summary>
    private static uint FillNeighbour(ArrayView<uint> argb, ArrayView<uint> s, int j, int jx, int jy, int x0, int x1, int y0, int x, int y)
    {
        bool walked = (jx >= x0) & (jx < x1) & (jy >= y0) & ((jy < y) | ((jy == y) & (jx < x)));
        uint v;
        if (walked) v = s[j];
        else v = argb[j];
        return v;
    }

    // One thread per tile: the mode with the smallest summed residual magnitude. Exact = false with transparent pixels
    // (WebPEncMeta.Transparent, not a palette image): libwebp's GetBestPredictorForTile scores every mode with GetResidual
    // (exact = 0), where a transparent pixel costs only its alpha residual and becomes its prediction for the pixels after
    // it; the fill (TransparentFill) then does the same to the picture with the chosen modes. Scoring on the cleared
    // (black) pixels instead picks modes for residuals that never reach the file. The walked pixels of the tile live in
    // <paramref name="s"/> (device memory, the tile's own region: no private array). Measured 2026-11-14: SpawnGen's LTX
    // clip as a WebPAnimEncoder animation 3.77 -> 3.54 MB, but transparent stripes covering 2/3 of each tile 2.9% larger:
    // the tile cost here is the summed residual magnitude, libwebp's is entropy-based (PredictionCostSpatialHistogram),
    // which is the next step for both opaque and transparent pictures.
    private static void PredictorModeKernel(Index1D index, ArrayView<uint> argb, ArrayView<uint> s, ArrayView<int> w, ArrayView<int> meta, int fillOn)
    {
        bool active = meta[WebPEncMeta.UsePredictor] == 1;
        if (active)
        {
            int width = meta[WebPEncMeta.PackedWidth];
            int height = meta[WebPEncMeta.Height];
            int bits = meta[WebPEncMeta.PredictorBits];
            int set = meta[WebPEncMeta.ModeSet];
            int tilesPerRow = (width + (1 << bits) - 1) >> bits;
            int t = index.X;
            int ty = t / tilesPerRow;
            int tx = t - ty * tilesPerRow;
            int x0 = tx << bits;
            int y0 = ty << bits;
            int x1 = x0 + (1 << bits);
            if (x1 > width) x1 = width;
            int y1 = y0 + (1 << bits);
            if (y1 > height) y1 = height;
            int xStart = x0 > 0 ? x0 : 1;
            int yStart = y0 > 0 ? y0 : 1;

            bool fill = (fillOn != 0) & (meta[WebPEncMeta.Transparent] == 1) & (meta[WebPEncMeta.PaletteCount] == 0);
            int candidates = set == 0 ? 5 : 14;
            int best = ModeCandidate(set, 0);
            int bestCost = int.MaxValue;
            for (int c = 0; c < candidates; c++)
            {
                int mode = ModeCandidate(set, c);
                int cost = 0;
                if (fill)
                {
                    // the whole tile in raster order (the image's first row / column use the fixed predictors, the
                    // same for every mode, but their fill feeds the pixels after them); only the mode's pixels are scored
                    for (int y = y0; y < y1; y++)
                    {
                        for (int x = x0; x < x1; x++)
                        {
                            int i = y * width + x;
                            uint pred;
                            if (i == 0) pred = 0xff000000u;
                            else if (y == 0) pred = FillNeighbour(argb, s, i - 1, x - 1, y, x0, x1, y0, x, y);
                            else if (x == 0) pred = FillNeighbour(argb, s, i - width, x, y - 1, x0, x1, y0, x, y);
                            else
                            {
                                // top-right of a row's last pixel is the row's first pixel (i - width + 1)
                                bool last = x + 1 == width;
                                int trx = last ? 0 : x + 1;
                                int try_ = last ? y : y - 1;
                                pred = WebPKernels.Predict(mode,
                                    FillNeighbour(argb, s, i - 1, x - 1, y, x0, x1, y0, x, y),
                                    FillNeighbour(argb, s, i - width, x, y - 1, x0, x1, y0, x, y),
                                    FillNeighbour(argb, s, i - width + 1, trx, try_, x0, x1, y0, x, y),
                                    FillNeighbour(argb, s, i - width - 1, x - 1, y - 1, x0, x1, y0, x, y));
                            }
                            uint p = argb[i];
                            uint residual = SubPixels(p, pred);
                            if ((p >> 24) == 0u)
                            {
                                residual &= 0xff000000u;
                                p = pred & 0x00ffffffu;
                            }
                            s[i] = p;
                            if ((x >= xStart) & (y >= yStart)) cost += AbsResidual(residual);
                        }
                    }
                }
                else
                {
                    for (int y = yStart; y < y1; y++)
                    {
                        for (int x = xStart; x < x1; x++)
                        {
                            int i = y * width + x;
                            uint pred = WebPKernels.Predict(mode, argb[i - 1], argb[i - width], argb[i - width + 1], argb[i - width - 1]);
                            cost += AbsResidual(SubPixels(argb[i], pred));
                        }
                    }
                }
                if (cost < bestCost)
                {
                    bestCost = cost;
                    best = mode;
                }
            }
            w[meta[WebPEncMeta.RegionPredictor] + t] = (int)(0xff000000u | ((uint)best << 8));
        }
    }

    /// <summary>One thread: pixels [start, start + TransparentFillChunk) of the transparent fill, in raster order (see TransparentFill).</summary>
    private static void TransparentFillKernel(Index1D _, ArrayView<uint> argb, ArrayView<int> w, ArrayView<int> meta, int start)
    {
        bool active = (meta[WebPEncMeta.UsePredictor] == 1) & (meta[WebPEncMeta.Transparent] == 1);
        if (!active) return;
        int width = meta[WebPEncMeta.PackedWidth];
        int n = width * meta[WebPEncMeta.Height];
        int bits = meta[WebPEncMeta.PredictorBits];
        int tilesPerRow = (width + (1 << bits) - 1) >> bits;
        int modes = meta[WebPEncMeta.RegionPredictor];
        int end = start + WebPEncodeKernels.TransparentFillChunk;
        if (end > n) end = n;
        for (int i = start; i < end; i++)
        {
            uint p = argb[i];
            if ((p >> 24) == 0u)
            {
                // the prediction PredictorResidualKernel makes for pixel i, from the (already filled) pixels before it
                int y = i / width;
                int x = i - y * width;
                uint pred;
                if (i == 0) pred = 0xff000000u;
                else if (y == 0) pred = argb[i - 1];
                else if (x == 0) pred = argb[i - width];
                else
                {
                    int mode = (w[modes + (y >> bits) * tilesPerRow + (x >> bits)] >> 8) & 0xF;
                    pred = WebPKernels.Predict(mode, argb[i - 1], argb[i - width], argb[i - width + 1], argb[i - width - 1]);
                }
                argb[i] = pred & 0x00ffffffu;
            }
        }
    }

    private static void PredictorResidualKernel(
        Index1D index, ArrayView<uint> src, ArrayView<uint> dst, ArrayView<int> w, ArrayView<int> meta)
    {
        bool active = meta[WebPEncMeta.UsePredictor] == 1;
        if (active)
        {
            int width = meta[WebPEncMeta.PackedWidth];
            int bits = meta[WebPEncMeta.PredictorBits];
            int tilesPerRow = (width + (1 << bits) - 1) >> bits;
            int i = index.X;
            int y = i / width;
            int x = i - y * width;
            uint pred;
            if (i == 0)
            {
                pred = 0xff000000u;
            }
            else if (y == 0)
            {
                pred = src[i - 1];
            }
            else if (x == 0)
            {
                pred = src[i - width];
            }
            else
            {
                int mode = (w[meta[WebPEncMeta.RegionPredictor] + (y >> bits) * tilesPerRow + (x >> bits)] >> 8) & 0xF;
                pred = WebPKernels.Predict(mode, src[i - 1], src[i - width], src[i - width + 1], src[i - width - 1]);
            }
            dst[i] = SubPixels(src[i], pred);
        }
    }

    private static int Signed8(int v) => v >= 128 ? v - 256 : v;

    private static int ColorDelta(int multiplier, int color) => (multiplier * color) >> 5;

    /// <summary>Residual magnitude of the red channel of <paramref name="p"/> after removing greenToRed * green.</summary>
    private static int RedCost(uint p, int greenToRed)
    {
        int green = Signed8((int)((p >> 8) & 0xFFu));
        int red = (int)((p >> 16) & 0xFFu);
        int r = (red - ColorDelta(greenToRed, green)) & 0xFF;
        return r < 128 ? r : 256 - r;
    }

    private static int BlueCost(uint p, int greenToBlue, int redToBlue)
    {
        int green = Signed8((int)((p >> 8) & 0xFFu));
        int red = Signed8((int)((p >> 16) & 0xFFu));
        int blue = (int)(p & 0xFFu);
        int b = (blue - ColorDelta(greenToBlue, green) - ColorDelta(redToBlue, red)) & 0xFF;
        return b < 128 ? b : 256 - b;
    }

    private static void CrossColorModeKernel(Index1D index, ArrayView<uint> argb, ArrayView<int> w, ArrayView<int> meta)
    {
        bool active = meta[WebPEncMeta.UseCrossColor] == 1;
        if (active)
        {
            int width = meta[WebPEncMeta.PackedWidth];
            int height = meta[WebPEncMeta.Height];
            int bits = meta[WebPEncMeta.CrossColorBits];
            int tilesPerRow = (width + (1 << bits) - 1) >> bits;
            int t = index.X;
            int ty = t / tilesPerRow;
            int tx = t - ty * tilesPerRow;
            int x0 = tx << bits;
            int y0 = ty << bits;
            int x1 = x0 + (1 << bits);
            if (x1 > width) x1 = width;
            int y1 = y0 + (1 << bits);
            if (y1 > height) y1 = height;

            // greedy search over a coarse grid of multipliers (-64..+63 in steps of 8), channel by channel
            int bestG2R = 0;
            int bestCost = int.MaxValue;
            for (int c = -8; c < 8; c++)
            {
                int m = c * 8;
                int cost = 0;
                for (int y = y0; y < y1; y++)
                    for (int x = x0; x < x1; x++)
                        cost += RedCost(argb[y * width + x], m);
                if (cost < bestCost)
                {
                    bestCost = cost;
                    bestG2R = m;
                }
            }
            int bestG2B = 0;
            bestCost = int.MaxValue;
            for (int c = -8; c < 8; c++)
            {
                int m = c * 8;
                int cost = 0;
                for (int y = y0; y < y1; y++)
                    for (int x = x0; x < x1; x++)
                        cost += BlueCost(argb[y * width + x], m, 0);
                if (cost < bestCost)
                {
                    bestCost = cost;
                    bestG2B = m;
                }
            }
            int bestR2B = 0;
            bestCost = int.MaxValue;
            for (int c = -8; c < 8; c++)
            {
                int m = c * 8;
                int cost = 0;
                for (int y = y0; y < y1; y++)
                    for (int x = x0; x < x1; x++)
                        cost += BlueCost(argb[y * width + x], bestG2B, m);
                if (cost < bestCost)
                {
                    bestCost = cost;
                    bestR2B = m;
                }
            }
            // element: red_to_blue in the red byte, green_to_blue in green, green_to_red in blue
            uint e = 0xff000000u | ((uint)(bestR2B & 0xFF) << 16) | ((uint)(bestG2B & 0xFF) << 8) | (uint)(bestG2R & 0xFF);
            w[meta[WebPEncMeta.RegionCross] + t] = (int)e;
        }
    }

    private static void CrossColorKernel(Index1D index, ArrayView<uint> argb, ArrayView<int> w, ArrayView<int> meta)
    {
        bool active = meta[WebPEncMeta.UseCrossColor] == 1;
        if (active)
        {
            int width = meta[WebPEncMeta.PackedWidth];
            int bits = meta[WebPEncMeta.CrossColorBits];
            int tilesPerRow = (width + (1 << bits) - 1) >> bits;
            int i = index.X;
            int y = i / width;
            int x = i - y * width;
            uint m = (uint)w[meta[WebPEncMeta.RegionCross] + (y >> bits) * tilesPerRow + (x >> bits)];
            int greenToRed = Signed8((int)(m & 0xFFu));
            int greenToBlue = Signed8((int)((m >> 8) & 0xFFu));
            int redToBlue = Signed8((int)((m >> 16) & 0xFFu));
            uint p = argb[i];
            int green = Signed8((int)((p >> 8) & 0xFFu));
            int red = (int)((p >> 16) & 0xFFu);
            int blue = (int)(p & 0xFFu);
            int newRed = (red - ColorDelta(greenToRed, green)) & 0xFF;
            // the decoder adds red_to_blue * (reconstructed = original) red, so the encoder subtracts it with the ORIGINAL red
            int newBlue = (blue - ColorDelta(greenToBlue, green) - ColorDelta(redToBlue, Signed8(red))) & 0xFF;
            argb[i] = (p & 0xff00ff00u) | ((uint)newRed << 16) | (uint)newBlue;
        }
    }

    // ------------------------------------------------------------------
    // 8-10. Entropy coding
    // ------------------------------------------------------------------

    private static void TokenizeInitKernel(Index1D index, ArrayView<int> w, ArrayView<int> meta, int cacheBits)
    {
        Vp8lEncodeGpu.TokenizeInit(index.X, w, meta, cacheBits);
    }

    private static void TokenizeKernel(
        Index1D _, ArrayView<uint> main, ArrayView<int> w, ArrayView<int> tok, ArrayView<int> meta, ArrayView<int> tables, int cacheBits)
    {
        int height = meta[WebPEncMeta.Height];
        int xsize = meta[WebPEncMeta.PackedWidth];
        Vp8lEncodeGpu.TokenizeStep(main, xsize, height, w, tok, meta, tables, cacheBits);
    }

    private static void TileHistInitKernel(Index1D index, ArrayView<int> w, ArrayView<int> meta)
    {
        Vp8lEncodeGroups.TileHistogramsInit(index.X, w, meta);
    }

    private static void TileHistKernel(Index1D _, ArrayView<int> w, ArrayView<int> tok, ArrayView<int> meta, ArrayView<int> tables)
    {
        Vp8lEncodeGroups.TileHistogramsStep(w, tok, meta, tables);
    }

    private static void ClusterSeedFirstKernel(Index1D _, ArrayView<int> w, ArrayView<int> meta)
    {
        Vp8lEncodeGroups.ClusterSeedFirst(w, meta);
    }

    private static void ClusterTileMinKernel(Index1D index, ArrayView<int> w, ArrayView<int> meta)
    {
        Vp8lEncodeGroups.ClusterTileMin(index.X, w, meta);
    }

    private static void ClusterSeedNextKernel(Index1D _, ArrayView<int> w, ArrayView<int> meta)
    {
        Vp8lEncodeGroups.ClusterSeedNext(w, meta);
    }

    private static void ClusterLengthsKernel(Index1D index, ArrayView<int> w, ArrayView<int> meta)
    {
        Vp8lEncodeGroups.ClusterLengths(index.X, w, meta);
    }

    private static void ClusterAssignKernel(Index1D index, ArrayView<int> w, ArrayView<int> meta)
    {
        Vp8lEncodeGroups.ClusterAssign(index.X, w, meta);
    }

    private static void ClusterAccumulateKernel(Index1D index, ArrayView<int> w, ArrayView<int> meta)
    {
        Vp8lEncodeGroups.ClusterAccumulate(index.X, w, meta);
    }

    private static void ClusterFinishKernel(Index1D _, ArrayView<int> w, ArrayView<int> meta)
    {
        Vp8lEncodeGroups.ClusterFinish(w, meta);
    }

    private static void BuildGroupCodesKernel(Index1D index, ArrayView<int> w, ArrayView<int> meta)
    {
        Vp8lEncodeGroups.BuildGroupCode(w, meta, index.X);
    }

    private static void BuildCodesKernel(Index1D index, ArrayView<int> w, ArrayView<int> meta)
    {
        int k = index.X;
        int cacheBits = meta[WebPEncMeta.CacheBits];
        int cacheSize = cacheBits > 0 ? 1 << cacheBits : 0;
        int set = meta[WebPEncMeta.RegionMain];
        int scratch = meta[WebPEncMeta.RegionBuild] + k * Vp8lEncLayout.BuildScratch;
        Vp8lEncodeGpu.BuildOneCode(w, set, cacheSize, k, scratch);
        meta[WebPEncMeta.CodeCost + k] = Vp8lEncodeGpu.CodeCostBits(w, set, cacheSize, k);
    }

    // Sub-images of the stream, in bitstream order (EmitAsync).
    private const int SubPalette = 0, SubPredictor = 1, SubCross = 2, SubEntropy = 3;

    /// <summary>Pixels of sub-image <paramref name="which"/>; 0 when the stream does not have it.</summary>
    private static int SubImageCount(ArrayView<int> meta, int which)
    {
        int packedWidth = meta[WebPEncMeta.PackedWidth];
        int height = meta[WebPEncMeta.Height];
        int count = 0;
        if (which == SubPalette)
        {
            count = meta[WebPEncMeta.PaletteCount];
        }
        else if (which == SubPredictor)
        {
            if (meta[WebPEncMeta.UsePredictor] == 1) count = Vp8lEncLayout.Tiles(packedWidth, height, meta[WebPEncMeta.PredictorBits]);
        }
        else if (which == SubCross)
        {
            if (meta[WebPEncMeta.UseCrossColor] == 1) count = Vp8lEncLayout.Tiles(packedWidth, height, meta[WebPEncMeta.CrossColorBits]);
        }
        else
        {
            if (meta[WebPEncMeta.Groups] > 1) count = Vp8lEncLayout.Tiles(packedWidth, height, meta[WebPEncMeta.MetaBits]);
        }
        return count;
    }

    /// <summary>Work-buffer offset of the pixels of sub-image <paramref name="which"/>.</summary>
    private static int SubImageSource(ArrayView<int> meta, int which)
    {
        int src = meta[WebPEncMeta.RegionEntropy];
        if (which == SubPalette) src = Vp8lEncLayout.PaletteDelta;
        else if (which == SubPredictor) src = meta[WebPEncMeta.RegionPredictor];
        else if (which == SubCross) src = meta[WebPEncMeta.RegionCross];
        return src;
    }

    /// <summary>The VP8L header (not for an ALPH stream) and the bits of the first transform when it is a palette or subtract green.</summary>
    private static void EmitHeaderKernel(Index1D _, ArrayView<byte> output, ArrayView<int> w, ArrayView<int> meta)
    {
        var bw = new Vp8lBitWriter { Pos = meta[WebPEncMeta.PayloadStart], Acc = 0, Nbits = 0, Cap = meta[WebPEncMeta.OutputCapacity], Overflow = 0 };
        // The 5-byte VP8L header (40 bits: byte aligned) is omitted for an ALPH stream, whose size comes from the VP8 frame.
        if (meta[WebPEncMeta.Headerless] == 0)
        {
            Vp8lEncodeGpu.PutBits(ref bw, output, 0x2Fu, 8);
            Vp8lEncodeGpu.PutBits(ref bw, output, (uint)(meta[WebPEncMeta.Width] - 1), 14);
            Vp8lEncodeGpu.PutBits(ref bw, output, (uint)(meta[WebPEncMeta.Height] - 1), 14);
            Vp8lEncodeGpu.PutBits(ref bw, output, (uint)meta[WebPEncMeta.HasAlpha], 1);
            Vp8lEncodeGpu.PutBits(ref bw, output, 0, 3);
        }
        int palette = meta[WebPEncMeta.PaletteCount];
        if (palette > 0)
        {
            // colour indexing; its palette image follows
            Vp8lEncodeGpu.PutBits(ref bw, output, 1, 1);
            Vp8lEncodeGpu.PutBits(ref bw, output, 3, 2);
            Vp8lEncodeGpu.PutBits(ref bw, output, (uint)(palette - 1), 8);
        }
        else if (meta[WebPEncMeta.UseSubtractGreen] == 1)
        {
            Vp8lEncodeGpu.PutBits(ref bw, output, 1, 1);
            Vp8lEncodeGpu.PutBits(ref bw, output, 2, 2);
        }
        SaveWriter(w, ref bw);
    }

    /// <summary>
    /// The bits between the sub-images: <paramref name="step"/> 1 = the predictor transform header, 2 = the cross-colour
    /// transform header, 3 = end of transforms, colour cache and meta prefix code flags (and the token position reset).
    /// </summary>
    private static void EmitBitsKernel(Index1D _, ArrayView<byte> output, ArrayView<int> w, ArrayView<int> meta, int step)
    {
        var bw = LoadWriter(w, meta);
        if (step == 1)
        {
            if (meta[WebPEncMeta.UsePredictor] == 1)
            {
                Vp8lEncodeGpu.PutBits(ref bw, output, 1, 1);
                Vp8lEncodeGpu.PutBits(ref bw, output, 0, 2);
                Vp8lEncodeGpu.PutBits(ref bw, output, (uint)(meta[WebPEncMeta.PredictorBits] - 2), 3);
            }
        }
        else if (step == 2)
        {
            if (meta[WebPEncMeta.UseCrossColor] == 1)
            {
                Vp8lEncodeGpu.PutBits(ref bw, output, 1, 1);
                Vp8lEncodeGpu.PutBits(ref bw, output, 1, 2);
                Vp8lEncodeGpu.PutBits(ref bw, output, (uint)(meta[WebPEncMeta.CrossColorBits] - 2), 3);
            }
        }
        else
        {
            Vp8lEncodeGpu.PutBits(ref bw, output, 0, 1); // no more transforms
            int cacheBits = meta[WebPEncMeta.CacheBits];
            if (cacheBits > 0)
            {
                Vp8lEncodeGpu.PutBits(ref bw, output, 1, 1);
                Vp8lEncodeGpu.PutBits(ref bw, output, (uint)cacheBits, 4);
            }
            else
            {
                Vp8lEncodeGpu.PutBits(ref bw, output, 0, 1);
            }
            if (meta[WebPEncMeta.Groups] > 1)
            {
                // meta prefix codes: the tile size; the entropy image (group per tile) and every group's codes follow
                Vp8lEncodeGpu.PutBits(ref bw, output, 1, 1);
                Vp8lEncodeGpu.PutBits(ref bw, output, (uint)(meta[WebPEncMeta.MetaBits] - 2), 3);
            }
            else
            {
                Vp8lEncodeGpu.PutBits(ref bw, output, 0, 1); // one group
            }
            int s = Vp8lEncLayout.StepState;
            w[s + Vp8lEncLayout.SEmitT] = 0;
            w[s + Vp8lEncLayout.SEmitX] = 0;
            w[s + Vp8lEncLayout.SEmitY] = 0;
        }
        SaveWriter(w, ref bw);
    }

    /// <summary>One thread per histogram slot of the sub-image code set (SetAlpha(0) threads): clears it.</summary>
    private static void SubImageBeginKernel(Index1D index, ArrayView<int> w, ArrayView<int> meta, int which)
    {
        if (SubImageCount(meta, which) > 0) w[meta[WebPEncMeta.RegionSub] + index.X] = 0;
    }

    /// <summary>
    /// Four threads (green, red, blue, alpha): histogram of chunk <paramref name="step"/> (StepChunk pixels) of the sub-image.
    /// Each thread owns one histogram, so the threads never write the same slot.
    /// </summary>
    private static void SubImageHistKernel(Index1D index, ArrayView<int> w, ArrayView<int> meta, int which, int step)
    {
        int count = SubImageCount(meta, which);
        int first = step * Vp8lEncLayout.StepChunk;
        int end = first + Vp8lEncLayout.StepChunk;
        if (end > count) end = count;
        int src = SubImageSource(meta, which);
        int k = index.X;
        int shift = 8;
        if (k == 1) shift = 16;
        else if (k == 2) shift = 0;
        else if (k == 3) shift = 24;
        int hist = meta[WebPEncMeta.RegionSub] + Vp8lEncodeGpu.AlphaOff(k, 0);
        for (int i = first; i < end; i++)
        {
            int v = (int)(((uint)w[src + i] >> shift) & 0xFFu);
            w[hist + v] = w[hist + v] + 1;
        }
    }

    /// <summary>Five threads: the Huffman code of each of the sub-image's five codes (each in its own scratch).</summary>
    private static void SubImageBuildKernel(Index1D index, ArrayView<int> w, ArrayView<int> meta, int which)
    {
        if (SubImageCount(meta, which) > 0)
        {
            int k = index.X;
            Vp8lEncodeGpu.BuildOneCode(w, meta[WebPEncMeta.RegionSub], 0, k, meta[WebPEncMeta.RegionBuild] + k * Vp8lEncLayout.BuildScratch);
        }
    }

    /// <summary>
    /// Writes the definition of code <paramref name="k"/> of one code set: <paramref name="set"/> -1 = the sub-image
    /// <paramref name="which"/> (preceded by its "no colour cache" bit at k = 0), otherwise prefix group <paramref name="set"/>
    /// of the main image (the main code set when there is a single group). A set the stream does not have writes nothing.
    /// </summary>
    private static void EmitCodeKernel(Index1D _, ArrayView<byte> output, ArrayView<int> w, ArrayView<int> meta, int set, int k, int which)
    {
        bool active;
        int setBase;
        int cacheSize = 0;
        if (set < 0)
        {
            active = SubImageCount(meta, which) > 0;
            setBase = meta[WebPEncMeta.RegionSub];
        }
        else
        {
            int cacheBits = meta[WebPEncMeta.CacheBits];
            cacheSize = cacheBits > 0 ? 1 << cacheBits : 0;
            int groups = meta[WebPEncMeta.Groups];
            if (groups > 1)
            {
                active = set < groups;
                setBase = meta[WebPEncMeta.RegionGroupSets] + set * Vp8lEncLayout.SetInts;
            }
            else
            {
                active = set == 0;
                setBase = meta[WebPEncMeta.RegionMain];
            }
        }
        if (active)
        {
            var bw = LoadWriter(w, meta);
            if ((set < 0) & (k == 0)) Vp8lEncodeGpu.PutBits(ref bw, output, 0, 1); // a sub-image has no colour cache
            int setAlpha = Vp8lEncodeGpu.SetAlpha(cacheSize);
            Vp8lEncodeGpu.WriteCodeDefinition(ref bw, output, w, setBase + setAlpha + Vp8lEncodeGpu.AlphaOff(k, cacheSize),
                Vp8lEncodeGpu.Alpha(k, cacheSize), meta[WebPEncMeta.RegionWriter]);
            SaveWriter(w, ref bw);
        }
    }

    /// <summary>Writes the pixels of chunk <paramref name="step"/> (StepChunk pixels) of the sub-image with its codes.</summary>
    private static void SubImageSymbolsKernel(Index1D _, ArrayView<byte> output, ArrayView<int> w, ArrayView<int> meta, int which, int step)
    {
        int count = SubImageCount(meta, which);
        int first = step * Vp8lEncLayout.StepChunk;
        if (first < count)
        {
            int end = first + Vp8lEncLayout.StepChunk;
            if (end > count) end = count;
            int src = SubImageSource(meta, which);
            int codes = meta[WebPEncMeta.RegionSub] + 2 * Vp8lEncodeGpu.SetAlpha(0);
            int cg = codes + Vp8lEncodeGpu.AlphaOff(0, 0);
            int cr = codes + Vp8lEncodeGpu.AlphaOff(1, 0);
            int cb = codes + Vp8lEncodeGpu.AlphaOff(2, 0);
            int ca = codes + Vp8lEncodeGpu.AlphaOff(3, 0);
            var bw = LoadWriter(w, meta);
            for (int i = first; i < end; i++)
            {
                uint px = (uint)w[src + i];
                Vp8lEncodeGpu.WriteSymbol(ref bw, output, w, cg, (int)((px >> 8) & 0xFFu));
                Vp8lEncodeGpu.WriteSymbol(ref bw, output, w, cr, (int)((px >> 16) & 0xFFu));
                Vp8lEncodeGpu.WriteSymbol(ref bw, output, w, cb, (int)(px & 0xFFu));
                Vp8lEncodeGpu.WriteSymbol(ref bw, output, w, ca, (int)(px >> 24));
            }
            SaveWriter(w, ref bw);
        }
    }

    private static void SaveWriter(ArrayView<int> w, ref Vp8lBitWriter bw)
    {
        int s = Vp8lEncLayout.StepState;
        w[s + Vp8lEncLayout.SEmitPos] = bw.Pos;
        w[s + Vp8lEncLayout.SEmitAcc] = (int)bw.Acc;
        w[s + Vp8lEncLayout.SEmitNbits] = bw.Nbits;
        w[s + Vp8lEncLayout.SEmitOverflow] = bw.Overflow;
    }

    private static Vp8lBitWriter LoadWriter(ArrayView<int> w, ArrayView<int> meta)
    {
        int s = Vp8lEncLayout.StepState;
        return new Vp8lBitWriter
        {
            Pos = w[s + Vp8lEncLayout.SEmitPos],
            Acc = (uint)w[s + Vp8lEncLayout.SEmitAcc],
            Nbits = w[s + Vp8lEncLayout.SEmitNbits],
            Cap = meta[WebPEncMeta.OutputCapacity],
            Overflow = w[s + Vp8lEncLayout.SEmitOverflow],
        };
    }

    /// <summary>The next <see cref="Vp8lEncLayout.StepChunk"/> tokens (a launch after the last token does nothing).</summary>
    private static void EmitTokensKernel(
        Index1D _, ArrayView<byte> output, ArrayView<int> w, ArrayView<int> tok, ArrayView<int> meta, ArrayView<int> tables)
    {
        int s = Vp8lEncLayout.StepState;
        int t = w[s + Vp8lEncLayout.SEmitT];
        if (t < meta[WebPEncMeta.TokenCount])
        {
            var bw = LoadWriter(w, meta);
            int x = w[s + Vp8lEncLayout.SEmitX];
            int y = w[s + Vp8lEncLayout.SEmitY];
            Vp8lEncodeGpu.WriteTokens(ref bw, output, w, tok, meta, tables, meta[WebPEncMeta.PackedWidth], meta[WebPEncMeta.CacheBits],
                ref t, ref x, ref y);
            SaveWriter(w, ref bw);
            w[s + Vp8lEncLayout.SEmitT] = t;
            w[s + Vp8lEncLayout.SEmitX] = x;
            w[s + Vp8lEncLayout.SEmitY] = y;
        }
    }

    /// <summary>Flush the last partial byte; the payload length and the overflow flag.</summary>
    private static void EmitTailKernel(Index1D _, ArrayView<byte> output, ArrayView<int> w, ArrayView<int> meta)
    {
        var bw = LoadWriter(w, meta);
        Vp8lEncodeGpu.Flush(ref bw, output);
        meta[WebPEncMeta.PayloadLength] = bw.Pos - meta[WebPEncMeta.PayloadStart];
        meta[WebPEncMeta.Error] = bw.Overflow;
    }

    // ------------------------------------------------------------------
    // 11. Container
    // ------------------------------------------------------------------

    private static void PutLe32(ArrayView<byte> o, int p, int v)
    {
        o[p] = (byte)(v & 0xFF);
        o[p + 1] = (byte)((v >> 8) & 0xFF);
        o[p + 2] = (byte)((v >> 16) & 0xFF);
        o[p + 3] = (byte)((v >> 24) & 0xFF);
    }

    private static void PutFcc(ArrayView<byte> o, int p, int a, int b, int c, int d)
    {
        o[p] = (byte)a;
        o[p + 1] = (byte)b;
        o[p + 2] = (byte)c;
        o[p + 3] = (byte)d;
    }

    private static void FinalizeKernel(Index1D _, ArrayView<byte> o, ArrayView<int> meta)
    {
        int cap = meta[WebPEncMeta.OutputCapacity];
        int start = meta[WebPEncMeta.PayloadStart];
        int payload = meta[WebPEncMeta.PayloadLength];
        int icc = meta[WebPEncMeta.IccLength];
        int exif = meta[WebPEncMeta.ExifLength];
        int xmp = meta[WebPEncMeta.XmpLength];
        int extended = meta[WebPEncMeta.ExtendedFile];
        int end = start + payload + (payload & 1);
        int p = end;
        int exifHeader = 0;
        int xmpHeader = 0;
        if (exif > 0)
        {
            exifHeader = p;
            meta[WebPEncMeta.ExifDst] = p + 8;
            p += 8 + exif + (exif & 1);
        }
        if (xmp > 0)
        {
            xmpHeader = p;
            meta[WebPEncMeta.XmpDst] = p + 8;
            p += 8 + xmp + (xmp & 1);
        }
        meta[WebPEncMeta.Total] = p;
        if ((meta[WebPEncMeta.Error] == 0) & (p <= cap))
        {
            PutFcc(o, 0, 'R', 'I', 'F', 'F');
            PutLe32(o, 4, p - 8);
            PutFcc(o, 8, 'W', 'E', 'B', 'P');
            int q = 12;
            if (extended == 1)
            {
                int flags = 0;
                if (meta[WebPEncMeta.HasAlpha] == 1) flags |= 0x10;
                if (icc > 0) flags |= 0x20;
                if (exif > 0) flags |= 0x08;
                if (xmp > 0) flags |= 0x04;
                int w1 = meta[WebPEncMeta.Width] - 1;
                int h1 = meta[WebPEncMeta.Height] - 1;
                PutFcc(o, q, 'V', 'P', '8', 'X');
                PutLe32(o, q + 4, 10);
                o[q + 8] = (byte)flags;
                o[q + 9] = 0;
                o[q + 10] = 0;
                o[q + 11] = 0;
                o[q + 12] = (byte)(w1 & 0xFF);
                o[q + 13] = (byte)((w1 >> 8) & 0xFF);
                o[q + 14] = (byte)((w1 >> 16) & 0xFF);
                o[q + 15] = (byte)(h1 & 0xFF);
                o[q + 16] = (byte)((h1 >> 8) & 0xFF);
                o[q + 17] = (byte)((h1 >> 16) & 0xFF);
                q += 18;
                if (icc > 0)
                {
                    PutFcc(o, q, 'I', 'C', 'C', 'P');
                    PutLe32(o, q + 4, icc);
                    meta[WebPEncMeta.IccDst] = q + 8;
                    if ((icc & 1) == 1) o[q + 8 + icc] = 0;
                }
            }
            PutFcc(o, start - 8, 'V', 'P', '8', 'L');
            PutLe32(o, start - 4, payload);
            if ((payload & 1) == 1) o[start + payload] = 0;
            if (exif > 0)
            {
                PutFcc(o, exifHeader, 'E', 'X', 'I', 'F');
                PutLe32(o, exifHeader + 4, exif);
                if ((exif & 1) == 1) o[exifHeader + 8 + exif] = 0;
            }
            if (xmp > 0)
            {
                PutFcc(o, xmpHeader, 'X', 'M', 'P', ' ');
                PutLe32(o, xmpHeader + 4, xmp);
                if ((xmp & 1) == 1) o[xmpHeader + 8 + xmp] = 0;
            }
        }
    }

    private static void CopyBlobKernel(
        Index1D index, ArrayView<byte> blobs, ArrayView<byte> output, ArrayView<int> meta, int srcOffset, int dstSlot)
    {
        bool ok = meta[WebPEncMeta.Error] == 0;
        if (ok) output[meta[dstSlot] + index.X] = blobs[srcOffset + index.X];
    }

    // ------------------------------------------------------------------
    // 12. ALPH: the alpha plane of a lossy image
    // ------------------------------------------------------------------

    /// <summary>
    /// ALPH prediction of sample (x, y) from the ORIGINAL plane (so every sample is independent): filter 1 = left, 2 = above,
    /// 3 = clip(left + above - aboveLeft); the top row is predicted from the left, the left column from above, (0, 0) from 0.
    /// </summary>
    private static int AlphaPrediction(ArrayView<byte> p, int x, int y, int w, int filter)
    {
        int i = y * w + x;
        bool notFirst = (x != 0) | (y != 0);
        bool filtered = (filter != 0) & notFirst;
        int pred = 0;
        if (filtered)
        {
            if (y == 0)
            {
                pred = p[i - 1];
            }
            else if (x == 0)
            {
                pred = p[i - w];
            }
            else if (filter == 1)
            {
                pred = p[i - 1];
            }
            else if (filter == 2)
            {
                pred = p[i - w];
            }
            else
            {
                int g = (int)p[i - 1] + (int)p[i - w] - (int)p[i - w - 1];
                pred = g < 0 ? 0 : (g > 255 ? 255 : g);
            }
        }
        return pred;
    }

    /// <summary>Elias-gamma-like code length of a residual's signed magnitude: the smoother the filtered plane, the cheaper.</summary>
    private static int AlphaResidualCost(int residual)
    {
        int m = residual < 128 ? residual : 256 - residual;
        int bits = 0;
        int t = m;
        while (t > 0)
        {
            bits++;
            t >>= 1;
        }
        return 1 + 2 * bits;
    }

    private static void AlphaCostKernel(Index1D index, ArrayView<byte> plane, ArrayView<int> rowCost, ArrayView<int> meta)
    {
        int w = meta[WebPEncMeta.Width];
        int h = meta[WebPEncMeta.Height];
        int filter = index.X / h;
        int y = index.X - filter * h;
        // keep the sum of all rows inside an int for planes of up to 2^28 samples
        int shift = w > (1 << 26) / h ? 3 : 0; // w * h > 2^26 without a 64-bit product
        int cost = 0;
        for (int x = 0; x < w; x++)
        {
            int r = (plane[y * w + x] - AlphaPrediction(plane, x, y, w, filter)) & 0xFF;
            cost += AlphaResidualCost(r);
        }
        rowCost[index.X] = cost >> shift;
    }

    private static void AlphaPickKernel(Index1D _, ArrayView<int> rowCost, ArrayView<int> meta)
    {
        int h = meta[WebPEncMeta.Height];
        int request = meta[WebPEncMeta.AlphaFilterRequest];
        int chosen = request;
        if (request >= WebPEncMeta.AlphaAuto)
        {
            int best = 0;
            int bestCost = int.MaxValue;
            for (int f = 0; f < 4; f++)
            {
                int sum = 0;
                for (int y = 0; y < h; y++) sum += rowCost[f * h + y];
                if (sum < bestCost)
                {
                    bestCost = sum;
                    best = f;
                }
            }
            chosen = best;
        }
        meta[WebPEncMeta.AlphaFilter] = chosen;
    }

    private static void AlphaToArgbKernel(Index1D index, ArrayView<byte> plane, ArrayView<uint> argb, ArrayView<int> meta)
    {
        int w = meta[WebPEncMeta.Width];
        int filter = meta[WebPEncMeta.AlphaFilter];
        int i = index.X;
        int y = i / w;
        int x = i - y * w;
        int r = (plane[i] - AlphaPrediction(plane, x, y, w, filter)) & 0xFF;
        argb[i] = 0xff000000u | ((uint)r << 8);
    }

    private static void AlphaRawKernel(Index1D index, ArrayView<byte> plane, ArrayView<byte> output, ArrayView<int> meta)
    {
        int w = meta[WebPEncMeta.Width];
        int filter = meta[WebPEncMeta.AlphaFilter];
        int i = index.X;
        int y = i / w;
        int x = i - y * w;
        output[1 + i] = (byte)((plane[i] - AlphaPrediction(plane, x, y, w, filter)) & 0xFF);
    }

    private static void FinalizeAlphaKernel(Index1D _, ArrayView<byte> o, ArrayView<int> meta, int raw)
    {
        int n = meta[WebPEncMeta.Width] * meta[WebPEncMeta.Height];
        // The compressed stream is only allowed 1 + n bytes (its capacity): when it did not fit, the plane is stored raw instead
        // and the overflow the writer flagged is not an error.
        if (raw == 1) meta[WebPEncMeta.Error] = 0;
        int total = 1 + (raw == 1 ? n : meta[WebPEncMeta.PayloadLength]);
        meta[WebPEncMeta.Total] = total;
        bool fits = (meta[WebPEncMeta.Error] == 0) & (total <= meta[WebPEncMeta.OutputCapacity]);
        if (fits)
        {
            // Rsv:2 = 0 | P:2 = 0 (no pre-processing) | F:2 = filter | C:2 = 0 raw, 1 lossless
            o[0] = (byte)((meta[WebPEncMeta.AlphaFilter] << 2) | (raw == 1 ? 0 : 1));
        }
    }

    // ------------------------------------------------------------------
    // 13. Animation container
    // ------------------------------------------------------------------

    private static void PutLe24(ArrayView<byte> o, int p, int v)
    {
        o[p] = (byte)(v & 0xFF);
        o[p + 1] = (byte)((v >> 8) & 0xFF);
        o[p + 2] = (byte)((v >> 16) & 0xFF);
    }

    /// <summary>ANMF header (8) + ANMF parameters (16) + VP8L chunk header (8) in front of the stream at PayloadStart.</summary>
    private static void FinalizeFrameKernel(Index1D _, ArrayView<byte> o, ArrayView<int> meta)
    {
        int start = meta[WebPEncMeta.PayloadStart];
        int payload = meta[WebPEncMeta.PayloadLength];
        int end = start + payload + (payload & 1);
        meta[WebPEncMeta.Total] = end;
        bool fits = (meta[WebPEncMeta.Error] == 0) & (end <= meta[WebPEncMeta.OutputCapacity]);
        if (fits)
        {
            int b = start - 32;
            PutFcc(o, b, 'A', 'N', 'M', 'F');
            PutLe32(o, b + 4, 16 + 8 + payload + (payload & 1));
            PutLe24(o, b + 8, meta[WebPEncMeta.FrameX] / 2);
            PutLe24(o, b + 11, meta[WebPEncMeta.FrameY] / 2);
            PutLe24(o, b + 14, meta[WebPEncMeta.Width] - 1);
            PutLe24(o, b + 17, meta[WebPEncMeta.Height] - 1);
            PutLe24(o, b + 20, meta[WebPEncMeta.FrameDuration]);
            o[b + 23] = (byte)meta[WebPEncMeta.FrameFlags];
            PutFcc(o, start - 8, 'V', 'P', '8', 'L');
            PutLe32(o, start - 4, payload);
            if ((payload & 1) == 1) o[start + payload] = 0;
        }
    }

    private static void FinalizeAnimKernel(Index1D _, ArrayView<byte> o, ArrayView<int> meta)
    {
        int cap = meta[WebPEncMeta.OutputCapacity];
        int icc = meta[WebPEncMeta.IccLength];
        int exif = meta[WebPEncMeta.ExifLength];
        int xmp = meta[WebPEncMeta.XmpLength];
        int p = meta[WebPEncMeta.AnimEnd];
        int exifHeader = 0;
        int xmpHeader = 0;
        if (exif > 0)
        {
            exifHeader = p;
            meta[WebPEncMeta.ExifDst] = p + 8;
            p += 8 + exif + (exif & 1);
        }
        if (xmp > 0)
        {
            xmpHeader = p;
            meta[WebPEncMeta.XmpDst] = p + 8;
            p += 8 + xmp + (xmp & 1);
        }
        meta[WebPEncMeta.Total] = p;
        if ((meta[WebPEncMeta.Error] == 0) & (p <= cap))
        {
            PutFcc(o, 0, 'R', 'I', 'F', 'F');
            PutLe32(o, 4, p - 8);
            PutFcc(o, 8, 'W', 'E', 'B', 'P');
            int flags = 0x02;
            if (meta[WebPEncMeta.AnimAlpha] == 1) flags |= 0x10;
            if (icc > 0) flags |= 0x20;
            if (exif > 0) flags |= 0x08;
            if (xmp > 0) flags |= 0x04;
            int w1 = meta[WebPEncMeta.CanvasWidth] - 1;
            int h1 = meta[WebPEncMeta.CanvasHeight] - 1;
            int q = 12;
            PutFcc(o, q, 'V', 'P', '8', 'X');
            PutLe32(o, q + 4, 10);
            o[q + 8] = (byte)flags;
            o[q + 9] = 0;
            o[q + 10] = 0;
            o[q + 11] = 0;
            PutLe24(o, q + 12, w1);
            PutLe24(o, q + 15, h1);
            q += 18;
            if (icc > 0)
            {
                PutFcc(o, q, 'I', 'C', 'C', 'P');
                PutLe32(o, q + 4, icc);
                meta[WebPEncMeta.IccDst] = q + 8;
                if ((icc & 1) == 1) o[q + 8 + icc] = 0;
                q += 8 + icc + (icc & 1);
            }
            PutFcc(o, q, 'A', 'N', 'I', 'M');
            PutLe32(o, q + 4, 6);
            PutLe32(o, q + 8, meta[WebPEncMeta.Background]);
            o[q + 12] = (byte)(meta[WebPEncMeta.LoopCount] & 0xFF);
            o[q + 13] = (byte)((meta[WebPEncMeta.LoopCount] >> 8) & 0xFF);
            if (exif > 0)
            {
                PutFcc(o, exifHeader, 'E', 'X', 'I', 'F');
                PutLe32(o, exifHeader + 4, exif);
                if ((exif & 1) == 1) o[exifHeader + 8 + exif] = 0;
            }
            if (xmp > 0)
            {
                PutFcc(o, xmpHeader, 'X', 'M', 'P', ' ');
                PutLe32(o, xmpHeader + 4, xmp);
                if ((xmp & 1) == 1) o[xmpHeader + 8 + xmp] = 0;
            }
        }
    }

    /// <summary>No-op (kernels are owned by the accelerator).</summary>
    public void Dispose()
    {
        _ptScratch?.Dispose();
        _peWords?.Dispose();
    }
}
