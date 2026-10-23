// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// WebP lossless (VP8L) encoder, 100% accelerator-resident. The host is a pure coordinator: it uploads the pixels
// (necessary I/O), dispatches the WebPEncodeKernels chain, reads back a few scalars (the output length, the cost of
// a colour-cache variant) and uploads the tiny ICC / EXIF / XMP blobs. The ARGB working images, the transform
// sub-images, the LZ77 tokens, the prefix codes and the bitstream never touch the host; the finished file is read
// back only into a caller-requested managed / JS sink.
//
// API shape (same as the video codecs and PngDecoderGpu): three overloads per entry point, all funnelling into the
// ArrayView1D one.
//   - ArrayView1D : RGBA already on the accelerator, the file written into a caller-owned view, returns the length.
//   - TypedArray  : browser. JS bytes in via CopyFromJS, a JS Uint8Array out; nothing enters the .NET heap.
//   - ReadOnlyMemory<byte> : managed. Desktop and small inputs; the file comes back as byte[].
//
// The output decodes to EXACTLY the input pixels in libwebp (and WebPDecoderGpu). With WebPEncodeOptions.Exact = false
// the RGB of fully transparent pixels is not preserved (it is cleared, which compresses better).
//
// Also in this class (partial files): EncodeAlphaAsync (WebPLosslessEncoderGpu.Alpha.cs, the ALPH chunk of a lossy WebP) and
// EncodeAnimationAsync (WebPLosslessEncoderGpu.Animation.cs). All three build on EncodeStreamAsync: the pixel half (colour
// analysis, strategy trials, transforms, entropy coding, bitstream) that writes one VP8L stream where the caller says; the
// container around it (RIFF, VP8X, ANMF, ALPH header byte) is the caller's.

using System.Runtime.Versioning;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.SpawnJS.JSObjects;

namespace SpawnDev.Codecs.Images.WebP;

/// <summary>WebP lossless encoder running on an ILGPU accelerator.</summary>
/// <remarks>Not thread-safe: one in-flight call per instance (the working buffers are shared across calls).</remarks>
public sealed partial class WebPLosslessEncoderGpu : IDisposable
{
    private const int MaxDimension = 16384;

    private readonly Accelerator _accelerator;
    private readonly WebPEncodeKernels _kernels;

    private readonly MemoryBuffer1D<int, Stride1D.Dense> _meta;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _tables;

    private readonly StageBuffer<byte> _stageIn;
    private readonly StageBuffer<byte> _stageOut;
    private readonly StageBuffer<uint> _argbA;
    private readonly StageBuffer<uint> _argbB;
    private readonly StageBuffer<int> _work;
    private readonly StageBuffer<int> _tokens;
    private readonly StageBuffer<byte> _blobs;
    private readonly StageBuffer<int> _rowCost;

    /// <summary>Compile the kernels and upload the constant tables.</summary>
    public WebPLosslessEncoderGpu(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernels = new WebPEncodeKernels(accelerator);
        _meta = accelerator.Allocate1D<int>(WebPEncMeta.Size);
        _tables = accelerator.Allocate1D<int>(Vp8lTables.Size);
        _tables.View.CopyFromCPU(Vp8lTables.Build());
        _stageIn = new StageBuffer<byte>(accelerator);
        _stageOut = new StageBuffer<byte>(accelerator);
        _argbA = new StageBuffer<uint>(accelerator);
        _argbB = new StageBuffer<uint>(accelerator);
        _work = new StageBuffer<int>(accelerator);
        _tokens = new StageBuffer<int>(accelerator);
        _blobs = new StageBuffer<byte>(accelerator);
        _rowCost = new StageBuffer<int>(accelerator);
    }

    // ------------------------------------------------------------------
    // Sizes and plan
    // ------------------------------------------------------------------

    private static byte[]? ExifPayload(byte[]? exif)
    {
        if (exif is null || exif.Length == 0) return null;
        // The EXIF chunk holds the bare TIFF structure; drop an "Exif\0\0" prefix if the caller kept it.
        if (exif.Length >= 6 && exif[0] == (byte)'E' && exif[1] == (byte)'x' && exif[2] == (byte)'i' && exif[3] == (byte)'f'
            && exif[4] == 0 && exif[5] == 0)
            return exif.AsSpan(6).ToArray();
        return exif;
    }

    /// <summary>
    /// Upper bound on the size of the file for a <paramref name="width"/> x <paramref name="height"/> image, for sizing the
    /// output buffer of <see cref="EncodeAsync(ArrayView1D{byte, Stride1D.Dense}, int, int, ArrayView1D{byte, Stride1D.Dense}, WebPEncodeOptions?)"/>.
    /// </summary>
    public static int GetMaxEncodedSize(int width, int height, WebPEncodeOptions? options = null)
    {
        ValidateSize(width, height);
        long size = 12 + 18 + 4 * 8 + 5L * width * height + 64 * 1024;
        if (options?.IccProfile is { } icc) size += icc.Length + 1;
        if (ExifPayload(options?.Exif) is { } exif) size += exif.Length + 1;
        if (options?.Xmp is { } xmp) size += xmp.Length + 1;
        if (size > int.MaxValue) throw new ArgumentOutOfRangeException(nameof(width), "Image is too large to encode.");
        return (int)size;
    }

    private static void ValidateSize(int width, int height)
    {
        if (width < 1 || width > MaxDimension) throw new ArgumentOutOfRangeException(nameof(width), $"Width must be 1..{MaxDimension}.");
        if (height < 1 || height > MaxDimension) throw new ArgumentOutOfRangeException(nameof(height), $"Height must be 1..{MaxDimension}.");
    }

    private sealed class Plan
    {
        public int Quality;
        public int PredictorBits, CrossBits, ModeSet, ChainDepth;
        public int TrialCache;
        public int MetaBits;
        public int[] CacheCandidates = { 0 };
    }

    /// <summary>One way of turning the image into the thing the entropy coder sees.</summary>
    private readonly record struct Strategy(bool Palette, bool SubtractGreen, bool Predictor, bool Cross);

    private static Plan MakePlan(int quality, int width, int height)
    {
        long n = (long)width * height;
        var p = new Plan { Quality = quality };
        p.PredictorBits = n <= 4096 ? 3 : n >= (4L << 20) ? 5 : 4;
        p.CrossBits = n <= 4096 ? 3 : n >= (4L << 20) ? 6 : 5;
        if (quality < 25)
        {
            p.ChainDepth = 1;
        }
        else if (quality < 50)
        {
            p.ChainDepth = 4;
        }
        else if (quality < 75)
        {
            p.ChainDepth = 16;
            p.TrialCache = 8;
            p.CacheCandidates = new[] { 0, 8 };
        }
        else
        {
            p.ModeSet = 1;
            p.ChainDepth = quality >= 90 ? 128 : 32;
            p.TrialCache = 8;
            p.CacheCandidates = quality >= 90 ? new[] { 0, 5, 7, 9, 10 } : new[] { 0, 6, 9 };
        }
        if (quality >= 50)
        {
            int mb = Vp8lEncodeGroups.TileBitsFor(width, height);
            if (Vp8lEncLayout.Tiles(width, height, mb) >= 4) p.MetaBits = mb;
        }
        return p;
    }

    /// <summary>The strategies worth trying for this quality, given how many colours the image has (0 = more than 256).</summary>
    private static List<Strategy> MakeStrategies(Plan plan, int paletteColors, bool plane = false)
    {
        int q = plan.Quality;
        var list = new List<Strategy>();
        if (plane)
        {
            // An alpha plane is green only (red and blue are 0): subtract green and cross colour cannot help. At most 256
            // distinct values, so a palette (with pixel bundling for few levels) is always possible.
            list.Add(new Strategy(true, false, false, false));
            if (q >= 25) list.Add(new Strategy(true, false, true, false));
            if (q >= 50)
            {
                list.Add(new Strategy(false, false, false, false));
                list.Add(new Strategy(false, false, true, false));
            }
            return list;
        }
        if ((paletteColors > 0) && (paletteColors <= 16 || q < 25))
        {
            list.Add(new Strategy(true, false, false, false));
            return list;
        }
        if (q < 50)
        {
            // no trial of the raw image below 50: the predictor wins on most pictures
            list.Add(new Strategy(false, true, true, false));
        }
        else
        {
            list.Add(new Strategy(false, false, false, false));
            if (q >= 75) list.Add(new Strategy(false, true, false, false));
            list.Add(new Strategy(false, true, true, false));
            if (q >= 60) list.Add(new Strategy(false, true, true, true));
        }
        if (paletteColors > 0)
        {
            list.Add(new Strategy(true, false, false, false));
            list.Add(new Strategy(true, false, true, false));
        }
        return list;
    }

    private static int PaletteBitsFor(int colors) => colors <= 2 ? 3 : colors <= 4 ? 2 : colors <= 16 ? 1 : 0;

    // ------------------------------------------------------------------
    // Encode
    // ------------------------------------------------------------------

    /// <summary>
    /// What <see cref="EncodeStreamAsync"/> encodes: an image of the given size and effort whose ARGB form the callback writes
    /// (called once per strategy trial, because the transforms work in place). <paramref name="Plane"/> marks an alpha plane
    /// (green only: palette / predictor strategies, no 5-byte VP8L header). <paramref name="AfterMeta"/> runs once the
    /// meta block is on the accelerator, before the first ARGB conversion.
    /// </summary>
    private sealed record StreamJob(
        int Width, int Height, int Quality, Action<ArrayView<uint>> ToArgb, bool Plane, Action? AfterMeta = null);

    /// <summary>
    /// The pixel half of every encode: colour analysis, strategy trials, transforms, entropy coding and the bitstream. Writes
    /// the VP8L stream at <c>hostMeta[PayloadStart]</c> of <paramref name="output"/> and leaves PayloadLength, Error and
    /// HasAlpha in the device meta block; the caller finishes the container around it. The caller fills the file-level host
    /// slots of <paramref name="hostMeta"/> (OutputCapacity, PayloadStart, ExtendedFile, blob lengths, frame fields, alpha
    /// filter request); this method fills the stream-level ones and uploads the block.
    /// </summary>
    private async Task EncodeStreamAsync(StreamJob job, int[] hostMeta, ArrayView1D<byte, Stride1D.Dense> output)
    {
        int width = job.Width;
        int height = job.Height;
        long pixels = (long)width * height;
        var plan = MakePlan(job.Quality, width, height);

        // ---- host-derived layout and choices go into the shared int block ----
        long workInts = Vp8lEncLayout.Compute(width, height, plan.PredictorBits, plan.CrossBits, plan.MetaBits, hostMeta);
        if (workInts > int.MaxValue) throw new ArgumentOutOfRangeException(nameof(job), "Image is too large to encode.");
        hostMeta[WebPEncMeta.Width] = width;
        hostMeta[WebPEncMeta.Height] = height;
        hostMeta[WebPEncMeta.PackedWidth] = width;
        hostMeta[WebPEncMeta.PredictorBits] = plan.PredictorBits;
        hostMeta[WebPEncMeta.CrossColorBits] = plan.CrossBits;
        hostMeta[WebPEncMeta.AllowPalette] = 1;
        hostMeta[WebPEncMeta.ChainDepth] = plan.ChainDepth;
        hostMeta[WebPEncMeta.ModeSet] = plan.ModeSet;
        hostMeta[WebPEncMeta.MetaBits] = plan.MetaBits;
        hostMeta[WebPEncMeta.Headerless] = job.Plane ? 1 : 0;

        // ---- buffers (grown at the start of the call) ----
        _argbA.Ensure(pixels);
        _argbB.Ensure(pixels);
        _work.Ensure(workInts);
        _tokens.Ensure(pixels * 2);
        _meta.View.CopyFromCPU(hostMeta);

        var a = _argbA.View;
        var bb = _argbB.View;
        var work = _work.View;
        var tok = _tokens.View;
        var meta = _meta.View;
        int n = (int)pixels;
        job.AfterMeta?.Invoke();

        // ---- colour analysis, once, on the untouched pixels ----
        job.ToArgb(a);
        _kernels.Analyze(a, work, meta, pixels);
        await _accelerator.SynchronizeAsync();
        var analysis = await meta.CopyToHostAsync();
        int paletteColors = analysis[WebPEncMeta.PaletteAvailable];

        // Applies a strategy: writes the host-side choices and runs the (parallel) transform kernels; returns the main image.
        ArrayView<uint> Apply(Strategy s)
        {
            int packed = width;
            hostMeta[WebPEncMeta.PaletteCount] = 0;
            hostMeta[WebPEncMeta.PaletteBits] = 0;
            if (s.Palette)
            {
                hostMeta[WebPEncMeta.PaletteCount] = paletteColors;
                int pbits = PaletteBitsFor(paletteColors);
                hostMeta[WebPEncMeta.PaletteBits] = pbits;
                packed = (width + (1 << pbits) - 1) >> pbits;
            }
            hostMeta[WebPEncMeta.PackedWidth] = packed;
            hostMeta[WebPEncMeta.UseSubtractGreen] = s.SubtractGreen ? 1 : 0;
            hostMeta[WebPEncMeta.UsePredictor] = s.Predictor ? 1 : 0;
            hostMeta[WebPEncMeta.UseCrossColor] = s.Cross ? 1 : 0;
            meta.SubView(WebPEncMeta.HostStart, WebPEncMeta.Size - WebPEncMeta.HostStart)
                .CopyFromCPU(ref hostMeta[WebPEncMeta.HostStart], WebPEncMeta.Size - WebPEncMeta.HostStart);

            long packedPixels = (long)packed * height;
            job.ToArgb(a);
            ArrayView<uint> cur;
            if (s.Palette)
            {
                _kernels.PaletteIndex(a, bb, work, meta, n);
                cur = bb;
                if (s.Predictor)
                {
                    _kernels.PredictorMode(cur, work, meta, Vp8lEncLayout.Tiles(packed, height, plan.PredictorBits));
                    _kernels.PredictorResidual(cur, a, work, meta, (int)packedPixels);
                    cur = a;
                }
            }
            else
            {
                cur = a;
                if (s.SubtractGreen) _kernels.SubtractGreen(a, meta, n);
                if (s.Predictor)
                {
                    _kernels.PredictorMode(a, work, meta, Vp8lEncLayout.Tiles(width, height, plan.PredictorBits));
                    _kernels.PredictorResidual(a, bb, work, meta, n);
                    cur = bb;
                }
                if (s.Cross)
                {
                    _kernels.CrossColorMode(cur, work, meta, Vp8lEncLayout.Tiles(width, height, plan.CrossBits));
                    _kernels.CrossColor(cur, work, meta, n);
                }
            }
            return cur;
        }

        // Entropy-codes the main image with a colour cache size and returns the estimated size in bits.
        async Task<long> CostAsync(ArrayView<uint> main, int cacheBits, bool readBack)
        {
            _kernels.Tokenize(main, work, tok, meta, _tables.View, cacheBits, (long)hostMeta[WebPEncMeta.PackedWidth] * height);
            _kernels.BuildCodes(work, meta);
            if (!readBack) return 0;
            await _accelerator.SynchronizeAsync();
            var m = await meta.CopyToHostAsync();
            long cost = m[WebPEncMeta.ExtraBits];
            for (int k = 0; k < 5; k++) cost += m[WebPEncMeta.CodeCost + k];
            return cost;
        }

        // ---- choose the strategy by the estimated size of the entropy-coded result ----
        var strategies = MakeStrategies(plan, paletteColors, job.Plane);
        Strategy chosen = strategies[0];
        ArrayView<uint> main;
        if (strategies.Count == 1)
        {
            main = Apply(chosen);
        }
        else
        {
            long bestCost = long.MaxValue;
            Strategy last = chosen;
            main = a;
            foreach (var s in strategies)
            {
                var cur = Apply(s);
                long cost = await CostAsync(cur, plan.TrialCache, true);
                // side information the transforms add to the stream
                if (s.Palette) cost += paletteColors * 10L + 200;
                if (s.Predictor) cost += Vp8lEncLayout.Tiles(hostMeta[WebPEncMeta.PackedWidth], height, plan.PredictorBits) * 4L + 400;
                if (s.Cross) cost += Vp8lEncLayout.Tiles(width, height, plan.CrossBits) * 6L + 400;
                if (cost < bestCost)
                {
                    bestCost = cost;
                    chosen = s;
                }
                last = s;
                main = cur;
            }
            if (chosen != last) main = Apply(chosen);
        }

        // ---- pick the colour cache size that gives the smallest estimate, then entropy-code for real ----
        int bestCache = plan.CacheCandidates[0];
        if (plan.CacheCandidates.Length > 1)
        {
            long bestCost = long.MaxValue;
            foreach (int cb in plan.CacheCandidates)
            {
                long cost = await CostAsync(main, cb, true);
                // a bigger cache widens the green alphabet: its code definition costs more to describe
                if (cb > 0) cost += 64L << Math.Min(cb - 4, 6);
                if (cost < bestCost)
                {
                    bestCost = cost;
                    bestCache = cb;
                }
            }
        }
        await CostAsync(main, bestCache, false);

        // ---- several prefix groups (meta prefix codes), when the tiles of the image differ enough to pay for them ----
        if (plan.MetaBits > 0)
        {
            await _accelerator.SynchronizeAsync();
            var single = await meta.CopyToHostAsync();
            long singleCost = single[WebPEncMeta.ExtraBits];
            for (int k = 0; k < 5; k++) singleCost += single[WebPEncMeta.CodeCost + k];

            int tileCount = Vp8lEncLayout.Tiles(hostMeta[WebPEncMeta.PackedWidth], height, plan.MetaBits);
            int costBase = hostMeta[WebPEncMeta.RegionGroupMisc] + Vp8lEncodeGroups.CostTableOffset;
            _kernels.TileHistograms(work, tok, meta, _tables.View, tileCount, (long)hostMeta[WebPEncMeta.PackedWidth] * height);
            long bestGroupsCost = long.MaxValue;
            int bestGroups = 0;
            int lastTried = 0;
            foreach (int k in new[] { 2, 4, 8 })
            {
                if (k > tileCount * 2) break;
                _kernels.Cluster(work, meta, k, tileCount);
                _kernels.BuildGroupCodes(work, meta);
                await _accelerator.SynchronizeAsync();
                var m = await meta.CopyToHostAsync();
                int groups = m[WebPEncMeta.Groups];
                var costs = await work.SubView(costBase, Vp8lEncodeGroups.MaxGroups * 5).CopyToHostAsync();
                long cost = m[WebPEncMeta.ExtraBits] + tileCount * 4L + 400;
                for (int i = 0; i < groups * 5; i++) cost += costs[i];
                lastTried = k;
                if (groups > 1 && cost < bestGroupsCost)
                {
                    bestGroupsCost = cost;
                    bestGroups = k;
                }
                if (groups < k) break; // fewer clusters than asked for: more would not help
            }
            if (bestGroups != 0 && bestGroupsCost + 64 < singleCost)
            {
                if (lastTried != bestGroups)
                {
                    _kernels.Cluster(work, meta, bestGroups, tileCount);
                    _kernels.BuildGroupCodes(work, meta);
                }
            }
            else
            {
                meta.SubView(WebPEncMeta.Groups, 1).CopyFromCPU(new[] { 1 });
            }
        }
        _kernels.Emit(output, work, tok, meta, _tables.View, (long)hostMeta[WebPEncMeta.PackedWidth] * height);
    }

    /// <summary>The ICC / EXIF / XMP payloads uploaded to one device buffer (small host blobs), in that order.</summary>
    private ArrayView1D<byte, Stride1D.Dense> UploadBlobs(byte[]? icc, byte[]? exif, byte[]? xmp)
    {
        int blobTotal = (icc?.Length ?? 0) + (exif?.Length ?? 0) + (xmp?.Length ?? 0);
        var blobs = _blobs.Ensure(Math.Max(1, blobTotal));
        if (blobTotal > 0)
        {
            var all = new byte[blobTotal];
            int at = 0;
            foreach (var b in new[] { icc, exif, xmp })
            {
                if (b == null) continue;
                b.CopyTo(all, at);
                at += b.Length;
            }
            DeviceTransfer.Upload(blobs, all);
        }
        return blobs;
    }

    /// <summary>Copies the metadata blobs to the offsets the container kernel recorded.</summary>
    private void CopyBlobs(ArrayView1D<byte, Stride1D.Dense> blobs, ArrayView1D<byte, Stride1D.Dense> output, byte[]? icc, byte[]? exif, byte[]? xmp)
    {
        var meta = _meta.View;
        int offset = 0;
        if (icc != null)
        {
            _kernels.CopyBlob(blobs, output, meta, offset, icc.Length, WebPEncMeta.IccDst);
            offset += icc.Length;
        }
        if (exif != null)
        {
            _kernels.CopyBlob(blobs, output, meta, offset, exif.Length, WebPEncMeta.ExifDst);
            offset += exif.Length;
        }
        if (xmp != null)
            _kernels.CopyBlob(blobs, output, meta, offset, xmp.Length, WebPEncMeta.XmpDst);
    }

    /// <summary>
    /// Encode RGBA8 pixels that are already on the accelerator into a caller-owned view. Only scalars cross to the host.
    /// </summary>
    /// <param name="rgba">Tightly packed RGBA8 pixels (straight alpha, rows top to bottom), at least width x height x 4 bytes.</param>
    /// <param name="width">Width in pixels (1..16384).</param>
    /// <param name="height">Height in pixels (1..16384).</param>
    /// <param name="output">Receives the WebP file; at least <see cref="GetMaxEncodedSize"/> bytes.</param>
    /// <param name="options">Encoder options (null = defaults).</param>
    /// <returns>The length of the file in <paramref name="output"/>.</returns>
    public async Task<int> EncodeAsync(
        ArrayView1D<byte, Stride1D.Dense> rgba, int width, int height,
        ArrayView1D<byte, Stride1D.Dense> output, WebPEncodeOptions? options = null)
    {
        ValidateSize(width, height);
        options ??= new WebPEncodeOptions();
        if (options.Quality < 0 || options.Quality > 100) throw new ArgumentOutOfRangeException(nameof(options), "Quality must be 0..100.");
        long pixels = (long)width * height;
        if (rgba.Length < pixels * 4)
            throw new ArgumentException($"rgba must hold {pixels * 4} bytes for {width}x{height}.", nameof(rgba));
        int maxSize = GetMaxEncodedSize(width, height, options);
        if (output.Length < maxSize)
            throw new ArgumentException($"output must hold {maxSize} bytes (GetMaxEncodedSize) for {width}x{height}.", nameof(output));

        byte[]? icc = options.IccProfile is { Length: > 0 } ? options.IccProfile : null;
        byte[]? exif = ExifPayload(options.Exif);
        byte[]? xmp = options.Xmp is { Length: > 0 } ? options.Xmp : null;
        bool extended = icc != null || exif != null || xmp != null;
        var blobs = UploadBlobs(icc, exif, xmp);

        var hostMeta = new int[WebPEncMeta.Size];
        hostMeta[WebPEncMeta.OutputCapacity] = (int)Math.Min(output.Length, int.MaxValue);
        hostMeta[WebPEncMeta.ExtendedFile] = extended ? 1 : 0;
        hostMeta[WebPEncMeta.IccLength] = icc?.Length ?? 0;
        hostMeta[WebPEncMeta.ExifLength] = exif?.Length ?? 0;
        hostMeta[WebPEncMeta.XmpLength] = xmp?.Length ?? 0;
        int payloadStart = 12 + 8;
        if (extended)
        {
            payloadStart = 12 + 18 + 8;
            if (icc != null) payloadStart += 8 + icc.Length + (icc.Length & 1);
        }
        hostMeta[WebPEncMeta.PayloadStart] = payloadStart;

        int n = (int)pixels;
        int exact = options.Exact ? 1 : 0;
        await EncodeStreamAsync(
            new StreamJob(width, height, options.Quality, a => _kernels.ToArgb(rgba, a, n, exact), Plane: false),
            hostMeta, output);

        _kernels.FinalizeFile(output, _meta.View);
        CopyBlobs(blobs, output, icc, exif, xmp);
        await _accelerator.SynchronizeAsync();

        var result = await _meta.View.CopyToHostAsync();
        if (result[WebPEncMeta.Error] != 0 || result[WebPEncMeta.Total] > output.Length)
            throw new InvalidOperationException(
                $"WebP encode overflowed the output buffer (needed {result[WebPEncMeta.Total]} bytes, GetMaxEncodedSize said {maxSize}).");
        return result[WebPEncMeta.Total];
    }

    // ------------------------------------------------------------------
    // Managed overload
    // ------------------------------------------------------------------

    /// <summary>
    /// Encode managed RGBA8 pixels. Returns the file as a managed array (a terminal CPU sink; in a browser prefer the
    /// <see cref="TypedArray"/> overload).
    /// </summary>
    public async Task<byte[]> EncodeAsync(ReadOnlyMemory<byte> rgba, int width, int height, WebPEncodeOptions? options = null)
    {
        ValidateSize(width, height);
        long needed = (long)width * height * 4;
        if (rgba.Length < needed) throw new ArgumentException($"rgba must hold {needed} bytes for {width}x{height}.", nameof(rgba));
        _stageIn.Ensure(needed);
        DeviceTransfer.Upload(_stageIn.View, rgba.Span.Slice(0, (int)needed));
        _stageOut.Ensure(GetMaxEncodedSize(width, height, options));
        int length = await EncodeAsync(_stageIn.View, width, height, _stageOut.View, options);
        return await DeviceTransfer.DownloadAsync(_stageOut.View, length);
    }

    // ------------------------------------------------------------------
    // TypedArray overload (browser: bulk bytes stay in JS)
    // ------------------------------------------------------------------

    /// <summary>
    /// Encode RGBA8 pixels held in a JS typed array (browser accelerators only). The bytes go to the GPU with
    /// <c>CopyFromJS</c> and the file comes back as a JS <see cref="Uint8Array"/>; none of it enters the .NET heap.
    /// The caller disposes the result.
    /// </summary>
    /// <exception cref="NotSupportedException">The accelerator is not a browser accelerator.</exception>
    [SupportedOSPlatform("browser")]
    public async Task<Uint8Array> EncodeAsync(TypedArray rgba, int width, int height, WebPEncodeOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(rgba);
        ValidateSize(width, height);
        long needed = (long)width * height * 4;
        if (rgba.ByteLength < needed) throw new ArgumentException($"rgba must hold {needed} bytes for {width}x{height}.", nameof(rgba));
        _stageIn.Ensure(needed);
        DeviceTransfer.Upload(_stageIn.View, rgba);
        _stageOut.Ensure(GetMaxEncodedSize(width, height, options));
        int length = await EncodeAsync(_stageIn.View, width, height, _stageOut.View, options);
        return await DeviceTransfer.DownloadUint8ArrayAsync(_stageOut.View, length);
    }

    /// <summary>Release kernel resources and buffers.</summary>
    public void Dispose()
    {
        _kernels.Dispose();
        _meta.Dispose();
        _tables.Dispose();
        _stageIn.Dispose();
        _stageOut.Dispose();
        _argbA.Dispose();
        _argbB.Dispose();
        _work.Dispose();
        _tokens.Dispose();
        _blobs.Dispose();
        _rowCost.Dispose();
    }
}
