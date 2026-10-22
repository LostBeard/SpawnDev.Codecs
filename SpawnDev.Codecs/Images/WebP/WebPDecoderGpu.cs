// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// WebP decoder, 100% accelerator-resident. The host is a pure coordinator: it uploads the file (necessary I/O),
// dispatches the WebPKernels chain, and reads back small metadata blocks (the meta block, the frame table, a few
// hundred ints at most). The compressed bytes, the ARGB working images and the RGBA pixels never touch the host
// unless the caller asks for managed output.
//
// API shape (same as the video codecs and PngDecoderGpu): three overloads per entry point, all funnelling into
// the ArrayView1D one.
//   - ArrayView1D : file already on the accelerator, RGBA pixels written into a caller-owned view.
//   - TypedArray  : browser. JS bytes in via CopyFromJS, JS Uint8Array pixels out; nothing enters the .NET heap.
//   - ReadOnlyMemory<byte> : managed. Desktop and small inputs; pixels come back as byte[].
//
// Supported: the RIFF container (simple VP8L files, VP8X extended files with ICCP / EXIF / XMP, ANIM / ANMF
// animations), the complete VP8L lossless bitstream (all four transforms, colour cache, meta prefix codes,
// LZ77 with the 120 short distance codes), animation compositing (blend + dispose) bit-exact with libwebp's
// WebPAnimDecoder. Lossy VP8 chunks are recognised and rejected with ImageDecodeException ("lossy WebP not yet
// supported"); ReadInfo / ReadFrames still describe such files. The alpha plane of a lossy image (the ALPH chunk:
// raw or VP8L-compressed, with the none / horizontal / vertical / gradient filters) IS decoded, by DecodeAlphaAsync,
// so a VP8 decoder only has to supply the colour half. The structure keeps the image decode behind one switch on the
// payload kind, so a VP8 decoder plugs in next to VP8L.
//
// DecodeAsync of an animated file returns its first frame (the canvas as it looks while frame 0 is shown);
// ReadFramesAsync / DecodeFrameAsync give every frame.

using System.Runtime.Versioning;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.SpawnJS.JSObjects;

namespace SpawnDev.Codecs.Images.WebP;

/// <summary>WebP decoder running on an ILGPU accelerator.</summary>
/// <remarks>
/// Not thread-safe: one in-flight call per instance (the metadata block, frame table and working buffers are
/// shared across calls).
/// </remarks>
public sealed class WebPDecoderGpu : IDisposable
{
    /// <summary>Default initial capacity (ints) of the prefix-code pool; it grows on demand.</summary>
    public const int DefaultInitialPoolInts = 1 << 18;
    private const int MaxFrames = 1 << 16;

    private readonly Accelerator _accelerator;
    private readonly WebPKernels _kernels;
    private readonly int _initialPoolInts;

    private readonly MemoryBuffer1D<int, Stride1D.Dense> _meta;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _tables;

    private readonly StageBuffer<byte> _stageIn;
    private readonly StageBuffer<byte> _stageOut;
    private readonly StageBuffer<int> _frames;
    private readonly StageBuffer<int> _work;
    private readonly StageBuffer<int> _pool;
    private readonly StageBuffer<uint> _pixA;
    private readonly StageBuffer<uint> _pixB;
    private readonly StageBuffer<byte> _frameRgba;

    /// <summary>Compile the kernels and upload the constant tables.</summary>
    /// <param name="accelerator">The accelerator to decode on.</param>
    /// <param name="initialPoolInts">
    /// Initial capacity of the prefix-code pool in ints (default <see cref="DefaultInitialPoolInts"/>). The pool holds the
    /// built prefix codes of the groups a file references; when a file needs more the decode reports the size it needs and the
    /// pool is grown and the decode repeated, so this only trades memory against a rare retry.
    /// </param>
    public WebPDecoderGpu(Accelerator accelerator, int initialPoolInts = DefaultInitialPoolInts)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        ArgumentOutOfRangeException.ThrowIfLessThan(initialPoolInts, 1);
        _initialPoolInts = initialPoolInts;
        _accelerator = accelerator;
        _kernels = new WebPKernels(accelerator);

        _meta = accelerator.Allocate1D<int>(WebPMeta.Size);
        _tables = accelerator.Allocate1D<int>(Vp8lTables.Size);
        _tables.View.CopyFromCPU(Vp8lTables.Build());

        _stageIn = new StageBuffer<byte>(accelerator);
        _stageOut = new StageBuffer<byte>(accelerator);
        _frames = new StageBuffer<int>(accelerator);
        _work = new StageBuffer<int>(accelerator);
        _pool = new StageBuffer<int>(accelerator);
        _pixA = new StageBuffer<uint>(accelerator);
        _pixB = new StageBuffer<uint>(accelerator);
        _frameRgba = new StageBuffer<byte>(accelerator);
    }

    // ------------------------------------------------------------------
    // Container facts
    // ------------------------------------------------------------------

    private sealed class Layout
    {
        public int[] Meta = null!;
        public int[]? Frames;

        public int Width => Meta[WebPMeta.Width];
        public int Height => Meta[WebPMeta.Height];
        public bool Animated => Meta[WebPMeta.IsAnimated] == 1;
        public int FrameCount => Meta[WebPMeta.FrameCount];
        public int StillKind => Meta[WebPMeta.ImageKind];

        public int FrameField(int frame, int field) => Frames![frame * WebPMeta.FrameStride + field];

        public bool IsLossless
        {
            get
            {
                if (!Animated) return StillKind == 1;
                for (int i = 0; i < FrameCount; i++)
                    if (FrameField(i, WebPMeta.FrKind) != 1) return false;
                return true;
            }
        }

        public ImageInfo ToInfo() => new(
            ImageFormat.WebP, Width, Height, Meta[WebPMeta.HasAlpha] == 1, 8,
            Math.Max(1, FrameCount), Meta[WebPMeta.Orientation], IsInterlaced: false);
    }

    private static string DescribeError(WebPError error) => error switch
    {
        WebPError.NotWebP => "not a WebP file (missing RIFF/WEBP signature)",
        WebPError.Truncated => "file is truncated (the RIFF size or a chunk runs past the end of the file)",
        WebPError.BadChunk => "malformed chunk (unexpected first chunk or a chunk of the wrong size)",
        WebPError.LossyUnsupported => "lossy WebP not yet supported",
        WebPError.BadHeader => "invalid VP8L/VP8 header (bad signature or version)",
        WebPError.NoImageData => "no image data (no VP8L/VP8 chunk, or an animation without frames)",
        WebPError.CanvasMismatch => "image or frame size does not match the canvas",
        WebPError.TooManyFrames => "more animation frames than the frame table holds",
        WebPError.TooLarge => "image is too large",
        WebPError.BadTransform => "a VP8L transform is used twice",
        WebPError.BadCacheBits => "invalid colour cache size",
        WebPError.BadPrefixCode => "invalid prefix (Huffman) code",
        WebPError.BadBackReference => "invalid backward reference",
        WebPError.TruncatedBitstream => "VP8L image data is truncated",
        WebPError.BadCacheIndex => "invalid colour cache index",
        WebPError.BadCodeLengths => "invalid prefix code lengths",
        WebPError.NoAlphaChunk => "the image has no ALPH chunk (only a lossy image in a VP8X file carries one)",
        WebPError.BadAlphaHeader => "invalid ALPH chunk (empty, or the compression method, pre-processing or reserved bits are invalid)",
        WebPError.AlphaTruncated => "the uncompressed ALPH chunk holds fewer bytes than the plane needs",
        _ => error.ToString(),
    };

    private static ImageDecodeException Error(WebPError error) => new(ImageFormat.WebP, DescribeError(error));

    private static void RequireRgba8(PixelFormat format)
    {
        if (format != PixelFormat.Rgba8)
            throw new ArgumentException("WebP is an 8-bit format: only PixelFormat.Rgba8 is supported.", nameof(format));
    }

    /// <summary>Container scan + readback of the metadata block (and the frame table of an animation).</summary>
    private async Task<Layout> ScanAsync(ArrayView1D<byte, Stride1D.Dense> file, int fileLength)
    {
        if (fileLength <= 0 || fileLength > file.Length) throw new ArgumentOutOfRangeException(nameof(fileLength));
        int frameCapacity = Math.Min(MaxFrames, fileLength / 32 + 1);
        _frames.Ensure((long)frameCapacity * WebPMeta.FrameStride);
        _kernels.Scan(file, fileLength, _meta.View, _frames.View, frameCapacity);
        await _accelerator.SynchronizeAsync();
        var meta = await _meta.View.CopyToHostAsync();
        var err = (WebPError)meta[WebPMeta.Error];
        if (err != WebPError.None) throw Error(err);
        var layout = new Layout { Meta = meta };
        if (layout.Animated)
            layout.Frames = await _frames.View.SubView(0, (long)layout.FrameCount * WebPMeta.FrameStride).CopyToHostAsync();
        return layout;
    }

    private static ImageFrameInfo[] ToFrameInfos(Layout l)
    {
        if (!l.Animated)
            return new[] { new ImageFrameInfo(0, 0, 0, l.Width, l.Height, 0, FrameDispose.None, FrameBlend.Source) };
        var frames = new ImageFrameInfo[l.FrameCount];
        for (int i = 0; i < frames.Length; i++)
        {
            int flags = l.FrameField(i, WebPMeta.FrFlags);
            frames[i] = new ImageFrameInfo(
                i, l.FrameField(i, WebPMeta.FrX), l.FrameField(i, WebPMeta.FrY),
                l.FrameField(i, WebPMeta.FrW), l.FrameField(i, WebPMeta.FrH), l.FrameField(i, WebPMeta.FrDuration),
                (flags & 1) != 0 ? FrameDispose.Background : FrameDispose.None,
                (flags & 2) != 0 ? FrameBlend.Source : FrameBlend.Over);
        }
        return frames;
    }

    /// <summary>
    /// Read the facts about a WebP that is on the accelerator (a RIFF walk + a metadata-sized readback; no pixels are
    /// decoded). Works for lossy files too. Use the result to size the output buffer for
    /// <see cref="DecodeAsync(ArrayView1D{byte, Stride1D.Dense}, int, ArrayView1D{byte, Stride1D.Dense}, PixelFormat)"/>.
    /// </summary>
    public async Task<ImageInfo> ReadInfoAsync(ArrayView1D<byte, Stride1D.Dense> file, int fileLength)
    {
        var layout = await ScanAsync(file, fileLength);
        return layout.ToInfo();
    }

    /// <summary>Everything the container says about the file: <see cref="ImageInfo"/> plus losslessness and which metadata chunks exist.</summary>
    public async Task<WebPFileInfo> ReadFileInfoAsync(ArrayView1D<byte, Stride1D.Dense> file, int fileLength)
    {
        var l = await ScanAsync(file, fileLength);
        var m = l.Meta;
        return new WebPFileInfo(
            l.ToInfo(), l.IsLossless, m[WebPMeta.HasVp8x] == 1, m[WebPMeta.IccOffset] >= 0, m[WebPMeta.ExifOffset] >= 0,
            m[WebPMeta.XmpOffset] >= 0, m[WebPMeta.LoopCount], (uint)m[WebPMeta.BackgroundColor]);
    }

    /// <summary>The ICCP / EXIF / XMP payloads (small blobs read back from the accelerator).</summary>
    public async Task<WebPMetadata> ReadMetadataAsync(ArrayView1D<byte, Stride1D.Dense> file, int fileLength)
    {
        var l = await ScanAsync(file, fileLength);
        async Task<byte[]?> Blob(int offsetSlot, int lengthSlot)
        {
            int off = l.Meta[offsetSlot];
            if (off < 0) return null;
            int len = l.Meta[lengthSlot];
            return len == 0 ? System.Array.Empty<byte>() : await file.SubView(off, len).CopyToHostAsync();
        }
        return new WebPMetadata(
            await Blob(WebPMeta.IccOffset, WebPMeta.IccLength),
            await Blob(WebPMeta.ExifOffset, WebPMeta.ExifLength),
            await Blob(WebPMeta.XmpOffset, WebPMeta.XmpLength));
    }

    /// <summary>
    /// The frames of the file: rectangle, duration, disposal and blend of each (a still image has one frame covering
    /// the image). Only metadata crosses to the host.
    /// </summary>
    public async Task<ImageFrameInfo[]> ReadFramesAsync(ArrayView1D<byte, Stride1D.Dense> file, int fileLength)
    {
        var layout = await ScanAsync(file, fileLength);
        return ToFrameInfos(layout);
    }

    // ------------------------------------------------------------------
    // Decode
    // ------------------------------------------------------------------

    /// <summary>
    /// Decode a WebP that is already on the accelerator into a caller-owned pixel view (the first frame of an
    /// animation, as a full canvas). Only metadata scalars cross to the host.
    /// </summary>
    /// <param name="file">The WebP file bytes (may be longer than <paramref name="fileLength"/>).</param>
    /// <param name="fileLength">Number of valid bytes in <paramref name="file"/>.</param>
    /// <param name="output">Receives the tightly packed RGBA8 pixels; at least <see cref="ImageInfo.GetPixelBufferSize"/> bytes.</param>
    /// <param name="format">Pixel layout; only <see cref="PixelFormat.Rgba8"/>.</param>
    /// <returns>The facts about the image.</returns>
    /// <exception cref="ImageDecodeException">The file is corrupt, truncated, lossy or uses an unsupported feature.</exception>
    public async Task<ImageInfo> DecodeAsync(
        ArrayView1D<byte, Stride1D.Dense> file, int fileLength,
        ArrayView1D<byte, Stride1D.Dense> output, PixelFormat format = PixelFormat.Rgba8)
    {
        RequireRgba8(format);
        var layout = await ScanAsync(file, fileLength);
        return await DecodeFrameCoreAsync(file, layout, 0, output);
    }

    /// <summary>
    /// Decode frame <paramref name="frameIndex"/> of a (possibly animated) WebP into a caller-owned view: the FULL CANVAS
    /// as it looks while that frame is displayed. Frames 0..<paramref name="frameIndex"/> are decoded and composited on the
    /// accelerator honouring each frame's blend method and the previous frames' disposal; the canvas never visits the host.
    /// A still image has one frame.
    /// </summary>
    public async Task<ImageInfo> DecodeFrameAsync(
        ArrayView1D<byte, Stride1D.Dense> file, int fileLength, int frameIndex,
        ArrayView1D<byte, Stride1D.Dense> output, PixelFormat format = PixelFormat.Rgba8)
    {
        RequireRgba8(format);
        var layout = await ScanAsync(file, fileLength);
        if (frameIndex < 0 || frameIndex >= Math.Max(1, layout.FrameCount))
            throw new ArgumentOutOfRangeException(nameof(frameIndex), $"The file has {Math.Max(1, layout.FrameCount)} frame(s).");
        return await DecodeFrameCoreAsync(file, layout, frameIndex, output);
    }

    private async Task<ImageInfo> DecodeFrameCoreAsync(
        ArrayView1D<byte, Stride1D.Dense> file, Layout layout, int frameIndex, ArrayView1D<byte, Stride1D.Dense> output)
    {
        var info = layout.ToInfo();
        long outBytes = info.GetPixelBufferSize(PixelFormat.Rgba8);
        if (output.Length < outBytes)
            throw new ArgumentException($"output must hold {outBytes} bytes for {info.Width}x{info.Height} Rgba8.", nameof(output));

        if (!layout.Animated)
        {
            if (layout.StillKind != 1) throw Error(WebPError.LossyUnsupported);
            EnsureDecodeBuffers(layout.Width, layout.Height, layout.Width, layout.Height);
            await DecodeVp8lAsync(file, layout.Meta[WebPMeta.ImageOffset], layout.Meta[WebPMeta.ImageLength],
                layout.Width, layout.Height, output);
        }
        else
        {
            // Check every frame we are about to composite before dispatching anything.
            for (int k = 0; k <= frameIndex; k++)
                if (layout.FrameField(k, WebPMeta.FrKind) != 1) throw Error(WebPError.LossyUnsupported);
            EnsureAnimationBuffers(layout, frameIndex);
            await CompositeFramesAsync(file, layout, frameIndex, output);
        }
        await _accelerator.SynchronizeAsync();
        return info;
    }

    /// <summary>Grow the working buffers for images up to the given sizes (called at the START of a public call).</summary>
    private void EnsureDecodeBuffers(int maxPixelsW, int maxPixelsH, int maxWorkW, int maxWorkH)
    {
        long pixels = (long)maxPixelsW * maxPixelsH;
        if (pixels > 0x1FFFFFFF) throw Error(WebPError.TooLarge);
        _pixA.Ensure(pixels);
        _pixB.Ensure(pixels);
        _work.Ensure(Vp8lLayout.WorkSize(maxWorkW, maxWorkH));
        _pool.Ensure(_initialPoolInts);
    }

    private void EnsureAnimationBuffers(Layout layout, int lastFrame)
    {
        long maxPixels = 1;
        long maxWork = 1;
        for (int k = 0; k <= lastFrame; k++)
        {
            int w = layout.FrameField(k, WebPMeta.FrW);
            int h = layout.FrameField(k, WebPMeta.FrH);
            maxPixels = Math.Max(maxPixels, (long)w * h);
            maxWork = Math.Max(maxWork, Vp8lLayout.WorkSize(w, h));
        }
        if (maxPixels > 0x1FFFFFFF) throw Error(WebPError.TooLarge);
        _pixA.Ensure(maxPixels);
        _pixB.Ensure(maxPixels);
        _work.Ensure(maxWork);
        _pool.Ensure(_initialPoolInts);
        _frameRgba.Ensure(maxPixels * 4);
    }

    /// <summary>libwebp anim_decode.c IsKeyFrame.</summary>
    private static bool IsKeyFrame(Layout l, int k, bool prevWasKey)
    {
        if (k == 0) return true;
        int cw = l.Width, ch = l.Height;
        bool full = l.FrameField(k, WebPMeta.FrW) == cw && l.FrameField(k, WebPMeta.FrH) == ch;
        bool hasAlpha = l.FrameField(k, WebPMeta.FrHasAlpha) != 0;
        bool noBlend = (l.FrameField(k, WebPMeta.FrFlags) & 2) != 0;
        if ((!hasAlpha || noBlend) && full) return true;
        bool prevBg = (l.FrameField(k - 1, WebPMeta.FrFlags) & 1) != 0;
        bool prevFull = l.FrameField(k - 1, WebPMeta.FrW) == cw && l.FrameField(k - 1, WebPMeta.FrH) == ch;
        return prevBg && (prevFull || prevWasKey);
    }

    private async Task CompositeFramesAsync(
        ArrayView1D<byte, Stride1D.Dense> file, Layout l, int lastFrame, ArrayView1D<byte, Stride1D.Dense> canvas)
    {
        int cw = l.Width, ch = l.Height;
        bool prevWasKey = false;
        for (int k = 0; k <= lastFrame; k++)
        {
            int fx = l.FrameField(k, WebPMeta.FrX), fy = l.FrameField(k, WebPMeta.FrY);
            int fw = l.FrameField(k, WebPMeta.FrW), fh = l.FrameField(k, WebPMeta.FrH);
            int flags = l.FrameField(k, WebPMeta.FrFlags);
            bool isKey = IsKeyFrame(l, k, prevWasKey);

            int px = 0, py = 0, pw = 0, ph = 0;
            bool prevBg = false;
            if (k > 0)
            {
                px = l.FrameField(k - 1, WebPMeta.FrX); py = l.FrameField(k - 1, WebPMeta.FrY);
                pw = l.FrameField(k - 1, WebPMeta.FrW); ph = l.FrameField(k - 1, WebPMeta.FrH);
                prevBg = (l.FrameField(k - 1, WebPMeta.FrFlags) & 1) != 0;
            }

            // Canvas state before this frame: the previous canvas with the previous frame disposed. A key frame starts from nothing.
            if (isKey) _kernels.ClearAll(canvas, cw * ch);
            else if (prevBg) _kernels.ClearRect(canvas, cw, px, py, pw, ph);

            await DecodeVp8lAsync(file, l.FrameField(k, WebPMeta.FrOffset), l.FrameField(k, WebPMeta.FrLength), fw, fh, _frameRgba.View);

            bool blend = !isKey && (flags & 2) == 0;
            _kernels.Composite(canvas, _frameRgba.View, cw, fx, fy, fw, fh, blend ? 1 : 0, prevBg ? 1 : 0, px, py, pw, ph);
            prevWasKey = isKey;
        }
    }

    /// <summary>
    /// Decode one VP8L stream of <paramref name="w"/> x <paramref name="h"/> into <paramref name="rgba"/>. Buffers are
    /// already sized by the caller. Syncs once (to read the decode result); the inverse transforms and the RGBA
    /// conversion are left dispatched.
    /// </summary>
    private async Task DecodeVp8lAsync(
        ArrayView1D<byte, Stride1D.Dense> file, int payloadOffset, int payloadLength, int w, int h,
        ArrayView1D<byte, Stride1D.Dense> rgba)
    {
        var cur = await DecodeVp8lPixelsAsync(file, payloadOffset, payloadLength, w, h);
        _kernels.ToRgba(cur, rgba, w * h);
    }

    /// <summary>
    /// Decode one VP8L stream and undo its transforms; returns the ARGB pixels (a view of one of the two pixel buffers,
    /// valid until the next decode). The stream's 5-byte header (signature, size, alpha bit, version) is skipped, not read:
    /// the container scan validated it, and an ALPH stream (which has no header) is passed 4 bytes early so the same 5-byte
    /// skip lands on its first bit.
    /// </summary>
    private async Task<ArrayView<uint>> DecodeVp8lPixelsAsync(
        ArrayView1D<byte, Stride1D.Dense> file, int payloadOffset, int payloadLength, int w, int h)
    {
        int[] m;
        int poolCap = (int)Math.Min(_pool.View.Length, int.MaxValue);
        while (true)
        {
            _kernels.Decode(file, payloadOffset, payloadLength, w, h, poolCap, _meta.View, _work.View, _pool.View, _pixA.View, _tables.View);
            await _accelerator.SynchronizeAsync();
            m = await _meta.View.CopyToHostAsync();
            var err = (WebPError)m[WebPMeta.DecodeError];
            if (err == WebPError.PoolTooSmall)
            {
                // The pool holds the prefix codes of the referenced groups; the kernel reported how much it needs.
                long grown = Math.Max((long)m[WebPMeta.PoolNeeded], 2L * poolCap);
                if (grown > int.MaxValue) throw Error(WebPError.TooLarge);
                _pool.Ensure(grown);
                poolCap = (int)_pool.View.Length;
                continue;
            }
            if (err != WebPError.None) throw Error(err);
            break;
        }

        var cur = _pixA.View;
        var other = _pixB.View;
        int curW = m[WebPMeta.PackedWidth];
        for (int t = m[WebPMeta.TransformCount] - 1; t >= 0; t--)
        {
            int rec = WebPMeta.Transforms + t * WebPMeta.TransformStride;
            int kind = m[rec + WebPMeta.TrKind];
            int bits = m[rec + WebPMeta.TrBits];
            int data = m[rec + WebPMeta.TrData];
            int tw = m[rec + WebPMeta.TrWidth];
            switch (kind)
            {
                case WebPMeta.KindSubtractGreen:
                    _kernels.SubtractGreen(cur, curW * h);
                    break;
                case WebPMeta.KindCrossColor:
                    _kernels.CrossColor(cur, _work.View, data, bits, curW, curW * h);
                    break;
                case WebPMeta.KindPredictor:
                    _kernels.Predictor(cur, _work.View, data, bits, curW, h);
                    break;
                default:
                    _kernels.ColorIndex(cur, other, _work.View, data, m[rec + WebPMeta.TrCount], bits, curW, tw, h);
                    (cur, other) = (other, cur);
                    curW = tw;
                    break;
            }
        }
        return cur;
    }

    // ------------------------------------------------------------------
    // ALPH: the alpha plane of a lossy image
    // ------------------------------------------------------------------

    /// <summary>
    /// Decode the alpha plane (the ALPH chunk) of a lossy WebP that is on the accelerator into a caller-owned view of
    /// width x height bytes (one byte per pixel, rows top to bottom). Handles both ALPH compression methods (raw, and the
    /// VP8L-compressed one with every transform) and the none / horizontal / vertical / gradient prediction filters; the
    /// plane is produced entirely on the accelerator. The VP8 (colour) part of the file is not touched, so this works
    /// before a lossy decoder exists and is the alpha half of one when it does.
    /// </summary>
    /// <param name="file">The WebP file bytes (may be longer than <paramref name="fileLength"/>).</param>
    /// <param name="fileLength">Number of valid bytes in <paramref name="file"/>.</param>
    /// <param name="alpha">Receives the plane; at least width x height bytes.</param>
    /// <param name="frameIndex">The animation frame whose ALPH chunk is decoded (0 for a still image).</param>
    /// <returns>The frame: its size is the size of the plane.</returns>
    /// <exception cref="ImageDecodeException">There is no ALPH chunk for the image, or it is malformed.</exception>
    public async Task<ImageFrameInfo> DecodeAlphaAsync(
        ArrayView1D<byte, Stride1D.Dense> file, int fileLength, ArrayView1D<byte, Stride1D.Dense> alpha, int frameIndex = 0)
    {
        var layout = await ScanAsync(file, fileLength);
        return await DecodeAlphaCoreAsync(file, layout, frameIndex, alpha);
    }

    private async Task<ImageFrameInfo> DecodeAlphaCoreAsync(
        ArrayView1D<byte, Stride1D.Dense> file, Layout l, int frameIndex, ArrayView1D<byte, Stride1D.Dense> alpha)
    {
        if (frameIndex < 0 || frameIndex >= Math.Max(1, l.FrameCount))
            throw new ArgumentOutOfRangeException(nameof(frameIndex), $"The file has {Math.Max(1, l.FrameCount)} frame(s).");
        int w, h, kind, off, len;
        if (l.Animated)
        {
            w = l.FrameField(frameIndex, WebPMeta.FrW);
            h = l.FrameField(frameIndex, WebPMeta.FrH);
            kind = l.FrameField(frameIndex, WebPMeta.FrKind);
            off = l.FrameField(frameIndex, WebPMeta.FrAlphOffset);
            len = l.FrameField(frameIndex, WebPMeta.FrAlphLength);
        }
        else
        {
            w = l.Width;
            h = l.Height;
            kind = l.StillKind;
            off = l.Meta[WebPMeta.AlphOffset];
            len = l.Meta[WebPMeta.AlphLength];
        }
        var frame = ToFrameInfos(l)[frameIndex];
        if (kind != 3) throw Error(WebPError.NoAlphaChunk);
        long n = (long)w * h;
        if (alpha.Length < n)
            throw new ArgumentException($"alpha must hold {n} bytes for {w}x{h}.", nameof(alpha));
        if (len < 2) throw Error(WebPError.BadAlphaHeader);

        int header = (await file.SubView(off, 1).CopyToHostAsync())[0];
        int method = header & WebPMeta.AlphMethodMask;
        int filter = (header >> WebPMeta.AlphFilterShift) & 3;
        int preprocessing = (header >> WebPMeta.AlphPreprocessShift) & 3;
        int reserved = header >> WebPMeta.AlphReservedShift;
        // libwebp (alpha_dec.c ALPHInit) rejects exactly these; pre-processing 1 (level reduction) needs nothing from the decoder
        if (method > 1 || preprocessing > 1 || reserved != 0) throw Error(WebPError.BadAlphaHeader);
        if (method == 0 && len - 1 < n) throw Error(WebPError.AlphaTruncated);

        EnsureDecodeBuffers(w, h, w, h);
        ArrayView<uint> plane;
        if (method == 0)
        {
            plane = _pixA.View;
            _kernels.AlphaCopy(file, off + 1, plane, (int)n);
        }
        else
        {
            // The stream has no VP8L header: it starts right after the ALPH header byte (see DecodeVp8lPixelsAsync).
            plane = await DecodeVp8lPixelsAsync(file, off - 4, len + 4, w, h);
            _kernels.AlphaGreen(plane, (int)n);
        }
        _kernels.AlphaUnfilter(plane, w, h, filter);
        _kernels.AlphaPack(plane, alpha, (int)n);
        await _accelerator.SynchronizeAsync();
        return frame;
    }

    // ------------------------------------------------------------------
    // Managed overloads
    // ------------------------------------------------------------------

    /// <summary>Not async: it holds a span.</summary>
    private void StageManaged(ReadOnlyMemory<byte> file)
    {
        if (file.IsEmpty) throw new ArgumentException("File is empty.", nameof(file));
        _stageIn.Ensure(file.Length);
        DeviceTransfer.Upload(_stageIn.View, file.Span);
    }

    /// <summary>Read the facts about a managed WebP file (no pixels are decoded).</summary>
    public async Task<ImageInfo> ReadInfoAsync(ReadOnlyMemory<byte> file)
    {
        StageManaged(file);
        return await ReadInfoAsync(_stageIn.View, file.Length);
    }

    /// <summary>Everything the container says about a managed WebP file.</summary>
    public async Task<WebPFileInfo> ReadFileInfoAsync(ReadOnlyMemory<byte> file)
    {
        StageManaged(file);
        return await ReadFileInfoAsync(_stageIn.View, file.Length);
    }

    /// <summary>The ICCP / EXIF / XMP payloads of a managed WebP file.</summary>
    public async Task<WebPMetadata> ReadMetadataAsync(ReadOnlyMemory<byte> file)
    {
        StageManaged(file);
        return await ReadMetadataAsync(_stageIn.View, file.Length);
    }

    /// <summary>The frames of a managed WebP file.</summary>
    public async Task<ImageFrameInfo[]> ReadFramesAsync(ReadOnlyMemory<byte> file)
    {
        StageManaged(file);
        return await ReadFramesAsync(_stageIn.View, file.Length);
    }

    private async Task<ImageInfo> DecodeToStageAsync(int fileLength, int frameIndex, PixelFormat format)
    {
        RequireRgba8(format);
        var layout = await ScanAsync(_stageIn.View, fileLength);
        if (frameIndex < 0 || frameIndex >= Math.Max(1, layout.FrameCount))
            throw new ArgumentOutOfRangeException(nameof(frameIndex), $"The file has {Math.Max(1, layout.FrameCount)} frame(s).");
        _stageOut.Ensure(layout.ToInfo().GetPixelBufferSize(PixelFormat.Rgba8));
        return await DecodeFrameCoreAsync(_stageIn.View, layout, frameIndex, _stageOut.View);
    }

    /// <summary>
    /// Decode a managed WebP file (the first frame of an animation). Returns the pixels as a managed array (a terminal
    /// CPU sink; in a browser prefer the <see cref="TypedArray"/> overload).
    /// </summary>
    public async Task<DecodedImage> DecodeAsync(ReadOnlyMemory<byte> file, PixelFormat format = PixelFormat.Rgba8)
        => await DecodeFrameAsync(file, 0, format);

    /// <summary>Decode frame <paramref name="frameIndex"/> (full canvas) of a managed WebP file.</summary>
    public async Task<DecodedImage> DecodeFrameAsync(ReadOnlyMemory<byte> file, int frameIndex, PixelFormat format = PixelFormat.Rgba8)
    {
        StageManaged(file);
        var info = await DecodeToStageAsync(file.Length, frameIndex, format);
        var pixels = await DeviceTransfer.DownloadAsync(_stageOut.View, info.GetPixelBufferSize(PixelFormat.Rgba8));
        return new DecodedImage(info, pixels, PixelFormat.Rgba8);
    }

    /// <summary>
    /// Decode the alpha plane (ALPH chunk) of a managed lossy WebP file; see
    /// <see cref="DecodeAlphaAsync(ArrayView1D{byte, Stride1D.Dense}, int, ArrayView1D{byte, Stride1D.Dense}, int)"/>.
    /// Returns the plane as a managed array (a terminal CPU sink; in a browser prefer the <see cref="TypedArray"/> overload).
    /// </summary>
    public async Task<WebPAlphaPlane> DecodeAlphaAsync(ReadOnlyMemory<byte> file, int frameIndex = 0)
    {
        StageManaged(file);
        var (frame, count) = await DecodeAlphaToStageAsync(file.Length, frameIndex);
        var plane = await DeviceTransfer.DownloadAsync(_stageOut.View, count);
        return new WebPAlphaPlane(frame.Width, frame.Height, plane);
    }

    private async Task<(ImageFrameInfo Frame, long Count)> DecodeAlphaToStageAsync(int fileLength, int frameIndex)
    {
        var layout = await ScanAsync(_stageIn.View, fileLength);
        if (frameIndex < 0 || frameIndex >= Math.Max(1, layout.FrameCount))
            throw new ArgumentOutOfRangeException(nameof(frameIndex), $"The file has {Math.Max(1, layout.FrameCount)} frame(s).");
        long count = layout.Animated
            ? (long)layout.FrameField(frameIndex, WebPMeta.FrW) * layout.FrameField(frameIndex, WebPMeta.FrH)
            : (long)layout.Width * layout.Height;
        _stageOut.Ensure(count);
        var frame = await DecodeAlphaCoreAsync(_stageIn.View, layout, frameIndex, _stageOut.View);
        return (frame, count);
    }

    // ------------------------------------------------------------------
    // TypedArray overloads (browser: bulk bytes stay in JS)
    // ------------------------------------------------------------------

    private int StageTyped(TypedArray file)
    {
        ArgumentNullException.ThrowIfNull(file);
        long length = file.ByteLength;
        _stageIn.Ensure(length);
        DeviceTransfer.Upload(_stageIn.View, file);
        return checked((int)length);
    }

    /// <summary>Read the facts about a WebP held in a JS typed array (browser accelerators only).</summary>
    [SupportedOSPlatform("browser")]
    public async Task<ImageInfo> ReadInfoAsync(TypedArray file)
    {
        int length = StageTyped(file);
        return await ReadInfoAsync(_stageIn.View, length);
    }

    /// <summary>Everything the container says about a WebP held in a JS typed array (browser accelerators only).</summary>
    [SupportedOSPlatform("browser")]
    public async Task<WebPFileInfo> ReadFileInfoAsync(TypedArray file)
    {
        int length = StageTyped(file);
        return await ReadFileInfoAsync(_stageIn.View, length);
    }

    /// <summary>The ICCP / EXIF / XMP payloads of a WebP held in a JS typed array (browser accelerators only; the blobs are small host arrays).</summary>
    [SupportedOSPlatform("browser")]
    public async Task<WebPMetadata> ReadMetadataAsync(TypedArray file)
    {
        int length = StageTyped(file);
        return await ReadMetadataAsync(_stageIn.View, length);
    }

    /// <summary>The frames of a WebP held in a JS typed array (browser accelerators only).</summary>
    [SupportedOSPlatform("browser")]
    public async Task<ImageFrameInfo[]> ReadFramesAsync(TypedArray file)
    {
        int length = StageTyped(file);
        return await ReadFramesAsync(_stageIn.View, length);
    }

    /// <summary>
    /// Decode the alpha plane (ALPH chunk) of a lossy WebP held in a JS typed array (browser accelerators only). The bytes go
    /// to the GPU with <c>CopyFromJS</c> and the plane comes back as a JS <see cref="Uint8Array"/>; none of it enters the .NET
    /// heap. The caller disposes the result.
    /// </summary>
    [SupportedOSPlatform("browser")]
    public async Task<WebPAlphaPlaneJS> DecodeAlphaAsync(TypedArray file, int frameIndex = 0)
    {
        int length = StageTyped(file);
        var (frame, count) = await DecodeAlphaToStageAsync(length, frameIndex);
        var plane = await DeviceTransfer.DownloadUint8ArrayAsync(_stageOut.View, count);
        return new WebPAlphaPlaneJS(frame.Width, frame.Height, plane);
    }

    /// <summary>
    /// Decode a WebP held in a JS typed array (browser accelerators only; the first frame of an animation). The bytes go to
    /// the GPU with <c>CopyFromJS</c> and the pixels come back as a JS <see cref="Uint8Array"/>; none of it enters the
    /// .NET heap. The caller disposes the result.
    /// </summary>
    /// <exception cref="NotSupportedException">The accelerator is not a browser accelerator.</exception>
    [SupportedOSPlatform("browser")]
    public async Task<DecodedImageJS> DecodeAsync(TypedArray file, PixelFormat format = PixelFormat.Rgba8)
        => await DecodeFrameAsync(file, 0, format);

    /// <summary>Decode frame <paramref name="frameIndex"/> (full canvas) of a WebP held in a JS typed array (browser accelerators only).</summary>
    [SupportedOSPlatform("browser")]
    public async Task<DecodedImageJS> DecodeFrameAsync(TypedArray file, int frameIndex, PixelFormat format = PixelFormat.Rgba8)
    {
        int length = StageTyped(file);
        var info = await DecodeToStageAsync(length, frameIndex, format);
        var pixels = await DeviceTransfer.DownloadUint8ArrayAsync(_stageOut.View, info.GetPixelBufferSize(PixelFormat.Rgba8));
        return new DecodedImageJS(info, pixels, PixelFormat.Rgba8);
    }

    /// <summary>Release kernel resources and buffers.</summary>
    public void Dispose()
    {
        _kernels.Dispose();
        _meta.Dispose();
        _tables.Dispose();
        _stageIn.Dispose();
        _stageOut.Dispose();
        _frames.Dispose();
        _work.Dispose();
        _pool.Dispose();
        _pixA.Dispose();
        _pixB.Dispose();
        _frameRgba.Dispose();
    }
}
