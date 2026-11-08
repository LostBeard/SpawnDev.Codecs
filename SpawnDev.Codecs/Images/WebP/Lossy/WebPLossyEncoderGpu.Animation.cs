// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Animated lossy WebP: a port of libwebp 1.6.0's WebPAnimEncoder (src/mux/anim_encode.c) as Pillow drives it for
// `save(save_all=True, lossless=False, ...)` (ComfyUI's Save Animated WEBP): allow_mixed off, minimize_size off,
// kmin / kmax from the options (Pillow's lossy defaults 3 / 5), one WebPConfig for every frame.
//
// Per input frame, exactly as libwebp:
//   * the change rectangle against the previous canvas (MinimizeChangeRectangle, lossless then lossy tolerance
//     QualityToMaxDiff(quality), snapped to even offsets), and against the previous canvas disposed to background when
//     that is allowed; the smaller lossy rectangle picks the previous frame's disposal;
//   * a frame whose lossy rectangle is empty is dropped and its duration goes to the previous frame;
//   * blending when every non-opaque pixel of the rectangle is similar to the previous canvas (IsLossyBlendingPossible);
//     a blended frame has its fully-similar 8x8 blocks made transparent (FlattenSimilarBlocks) and is encoded with the
//     loop filter off (filter_strength 0);
//   * key frames: after kmin sub-frames each frame is encoded twice (sub-frame and full-canvas key frame) and the one with
//     the smallest size penalty becomes the key frame, decided by kmax;
//   * frames reach the file when no later frame can change them (FlushFrames); a lone frame is turned into a still image
//     (MuxCleanup / OptimizeSingleFrame).
// Every frame is a still lossy picture of this encoder (VP8 chunk byte-identical with libwebp's WebPEncode).
//
// The host is a coordinator: the pixel tests, rectangles, block flattening, canvases and every byte of the file are
// kernels (WebPAnimKernels); the host reads back the rectangles, two flags and the encoded sizes per frame and makes
// libwebp's decisions from those scalars.
//
// Where the output can differ from libwebp's: the ALPH chunk of a frame with transparent pixels (flattened blocks or a
// transparent source) is coded by our VP8L coder, not libwebp's; it decodes to the same alpha plane but has another size,
// and libwebp's key-frame choice compares sizes. Without transparency (or with key frames off) the decisions are the
// same, frame for frame.

using System.Runtime.Versioning;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.SpawnJS.JSObjects;
using K = SpawnDev.Codecs.Images.WebP.Lossy.WebPAnimKernels;

namespace SpawnDev.Codecs.Images.WebP.Lossy;

/// <summary>Animation-wide settings of <see cref="WebPLossyEncoderGpu"/>'s animated encode (libwebp WebPAnimEncoderOptions + chunks).</summary>
public sealed record WebPLossyAnimationOptions
{
    /// <summary>How many times the animation plays (0 = forever, the default).</summary>
    public int LoopCount { get; init; }

    /// <summary>ANIM background colour, 0xAARRGGBB (written little-endian: B, G, R, A). Pillow's default is 0.</summary>
    public uint BackgroundColor { get; init; }

    /// <summary>
    /// libwebp kmin: the minimum distance between key frames (Pillow's lossy default 3). Sanitized as libwebp does
    /// (see <see cref="KeyframeMax"/>).
    /// </summary>
    public int KeyframeMin { get; init; } = 3;

    /// <summary>
    /// libwebp kmax: the maximum distance between key frames (Pillow's lossy default 5). 0 or less turns key frames off
    /// (every frame after the first is a sub-frame); 1 makes every frame a key frame.
    /// </summary>
    public int KeyframeMax { get; init; } = 5;

    /// <summary>ICC profile written as an ICCP chunk (null = none).</summary>
    public byte[]? IccProfile { get; init; }

    /// <summary>
    /// EXIF chunk payload, written EXACTLY as given (null = none). Pillow's animated save writes <c>Image.Exif.tobytes()</c>
    /// as is, "Exif\0\0" prefix included (its still save strips the prefix); pass those bytes for the same file.
    /// </summary>
    public byte[]? Exif { get; init; }

    /// <summary>XMP packet written as an XMP chunk (null = none).</summary>
    public byte[]? Xmp { get; init; }
}

public sealed partial class WebPLossyEncoderGpu
{
    /// <summary>libwebp MAX_DURATION: frame durations are 24-bit.</summary>
    private const int MaxDuration = 1 << 24;
    private const long DeltaInfinity = 1L << 32;

    /// <summary>libwebp's lossy 1x1 transparent frame (IncreasePreviousDuration's split of an over-long duration).</summary>
    private static readonly byte[] Lossy1x1 =
    {
        0x52, 0x49, 0x46, 0x46, 0x40, 0x00, 0x00, 0x00, 0x57, 0x45, 0x42, 0x50,
        0x56, 0x50, 0x38, 0x58, 0x0a, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x41, 0x4c, 0x50, 0x48, 0x02, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x56, 0x50, 0x38, 0x20, 0x18, 0x00, 0x00, 0x00,
        0x30, 0x01, 0x00, 0x9d, 0x01, 0x2a, 0x01, 0x00, 0x01, 0x00, 0x02, 0x00,
        0x34, 0x25, 0xa4, 0x00, 0x03, 0x70, 0x00, 0xfe, 0xfb, 0xfd, 0x50, 0x00,
    };

    private AnimBuffers? _anim;

    /// <summary>Kernels and buffers of the animated encode, created on its first use.</summary>
    private sealed class AnimBuffers : IDisposable
    {
        public readonly Action<Index1D, ArrayView<int>, int, int, int, int, int> SetRect;
        public readonly Action<Index1D, ArrayView<int>, int, int> CopyRect;
        public readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, int, int, int> Columns, Rows, BlendRows;
        public readonly Action<Index1D, ArrayView<int>, ArrayView<int>, int> ColumnsReduce;
        public readonly Action<Index1D, ArrayView<int>, ArrayView<int>, int, int> RowsReduce;
        public readonly Action<Index1D, ArrayView<int>, ArrayView<int>, int, int, int> Any;
        public readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, int, int, int, int> Extract, Place;
        public readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, int, int, int, int, int, int, int, int> Flatten;
        public readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>> CopyCanvas;
        public readonly Action<Index1D, ArrayView<byte>> ClearCanvas;
        public readonly Action<Index1D, ArrayView<byte>, int, int, int, int> ClearRect;
        public readonly Action<Index1D, ArrayView<byte>, int, int, int, int> FileHeader;
        public readonly Action<Index1D, ArrayView<byte>, int> RiffHeader;
        public readonly Action<Index1D, ArrayView<byte>, int, int, int> AnimChunk;
        public readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, int, int, int, int> BlobChunk;
        public readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, int, int, int, int, int, int, int, int, int> Frame;
        public readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, int, int> Copy;

        public readonly StageBuffer<byte> Prev, PrevDisposed, Canvas, Picture, FrameIn, Blobs;
        public readonly StageBuffer<int> ColFlags, RowFlags;
        public readonly MemoryBuffer1D<int, Stride1D.Dense> Meta;
        public readonly MemoryBuffer1D<byte, Stride1D.Dense> OneByOne;
        public readonly Stack<StageBuffer<byte>> FreeFiles = new();
        public readonly List<StageBuffer<byte>> AllFiles = new();
        public MemoryBuffer1D<byte, Stride1D.Dense>? Out;
        public WebPDecoderGpu? Decoder;
        private readonly Accelerator _acc;

        public AnimBuffers(Accelerator a)
        {
            _acc = a;
            SetRect = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, int, int, int, int, int>(K.SetRectKernel);
            CopyRect = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, int, int>(K.CopyRectKernel);
            Columns = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, int, int, int>(K.ColumnsKernel);
            Rows = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, int, int, int>(K.RowsKernel);
            BlendRows = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, int, int, int>(K.BlendRowsKernel);
            ColumnsReduce = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, int>(K.ColumnsReduceKernel);
            RowsReduce = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, int, int>(K.RowsReduceKernel);
            Any = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, int, int, int>(K.AnyKernel);
            Extract = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<byte>, int, int, int, int>(K.ExtractKernel);
            Place = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<byte>, int, int, int, int>(K.PlaceKernel);
            Flatten = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<byte>, int, int, int, int, int, int, int, int>(K.FlattenKernel);
            CopyCanvas = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<byte>>(K.CopyCanvasKernel);
            ClearCanvas = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>>(K.ClearCanvasKernel);
            ClearRect = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, int, int, int, int>(K.ClearRectKernel);
            FileHeader = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, int, int, int, int>(K.FileHeaderKernel);
            RiffHeader = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, int>(K.RiffHeaderKernel);
            AnimChunk = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, int, int, int>(K.AnimChunkKernel);
            BlobChunk = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<byte>, int, int, int, int>(K.BlobChunkKernel);
            Frame = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<byte>, int, int, int, int, int, int, int, int, int>(K.FrameKernel);
            Copy = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<byte>, int, int>(K.CopyKernel);
            Prev = new StageBuffer<byte>(a); PrevDisposed = new StageBuffer<byte>(a); Canvas = new StageBuffer<byte>(a);
            Picture = new StageBuffer<byte>(a); FrameIn = new StageBuffer<byte>(a); Blobs = new StageBuffer<byte>(a);
            ColFlags = new StageBuffer<int>(a); RowFlags = new StageBuffer<int>(a);
            Meta = a.Allocate1D<int>(K.MetaSize);
            OneByOne = a.Allocate1D<byte>(Lossy1x1.Length);
            OneByOne.View.CopyFromCPU(Lossy1x1);
        }

        public void ResetPool()
        {
            FreeFiles.Clear();
            foreach (var b in AllFiles) FreeFiles.Push(b);
        }

        public StageBuffer<byte> RentFile()
        {
            if (FreeFiles.Count > 0) return FreeFiles.Pop();
            var b = new StageBuffer<byte>(_acc);
            AllFiles.Add(b);
            return b;
        }

        public void Dispose()
        {
            Prev.Dispose(); PrevDisposed.Dispose(); Canvas.Dispose(); Picture.Dispose(); FrameIn.Dispose(); Blobs.Dispose();
            ColFlags.Dispose(); RowFlags.Dispose(); Meta.Dispose(); OneByOne.Dispose(); Out?.Dispose();
            foreach (var b in AllFiles) b.Dispose();
            Decoder?.Dispose();
        }
    }

    private record struct AnimRect(int X, int Y, int W, int H)
    {
        public bool IsEmpty => W == 0 || H == 0;
        public long Area => (long)W * H;
    }

    /// <summary>One encoded candidate (libwebp WebPMuxFrameInfo + its bitstream): a still lossy file on the accelerator.</summary>
    private sealed class AnimVariant
    {
        public StageBuffer<byte>? File;     // null: the shared 1x1 frame
        public int Length;
        public bool HasAlpha;
        public AnimRect Rect;
        public bool Blend;
        public int Dispose;                 // 0 NONE, 1 BACKGROUND
    }

    /// <summary>libwebp EncodedFrame: the sub-frame and key-frame candidates of one frame.</summary>
    private sealed class AnimFrame
    {
        public AnimVariant? Sub, Key;
        public bool IsKey;
        public int Duration;
        public AnimVariant Chosen => IsKey ? Key! : Sub!;
    }

    /// <summary>Where the file is written: a caller-owned view (overflow is measured, then reported) or a growing buffer.</summary>
    private sealed class AnimSink
    {
        private readonly Accelerator _acc;
        private readonly AnimBuffers? _grow;
        private readonly ArrayView1D<byte, Stride1D.Dense> _fixed;
        private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, int, int>? _copy;
        private long _written;

        public AnimSink(Accelerator acc, ArrayView1D<byte, Stride1D.Dense> output) { _acc = acc; _fixed = output; }

        public AnimSink(Accelerator acc, AnimBuffers grow) { _acc = acc; _grow = grow; _copy = grow.Copy; }

        public bool Overflowed { get; private set; }

        /// <summary>A view that holds bytes [0, end), or null (fixed output too small: the write is skipped and reported at the end).</summary>
        public async ValueTask<ArrayView1D<byte, Stride1D.Dense>?> ReserveAsync(long end)
        {
            if (_grow == null)
            {
                if (end > _fixed.Length) { Overflowed = true; return null; }
                return _fixed;
            }
            var cur = _grow.Out;
            if (cur == null || cur.Length < end)
            {
                long size = Math.Max(end, Math.Max(64 * 1024, (cur?.Length ?? 0) * 2));
                if (size > int.MaxValue) throw new InvalidOperationException("The animation is too large to encode (over 2 GB).");
                var next = _acc.Allocate1D<byte>(size);
                if (cur != null)
                {
                    if (_written > 0) _copy!((int)_written, next.View, cur.View, 0, 0);
                    await _acc.SynchronizeAsync();
                    cur.Dispose();
                }
                _grow.Out = next;
            }
            _written = Math.Max(_written, end);
            return _grow.Out!.View;
        }
    }

    /// <summary>The state of one WebPAnimEncoder run.</summary>
    private sealed class AnimRun
    {
        public int Width, Height, MaxDiff, Kmin, Kmax, LoopCount;
        public uint Background;
        public WebPLossyEncodeOptions Config = null!;
        public readonly List<AnimFrame> Frames = new();   // encoded_frames[start ..], position 0 = start
        public int FlushCount, Keyframe = -1, CountSinceKey, FirstTimestamp, PrevTimestamp, InFrames, OutFrames;
        public long BestDelta = DeltaInfinity;
        public bool PrevUndecided, IsFirst = true;
        public AnimRect PrevRect;
        public AnimFrame? FirstOut;          // the first frame written, kept until a second one is (single-frame file)
        public long Cursor, FramesStart;
        public bool AnyAlpha;
        public byte[]? Icc, Exif, Xmp;
        public AnimSink Sink = null!;
    }

    /// <summary>libwebp SanitizeEncoderOptions (minimize_size off).</summary>
    private static (int Kmin, int Kmax) SanitizeKeyframes(int kmin, int kmax)
    {
        if (kmax == 1) return (0, 0);
        if (kmax <= 0) { kmax = int.MaxValue; kmin = kmax - 1; }
        if (kmin >= kmax) kmin = kmax - 1;
        else
        {
            int kminLimit = kmax / 2 + 1;
            if (kmin < kminLimit && kminLimit < kmax) kmin = kminLimit;
        }
        if (kmax - kmin > 30) kmin = kmax - 30;
        return (kmin, kmax);
    }

    private static int ChunkLength(byte[]? payload) => payload == null ? 0 : 8 + payload.Length + (payload.Length & 1);

    private static void ValidateAnimation(int frameCount, int width, int height, IReadOnlyList<int> durationsMs, WebPLossyEncodeOptions o)
    {
        if (frameCount < 1) throw new ArgumentException("An animation needs at least one frame.");
        ArgumentNullException.ThrowIfNull(durationsMs);
        if (o.IccProfile != null || o.Exif != null || o.Xmp != null)
            throw new ArgumentException("An animation takes its ICC / EXIF / XMP from WebPLossyAnimationOptions, not the per-frame options.");
        if (durationsMs.Count != frameCount)
            throw new ArgumentException($"{durationsMs.Count} durations for {frameCount} frames.", nameof(durationsMs));
        for (int k = 0; k < frameCount; k++)
            if (durationsMs[k] < 0 || durationsMs[k] >= MaxDuration)
                throw new ArgumentException($"frame {k}: duration must be 0..{MaxDuration - 1} ms.", nameof(durationsMs));
        Validate(width, height, o);
    }

    /// <summary>
    /// Encode an animation whose frames (full-canvas RGBA8, straight alpha) are on the accelerator, as libwebp's
    /// WebPAnimEncoder does for Pillow's animated lossy save. The file goes to <paramref name="output"/>; returns its length.
    /// Throws <see cref="ArgumentException"/> naming the needed size when <paramref name="output"/> is too small.
    /// </summary>
    /// <param name="frames">Per frame, <paramref name="width"/> x <paramref name="height"/> x 4 bytes of RGBA8 (rows top to bottom).</param>
    /// <param name="width">Canvas width (1..16383).</param>
    /// <param name="height">Canvas height (1..16383).</param>
    /// <param name="durationsMs">Per frame, how long it shows (0..2^24 - 1 ms).</param>
    /// <param name="output">Receives the WebP file.</param>
    /// <param name="options">Lossy settings of every frame (quality, method; Pillow: quality 80, method as given).</param>
    /// <param name="animation">Animation settings and metadata chunks (null = Pillow's lossy defaults, no metadata).</param>
    public async Task<int> EncodeAnimationAsync(
        IReadOnlyList<ArrayView1D<byte, Stride1D.Dense>> frames, int width, int height, IReadOnlyList<int> durationsMs,
        ArrayView1D<byte, Stride1D.Dense> output, WebPLossyEncodeOptions? options = null, WebPLossyAnimationOptions? animation = null)
    {
        ArgumentNullException.ThrowIfNull(frames);
        var o = options ?? new WebPLossyEncodeOptions();
        ValidateAnimation(frames.Count, width, height, durationsMs, o);
        long need = (long)width * height * 4;
        for (int k = 0; k < frames.Count; k++)
            if (frames[k].Length < need) throw new ArgumentException($"frame {k}: must hold {need} bytes.", nameof(frames));
        var sink = new AnimSink(_accelerator, output);
        int length = await EncodeAnimationCoreAsync(frames.Count, k => new ValueTask<ArrayView1D<byte, Stride1D.Dense>>(frames[k]),
            width, height, durationsMs, o, animation ?? new WebPLossyAnimationOptions(), sink);
        if (sink.Overflowed) throw new ArgumentException($"output must hold {length} bytes.", nameof(output));
        return length;
    }

    /// <summary>
    /// Encode an animation of managed RGBA8 frames (full canvas each). Returns the file as a managed array (a terminal CPU
    /// sink; in a browser prefer the <see cref="TypedArray"/> overload). Frames are uploaded one at a time.
    /// </summary>
    public async Task<byte[]> EncodeAnimationAsync(
        IReadOnlyList<ReadOnlyMemory<byte>> frames, int width, int height, IReadOnlyList<int> durationsMs,
        WebPLossyEncodeOptions? options = null, WebPLossyAnimationOptions? animation = null)
    {
        ArgumentNullException.ThrowIfNull(frames);
        var o = options ?? new WebPLossyEncodeOptions();
        ValidateAnimation(frames.Count, width, height, durationsMs, o);
        int need = width * height * 4;
        for (int k = 0; k < frames.Count; k++)
            if (frames[k].Length < need) throw new ArgumentException($"frame {k}: must hold {need} bytes.", nameof(frames));
        var a = _anim ??= new AnimBuffers(_accelerator);
        var input = await a.FrameIn.EnsureDrainedAsync(need);
        int length = await EncodeAnimationCoreAsync(frames.Count, async k =>
        {
            await _accelerator.SynchronizeAsync();   // the previous frame's kernels are done with the input buffer
            DeviceTransfer.Upload(input.SubView(0, need), frames[k].Span.Slice(0, need));
            return input;
        }, width, height, durationsMs, o, animation ?? new WebPLossyAnimationOptions(), new AnimSink(_accelerator, a));
        return await DeviceTransfer.DownloadAsync(a.Out!.View, length);
    }

    /// <summary>Encode an animation of RGBA8 frames held in JS typed arrays (browser accelerators); the file comes back as a JS Uint8Array.</summary>
    [SupportedOSPlatform("browser")]
    public async Task<Uint8Array> EncodeAnimationAsync(
        IReadOnlyList<TypedArray> frames, int width, int height, IReadOnlyList<int> durationsMs,
        WebPLossyEncodeOptions? options = null, WebPLossyAnimationOptions? animation = null)
    {
        ArgumentNullException.ThrowIfNull(frames);
        var o = options ?? new WebPLossyEncodeOptions();
        ValidateAnimation(frames.Count, width, height, durationsMs, o);
        int need = width * height * 4;
        for (int k = 0; k < frames.Count; k++)
            if (frames[k] == null || frames[k].ByteLength < need) throw new ArgumentException($"frame {k}: must hold {need} bytes.", nameof(frames));
        var a = _anim ??= new AnimBuffers(_accelerator);
        var input = await a.FrameIn.EnsureDrainedAsync(need);
        int length = await EncodeAnimationCoreAsync(frames.Count, async k =>
        {
            await _accelerator.SynchronizeAsync();
            DeviceTransfer.Upload(input.SubView(0, need), frames[k]);
            return input;
        }, width, height, durationsMs, o, animation ?? new WebPLossyAnimationOptions(), new AnimSink(_accelerator, a));
        return await DeviceTransfer.DownloadUint8ArrayAsync(a.Out!.View, length);
    }

    private async Task<int> EncodeAnimationCoreAsync(int frameCount, Func<int, ValueTask<ArrayView1D<byte, Stride1D.Dense>>> frameAt,
        int width, int height, IReadOnlyList<int> durationsMs, WebPLossyEncodeOptions o, WebPLossyAnimationOptions anim, AnimSink sink)
    {
        if (anim.LoopCount < 0 || anim.LoopCount > 65535) throw new ArgumentOutOfRangeException(nameof(anim), "LoopCount must be 0..65535.");
        var a = _anim ??= new AnimBuffers(_accelerator);
        a.ResetPool();   // runs never overlap: every candidate file of an earlier (possibly failed) run is free again
        var (kmin, kmax) = SanitizeKeyframes(anim.KeyframeMin, anim.KeyframeMax);
        var run = new AnimRun
        {
            Width = width, Height = height, Config = o, Kmin = kmin, Kmax = kmax,
            LoopCount = anim.LoopCount, Background = anim.BackgroundColor,
            MaxDiff = K.QualityToMaxDiff(o.Quality),
            Icc = anim.IccProfile is { Length: > 0 } ? anim.IccProfile : null,
            Exif = anim.Exif is { Length: > 0 } ? anim.Exif : null,
            Xmp = anim.Xmp is { Length: > 0 } ? anim.Xmp : null,
            Sink = sink,
        };
        int pixels = width * height;
        await a.Prev.EnsureDrainedAsync((long)pixels * 4);
        await a.PrevDisposed.EnsureDrainedAsync((long)pixels * 4);
        await a.ColFlags.EnsureDrainedAsync(width);
        await a.RowFlags.EnsureDrainedAsync(height);
        a.ClearCanvas(pixels, a.Prev.View);   // the previous canvas starts fully transparent (WebPUtilClearPic)

        // metadata payloads (caller-provided blobs) go up once; the chunk headers around them are written by kernels
        int blobTotal = (run.Icc?.Length ?? 0) + (run.Exif?.Length ?? 0) + (run.Xmp?.Length ?? 0);
        var blobs = await a.Blobs.EnsureDrainedAsync(Math.Max(1, blobTotal));
        int at = 0;
        foreach (var b in new[] { run.Icc, run.Exif, run.Xmp })
        {
            if (b == null) continue;
            DeviceTransfer.Upload(blobs.SubView(at, b.Length), b);
            at += b.Length;
        }

        run.FramesStart = run.Cursor = 12 + 18 + ChunkLength(run.Icc) + 14;   // RIFF, VP8X, ICCP, ANIM, then the frames
        int timestamp = 0;
        for (int k = 0; k < frameCount; k++)
        {
            var frame = await frameAt(k);
            await AddFrameAsync(run, frame, timestamp);
            timestamp = unchecked(timestamp + durationsMs[k]);
        }
        AddEnd(run, timestamp);
        return await AssembleAnimationAsync(run);
    }

    /// <summary>WebPAnimEncoderAdd with a frame.</summary>
    private async Task AddFrameAsync(AnimRun run, ArrayView1D<byte, Stride1D.Dense> frame, int timestamp)
    {
        if (!run.IsFirst) IncreasePreviousDuration(run, PreviousDuration(run, timestamp));
        else run.FirstTimestamp = timestamp;
        await CacheFrameAsync(run, frame);
        await FlushFramesAsync(run);
        run.PrevTimestamp = timestamp;
    }

    /// <summary>WebPAnimEncoderAdd(NULL): the end timestamp gives the last frame its duration.</summary>
    private static void AddEnd(AnimRun run, int timestamp)
    {
        IncreasePreviousDuration(run, PreviousDuration(run, timestamp));
        run.PrevTimestamp = timestamp;
    }

    private static int PreviousDuration(AnimRun run, int timestamp)
    {
        uint d = unchecked((uint)timestamp - (uint)run.PrevTimestamp);
        if (d >= MaxDuration) throw new ArgumentException("Frame timestamps must be non-decreasing.");
        return (int)d;
    }

    private static void IncreasePreviousDuration(AnimRun run, int duration)
    {
        var prev = run.Frames[^1];
        int newDuration = prev.Duration + duration;
        if (newDuration >= MaxDuration)
        {
            // Separate the previous frame from the merged ones: a 1x1 transparent frame, blended, carries this duration.
            var f = new AnimFrame
            {
                IsKey = false, Duration = duration,
                Sub = new AnimVariant { File = null, Length = Lossy1x1.Length, HasAlpha = true, Rect = new AnimRect(0, 0, 1, 1), Blend = true },
            };
            run.Frames.Add(f);
            run.CountSinceKey++;
            run.FlushCount = run.Frames.Count - 1;
            run.PrevUndecided = false;
            run.PrevRect = new AnimRect(0, 0, 1, 1);
        }
        else
        {
            prev.Duration = newDuration;
        }
    }

    /// <summary>libwebp CacheFrame.</summary>
    private async Task CacheFrameAsync(AnimRun run, ArrayView1D<byte, Stride1D.Dense> curr)
    {
        var a = _anim!;
        int position = run.Frames.Count;
        var f = new AnimFrame();
        run.Frames.Add(f);
        bool skipped = false;
        if (run.IsFirst)
        {
            await SetFrameAsync(run, curr, true, f);   // the first frame cannot be skipped (empty rectangle not allowed)
            f.IsKey = true;
            run.FlushCount = 0;
            run.CountSinceKey = 0;
            run.PrevUndecided = false;
        }
        else
        {
            run.CountSinceKey++;
            if (run.CountSinceKey <= run.Kmin)
            {
                skipped = await SetFrameAsync(run, curr, false, f);
                if (!skipped)
                {
                    f.IsKey = false;
                    run.FlushCount = run.Frames.Count - 1;
                    run.PrevUndecided = false;
                }
            }
            else
            {
                skipped = await SetFrameAsync(run, curr, false, f);
                if (!skipped)
                {
                    var prevRectSub = run.PrevRect;
                    await SetFrameAsync(run, curr, true, f);
                    var prevRectKey = run.PrevRect;
                    long delta = (long)f.Key!.Length - f.Sub!.Length;   // KeyFramePenalty
                    if (delta <= run.BestDelta)
                    {
                        if (run.Keyframe != -1) run.Frames[run.Keyframe].IsKey = false;
                        f.IsKey = true;
                        run.PrevUndecided = true;
                        run.Keyframe = position;
                        run.BestDelta = delta;
                        run.FlushCount = run.Frames.Count - 1;
                    }
                    else
                    {
                        f.IsKey = false;
                        run.PrevUndecided = false;
                    }
                    if (run.CountSinceKey >= run.Kmax)
                    {
                        run.FlushCount = run.Frames.Count - 1;
                        run.CountSinceKey = 0;
                        run.Keyframe = -1;
                        run.BestDelta = DeltaInfinity;
                    }
                    if (!run.PrevUndecided) run.PrevRect = f.IsKey ? prevRectKey : prevRectSub;
                }
            }
        }

        run.InFrames++;
        if (skipped)
        {
            ReleaseFrame(f);
            run.Frames.RemoveAt(run.Frames.Count - 1);
            run.CountSinceKey--;   // never the first frame
            return;
        }
        a.CopyCanvas(run.Width * run.Height, curr, a.Prev.View);   // the previous canvas is the input frame, unmodified
        run.IsFirst = false;
    }

    /// <summary>
    /// GetSubRects: the lossless change rectangle (seed: the canvas), then the lossy one seeded with it, into two meta slots.
    /// A key frame other than the first keeps the whole canvas.
    /// </summary>
    private void GetSubRects(AnimRun run, ArrayView1D<byte, Stride1D.Dense> prev, ArrayView1D<byte, Stride1D.Dense> curr,
        bool isKey, bool isFirst, int slotLl, int slotLossy, int emptyAllowed)
    {
        var a = _anim!;
        var meta = a.Meta.View;
        int w = run.Width, h = run.Height;
        a.SetRect(1, meta, slotLl, 0, 0, w, h);
        if (!isKey || isFirst)
        {
            Minimize(run, prev, curr, slotLl, -1, emptyAllowed);
            a.CopyRect(1, meta, slotLl, slotLossy);
            Minimize(run, prev, curr, slotLossy, run.MaxDiff, emptyAllowed);
        }
        else
        {
            a.CopyRect(1, meta, slotLl, slotLossy);   // the whole canvas: not empty, offsets already even
        }
    }

    private void Minimize(AnimRun run, ArrayView1D<byte, Stride1D.Dense> prev, ArrayView1D<byte, Stride1D.Dense> curr, int slot, int maxDiff, int emptyAllowed)
    {
        var a = _anim!;
        var meta = a.Meta.View;
        a.Columns(run.Width, prev, curr, meta, a.ColFlags.View, run.Width, slot, maxDiff);
        a.ColumnsReduce(1, meta, a.ColFlags.View, slot);
        a.Rows(run.Height, prev, curr, meta, a.RowFlags.View, run.Width, slot, maxDiff);
        a.RowsReduce(1, meta, a.RowFlags.View, slot, emptyAllowed);
    }

    /// <summary>libwebp SetFrame for a lossy-only encoder: one candidate. Returns true when the frame is skipped.</summary>
    private async Task<bool> SetFrameAsync(AnimRun run, ArrayView1D<byte, Stride1D.Dense> curr, bool isKey, AnimFrame f)
    {
        var a = _anim!;
        var meta = a.Meta.View;
        bool isFirst = run.IsFirst;
        int emptyAllowedNone = isFirst ? 0 : 1;
        bool disposeBgPossible = !isKey && !run.PrevUndecided;
        var prev = a.Prev.View;
        var prevDisposed = a.PrevDisposed.View;

        GetSubRects(run, prev, curr, isKey, isFirst, K.SlotLlNone, K.SlotLossyNone, emptyAllowedNone);
        if (disposeBgPossible)
        {
            a.CopyCanvas(run.Width * run.Height, prev, prevDisposed);
            var pr = run.PrevRect;
            if (!pr.IsEmpty) a.ClearRect(pr.W * pr.H, prevDisposed, run.Width, pr.X, pr.Y, pr.W);
            GetSubRects(run, prevDisposed, curr, isKey, isFirst, K.SlotLlBg, K.SlotLossyBg, 0);
        }
        if (!isKey)
        {
            // IsLossyBlendingPossible for each candidate rectangle (key frames never blend)
            a.BlendRows(run.Height, prev, curr, meta, a.RowFlags.View, run.Width, K.SlotLossyNone, run.MaxDiff);
            a.Any(1, meta, a.RowFlags.View, 0, run.Height, K.FlagBlendImpossible);
            if (disposeBgPossible)
            {
                a.BlendRows(run.Height, prevDisposed, curr, meta, a.RowFlags.View, run.Width, K.SlotLossyBg, run.MaxDiff);
                a.Any(1, meta, a.RowFlags.View, 0, run.Height, K.FlagBlendImpossible + 1);
            }
        }
        await _accelerator.SynchronizeAsync();
        var m = await meta.CopyToHostAsync();

        var none = new AnimRect(m[K.SlotLossyNone], m[K.SlotLossyNone + 1], m[K.SlotLossyNone + 2], m[K.SlotLossyNone + 3]);
        if (none.IsEmpty) return true;   // nothing changed beyond the lossy tolerance: the previous frame shows longer
        bool tryBg = false;
        var bg = default(AnimRect);
        if (disposeBgPossible)
        {
            bg = new AnimRect(m[K.SlotLossyBg], m[K.SlotLossyBg + 1], m[K.SlotLossyBg + 2], m[K.SlotLossyBg + 3]);
            tryBg = bg.Area < none.Area;
        }

        AnimVariant v;
        int dispose;
        if (!tryBg)
        {
            bool blend = !isKey && m[K.FlagBlendImpossible] == 0;
            v = await EncodeCandidateAsync(run, curr, prev, none, blend);
            dispose = 0;
        }
        else
        {
            bool blend = m[K.FlagBlendImpossible + 1] == 0;
            v = await EncodeCandidateAsync(run, curr, prevDisposed, bg, blend);
            dispose = 1;
        }

        // PickBestCandidate (one candidate)
        if (isKey)
        {
            f.Key = v;
        }
        else
        {
            f.Sub = v;
            SetPreviousDisposeMethod(run, dispose);
        }
        run.PrevRect = v.Rect;
        return false;
    }

    private static void SetPreviousDisposeMethod(AnimRun run, int dispose)
    {
        var prev = run.Frames[run.Frames.Count - 2];
        if (run.PrevUndecided)
        {
            if (prev.Sub != null) prev.Sub.Dispose = dispose;
            if (prev.Key != null) prev.Key.Dispose = dispose;
        }
        else
        {
            prev.Chosen.Dispose = dispose;
        }
    }

    /// <summary>
    /// EncodeCandidate: the rectangle of the current canvas as a still picture (flattened against the reference canvas
    /// when blending, then encoded with the loop filter off), written as a complete still file into a pooled buffer.
    /// </summary>
    private async Task<AnimVariant> EncodeCandidateAsync(AnimRun run, ArrayView1D<byte, Stride1D.Dense> curr,
        ArrayView1D<byte, Stride1D.Dense> reference, AnimRect r, bool blend)
    {
        var a = _anim!;
        var picture = await a.Picture.EnsureDrainedAsync((long)r.W * r.H * 4);
        a.Extract(r.W * r.H, curr, picture, run.Width, r.X, r.Y, r.W);
        if (blend)
        {
            // FlattenSimilarBlocks walks 8x8 blocks of the canvas grid: x from (x + 8) & ~7 to (x + w) & ~7, same for y
            int xStart = (r.X + 8) & ~7, xEnd = (r.X + r.W) & ~7;
            int yStart = (r.Y + 8) & ~7, yEnd = (r.Y + r.H) & ~7;
            if (xEnd > xStart && yEnd > yStart)
            {
                int bx = (xEnd - xStart) / 8, by = (yEnd - yStart) / 8;
                a.Flatten(bx * by, reference, picture, run.Width, r.X, r.Y, r.W, xStart, yStart, bx, run.MaxDiff);
            }
        }
        var o = blend ? run.Config with { FilterStrength = 0 } : run.Config;
        var (simpleLength, alphaLength) = await EncodeToScratchAsync(picture, r.W, r.H, o);
        int length = FileLength(simpleLength, alphaLength);
        var file = a.RentFile();
        var view = await file.EnsureDrainedAsync(length);
        Assemble(view, length, alphaLength);
        return new AnimVariant { File = file, Length = length, HasAlpha = alphaLength >= 0, Rect = r, Blend = blend };
    }

    private void ReleaseVariant(AnimVariant? v)
    {
        if (v?.File != null) _anim!.FreeFiles.Push(v.File);
        if (v != null) v.File = null;
    }

    private void ReleaseFrame(AnimFrame f)
    {
        ReleaseVariant(f.Sub);
        ReleaseVariant(f.Key);
    }

    private ArrayView1D<byte, Stride1D.Dense> FileView(AnimVariant v) => v.File?.View ?? _anim!.OneByOne.View;

    /// <summary>Bytes before a still file's image chunks: RIFF (12), plus VP8X (18) when it has alpha.</summary>
    private static int ImageChunksStart(AnimVariant v) => v.HasAlpha ? 30 : 12;

    /// <summary>libwebp FlushFrames: frames no later frame can change are written as ANMF chunks.</summary>
    private async Task FlushFramesAsync(AnimRun run)
    {
        var a = _anim!;
        while (run.FlushCount > 0)
        {
            var f = run.Frames[0];
            var v = f.Chosen;
            int skip = ImageChunksStart(v);
            int payload = v.Length - skip;
            long end = run.Cursor + 24 + payload;
            var output = await run.Sink.ReserveAsync(end);
            if (output is { } o)
            {
                int flags = (v.Dispose == 1 ? 1 : 0) | (v.Blend ? 0 : 2);
                a.Frame(24 + payload, o, FileView(v), (int)run.Cursor, skip, payload, v.Rect.X, v.Rect.Y, v.Rect.W, v.Rect.H, f.Duration, flags);
            }
            run.Cursor = end;
            run.AnyAlpha |= v.HasAlpha;
            run.OutFrames++;
            if (run.OutFrames == 1)
            {
                run.FirstOut = f;   // kept: a lone frame becomes a still image at the end
            }
            else
            {
                if (run.FirstOut != null) { ReleaseFrame(run.FirstOut); run.FirstOut = null; }
                ReleaseFrame(f);
            }
            run.Frames.RemoveAt(0);
            run.FlushCount--;
            if (run.Keyframe != -1) run.Keyframe--;
        }
    }

    /// <summary>WebPAnimEncoderAssemble: flush the rest, then the container; a lone frame becomes a still (MuxCleanup, OptimizeSingleFrame).</summary>
    private async Task<int> AssembleAnimationAsync(AnimRun run)
    {
        var a = _anim!;
        run.FlushCount = run.Frames.Count;
        await FlushFramesAsync(run);

        int tail = ChunkLength(run.Exif) + ChunkLength(run.Xmp);
        long animatedLength = run.Cursor + tail;
        if (run.OutFrames == 1)
        {
            var f = run.FirstOut!;
            var v = f.Chosen;
            bool fullCanvas = v.Rect.X == 0 && v.Rect.Y == 0 && v.Rect.W == run.Width && v.Rect.H == run.Height;
            if (fullCanvas)
            {
                // MuxCleanup: one frame covering the canvas is written as a non-animated image
                int len = await WriteStillAsync(run, v);
                ReleaseFrame(f);
                return len;
            }
            // OptimizeSingleFrame: the frame decoded onto a cleared canvas, re-encoded whole; the smaller file wins
            var full = await EncodeFrameOnCanvasAsync(run, v);
            long stillLength = StillLength(run, full);
            if (stillLength < animatedLength)
            {
                int len = await WriteStillAsync(run, full);
                ReleaseVariant(full);
                ReleaseFrame(f);
                return len;
            }
            ReleaseVariant(full);
            ReleaseFrame(f);
        }
        else if (run.FirstOut != null)
        {
            ReleaseFrame(run.FirstOut);
            run.FirstOut = null;
        }

        var output = await run.Sink.ReserveAsync(animatedLength);
        if (animatedLength > int.MaxValue) throw new InvalidOperationException("The animation is too large to encode (over 2 GB).");
        if (output is { } o)
        {
            int flags = 0x02 | (run.AnyAlpha ? 0x10 : 0) | (run.Icc != null ? 0x20 : 0) | (run.Exif != null ? 0x08 : 0) | (run.Xmp != null ? 0x04 : 0);
            a.FileHeader(30, o, (int)animatedLength - 8, flags, run.Width, run.Height);
            int p = 30;
            int blob = 0;
            if (run.Icc != null)
            {
                a.BlobChunk(ChunkLength(run.Icc), o, a.Blobs.View, p, 0x50434349, blob, run.Icc.Length);   // "ICCP"
                p += ChunkLength(run.Icc);
                blob += run.Icc.Length;
            }
            a.AnimChunk(14, o, p, unchecked((int)run.Background), run.LoopCount);
            WriteMetadataTail(o, (int)run.Cursor, blob, run.Exif, run.Xmp);
            await _accelerator.SynchronizeAsync();
        }
        return (int)animatedLength;
    }

    private static long StillLength(AnimRun run, AnimVariant v) => StillFileLength(v.Length, v.HasAlpha, run.Icc, run.Exif, run.Xmp);

    /// <summary>The non-animated file of one canvas-sized image (MuxCleanup / OptimizeSingleFrame).</summary>
    private async Task<int> WriteStillAsync(AnimRun run, AnimVariant v)
    {
        long length = StillLength(run, v);
        var output = await run.Sink.ReserveAsync(length);
        if (output is { } o)
        {
            WriteStillFile(o, FileView(v), v.Length, v.HasAlpha, run.Width, run.Height, run.Icc, run.Exif, run.Xmp);
            await _accelerator.SynchronizeAsync();
        }
        return (int)length;
    }

    /// <summary>FrameToFullCanvas: decode the frame onto a cleared canvas at its offset and encode the whole canvas.</summary>
    private async Task<AnimVariant> EncodeFrameOnCanvasAsync(AnimRun run, AnimVariant v)
    {
        var a = _anim!;
        a.Decoder ??= new WebPDecoderGpu(_accelerator);
        var picture = await a.Picture.EnsureDrainedAsync((long)v.Rect.W * v.Rect.H * 4);
        await a.Decoder.DecodeAsync(FileView(v), v.Length, picture);
        int pixels = run.Width * run.Height;
        var canvas = await a.Canvas.EnsureDrainedAsync((long)pixels * 4);
        a.ClearCanvas(pixels, canvas);
        a.Place(v.Rect.W * v.Rect.H, picture, canvas, run.Width, v.Rect.X, v.Rect.Y, v.Rect.W);
        var (simpleLength, alphaLength) = await EncodeToScratchAsync(canvas, run.Width, run.Height, run.Config);
        int length = FileLength(simpleLength, alphaLength);
        var file = a.RentFile();
        var view = await file.EnsureDrainedAsync(length);
        Assemble(view, length, alphaLength);
        return new AnimVariant { File = file, Length = length, HasAlpha = alphaLength >= 0, Rect = new AnimRect(0, 0, run.Width, run.Height) };
    }
}
