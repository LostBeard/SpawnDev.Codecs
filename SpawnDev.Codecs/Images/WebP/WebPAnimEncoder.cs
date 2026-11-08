// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// WebPAnimEncoder: a port of libwebp 1.6.0's src/mux/anim_encode.c as Pillow drives it for `save(save_all=True, ...)`
// (ComfyUI's Save Animated WEBP): allow_mixed off, minimize_size off, kmin / kmax from the options, one WebPConfig for
// every frame, lossy OR lossless. Shared by WebPLossyEncoderGpu.EncodeAnimationAsync and
// WebPLosslessEncoderGpu.EncodeAnimationAsync, which hand it their still encoder as an AnimFrameEncoder.
//
// Per input frame, exactly as libwebp:
//   * the change rectangle against the previous canvas (MinimizeChangeRectangle: exact for lossless; exact then within
//     QualityToMaxDiff(quality) for lossy), snapped to even offsets, and against the previous canvas disposed to
//     background when that is allowed; the smaller rectangle picks the previous frame's disposal;
//   * a frame whose rectangle is empty is dropped and its duration goes to the previous frame;
//   * blending when it can reproduce the frame (IsLosslessBlendingPossible / IsLossyBlendingPossible); a blended lossless
//     frame has the pixels equal to the canvas made transparent (IncreaseTransparency), a blended lossy one its
//     fully-similar 8x8 blocks (FlattenSimilarBlocks) and the loop filter off;
//   * key frames: after kmin sub-frames each frame is encoded twice (sub-frame and full-canvas key frame) and the one with
//     the smallest size penalty becomes the key frame, decided by kmax;
//   * frames reach the file when no later frame can change them (FlushFrames); a lone frame becomes a still image
//     (MuxCleanup / OptimizeSingleFrame); durations past 24 bits are split with libwebp's 1x1 frame.
//
// The host is a coordinator: the pixel tests, rectangles, transparency, canvases and every byte of the file are kernels
// (WebPAnimKernels); the host reads back the rectangles, two flags and the encoded sizes per frame and makes libwebp's
// decisions from those scalars. Key frames are chosen by encoded size, so they match libwebp's where our frame sizes do
// (the lossy VP8 chunks are byte-identical with libwebp's; our VP8L coder - lossless frames, lossy ALPH - is not).

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using K = SpawnDev.Codecs.Images.WebP.Lossy.WebPAnimKernels;

namespace SpawnDev.Codecs.Images.WebP;

/// <summary>Animation-wide settings of an animated WebP encode (libwebp WebPAnimEncoderOptions + metadata chunks).</summary>
public sealed record WebPAnimEncoderOptions
{
    /// <summary>How many times the animation plays (0 = forever, the default).</summary>
    public int LoopCount { get; init; }

    /// <summary>ANIM background colour, 0xAARRGGBB (written little-endian: B, G, R, A). Pillow's default is 0.</summary>
    public uint BackgroundColor { get; init; }

    /// <summary>
    /// libwebp kmin, the minimum distance between key frames. Null = Pillow's default for the mode: 3 lossy, 9 lossless.
    /// Sanitized as libwebp does.
    /// </summary>
    public int? KeyframeMin { get; init; }

    /// <summary>
    /// libwebp kmax, the maximum distance between key frames. Null = Pillow's default: 5 lossy, 17 lossless. 0 or less
    /// turns key frames off (every frame after the first is a sub-frame); 1 makes every frame a key frame.
    /// </summary>
    public int? KeyframeMax { get; init; }

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

/// <summary>One encoded still: length of the complete RIFF file, the VP8X alpha fact, where its image chunks start.</summary>
internal readonly record struct AnimEncoded(int Length, bool HasAlpha, int ImageChunksStart);

/// <summary>
/// Encode an RGBA8 picture (on the accelerator) as a complete still WebP file into <paramref name="file"/> (ensure it,
/// drained, to the length). <paramref name="blend"/>: the frame is blended onto the canvas (lossy: loop filter off).
/// </summary>
internal delegate Task<AnimEncoded> AnimFrameEncoder(ArrayView1D<byte, Stride1D.Dense> rgba, int width, int height, bool blend, StageBuffer<byte> file);

/// <summary>The WebPAnimEncoder port (see the file header). Not thread-safe; owned by one encoder.</summary>
internal sealed class WebPAnimEncoder : IDisposable
{
    /// <summary>libwebp MAX_DURATION: frame durations are 24-bit.</summary>
    public const int MaxDuration = 1 << 24;
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

    /// <summary>libwebp's lossless 1x1 transparent frame.</summary>
    private static readonly byte[] Lossless1x1 =
    {
        0x52, 0x49, 0x46, 0x46, 0x14, 0x00, 0x00, 0x00, 0x57, 0x45, 0x42, 0x50,
        0x56, 0x50, 0x38, 0x4c, 0x08, 0x00, 0x00, 0x00, 0x2f, 0x00, 0x00, 0x00,
        0x10, 0x88, 0x88, 0x08,
    };

    private const int FourccIccp = 0x50434349, FourccExif = 0x46495845, FourccXmp = 0x20504D58;

    private readonly Accelerator _acc;
    private readonly Action<Index1D, ArrayView<int>, int, int, int, int, int> _setRect;
    private readonly Action<Index1D, ArrayView<int>, int, int> _copyRect;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, int, int, int> _columns, _rows, _blendRows;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, int> _columnsReduce;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, int, int> _rowsReduce;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, int, int, int> _any;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, int, int, int, int> _extract, _place;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, int, int, int, int, int, int, int, int> _flatten;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, int, int, int, int> _increaseTransparency;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>> _copyCanvas;
    private readonly Action<Index1D, ArrayView<byte>> _clearCanvas;
    private readonly Action<Index1D, ArrayView<byte>, int, int, int, int> _clearRect;
    private readonly Action<Index1D, ArrayView<byte>, int, int, int, int> _fileHeader;
    private readonly Action<Index1D, ArrayView<byte>, int, int, int> _animChunk;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, int, int, int, int> _blobChunk;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, int, int, int, int, int, int, int, int, int> _frame;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, int, int> _copy;

    private readonly StageBuffer<byte> _prev, _prevDisposed, _canvas, _picture, _blobs;
    private readonly StageBuffer<int> _colFlags, _rowFlags;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _meta;
    private readonly MemoryBuffer1D<byte, Stride1D.Dense> _lossy1x1, _lossless1x1;
    private readonly Stack<StageBuffer<byte>> _freeFiles = new();
    private readonly List<StageBuffer<byte>> _allFiles = new();
    private MemoryBuffer1D<byte, Stride1D.Dense>? _out;
    private WebPDecoderGpu? _decoder;

    /// <summary>Staging buffer of the managed / TypedArray overloads: the current input frame.</summary>
    public StageBuffer<byte> FrameIn { get; }

    /// <summary>The growing output of a managed / TypedArray encode (valid after one).</summary>
    public ArrayView1D<byte, Stride1D.Dense> Output => _out!.View;

    public WebPAnimEncoder(Accelerator a)
    {
        _acc = a;
        _setRect = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, int, int, int, int, int>(K.SetRectKernel);
        _copyRect = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, int, int>(K.CopyRectKernel);
        _columns = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, int, int, int>(K.ColumnsKernel);
        _rows = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, int, int, int>(K.RowsKernel);
        _blendRows = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, int, int, int>(K.BlendRowsKernel);
        _columnsReduce = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, int>(K.ColumnsReduceKernel);
        _rowsReduce = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, int, int>(K.RowsReduceKernel);
        _any = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, int, int, int>(K.AnyKernel);
        _extract = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<byte>, int, int, int, int>(K.ExtractKernel);
        _place = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<byte>, int, int, int, int>(K.PlaceKernel);
        _flatten = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<byte>, int, int, int, int, int, int, int, int>(K.FlattenKernel);
        _increaseTransparency = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<byte>, int, int, int, int>(K.IncreaseTransparencyKernel);
        _copyCanvas = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<byte>>(K.CopyCanvasKernel);
        _clearCanvas = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>>(K.ClearCanvasKernel);
        _clearRect = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, int, int, int, int>(K.ClearRectKernel);
        _fileHeader = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, int, int, int, int>(K.FileHeaderKernel);
        _animChunk = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, int, int, int>(K.AnimChunkKernel);
        _blobChunk = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<byte>, int, int, int, int>(K.BlobChunkKernel);
        _frame = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<byte>, int, int, int, int, int, int, int, int, int>(K.FrameKernel);
        _copy = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<byte>, int, int>(K.CopyKernel);
        _prev = new StageBuffer<byte>(a); _prevDisposed = new StageBuffer<byte>(a); _canvas = new StageBuffer<byte>(a);
        _picture = new StageBuffer<byte>(a); FrameIn = new StageBuffer<byte>(a); _blobs = new StageBuffer<byte>(a);
        _colFlags = new StageBuffer<int>(a); _rowFlags = new StageBuffer<int>(a);
        _meta = a.Allocate1D<int>(K.MetaSize);
        _lossy1x1 = a.Allocate1D<byte>(Lossy1x1.Length);
        _lossy1x1.View.CopyFromCPU(Lossy1x1);
        _lossless1x1 = a.Allocate1D<byte>(Lossless1x1.Length);
        _lossless1x1.View.CopyFromCPU(Lossless1x1);
    }

    public void Dispose()
    {
        _prev.Dispose(); _prevDisposed.Dispose(); _canvas.Dispose(); _picture.Dispose(); FrameIn.Dispose(); _blobs.Dispose();
        _colFlags.Dispose(); _rowFlags.Dispose(); _meta.Dispose(); _lossy1x1.Dispose(); _lossless1x1.Dispose(); _out?.Dispose();
        foreach (var b in _allFiles) b.Dispose();
        _decoder?.Dispose();
    }

    /// <summary>A pooled file buffer (runs never overlap: the pool is reset at the start of each).</summary>
    public StageBuffer<byte> RentFile()
    {
        if (_freeFiles.Count > 0) return _freeFiles.Pop();
        var b = new StageBuffer<byte>(_acc);
        _allFiles.Add(b);
        return b;
    }

    public void ResetPool()
    {
        _freeFiles.Clear();
        foreach (var b in _allFiles) _freeFiles.Push(b);
    }

    // ------------------------------------------------------------------ sinks

    /// <summary>Where the file is written: a caller-owned view (overflow is measured, then reported) or the growing output.</summary>
    public sealed class Sink
    {
        private readonly WebPAnimEncoder _e;
        private readonly bool _grow;
        private readonly ArrayView1D<byte, Stride1D.Dense> _fixed;
        private long _written;

        private Sink(WebPAnimEncoder e, bool grow, ArrayView1D<byte, Stride1D.Dense> output) { _e = e; _grow = grow; _fixed = output; }

        public static Sink Fixed(WebPAnimEncoder e, ArrayView1D<byte, Stride1D.Dense> output) => new(e, false, output);

        public static Sink Growing(WebPAnimEncoder e) => new(e, true, default);

        public bool Overflowed { get; private set; }

        /// <summary>A view that holds bytes [0, end), or null (fixed output too small: skipped, reported at the end).</summary>
        public async ValueTask<ArrayView1D<byte, Stride1D.Dense>?> ReserveAsync(long end)
        {
            if (!_grow)
            {
                if (end > _fixed.Length) { Overflowed = true; return null; }
                return _fixed;
            }
            var cur = _e._out;
            if (cur == null || cur.Length < end)
            {
                long size = Math.Max(end, Math.Max(64 * 1024, (cur?.Length ?? 0) * 2));
                if (size > int.MaxValue) throw new InvalidOperationException("The animation is too large to encode (over 2 GB).");
                var next = _e._acc.Allocate1D<byte>(size);
                if (cur != null)
                {
                    if (_written > 0) _e._copy((int)_written, next.View, cur.View, 0, 0);
                    await _e._acc.SynchronizeAsync();
                    cur.Dispose();
                }
                _e._out = next;
            }
            _written = Math.Max(_written, end);
            return _e._out!.View;
        }
    }

    // ------------------------------------------------------------------ run state

    private record struct Rect(int X, int Y, int W, int H)
    {
        public bool IsEmpty => W == 0 || H == 0;
        public long Area => (long)W * H;
    }

    /// <summary>One encoded candidate (libwebp WebPMuxFrameInfo + its bitstream): a still file on the accelerator.</summary>
    private sealed class Variant
    {
        public StageBuffer<byte>? File;     // null: a shared 1x1 frame
        public bool Lossless1x1;
        public int Length, Skip;
        public bool HasAlpha;
        public Rect Rect;
        public bool Blend;
        public int Dispose;                 // 0 NONE, 1 BACKGROUND
    }

    /// <summary>libwebp EncodedFrame: the sub-frame and key-frame candidates of one frame.</summary>
    private sealed class Frame
    {
        public Variant? Sub, Key;
        public bool IsKey;
        public int Duration;
        public Variant Chosen => IsKey ? Key! : Sub!;
    }

    private sealed class Run
    {
        public int Width, Height, MaxDiff, Kmin, Kmax, LoopCount;
        public uint Background;
        public bool Lossless;
        public AnimFrameEncoder Encode = null!;
        public readonly List<Frame> Frames = new();   // encoded_frames[start ..], position 0 = start
        public int FlushCount, Keyframe = -1, CountSinceKey, PrevTimestamp, OutFrames;
        public long BestDelta = DeltaInfinity;
        public bool PrevUndecided, IsFirst = true;
        public Rect PrevRect;
        public Frame? FirstOut;
        public long Cursor;
        public bool AnyAlpha;
        public byte[]? Icc, Exif, Xmp;
        public Sink Sink = null!;
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

    public static int ChunkLength(byte[]? payload) => payload == null ? 0 : 8 + payload.Length + (payload.Length & 1);

    private static byte[]? Blob(byte[]? b) => b is { Length: > 0 } ? b : null;

    /// <summary>Durations and frame count as libwebp accepts them.</summary>
    public static void ValidateDurations(int frameCount, IReadOnlyList<int> durationsMs)
    {
        if (frameCount < 1) throw new ArgumentException("An animation needs at least one frame.");
        ArgumentNullException.ThrowIfNull(durationsMs);
        if (durationsMs.Count != frameCount)
            throw new ArgumentException($"{durationsMs.Count} durations for {frameCount} frames.", nameof(durationsMs));
        for (int k = 0; k < frameCount; k++)
            if (durationsMs[k] < 0 || durationsMs[k] >= MaxDuration)
                throw new ArgumentException($"frame {k}: duration must be 0..{MaxDuration - 1} ms.", nameof(durationsMs));
    }

    // ------------------------------------------------------------------ the encode

    /// <summary>
    /// Run WebPAnimEncoder over <paramref name="frameCount"/> full-canvas frames (<paramref name="frameAt"/> gives each on
    /// the accelerator); returns the file length written to <paramref name="sink"/>.
    /// </summary>
    public async Task<int> EncodeAsync(int frameCount, Func<int, ValueTask<ArrayView1D<byte, Stride1D.Dense>>> frameAt,
        int width, int height, IReadOnlyList<int> durationsMs, bool lossless, float quality, AnimFrameEncoder encode,
        WebPAnimEncoderOptions anim, Sink sink)
    {
        if (anim.LoopCount < 0 || anim.LoopCount > 65535) throw new ArgumentOutOfRangeException(nameof(anim), "LoopCount must be 0..65535.");
        ResetPool();
        var (kmin, kmax) = SanitizeKeyframes(anim.KeyframeMin ?? (lossless ? 9 : 3), anim.KeyframeMax ?? (lossless ? 17 : 5));
        var run = new Run
        {
            Width = width, Height = height, Kmin = kmin, Kmax = kmax, Lossless = lossless, Encode = encode,
            MaxDiff = K.QualityToMaxDiff(quality), LoopCount = anim.LoopCount, Background = anim.BackgroundColor,
            Icc = Blob(anim.IccProfile), Exif = Blob(anim.Exif), Xmp = Blob(anim.Xmp), Sink = sink,
        };
        int pixels = width * height;
        await _prev.EnsureDrainedAsync((long)pixels * 4);
        await _prevDisposed.EnsureDrainedAsync((long)pixels * 4);
        await _colFlags.EnsureDrainedAsync(width);
        await _rowFlags.EnsureDrainedAsync(height);
        _clearCanvas(pixels, _prev.View);   // the previous canvas starts fully transparent (WebPUtilClearPic)
        await UploadBlobsAsync(run.Icc, run.Exif, run.Xmp);

        run.Cursor = 12 + 18 + ChunkLength(run.Icc) + 14;   // RIFF, VP8X, ICCP, ANIM, then the frames
        int timestamp = 0;
        for (int k = 0; k < frameCount; k++)
        {
            var frame = await frameAt(k);
            if (!run.IsFirst) IncreasePreviousDuration(run, PreviousDuration(run, timestamp));
            await CacheFrameAsync(run, frame);
            await FlushFramesAsync(run);
            run.PrevTimestamp = timestamp;
            timestamp = unchecked(timestamp + durationsMs[k]);
        }
        // WebPAnimEncoderAdd(NULL): the end timestamp gives the last frame its duration
        IncreasePreviousDuration(run, PreviousDuration(run, timestamp));
        run.PrevTimestamp = timestamp;
        return await AssembleAsync(run);
    }

    private static int PreviousDuration(Run run, int timestamp)
    {
        uint d = unchecked((uint)timestamp - (uint)run.PrevTimestamp);
        if (d >= MaxDuration) throw new ArgumentException("Frame timestamps must be non-decreasing.");
        return (int)d;
    }

    private static void IncreasePreviousDuration(Run run, int duration)
    {
        var prev = run.Frames[^1];
        int newDuration = prev.Duration + duration;
        if (newDuration >= MaxDuration)
        {
            // Separate the previous frame from the merged ones: a 1x1 transparent frame, blended, carries this duration.
            var f = new Frame
            {
                IsKey = false, Duration = duration,
                Sub = run.Lossless
                    ? new Variant { Lossless1x1 = true, Length = Lossless1x1.Length, Skip = 12, HasAlpha = true, Rect = new Rect(0, 0, 1, 1), Blend = true }
                    : new Variant { Length = Lossy1x1.Length, Skip = 30, HasAlpha = true, Rect = new Rect(0, 0, 1, 1), Blend = true },
            };
            run.Frames.Add(f);
            run.CountSinceKey++;
            run.FlushCount = run.Frames.Count - 1;
            run.PrevUndecided = false;
            run.PrevRect = new Rect(0, 0, 1, 1);
        }
        else
        {
            prev.Duration = newDuration;
        }
    }

    /// <summary>libwebp CacheFrame.</summary>
    private async Task CacheFrameAsync(Run run, ArrayView1D<byte, Stride1D.Dense> curr)
    {
        int position = run.Frames.Count;
        var f = new Frame();
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

        if (skipped)
        {
            ReleaseFrame(f);
            run.Frames.RemoveAt(run.Frames.Count - 1);
            run.CountSinceKey--;   // never the first frame
            return;
        }
        _copyCanvas(run.Width * run.Height, curr, _prev.View);   // the previous canvas is the input frame, unmodified
        run.IsFirst = false;
    }

    /// <summary>
    /// GetSubRects: the lossless change rectangle (seed: the canvas), then (lossy) the lossy one seeded with it. A key frame
    /// other than the first keeps the whole canvas.
    /// </summary>
    private void GetSubRects(Run run, ArrayView1D<byte, Stride1D.Dense> prev, ArrayView1D<byte, Stride1D.Dense> curr,
        bool isKey, bool isFirst, int slotLl, int slotLossy, int emptyAllowed)
    {
        var meta = _meta.View;
        _setRect(1, meta, slotLl, 0, 0, run.Width, run.Height);
        if (!isKey || isFirst)
        {
            Minimize(run, prev, curr, slotLl, -1, emptyAllowed);
            if (!run.Lossless)
            {
                _copyRect(1, meta, slotLl, slotLossy);
                Minimize(run, prev, curr, slotLossy, run.MaxDiff, emptyAllowed);
            }
        }
        else if (!run.Lossless)
        {
            _copyRect(1, meta, slotLl, slotLossy);   // the whole canvas: not empty, offsets already even
        }
    }

    private void Minimize(Run run, ArrayView1D<byte, Stride1D.Dense> prev, ArrayView1D<byte, Stride1D.Dense> curr, int slot, int maxDiff, int emptyAllowed)
    {
        var meta = _meta.View;
        _columns(run.Width, prev, curr, meta, _colFlags.View, run.Width, slot, maxDiff);
        _columnsReduce(1, meta, _colFlags.View, slot);
        _rows(run.Height, prev, curr, meta, _rowFlags.View, run.Width, slot, maxDiff);
        _rowsReduce(1, meta, _rowFlags.View, slot, emptyAllowed);
    }

    /// <summary>libwebp SetFrame for a single-mode encoder: one candidate. Returns true when the frame is skipped.</summary>
    private async Task<bool> SetFrameAsync(Run run, ArrayView1D<byte, Stride1D.Dense> curr, bool isKey, Frame f)
    {
        var meta = _meta.View;
        bool isFirst = run.IsFirst;
        int emptyAllowedNone = isFirst ? 0 : 1;
        bool disposeBgPossible = !isKey && !run.PrevUndecided;
        var prev = _prev.View;
        var prevDisposed = _prevDisposed.View;
        // the rectangles this mode decides with: lossless (exact) or lossy (within the tolerance)
        int slotNone = run.Lossless ? K.SlotLlNone : K.SlotLossyNone, slotBg = run.Lossless ? K.SlotLlBg : K.SlotLossyBg;
        int blendDiff = run.Lossless ? -1 : run.MaxDiff;

        GetSubRects(run, prev, curr, isKey, isFirst, K.SlotLlNone, K.SlotLossyNone, emptyAllowedNone);
        if (disposeBgPossible)
        {
            _copyCanvas(run.Width * run.Height, prev, prevDisposed);
            var pr = run.PrevRect;
            if (!pr.IsEmpty) _clearRect(pr.W * pr.H, prevDisposed, run.Width, pr.X, pr.Y, pr.W);
            GetSubRects(run, prevDisposed, curr, isKey, isFirst, K.SlotLlBg, K.SlotLossyBg, 0);
        }
        if (!isKey)
        {
            // IsLosslessBlendingPossible / IsLossyBlendingPossible for each candidate rectangle (key frames never blend)
            _blendRows(run.Height, prev, curr, meta, _rowFlags.View, run.Width, slotNone, blendDiff);
            _any(1, meta, _rowFlags.View, 0, run.Height, K.FlagBlendImpossible);
            if (disposeBgPossible)
            {
                _blendRows(run.Height, prevDisposed, curr, meta, _rowFlags.View, run.Width, slotBg, blendDiff);
                _any(1, meta, _rowFlags.View, 0, run.Height, K.FlagBlendImpossible + 1);
            }
        }
        await _acc.SynchronizeAsync();
        var m = await meta.CopyToHostAsync();

        var none = new Rect(m[slotNone], m[slotNone + 1], m[slotNone + 2], m[slotNone + 3]);
        if (none.IsEmpty) return true;   // nothing changed: the previous frame shows longer
        bool tryBg = false;
        var bg = default(Rect);
        if (disposeBgPossible)
        {
            bg = new Rect(m[slotBg], m[slotBg + 1], m[slotBg + 2], m[slotBg + 3]);
            tryBg = bg.Area < none.Area;
        }

        Variant v;
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

    private static void SetPreviousDisposeMethod(Run run, int dispose)
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
    /// EncodeCandidate: the rectangle of the current canvas as a still picture (blended: IncreaseTransparency for lossless,
    /// FlattenSimilarBlocks for lossy, against the reference canvas), encoded into a pooled file buffer.
    /// </summary>
    private async Task<Variant> EncodeCandidateAsync(Run run, ArrayView1D<byte, Stride1D.Dense> curr,
        ArrayView1D<byte, Stride1D.Dense> reference, Rect r, bool blend)
    {
        var picture = await _picture.EnsureDrainedAsync((long)r.W * r.H * 4);
        _extract(r.W * r.H, curr, picture, run.Width, r.X, r.Y, r.W);
        if (blend && run.Lossless)
        {
            _increaseTransparency(r.W * r.H, reference, picture, run.Width, r.X, r.Y, r.W);
        }
        else if (blend)
        {
            // FlattenSimilarBlocks walks 8x8 blocks of the canvas grid: x from (x + 8) & ~7 to (x + w) & ~7, same for y
            int xStart = (r.X + 8) & ~7, xEnd = (r.X + r.W) & ~7;
            int yStart = (r.Y + 8) & ~7, yEnd = (r.Y + r.H) & ~7;
            if (xEnd > xStart && yEnd > yStart)
            {
                int bx = (xEnd - xStart) / 8, by = (yEnd - yStart) / 8;
                _flatten(bx * by, reference, picture, run.Width, r.X, r.Y, r.W, xStart, yStart, bx, run.MaxDiff);
            }
        }
        var file = RentFile();
        var e = await run.Encode(picture, r.W, r.H, blend, file);
        return new Variant { File = file, Length = e.Length, Skip = e.ImageChunksStart, HasAlpha = e.HasAlpha, Rect = r, Blend = blend };
    }

    private void ReleaseVariant(Variant? v)
    {
        if (v?.File != null) _freeFiles.Push(v.File);
        if (v != null) v.File = null;
    }

    private void ReleaseFrame(Frame f)
    {
        ReleaseVariant(f.Sub);
        ReleaseVariant(f.Key);
    }

    private ArrayView1D<byte, Stride1D.Dense> FileView(Variant v) =>
        v.File?.View ?? (v.Lossless1x1 ? _lossless1x1.View : _lossy1x1.View);

    /// <summary>libwebp FlushFrames: frames no later frame can change are written as ANMF chunks.</summary>
    private async Task FlushFramesAsync(Run run)
    {
        while (run.FlushCount > 0)
        {
            var f = run.Frames[0];
            var v = f.Chosen;
            int payload = v.Length - v.Skip;
            long end = run.Cursor + 24 + payload;
            var output = await run.Sink.ReserveAsync(end);
            if (output is { } o)
            {
                int flags = (v.Dispose == 1 ? 1 : 0) | (v.Blend ? 0 : 2);
                _frame(24 + payload, o, FileView(v), (int)run.Cursor, v.Skip, payload, v.Rect.X, v.Rect.Y, v.Rect.W, v.Rect.H, f.Duration, flags);
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

    /// <summary>WebPAnimEncoderAssemble: flush the rest, then the container; a lone frame becomes a still.</summary>
    private async Task<int> AssembleAsync(Run run)
    {
        run.FlushCount = run.Frames.Count;
        await FlushFramesAsync(run);

        long animatedLength = run.Cursor + ChunkLength(run.Exif) + ChunkLength(run.Xmp);
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
            long stillLength = StillFileLength(full.Length, full.HasAlpha, full.Skip, run.Icc, run.Exif, run.Xmp);
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

        if (animatedLength > int.MaxValue) throw new InvalidOperationException("The animation is too large to encode (over 2 GB).");
        var output = await run.Sink.ReserveAsync(animatedLength);
        if (output is { } o)
        {
            int flags = 0x02 | (run.AnyAlpha ? 0x10 : 0) | (run.Icc != null ? 0x20 : 0) | (run.Exif != null ? 0x08 : 0) | (run.Xmp != null ? 0x04 : 0);
            _fileHeader(30, o, (int)animatedLength - 8, flags, run.Width, run.Height);
            int p = 30, blob = 0;
            if (run.Icc != null)
            {
                _blobChunk(ChunkLength(run.Icc), o, _blobs.View, p, FourccIccp, blob, run.Icc.Length);
                p += ChunkLength(run.Icc);
                blob += run.Icc.Length;
            }
            _animChunk(14, o, p, unchecked((int)run.Background), run.LoopCount);
            WriteMetadataTail(o, (int)run.Cursor, blob, run.Exif, run.Xmp);
            await _acc.SynchronizeAsync();
        }
        return (int)animatedLength;
    }

    private async Task<int> WriteStillAsync(Run run, Variant v)
    {
        long length = StillFileLength(v.Length, v.HasAlpha, v.Skip, run.Icc, run.Exif, run.Xmp);
        var output = await run.Sink.ReserveAsync(length);
        if (output is { } o)
        {
            WriteStillFile(o, FileView(v), v.Length, v.HasAlpha, v.Skip, run.Width, run.Height, run.Icc, run.Exif, run.Xmp);
            await _acc.SynchronizeAsync();
        }
        return (int)length;
    }

    /// <summary>FrameToFullCanvas: decode the frame onto a cleared canvas at its offset and encode the whole canvas.</summary>
    private async Task<Variant> EncodeFrameOnCanvasAsync(Run run, Variant v)
    {
        _decoder ??= new WebPDecoderGpu(_acc);
        var picture = await _picture.EnsureDrainedAsync((long)v.Rect.W * v.Rect.H * 4);
        await _decoder.DecodeAsync(FileView(v), v.Length, picture);
        int pixels = run.Width * run.Height;
        var canvas = await _canvas.EnsureDrainedAsync((long)pixels * 4);
        _clearCanvas(pixels, canvas);
        _place(v.Rect.W * v.Rect.H, picture, canvas, run.Width, v.Rect.X, v.Rect.Y, v.Rect.W);
        var file = RentFile();
        var e = await run.Encode(canvas, run.Width, run.Height, false, file);
        return new Variant { File = file, Length = e.Length, Skip = e.ImageChunksStart, HasAlpha = e.HasAlpha, Rect = new Rect(0, 0, run.Width, run.Height) };
    }

    // ------------------------------------------------------------------ still files with metadata (WebPMuxAssemble, one image)

    private static int StillFlags(bool hasAlpha, byte[]? icc, byte[]? exif, byte[]? xmp) =>
        (hasAlpha ? 0x10 : 0) | (icc != null ? 0x20 : 0) | (exif != null ? 0x08 : 0) | (xmp != null ? 0x04 : 0);

    /// <summary>
    /// Length of the still WebPMuxAssemble writes for one encoded image file (<paramref name="imageChunksStart"/>: where
    /// its image chunks start). Without ICC / EXIF / XMP and alpha flag the image's own file is kept.
    /// </summary>
    public static long StillFileLength(int fileLength, bool hasAlpha, int imageChunksStart, byte[]? icc, byte[]? exif, byte[]? xmp)
    {
        if (StillFlags(hasAlpha, icc, exif, xmp) == 0) return fileLength;
        if (icc == null && exif == null && xmp == null && imageChunksStart == 12) return fileLength;   // lossless: no VP8X just for alpha
        return 30 + ChunkLength(icc) + (fileLength - imageChunksStart) + ChunkLength(exif) + ChunkLength(xmp);
    }

    /// <summary>Upload ICC, EXIF, XMP (in that order) into the blob buffer the chunk kernels read.</summary>
    public async Task UploadBlobsAsync(byte[]? icc, byte[]? exif, byte[]? xmp)
    {
        int total = (icc?.Length ?? 0) + (exif?.Length ?? 0) + (xmp?.Length ?? 0);
        var blobs = await _blobs.EnsureDrainedAsync(Math.Max(1, total));
        int at = 0;
        foreach (var b in new[] { icc, exif, xmp })
        {
            if (b == null) continue;
            DeviceTransfer.Upload(blobs.SubView(at, b.Length), b);
            at += b.Length;
        }
    }

    private void WriteMetadataTail(ArrayView1D<byte, Stride1D.Dense> o, int at, int blob, byte[]? exif, byte[]? xmp)
    {
        if (exif != null)
        {
            _blobChunk(ChunkLength(exif), o, _blobs.View, at, FourccExif, blob, exif.Length);
            at += ChunkLength(exif);
            blob += exif.Length;
        }
        if (xmp != null)
            _blobChunk(ChunkLength(xmp), o, _blobs.View, at, FourccXmp, blob, xmp.Length);
    }

    /// <summary>
    /// The still file of one encoded image (<paramref name="file"/>: a complete RIFF of <paramref name="fileLength"/> bytes)
    /// with the metadata chunks, into <paramref name="o"/>. The blobs must already be uploaded.
    /// </summary>
    public void WriteStillFile(ArrayView1D<byte, Stride1D.Dense> o, ArrayView1D<byte, Stride1D.Dense> file, int fileLength, bool hasAlpha,
        int imageChunksStart, int width, int height, byte[]? icc, byte[]? exif, byte[]? xmp)
    {
        long length = StillFileLength(fileLength, hasAlpha, imageChunksStart, icc, exif, xmp);
        if (length == fileLength)
        {
            _copy(fileLength, o, file, 0, 0);   // the encoder's own file
            return;
        }
        _fileHeader(30, o, (int)length - 8, StillFlags(hasAlpha, icc, exif, xmp), width, height);
        int p = 30, blob = 0;
        if (icc != null)
        {
            _blobChunk(ChunkLength(icc), o, _blobs.View, p, FourccIccp, blob, icc.Length);
            p += ChunkLength(icc);
            blob += icc.Length;
        }
        _copy(fileLength - imageChunksStart, o, file, p, imageChunksStart);
        WriteMetadataTail(o, p + fileLength - imageChunksStart, blob, exif, xmp);
    }
}
