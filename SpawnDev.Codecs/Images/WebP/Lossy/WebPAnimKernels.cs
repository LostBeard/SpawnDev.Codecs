// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Accelerator side of the animated lossy WebP encoder (WebPLossyEncoderGpu.Animation.cs): libwebp 1.6.0 anim_encode.c's
// pixel work, as kernels. The canvases are tightly packed RGBA8 (libwebp keeps ARGB words; every test below is per channel
// or on the whole pixel, so the byte order does not matter, and libwebp's TRANSPARENT_COLOR 0x00000000 is four zero bytes).
//
//   MinimizeChangeRectangle   ColumnsKernel (one thread per canvas column: does the column differ inside the rectangle?)
//                             -> ColumnsReduceKernel (one thread: first / last differing column, as libwebp's left and right
//                             scans) -> RowsKernel (one thread per row, over the narrowed columns) -> RowsReduceKernel (top /
//                             bottom, then GetSubRect's empty-rectangle rule and SnapToEvenOffsets). Same result as the
//                             serial scans: a column is "redundant" exactly when no pixel of it differs.
//   IsLossyBlendingPossible   BlendRowsKernel (one thread per row) -> AnyKernel (one thread).
//   sub-frame picture         ExtractKernel (one thread per pixel: the rectangle of the canvas, tightly packed for the
//                             still encoder) then FlattenKernel (one thread per 8x8 block of FlattenSimilarBlocks, with the
//                             block grid of the CANVAS, written into the extracted picture).
//   DisposeFrameRectangle     ClearRectKernel (one thread per pixel of the rectangle).
//   OptimizeSingleFrame       PlaceKernel (a decoded frame onto a cleared canvas).
//   container                 FileHeaderKernel (RIFF + VP8X), AnimChunkKernel (ANIM), BlobChunkKernel (ICCP / EXIF / XMP),
//                             FrameKernel (ANMF header + the frame's chunks copied out of its encoded file), StillKernel
//                             (the chunks of a still file after a new RIFF / VP8X header).
//
// Rectangles live in an int meta buffer, 4 ints per slot (x, y, width, height); the host reads them back (a few scalars per
// frame) to make libwebp's decisions, which are orchestration, not pixel work.

using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.Codecs.Images.WebP.Lossy;

internal static class WebPAnimKernels
{
    /// <summary>Ints per rectangle slot: x, y, width, height.</summary>
    public const int RectInts = 4;

    // meta slots (int offsets)
    public const int SlotLlNone = 0;          // lossless change rectangle, previous frame disposed NONE
    public const int SlotLossyNone = 4;       // lossy change rectangle, NONE
    public const int SlotLlBg = 8;            // lossless, previous frame disposed to BACKGROUND
    public const int SlotLossyBg = 12;        // lossy, BACKGROUND
    public const int FlagBlendImpossible = 16; // IsLossyBlendingPossible == false
    public const int MetaSize = 20;

    /// <summary>libwebp QualityToMaxDiff: the per-channel tolerance of the lossy pixel comparisons.</summary>
    public static int QualityToMaxDiff(float quality)
    {
        double val = Math.Pow(quality / 100.0, 0.5);
        double maxDiff = 31 * (1 - val) + 1 * val;
        return (int)(maxDiff + 0.5);
    }

    private static int Abs(int v) => v < 0 ? -v : v;

    /// <summary>
    /// libwebp's pixel test: lossless (<paramref name="maxDiff"/> &lt; 0) = the four bytes equal; lossy = PixelsAreSimilar
    /// (same alpha, each colour channel off by at most maxDiff scaled by dst's alpha).
    /// </summary>
    private static bool Same(ArrayView<byte> src, ArrayView<byte> dst, int p, int maxDiff)
    {
        int sr = src[p], sg = src[p + 1], sb = src[p + 2], sa = src[p + 3];
        int dr = dst[p], dg = dst[p + 1], db = dst[p + 2], da = dst[p + 3];
        bool lossless = maxDiff < 0;
        bool exact = (sr == dr) & (sg == dg) & (sb == db) & (sa == da);
        int lim = maxDiff * 255;
        bool similar = (sa == da) & (Abs(sr - dr) * da <= lim) & (Abs(sg - dg) * da <= lim) & (Abs(sb - db) * da <= lim);
        return lossless ? exact : similar;
    }

    /// <summary>meta[dst..dst+4) = (x, y, w, h).</summary>
    public static void SetRectKernel(Index1D _, ArrayView<int> meta, int dst, int x, int y, int w, int h)
    {
        meta[dst] = x; meta[dst + 1] = y; meta[dst + 2] = w; meta[dst + 3] = h;
    }

    /// <summary>meta[dst..] = meta[src..] (the lossy rectangle is seeded with the lossless one).</summary>
    public static void CopyRectKernel(Index1D _, ArrayView<int> meta, int src, int dst)
    {
        int x = meta[src], y = meta[src + 1], w = meta[src + 2], h = meta[src + 3];
        meta[dst] = x; meta[dst + 1] = y; meta[dst + 2] = w; meta[dst + 3] = h;
    }

    /// <summary>flags[column] = 1 when a pixel of that column inside the rectangle differs between prev and curr.</summary>
    public static void ColumnsKernel(Index1D column, ArrayView<byte> prev, ArrayView<byte> curr, ArrayView<int> meta,
        ArrayView<int> flags, int canvasWidth, int slot, int maxDiff)
    {
        int i = column.X;
        int x = meta[slot], y = meta[slot + 1], w = meta[slot + 2], h = meta[slot + 3];
        int differs = 0;
        bool inside = (i >= x) & (i < x + w);
        if (inside)
        {
            for (int j = y; j < y + h; j++)
            {
                int p = (j * canvasWidth + i) * 4;
                if (!Same(prev, curr, p, maxDiff))
                {
                    differs = 1;
                    break;
                }
            }
        }
        flags[i] = differs;
    }

    /// <summary>libwebp's left and right scans: the rectangle shrinks to the first..last differing column (empty: none).</summary>
    public static void ColumnsReduceKernel(Index1D _, ArrayView<int> meta, ArrayView<int> flags, int slot)
    {
        int x = meta[slot], w = meta[slot + 2];
        int left = -1, right = -1;
        for (int i = x; i < x + w; i++)
        {
            if (flags[i] != 0)
            {
                if (left < 0) left = i;
                right = i;
            }
        }
        if (left < 0)
        {
            meta[slot] = 0; meta[slot + 1] = 0; meta[slot + 2] = 0; meta[slot + 3] = 0;
        }
        else
        {
            meta[slot] = left;
            meta[slot + 2] = right - left + 1;
        }
    }

    /// <summary>flags[row] = 1 when a pixel of that row inside the (column-narrowed) rectangle differs.</summary>
    public static void RowsKernel(Index1D row, ArrayView<byte> prev, ArrayView<byte> curr, ArrayView<int> meta,
        ArrayView<int> flags, int canvasWidth, int slot, int maxDiff)
    {
        int j = row.X;
        int x = meta[slot], y = meta[slot + 1], w = meta[slot + 2], h = meta[slot + 3];
        int differs = 0;
        bool inside = (j >= y) & (j < y + h);
        if (inside)
        {
            for (int i = x; i < x + w; i++)
            {
                int p = (j * canvasWidth + i) * 4;
                if (!Same(prev, curr, p, maxDiff))
                {
                    differs = 1;
                    break;
                }
            }
        }
        flags[j] = differs;
    }

    /// <summary>
    /// Top and bottom scans, then GetSubRect: an empty rectangle stays empty when allowed, else becomes 1x1 at (0, 0); a
    /// non-empty one is snapped to even offsets (SnapToEvenOffsets).
    /// </summary>
    public static void RowsReduceKernel(Index1D _, ArrayView<int> meta, ArrayView<int> flags, int slot, int emptyAllowed)
    {
        int x = meta[slot], y = meta[slot + 1], w = meta[slot + 2], h = meta[slot + 3];
        int top = -1, bottom = -1;
        bool hasColumns = w > 0;
        if (hasColumns)
        {
            for (int j = y; j < y + h; j++)
            {
                if (flags[j] != 0)
                {
                    if (top < 0) top = j;
                    bottom = j;
                }
            }
        }
        if (top < 0)
        {
            x = 0; y = 0; w = 0; h = 0;
        }
        else
        {
            y = top;
            h = bottom - top + 1;
        }
        PostRect(meta, slot, x, y, w, h, emptyAllowed);
    }

    /// <summary>GetSubRect's tail for a rectangle that was not minimized (a key frame): the same empty / snap rules.</summary>
    public static void PostRectKernel(Index1D _, ArrayView<int> meta, int slot, int emptyAllowed)
    {
        PostRect(meta, slot, meta[slot], meta[slot + 1], meta[slot + 2], meta[slot + 3], emptyAllowed);
    }

    private static void PostRect(ArrayView<int> meta, int slot, int x, int y, int w, int h, int emptyAllowed)
    {
        bool empty = (w == 0) | (h == 0);
        bool keepEmpty = empty & (emptyAllowed != 0);
        if (empty & !keepEmpty)
        {
            x = 0; y = 0; w = 1; h = 1;
        }
        if (!keepEmpty)
        {
            w += x & 1;
            h += y & 1;
            x &= ~1;
            y &= ~1;
        }
        meta[slot] = x; meta[slot + 1] = y; meta[slot + 2] = w; meta[slot + 3] = h;
    }

    /// <summary>
    /// IsLossyBlendingPossible, one row: flags[row] = 1 when a pixel of the rectangle is not opaque in curr and not similar
    /// to prev (blending could not reproduce it).
    /// </summary>
    public static void BlendRowsKernel(Index1D row, ArrayView<byte> prev, ArrayView<byte> curr, ArrayView<int> meta,
        ArrayView<int> flags, int canvasWidth, int slot, int maxDiff)
    {
        int j = row.X;
        int x = meta[slot], y = meta[slot + 1], w = meta[slot + 2], h = meta[slot + 3];
        int bad = 0;
        bool inside = (j >= y) & (j < y + h);
        if (inside)
        {
            for (int i = x; i < x + w; i++)
            {
                int p = (j * canvasWidth + i) * 4;
                bool opaque = curr[p + 3] == 255;
                if (!opaque)
                {
                    if (!Same(prev, curr, p, maxDiff))
                    {
                        bad = 1;
                        break;
                    }
                }
            }
        }
        flags[j] = bad;
    }

    /// <summary>meta[dst] = 1 when any of flags[start..start+count) is set.</summary>
    public static void AnyKernel(Index1D _, ArrayView<int> meta, ArrayView<int> flags, int start, int count, int dst)
    {
        int any = 0;
        for (int i = start; i < start + count; i++)
        {
            if (flags[i] != 0)
            {
                any = 1;
                break;
            }
        }
        meta[dst] = any;
    }

    /// <summary>The rectangle (x, y, w, h) of the canvas into a tightly packed RGBA8 picture, one thread per pixel.</summary>
    public static void ExtractKernel(Index1D pixel, ArrayView<byte> canvas, ArrayView<byte> picture, int canvasWidth, int x, int y, int w)
    {
        int px = pixel.X % w, py = pixel.X / w;
        int s = ((y + py) * canvasWidth + x + px) * 4;
        int d = pixel.X * 4;
        picture[d] = canvas[s];
        picture[d + 1] = canvas[s + 1];
        picture[d + 2] = canvas[s + 2];
        picture[d + 3] = canvas[s + 3];
    }

    /// <summary>
    /// FlattenSimilarBlocks on the extracted picture of rectangle (x, y, w): block <paramref name="block"/> of the canvas
    /// grid that libwebp walks (8x8 blocks from the first multiple of 8 above x / y to the last one inside the rectangle).
    /// When all 64 pixels are opaque in prev and similar to the picture, the block becomes transparent with the average
    /// colour of prev (libwebp: alpha 0, integer means).
    /// </summary>
    public static void FlattenKernel(Index1D block, ArrayView<byte> prev, ArrayView<byte> picture, int canvasWidth,
        int x, int y, int w, int xStart, int yStart, int blocksPerRow, int maxDiff)
    {
        int bi = (block.X % blocksPerRow) * 8 + xStart;
        int bj = (block.X / blocksPerRow) * 8 + yStart;
        int cnt = 0, avgR = 0, avgG = 0, avgB = 0;
        for (int yy = 0; yy < 8; yy++)
        {
            for (int xx = 0; xx < 8; xx++)
            {
                int s = ((bj + yy) * canvasWidth + bi + xx) * 4;
                int d = ((bj + yy - y) * w + bi + xx - x) * 4;
                int sr = prev[s], sg = prev[s + 1], sb = prev[s + 2], sa = prev[s + 3];
                int dr = picture[d], dg = picture[d + 1], db = picture[d + 2], da = picture[d + 3];
                int lim = maxDiff * 255;
                bool similar = (sa == da) & (Abs(sr - dr) * da <= lim) & (Abs(sg - dg) * da <= lim) & (Abs(sb - db) * da <= lim);
                if ((sa == 255) & similar)
                {
                    cnt++;
                    avgR += sr; avgG += sg; avgB += sb;
                }
            }
        }
        if (cnt == 64)
        {
            byte r = (byte)(avgR / 64), g = (byte)(avgG / 64), b = (byte)(avgB / 64);
            for (int yy = 0; yy < 8; yy++)
            {
                for (int xx = 0; xx < 8; xx++)
                {
                    int d = ((bj + yy - y) * w + bi + xx - x) * 4;
                    picture[d] = r; picture[d + 1] = g; picture[d + 2] = b; picture[d + 3] = 0;
                }
            }
        }
    }

    /// <summary>WebPCopyPixels of a whole canvas, one thread per pixel.</summary>
    public static void CopyCanvasKernel(Index1D pixel, ArrayView<byte> src, ArrayView<byte> dst)
    {
        int p = pixel.X * 4;
        dst[p] = src[p]; dst[p + 1] = src[p + 1]; dst[p + 2] = src[p + 2]; dst[p + 3] = src[p + 3];
    }

    /// <summary>A whole canvas of transparent black (TRANSPARENT_COLOR), one thread per pixel.</summary>
    public static void ClearCanvasKernel(Index1D pixel, ArrayView<byte> canvas)
    {
        int p = pixel.X * 4;
        canvas[p] = 0; canvas[p + 1] = 0; canvas[p + 2] = 0; canvas[p + 3] = 0;
    }

    /// <summary>DisposeFrameRectangle(BACKGROUND): the rectangle (x, y, w) of the canvas becomes transparent black.</summary>
    public static void ClearRectKernel(Index1D pixel, ArrayView<byte> canvas, int canvasWidth, int x, int y, int w)
    {
        int px = pixel.X % w, py = pixel.X / w;
        int d = ((y + py) * canvasWidth + x + px) * 4;
        canvas[d] = 0; canvas[d + 1] = 0; canvas[d + 2] = 0; canvas[d + 3] = 0;
    }

    /// <summary>A decoded w-wide picture onto the canvas at (x, y) (DecodeFrameOntoCanvas after the canvas is cleared).</summary>
    public static void PlaceKernel(Index1D pixel, ArrayView<byte> picture, ArrayView<byte> canvas, int canvasWidth, int x, int y, int w)
    {
        int px = pixel.X % w, py = pixel.X / w;
        int d = ((y + py) * canvasWidth + x + px) * 4;
        int s = pixel.X * 4;
        canvas[d] = picture[s]; canvas[d + 1] = picture[s + 1]; canvas[d + 2] = picture[s + 2]; canvas[d + 3] = picture[s + 3];
    }

    private static void PutLe24(ArrayView<byte> o, int at, int i, int v)
    {
        // byte i of a 24-bit little-endian field that starts at 'at'
        o[at + i] = (byte)((v >> (8 * i)) & 0xFF);
    }

    private static byte Le32Byte(int v, int i) => (byte)((v >> (8 * i)) & 0xFF);

    /// <summary>Byte i of a four-character code packed little-endian ('R' | 'I' << 8 | ...).</summary>
    private static byte FccByte(int fcc, int i) => (byte)((fcc >> (8 * i)) & 0xFF);

    /// <summary>
    /// RIFF header (12) + VP8X chunk (18), one thread per byte (30 threads): RIFF size, VP8X flags, canvas size - 1.
    /// </summary>
    public static void FileHeaderKernel(Index1D index, ArrayView<byte> o, int riffSize, int flags, int canvasWidth, int canvasHeight)
    {
        int i = index.X;
        byte v = 0;
        if (i < 4) v = FccByte(0x46464952, i);                 // "RIFF"
        else if (i < 8) v = Le32Byte(riffSize, i - 4);
        else if (i < 12) v = FccByte(0x50424557, i - 8);       // "WEBP"
        else if (i < 16) v = FccByte(0x58385056, i - 12);      // "VP8X"
        else if (i < 20) v = Le32Byte(10, i - 16);
        else if (i == 20) v = (byte)flags;
        else if (i < 24) v = 0;
        else if (i < 27) v = Le32Byte(canvasWidth - 1, i - 24);
        else v = Le32Byte(canvasHeight - 1, i - 27);
        o[i] = v;
    }

    /// <summary>RIFF header only (12 bytes, one thread per byte): a simple-format still file.</summary>
    public static void RiffHeaderKernel(Index1D index, ArrayView<byte> o, int riffSize)
    {
        int i = index.X;
        byte v;
        if (i < 4) v = FccByte(0x46464952, i);
        else if (i < 8) v = Le32Byte(riffSize, i - 4);
        else v = FccByte(0x50424557, i - 8);
        o[i] = v;
    }

    /// <summary>ANIM chunk (14 bytes) at <paramref name="at"/>: background colour (LE 32) and loop count (LE 16).</summary>
    public static void AnimChunkKernel(Index1D index, ArrayView<byte> o, int at, int background, int loopCount)
    {
        int i = index.X;
        byte v;
        if (i < 4) v = FccByte(0x4D494E41, i);                 // "ANIM"
        else if (i < 8) v = Le32Byte(6, i - 4);
        else if (i < 12) v = Le32Byte(background, i - 8);
        else v = Le32Byte(loopCount, i - 12);
        o[at + i] = v;
    }

    /// <summary>
    /// A metadata chunk at <paramref name="at"/>: header (fourcc, size), the payload from blobs[blobOffset..], a pad byte
    /// when the size is odd. One thread per byte (8 + size + (size &amp; 1)).
    /// </summary>
    public static void BlobChunkKernel(Index1D index, ArrayView<byte> o, ArrayView<byte> blobs, int at, int fcc, int blobOffset, int size)
    {
        int i = index.X;
        byte v;
        if (i < 4) v = FccByte(fcc, i);
        else if (i < 8) v = Le32Byte(size, i - 4);
        else if (i < 8 + size) v = blobs[blobOffset + i - 8];
        else v = 0;
        o[at + i] = v;
    }

    /// <summary>
    /// One ANMF chunk at <paramref name="at"/>: header (8) + frame parameters (16: offset / 2, size - 1, duration, flags) +
    /// the frame's chunks, copied from its encoded file starting at <paramref name="skip"/> (after the RIFF and, for a file
    /// with alpha, the VP8X header). One thread per byte (24 + payload).
    /// </summary>
    public static void FrameKernel(Index1D index, ArrayView<byte> o, ArrayView<byte> file, int at, int skip, int payload,
        int x, int y, int w, int h, int duration, int flags)
    {
        int i = index.X;
        byte v;
        if (i < 4) v = FccByte(0x464D4E41, i);                 // "ANMF"
        else if (i < 8) v = Le32Byte(16 + payload, i - 4);
        else if (i < 11) v = Le32Byte(x >> 1, i - 8);
        else if (i < 14) v = Le32Byte(y >> 1, i - 11);
        else if (i < 17) v = Le32Byte(w - 1, i - 14);
        else if (i < 20) v = Le32Byte(h - 1, i - 17);
        else if (i < 23) v = Le32Byte(duration, i - 20);
        else if (i == 23) v = (byte)flags;
        else v = file[skip + i - 24];
        o[at + i] = v;
    }

    /// <summary>Copy file[skip .. skip + count) to o[at ..] (the image chunks of a still file), one thread per byte.</summary>
    public static void CopyKernel(Index1D index, ArrayView<byte> o, ArrayView<byte> file, int at, int skip)
    {
        o[at + index.X] = file[skip + index.X];
    }
}
