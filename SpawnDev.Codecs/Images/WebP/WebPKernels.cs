// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// The WebP decode kernels. Every stage runs on the accelerator; the host only dispatches and reads back
// small metadata blocks (the meta block, the frame table):
//
//   1. ScanKernel       (1 thread)  RIFF walk: VP8X flags, canvas, ICCP / EXIF / XMP locations, EXIF orientation,
//                                   ANIM, the ANMF frame table, the image chunk, and the VP8L header facts.
//   2. DecodeKernel     (1 thread)  VP8L bitstream -> ARGB main image + transform list (Vp8lDecodeGpu).
//   3. Inverse transforms           subtract green (parallel), cross colour (parallel), colour indexing with pixel
//                                   unbundling (parallel), predictor (sequential in raster order, see below).
//   4. ToRgbaKernel     (1 thread per pixel)  ARGB -> RGBA8 bytes.
//   4b. ALPH            AlphaGreen / AlphaCopy / AlphaUnfilter / AlphaPack: the alpha plane of a lossy image (see DecodeAlphaAsync).
//   5. Animation        ClearKernel / CompositeKernel (1 thread per pixel) dispose + blend a frame onto the canvas,
//                                   bit-exact with libwebp's WebPAnimDecoder.
//
// The predictor inverse is a dependency chain in raster order (each pixel reads its left, top, top-left and
// top-right neighbours AFTER they were reconstructed), so it is one GPU thread - still accelerator-resident.
// Pixels on one anti-diagonal x + 2y = const are independent, so it can become a wavefront later.

using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.Codecs.Images.WebP;

/// <summary>Loads and launches the WebP decode kernels.</summary>
public sealed class WebPKernels : IDisposable
{
    private const uint FccVp8 = 'V' | ('P' << 8) | ('8' << 16) | (' ' << 24);
    private const uint FccVp8L = 'V' | ('P' << 8) | ('8' << 16) | ('L' << 24);
    private const uint FccVp8X = 'V' | ('P' << 8) | ('8' << 16) | ('X' << 24);
    private const uint FccAlph = 'A' | ('L' << 8) | ('P' << 16) | ('H' << 24);
    private const uint FccAnim = 'A' | ('N' << 8) | ('I' << 16) | ('M' << 24);
    private const uint FccAnmf = 'A' | ('N' << 8) | ('M' << 16) | ('F' << 24);
    private const uint FccIccp = 'I' | ('C' << 8) | ('C' << 16) | ('P' << 24);
    private const uint FccExif = 'E' | ('X' << 8) | ('I' << 16) | ('F' << 24);
    private const uint FccXmp = 'X' | ('M' << 8) | ('P' << 16) | (' ' << 24);

    private readonly Action<Index1D, ArrayView<byte>, int, ArrayView<int>, ArrayView<int>, int> _scan;
    private readonly Action<Index1D, ArrayView<byte>, int, int, int, int, int,
        ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<uint>, ArrayView<int>> _decode;
    private readonly Action<Index1D, ArrayView<uint>> _subtractGreen;
    private readonly Action<Index1D, ArrayView<uint>, ArrayView<int>, int, int, int> _crossColor;
    private readonly Action<Index1D, ArrayView<uint>, ArrayView<int>, int, int, int, int> _predictor;
    private readonly Action<Index1D, ArrayView<uint>, ArrayView<uint>, ArrayView<int>, int, int, int, int, int> _colorIndex;
    private readonly Action<Index1D, ArrayView<uint>, ArrayView<byte>> _toRgba;
    private readonly Action<Index1D, ArrayView<uint>> _alphaGreen;
    private readonly Action<Index1D, ArrayView<byte>, int, ArrayView<uint>> _alphaCopy;
    private readonly Action<Index1D, ArrayView<uint>, int, int, int, int> _alphaUnfilter;
    private readonly Action<Index1D, ArrayView<uint>, ArrayView<byte>> _alphaPack;
    private readonly Action<Index1D, ArrayView<byte>, int, int, int, int, int> _clearRect;
    private readonly Action<Index1D, ArrayView<byte>> _clearAll;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, int, int, int, int, int, int, int, int, int, int, int> _composite;

    /// <summary>Compile the kernels.</summary>
    public WebPKernels(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _scan = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, int, ArrayView<int>, ArrayView<int>, int>(ScanKernel);
        _decode = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, int, int, int, int, int,
            ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<uint>, ArrayView<int>>(DecodeKernel);
        _subtractGreen = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<uint>>(SubtractGreenKernel);
        _crossColor = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<uint>, ArrayView<int>, int, int, int>(CrossColorKernel);
        _predictor = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<uint>, ArrayView<int>, int, int, int, int>(PredictorKernel);
        _colorIndex = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<uint>, ArrayView<uint>, ArrayView<int>, int, int, int, int, int>(ColorIndexKernel);
        _toRgba = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<uint>, ArrayView<byte>>(ToRgbaKernel);
        _alphaGreen = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<uint>>(AlphaGreenKernel);
        _alphaCopy = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, int, ArrayView<uint>>(AlphaCopyKernel);
        _alphaUnfilter = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<uint>, int, int, int, int>(AlphaUnfilterKernel);
        _alphaPack = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<uint>, ArrayView<byte>>(AlphaPackKernel);
        _clearRect = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, int, int, int, int, int>(ClearRectKernel);
        _clearAll = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>>(ClearAllKernel);
        _composite = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<byte>, int, int, int, int, int, int, int, int, int, int, int>(CompositeKernel);
    }

    // ------------------------------------------------------------------
    // Launchers
    // ------------------------------------------------------------------

    /// <summary>RIFF scan: fills <paramref name="meta"/> and the frame table (<see cref="WebPMeta.FrameStride"/> ints per frame).</summary>
    public void Scan(ArrayView<byte> file, int fileLength, ArrayView<int> meta, ArrayView<int> frames, int frameCapacity)
    {
        _scan(1, file, fileLength, meta, frames, frameCapacity);
    }

    /// <summary>Decode the VP8L stream at <c>file[payloadOffset ..]</c> into <paramref name="pix"/>.</summary>
    public void Decode(
        ArrayView<byte> file, int payloadOffset, int payloadLength, int width, int height, int poolCapacity,
        ArrayView<int> meta, ArrayView<int> work, ArrayView<int> pool, ArrayView<uint> pix, ArrayView<int> tables)
    {
        _decode(1, file, payloadOffset, payloadLength, width, height, poolCapacity, meta, work, pool, pix, tables);
    }

    /// <summary>Inverse subtract-green over <paramref name="count"/> pixels, in place.</summary>
    public void SubtractGreen(ArrayView<uint> pix, int count) => _subtractGreen(count, pix);

    /// <summary>Inverse cross-colour transform over <paramref name="count"/> pixels of width <paramref name="width"/>, in place.</summary>
    public void CrossColor(ArrayView<uint> pix, ArrayView<int> work, int dataBase, int bits, int width, int count)
        => _crossColor(count, pix, work, dataBase, bits, width);

    /// <summary>Inverse predictor transform, in place (one thread, raster order).</summary>
    public void Predictor(ArrayView<uint> pix, ArrayView<int> work, int dataBase, int bits, int width, int height)
        => _predictor(1, pix, work, dataBase, bits, width, height);

    /// <summary>Inverse colour indexing: <paramref name="src"/> (packed, width <paramref name="srcWidth"/>) to <paramref name="dst"/> (width <paramref name="width"/>).</summary>
    public void ColorIndex(
        ArrayView<uint> src, ArrayView<uint> dst, ArrayView<int> work, int paletteBase, int paletteCount,
        int bits, int srcWidth, int width, int height)
        => _colorIndex(width * height, src, dst, work, paletteBase, paletteCount, bits, srcWidth, width);

    /// <summary>ARGB pixels to RGBA8 bytes.</summary>
    public void ToRgba(ArrayView<uint> pix, ArrayView<byte> rgba, int count) => _toRgba(count, pix, rgba);

    /// <summary>The alpha plane of a decoded VP8L alpha stream is its green channel: keep only that byte, in place.</summary>
    public void AlphaGreen(ArrayView<uint> pix, int count) => _alphaGreen(count, pix);

    /// <summary>Uncompressed ALPH data: <paramref name="count"/> bytes from <c>file[offset ..]</c> into the plane.</summary>
    public void AlphaCopy(ArrayView<byte> file, int offset, ArrayView<uint> plane, int count) => _alphaCopy(count, file, offset, plane);

    /// <summary>
    /// Undo the ALPH prediction filter (1 horizontal, 2 vertical, 3 gradient; 0 = nothing) of a <paramref name="width"/> x
    /// <paramref name="height"/> plane held one sample per uint, in place. Two dispatches for the separable filters (a short
    /// serial edge pass, then every row / column in parallel); the gradient filter depends on its three causal neighbours in
    /// both directions, so it is one thread in raster order (an anti-diagonal wavefront can replace it later).
    /// </summary>
    public void AlphaUnfilter(ArrayView<uint> plane, int width, int height, int filter)
    {
        if (filter == 0) return;
        _alphaUnfilter(1, plane, width, height, filter, 0);
        if (filter == 1) _alphaUnfilter(height, plane, width, height, filter, 1);
        else if (filter == 2) _alphaUnfilter(width, plane, width, height, filter, 1);
    }

    /// <summary>One sample per uint to bytes.</summary>
    public void AlphaPack(ArrayView<uint> plane, ArrayView<byte> output, int count) => _alphaPack(count, plane, output);

    /// <summary>Clear a rectangle of an RGBA8 canvas to transparent black.</summary>
    public void ClearRect(ArrayView<byte> canvas, int canvasWidth, int x, int y, int w, int h)
    {
        if ((w > 0) & (h > 0)) _clearRect(w * h, canvas, canvasWidth, x, y, w, h);
    }

    /// <summary>Clear a whole RGBA8 canvas of <paramref name="pixels"/> pixels to transparent black.</summary>
    public void ClearAll(ArrayView<byte> canvas, int pixels) => _clearAll(pixels, canvas);

    /// <summary>
    /// Draw an RGBA8 frame onto the canvas. <paramref name="blend"/> 1 = alpha-blend like libwebp's WebPAnimDecoder
    /// (opaque source pixels replace, translucent ones blend, except inside the previous frame's rectangle when it was
    /// disposed to background: <paramref name="prevBackground"/> + prev rect), 0 = replace.
    /// </summary>
    public void Composite(
        ArrayView<byte> canvas, ArrayView<byte> frame, int canvasWidth, int x, int y, int w, int h,
        int blend, int prevBackground, int prevX, int prevY, int prevW, int prevH)
        => _composite(w * h, canvas, frame, canvasWidth, x, y, w, h, blend, prevBackground, prevX, prevY, prevW, prevH);

    // ------------------------------------------------------------------
    // 1. RIFF scan
    // ------------------------------------------------------------------

    private static int Le16(ArrayView<byte> f, int p) => f[p] | (f[p + 1] << 8);

    private static int Le24(ArrayView<byte> f, int p) => f[p] | (f[p + 1] << 8) | (f[p + 2] << 16);

    private static uint Le32(ArrayView<byte> f, int p)
        => (uint)f[p] | ((uint)f[p + 1] << 8) | ((uint)f[p + 2] << 16) | ((uint)f[p + 3] << 24);

    /// <summary>Little-endian 16-bit read honouring the TIFF byte order flag (1 = little endian).</summary>
    private static int Tiff16(ArrayView<byte> f, int p, int little)
        => little == 1 ? (f[p] | (f[p + 1] << 8)) : ((f[p] << 8) | f[p + 1]);

    private static int Tiff32(ArrayView<byte> f, int p, int little)
        => little == 1
            ? (f[p] | (f[p + 1] << 8) | (f[p + 2] << 16) | (f[p + 3] << 24))
            : ((f[p] << 24) | (f[p + 1] << 16) | (f[p + 2] << 8) | f[p + 3]);

    /// <summary>EXIF orientation (tag 0x0112) of a TIFF block at <paramref name="p"/> (optionally behind "Exif\0\0"); 1 when absent or invalid.</summary>
    private static int ExifOrientation(ArrayView<byte> f, int p, int len)
    {
        int result = 1;
        int start = p;
        int length = len;
        if (len >= 6)
        {
            bool hasPrefix = (f[p] == 'E') & (f[p + 1] == 'x') & (f[p + 2] == 'i') & (f[p + 3] == 'f')
                             & (f[p + 4] == 0) & (f[p + 5] == 0);
            if (hasPrefix)
            {
                start = p + 6;
                length = len - 6;
            }
        }
        if (length >= 8)
        {
            int little = 0;
            bool ok = true;
            bool isLittle = (f[start] == 'I') & (f[start + 1] == 'I');
            bool isBig = (f[start] == 'M') & (f[start + 1] == 'M');
            if (isLittle) little = 1;
            else if (!isBig) ok = false;
            if (ok)
            {
                int magic = Tiff16(f, start + 2, little);
                int ifd = Tiff32(f, start + 4, little);
                bool ifdOk = (magic == 42) & (ifd >= 8) & (ifd + 2 <= length);
                if (ifdOk)
                {
                    int count = Tiff16(f, start + ifd, little);
                    int e = 0;
                    int found = 0;
                    bool go = count > 0;
                    while (go)
                    {
                        int ep = ifd + 2 + e * 12;
                        if (ep + 12 > length)
                        {
                            found = 1;
                        }
                        else if (Tiff16(f, start + ep, little) == 0x0112)
                        {
                            found = 1;
                            int type = Tiff16(f, start + ep + 2, little);
                            int cnt = Tiff32(f, start + ep + 4, little);
                            int v = Tiff16(f, start + ep + 8, little);
                            bool valid = (type == 3) & (cnt == 1) & (v >= 1) & (v <= 8);
                            if (valid) result = v;
                        }
                        e++;
                        go = (e < count) & (found == 0);
                    }
                }
            }
        }
        return result;
    }

    private static void ScanKernel(
        Index1D _, ArrayView<byte> file, int fileLen, ArrayView<int> meta, ArrayView<int> frames, int frameCap)
    {
        for (int i = 0; i < WebPMeta.Size; i++) meta[i] = 0;
        meta[WebPMeta.IccOffset] = -1;
        meta[WebPMeta.ExifOffset] = -1;
        meta[WebPMeta.XmpOffset] = -1;
        meta[WebPMeta.Orientation] = 1;

        int err = 0;
        int end = 0;
        if (fileLen < 12)
        {
            err = (int)WebPError.NotWebP;
        }
        else
        {
            uint magic = Le32(file, 0);
            uint webp = Le32(file, 8);
            bool okSig = (magic == ('R' | ('I' << 8) | ('F' << 16) | ('F' << 24))) & (webp == ('W' | ('E' << 8) | ('B' << 16) | ('P' << 24)));
            if (!okSig)
            {
                err = (int)WebPError.NotWebP;
            }
            else
            {
                uint riff = Le32(file, 4);
                if (riff < 12)
                {
                    err = (int)WebPError.BadChunk;
                }
                else if (riff > (uint)(fileLen - 8))
                {
                    err = (int)WebPError.Truncated;
                }
                else
                {
                    end = (int)riff + 8;
                }
            }
        }
        meta[WebPMeta.FileEnd] = end;

        int pos = 12;
        int chunkIndex = 0;
        int done = 0;
        int hasVp8x = 0;
        int animated = 0;
        int frameCount = 0;
        int stillKind = 0;
        int stillOff = 0;
        int stillLen = 0;
        int stillW = 0;
        int stillH = 0;
        int alphaBit = 0;
        int flags = 0;
        int canvasW = 0;
        int canvasH = 0;

        bool scanning = true;
        while (scanning)
        {
            if (pos + 8 > end)
            {
                done = 1;
            }
            else
            {
                uint fcc = Le32(file, pos);
                uint size = Le32(file, pos + 4);
                int payload = pos + 8;
                if (size > (uint)(end - payload))
                {
                    err = (int)WebPError.Truncated;
                }
                else
                {
                    int len = (int)size;
                    bool badFirst = (chunkIndex == 0) & (fcc != FccVp8X) & (fcc != FccVp8L) & (fcc != FccVp8);
                    if (badFirst)
                    {
                        err = (int)WebPError.BadChunk;
                    }
                    else if (fcc == FccVp8X)
                    {
                        if ((chunkIndex != 0) | (len < 10))
                        {
                            err = (int)WebPError.BadChunk;
                        }
                        else
                        {
                            hasVp8x = 1;
                            flags = file[payload];
                            canvasW = Le24(file, payload + 4) + 1;
                            canvasH = Le24(file, payload + 7) + 1;
                            animated = (flags & 2) != 0 ? 1 : 0;
                            if (canvasW > 0x1FFFFFFF / canvasH) err = (int)WebPError.TooLarge;
                        }
                    }
                    else if (fcc == FccIccp)
                    {
                        // libwebp (demux.c) only keeps a metadata chunk whose VP8X flag is set, and the first one of a kind
                        bool wanted = (meta[WebPMeta.IccOffset] < 0) & ((flags & 0x20) != 0);
                        if (wanted)
                        {
                            meta[WebPMeta.IccOffset] = payload;
                            meta[WebPMeta.IccLength] = len;
                        }
                    }
                    else if (fcc == FccExif)
                    {
                        bool wanted = (meta[WebPMeta.ExifOffset] < 0) & ((flags & 0x08) != 0);
                        if (wanted)
                        {
                            meta[WebPMeta.ExifOffset] = payload;
                            meta[WebPMeta.ExifLength] = len;
                            meta[WebPMeta.Orientation] = ExifOrientation(file, payload, len);
                        }
                    }
                    else if (fcc == FccXmp)
                    {
                        bool wanted = (meta[WebPMeta.XmpOffset] < 0) & ((flags & 0x04) != 0);
                        if (wanted)
                        {
                            meta[WebPMeta.XmpOffset] = payload;
                            meta[WebPMeta.XmpLength] = len;
                        }
                    }
                    else if (fcc == FccAnim)
                    {
                        if (len >= 6)
                        {
                            meta[WebPMeta.BackgroundColor] = (int)Le32(file, payload);
                            meta[WebPMeta.LoopCount] = Le16(file, payload + 4);
                        }
                    }
                    else if (fcc == FccAnmf)
                    {
                        if (animated == 1)
                        {
                            if (len < 16)
                            {
                                err = (int)WebPError.BadChunk;
                            }
                            else if (frameCount >= frameCap)
                            {
                                err = (int)WebPError.TooManyFrames;
                            }
                            else
                            {
                                int fb = frameCount * WebPMeta.FrameStride;
                                int fx = Le24(file, payload) * 2;
                                int fy = Le24(file, payload + 3) * 2;
                                int fw = Le24(file, payload + 6) + 1;
                                int fh = Le24(file, payload + 9) + 1;
                                frames[fb + WebPMeta.FrX] = fx;
                                frames[fb + WebPMeta.FrY] = fy;
                                frames[fb + WebPMeta.FrW] = fw;
                                frames[fb + WebPMeta.FrH] = fh;
                                frames[fb + WebPMeta.FrDuration] = Le24(file, payload + 12);
                                frames[fb + WebPMeta.FrFlags] = file[payload + 15];
                                frames[fb + WebPMeta.FrKind] = 0;
                                frames[fb + WebPMeta.FrHasAlpha] = 0;
                                frames[fb + WebPMeta.FrAlphOffset] = 0;
                                frames[fb + WebPMeta.FrAlphLength] = 0;
                                bool inside = (fx + fw <= canvasW) & (fy + fh <= canvasH);
                                if (!inside) err = (int)WebPError.CanvasMismatch;

                                // sub-chunks of the frame: [ALPH] then VP8 / VP8L
                                int sp = payload + 16;
                                int se = payload + len;
                                int sdone = 0;
                                int sawAlph = 0;
                                bool sscan = err == 0;
                                while (sscan)
                                {
                                    if (sp + 8 > se)
                                    {
                                        sdone = 1;
                                    }
                                    else
                                    {
                                        uint sf = Le32(file, sp);
                                        uint ss = Le32(file, sp + 4);
                                        int sPayload = sp + 8;
                                        if (ss > (uint)(se - sPayload))
                                        {
                                            err = (int)WebPError.Truncated;
                                        }
                                        else
                                        {
                                            if (sf == FccAlph)
                                            {
                                                sawAlph = 1;
                                                frames[fb + WebPMeta.FrAlphOffset] = sPayload;
                                                frames[fb + WebPMeta.FrAlphLength] = (int)ss;
                                            }
                                            else if (sf == FccVp8L)
                                            {
                                                int sl = (int)ss;
                                                frames[fb + WebPMeta.FrKind] = 1;
                                                frames[fb + WebPMeta.FrOffset] = sPayload;
                                                frames[fb + WebPMeta.FrLength] = sl;
                                                sdone = 1;
                                                int sigOk = 0;
                                                if (sl >= 5)
                                                {
                                                    if (file[sPayload] == 0x2F) sigOk = 1;
                                                }
                                                if (sigOk == 0)
                                                {
                                                    err = (int)WebPError.BadHeader;
                                                }
                                                else
                                                {
                                                    uint hdr = Le32(file, sPayload + 1);
                                                    int vw = (int)(hdr & 0x3FFF) + 1;
                                                    int vh = (int)((hdr >> 14) & 0x3FFF) + 1;
                                                    frames[fb + WebPMeta.FrHasAlpha] = (int)((hdr >> 28) & 1);
                                                    if (((hdr >> 29) & 7) != 0) err = (int)WebPError.BadHeader;
                                                    else if ((vw != fw) | (vh != fh)) err = (int)WebPError.CanvasMismatch;
                                                }
                                            }
                                            else if (sf == FccVp8)
                                            {
                                                frames[fb + WebPMeta.FrKind] = sawAlph == 1 ? 3 : 2;
                                                frames[fb + WebPMeta.FrOffset] = sPayload;
                                                frames[fb + WebPMeta.FrLength] = (int)ss;
                                                frames[fb + WebPMeta.FrHasAlpha] = sawAlph;
                                                sdone = 1;
                                            }
                                            sp = sPayload + (int)ss + (int)(ss & 1);
                                        }
                                    }
                                    sscan = (sdone == 0) & (err == 0);
                                }
                                bool noImage = (err == 0) & (frames[fb + WebPMeta.FrKind] == 0);
                                if (noImage)
                                    err = (int)WebPError.NoImageData;
                                frameCount++;
                            }
                        }
                    }
                    else if (fcc == FccAlph)
                    {
                        // only an ALPH chunk BEFORE the image chunk belongs to it; the last one wins (libwebp ParseOptionalChunks)
                        if ((stillKind == 0) & (animated == 0))
                        {
                            meta[WebPMeta.HasAlph] = 1;
                            meta[WebPMeta.AlphOffset] = payload;
                            meta[WebPMeta.AlphLength] = len;
                        }
                    }
                    else if (fcc == FccVp8L)
                    {
                        if ((stillKind == 0) & (animated == 0))
                        {
                            stillKind = 1;
                            stillOff = payload;
                            stillLen = len;
                            int sigOk = 0;
                            if (len >= 5)
                            {
                                if (file[payload] == 0x2F) sigOk = 1;
                            }
                            if (sigOk == 0)
                            {
                                err = (int)WebPError.BadHeader;
                            }
                            else
                            {
                                uint hdr = Le32(file, payload + 1);
                                stillW = (int)(hdr & 0x3FFF) + 1;
                                stillH = (int)((hdr >> 14) & 0x3FFF) + 1;
                                alphaBit = (int)((hdr >> 28) & 1);
                                if (((hdr >> 29) & 7) != 0) err = (int)WebPError.BadHeader;
                            }
                            if (hasVp8x == 0) done = 1;
                        }
                    }
                    else if (fcc == FccVp8)
                    {
                        if ((stillKind == 0) & (animated == 0))
                        {
                            stillKind = meta[WebPMeta.HasAlph] == 1 ? 3 : 2;
                            stillOff = payload;
                            stillLen = len;
                            // VP8 key frame header: 3-byte tag, start code 9d 01 2a, 14-bit width, 14-bit height.
                            int vp8Ok = 0;
                            if (len >= 10)
                            {
                                bool tag = ((file[payload] & 1) == 0) & (file[payload + 3] == 0x9d)
                                           & (file[payload + 4] == 0x01) & (file[payload + 5] == 0x2a);
                                if (tag) vp8Ok = 1;
                            }
                            if (vp8Ok == 0)
                            {
                                err = (int)WebPError.BadHeader;
                            }
                            else
                            {
                                stillW = Le16(file, payload + 6) & 0x3FFF;
                                stillH = Le16(file, payload + 8) & 0x3FFF;
                            }
                            if (hasVp8x == 0) done = 1;
                        }
                    }
                    pos = payload + len + (len & 1);
                    chunkIndex++;
                }
            }
            scanning = (done == 0) & (err == 0);
        }

        meta[WebPMeta.ChunkCount] = chunkIndex;
        meta[WebPMeta.HasVp8x] = hasVp8x;
        meta[WebPMeta.Flags] = flags;
        meta[WebPMeta.IsAnimated] = animated;
        meta[WebPMeta.ImageKind] = stillKind;
        meta[WebPMeta.ImageOffset] = stillOff;
        meta[WebPMeta.ImageLength] = stillLen;
        meta[WebPMeta.AlphaBit] = alphaBit;

        if (err == 0)
        {
            if (animated == 1)
            {
                if (frameCount == 0) err = (int)WebPError.NoImageData;
                meta[WebPMeta.FrameCount] = frameCount;
                meta[WebPMeta.CanvasWidth] = canvasW;
                meta[WebPMeta.CanvasHeight] = canvasH;
                meta[WebPMeta.Width] = canvasW;
                meta[WebPMeta.Height] = canvasH;
                meta[WebPMeta.HasAlpha] = (flags & 0x10) != 0 ? 1 : 0;
            }
            else
            {
                if (stillKind == 0)
                {
                    err = (int)WebPError.NoImageData;
                }
                else if ((hasVp8x == 1) & ((stillW != canvasW) | (stillH != canvasH)))
                {
                    err = (int)WebPError.CanvasMismatch;
                }
                meta[WebPMeta.FrameCount] = 1;
                meta[WebPMeta.CanvasWidth] = stillW;
                meta[WebPMeta.CanvasHeight] = stillH;
                meta[WebPMeta.Width] = stillW;
                meta[WebPMeta.Height] = stillH;
                int hasAlpha = 0;
                if (stillKind == 1) hasAlpha = alphaBit;
                else if (((flags & 0x10) != 0) | (meta[WebPMeta.HasAlph] == 1)) hasAlpha = 1;
                meta[WebPMeta.HasAlpha] = hasAlpha;
            }
        }
        meta[WebPMeta.Error] = err;
    }

    // ------------------------------------------------------------------
    // 2. VP8L decode
    // ------------------------------------------------------------------

    private static void DecodeKernel(
        Index1D _, ArrayView<byte> file, int payloadOffset, int payloadLength, int width, int height, int poolCapacity,
        ArrayView<int> meta, ArrayView<int> work, ArrayView<int> pool, ArrayView<uint> pix, ArrayView<int> tables)
    {
        Vp8lDecodeGpu.Decode(file, payloadOffset, payloadLength, width, height, poolCapacity, meta, work, pool, pix, tables);
    }

    // ------------------------------------------------------------------
    // 3. Inverse transforms
    // ------------------------------------------------------------------

    private static void SubtractGreenKernel(Index1D index, ArrayView<uint> pix)
    {
        uint p = pix[index.X];
        uint green = (p >> 8) & 0xFFu;
        uint redBlue = (p & 0x00ff00ffu) + ((green << 16) | green);
        pix[index.X] = (p & 0xff00ff00u) | (redBlue & 0x00ff00ffu);
    }

    private static int Signed8(int v) => v >= 128 ? v - 256 : v;

    private static void CrossColorKernel(
        Index1D index, ArrayView<uint> pix, ArrayView<int> work, int dataBase, int bits, int width)
    {
        int i = index.X;
        int y = i / width;
        int x = i - y * width;
        int tilesPerRow = (width + (1 << bits) - 1) >> bits;
        uint m = (uint)work[dataBase + (y >> bits) * tilesPerRow + (x >> bits)];
        int greenToRed = Signed8((int)(m & 0xFFu));
        int greenToBlue = Signed8((int)((m >> 8) & 0xFFu));
        int redToBlue = Signed8((int)((m >> 16) & 0xFFu));

        uint p = pix[i];
        int green = Signed8((int)((p >> 8) & 0xFFu));
        int red = (int)((p >> 16) & 0xFFu);
        int blue = (int)(p & 0xFFu);
        int newRed = (red + ((greenToRed * green) >> 5)) & 0xFF;
        int newBlue = (blue + ((greenToBlue * green) >> 5) + ((redToBlue * Signed8(newRed)) >> 5)) & 0xFF;
        pix[i] = (p & 0xff00ff00u) | ((uint)newRed << 16) | (uint)newBlue;
    }

    private static uint Average2(uint a, uint b)
    {
        return (((a ^ b) & 0xfefefefeu) >> 1) + (a & b);
    }

    private static uint AddPixels(uint a, uint b)
    {
        uint ag = (a & 0xff00ff00u) + (b & 0xff00ff00u);
        uint rb = (a & 0x00ff00ffu) + (b & 0x00ff00ffu);
        return (ag & 0xff00ff00u) | (rb & 0x00ff00ffu);
    }

    private static int Clip255(int v)
    {
        int r = v;
        if (r < 0) r = 0;
        if (r > 255) r = 255;
        return r;
    }

    private static int Abs(int v) => v < 0 ? -v : v;

    private static uint Select(uint left, uint top, uint topLeft)
    {
        // Manhattan distance of p = L + T - TL to L is the distance of T to TL, and to T the distance of L to TL.
        int pL = 0;
        int pT = 0;
        for (int i = 0; i < 4; i++)
        {
            int s = i * 8;
            int l = (int)((left >> s) & 0xFFu);
            int t = (int)((top >> s) & 0xFFu);
            int tl = (int)((topLeft >> s) & 0xFFu);
            pL += Abs(t - tl);
            pT += Abs(l - tl);
        }
        return pL < pT ? left : top;
    }

    private static uint ClampAddSubtractFull(uint a, uint b, uint c)
    {
        uint r = 0;
        for (int i = 0; i < 4; i++)
        {
            int s = i * 8;
            int v = Clip255((int)((a >> s) & 0xFFu) + (int)((b >> s) & 0xFFu) - (int)((c >> s) & 0xFFu));
            r |= (uint)v << s;
        }
        return r;
    }

    private static uint ClampAddSubtractHalf(uint a, uint b)
    {
        uint r = 0;
        for (int i = 0; i < 4; i++)
        {
            int s = i * 8;
            int av = (int)((a >> s) & 0xFFu);
            int bv = (int)((b >> s) & 0xFFu);
            int d = av - bv;
            // C integer division truncates toward zero.
            int half = d >= 0 ? d >> 1 : -((-d) >> 1);
            r |= (uint)Clip255(av + half) << s;
        }
        return r;
    }

    internal static uint Predict(int mode, uint left, uint top, uint topRight, uint topLeft)
    {
        uint r = 0xff000000u;
        if (mode == 1) r = left;
        else if (mode == 2) r = top;
        else if (mode == 3) r = topRight;
        else if (mode == 4) r = topLeft;
        else if (mode == 5) r = Average2(Average2(left, topRight), top);
        else if (mode == 6) r = Average2(left, topLeft);
        else if (mode == 7) r = Average2(left, top);
        else if (mode == 8) r = Average2(topLeft, top);
        else if (mode == 9) r = Average2(top, topRight);
        else if (mode == 10) r = Average2(Average2(left, topLeft), Average2(top, topRight));
        else if (mode == 11) r = Select(left, top, topLeft);
        else if (mode == 12) r = ClampAddSubtractFull(left, top, topLeft);
        else if (mode == 13) r = ClampAddSubtractHalf(Average2(left, top), topLeft);
        return r;
    }

    private static void PredictorKernel(
        Index1D _, ArrayView<uint> pix, ArrayView<int> work, int dataBase, int bits, int width, int height)
    {
        int tilesPerRow = (width + (1 << bits) - 1) >> bits;
        // First row: the very first pixel predicts from opaque black, the others from their left neighbour.
        pix[0] = AddPixels(pix[0], 0xff000000u);
        for (int x = 1; x < width; x++)
            pix[x] = AddPixels(pix[x], pix[x - 1]);

        for (int y = 1; y < height; y++)
        {
            int row = y * width;
            // Leftmost column predicts from the pixel above.
            pix[row] = AddPixels(pix[row], pix[row - width]);
            for (int x = 1; x < width; x++)
            {
                int mode = (work[dataBase + (y >> bits) * tilesPerRow + (x >> bits)] >> 8) & 0xF;
                uint left = pix[row + x - 1];
                uint top = pix[row + x - width];
                // On the rightmost column x - width + 1 is the first pixel of the CURRENT row, which the spec prescribes.
                uint topRight = pix[row + x - width + 1];
                uint topLeft = pix[row + x - width - 1];
                pix[row + x] = AddPixels(pix[row + x], Predict(mode, left, top, topRight, topLeft));
            }
        }
    }

    private static void ColorIndexKernel(
        Index1D index, ArrayView<uint> src, ArrayView<uint> dst, ArrayView<int> work,
        int paletteBase, int paletteCount, int bits, int srcWidth, int width)
    {
        int i = index.X;
        int y = i / width;
        int x = i - y * width;
        uint packed = src[y * srcWidth + (x >> bits)];
        int idx = (int)((packed >> 8) & 0xFFu);
        if (bits > 0)
        {
            int bitsPerPixel = 8 >> bits;
            idx = (idx >> ((x & ((1 << bits) - 1)) * bitsPerPixel)) & ((1 << bitsPerPixel) - 1);
        }
        uint color = 0;
        if (idx < paletteCount) color = (uint)work[paletteBase + idx];
        dst[i] = color;
    }

    // ------------------------------------------------------------------
    // 4. ARGB -> RGBA
    // ------------------------------------------------------------------

    private static void ToRgbaKernel(Index1D index, ArrayView<uint> pix, ArrayView<byte> rgba)
    {
        uint p = pix[index.X];
        int o = index.X * 4;
        rgba[o + 0] = (byte)((p >> 16) & 0xFFu);
        rgba[o + 1] = (byte)((p >> 8) & 0xFFu);
        rgba[o + 2] = (byte)(p & 0xFFu);
        rgba[o + 3] = (byte)(p >> 24);
    }

    // ------------------------------------------------------------------
    // 4b. ALPH alpha plane (VP8X lossy images): the plane is held one sample per uint while it is unfiltered, so that
    //     no two threads ever write bytes of the same word
    // ------------------------------------------------------------------

    private static void AlphaGreenKernel(Index1D index, ArrayView<uint> pix)
    {
        pix[index.X] = (pix[index.X] >> 8) & 0xFFu;
    }

    private static void AlphaCopyKernel(Index1D index, ArrayView<byte> file, int offset, ArrayView<uint> plane)
    {
        plane[index.X] = file[offset + index.X];
    }

    private static void AlphaPackKernel(Index1D index, ArrayView<uint> plane, ArrayView<byte> output)
    {
        output[index.X] = (byte)plane[index.X];
    }

    /// <summary>
    /// libwebp filters.c: the predictor of (x, y) is the left sample (horizontal), the sample above (vertical) or
    /// clip(left + above - aboveLeft) (gradient); the top row is always predicted from the left, the left column from above,
    /// and (0, 0) from 0. Phase 0 is the serial edge pass (column 0 for horizontal, row 0 for vertical, everything for
    /// gradient), phase 1 runs one thread per row (horizontal) or per column (vertical).
    /// </summary>
    private static void AlphaUnfilterKernel(Index1D index, ArrayView<uint> p, int width, int height, int filter, int phase)
    {
        int t = index.X;
        if (filter == 1)
        {
            if (phase == 0)
            {
                for (int y = 1; y < height; y++)
                    p[y * width] = (p[y * width] + p[(y - 1) * width]) & 0xFFu;
            }
            else
            {
                int row = t * width;
                for (int x = 1; x < width; x++)
                    p[row + x] = (p[row + x] + p[row + x - 1]) & 0xFFu;
            }
        }
        else if (filter == 2)
        {
            if (phase == 0)
            {
                for (int x = 1; x < width; x++)
                    p[x] = (p[x] + p[x - 1]) & 0xFFu;
            }
            else
            {
                for (int y = 1; y < height; y++)
                    p[y * width + t] = (p[y * width + t] + p[(y - 1) * width + t]) & 0xFFu;
            }
        }
        else if (filter == 3)
        {
            for (int x = 1; x < width; x++)
                p[x] = (p[x] + p[x - 1]) & 0xFFu;
            for (int y = 1; y < height; y++)
            {
                int row = y * width;
                p[row] = (p[row] + p[row - width]) & 0xFFu;
                for (int x = 1; x < width; x++)
                {
                    int g = (int)p[row + x - 1] + (int)p[row - width + x] - (int)p[row - width + x - 1];
                    int pred = g < 0 ? 0 : (g > 255 ? 255 : g);
                    p[row + x] = (p[row + x] + (uint)pred) & 0xFFu;
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // 5. Animation compositing
    // ------------------------------------------------------------------

    private static void ClearAllKernel(Index1D index, ArrayView<byte> canvas)
    {
        int o = index.X * 4;
        canvas[o + 0] = 0;
        canvas[o + 1] = 0;
        canvas[o + 2] = 0;
        canvas[o + 3] = 0;
    }

    private static void ClearRectKernel(Index1D index, ArrayView<byte> canvas, int canvasWidth, int rx, int ry, int rw, int rh)
    {
        int i = index.X;
        int y = i / rw;
        int x = i - y * rw;
        int o = ((ry + y) * canvasWidth + rx + x) * 4;
        canvas[o + 0] = 0;
        canvas[o + 1] = 0;
        canvas[o + 2] = 0;
        canvas[o + 3] = 0;
    }

    /// <summary>libwebp anim_decode.c BlendChannelNonPremult.</summary>
    private static int BlendChannel(int src, int srcA, int dst, int dstA, uint scale)
    {
        uint unscaled = (uint)(src * srcA + dst * dstA);
        return (int)((unscaled * scale) >> 24);
    }

    private static void CompositeKernel(
        Index1D index, ArrayView<byte> canvas, ArrayView<byte> frame, int canvasWidth, int fx, int fy, int fw, int fh,
        int blend, int prevBackground, int px, int py, int pw, int ph)
    {
        int i = index.X;
        int y = i / fw;
        int x = i - y * fw;
        int s = i * 4;
        int cx = fx + x;
        int cy = fy + y;
        int o = (cy * canvasWidth + cx) * 4;

        int sr = frame[s + 0];
        int sg = frame[s + 1];
        int sb = frame[s + 2];
        int sa = frame[s + 3];

        int outR = sr;
        int outG = sg;
        int outB = sb;
        int outA = sa;

        bool insidePrev = (prevBackground != 0) & (cx >= px) & (cx < px + pw) & (cy >= py) & (cy < py + ph);
        bool doBlend = (blend != 0) & (sa != 255) & !insidePrev;
        if (doBlend)
        {
            if (sa == 0)
            {
                outR = canvas[o + 0];
                outG = canvas[o + 1];
                outB = canvas[o + 2];
                outA = canvas[o + 3];
            }
            else
            {
                int dr = canvas[o + 0];
                int dg = canvas[o + 1];
                int db = canvas[o + 2];
                int da = canvas[o + 3];
                // dst_factor_a = dst_a * (256 - src_a) / 256, the integer form of dst_a * (255 - src_a) / 255.
                int dstFactorA = (da * (256 - sa)) >> 8;
                int blendA = sa + dstFactorA;
                uint scale = (1u << 24) / (uint)blendA;
                outR = BlendChannel(sr, sa, dr, dstFactorA, scale);
                outG = BlendChannel(sg, sa, dg, dstFactorA, scale);
                outB = BlendChannel(sb, sa, db, dstFactorA, scale);
                outA = blendA;
            }
        }
        canvas[o + 0] = (byte)outR;
        canvas[o + 1] = (byte)outG;
        canvas[o + 2] = (byte)outB;
        canvas[o + 3] = (byte)outA;
    }

    /// <summary>No-op (kernels are owned by the accelerator).</summary>
    public void Dispose() { }
}
