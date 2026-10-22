// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// The BMP kernels. Every stage that touches the file bytes or the pixels runs on the accelerator; the host
// only dispatches and reads back the metadata block.
//
// Decode:
//   1. ScanKernel      (1 thread)            BITMAPFILEHEADER + DIB header (core / info / V2..V5) + channel masks.
//   2. AlphaScanKernel (1 thread per pixel)  16/32-bit images with an alpha channel: is any alpha value non-zero?
//   3. ConvertKernel   (1 thread per pixel)  1/4/8-bit palette, 16/24/32-bit, BITFIELDS masks -> RGBA8, flips
//                                            bottom-up files, skips the row padding.
//      RleKernel       (1 thread)            RLE4 / RLE8 are sequential run streams: one thread decodes the whole
//                                            stream straight to RGBA8 (palette lookup inline).
// Encode:
//   EncodeKernel       (1 thread per output byte)  header, palette and padded bottom-up rows from RGBA8.
//
// Channel widths other than 8 bits are expanded by bit replication (a 5-bit v becomes (v << 3) | (v >> 2)), so
// full-scale input maps to 255.

using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.Codecs.Images.Bmp;

/// <summary>Loads and launches the BMP kernels.</summary>
public sealed class BmpKernels : IDisposable
{
    private readonly Action<Index1D, ArrayView<byte>, int, ArrayView<int>> _scan;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>> _alphaScan;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>> _convert;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>> _rle;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>> _encode;

    /// <summary>Compile the kernels.</summary>
    public BmpKernels(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _scan = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, int, ArrayView<int>>(ScanKernel);
        _alphaScan = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<int>>(AlphaScanKernel);
        _convert = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>>(ConvertKernel);
        _rle = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>>(RleKernel);
        _encode = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>>(EncodeKernel);
    }

    /// <summary>Parse the file and DIB headers into <paramref name="meta"/>.</summary>
    public void Scan(ArrayView<byte> file, int fileLength, ArrayView<int> meta)
    {
        _scan(1, file, fileLength, meta);
    }

    /// <summary>Set <see cref="BmpMeta.AlphaSeen"/> when any of the <paramref name="pixelCount"/> pixels has a non-zero alpha value.</summary>
    public void AlphaScan(ArrayView<byte> file, ArrayView<int> meta, int pixelCount)
    {
        _alphaScan(pixelCount, file, meta);
    }

    /// <summary>Convert an uncompressed image to RGBA8 in <paramref name="canvas"/>.</summary>
    public void Convert(ArrayView<byte> file, ArrayView<byte> canvas, ArrayView<int> meta, int pixelCount)
    {
        _convert(pixelCount, file, canvas, meta);
    }

    /// <summary>Decode an RLE4 / RLE8 stream to RGBA8 in <paramref name="canvas"/> (pre-cleared to 0: uncovered pixels stay transparent).</summary>
    public void Rle(ArrayView<byte> file, ArrayView<byte> canvas, ArrayView<int> meta)
    {
        _rle(1, file, canvas, meta);
    }

    /// <summary>
    /// Write <paramref name="totalBytes"/> bytes of a BMP file. <paramref name="p"/> = mode (0 BGRA32 V4,
    /// 1 BGR24, 2 indexed 8), width, height, file size, pixel data offset, row stride, image size, palette count.
    /// </summary>
    public void Encode(ArrayView<byte> rgba, ArrayView<byte> output, ArrayView<int> palette, ArrayView<int> p, int totalBytes)
    {
        _encode(totalBytes, rgba, output, palette, p);
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    private static int Le16(ArrayView<byte> f, int p) => f[p] | (f[p + 1] << 8);

    private static int Le32(ArrayView<byte> f, int p) => f[p] | (f[p + 1] << 8) | (f[p + 2] << 16) | (f[p + 3] << 24);

    /// <summary>Expand a <paramref name="bits"/>-wide channel value to 8 bits by bit replication.</summary>
    private static int Expand8(uint v, int bits)
    {
        int r;
        if (bits <= 0)
        {
            r = 0;
        }
        else if (bits >= 8)
        {
            r = (int)(v >> (bits - 8));
        }
        else
        {
            // Repeat the pattern until at least 8 bits are filled, then keep the top 8.
            int acc = 0;
            int filled = 0;
            for (int k = 0; k < 8; k++)
            {
                if (filled < 8)
                {
                    acc = (acc << bits) | (int)v;
                    filled += bits;
                }
            }
            r = (acc >> (filled - 8)) & 0xFF;
        }
        return r;
    }

    private static int Channel(uint raw, int mask, int shift, int bits)
    {
        return Expand8((raw & (uint)mask) >> shift, bits);
    }

    /// <summary>File offset of the first byte of output row <paramref name="y"/> (top row = 0).</summary>
    private static int RowStart(ArrayView<int> meta, int y)
    {
        int row = y;
        if (meta[BmpMeta.TopDown] == 0) row = meta[BmpMeta.AbsHeight] - 1 - y;
        return meta[BmpMeta.OffBits] + row * meta[BmpMeta.Stride];
    }

    private static uint ReadRaw(ArrayView<byte> file, int bpp, int rowStart, int x)
    {
        uint v;
        if (bpp == 16) v = (uint)Le16(file, rowStart + x * 2);
        else v = (uint)Le32(file, rowStart + x * 4);
        return v;
    }

    // ------------------------------------------------------------------
    // 1. Header scan
    // ------------------------------------------------------------------

    private static void ScanKernel(Index1D _, ArrayView<byte> file, int fileLen, ArrayView<int> meta)
    {
        for (int i = 0; i < BmpMeta.Size; i++) meta[i] = 0;
        meta[BmpMeta.FileLength] = fileLen;

        int err = 0;
        if (fileLen < 2)
        {
            err = (int)BmpError.TruncatedHeader;
        }
        else if (((file[0] ^ 0x42) | (file[1] ^ 0x4D)) != 0)
        {
            err = (int)BmpError.BadSignature;
        }
        else if (fileLen < 18)
        {
            err = (int)BmpError.TruncatedHeader;
        }

        if (err == 0)
        {
            int hs = Le32(file, 14);
            meta[BmpMeta.HeaderSize] = hs;
            bool known = (hs == 12) | (hs == 40) | (hs == 52) | (hs == 56) | (hs == 108) | (hs == 124);
            if (!known)
            {
                err = (int)BmpError.UnsupportedHeader;
            }
            else if (14 + hs > fileLen)
            {
                err = (int)BmpError.TruncatedHeader;
            }
            else
            {
                meta[BmpMeta.OffBits] = Le32(file, 10);
                int comp = 0;
                if (hs == 12)
                {
                    meta[BmpMeta.Width] = Le16(file, 18);
                    meta[BmpMeta.Height] = Le16(file, 20);
                    meta[BmpMeta.Planes] = Le16(file, 22);
                    meta[BmpMeta.Bpp] = Le16(file, 24);
                }
                else
                {
                    comp = Le32(file, 30);
                    meta[BmpMeta.Width] = Le32(file, 18);
                    meta[BmpMeta.Height] = Le32(file, 22);
                    meta[BmpMeta.Planes] = Le16(file, 26);
                    meta[BmpMeta.Bpp] = Le16(file, 28);
                    meta[BmpMeta.Compression] = comp;
                    meta[BmpMeta.ImageSize] = Le32(file, 34);
                    meta[BmpMeta.ClrUsed] = Le32(file, 46);
                }

                // Channel masks: after a 40-byte header when BITFIELDS / ALPHABITFIELDS, inside V2+ headers.
                int palOff = 14 + hs;
                int maskBytes = 0;
                if (hs == 40)
                {
                    if (comp == 3) maskBytes = 12;
                    else if (comp == 6) maskBytes = 16;
                    palOff += maskBytes;
                }
                else if (hs == 52)
                {
                    maskBytes = 12;
                }
                else if (hs >= 56)
                {
                    maskBytes = 16;
                }
                if (maskBytes > 0)
                {
                    if (54 + maskBytes > fileLen)
                    {
                        err = (int)BmpError.TruncatedHeader;
                    }
                    else
                    {
                        meta[BmpMeta.MaskR] = Le32(file, 54);
                        meta[BmpMeta.MaskG] = Le32(file, 58);
                        meta[BmpMeta.MaskB] = Le32(file, 62);
                        if (maskBytes == 16) meta[BmpMeta.MaskA] = Le32(file, 66);
                        meta[BmpMeta.FileHasMasks] = 1;
                    }
                }
                meta[BmpMeta.PaletteOffset] = palOff;
            }
        }
        meta[BmpMeta.Error] = err;
    }

    // ------------------------------------------------------------------
    // 2. Alpha scan
    // ------------------------------------------------------------------

    private static void AlphaScanKernel(Index1D index, ArrayView<byte> file, ArrayView<int> meta)
    {
        int w = meta[BmpMeta.Width];
        int i = index.X;
        int y = i / w;
        int x = i - y * w;
        uint v = ReadRaw(file, meta[BmpMeta.Bpp], RowStart(meta, y), x);
        if ((v & (uint)meta[BmpMeta.EffMaskA]) != 0u) meta[BmpMeta.AlphaSeen] = 1;
    }

    // ------------------------------------------------------------------
    // 3. Uncompressed pixels -> RGBA8
    // ------------------------------------------------------------------

    private static void ConvertKernel(Index1D index, ArrayView<byte> file, ArrayView<byte> canvas, ArrayView<int> meta)
    {
        int w = meta[BmpMeta.Width];
        int bpp = meta[BmpMeta.Bpp];
        int i = index.X;
        int y = i / w;
        int x = i - y * w;
        int rowStart = RowStart(meta, y);

        int r = 0, g = 0, b = 0, a = 255;
        if (bpp <= 8)
        {
            int bitPos = x * bpp;
            int at = rowStart + (bitPos >> 3);
            int shift = 8 - bpp - (bitPos & 7);
            int idx = (file[at] >> shift) & ((1 << bpp) - 1);
            if (idx < meta[BmpMeta.PaletteCount])
            {
                int e = meta[BmpMeta.PaletteOffset] + idx * meta[BmpMeta.EntrySize];
                b = file[e];
                g = file[e + 1];
                r = file[e + 2];
            }
        }
        else if (bpp == 24)
        {
            int at = rowStart + x * 3;
            b = file[at];
            g = file[at + 1];
            r = file[at + 2];
        }
        else
        {
            uint v = ReadRaw(file, bpp, rowStart, x);
            r = Channel(v, meta[BmpMeta.EffMaskR], meta[BmpMeta.ShiftR], meta[BmpMeta.BitsR]);
            g = Channel(v, meta[BmpMeta.EffMaskG], meta[BmpMeta.ShiftG], meta[BmpMeta.BitsG]);
            b = Channel(v, meta[BmpMeta.EffMaskB], meta[BmpMeta.ShiftB], meta[BmpMeta.BitsB]);
            // An alpha channel whose every value is 0 is treated as opaque (browsers do the same: such files
            // come from writers that store a padding byte there).
            bool alpha = (meta[BmpMeta.HasAlphaChannel] != 0) & (meta[BmpMeta.AlphaSeen] != 0);
            if (alpha) a = Channel(v, meta[BmpMeta.EffMaskA], meta[BmpMeta.ShiftA], meta[BmpMeta.BitsA]);
        }

        int o = i * 4;
        canvas[o] = (byte)r;
        canvas[o + 1] = (byte)g;
        canvas[o + 2] = (byte)b;
        canvas[o + 3] = (byte)a;
    }

    // ------------------------------------------------------------------
    // 3b. RLE4 / RLE8 (sequential)
    // ------------------------------------------------------------------

    private static void PutIndexed(
        ArrayView<byte> file, ArrayView<byte> canvas, ArrayView<int> meta, int w, int h, int x, int y, int idx)
    {
        // RLE bitmaps are bottom-up: file row y is output row h - 1 - y.
        int o = ((h - 1 - y) * w + x) * 4;
        int r = 0, g = 0, b = 0;
        if (idx < meta[BmpMeta.PaletteCount])
        {
            int e = meta[BmpMeta.PaletteOffset] + idx * meta[BmpMeta.EntrySize];
            b = file[e];
            g = file[e + 1];
            r = file[e + 2];
        }
        canvas[o] = (byte)r;
        canvas[o + 1] = (byte)g;
        canvas[o + 2] = (byte)b;
        canvas[o + 3] = 255;
    }

    private static void RleKernel(Index1D _, ArrayView<byte> file, ArrayView<byte> canvas, ArrayView<int> meta)
    {
        int w = meta[BmpMeta.Width];
        int h = meta[BmpMeta.AbsHeight];
        int end = meta[BmpMeta.FileLength];
        bool rle4 = meta[BmpMeta.Compression] == 2;
        int pos = meta[BmpMeta.OffBits];
        int x = 0;
        int y = 0;
        int err = 0;
        bool done = false;
        while (!done && err == 0)
        {
            if (pos >= end)
            {
                // The stream may end without an end-of-bitmap code.
                done = true;
            }
            else if (pos + 2 > end)
            {
                err = (int)BmpError.RleTruncated;
            }
            else
            {
                int n = file[pos];
                int v = file[pos + 1];
                pos += 2;
                if (n > 0)
                {
                    // Encoded run: n pixels of v (RLE8) or n pixels alternating v's two nibbles (RLE4).
                    for (int k = 0; k < n; k++)
                    {
                        int idx = v;
                        if (rle4)
                        {
                            if ((k & 1) == 0) idx = v >> 4;
                            else idx = v & 15;
                        }
                        bool inside = (x < w) & (y < h);
                        if (err == 0)
                        {
                            if (inside)
                            {
                                PutIndexed(file, canvas, meta, w, h, x, y, idx);
                                x++;
                            }
                            else
                            {
                                err = (int)BmpError.RleOutOfBounds;
                            }
                        }
                    }
                }
                else if (v == 0)
                {
                    x = 0;
                    y++;
                }
                else if (v == 1)
                {
                    done = true;
                }
                else if (v == 2)
                {
                    if (pos + 2 > end)
                    {
                        err = (int)BmpError.RleTruncated;
                    }
                    else
                    {
                        x += file[pos];
                        y += file[pos + 1];
                        pos += 2;
                    }
                }
                else
                {
                    // Absolute run: v literal pixels, padded to a 16-bit boundary.
                    int nbytes = v;
                    if (rle4) nbytes = (v + 1) >> 1;
                    if (pos + nbytes > end)
                    {
                        err = (int)BmpError.RleTruncated;
                    }
                    else
                    {
                        for (int k = 0; k < v; k++)
                        {
                            int idx;
                            if (rle4)
                            {
                                int by = file[pos + (k >> 1)];
                                if ((k & 1) == 0) idx = by >> 4;
                                else idx = by & 15;
                            }
                            else
                            {
                                idx = file[pos + k];
                            }
                            bool inside = (x < w) & (y < h);
                            if (err == 0)
                            {
                                if (inside)
                                {
                                    PutIndexed(file, canvas, meta, w, h, x, y, idx);
                                    x++;
                                }
                                else
                                {
                                    err = (int)BmpError.RleOutOfBounds;
                                }
                            }
                        }
                        pos += nbytes + (nbytes & 1);
                    }
                }
            }
        }
        if (err != 0) meta[BmpMeta.Error] = err;
    }

    // ------------------------------------------------------------------
    // Encode
    // ------------------------------------------------------------------

    private static int LeByte(int value, int byteIndex) => (value >> (byteIndex * 8)) & 0xFF;

    /// <summary>Byte <paramref name="o"/> of BITMAPFILEHEADER + BITMAPINFOHEADER (+ the V4 extension for mode 0).</summary>
    private static int HeaderByte(int o, int mode, int w, int h, int fileSize, int dataStart, int imageSize, int clrUsed)
    {
        int dib = 40;
        int bpp = 8;
        if (mode == 0) { dib = 108; bpp = 32; }
        else if (mode == 1) { bpp = 24; }
        int v;
        if (o < 2) v = o == 0 ? 0x42 : 0x4D;
        else if (o < 6) v = LeByte(fileSize, o - 2);
        else if (o < 10) v = 0;
        else if (o < 14) v = LeByte(dataStart, o - 10);
        else if (o < 18) v = LeByte(dib, o - 14);
        else if (o < 22) v = LeByte(w, o - 18);
        else if (o < 26) v = LeByte(h, o - 22);
        else if (o < 28) v = o == 26 ? 1 : 0;
        else if (o < 30) v = LeByte(bpp, o - 28);
        else if (o < 34) v = LeByte(mode == 0 ? 3 : 0, o - 30);
        else if (o < 38) v = LeByte(imageSize, o - 34);
        else if (o < 46) v = LeByte(2835, (o - 38) & 3);
        else if (o < 50) v = LeByte(clrUsed, o - 46);
        else if (o < 54) v = 0;
        else if (o < 58) v = LeByte(0x00FF0000, o - 54);
        else if (o < 62) v = LeByte(0x0000FF00, o - 58);
        else if (o < 66) v = LeByte(0x000000FF, o - 62);
        else if (o < 70) v = LeByte(-16777216, o - 66);
        else if (o < 74) v = LeByte(0x57696E20, o - 70);
        else v = 0;
        return v;
    }

    private static void EncodeKernel(
        Index1D index, ArrayView<byte> rgba, ArrayView<byte> output, ArrayView<int> palette, ArrayView<int> p)
    {
        int i = index.X;
        int mode = p[0];
        int w = p[1];
        int h = p[2];
        int fileSize = p[3];
        int dataStart = p[4];
        int stride = p[5];
        int imageSize = p[6];
        int palCount = p[7];
        int headerLen = mode == 0 ? 14 + 108 : 14 + 40;

        int value;
        if (i < headerLen)
        {
            value = HeaderByte(i, mode, w, h, fileSize, dataStart, imageSize, palCount);
        }
        else if (i < dataStart)
        {
            // Palette entries B, G, R, 0.
            int k = i - headerLen;
            int e = k >> 2;
            int c = k & 3;
            int col = palette[e];
            if (c == 0) value = col & 0xFF;
            else if (c == 1) value = (col >> 8) & 0xFF;
            else if (c == 2) value = (col >> 16) & 0xFF;
            else value = 0;
        }
        else
        {
            int o = i - dataStart;
            int row = o / stride;
            int col = o - row * stride;
            int srcY = h - 1 - row;
            value = 0;
            if (mode == 0)
            {
                int x = col >> 2;
                int ch = col & 3;
                int src = (srcY * w + x) * 4;
                if (ch == 0) value = rgba[src + 2];
                else if (ch == 1) value = rgba[src + 1];
                else if (ch == 2) value = rgba[src];
                else value = rgba[src + 3];
            }
            else if (mode == 1)
            {
                int x = col / 3;
                int ch = col - x * 3;
                if (x < w)
                {
                    int src = (srcY * w + x) * 4;
                    if (ch == 0) value = rgba[src + 2];
                    else if (ch == 1) value = rgba[src + 1];
                    else value = rgba[src];
                }
            }
            else
            {
                int x = col;
                if (x < w)
                {
                    int src = (srcY * w + x) * 4;
                    int c = (rgba[src] << 16) | (rgba[src + 1] << 8) | rgba[src + 2];
                    // Lower bound in the sorted palette: the colour is present by construction.
                    int lo = 0;
                    int hi = palCount - 1;
                    while (lo < hi)
                    {
                        int mid = (lo + hi) >> 1;
                        if (palette[mid] < c) lo = mid + 1;
                        else hi = mid;
                    }
                    value = lo;
                }
            }
        }
        output[i] = (byte)value;
    }

    /// <summary>No-op (kernels are owned by the accelerator).</summary>
    public void Dispose() { }
}
