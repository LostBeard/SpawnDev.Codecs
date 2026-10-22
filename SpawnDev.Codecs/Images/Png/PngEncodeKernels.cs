// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// The PNG encode kernels. Pixels never leave the accelerator:
//
//   1. AnalyzeKernel (1 thread per pixel)  clears the "opaque" / "gray" flags when a pixel disproves them
//                                          (drives ColorMode.Auto).
//   2. PackKernel    (1 thread per pixel)  RGBA -> the chosen colour type / bit depth, into unfiltered scanlines.
//   3. FilterKernel  (1 thread per row)    picks the best of the five PNG filters (libpng's minimum-sum-of-absolute-
//                                          differences heuristic) or applies a fixed one. Encoder-side filtering reads
//                                          the ORIGINAL previous row, so rows are fully independent.
//   4. DeflateKernels (Compression/)       block-parallel LZ77 + dynamic Huffman -> byte-aligned blocks.
//   5. LayoutKernel  (1 thread)            IDAT chunk offsets + combined Adler-32.
//   6. FrameKernel   (1 thread per block)  writes each block as its own IDAT chunk (length, type, zlib header on the
//                                          first, Adler-32 on the last, CRC-32), plus the header blob and IEND.
//
// Every compressed block becomes one IDAT chunk, so each chunk's CRC is computed by the thread that wrote it:
// no CRC combining is needed.

using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.Codecs.Images.Png;

/// <summary>Slots of the small encoder metadata block.</summary>
internal static class PngEncMeta
{
    public const int Size = 16;
    public const int Opaque = 0;          // 1 until a pixel with alpha != max is found
    public const int Gray = 1;            // 1 until a pixel with r != g or g != b is found
    public const int ColorType = 2;
    public const int BitDepth = 3;        // 8 or 16
    public const int Channels = 4;
    public const int RowBytes = 5;
    public const int InputBytesPerPixel = 6;   // 4 (Rgba8) or 8 (Rgba16)
    public const int Width = 7;
    public const int Height = 8;
    public const int FilterMode = 9;      // 0 adaptive, 1..5 = fixed None, Sub, Up, Average, Paeth
    public const int HeaderLength = 10;   // bytes of the host-built header blob
    public const int Adler = 11;          // combined Adler-32 of the filtered data (written by LayoutKernel)
    public const int Total = 12;          // total file bytes (written by LayoutKernel), -1 on error
}

/// <summary>Loads and launches the PNG encode kernels.</summary>
internal sealed class PngEncodeKernels : IDisposable
{
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>> _analyze;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>> _pack;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>> _filter;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>> _layout;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<uint>, ArrayView<int>> _frame;

    public PngEncodeKernels(Accelerator accelerator)
    {
        _analyze = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<int>>(AnalyzeKernel);
        _pack = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>>(PackKernel);
        _filter = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>>(FilterKernel);
        _layout = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>>(LayoutKernel);
        _frame = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>,
            ArrayView<uint>, ArrayView<int>>(FrameKernel);
    }

    public void Analyze(ArrayView<byte> rgba, ArrayView<int> meta, int pixels) => _analyze(pixels, rgba, meta);
    public void Pack(ArrayView<byte> rgba, ArrayView<byte> raw, ArrayView<int> meta, int pixels) => _pack(pixels, rgba, raw, meta);
    public void Filter(ArrayView<byte> raw, ArrayView<byte> filtered, ArrayView<int> meta, int rows) => _filter(rows, raw, filtered, meta);

    /// <param name="res">Per-block results from the deflate block kernel (4 ints per block).</param>
    /// <param name="prm">Deflate prm block (blockCount at [4]).</param>
    /// <param name="layout">blockCount + 1 ints: chunk start offsets.</param>
    /// <param name="meta">Encoder meta (header length in, adler / total out).</param>
    public void Layout(ArrayView<int> res, ArrayView<int> prm, ArrayView<int> layout, ArrayView<int> meta) =>
        _layout(1, res, prm, layout, meta);

    public void Frame(
        ArrayView<byte> slots, ArrayView<byte> header, ArrayView<byte> output, ArrayView<int> res,
        ArrayView<int> layout, ArrayView<int> prm, ArrayView<uint> crcTable, ArrayView<int> meta, int blocks) =>
        _frame(blocks, slots, header, output, res, layout, prm, crcTable, meta);

    // ------------------------------------------------------------------

    private static void AnalyzeKernel(Index1D idx, ArrayView<byte> rgba, ArrayView<int> meta)
    {
        int bpp = meta[PngEncMeta.InputBytesPerPixel];
        int o = idx.X * bpp;
        if (bpp == 4)
        {
            if (rgba[o + 3] != 255) meta[PngEncMeta.Opaque] = 0;
            bool notGray = (rgba[o] != rgba[o + 1]) | (rgba[o + 1] != rgba[o + 2]);
            if (notGray) meta[PngEncMeta.Gray] = 0;
        }
        else
        {
            // 16-bit samples, little-endian in memory.
            bool notOpaque = (rgba[o + 6] != 255) | (rgba[o + 7] != 255);
            if (notOpaque) meta[PngEncMeta.Opaque] = 0;
            bool notGray = (rgba[o] != rgba[o + 2]) | (rgba[o + 1] != rgba[o + 3]) |
                           (rgba[o + 2] != rgba[o + 4]) | (rgba[o + 3] != rgba[o + 5]);
            if (notGray) meta[PngEncMeta.Gray] = 0;
        }
    }

    private static void PackKernel(Index1D idx, ArrayView<byte> rgba, ArrayView<byte> raw, ArrayView<int> meta)
    {
        int width = meta[PngEncMeta.Width];
        int colorType = meta[PngEncMeta.ColorType];
        int depth = meta[PngEncMeta.BitDepth];
        int channels = meta[PngEncMeta.Channels];
        int rowBytes = meta[PngEncMeta.RowBytes];
        int inBpp = meta[PngEncMeta.InputBytesPerPixel];
        int i = idx.X;
        int y = i / width;
        int x = i - y * width;
        int src = i * inBpp;
        int rowStart = y * (rowBytes + 1);
        if (x == 0) raw[rowStart] = 0;
        int dst = rowStart + 1 + x * channels * (depth >> 3);

        // Sample k (0 = R, 1 = G, 2 = B, 3 = A) of the input pixel as a 16-bit value, then as 8 or 16 bits.
        if (depth == 8)
        {
            if (colorType == 6)
            {
                raw[dst] = rgba[src]; raw[dst + 1] = rgba[src + 1]; raw[dst + 2] = rgba[src + 2]; raw[dst + 3] = rgba[src + 3];
            }
            else if (colorType == 2)
            {
                raw[dst] = rgba[src]; raw[dst + 1] = rgba[src + 1]; raw[dst + 2] = rgba[src + 2];
            }
            else if (colorType == 4)
            {
                raw[dst] = rgba[src]; raw[dst + 1] = rgba[src + 3];
            }
            else
            {
                raw[dst] = rgba[src];
            }
        }
        else
        {
            // 16-bit PNG samples are big-endian; the input is little-endian: swap each pair.
            if (colorType == 6)
            {
                for (int k = 0; k < 4; k++) { raw[dst + 2 * k] = rgba[src + 2 * k + 1]; raw[dst + 2 * k + 1] = rgba[src + 2 * k]; }
            }
            else if (colorType == 2)
            {
                for (int k = 0; k < 3; k++) { raw[dst + 2 * k] = rgba[src + 2 * k + 1]; raw[dst + 2 * k + 1] = rgba[src + 2 * k]; }
            }
            else if (colorType == 4)
            {
                raw[dst] = rgba[src + 1]; raw[dst + 1] = rgba[src];
                raw[dst + 2] = rgba[src + 7]; raw[dst + 3] = rgba[src + 6];
            }
            else
            {
                raw[dst] = rgba[src + 1]; raw[dst + 1] = rgba[src];
            }
        }
    }

    private static int FilterPredictor(int type, int a, int b, int c)
    {
        int p = 0;
        if (type == 1) p = a;
        else if (type == 2) p = b;
        else if (type == 3) p = (a + b) >> 1;
        else if (type == 4)
        {
            int pp = a + b - c;
            int pa = pp - a; if (pa < 0) pa = -pa;
            int pb = pp - b; if (pb < 0) pb = -pb;
            int pc = pp - c; if (pc < 0) pc = -pc;
            bool useA = (pa <= pb) & (pa <= pc);
            bool useB = pb <= pc;
            p = c;
            if (useA) p = a;
            else if (useB) p = b;
        }
        return p;
    }

    private static void FilterKernel(Index1D idx, ArrayView<byte> raw, ArrayView<byte> filtered, ArrayView<int> meta)
    {
        int rowBytes = meta[PngEncMeta.RowBytes];
        int bpp = meta[PngEncMeta.Channels] * (meta[PngEncMeta.BitDepth] >> 3);
        int mode = meta[PngEncMeta.FilterMode];
        int y = idx.X;
        int stride = rowBytes + 1;
        int cur = y * stride + 1;
        int prev = cur - stride;

        int best = 0;
        if (mode == 0)
        {
            int bestSum = int.MaxValue;
            for (int type = 0; type < 5; type++)
            {
                int sum = 0;
                for (int i = 0; i < rowBytes; i++)
                {
                    int x = raw[cur + i];
                    int a = 0;
                    int c = 0;
                    if (i >= bpp)
                    {
                        a = raw[cur + i - bpp];
                        if (y > 0) c = raw[prev + i - bpp];
                    }
                    int b = 0;
                    if (y > 0) b = raw[prev + i];
                    int f = (x - FilterPredictor(type, a, b, c)) & 0xFF;
                    sum += f < 128 ? f : 256 - f;
                }
                if (sum < bestSum)
                {
                    bestSum = sum;
                    best = type;
                }
            }
        }
        else
        {
            best = mode - 1;
        }

        filtered[y * stride] = (byte)best;
        for (int i = 0; i < rowBytes; i++)
        {
            int x = raw[cur + i];
            int a = 0;
            int c = 0;
            if (i >= bpp)
            {
                a = raw[cur + i - bpp];
                if (y > 0) c = raw[prev + i - bpp];
            }
            int b = 0;
            if (y > 0) b = raw[prev + i];
            filtered[cur + i] = (byte)(x - FilterPredictor(best, a, b, c));
        }
    }

    private static void LayoutKernel(Index1D _, ArrayView<int> res, ArrayView<int> prm, ArrayView<int> layout, ArrayView<int> meta)
    {
        int blocks = prm[4];
        const uint Base = 65521u;
        uint a = 1;
        uint b = 0;
        int pos = meta[PngEncMeta.HeaderLength];
        int failed = 0;
        for (int i = 0; i < blocks; i++)
        {
            int outLen = res[i * 4 + 0];
            if (outLen < 0) failed = 1;
            layout[i] = pos;
            int dataLen = (outLen > 0 ? outLen : 0) + (i == 0 ? 2 : 0) + (i == blocks - 1 ? 4 : 0);
            pos += 12 + dataLen;
            uint a2 = (uint)res[i * 4 + 1];
            uint b2 = (uint)res[i * 4 + 2];
            uint len2 = (uint)res[i * 4 + 3] % Base;
            b = (b + b2 + len2 * ((a + Base - 1u) % Base)) % Base;
            a = (a + a2 + Base - 1u) % Base;
        }
        layout[blocks] = pos;                        // where IEND goes
        meta[PngEncMeta.Adler] = (int)((b << 16) | a);
        meta[PngEncMeta.Total] = failed != 0 ? -1 : pos + 12;
    }

    private static void PutBe32(ArrayView<byte> o, int at, uint v)
    {
        o[at] = (byte)(v >> 24);
        o[at + 1] = (byte)(v >> 16);
        o[at + 2] = (byte)(v >> 8);
        o[at + 3] = (byte)v;
    }

    private static uint CrcByte(ArrayView<uint> table, uint crc, int b)
    {
        return table[(int)((crc ^ (uint)b) & 0xFF)] ^ (crc >> 8);
    }

    private static void FrameKernel(
        Index1D idx, ArrayView<byte> slots, ArrayView<byte> header, ArrayView<byte> output, ArrayView<int> res,
        ArrayView<int> layout, ArrayView<int> prm, ArrayView<uint> crcTable, ArrayView<int> meta)
    {
        int block = idx.X;
        int blocks = prm[4];
        int slotCap = prm[3];
        bool first = block == 0;
        bool last = block == blocks - 1;
        int outLen = res[block * 4 + 0];
        if (outLen < 0) return;

        if (first)
        {
            int hdrLen = meta[PngEncMeta.HeaderLength];
            for (int i = 0; i < hdrLen; i++) output[i] = header[i];
        }

        int start = layout[block];
        int dataLen = outLen + (first ? 2 : 0) + (last ? 4 : 0);
        PutBe32(output, start, (uint)dataLen);
        output[start + 4] = 0x49; output[start + 5] = 0x44; output[start + 6] = 0x41; output[start + 7] = 0x54; // "IDAT"
        uint crc = Compression.Crc32Gpu.Seed;
        crc = CrcByte(crcTable, crc, 0x49);
        crc = CrcByte(crcTable, crc, 0x44);
        crc = CrcByte(crcTable, crc, 0x41);
        crc = CrcByte(crcTable, crc, 0x54);
        int p = start + 8;
        if (first)
        {
            output[p] = 0x78; output[p + 1] = 0x01;
            crc = CrcByte(crcTable, crc, 0x78);
            crc = CrcByte(crcTable, crc, 0x01);
            p += 2;
        }
        int slot = block * slotCap;
        for (int i = 0; i < outLen; i++)
        {
            int v = slots[slot + i];
            output[p + i] = (byte)v;
            crc = CrcByte(crcTable, crc, v);
        }
        p += outLen;
        if (last)
        {
            uint adler = (uint)meta[PngEncMeta.Adler];
            int b0 = (int)(adler >> 24), b1 = (int)((adler >> 16) & 0xFF), b2 = (int)((adler >> 8) & 0xFF), b3 = (int)(adler & 0xFF);
            output[p] = (byte)b0; output[p + 1] = (byte)b1; output[p + 2] = (byte)b2; output[p + 3] = (byte)b3;
            crc = CrcByte(crcTable, crc, b0);
            crc = CrcByte(crcTable, crc, b1);
            crc = CrcByte(crcTable, crc, b2);
            crc = CrcByte(crcTable, crc, b3);
            p += 4;
        }
        PutBe32(output, p, Compression.Crc32Gpu.Finish(crc));

        if (last)
        {
            // IEND: length 0, "IEND", CRC of "IEND".
            int e = layout[blocks];
            PutBe32(output, e, 0);
            output[e + 4] = 0x49; output[e + 5] = 0x45; output[e + 6] = 0x4E; output[e + 7] = 0x44;
            output[e + 8] = 0xAE; output[e + 9] = 0x42; output[e + 10] = 0x60; output[e + 11] = 0x82;
        }
    }

    public void Dispose() { }
}
