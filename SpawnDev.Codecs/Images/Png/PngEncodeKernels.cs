// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// The PNG encode kernels. Pixels never leave the accelerator:
//
//   1. AnalyzeKernel (1 thread per pixel)  clears the "opaque" / "gray" flags when a pixel disproves them
//                                          (drives ColorMode.Auto).
//   2. PackKernel    (1 thread per pixel)  RGBA -> the chosen colour type / bit depth, into unfiltered scanlines.
//   3. FilterCost / FilterPick / FilterApply  pick the best of the five PNG filters per row (libpng's minimum-sum-of-
//                    absolute-differences heuristic) or apply a fixed one: the five sums per FilterSegment bytes of a
//                    row (1 thread each), the choice per row (1 thread each), the filtered bytes (1 thread each).
//                    Encoder-side filtering reads the ORIGINAL previous row, so rows and bytes are fully independent.
//   4. DeflateKernels (Compression/)       block-parallel LZ77 + dynamic Huffman -> byte-aligned blocks.
//   5. LayoutKernel  (resumable)           IDAT chunk offsets + combined Adler-32, LayoutChunk blocks per launch.
//   6. Frame kernels                       the header blob (1 thread per byte); each block's IDAT chunk framing (1 thread
//                                          per block: length, type, zlib header on the first, Adler-32 on the last, IEND);
//                                          the block bytes (1 thread per byte); the chunk CRC-32s in Crc32Gpu.PieceBytes
//                                          pieces (1 thread per piece) combined per chunk.
//
// Every compressed block becomes one IDAT chunk, so each chunk's CRC covers one block. No launch does work that grows
// with the image (a long launch stalls the desktop compositor and trips the OS GPU watchdog).

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Compression;

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
    public const int FilterSegments = 13; // FilterSegment-byte segments per row (adaptive filter costs)
}

/// <summary>Loads and launches the PNG encode kernels.</summary>
internal sealed class PngEncodeKernels : IDisposable
{
    /// <summary>Row bytes one filter-cost thread sums (five filters each).</summary>
    public const int FilterSegment = 4096;

    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>> _analyze;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>> _pack;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>> _filterCost;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>> _filterPick;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>> _filterApply;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int> _layout;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>> _frameHeader;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>> _frameChunk;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>> _frameCopy;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<uint>, ArrayView<uint>, int> _crcPiece;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<uint>, ArrayView<uint>, int> _crcCombine;

    public PngEncodeKernels(Accelerator accelerator)
    {
        _analyze = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<int>>(AnalyzeKernel);
        _pack = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>>(PackKernel);
        _filterCost = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>>(FilterCostKernel);
        _filterPick = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>>(FilterPickKernel);
        _filterApply = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>>(FilterApplyKernel);
        _layout = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int>(LayoutKernel);
        _frameHeader = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<byte>>(FrameHeaderKernel);
        _frameChunk = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>>(FrameChunkKernel);
        _frameCopy = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>>(FrameCopyKernel);
        _crcPiece = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<uint>, ArrayView<uint>, int>(FrameCrcPieceKernel);
        _crcCombine = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<uint>, ArrayView<uint>, int>(FrameCrcCombineKernel);
    }

    /// <summary>Filter-cost segments of a row of <paramref name="rowBytes"/> bytes.</summary>
    public static int FilterSegments(int rowBytes) => (rowBytes + FilterSegment - 1) / FilterSegment;

    /// <summary>CRC pieces of the largest IDAT chunk a block of <paramref name="blockSize"/> input bytes can make.</summary>
    public static int MaxCrcPiecesPerChunk(int blockSize) => Crc32Gpu.PiecesOf(4 + DeflateGpu.MaxBlockOutput(blockSize) + 6);

    public void Analyze(ArrayView<byte> rgba, ArrayView<int> meta, int pixels) => _analyze(pixels, rgba, meta);
    public void Pack(ArrayView<byte> rgba, ArrayView<byte> raw, ArrayView<int> meta, int pixels) => _pack(pixels, rgba, raw, meta);

    /// <summary>
    /// Filters <paramref name="rows"/> scanlines of <paramref name="rowBytes"/> bytes from <paramref name="raw"/> into
    /// <paramref name="filtered"/>: per-segment filter costs into <paramref name="costs"/> (adaptive mode only; at least
    /// rows * <see cref="FilterSegments"/> * 5 ints), the choice per row, the filtered bytes.
    /// </summary>
    public void Filter(ArrayView<byte> raw, ArrayView<byte> filtered, ArrayView<int> costs, ArrayView<int> meta, int rows, int rowBytes, bool adaptive)
    {
        if (adaptive) _filterCost(rows * FilterSegments(rowBytes), raw, costs, meta);
        _filterPick(rows, filtered, costs, meta);
        _filterApply(rows * (rowBytes + 1), raw, filtered, meta);
    }

    /// <param name="res">Per-block results from the deflate block kernel (4 ints per block).</param>
    /// <param name="prm">Deflate prm block (blockCount at [4], the resumable layout state at [8..11]).</param>
    /// <param name="layout">blockCount + 1 ints: chunk start offsets.</param>
    /// <param name="meta">Encoder meta (header length in, adler / total out).</param>
    /// <param name="blocks">Block count (launches of <see cref="DeflateKernels.LayoutChunk"/> blocks each).</param>
    public void Layout(ArrayView<int> res, ArrayView<int> prm, ArrayView<int> layout, ArrayView<int> meta, int blocks)
    {
        for (int step = 0; step * DeflateKernels.LayoutChunk < blocks; step++) _layout(1, res, prm, layout, meta, step);
    }

    /// <summary>Writes the PNG: header blob, IDAT chunks (framing, block bytes, CRCs), IEND.</summary>
    public void Frame(
        ArrayView<byte> slots, ArrayView<byte> header, int headerLength, ArrayView<byte> output, ArrayView<int> res,
        ArrayView<int> layout, ArrayView<int> prm, ArrayView<uint> crcTable, ArrayView<uint> crcPieces, ArrayView<int> meta,
        int blocks, int slotBytes, int piecesPerChunk)
    {
        if (headerLength > 0) _frameHeader(headerLength, header, output);
        _frameChunk(blocks, output, res, layout, prm, meta);
        _frameCopy(slotBytes, slots, output, res, layout, prm);
        _crcPiece(blocks * piecesPerChunk, output, res, layout, prm, crcTable, crcPieces, piecesPerChunk);
        _crcCombine(blocks, output, res, layout, prm, crcTable, crcPieces, piecesPerChunk);
    }

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

    /// <summary>Thread per (row, segment): the five filters' sums of absolute differences over the segment's bytes.</summary>
    private static void FilterCostKernel(Index1D idx, ArrayView<byte> raw, ArrayView<int> costs, ArrayView<int> meta)
    {
        int rowBytes = meta[PngEncMeta.RowBytes];
        int bpp = meta[PngEncMeta.Channels] * (meta[PngEncMeta.BitDepth] >> 3);
        int segs = meta[PngEncMeta.FilterSegments];
        int y = idx.X / segs;
        int seg = idx.X - y * segs;
        int stride = rowBytes + 1;
        int cur = y * stride + 1;
        int prev = cur - stride;
        int i0 = seg * FilterSegment;
        int i1 = i0 + FilterSegment;
        if (i1 > rowBytes) i1 = rowBytes;
        int s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0;
        for (int i = i0; i < i1; i++)
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
            s0 += AbsFiltered(x - FilterPredictor(0, a, b, c));
            s1 += AbsFiltered(x - FilterPredictor(1, a, b, c));
            s2 += AbsFiltered(x - FilterPredictor(2, a, b, c));
            s3 += AbsFiltered(x - FilterPredictor(3, a, b, c));
            s4 += AbsFiltered(x - FilterPredictor(4, a, b, c));
        }
        int o = idx.X * 5;
        costs[o + 0] = s0;
        costs[o + 1] = s1;
        costs[o + 2] = s2;
        costs[o + 3] = s3;
        costs[o + 4] = s4;
    }

    /// <summary>A filtered byte as a signed magnitude (libpng's heuristic: 0..127 and 256 - (128..255)).</summary>
    private static int AbsFiltered(int d)
    {
        int f = d & 0xFF;
        return f < 128 ? f : 256 - f;
    }

    /// <summary>Thread per row: the filter type (smallest sum, the lowest type on a tie; or the fixed mode) into the row's filter byte.</summary>
    private static void FilterPickKernel(Index1D idx, ArrayView<byte> filtered, ArrayView<int> costs, ArrayView<int> meta)
    {
        int rowBytes = meta[PngEncMeta.RowBytes];
        int mode = meta[PngEncMeta.FilterMode];
        int segs = meta[PngEncMeta.FilterSegments];
        int y = idx.X;
        int best = 0;
        if (mode == 0)
        {
            int bestSum = int.MaxValue;
            for (int type = 0; type < 5; type++)
            {
                int sum = 0;
                for (int seg = 0; seg < segs; seg++) sum += costs[(y * segs + seg) * 5 + type];
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
        filtered[y * (rowBytes + 1)] = (byte)best;
    }

    /// <summary>Thread per byte of the filtered scanlines: applies its row's filter (the filter bytes are already written).</summary>
    private static void FilterApplyKernel(Index1D idx, ArrayView<byte> raw, ArrayView<byte> filtered, ArrayView<int> meta)
    {
        int rowBytes = meta[PngEncMeta.RowBytes];
        int bpp = meta[PngEncMeta.Channels] * (meta[PngEncMeta.BitDepth] >> 3);
        int stride = rowBytes + 1;
        int y = idx.X / stride;
        int col = idx.X - y * stride;
        if (col == 0) return;
        int i = col - 1;
        int cur = y * stride + 1;
        int prev = cur - stride;
        int type = filtered[y * stride];
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
        filtered[cur + i] = (byte)(x - FilterPredictor(type, a, b, c));
    }

    /// <summary>
    /// Layout launch <paramref name="step"/>: chunk offsets and the Adler-32 combine of blocks
    /// [step * LayoutChunk, ...), continuing from the state kept in prm; the last launch writes the totals.
    /// </summary>
    private static void LayoutKernel(Index1D _, ArrayView<int> res, ArrayView<int> prm, ArrayView<int> layout, ArrayView<int> meta, int step)
    {
        int blocks = prm[DeflateKernels.PrmBlocks];
        int lo = step * DeflateKernels.LayoutChunk;
        int hi = lo + DeflateKernels.LayoutChunk;
        if (hi > blocks) hi = blocks;
        const uint Base = 65521u;
        uint a = 1;
        uint b = 0;
        int pos = meta[PngEncMeta.HeaderLength];
        int failed = 0;
        if (step > 0)
        {
            pos = prm[DeflateKernels.PrmLayoutPos];
            a = (uint)prm[DeflateKernels.PrmLayoutA];
            b = (uint)prm[DeflateKernels.PrmLayoutB];
            failed = prm[DeflateKernels.PrmLayoutFailed];
        }
        for (int i = lo; i < hi; i++)
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
        prm[DeflateKernels.PrmLayoutPos] = pos;
        prm[DeflateKernels.PrmLayoutA] = (int)a;
        prm[DeflateKernels.PrmLayoutB] = (int)b;
        prm[DeflateKernels.PrmLayoutFailed] = failed;
        if (hi == blocks)
        {
            layout[blocks] = pos;                        // where IEND goes
            meta[PngEncMeta.Adler] = (int)((b << 16) | a);
            meta[PngEncMeta.Total] = failed != 0 ? -1 : pos + 12;
        }
    }

    private static void PutBe32(ArrayView<byte> o, int at, uint v)
    {
        o[at] = (byte)(v >> 24);
        o[at + 1] = (byte)(v >> 16);
        o[at + 2] = (byte)(v >> 8);
        o[at + 3] = (byte)v;
    }

    /// <summary>Thread per byte of the host-built header blob (signature, IHDR, text chunks).</summary>
    private static void FrameHeaderKernel(Index1D idx, ArrayView<byte> header, ArrayView<byte> output)
    {
        output[idx.X] = header[idx.X];
    }

    /// <summary>Bytes of block <paramref name="block"/>'s IDAT chunk payload (zlib header on the first, Adler-32 on the last).</summary>
    private static int ChunkDataLength(int block, int blocks, int outLen) =>
        outLen + (block == 0 ? 2 : 0) + (block == blocks - 1 ? 4 : 0);

    /// <summary>Thread per block: the IDAT chunk's length, type, zlib header (first) and Adler-32 (last); IEND (last).</summary>
    private static void FrameChunkKernel(
        Index1D idx, ArrayView<byte> output, ArrayView<int> res, ArrayView<int> layout, ArrayView<int> prm, ArrayView<int> meta)
    {
        int block = idx.X;
        int blocks = prm[DeflateKernels.PrmBlocks];
        bool first = block == 0;
        bool last = block == blocks - 1;
        int outLen = res[block * 4 + 0];
        if (outLen < 0) return;

        int start = layout[block];
        PutBe32(output, start, (uint)ChunkDataLength(block, blocks, outLen));
        output[start + 4] = 0x49; output[start + 5] = 0x44; output[start + 6] = 0x41; output[start + 7] = 0x54; // "IDAT"
        int p = start + 8;
        if (first)
        {
            output[p] = 0x78; output[p + 1] = 0x01;
            p += 2;
        }
        if (last)
        {
            PutBe32(output, p + outLen, (uint)meta[PngEncMeta.Adler]);
            // IEND: length 0, "IEND", CRC of "IEND".
            int e = layout[blocks];
            PutBe32(output, e, 0);
            output[e + 4] = 0x49; output[e + 5] = 0x45; output[e + 6] = 0x4E; output[e + 7] = 0x44;
            output[e + 8] = 0xAE; output[e + 9] = 0x42; output[e + 10] = 0x60; output[e + 11] = 0x82;
        }
    }

    /// <summary>Thread per slot byte: copies each block's compressed bytes into its IDAT chunk.</summary>
    private static void FrameCopyKernel(
        Index1D idx, ArrayView<byte> slots, ArrayView<byte> output, ArrayView<int> res, ArrayView<int> layout, ArrayView<int> prm)
    {
        int slotCap = prm[DeflateKernels.PrmSlotCap];
        int i = idx.X;
        int block = i / slotCap;
        int off = i - block * slotCap;
        if (block >= prm[DeflateKernels.PrmBlocks]) return;
        if (off < res[block * 4 + 0])
            output[layout[block] + 8 + (block == 0 ? 2 : 0) + off] = slots[i];
    }

    /// <summary>Thread per (block, piece): the CRC of one piece of the IDAT chunk's type + payload.</summary>
    private static void FrameCrcPieceKernel(
        Index1D idx, ArrayView<byte> output, ArrayView<int> res, ArrayView<int> layout, ArrayView<int> prm,
        ArrayView<uint> table, ArrayView<uint> pieces, int piecesPerChunk)
    {
        int block = idx.X / piecesPerChunk;
        int j = idx.X - block * piecesPerChunk;
        int blocks = prm[DeflateKernels.PrmBlocks];
        if (block >= blocks) return;
        int outLen = res[block * 4 + 0];
        if (outLen < 0) return;
        int regionLen = 4 + ChunkDataLength(block, blocks, outLen);
        if (j >= Crc32Gpu.PiecesOf(regionLen)) return;
        pieces[idx.X] = Crc32Gpu.PieceOfRange(output, layout[block] + 4, regionLen, j, table);
    }

    /// <summary>Thread per block: combines the chunk's CRC pieces and writes the CRC after the payload.</summary>
    private static void FrameCrcCombineKernel(
        Index1D idx, ArrayView<byte> output, ArrayView<int> res, ArrayView<int> layout, ArrayView<int> prm,
        ArrayView<uint> table, ArrayView<uint> pieces, int piecesPerChunk)
    {
        int block = idx.X;
        int blocks = prm[DeflateKernels.PrmBlocks];
        int outLen = res[block * 4 + 0];
        if (outLen < 0) return;
        int dataLen = ChunkDataLength(block, blocks, outLen);
        int at = block * piecesPerChunk;
        // A chunk has at most MaxCrcPiecesPerChunk (<= Crc32Gpu.CombinePieces + 1) pieces: one combine step folds them all.
        Crc32Gpu.CombineStep(pieces, at, Crc32Gpu.PiecesOf(4 + dataLen), 0, table);
        PutBe32(output, layout[block] + 8 + dataLen, Crc32Gpu.Finish(pieces[at]));
    }

    public void Dispose() { }
}
