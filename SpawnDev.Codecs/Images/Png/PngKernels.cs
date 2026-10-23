// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// The PNG decode kernels. Every stage of the decode runs on the accelerator; the host only dispatches and
// reads back a few scalars:
//
//   1. Scan          (resumable, 1 thread)  walks the chunk list: chunk table, IDAT gather table, IHDR/PLTE/tRNS/acTL
//                                    facts, and the CRC piece table. ScanChunks chunks per launch.
//   2. CrcPiece      (1 thread per CRC piece)  CRC-32 of one Crc32Gpu.PieceBytes piece of a chunk.
//      CrcCombine    (1 thread per chunk, resumable)  folds the pieces of each chunk, verifies critical-chunk CRCs.
//   3. GatherKernel  (1 thread per byte)   concatenates the IDAT payloads into one zlib stream.
//   4. InflateKernel (resumable, 1 thread) zlib -> filtered scanlines (Compression/InflateKernel).
//   5. Unfilter      (1 thread per filter byte lane)  reverses the per-scanline filters, in place, UnfilterLaneBytes
//                                    bytes per lane per launch.
//   6. ToRgbaKernel  (1 thread per output pixel)  samples -> RGBA8 / RGBA16: palette, tRNS, bit depths 1..16,
//                                  gray / RGB / gray-alpha / RGBA, Adam7 de-interlacing.
//
// Inflate and unfilter are sequential dependency chains (each output byte depends on earlier ones), so per
// the library's cardinal rule they run on single GPU threads (unfilter: one per byte lane, the lanes never
// read each other): still accelerator-resident, never on the host. No launch does work that grows with the
// file: every serial stage is a chain of bounded launches whose count the host knows from the sizes it
// already has (a long launch stalls the desktop compositor and trips the OS GPU watchdog).
// All scalar metadata travels in one shared int buffer (PngMeta) so a kernel and the host cannot disagree.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Compression;

namespace SpawnDev.Codecs.Images.Png;

/// <summary>Loads and launches the PNG decode kernels.</summary>
public sealed class PngKernels : IDisposable
{
    private const uint TypeIhdr = 0x49484452;
    private const uint TypeIdat = 0x49444154;
    private const uint TypePlte = 0x504C5445;
    private const uint TypeTrns = 0x74524E53;
    private const uint TypeIend = 0x49454E44;
    private const uint TypeActl = 0x6163544C;
    private const uint TypeSrgb = 0x73524742;

    /// <summary>Chunks one scan launch walks at most.</summary>
    public const int ScanChunks = 4096;

    /// <summary>
    /// Bytes one unfilter lane reverses per launch. One thread per filter byte lane (bpp lanes) walks its bytes in
    /// scanline order; 32 KiB is about 2 ms on a desktop GPU.
    /// </summary>
    public const int UnfilterLaneBytes = 32768;

    private readonly Action<Index1D, ArrayView<byte>, int, ArrayView<int>> _scanInit;
    private readonly Action<Index1D, ArrayView<byte>, int, ArrayView<int>, ArrayView<int>, ArrayView<int>, int> _scan;
    private readonly Action<Index1D, ArrayView<int>> _scanFinish;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<uint>, ArrayView<uint>, ArrayView<int>> _crcPiece;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<uint>, ArrayView<uint>, ArrayView<int>, int> _crcCombine;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<byte>, ArrayView<int>> _gather;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>, int, int> _unfilter;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<int>> _toRgba;

    /// <summary>Compile the kernels.</summary>
    private readonly LaunchPacer _pacer;

    public PngKernels(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _pacer = new LaunchPacer(accelerator);
        _scanInit = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, int, ArrayView<int>>(ScanInitKernel);
        _scan = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, int, ArrayView<int>, ArrayView<int>, ArrayView<int>, int>(ScanKernel);
        _scanFinish = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>>(ScanFinishKernel);
        _crcPiece = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<uint>, ArrayView<uint>, ArrayView<int>>(CrcPieceKernel);
        _crcCombine = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<uint>, ArrayView<uint>, ArrayView<int>, int>(CrcCombineKernel);
        _gather = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<byte>, ArrayView<int>>(GatherKernel);
        _unfilter = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<int>, int, int>(UnfilterKernel);
        _toRgba = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<int>>(ToRgbaKernel);
    }

    /// <summary>Upper bound of the chunks a file of <paramref name="fileLength"/> bytes holds (each chunk is at least 12 bytes).</summary>
    public static int MaxChunks(int fileLength, int chunkCapacity) =>
        (int)Math.Min(chunkCapacity, Math.Max(0, fileLength - 8) / 12 + 1);

    /// <summary>Upper bound of the CRC pieces of a file's chunks (the size the piece buffer of <see cref="CheckCrc"/> needs).</summary>
    public static long MaxCrcPieces(int fileLength, int chunkCapacity) =>
        (long)Crc32Gpu.PiecesOf(Math.Max(1, fileLength)) + MaxChunks(fileLength, chunkCapacity);

    /// <summary>
    /// Chunk scan: fills <paramref name="chunks"/> (4 ints per chunk: type, data offset, length, first CRC piece),
    /// <paramref name="idat"/> (3 per IDAT) and <paramref name="meta"/>. A set-up launch, then launches of
    /// <see cref="ScanChunks"/> chunks each (as many as the file can hold), then a finishing launch.
    /// </summary>
    public void Scan(ArrayView<byte> file, int fileLength, ArrayView<int> chunks, ArrayView<int> idat, ArrayView<int> meta, int chunkCapacity)
    {
        _scanInit(1, file, fileLength, meta);
        // One iteration per chunk plus the one that ends the walk (IEND, an error, or a full table).
        int steps = (MaxChunks(fileLength, chunkCapacity) + 1 + ScanChunks - 1) / ScanChunks;
        for (int i = 0; i < steps; i++) _scan(1, file, fileLength, chunks, idat, meta, chunkCapacity);
        _scanFinish(1, meta);
    }

    /// <summary>
    /// CRC check of the scanned chunks of a file of <paramref name="fileLength"/> bytes: piece CRCs in parallel into
    /// <paramref name="pieces"/> (at least <see cref="MaxCrcPieces"/> uints), then bounded combine launches per chunk.
    /// <paramref name="crcTable"/> is <see cref="Crc32Gpu.BuildTableWithShift"/>. Surplus threads exit at once.
    /// </summary>
    public void CheckCrc(
        ArrayView<byte> file, int fileLength, ArrayView<int> chunks, ArrayView<uint> crcTable, ArrayView<uint> pieces,
        ArrayView<int> meta, int chunkCapacity)
    {
        int maxChunks = MaxChunks(fileLength, chunkCapacity);
        long maxPieces = MaxCrcPieces(fileLength, chunkCapacity);
        _crcPiece((int)maxPieces, file, chunks, crcTable, pieces, meta);
        int steps = Crc32Gpu.CombineSteps(Crc32Gpu.PiecesOf(Math.Max(1, fileLength)));
        for (int s = 0; s < steps; s++) _crcCombine(maxChunks, file, chunks, crcTable, pieces, meta, s);
    }

    /// <summary>Concatenate the IDAT payloads into <paramref name="zlib"/> (<paramref name="idatTotal"/> bytes).</summary>
    public void GatherIdat(ArrayView<byte> file, ArrayView<int> idat, ArrayView<byte> zlib, ArrayView<int> meta, int idatTotal)
    {
        if (idatTotal > 0) _gather(idatTotal, file, idat, zlib, meta);
    }

    /// <summary>
    /// Reverse the scanline filters of the first <paramref name="rawSize"/> bytes of <paramref name="raw"/> in place
    /// (<paramref name="filterBpp"/> = <see cref="PngMeta.FilterBytesPerPixel"/>): launches of
    /// <see cref="UnfilterLaneBytes"/> bytes per lane, in order.
    /// </summary>
    public async ValueTask UnfilterAsync(ArrayView<byte> raw, ArrayView<int> meta, long rawSize, int filterBpp)
    {
        long chunk = (long)UnfilterLaneBytes * filterBpp;
        for (long start = 0; start < rawSize; start += chunk)
        {
            _unfilter(filterBpp, raw, meta, (int)start, (int)Math.Min(rawSize, start + chunk));
            await _pacer.LaunchedAsync();
        }
    }

    /// <summary>Convert the unfiltered samples to RGBA8 / RGBA16 in <paramref name="output"/>.</summary>
    public void ToRgba(ArrayView<byte> file, ArrayView<byte> raw, ArrayView<byte> output, ArrayView<int> meta, int pixelCount)
    {
        _toRgba(pixelCount, file, raw, output, meta);
    }

    // ------------------------------------------------------------------
    // 1. Chunk scan (resumable)
    // ------------------------------------------------------------------

    private static uint Be32(ArrayView<byte> f, int p)
    {
        return ((uint)f[p] << 24) | ((uint)f[p + 1] << 16) | ((uint)f[p + 2] << 8) | f[p + 3];
    }

    private static void ScanInitKernel(Index1D _, ArrayView<byte> file, int fileLen, ArrayView<int> meta)
    {
        for (int i = 0; i < PngMeta.Size; i++) meta[i] = 0;
        meta[PngMeta.PlteOffset] = -1;
        meta[PngMeta.TrnsOffset] = -1;

        int bad = 0;
        if (fileLen < 8)
        {
            bad = 1;
        }
        else
        {
            bad |= file[0] ^ 0x89;
            bad |= file[1] ^ 0x50;
            bad |= file[2] ^ 0x4E;
            bad |= file[3] ^ 0x47;
            bad |= file[4] ^ 0x0D;
            bad |= file[5] ^ 0x0A;
            bad |= file[6] ^ 0x1A;
            bad |= file[7] ^ 0x0A;
        }
        if (bad != 0) meta[PngMeta.Error] = (int)PngError.BadSignature;
        meta[PngMeta.ScanPos] = 8;
    }

    /// <summary>
    /// Walks up to <see cref="ScanChunks"/> chunks from <see cref="PngMeta.ScanPos"/>; the running counts live in their
    /// meta slots and the error in <see cref="PngMeta.Error"/>. A launch after IEND or an error does nothing.
    /// </summary>
    private static void ScanKernel(
        Index1D _, ArrayView<byte> file, int fileLen,
        ArrayView<int> chunks, ArrayView<int> idat, ArrayView<int> meta, int chunkCap)
    {
        int err = meta[PngMeta.Error];
        int done = meta[PngMeta.ScanDone];
        int pos = meta[PngMeta.ScanPos];
        int count = meta[PngMeta.ChunkCount];
        int idatCount = meta[PngMeta.IdatCount];
        int idatTotal = meta[PngMeta.IdatTotal];
        int pieces = meta[PngMeta.CrcPieces];
        int walked = 0;
        bool scanning = (done == 0) & (err == 0);
        while (scanning)
        {
            if (pos + 12 > fileLen)
            {
                err = (int)PngError.Truncated;
            }
            else
            {
                uint len = Be32(file, pos);
                uint type = Be32(file, pos + 4);
                if (len > (uint)fileLen)
                {
                    err = (int)PngError.Truncated;
                }
                else if (pos + 12 + (int)len > fileLen)
                {
                    err = (int)PngError.Truncated;
                }
                else if (count >= chunkCap)
                {
                    err = (int)PngError.TooManyChunks;
                }
                else
                {
                    int dataOff = pos + 8;
                    chunks[count * 4 + 0] = (int)type;
                    chunks[count * 4 + 1] = dataOff;
                    chunks[count * 4 + 2] = (int)len;
                    // The CRC covers the 4 type bytes and the data, checksummed in pieces from this index on.
                    chunks[count * 4 + 3] = pieces;
                    pieces += Crc32Gpu.PiecesOf((int)len + 4);
                    count++;

                    if (count == 1)
                    {
                        bool badIhdr = (type != TypeIhdr) | (len != 13);
                        if (badIhdr)
                        {
                            err = (int)PngError.BadHeader;
                        }
                        else
                        {
                            meta[PngMeta.Width] = (int)Be32(file, dataOff);
                            meta[PngMeta.Height] = (int)Be32(file, dataOff + 4);
                            meta[PngMeta.BitDepth] = file[dataOff + 8];
                            meta[PngMeta.ColorType] = file[dataOff + 9];
                            meta[PngMeta.Compression] = file[dataOff + 10];
                            meta[PngMeta.FilterMethod] = file[dataOff + 11];
                            meta[PngMeta.Interlace] = file[dataOff + 12];
                        }
                    }
                    else if (type == TypeIdat)
                    {
                        idat[idatCount * 3 + 0] = dataOff;
                        idat[idatCount * 3 + 1] = idatTotal;
                        idat[idatCount * 3 + 2] = (int)len;
                        idatCount++;
                        idatTotal += (int)len;
                    }
                    else if (type == TypePlte)
                    {
                        meta[PngMeta.PlteOffset] = dataOff;
                        meta[PngMeta.PlteLength] = (int)len;
                    }
                    else if (type == TypeTrns)
                    {
                        meta[PngMeta.TrnsOffset] = dataOff;
                        meta[PngMeta.TrnsLength] = (int)len;
                    }
                    else if (type == TypeActl)
                    {
                        if (len == 8)
                        {
                            meta[PngMeta.ApngFrames] = (int)Be32(file, dataOff);
                            meta[PngMeta.ApngPlays] = (int)Be32(file, dataOff + 4);
                        }
                    }
                    else if (type == TypeSrgb)
                    {
                        meta[PngMeta.HasSrgb] = 1;
                    }
                    else if (type == TypeIend)
                    {
                        meta[PngMeta.IendSeen] = 1;
                        done = 1;
                    }
                    pos += 12 + (int)len;
                }
            }
            walked++;
            scanning = (done == 0) & (err == 0) & (walked < ScanChunks);
        }

        meta[PngMeta.Error] = err;
        meta[PngMeta.ScanDone] = done;
        meta[PngMeta.ScanPos] = pos;
        meta[PngMeta.ChunkCount] = count;
        meta[PngMeta.IdatTotal] = idatTotal;
        meta[PngMeta.IdatCount] = idatCount;
        meta[PngMeta.CrcPieces] = pieces;
    }

    private static void ScanFinishKernel(Index1D _, ArrayView<int> meta)
    {
        bool missingData = (meta[PngMeta.Error] == 0) & (meta[PngMeta.IdatCount] == 0);
        if (missingData) meta[PngMeta.Error] = (int)PngError.NoImageData;
    }

    // ------------------------------------------------------------------
    // 2. CRC verification (critical chunks only; ancillary CRC errors are ignored like libpng's default)
    // ------------------------------------------------------------------

    /// <summary>Thread <c>g</c>: the CRC of piece g (pieces of all chunks numbered in chunk order, see the scan).</summary>
    private static void CrcPieceKernel(
        Index1D index, ArrayView<byte> file, ArrayView<int> chunks, ArrayView<uint> table, ArrayView<uint> pieces,
        ArrayView<int> meta)
    {
        int g = index.X;
        if (g >= meta[PngMeta.CrcPieces]) return;
        // Largest chunk k whose first piece is <= g (every chunk has at least one piece).
        int lo = 0;
        int hi = meta[PngMeta.ChunkCount] - 1;
        while (lo < hi)
        {
            int mid = (lo + hi + 1) >> 1;
            if (chunks[mid * 4 + 3] <= g) lo = mid; else hi = mid - 1;
        }
        int dataOff = chunks[lo * 4 + 1];
        int len = chunks[lo * 4 + 2];
        pieces[g] = Crc32Gpu.PieceOfRange(file, dataOff - 4, len + 4, g - chunks[lo * 4 + 3], table);
    }

    /// <summary>Thread <c>k</c>, combine launch <paramref name="step"/>: folds chunk k's pieces; the launch that folds its last one verifies it.</summary>
    private static void CrcCombineKernel(
        Index1D index, ArrayView<byte> file, ArrayView<int> chunks, ArrayView<uint> table, ArrayView<uint> pieces,
        ArrayView<int> meta, int step)
    {
        int i = index.X;
        if (i >= meta[PngMeta.ChunkCount]) return;
        uint type = (uint)chunks[i * 4 + 0];
        int dataOff = chunks[i * 4 + 1];
        int len = chunks[i * 4 + 2];
        int at = chunks[i * 4 + 3];
        int finished = Crc32Gpu.CombineStep(pieces, at, Crc32Gpu.PiecesOf(len + 4), step, table);
        if (finished != 0)
        {
            uint crc = Crc32Gpu.Finish(pieces[at]);
            uint stored = Be32(file, dataOff + len);
            bool critical = ((type >> 24) & 0x20u) == 0;
            bool mismatch = crc != stored;
            if (mismatch & critical)
                meta[PngMeta.BadCrcChunk] = i + 1;
        }
    }

    // ------------------------------------------------------------------
    // 3. IDAT gather
    // ------------------------------------------------------------------

    private static void GatherKernel(
        Index1D index, ArrayView<byte> file, ArrayView<int> idat, ArrayView<byte> zlib, ArrayView<int> meta)
    {
        int i = index.X;
        // Largest chunk k whose destination start is <= i (empty chunks share a start; the later,
        // non-empty one wins).
        int lo = 0;
        int hi = meta[PngMeta.IdatCount] - 1;
        while (lo < hi)
        {
            int mid = (lo + hi + 1) >> 1;
            if (idat[mid * 3 + 1] <= i) lo = mid; else hi = mid - 1;
        }
        zlib[i] = file[idat[lo * 3 + 0] + (i - idat[lo * 3 + 1])];
    }

    // ------------------------------------------------------------------
    // 5. Unfilter (in place; the filter byte of each scanline stays where it is)
    // ------------------------------------------------------------------

    private static int Paeth(int a, int b, int c)
    {
        int p = a + b - c;
        int pa = p - a; if (pa < 0) pa = -pa;
        int pb = p - b; if (pb < 0) pb = -pb;
        int pc = p - c; if (pc < 0) pc = -pc;
        bool useA = (pa <= pb) & (pa <= pc);
        bool useB = pb <= pc;
        int r = c;
        if (useA) r = a;
        else if (useB) r = b;
        return r;
    }

    /// <summary>
    /// Lane <c>index</c> (0 .. bpp-1) reverses the filters of the bytes in [<paramref name="start"/>,
    /// <paramref name="end"/>) of the inflated buffer whose scanline byte index is congruent to the lane mod bpp. Every
    /// byte a filter reads (left: i - bpp, up: the previous scanline's i, up-left: its i - bpp) is in the same lane, so
    /// the lanes never read each other, and each lane walks its bytes in scanline order: launches over consecutive
    /// ranges reproduce the serial unfilter exactly.
    /// </summary>
    private static void UnfilterKernel(Index1D index, ArrayView<byte> raw, ArrayView<int> meta, int start, int end)
    {
        int lane = index.X;
        int passCount = meta[PngMeta.PassCount];
        int bpp = meta[PngMeta.FilterBytesPerPixel];
        int badFilter = 0;
        int pos = start;
        bool going = pos < end;
        while (going)
        {
            // The pass holding pos (passes without pixels own no bytes).
            int pass = 0;
            for (int p = 0; p < passCount; p++)
            {
                int ep = PngMeta.PassTable + p * PngMeta.PassStride;
                int poff = meta[ep + PngMeta.PassOffset];
                int prows = meta[ep + PngMeta.PassHeight];
                int psize = prows * (meta[ep + PngMeta.PassRowBytes] + 1);
                bool holds = (prows > 0) & (meta[ep + PngMeta.PassWidth] > 0) & (pos >= poff) & (pos < poff + psize);
                if (holds) pass = p;
            }
            int e = PngMeta.PassTable + pass * PngMeta.PassStride;
            int off = meta[e + PngMeta.PassOffset];
            int rb = meta[e + PngMeta.PassRowBytes];
            int stride = rb + 1;
            int y = (pos - off) / stride;
            int rowStart = off + y * stride;
            int segEnd = rowStart + stride;
            if (segEnd > end) segEnd = end;
            int i0 = pos - rowStart - 1;
            if (i0 < 0) i0 = 0;
            int i1 = segEnd - rowStart - 1;

            int ft = raw[rowStart];
            int cur = rowStart + 1;
            int prev = cur - stride;
            if (ft > 4)
            {
                badFilter = 1;
            }
            else if (ft != 0)
            {
                // First byte index >= i0 of this lane.
                int i = i0 + (lane - i0 % bpp + bpp) % bpp;
                for (; i < i1; i += bpp)
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
                    int v;
                    if (ft == 1) v = x + a;
                    else if (ft == 2) v = x + b;
                    else if (ft == 3) v = x + ((a + b) >> 1);
                    else v = x + Paeth(a, b, c);
                    raw[cur + i] = (byte)v;
                }
            }
            pos = segEnd;
            going = pos < end;
        }
        bool report = (badFilter != 0) & (lane == 0);
        if (report) meta[PngMeta.Error] = (int)PngError.BadFilter;
    }

    // ------------------------------------------------------------------
    // 6. Samples -> RGBA
    // ------------------------------------------------------------------

    /// <summary>Sample <paramref name="k"/> of pixel <paramref name="px"/> in the scanline starting at <paramref name="rowData"/>.</summary>
    private static int SampleAt(ArrayView<byte> raw, int rowData, int px, int k, int channels, int depth)
    {
        int bitOffset = (px * channels + k) * depth;
        int byteIndex = rowData + (bitOffset >> 3);
        int v;
        if (depth == 16)
        {
            v = (raw[byteIndex] << 8) | raw[byteIndex + 1];
        }
        else if (depth == 8)
        {
            v = raw[byteIndex];
        }
        else
        {
            int shift = 8 - depth - (bitOffset & 7);
            v = (raw[byteIndex] >> shift) & ((1 << depth) - 1);
        }
        return v;
    }

    private static int To8(int v, int depth)
    {
        int r;
        if (depth == 16) r = v >> 8;
        else if (depth == 8) r = v;
        else r = v * (255 / ((1 << depth) - 1));
        return r;
    }

    private static int To16(int v, int depth)
    {
        int r;
        if (depth == 16) r = v;
        else if (depth == 8) r = v * 257;
        else r = v * (65535 / ((1 << depth) - 1));
        return r;
    }

    /// <summary>Adam7 pass (0..6) containing image pixel (x, y); 0 when not interlaced.</summary>
    private static int AdamPass(int x, int y)
    {
        // Each membership test is a bool local built with non-short-circuit &: the WGSL codegen has
        // dropped threads on compound predicates written inline in a branch condition.
        bool p0 = ((x & 7) == 0) & ((y & 7) == 0);
        bool p1 = ((x & 7) == 4) & ((y & 7) == 0);
        bool p2 = ((x & 3) == 0) & ((y & 7) == 4);
        bool p3 = ((x & 3) == 2) & ((y & 3) == 0);
        bool p4 = ((x & 1) == 0) & ((y & 3) == 2);
        bool p5 = ((x & 1) == 1) & ((y & 1) == 0);
        int pass = 6;
        if (p5) pass = 5;
        if (p4) pass = 4;
        if (p3) pass = 3;
        if (p2) pass = 2;
        if (p1) pass = 1;
        if (p0) pass = 0;
        return pass;
    }

    private static void ToRgbaKernel(
        Index1D index, ArrayView<byte> file, ArrayView<byte> raw, ArrayView<byte> output, ArrayView<int> meta)
    {
        int width = meta[PngMeta.Width];
        int depth = meta[PngMeta.BitDepth];
        int colorType = meta[PngMeta.ColorType];
        int interlace = meta[PngMeta.Interlace];
        int outFormat = meta[PngMeta.OutputFormat];
        int i = index.X;
        int y = i / width;
        int x = i - y * width;

        int pass = 0;
        if (interlace != 0) pass = AdamPass(x, y);
        int e = PngMeta.PassTable + pass * PngMeta.PassStride;
        int px = (x - meta[e + PngMeta.PassXStart]) / meta[e + PngMeta.PassXStep];
        int py = (y - meta[e + PngMeta.PassYStart]) / meta[e + PngMeta.PassYStep];
        int rb = meta[e + PngMeta.PassRowBytes];
        int rowData = meta[e + PngMeta.PassOffset] + py * (rb + 1) + 1;

        int channels = 1;
        if (colorType == 2) channels = 3;
        else if (colorType == 4) channels = 2;
        else if (colorType == 6) channels = 4;
        int maxSample = depth == 16 ? 65535 : (1 << depth) - 1;

        int r = 0, g = 0, b = 0;
        int a = maxSample;
        if (colorType == 3)
        {
            // Palette: always expressed in 8 bits, scaled to 16 below.
            int idx = SampleAt(raw, rowData, px, 0, 1, depth);
            int plteOff = meta[PngMeta.PlteOffset];
            int plteLen = meta[PngMeta.PlteLength];
            int entry = idx * 3;
            bool inPalette = (plteOff >= 0) & (entry + 2 < plteLen);
            if (inPalette)
            {
                r = file[plteOff + entry];
                g = file[plteOff + entry + 1];
                b = file[plteOff + entry + 2];
            }
            a = 255;
            int trnsOff = meta[PngMeta.TrnsOffset];
            bool hasAlphaEntry = (trnsOff >= 0) & (idx < meta[PngMeta.TrnsLength]);
            if (hasAlphaEntry) a = file[trnsOff + idx];
            if (outFormat == 1) { r *= 257; g *= 257; b *= 257; a *= 257; }
        }
        else
        {
            int sr, sg, sb;
            bool grayish = (colorType == 0) | (colorType == 4);
            if (grayish)
            {
                sr = SampleAt(raw, rowData, px, 0, channels, depth);
                sg = sr;
                sb = sr;
                if (colorType == 4) a = SampleAt(raw, rowData, px, 1, channels, depth);
            }
            else
            {
                sr = SampleAt(raw, rowData, px, 0, channels, depth);
                sg = SampleAt(raw, rowData, px, 1, channels, depth);
                sb = SampleAt(raw, rowData, px, 2, channels, depth);
                if (colorType == 6) a = SampleAt(raw, rowData, px, 3, channels, depth);
            }
            bool maybeKeyed = (colorType == 0) | (colorType == 2);
            if (maybeKeyed)
            {
                if (meta[PngMeta.HasColorKey] != 0)
                {
                    // tRNS for gray / RGB is a colour key: 16-bit big-endian samples, only the low
                    // 'depth' bits are significant.
                    int k = meta[PngMeta.TrnsOffset];
                    int kr = ((file[k] << 8) | file[k + 1]) & maxSample;
                    int kg = kr;
                    int kb = kr;
                    if (colorType == 2)
                    {
                        kg = ((file[k + 2] << 8) | file[k + 3]) & maxSample;
                        kb = ((file[k + 4] << 8) | file[k + 5]) & maxSample;
                    }
                    bool keyed = (sr == kr) & (sg == kg) & (sb == kb);
                    if (keyed) a = 0;
                }
            }
            if (outFormat == 1)
            {
                r = To16(sr, depth); g = To16(sg, depth); b = To16(sb, depth); a = To16(a, depth);
            }
            else
            {
                r = To8(sr, depth); g = To8(sg, depth); b = To8(sb, depth); a = To8(a, depth);
            }
        }

        if (outFormat == 1)
        {
            int o = i * 8;
            output[o + 0] = (byte)(r & 0xFF); output[o + 1] = (byte)(r >> 8);
            output[o + 2] = (byte)(g & 0xFF); output[o + 3] = (byte)(g >> 8);
            output[o + 4] = (byte)(b & 0xFF); output[o + 5] = (byte)(b >> 8);
            output[o + 6] = (byte)(a & 0xFF); output[o + 7] = (byte)(a >> 8);
        }
        else
        {
            int o = i * 4;
            output[o + 0] = (byte)r;
            output[o + 1] = (byte)g;
            output[o + 2] = (byte)b;
            output[o + 3] = (byte)a;
        }
    }

    /// <summary>No-op (kernels are owned by the accelerator).</summary>
    public void Dispose() { }
}
