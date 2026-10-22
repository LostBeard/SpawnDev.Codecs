// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// The JPEG decode kernels. Every stage runs on the accelerator; the host only dispatches and reads back a few
// scalars (the metadata block):
//
//   1. ScanMarkers  (1 thread)         walks the marker segments: frame header, quantization / Huffman tables, restart
//                                      interval, JFIF / Adobe / EXIF facts, one scan-table entry per SOS (including the
//                                      end of its entropy-coded data). Quantization tables are latched per component at
//                                      the component's first scan, Huffman tables are snapshotted per scan.
//   2. ClearCoef    (1 thread / coef)  zeroes the coefficient buffer.
//   per scan:
//   3. BuildHuff    (8 threads)        raw DHT tables -> derived decode tables (9-bit lookahead + canonical maxcode).
//   4. FindRestarts (1 thread)         locates the RSTn markers of the scan: start / end of every restart segment.
//   5. DecodeScan   (1 thread / restart segment, chunked) Huffman decode of baseline / extended / progressive scans
//                                      into the coefficient buffer. Each dispatch advances a bounded number of MCUs and
//                                      saves its bit-reader state, so no single launch runs for long (GPU watchdogs).
//   6. Idct         (1 thread / block) dequantize + libjpeg ISLOW integer IDCT (jidctint.c) -> sample planes.
//   7. ColorConvert (1 thread / pixel) libjpeg "fancy" upsampling (h2v1, h1v2, h2v2 triangle filters, replication for
//                                      other integral ratios) + libjpeg fixed-point YCbCr / YCCK colour conversion ->
//                                      RGBA8 / RGBA16.
//   8. Orient       (1 thread / pixel) optional EXIF orientation transform.
//
// The arithmetic mirrors libjpeg-turbo (jidctint.c, jdsample.c, jdcolor.c) so the output is bit-exact with it.

using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.Codecs.Images.Jpeg;

/// <summary>Loads and launches the JPEG decode kernels.</summary>
public sealed class JpegKernels : IDisposable
{
    private readonly Action<Index1D, ArrayView<byte>, int, ArrayView<int>, ArrayView<int>, ArrayView<int>, int> _scan;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, int> _buildHuff;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, int, int> _findRestarts;
    private readonly Action<Index1D, ArrayView<short>> _clearCoef;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<short>, ArrayView<int>, ArrayView<int>, ArrayView<int>,
        ArrayView<int>, ArrayView<int>, int, int, int> _decodeScan;
    private readonly Action<Index1D, ArrayView<short>, ArrayView<byte>, ArrayView<int>> _idct;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<byte>, int> _color;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, int, int, int, int> _orient;

    /// <summary>Compile the kernels.</summary>
    public JpegKernels(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _scan = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, int, ArrayView<int>, ArrayView<int>, ArrayView<int>, int>(ScanMarkersKernel);
        _buildHuff = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, int>(BuildHuffKernel);
        _findRestarts = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, int, int>(FindRestartsKernel);
        _clearCoef = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<short>>(ClearCoefKernel);
        _decodeScan = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<short>, ArrayView<int>, ArrayView<int>, ArrayView<int>,
            ArrayView<int>, ArrayView<int>, int, int, int>(DecodeScanKernel);
        _idct = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<short>, ArrayView<byte>, ArrayView<int>>(IdctKernel);
        _color = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<byte>, int>(ColorKernel);
        _orient = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<byte>, int, int, int, int>(OrientKernel);
    }

    /// <summary>Marker scan; fills <paramref name="meta"/> and the Huffman set table. <paramref name="headerOnly"/> stops at the first SOS.</summary>
    public void ScanMarkers(ArrayView<byte> file, int fileLength, ArrayView<int> meta, ArrayView<int> huffRaw,
        ArrayView<int> consts, bool headerOnly)
    {
        _scan(1, file, fileLength, meta, huffRaw, consts, headerOnly ? 1 : 0);
    }

    /// <summary>Derive the Huffman decode tables of scan <paramref name="scan"/> into <paramref name="huff"/>.</summary>
    public void BuildHuff(ArrayView<int> meta, ArrayView<int> huffRaw, ArrayView<int> huff, int scan)
    {
        _buildHuff(8, meta, huffRaw, huff, scan);
    }

    /// <summary>Locate the <paramref name="segmentCount"/> restart segments of a scan.</summary>
    public void FindRestarts(ArrayView<byte> file, ArrayView<int> meta, ArrayView<int> segs, int scan, int segmentCount)
    {
        _findRestarts(1, file, meta, segs, scan, segmentCount);
    }

    /// <summary>Zero the first <paramref name="count"/> coefficients.</summary>
    public void ClearCoef(ArrayView<short> coef, int count)
    {
        if (count > 0) _clearCoef(count, coef);
    }

    /// <summary>
    /// Advance every restart segment of a scan by up to <paramref name="budget"/> MCUs (round 0 initialises the
    /// per-segment state, later rounds resume it).
    /// </summary>
    public void DecodeScan(ArrayView<byte> file, ArrayView<short> coef, ArrayView<int> meta, ArrayView<int> huff,
        ArrayView<int> segs, ArrayView<int> consts, ArrayView<int> state, int scan, int segmentCount, int budget, int round)
    {
        _decodeScan(segmentCount, file, coef, meta, huff, segs, consts, state, scan, budget, round);
    }

    /// <summary>Dequantize + inverse DCT every block into the sample planes.</summary>
    public void Idct(ArrayView<short> coef, ArrayView<byte> planes, ArrayView<int> meta, int totalBlocks)
    {
        _idct(totalBlocks, coef, planes, meta);
    }

    /// <summary>Upsample + colour convert to RGBA.</summary>
    public void ColorConvert(ArrayView<byte> planes, ArrayView<int> meta, ArrayView<byte> output, int width, int pixelCount)
    {
        _color(pixelCount, planes, meta, output, width);
    }

    /// <summary>Apply an EXIF orientation (1..8) to a tightly packed image of <paramref name="bytesPerPixel"/> bytes per pixel.</summary>
    public void Orient(ArrayView<byte> src, ArrayView<byte> dst, int width, int height, int orientation, int bytesPerPixel)
    {
        _orient(width * height, src, dst, width, height, orientation, bytesPerPixel);
    }

    // =================================================================================================================
    // 1. Marker scan
    // =================================================================================================================

    private static int Be16(ArrayView<byte> f, int p) => (f[p] << 8) | f[p + 1];

    private static int Rd16(ArrayView<byte> f, int p, int le)
    {
        int v = (f[p] << 8) | f[p + 1];
        if (le != 0) v = (f[p + 1] << 8) | f[p];
        return v;
    }

    private static int Rd32(ArrayView<byte> f, int p, int le)
    {
        int v = (f[p] << 24) | (f[p + 1] << 16) | (f[p + 2] << 8) | f[p + 3];
        if (le != 0) v = (f[p + 3] << 24) | (f[p + 2] << 16) | (f[p + 1] << 8) | f[p];
        return v;
    }

    /// <summary>EXIF orientation (1..8) from an APP1 payload at <paramref name="p"/>, or 0 when absent / invalid.</summary>
    private static int ParseExifOrientation(ArrayView<byte> f, int p, int n)
    {
        int result = 0;
        if (n >= 14)
        {
            bool hdr = (f[p] == 'E') & (f[p + 1] == 'x') & (f[p + 2] == 'i') & (f[p + 3] == 'f')
                     & (f[p + 4] == 0) & (f[p + 5] == 0);
            int t = p + 6;
            bool isLe = (f[t] == 'I') & (f[t + 1] == 'I');
            bool isBe = (f[t] == 'M') & (f[t + 1] == 'M');
            bool known = hdr & (isLe | isBe);
            if (known)
            {
                int le = isLe ? 1 : 0;
                int magic = Rd16(f, t + 2, le);
                int ifd = Rd32(f, t + 4, le);
                bool sane = (magic == 42) & (ifd >= 8) & (ifd < n - 8);
                if (sane)
                {
                    int count = Rd16(f, t + ifd, le);
                    int e = t + ifd + 2;
                    int end = p + n;
                    for (int i = 0; i < count; i++)
                    {
                        if (e + 12 <= end)
                        {
                            int tag = Rd16(f, e, le);
                            if (tag == 0x0112)
                            {
                                int type = Rd16(f, e + 2, le);
                                int cnt = Rd32(f, e + 4, le);
                                int v = Rd16(f, e + 8, le);
                                bool ok = (type == 3) & (cnt == 1) & (v >= 1) & (v <= 8);
                                bool take = ok & (result == 0);
                                if (take) result = v;
                            }
                        }
                        e += 12;
                    }
                }
            }
        }
        return result;
    }

    /// <summary>
    /// Offset of the first marker after entropy-coded data starting at <paramref name="start"/> (byte stuffing, fill
    /// bytes and RSTn markers are skipped; RSTn markers are counted).
    /// </summary>
    private static int FindEntropyEnd(ArrayView<byte> f, int start, int fileLen, ref int rstCount)
    {
        int p = start;
        int result = -1;
        bool go1 = (result < 0) & (p < fileLen);
        while (go1)
        {
            if (f[p] != 0xFF)
            {
                p++;
            }
            else if (p + 1 >= fileLen)
            {
                result = fileLen;
            }
            else
            {
                int n = f[p + 1];
                bool isRst = (n >= 0xD0) & (n <= 0xD7);
                if (n == 0) p += 2;
                else if (n == 0xFF) p += 1;
                else if (isRst) { rstCount++; p += 2; }
                else result = p;
            }
            go1 = (result < 0) & (p < fileLen);
        }
        if (result < 0) result = fileLen;
        return result;
    }

    private static int ParseSof(ArrayView<byte> f, int p, int n, ArrayView<int> meta, int marker)
    {
        int err = 0;
        if (meta[JpegMeta.SofSeen] != 0)
        {
            err = (int)JpegError.BadFrameHeader;
            meta[JpegMeta.ErrorDetail] = 3;       // second frame header
        }
        else if (n < 6)
        {
            err = (int)JpegError.BadFrameHeader;
        }
        else
        {
            int prec = f[p];
            int height = Be16(f, p + 1);
            int width = Be16(f, p + 3);
            int nc = f[p + 5];
            bool badCompCount = (nc != 1) & (nc != 3) & (nc != 4);
            if (prec != 8)
            {
                err = (int)JpegError.UnsupportedPrecision;
                meta[JpegMeta.ErrorDetail] = prec;
            }
            else if (n != 6 + 3 * nc)
            {
                err = (int)JpegError.BadFrameHeader;
            }
            else if (badCompCount)
            {
                err = (int)JpegError.BadFrameHeader;
                meta[JpegMeta.ErrorDetail] = 10 + nc;     // unsupported component count
            }
            else if (width == 0)
            {
                err = (int)JpegError.BadFrameHeader;
                meta[JpegMeta.ErrorDetail] = 1;
            }
            else if (height == 0)
            {
                err = (int)JpegError.BadFrameHeader;
                meta[JpegMeta.ErrorDetail] = 2;
            }
            else
            {
                int maxH = 1;
                int maxV = 1;
                for (int i = 0; i < nc; i++)
                {
                    int q = p + 6 + 3 * i;
                    int id = f[q];
                    int h = f[q + 1] >> 4;
                    int v = f[q + 1] & 15;
                    int tq = f[q + 2];
                    bool bad = (h < 1) | (h > 4) | (v < 1) | (v > 4) | (tq > 3);
                    if (bad)
                    {
                        err = (int)JpegError.BadFrameHeader;
                        meta[JpegMeta.ErrorDetail] = 4;      // bad sampling factors / table selector
                    }
                    for (int j = 0; j < i; j++)
                    {
                        if (meta[JpegMeta.CompIn + j * JpegMeta.CompInStride + JpegMeta.CiId] == id)
                        {
                            err = (int)JpegError.BadFrameHeader;
                            meta[JpegMeta.ErrorDetail] = 5;  // duplicate component id
                        }
                    }
                    int c = JpegMeta.CompIn + i * JpegMeta.CompInStride;
                    meta[c + JpegMeta.CiId] = id;
                    meta[c + JpegMeta.CiH] = h;
                    meta[c + JpegMeta.CiV] = v;
                    meta[c + JpegMeta.CiTq] = tq;
                    meta[c + JpegMeta.CiLatched] = 0;
                    if (h > maxH) maxH = h;
                    if (v > maxV) maxV = v;
                }
                meta[JpegMeta.Width] = width;
                meta[JpegMeta.Height] = height;
                meta[JpegMeta.Precision] = prec;
                meta[JpegMeta.NumComps] = nc;
                meta[JpegMeta.MaxH] = maxH;
                meta[JpegMeta.MaxV] = maxV;
                meta[JpegMeta.SofSeen] = 1;
                meta[JpegMeta.FirstCompId] = f[p + 6];
                int sofType = 0;
                if (marker == 0xC1) sofType = 1;
                if (marker == 0xC2) sofType = 2;
                meta[JpegMeta.SofType] = sofType;
            }
        }
        return err;
    }

    private static int ParseDqt(ArrayView<byte> f, int p, int n, ArrayView<int> meta, ArrayView<int> consts)
    {
        int err = 0;
        int q = p;
        int end = p + n;
        bool go2 = (q < end) & (err == 0);
        while (go2)
        {
            int pt = f[q];
            int pq = pt >> 4;
            int tq = pt & 15;
            int need = 64 * (pq + 1);
            bool bad = (pq > 1) | (tq > 3) | (q + 1 + need > end);
            if (bad)
            {
                err = (int)JpegError.BadQuantTable;
            }
            else
            {
                for (int i = 0; i < 64; i++)
                {
                    int v = f[q + 1 + i];
                    if (pq != 0) v = (f[q + 1 + 2 * i] << 8) | f[q + 2 + 2 * i];
                    meta[JpegMeta.QuantWork + tq * 64 + consts[i]] = v;
                }
                meta[JpegMeta.QuantDefined + tq] = 1;
                q += 1 + need;
            }
            go2 = (q < end) & (err == 0);
        }
        return err;
    }

    private static int ParseDht(ArrayView<byte> f, int p, int n, ArrayView<int> meta, ArrayView<int> huffRaw)
    {
        int err = 0;
        int q = p;
        int end = p + n;
        int work = JpegMeta.MaxHuffSets * JpegMeta.RawSetStride;
        bool go3 = (q < end) & (err == 0);
        while (go3)
        {
            if (q + 17 > end)
            {
                err = (int)JpegError.BadHuffmanTable;
            }
            else
            {
                int tc = f[q] >> 4;
                int th = f[q] & 15;
                int total = 0;
                for (int i = 0; i < 16; i++) total += f[q + 1 + i];
                bool bad = (tc > 1) | (th > 3) | (total > 256) | (q + 17 + total > end);
                if (bad)
                {
                    err = (int)JpegError.BadHuffmanTable;
                }
                else
                {
                    int slot = tc * 4 + th;
                    int b = work + slot * JpegMeta.RawTableStride;
                    huffRaw[b] = 1;
                    for (int i = 0; i < 16; i++) huffRaw[b + 1 + i] = f[q + 1 + i];
                    for (int i = 0; i < total; i++) huffRaw[b + 17 + i] = f[q + 17 + i];
                    meta[JpegMeta.HuffDirty] = 1;
                    q += 17 + total;
                }
            }
            go3 = (q < end) & (err == 0);
        }
        return err;
    }

    private static int ParseSos(ArrayView<byte> f, int p, int n, int fileLen, ArrayView<int> meta, ArrayView<int> huffRaw)
    {
        int err = 0;
        if (meta[JpegMeta.SofSeen] == 0)
        {
            err = (int)JpegError.NoFrame;
        }
        else
        {
            int nc = meta[JpegMeta.NumComps];
            int ns = f[p];
            int sc = meta[JpegMeta.ScanCount];
            bool c4 = (ns < 1) | (ns > 4) | (ns > nc) | (n != 4 + 2 * ns);
            if (c4)
            {
                err = (int)JpegError.BadScanHeader;
            }
            else if (sc >= JpegMeta.MaxScans)
            {
                err = (int)JpegError.TooManyScans;
            }
            else
            {
                int sb = JpegMeta.Scans + sc * JpegMeta.ScanStride;
                int ss = f[p + 1 + 2 * ns];
                int se = f[p + 2 + 2 * ns];
                int ah = f[p + 3 + 2 * ns] >> 4;
                int al = f[p + 3 + 2 * ns] & 15;
                int sofType = meta[JpegMeta.SofType];
                int blocksInMcu = 0;
                int dcNeeded = 0;
                int acNeeded = 0;
                meta[sb + JpegMeta.ScNs] = ns;
                for (int i = 0; i < ns; i++)
                {
                    int cs = f[p + 1 + 2 * i];
                    int tt = f[p + 2 + 2 * i];
                    int ci = -1;
                    for (int j = 0; j < nc; j++)
                    {
                        if (meta[JpegMeta.CompIn + j * JpegMeta.CompInStride + JpegMeta.CiId] == cs) ci = j;
                    }
                    if (ci < 0)
                    {
                        err = (int)JpegError.BadScanHeader;
                        meta[JpegMeta.ErrorDetail] = 1;       // unknown component selector
                    }
                    else
                    {
                        for (int j = 0; j < i; j++)
                        {
                            if (meta[sb + JpegMeta.ScComp + j] == ci)
                            {
                                err = (int)JpegError.BadScanHeader;
                                meta[JpegMeta.ErrorDetail] = 2;   // component twice in one scan
                            }
                        }
                        meta[sb + JpegMeta.ScComp + i] = ci;
                        meta[sb + JpegMeta.ScTables + i] = tt;
                        int ch = meta[JpegMeta.CompIn + ci * JpegMeta.CompInStride + JpegMeta.CiH];
                        int cv = meta[JpegMeta.CompIn + ci * JpegMeta.CompInStride + JpegMeta.CiV];
                        blocksInMcu += ch * cv;
                        bool c5 = ((tt >> 4) > 3) | ((tt & 15) > 3);
                        if (c5)
                        {
                            err = (int)JpegError.BadScanHeader;
                            meta[JpegMeta.ErrorDetail] = 3;       // Huffman table selector out of range
                        }
                    }
                }
                bool c6 = (err == 0) & (ns > 1) & (blocksInMcu > 10);
                if (c6)
                {
                    err = (int)JpegError.BadScanHeader;
                    meta[JpegMeta.ErrorDetail] = 4;               // MCU with more than 10 blocks
                }
                if (err == 0)
                {
                    bool badParams;
                    if (sofType == 2)
                    {
                        bool dcBand = ss == 0;
                        badParams = (al > 13);
                        if (dcBand) badParams = badParams | (se != 0);
                        else badParams = badParams | (ss > se) | (se > 63) | (ns != 1);
                        if (ah != 0) badParams = badParams | (al != ah - 1);
                        if (dcBand) { if (ah == 0) dcNeeded = 1; }
                        else acNeeded = 1;
                    }
                    else
                    {
                        badParams = (ss != 0) | (se != 63) | (ah != 0) | (al != 0);
                        dcNeeded = 1;
                        acNeeded = 1;
                    }
                    if (badParams)
                    {
                        err = (int)JpegError.BadScanHeader;
                        meta[JpegMeta.ErrorDetail] = 5;           // invalid spectral / approximation parameters
                    }
                }
                if (err == 0)
                {
                    int work = JpegMeta.MaxHuffSets * JpegMeta.RawSetStride;
                    for (int i = 0; i < ns; i++)
                    {
                        int ci = meta[sb + JpegMeta.ScComp + i];
                        int tt = meta[sb + JpegMeta.ScTables + i];
                        int c = JpegMeta.CompIn + ci * JpegMeta.CompInStride;
                        if (meta[c + JpegMeta.CiLatched] == 0)
                        {
                            int tq = meta[c + JpegMeta.CiTq];
                            if (meta[JpegMeta.QuantDefined + tq] == 0)
                            {
                                err = (int)JpegError.MissingQuantTable;
                                meta[JpegMeta.ErrorDetail] = tq;
                            }
                            else
                            {
                                for (int k = 0; k < 64; k++)
                                    meta[JpegMeta.Quant + ci * 64 + k] = meta[JpegMeta.QuantWork + tq * 64 + k];
                                meta[c + JpegMeta.CiLatched] = 1;
                            }
                        }
                        if (dcNeeded != 0)
                        {
                            int slot = tt >> 4;
                            if (huffRaw[work + slot * JpegMeta.RawTableStride] == 0)
                            {
                                err = (int)JpegError.MissingHuffmanTable;
                                meta[JpegMeta.ErrorDetail] = slot;
                            }
                        }
                        if (acNeeded != 0)
                        {
                            int slot = 4 + (tt & 15);
                            if (huffRaw[work + slot * JpegMeta.RawTableStride] == 0)
                            {
                                err = (int)JpegError.MissingHuffmanTable;
                                meta[JpegMeta.ErrorDetail] = slot;
                            }
                        }
                    }
                }
                if (err == 0)
                {
                    // Snapshot the working Huffman tables when they changed since the last scan.
                    int sets = meta[JpegMeta.HuffSetCount];
                    bool c7 = (meta[JpegMeta.HuffDirty] != 0) | (sets == 0);
                    if (c7)
                    {
                        if (sets >= JpegMeta.MaxHuffSets)
                        {
                            err = (int)JpegError.TooManyScans;
                        }
                        else
                        {
                            int work = JpegMeta.MaxHuffSets * JpegMeta.RawSetStride;
                            int dst = sets * JpegMeta.RawSetStride;
                            for (int i = 0; i < JpegMeta.RawSetStride; i++) huffRaw[dst + i] = huffRaw[work + i];
                            meta[JpegMeta.HuffSetCount] = sets + 1;
                            meta[JpegMeta.HuffDirty] = 0;
                            sets++;
                        }
                    }
                    meta[sb + JpegMeta.ScHuffSet] = sets - 1;
                }
                if (err == 0)
                {
                    int ecsStart = p + n;
                    int rst = 0;
                    int ecsEnd = FindEntropyEnd(f, ecsStart, fileLen, ref rst);
                    meta[sb + JpegMeta.ScSs] = ss;
                    meta[sb + JpegMeta.ScSe] = se;
                    meta[sb + JpegMeta.ScAh] = ah;
                    meta[sb + JpegMeta.ScAl] = al;
                    meta[sb + JpegMeta.ScEcsStart] = ecsStart;
                    meta[sb + JpegMeta.ScEcsEnd] = ecsEnd;
                    meta[sb + JpegMeta.ScRestartInterval] = meta[JpegMeta.RestartInterval];
                    meta[sb + JpegMeta.ScRstCount] = rst;
                    meta[JpegMeta.ScanCount] = sc + 1;
                }
            }
        }
        return err;
    }

    private static void ScanMarkersKernel(
        Index1D _, ArrayView<byte> file, int fileLen, ArrayView<int> meta, ArrayView<int> huffRaw,
        ArrayView<int> consts, int headerOnly)
    {
        for (int i = 0; i < JpegMeta.Size; i++) meta[i] = 0;
        meta[JpegMeta.Orientation] = 1;
        int work = JpegMeta.MaxHuffSets * JpegMeta.RawSetStride;
        for (int t = 0; t < 8; t++) huffRaw[work + t * JpegMeta.RawTableStride] = 0;

        int err = 0;
        bool soi = false;
        if (fileLen >= 2) soi = (file[0] == 0xFF) & (file[1] == 0xD8);
        if (!soi) err = (int)JpegError.NotJpeg;

        int pos = 2;
        int done = 0;
        int exifDone = 0;
        bool go8 = (err == 0) & (done == 0);
        while (go8)
        {
            if (pos >= fileLen)
            {
                if (meta[JpegMeta.ScanCount] == 0) err = (int)JpegError.TruncatedHeaders;
                done = 1;
            }
            else if (file[pos] != 0xFF)
            {
                err = (int)JpegError.BadMarker;
            }
            else
            {
                int skipping = 1;
                while (skipping != 0)
                {
                    if (pos < fileLen)
                    {
                        if (file[pos] == 0xFF) pos++;
                        else skipping = 0;
                    }
                    else
                    {
                        skipping = 0;
                    }
                }
                if (pos >= fileLen)
                {
                    err = (int)JpegError.TruncatedHeaders;
                }
                else
                {
                    int marker = file[pos];
                    pos++;
                    bool standalone = (marker == 0x01) | ((marker >= 0xD0) & (marker <= 0xD7));
                    bool strayMarker = (marker == 0) | (marker == 0xD8);
                    if (marker == 0xD9)
                    {
                        meta[JpegMeta.EoiSeen] = 1;
                        done = 1;
                    }
                    else if (strayMarker)
                    {
                        err = (int)JpegError.BadMarker;
                        meta[JpegMeta.ErrorDetail] = marker;
                    }
                    else if (!standalone)
                    {
                        if (pos + 2 > fileLen)
                        {
                            err = (int)JpegError.TruncatedHeaders;
                        }
                        else
                        {
                            int len = Be16(file, pos);
                            if (len < 2)
                            {
                                err = (int)JpegError.BadMarker;
                                meta[JpegMeta.ErrorDetail] = marker;
                            }
                            else if (pos + len > fileLen)
                            {
                                err = (int)JpegError.TruncatedHeaders;
                                meta[JpegMeta.ErrorDetail] = marker;
                            }
                            else
                            {
                                int p = pos + 2;
                                int n = len - 2;
                                int segEnd = pos + len;
                                bool sofKnown = (marker == 0xC0) | (marker == 0xC1) | (marker == 0xC2);
                                bool unsupportedSof = (marker == 0xC3) | ((marker >= 0xC5) & (marker <= 0xC7))
                                                    | ((marker >= 0xC9) & (marker <= 0xCB)) | ((marker >= 0xCD) & (marker <= 0xCF))
                                                    | (marker == 0xC8) | (marker == 0xCC) | (marker == 0xDC);
                                if (sofKnown)
                                {
                                    err = ParseSof(file, p, n, meta, marker);
                                }
                                else if (unsupportedSof)
                                {
                                    err = (int)JpegError.UnsupportedFrame;
                                    meta[JpegMeta.ErrorDetail] = marker;
                                }
                                else if (marker == 0xC4)
                                {
                                    err = ParseDht(file, p, n, meta, huffRaw);
                                }
                                else if (marker == 0xDB)
                                {
                                    err = ParseDqt(file, p, n, meta, consts);
                                }
                                else if (marker == 0xDD)
                                {
                                    if (n != 2) err = (int)JpegError.BadMarker;
                                    else meta[JpegMeta.RestartInterval] = Be16(file, p);
                                    if (err != 0) meta[JpegMeta.ErrorDetail] = marker;
                                }
                                else if (marker == 0xDA)
                                {
                                    err = ParseSos(file, p, n, fileLen, meta, huffRaw);
                                    if (err == 0)
                                    {
                                        int sb = JpegMeta.Scans + (meta[JpegMeta.ScanCount] - 1) * JpegMeta.ScanStride;
                                        segEnd = meta[sb + JpegMeta.ScEcsEnd];
                                        if (headerOnly != 0) done = 1;
                                    }
                                }
                                else if (marker == 0xE0)
                                {
                                    if (n >= 14)
                                    {
                                        bool jfif = (file[p] == 'J') & (file[p + 1] == 'F') & (file[p + 2] == 'I')
                                                  & (file[p + 3] == 'F') & (file[p + 4] == 0);
                                        if (jfif) meta[JpegMeta.JfifSeen] = 1;
                                    }
                                }
                                else if (marker == 0xE1)
                                {
                                    if (exifDone == 0)
                                    {
                                        int o = ParseExifOrientation(file, p, n);
                                        if (o != 0)
                                        {
                                            meta[JpegMeta.Orientation] = o;
                                            exifDone = 1;
                                        }
                                    }
                                }
                                else if (marker == 0xEE)
                                {
                                    if (n >= 12)
                                    {
                                        bool adobe = (file[p] == 'A') & (file[p + 1] == 'd') & (file[p + 2] == 'o')
                                                   & (file[p + 3] == 'b') & (file[p + 4] == 'e');
                                        if (adobe)
                                        {
                                            meta[JpegMeta.AdobeSeen] = 1;
                                            meta[JpegMeta.AdobeTransform] = file[p + 11];
                                        }
                                    }
                                }
                                else
                                {
                                    // APPn, COM, JPGn: skipped. Reserved / unknown markers are an error.
                                    bool skippable = ((marker >= 0xE0) & (marker <= 0xEF)) | (marker == 0xFE)
                                                   | ((marker >= 0xF0) & (marker <= 0xFD));
                                    if (!skippable)
                                    {
                                        err = (int)JpegError.BadMarker;
                                        meta[JpegMeta.ErrorDetail] = marker;
                                    }
                                }
                                pos = segEnd;
                            }
                        }
                    }
                }
            }
            go8 = (err == 0) & (done == 0);
        }

        if (err == 0)
        {
            if (meta[JpegMeta.SofSeen] == 0) err = (int)JpegError.NoFrame;
            else if (meta[JpegMeta.ScanCount] == 0) err = (int)JpegError.NoImageData;
        }
        meta[JpegMeta.Error] = err;
    }

    // =================================================================================================================
    // 3. Derived Huffman tables
    // =================================================================================================================

    private static void BuildHuffKernel(Index1D index, ArrayView<int> meta, ArrayView<int> huffRaw, ArrayView<int> huff, int scan)
    {
        int t = index.X;
        int set = meta[JpegMeta.Scans + scan * JpegMeta.ScanStride + JpegMeta.ScHuffSet];
        int raw = set * JpegMeta.RawSetStride + t * JpegMeta.RawTableStride;
        int d = t * JpegMeta.HuffStride;
        huff[d + JpegMeta.HuffDefined] = 0;
        if (huffRaw[raw] != 0)
        {
            for (int i = 0; i < 512; i++) huff[d + JpegMeta.HuffLookup + i] = 0;
            int code = 0;
            int k = 0;
            int bad = 0;
            for (int l = 1; l <= 16; l++)
            {
                int cnt = huffRaw[raw + l];
                huff[d + JpegMeta.HuffMaxCode + l] = -1;
                huff[d + JpegMeta.HuffValOffset + l] = 0;
                if (cnt > 0)
                {
                    huff[d + JpegMeta.HuffValOffset + l] = k - code;
                    for (int j = 0; j < cnt; j++)
                    {
                        int sym = huffRaw[raw + 17 + k + j];
                        huff[d + JpegMeta.HuffSymbols + k + j] = sym;
                        if (l <= 9)
                        {
                            int c = code + j;
                            int span = 1 << (9 - l);
                            int baseIdx = c << (9 - l);
                            for (int s = 0; s < span; s++)
                                if (baseIdx + s < 512) huff[d + JpegMeta.HuffLookup + baseIdx + s] = (l << 8) | sym;
                        }
                        // DC tables must not describe a category above 15 (checked for every table: AC symbols are 0..255).
                        bool dcTooBig = (t < 4) & (sym > 15);
                        if (dcTooBig) bad = 1;
                    }
                    code += cnt;
                    k += cnt;
                    huff[d + JpegMeta.HuffMaxCode + l] = code - 1;
                    if (code > (1 << l)) bad = 1;     // oversubscribed
                }
                code <<= 1;
            }
            huff[d + JpegMeta.HuffMaxCode + 17] = 0xFFFFF;
            huff[d + JpegMeta.HuffDefined] = 1;
            if (bad != 0)
            {
                if (meta[JpegMeta.Error] == 0)
                {
                    meta[JpegMeta.Error] = (int)JpegError.BadHuffmanTable;
                    meta[JpegMeta.ErrorDetail] = t;
                }
            }
        }
    }

    // =================================================================================================================
    // 4. Restart segments
    // =================================================================================================================

    private static void FindRestartsKernel(
        Index1D _, ArrayView<byte> file, ArrayView<int> meta, ArrayView<int> segs, int scan, int segmentCount)
    {
        int sb = JpegMeta.Scans + scan * JpegMeta.ScanStride;
        int start = meta[sb + JpegMeta.ScEcsStart];
        int end = meta[sb + JpegMeta.ScEcsEnd];
        segs[0] = start;
        segs[1] = end;
        if (segmentCount > 1)
        {
            int k = 0;
            int p = start;
            int err = 0;
            bool go9 = (p < end - 1) & (err == 0);
            while (go9)
            {
                if (file[p] != 0xFF)
                {
                    p++;
                }
                else
                {
                    int n = file[p + 1];
                    bool isRst = (n >= 0xD0) & (n <= 0xD7);
                    if (n == 0)
                    {
                        p += 2;
                    }
                    else if (n == 0xFF)
                    {
                        p += 1;
                    }
                    else if (isRst)
                    {
                        if (n - 0xD0 != (k & 7))
                        {
                            err = (int)JpegError.BadRestart;
                        }
                        else if (k + 1 >= segmentCount)
                        {
                            err = (int)JpegError.BadRestart;
                        }
                        else
                        {
                            segs[2 * k + 1] = p;
                            k++;
                            segs[2 * k] = p + 2;
                            segs[2 * k + 1] = end;
                            p += 2;
                        }
                    }
                    else
                    {
                        p += 2;
                    }
                }
                go9 = (p < end - 1) & (err == 0);
            }
            bool c10 = (err == 0) & (k + 1 != segmentCount);
            if (c10) err = (int)JpegError.BadRestart;
            if (err != 0)
            {
                if (meta[JpegMeta.Error] == 0) meta[JpegMeta.Error] = err;
            }
        }
    }

    // =================================================================================================================
    // 2/5. Coefficient buffer + entropy decode
    // =================================================================================================================

    private static void ClearCoefKernel(Index1D index, ArrayView<short> coef)
    {
        coef[index.X] = 0;
    }

    private static void DecodeScanKernel(
        Index1D index, ArrayView<byte> file, ArrayView<short> coef, ArrayView<int> meta, ArrayView<int> huff,
        ArrayView<int> segs, ArrayView<int> consts, ArrayView<int> state, int scan, int budget, int round)
    {
        int seg = index.X;
        int sb = JpegMeta.Scans + scan * JpegMeta.ScanStride;
        int ns = meta[sb + JpegMeta.ScNs];
        int ss = meta[sb + JpegMeta.ScSs];
        int se = meta[sb + JpegMeta.ScSe];
        int ah = meta[sb + JpegMeta.ScAh];
        int al = meta[sb + JpegMeta.ScAl];
        int ri = meta[sb + JpegMeta.ScRestartInterval];
        int sofType = meta[JpegMeta.SofType];

        int mcusX;
        int mcusY;
        if (ns == 1)
        {
            int c0 = JpegMeta.CompLayout + meta[sb + JpegMeta.ScComp] * JpegMeta.CompLayoutStride;
            mcusX = meta[c0 + JpegMeta.ClWidthInBlocks];
            mcusY = meta[c0 + JpegMeta.ClHeightInBlocks];
        }
        else
        {
            int mw = 8 * meta[JpegMeta.MaxH];
            int mh = 8 * meta[JpegMeta.MaxV];
            mcusX = (meta[JpegMeta.Width] + mw - 1) / mw;
            mcusY = (meta[JpegMeta.Height] + mh - 1) / mh;
        }
        int total = mcusX * mcusY;
        int first = 0;
        int last = total;
        if (ri > 0)
        {
            first = seg * ri;
            last = first + ri;
            if (last > total) last = total;
        }

        // 0 sequential, 1 DC first, 2 DC refine, 3 AC first, 4 AC refine
        int mode = 0;
        if (sofType == 2)
        {
            if (ss == 0) mode = ah == 0 ? 1 : 2;
            else mode = ah == 0 ? 3 : 4;
        }

        var st = new JpegBitState();
        int mcu = first;
        int sbase = seg * JpegMeta.SegStateStride;
        if (round == 0)
        {
            st.Pos = segs[2 * seg];
            st.End = segs[2 * seg + 1];
        }
        else
        {
            st.Pos = state[sbase + 0];
            st.End = state[sbase + 1];
            st.Buf = (uint)state[sbase + 2];
            st.Cnt = state[sbase + 3];
            st.Pad = state[sbase + 4];
            st.Err = state[sbase + 5];
            st.EobRun = state[sbase + 6];
            st.Pred0 = state[sbase + 7];
            st.Pred1 = state[sbase + 8];
            st.Pred2 = state[sbase + 9];
            st.Pred3 = state[sbase + 10];
            mcu = state[sbase + 11];
        }

        int stop = mcu + budget;
        if (stop > last) stop = last;
        bool go11 = (mcu < stop) & (st.Err == 0);
        while (go11)
        {
            int mcuRow = mcu / mcusX;
            int mcuCol = mcu - mcuRow * mcusX;
            for (int k = 0; k < ns; k++)
            {
                int ci = meta[sb + JpegMeta.ScComp + k];
                int cb = JpegMeta.CompLayout + ci * JpegMeta.CompLayoutStride;
                int h = 1;
                int v = 1;
                if (ns > 1)
                {
                    h = meta[cb + JpegMeta.ClH];
                    v = meta[cb + JpegMeta.ClV];
                }
                int bpl = meta[cb + JpegMeta.ClBlocksPerLine];
                int coefBase = meta[cb + JpegMeta.ClCoefOffset];
                int tt = meta[sb + JpegMeta.ScTables + k];
                int dcTb = (tt >> 4) * JpegMeta.HuffStride;
                int acTb = (4 + (tt & 15)) * JpegMeta.HuffStride;
                for (int by = 0; by < v; by++)
                {
                    for (int bx = 0; bx < h; bx++)
                    {
                        int row = mcuRow * v + by;
                        int col = mcuCol * h + bx;
                        int off = coefBase + (row * bpl + col) * 64;
                        if (st.Err == 0)
                        {
                            if (mode == 0)
                                JpegEntropyGpu.DecodeBlockSequential(ref st, file, coef, huff, consts, off, dcTb, acTb, k);
                            else if (mode == 1)
                                JpegEntropyGpu.DecodeBlockDcFirst(ref st, file, coef, huff, off, dcTb, k, al);
                            else if (mode == 2)
                                JpegEntropyGpu.DecodeBlockDcRefine(ref st, file, coef, off, al);
                            else if (mode == 3)
                                JpegEntropyGpu.DecodeBlockAcFirst(ref st, file, coef, huff, consts, off, acTb, ss, se, al);
                            else
                                JpegEntropyGpu.DecodeBlockAcRefine(ref st, file, coef, huff, consts, off, acTb, ss, se, al);
                        }
                    }
                }
            }
            mcu++;
            // Progressive scans reset the end-of-band run only at restart boundaries (handled by the segment start).
            go11 = (mcu < stop) & (st.Err == 0);
        }

        bool overran = (st.Err == 0) & (mcu >= last) & (st.Pad > st.Cnt);
        if (overran) st.Err = (int)JpegError.EntropyTruncated;
        if (st.Err != 0)
        {
            // The first error wins (an earlier stage's cause must not be overwritten by its consequences).
            if (meta[JpegMeta.Error] == 0) meta[JpegMeta.Error] = st.Err;
        }

        state[sbase + 0] = st.Pos;
        state[sbase + 1] = st.End;
        state[sbase + 2] = (int)st.Buf;
        state[sbase + 3] = st.Cnt;
        state[sbase + 4] = st.Pad;
        state[sbase + 5] = st.Err;
        state[sbase + 6] = st.EobRun;
        state[sbase + 7] = st.Pred0;
        state[sbase + 8] = st.Pred1;
        state[sbase + 9] = st.Pred2;
        state[sbase + 10] = st.Pred3;
        state[sbase + 11] = mcu;
    }

    // =================================================================================================================
    // 6. Dequantize + ISLOW inverse DCT (jidctint.c, CONST_BITS = 13, PASS1_BITS = 2)
    // =================================================================================================================

    private struct Idct8
    {
        public int V0, V1, V2, V3, V4, V5, V6, V7;
    }

    private static Idct8 Idct1D(int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7, int shift)
    {
        // Even part
        int z2 = i2;
        int z3 = i6;
        int z1 = (z2 + z3) * 4433;
        int tmp2 = z1 + z3 * -15137;
        int tmp3 = z1 + z2 * 6270;
        z2 = i0;
        z3 = i4;
        int tmp0 = (z2 + z3) << 13;
        int tmp1 = (z2 - z3) << 13;
        int tmp10 = tmp0 + tmp3;
        int tmp13 = tmp0 - tmp3;
        int tmp11 = tmp1 + tmp2;
        int tmp12 = tmp1 - tmp2;
        // Odd part
        tmp0 = i7;
        tmp1 = i5;
        tmp2 = i3;
        tmp3 = i1;
        z1 = tmp0 + tmp3;
        z2 = tmp1 + tmp2;
        z3 = tmp0 + tmp2;
        int z4 = tmp1 + tmp3;
        int z5 = (z3 + z4) * 9633;
        tmp0 *= 2446;
        tmp1 *= 16819;
        tmp2 *= 25172;
        tmp3 *= 12299;
        z1 *= -7373;
        z2 *= -20995;
        z3 *= -16069;
        z4 *= -3196;
        z3 += z5;
        z4 += z5;
        tmp0 += z1 + z3;
        tmp1 += z2 + z4;
        tmp2 += z2 + z3;
        tmp3 += z1 + z4;

        int rnd = 1 << (shift - 1);
        var r = new Idct8();
        r.V0 = (tmp10 + tmp3 + rnd) >> shift;
        r.V7 = (tmp10 - tmp3 + rnd) >> shift;
        r.V1 = (tmp11 + tmp2 + rnd) >> shift;
        r.V6 = (tmp11 - tmp2 + rnd) >> shift;
        r.V2 = (tmp12 + tmp1 + rnd) >> shift;
        r.V5 = (tmp12 - tmp1 + rnd) >> shift;
        r.V3 = (tmp13 + tmp0 + rnd) >> shift;
        r.V4 = (tmp13 - tmp0 + rnd) >> shift;
        return r;
    }

    private static int Pick(Idct8 s, int r)
    {
        int v = s.V0;
        if (r == 1) v = s.V1;
        if (r == 2) v = s.V2;
        if (r == 3) v = s.V3;
        if (r == 4) v = s.V4;
        if (r == 5) v = s.V5;
        if (r == 6) v = s.V6;
        if (r == 7) v = s.V7;
        return v;
    }

    /// <summary>libjpeg's post-IDCT range limit: clamp(v + 128, 0, 255) with the table's 1024-entry wrap-around.</summary>
    private static byte RangeLimit(int v)
    {
        int idx = v & 1023;
        int r;
        if (idx < 128) r = idx + 128;
        else if (idx < 512) r = 255;
        else if (idx < 896) r = 0;
        else r = idx - 896;
        return (byte)r;
    }

    private static void IdctKernel(Index1D index, ArrayView<short> coef, ArrayView<byte> planes, ArrayView<int> meta)
    {
        int b = index.X;
        int nc = meta[JpegMeta.NumComps];
        int ci = 0;
        for (int c = 1; c < nc; c++)
        {
            if (b >= meta[JpegMeta.CompLayout + c * JpegMeta.CompLayoutStride + JpegMeta.ClBlockPrefix]) ci = c;
        }
        int cb = JpegMeta.CompLayout + ci * JpegMeta.CompLayoutStride;
        int local = b - meta[cb + JpegMeta.ClBlockPrefix];
        int bpl = meta[cb + JpegMeta.ClBlocksPerLine];
        int row = local / bpl;
        int col = local - row * bpl;
        int co = meta[cb + JpegMeta.ClCoefOffset] + local * 64;
        int qo = JpegMeta.Quant + ci * 64;

        // Pass 1: columns. Column c of the dequantized block -> w_c (rows 0..7 of workspace column c).
        Idct8 w0 = Idct1D(
            coef[co + 0] * meta[qo + 0], coef[co + 8] * meta[qo + 8], coef[co + 16] * meta[qo + 16], coef[co + 24] * meta[qo + 24],
            coef[co + 32] * meta[qo + 32], coef[co + 40] * meta[qo + 40], coef[co + 48] * meta[qo + 48], coef[co + 56] * meta[qo + 56], 11);
        Idct8 w1 = Idct1D(
            coef[co + 1] * meta[qo + 1], coef[co + 9] * meta[qo + 9], coef[co + 17] * meta[qo + 17], coef[co + 25] * meta[qo + 25],
            coef[co + 33] * meta[qo + 33], coef[co + 41] * meta[qo + 41], coef[co + 49] * meta[qo + 49], coef[co + 57] * meta[qo + 57], 11);
        Idct8 w2 = Idct1D(
            coef[co + 2] * meta[qo + 2], coef[co + 10] * meta[qo + 10], coef[co + 18] * meta[qo + 18], coef[co + 26] * meta[qo + 26],
            coef[co + 34] * meta[qo + 34], coef[co + 42] * meta[qo + 42], coef[co + 50] * meta[qo + 50], coef[co + 58] * meta[qo + 58], 11);
        Idct8 w3 = Idct1D(
            coef[co + 3] * meta[qo + 3], coef[co + 11] * meta[qo + 11], coef[co + 19] * meta[qo + 19], coef[co + 27] * meta[qo + 27],
            coef[co + 35] * meta[qo + 35], coef[co + 43] * meta[qo + 43], coef[co + 51] * meta[qo + 51], coef[co + 59] * meta[qo + 59], 11);
        Idct8 w4 = Idct1D(
            coef[co + 4] * meta[qo + 4], coef[co + 12] * meta[qo + 12], coef[co + 20] * meta[qo + 20], coef[co + 28] * meta[qo + 28],
            coef[co + 36] * meta[qo + 36], coef[co + 44] * meta[qo + 44], coef[co + 52] * meta[qo + 52], coef[co + 60] * meta[qo + 60], 11);
        Idct8 w5 = Idct1D(
            coef[co + 5] * meta[qo + 5], coef[co + 13] * meta[qo + 13], coef[co + 21] * meta[qo + 21], coef[co + 29] * meta[qo + 29],
            coef[co + 37] * meta[qo + 37], coef[co + 45] * meta[qo + 45], coef[co + 53] * meta[qo + 53], coef[co + 61] * meta[qo + 61], 11);
        Idct8 w6 = Idct1D(
            coef[co + 6] * meta[qo + 6], coef[co + 14] * meta[qo + 14], coef[co + 22] * meta[qo + 22], coef[co + 30] * meta[qo + 30],
            coef[co + 38] * meta[qo + 38], coef[co + 46] * meta[qo + 46], coef[co + 54] * meta[qo + 54], coef[co + 62] * meta[qo + 62], 11);
        Idct8 w7 = Idct1D(
            coef[co + 7] * meta[qo + 7], coef[co + 15] * meta[qo + 15], coef[co + 23] * meta[qo + 23], coef[co + 31] * meta[qo + 31],
            coef[co + 39] * meta[qo + 39], coef[co + 47] * meta[qo + 47], coef[co + 55] * meta[qo + 55], coef[co + 63] * meta[qo + 63], 11);

        // Pass 2: rows. Row r of the workspace is (w0.r, w1.r, ... w7.r).
        int stride = meta[cb + JpegMeta.ClPlaneStride];
        int dst = meta[cb + JpegMeta.ClPlaneOffset] + (row * 8) * stride + col * 8;
        for (int r = 0; r < 8; r++)
        {
            Idct8 o = Idct1D(Pick(w0, r), Pick(w1, r), Pick(w2, r), Pick(w3, r), Pick(w4, r), Pick(w5, r), Pick(w6, r), Pick(w7, r), 18);
            int d = dst + r * stride;
            planes[d + 0] = RangeLimit(o.V0);
            planes[d + 1] = RangeLimit(o.V1);
            planes[d + 2] = RangeLimit(o.V2);
            planes[d + 3] = RangeLimit(o.V3);
            planes[d + 4] = RangeLimit(o.V4);
            planes[d + 5] = RangeLimit(o.V5);
            planes[d + 6] = RangeLimit(o.V6);
            planes[d + 7] = RangeLimit(o.V7);
        }
    }

    // =================================================================================================================
    // 7. Upsampling + colour conversion
    // =================================================================================================================

    private static int Clamp255(int v)
    {
        int r = v;
        if (r < 0) r = 0;
        if (r > 255) r = 255;
        return r;
    }

    /// <summary>Component sample at output pixel (x, y) after libjpeg's upsampling.</summary>
    private static int Sample(ArrayView<byte> planes, ArrayView<int> meta, int cb, int x, int y)
    {
        int po = meta[cb + JpegMeta.ClPlaneOffset];
        int stride = meta[cb + JpegMeta.ClPlaneStride];
        int dw = meta[cb + JpegMeta.ClDownW];
        int dh = meta[cb + JpegMeta.ClDownH];
        int mode = meta[cb + JpegMeta.ClUpMode];
        int r = 0;
        if (mode == JpegMeta.UpFull)
        {
            r = planes[po + y * stride + x];
        }
        else if (mode == JpegMeta.UpReplicate)
        {
            int hr = meta[cb + JpegMeta.ClHr];
            int vr = meta[cb + JpegMeta.ClVr];
            r = planes[po + (y / vr) * stride + x / hr];
        }
        else if (mode == JpegMeta.UpH2V1)
        {
            int rowBase = po + y * stride;
            int ic = x >> 1;
            int cur = planes[rowBase + ic];
            if ((x & 1) == 0)
            {
                if (ic == 0) r = cur;
                else r = (cur * 3 + planes[rowBase + ic - 1] + 1) >> 2;
            }
            else
            {
                if (ic == dw - 1) r = cur;
                else r = (cur * 3 + planes[rowBase + ic + 1] + 2) >> 2;
            }
        }
        else if (mode == JpegMeta.UpH1V2)
        {
            int ir = y >> 1;
            int other = (y & 1) == 0 ? ir - 1 : ir + 1;
            if (other < 0) other = 0;
            if (other > dh - 1) other = dh - 1;
            int bias = (y & 1) == 0 ? 1 : 2;
            r = (planes[po + ir * stride + x] * 3 + planes[po + other * stride + x] + bias) >> 2;
        }
        else
        {
            // h2v2 fancy: vertical 3:1 blend of two input rows, then horizontal 3:1 blend of the column sums.
            int ir = y >> 1;
            int other = (y & 1) == 0 ? ir - 1 : ir + 1;
            if (other < 0) other = 0;
            if (other > dh - 1) other = dh - 1;
            int r0 = po + ir * stride;
            int r1 = po + other * stride;
            int ic = x >> 1;
            int cs = planes[r0 + ic] * 3 + planes[r1 + ic];
            if ((x & 1) == 0)
            {
                if (ic == 0)
                {
                    r = (cs * 4 + 8) >> 4;
                }
                else
                {
                    int ls = planes[r0 + ic - 1] * 3 + planes[r1 + ic - 1];
                    r = (cs * 3 + ls + 8) >> 4;
                }
            }
            else
            {
                if (ic == dw - 1)
                {
                    r = (cs * 4 + 7) >> 4;
                }
                else
                {
                    int ns = planes[r0 + ic + 1] * 3 + planes[r1 + ic + 1];
                    r = (cs * 3 + ns + 7) >> 4;
                }
            }
        }
        return r;
    }

    private static int MulDiv255(int a, int b)
    {
        int tmp = a * b + 128;
        return ((tmp >> 8) + tmp) >> 8;
    }

    private static void ColorKernel(Index1D index, ArrayView<byte> planes, ArrayView<int> meta, ArrayView<byte> output, int width)
    {
        int i = index.X;
        int y = i / width;
        int x = i - y * width;
        int nc = meta[JpegMeta.NumComps];
        int cs = meta[JpegMeta.ColorSpace];
        int fmt = meta[JpegMeta.OutputFormat];

        int c0 = Sample(planes, meta, JpegMeta.CompLayout, x, y);
        int c1 = 0;
        int c2 = 0;
        int c3 = 0;
        if (nc > 1)
        {
            c1 = Sample(planes, meta, JpegMeta.CompLayout + JpegMeta.CompLayoutStride, x, y);
            c2 = Sample(planes, meta, JpegMeta.CompLayout + 2 * JpegMeta.CompLayoutStride, x, y);
        }
        if (nc > 3) c3 = Sample(planes, meta, JpegMeta.CompLayout + 3 * JpegMeta.CompLayoutStride, x, y);

        int r = c0;
        int g = c0;
        int b = c0;
        bool isYccLike = (cs == JpegMeta.ColorYcc) | (cs == JpegMeta.ColorYcck);
        if (cs == JpegMeta.ColorRgb)
        {
            g = c1;
            b = c2;
        }
        else if (isYccLike)
        {
            // jdcolor.c fixed point (SCALEBITS = 16): R = y + Cr_r, G = y + ((Cb_g + Cr_g) >> 16), B = y + Cb_b.
            int cb = c1 - 128;
            int cr = c2 - 128;
            int rr = Clamp255(c0 + ((91881 * cr + 32768) >> 16));
            int gg = Clamp255(c0 + ((-22554 * cb + 32768 - 46802 * cr) >> 16));
            int bb = Clamp255(c0 + ((116130 * cb + 32768) >> 16));
            if (cs == JpegMeta.ColorYcc)
            {
                r = rr;
                g = gg;
                b = bb;
            }
            else
            {
                // YCCK -> CMYK (ycck_cmyk_convert): C = 255 - R, ... K as is.
                r = 255 - rr;
                g = 255 - gg;
                b = 255 - bb;
            }
        }
        else if (cs == JpegMeta.ColorCmyk)
        {
            r = c0;
            g = c1;
            b = c2;
        }

        bool c12 = (cs == JpegMeta.ColorCmyk) | (cs == JpegMeta.ColorYcck);
        if (c12)
        {
            // Pillow reads four-component files as inverted CMYK ("CMYK;I") and converts with
            // R = nk - (c * nk) / 255 where c is the ink amount (255 - stored) and nk = 255 - K ink = stored K.
            int nk = c3;
            int rr = Clamp255(nk - MulDiv255(255 - r, nk));
            int gg = Clamp255(nk - MulDiv255(255 - g, nk));
            int bb = Clamp255(nk - MulDiv255(255 - b, nk));
            r = rr;
            g = gg;
            b = bb;
        }

        if (fmt == 1)
        {
            int o = i * 8;
            output[o + 0] = (byte)r;
            output[o + 1] = (byte)r;
            output[o + 2] = (byte)g;
            output[o + 3] = (byte)g;
            output[o + 4] = (byte)b;
            output[o + 5] = (byte)b;
            output[o + 6] = 255;
            output[o + 7] = 255;
        }
        else
        {
            int o = i * 4;
            output[o + 0] = (byte)r;
            output[o + 1] = (byte)g;
            output[o + 2] = (byte)b;
            output[o + 3] = 255;
        }
    }

    // =================================================================================================================
    // 8. EXIF orientation
    // =================================================================================================================

    private static void OrientKernel(Index1D index, ArrayView<byte> src, ArrayView<byte> dst, int width, int height,
        int orientation, int bpp)
    {
        int i = index.X;
        int ow = width;
        if (orientation >= 5) ow = height;
        int oy = i / ow;
        int ox = i - oy * ow;
        int sx = ox;
        int sy = oy;
        if (orientation == 2) { sx = width - 1 - ox; }
        else if (orientation == 3) { sx = width - 1 - ox; sy = height - 1 - oy; }
        else if (orientation == 4) { sy = height - 1 - oy; }
        else if (orientation == 5) { sx = oy; sy = ox; }
        else if (orientation == 6) { sx = oy; sy = height - 1 - ox; }
        else if (orientation == 7) { sx = width - 1 - oy; sy = height - 1 - ox; }
        else if (orientation == 8) { sx = width - 1 - oy; sy = ox; }
        int s = (sy * width + sx) * bpp;
        int d = i * bpp;
        for (int k = 0; k < bpp; k++) dst[d + k] = src[s + k];
    }

    /// <summary>No-op (kernels are owned by the accelerator).</summary>
    public void Dispose() { }
}
