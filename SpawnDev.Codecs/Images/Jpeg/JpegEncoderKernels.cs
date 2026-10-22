// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// The JPEG encode kernels. Every stage runs on the accelerator; the host builds the < 1 KB header blob and
// reads back one scalar (the file length):
//
//   1. Prep         (1 thread / plane sample)  RGBA -> YCbCr (libjpeg jccolor.c fixed point) + chroma downsampling
//                                              (jcsample.c box filters with the alternating rounding bias) + the edge
//                                              replication jcprepct.c performs, into the component planes.
//   2. FdctQuant    (1 thread / block)         libjpeg ISLOW forward DCT (jfdctint.c) + quantization (jcdctmgr.c
//                                              rounding division), coefficients stored in zigzag order.
//   3. Entropy      (1 thread / restart segment, chunked) baseline Huffman coding with the Annex K tables (jchuff.c):
//                                              DC differences, run/size AC symbols, byte stuffing, dummy blocks at the
//                                              MCU-grid edge. Each dispatch advances a bounded number of MCUs and saves
//                                              its bit-writer state.
//   4. Prefix       (1 thread)                 segment output offsets, the EOI marker, the total length.
//   5. Assemble     (1 thread / scratch byte)  with restart markers: copies the segments (+ RSTn) into the output.
//
// The arithmetic mirrors libjpeg-turbo so the produced file is byte-identical to Pillow's for the same input,
// quality, subsampling and restart interval.

using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.Codecs.Images.Jpeg;

/// <summary>Slot indices of the encoder metadata block (host-written, read by the encode kernels).</summary>
internal static class JpegEncMeta
{
    public const int Size = 1024;
    public const int Width = 0;
    public const int Height = 1;
    public const int NumComps = 2;
    public const int MaxH = 3;
    public const int MaxV = 4;
    public const int RestartInterval = 5;
    public const int McusX = 6;
    public const int McusY = 7;
    public const int TotalMcus = 8;
    public const int SegCount = 9;
    public const int SlotSize = 10;
    public const int HeaderLen = 11;
    public const int TotalLen = 12;
    public const int Error = 13;
    public const int DstBase = 14;
    public const int BlocksPerMcu = 15;

    // ---- optimized / progressive path (JpegProgressiveKernels) ----
    /// <summary>Write position in the output file (device-side running offset).</summary>
    public const int OutPos = 16;
    /// <summary>Size of the output buffer the caller provided.</summary>
    public const int OutBound = 17;
    /// <summary>Output offset of the current scan's entropy-coded data.</summary>
    public const int ScanDataStart = 18;
    /// <summary>First slot of the scan table (<see cref="ScanStride"/> ints per scan, at most <see cref="MaxScans"/>).</summary>
    public const int Scans = 256;
    public const int ScanStride = 32;
    public const int MaxScans = 16;
    public const int SKind = 0;          // 0 sequential (optimized baseline), 1 DC first, 2 DC refine, 3 AC first, 4 AC refine
    public const int SNs = 1;
    public const int SComp = 2;          // 3 slots: frame component indices
    public const int SSs = 5;
    public const int SSe = 6;
    public const int SAh = 7;
    public const int SAl = 8;
    public const int STotalMcus = 10;
    public const int SSegCount = 11;
    public const int SSlot = 12;         // per-segment scratch slot (multi-segment scans)
    public const int SMcusX = 13;
    public const int SSosOff = 14;
    public const int SSosLen = 15;
    public const int SDhtCount = 16;
    public const int SDht = 17;          // 4 slots: Huffman table slot indices (0 DC0, 1 DC1, 2 AC0, 3 AC1) in DHT order
    public const int SBlocksPerMcu = 21;
    public const int SBand = 22;         // Se - Ss + 1

    /// <summary>Ints per Huffman table in the optimized-table buffer.</summary>
    public const int TabStride = 2048;
    public const int TabCodes = 0;       // 256: code | length &lt;&lt; 16 per symbol
    public const int TabBits = 256;      // 17: number of codes of each length 1..16
    public const int TabCount = 273;     // number of symbols
    public const int TabVals = 280;      // 256 symbols in code order
    public const int TabFreq = 600;      // 257 working frequencies
    public const int TabCodeSize = 900;  // 257
    public const int TabOthers = 1200;   // 257
    public const int TabBits32 = 1500;   // 34
    public const int HistStride = 257;

    public const int Comp = 32;
    public const int CompStride = 16;
    public const int CH = 0;
    public const int CV = 1;
    public const int CWidthInBlocks = 2;
    public const int CHeightInBlocks = 3;
    public const int CCoefOffset = 4;
    public const int CPlaneOffset = 5;
    public const int CPlaneStride = 6;
    public const int CBlockPrefix = 7;
    public const int CQuant = 8;       // 0 or 1: which divisor table
    public const int CDcTbl = 9;       // 0 luminance, 1 chrominance
    public const int CAcTbl = 10;
    public const int CHf = 11;         // horizontal downsample factor
    public const int CVf = 12;
    public const int CDsRows = 13;     // downsampled rows that carry image data
    public const int CPlaneEnd = 14;   // PlaneOffset + plane size

    /// <summary>Constants buffer: divisor table 0 (natural order, quant * 8), table 1, natural -> zigzag index, Huffman codes.</summary>
    public const int KQuant0 = 0;
    public const int KQuant1 = 64;
    public const int KUnZigzag = 128;
    public const int KHuff = 192;      // 4 tables x 256: DC lum, DC chr, AC lum, AC chr; value = code | size << 16
    public const int KSize = 192 + 4 * 256;

    /// <summary>Ints of per-segment encoder state.</summary>
    public const int StateStride = 16;
}

/// <summary>Bit-writer state of one restart segment.</summary>
internal struct JpegBitWriter
{
    public int Pos;
    public uint Buf;
    public int Cnt;
    public int Pred0;
    public int Pred1;
    public int Pred2;
    public int Err;
}

/// <summary>Loads and launches the JPEG encode kernels.</summary>
public sealed class JpegEncoderKernels : IDisposable
{
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>> _prep;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<short>, ArrayView<int>, ArrayView<int>> _fdct;
    private readonly Action<Index1D, ArrayView<short>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int, int> _entropy;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>> _prefix;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>> _assemble;

    /// <summary>Compile the kernels.</summary>
    public JpegEncoderKernels(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _prep = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>>(PrepKernel);
        _fdct = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<short>, ArrayView<int>, ArrayView<int>>(FdctQuantKernel);
        _entropy = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<short>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int, int>(EntropyKernel);
        _prefix = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>>(PrefixKernel);
        _assemble = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>>(AssembleKernel);
    }

    /// <summary>RGBA -> component planes.</summary>
    public void Prep(ArrayView<byte> rgba, ArrayView<byte> planes, ArrayView<int> meta, int sampleCount)
    {
        _prep(sampleCount, rgba, planes, meta);
    }

    /// <summary>Forward DCT + quantization of every block.</summary>
    public void FdctQuant(ArrayView<byte> planes, ArrayView<short> coefs, ArrayView<int> meta, ArrayView<int> consts, int blockCount)
    {
        _fdct(blockCount, planes, coefs, meta, consts);
    }

    /// <summary>Advance every restart segment by up to <paramref name="budget"/> MCUs.</summary>
    public void Entropy(ArrayView<short> coefs, ArrayView<byte> dst, ArrayView<int> meta, ArrayView<int> consts,
        ArrayView<int> state, int segmentCount, int budget, int round)
    {
        _entropy(segmentCount, coefs, dst, meta, consts, state, budget, round);
    }

    /// <summary>Segment offsets, EOI marker, total length.</summary>
    public void Prefix(ArrayView<byte> output, ArrayView<int> meta, ArrayView<int> state)
    {
        _prefix(1, output, meta, state);
    }

    /// <summary>Copy the scratch segments into the output with restart markers between them.</summary>
    public void Assemble(ArrayView<byte> scratch, ArrayView<byte> output, ArrayView<int> meta, ArrayView<int> state, long extent)
    {
        _assemble((Index1D)(int)extent, scratch, output, meta, state);
    }

    // =================================================================================================================
    // 1. Colour conversion + downsampling
    // =================================================================================================================

    private static int Luma(int r, int g, int b) => (19595 * r + 38470 * g + 7471 * b + 32768) >> 16;

    private static int ChromaB(int r, int g, int b) => (-11059 * r - 21709 * g + 32768 * b + 8421375) >> 16;

    private static int ChromaR(int r, int g, int b) => (32768 * r - 27439 * g - 5329 * b + 8421375) >> 16;

    /// <summary>Component <paramref name="comp"/> (0 = Y, 1 = Cb, 2 = Cr) of the pixel at (x, y), edge-clamped.</summary>
    private static int PixelComponent(ArrayView<byte> rgba, int width, int height, int x, int y, int comp)
    {
        int xx = x;
        int yy = y;
        if (xx > width - 1) xx = width - 1;
        if (yy > height - 1) yy = height - 1;
        int o = (yy * width + xx) * 4;
        int r = rgba[o];
        int g = rgba[o + 1];
        int b = rgba[o + 2];
        int v = Luma(r, g, b);
        if (comp == 1) v = ChromaB(r, g, b);
        else if (comp == 2) v = ChromaR(r, g, b);
        return v;
    }

    private static void PrepKernel(Index1D index, ArrayView<byte> rgba, ArrayView<byte> planes, ArrayView<int> meta)
    {
        int i = index.X;
        int nc = meta[JpegEncMeta.NumComps];
        int ci = 0;
        for (int c = 1; c < nc; c++)
        {
            if (i >= meta[JpegEncMeta.Comp + c * JpegEncMeta.CompStride + JpegEncMeta.CPlaneOffset]) ci = c;
        }
        int cb = JpegEncMeta.Comp + ci * JpegEncMeta.CompStride;
        int local = i - meta[cb + JpegEncMeta.CPlaneOffset];
        int stride = meta[cb + JpegEncMeta.CPlaneStride];
        int row = local / stride;
        int col = local - row * stride;
        int width = meta[JpegEncMeta.Width];
        int height = meta[JpegEncMeta.Height];
        int hf = meta[cb + JpegEncMeta.CHf];
        int vf = meta[cb + JpegEncMeta.CVf];
        int dsRows = meta[cb + JpegEncMeta.CDsRows];

        // Rows past the last real downsampled row repeat it (jcprepct.c expand_bottom_edge on the downsampled data).
        int r2 = row;
        if (r2 > dsRows - 1) r2 = dsRows - 1;

        int sum = 0;
        for (int dy = 0; dy < vf; dy++)
        {
            for (int dx = 0; dx < hf; dx++)
                sum += PixelComponent(rgba, width, height, col * hf + dx, r2 * vf + dy, ci);
        }
        int value = sum;
        if (hf * vf == 2) value = (sum + (col & 1)) >> 1;          // h2v1: bias 0, 1, 0, 1 ...
        else if (hf * vf == 4) value = (sum + 1 + (col & 1)) >> 2; // h2v2: bias 1, 2, 1, 2 ...
        planes[i] = (byte)value;
    }

    // =================================================================================================================
    // 2. Forward DCT (jfdctint.c, CONST_BITS = 13, PASS1_BITS = 2) + quantization (jcdctmgr.c)
    // =================================================================================================================

    private struct Fdct8
    {
        public int V0, V1, V2, V3, V4, V5, V6, V7;
    }

    private static Fdct8 Fdct1D(int d0, int d1, int d2, int d3, int d4, int d5, int d6, int d7, int pass)
    {
        int tmp0 = d0 + d7;
        int tmp7 = d0 - d7;
        int tmp1 = d1 + d6;
        int tmp6 = d1 - d6;
        int tmp2 = d2 + d5;
        int tmp5 = d2 - d5;
        int tmp3 = d3 + d4;
        int tmp4 = d3 - d4;

        // Even part
        int tmp10 = tmp0 + tmp3;
        int tmp13 = tmp0 - tmp3;
        int tmp11 = tmp1 + tmp2;
        int tmp12 = tmp1 - tmp2;

        int shift = pass == 1 ? 11 : 15;          // CONST_BITS - PASS1_BITS, CONST_BITS + PASS1_BITS
        int rnd = 1 << (shift - 1);
        var o = new Fdct8();
        if (pass == 1)
        {
            o.V0 = (tmp10 + tmp11) << 2;
            o.V4 = (tmp10 - tmp11) << 2;
        }
        else
        {
            o.V0 = (tmp10 + tmp11 + 2) >> 2;
            o.V4 = (tmp10 - tmp11 + 2) >> 2;
        }
        int z1 = (tmp12 + tmp13) * 4433;
        o.V2 = (z1 + tmp13 * 6270 + rnd) >> shift;
        o.V6 = (z1 + tmp12 * -15137 + rnd) >> shift;

        // Odd part
        z1 = tmp4 + tmp7;
        int z2 = tmp5 + tmp6;
        int z3 = tmp4 + tmp6;
        int z4 = tmp5 + tmp7;
        int z5 = (z3 + z4) * 9633;
        tmp4 *= 2446;
        tmp5 *= 16819;
        tmp6 *= 25172;
        tmp7 *= 12299;
        z1 *= -7373;
        z2 *= -20995;
        z3 *= -16069;
        z4 *= -3196;
        z3 += z5;
        z4 += z5;
        o.V7 = (tmp4 + z1 + z3 + rnd) >> shift;
        o.V5 = (tmp5 + z2 + z4 + rnd) >> shift;
        o.V3 = (tmp6 + z2 + z3 + rnd) >> shift;
        o.V1 = (tmp7 + z1 + z4 + rnd) >> shift;
        return o;
    }

    private static int Pick(Fdct8 s, int c)
    {
        int v = s.V0;
        if (c == 1) v = s.V1;
        if (c == 2) v = s.V2;
        if (c == 3) v = s.V3;
        if (c == 4) v = s.V4;
        if (c == 5) v = s.V5;
        if (c == 6) v = s.V6;
        if (c == 7) v = s.V7;
        return v;
    }

    /// <summary>jcdctmgr.c quantization: round half away from zero of value / divisor.</summary>
    private static int Quantize(int value, int divisor)
    {
        int t = value;
        if (t < 0)
        {
            t = -t;
            t += divisor >> 1;
            t = t >= divisor ? t / divisor : 0;
            t = -t;
        }
        else
        {
            t += divisor >> 1;
            t = t >= divisor ? t / divisor : 0;
        }
        return t;
    }

    private static Fdct8 RowPass(ArrayView<byte> planes, int p, int pass)
    {
        return Fdct1D(planes[p] - 128, planes[p + 1] - 128, planes[p + 2] - 128, planes[p + 3] - 128,
                      planes[p + 4] - 128, planes[p + 5] - 128, planes[p + 6] - 128, planes[p + 7] - 128, pass);
    }

    private static void FdctQuantKernel(Index1D index, ArrayView<byte> planes, ArrayView<short> coefs, ArrayView<int> meta, ArrayView<int> consts)
    {
        int b = index.X;
        int nc = meta[JpegEncMeta.NumComps];
        int ci = 0;
        for (int c = 1; c < nc; c++)
        {
            if (b >= meta[JpegEncMeta.Comp + c * JpegEncMeta.CompStride + JpegEncMeta.CBlockPrefix]) ci = c;
        }
        int cb = JpegEncMeta.Comp + ci * JpegEncMeta.CompStride;
        int local = b - meta[cb + JpegEncMeta.CBlockPrefix];
        int wib = meta[cb + JpegEncMeta.CWidthInBlocks];
        int by = local / wib;
        int bx = local - by * wib;
        int stride = meta[cb + JpegEncMeta.CPlaneStride];
        int p0 = meta[cb + JpegEncMeta.CPlaneOffset] + (by * 8) * stride + bx * 8;
        int qbase = meta[cb + JpegEncMeta.CQuant] == 0 ? JpegEncMeta.KQuant0 : JpegEncMeta.KQuant1;

        Fdct8 r0 = RowPass(planes, p0, 1);
        Fdct8 r1 = RowPass(planes, p0 + stride, 1);
        Fdct8 r2 = RowPass(planes, p0 + 2 * stride, 1);
        Fdct8 r3 = RowPass(planes, p0 + 3 * stride, 1);
        Fdct8 r4 = RowPass(planes, p0 + 4 * stride, 1);
        Fdct8 r5 = RowPass(planes, p0 + 5 * stride, 1);
        Fdct8 r6 = RowPass(planes, p0 + 6 * stride, 1);
        Fdct8 r7 = RowPass(planes, p0 + 7 * stride, 1);

        int outBase = meta[cb + JpegEncMeta.CCoefOffset] + local * 64;
        for (int c = 0; c < 8; c++)
        {
            Fdct8 o = Fdct1D(Pick(r0, c), Pick(r1, c), Pick(r2, c), Pick(r3, c), Pick(r4, c), Pick(r5, c), Pick(r6, c), Pick(r7, c), 2);
            for (int r = 0; r < 8; r++)
            {
                int n = r * 8 + c;
                coefs[outBase + consts[JpegEncMeta.KUnZigzag + n]] = (short)Quantize(Pick(o, r), consts[qbase + n]);
            }
        }
    }

    // =================================================================================================================
    // 3. Entropy coding
    // =================================================================================================================

    private static void PutBits(ref JpegBitWriter st, ArrayView<byte> dst, int slotBase, int code, int size)
    {
        st.Buf = (st.Buf << size) | ((uint)code & ((1u << size) - 1u));
        st.Cnt += size;
        bool go = st.Cnt >= 8;
        while (go)
        {
            int b = (int)((st.Buf >> (st.Cnt - 8)) & 0xFFu);
            dst[slotBase + st.Pos] = (byte)b;
            st.Pos++;
            if (b == 0xFF)
            {
                dst[slotBase + st.Pos] = 0;
                st.Pos++;
            }
            st.Cnt -= 8;
            go = st.Cnt >= 8;
        }
    }

    private static void PutSymbol(ref JpegBitWriter st, ArrayView<byte> dst, int slotBase, ArrayView<int> consts, int table, int symbol)
    {
        int e = consts[JpegEncMeta.KHuff + table * 256 + symbol];
        PutBits(ref st, dst, slotBase, e & 0xFFFF, e >> 16);
    }

    private static int GetPred(ref JpegBitWriter st, int k)
    {
        int p = st.Pred0;
        if (k == 1) p = st.Pred1;
        if (k == 2) p = st.Pred2;
        return p;
    }

    private static void SetPred(ref JpegBitWriter st, int k, int v)
    {
        if (k == 0) st.Pred0 = v;
        if (k == 1) st.Pred1 = v;
        if (k == 2) st.Pred2 = v;
    }

    private static int BitLength(int v)
    {
        int n = 0;
        int t = v;
        bool go = t != 0;
        while (go)
        {
            n++;
            t >>= 1;
            go = t != 0;
        }
        return n;
    }

    private static void EncodeBlock(ref JpegBitWriter st, ArrayView<byte> dst, int slotBase, ArrayView<short> coefs, int off,
        ArrayView<int> consts, int dcTable, int acTable, int comp)
    {
        int dc = coefs[off];
        int diff = dc - GetPred(ref st, comp);
        SetPred(ref st, comp, dc);
        int temp = diff;
        int temp2 = diff;
        if (temp < 0)
        {
            temp = -temp;
            temp2--;
        }
        int nbits = BitLength(temp);
        PutSymbol(ref st, dst, slotBase, consts, dcTable, nbits);
        if (nbits != 0) PutBits(ref st, dst, slotBase, temp2, nbits);

        int run = 0;
        for (int k = 1; k < 64; k++)
        {
            int v = coefs[off + k];
            if (v == 0)
            {
                run++;
            }
            else
            {
                bool zrl = run > 15;
                while (zrl)
                {
                    PutSymbol(ref st, dst, slotBase, consts, acTable, 0xF0);
                    run -= 16;
                    zrl = run > 15;
                }
                int t = v;
                int t2 = v;
                if (t < 0)
                {
                    t = -t;
                    t2--;
                }
                int nb = BitLength(t);
                PutSymbol(ref st, dst, slotBase, consts, acTable, (run << 4) + nb);
                PutBits(ref st, dst, slotBase, t2, nb);
                run = 0;
            }
        }
        if (run > 0) PutSymbol(ref st, dst, slotBase, consts, acTable, 0);
    }

    /// <summary>A block outside the component's image area: DC difference 0 and an immediate end-of-block (jccoefct.c dummy block).</summary>
    private static void EncodeDummyBlock(ref JpegBitWriter st, ArrayView<byte> dst, int slotBase, ArrayView<int> consts, int dcTable, int acTable)
    {
        PutSymbol(ref st, dst, slotBase, consts, dcTable, 0);
        PutSymbol(ref st, dst, slotBase, consts, acTable, 0);
    }

    private static void EntropyKernel(
        Index1D index, ArrayView<short> coefs, ArrayView<byte> dst, ArrayView<int> meta, ArrayView<int> consts,
        ArrayView<int> state, int budget, int round)
    {
        int seg = index.X;
        int nc = meta[JpegEncMeta.NumComps];
        int ri = meta[JpegEncMeta.RestartInterval];
        int total = meta[JpegEncMeta.TotalMcus];
        int mcusX = meta[JpegEncMeta.McusX];
        int slot = meta[JpegEncMeta.SlotSize];
        int slotBase = meta[JpegEncMeta.DstBase] + seg * slot;
        int first = 0;
        int last = total;
        if (ri > 0)
        {
            first = seg * ri;
            last = first + ri;
            if (last > total) last = total;
        }

        var st = new JpegBitWriter();
        int mcu = first;
        int sbase = seg * JpegEncMeta.StateStride;
        if (round != 0)
        {
            st.Pos = state[sbase + 0];
            st.Buf = (uint)state[sbase + 1];
            st.Cnt = state[sbase + 2];
            st.Pred0 = state[sbase + 3];
            st.Pred1 = state[sbase + 4];
            st.Pred2 = state[sbase + 5];
            st.Err = state[sbase + 6];
            mcu = state[sbase + 7];
        }

        int stop = mcu + budget;
        if (stop > last) stop = last;
        // Worst case bytes one MCU can add: blocks * 400 (see JpegEncoderGpu.GetMaxEncodedSize).
        int mcuBytes = meta[JpegEncMeta.BlocksPerMcu] * 400;
        bool go = (mcu < stop) & (st.Err == 0);
        while (go)
        {
            if (st.Pos + mcuBytes + 8 > slot)
            {
                st.Err = 1;
            }
            else
            {
                int mcuRow = mcu / mcusX;
                int mcuCol = mcu - mcuRow * mcusX;
                for (int c = 0; c < nc; c++)
                {
                    int cb = JpegEncMeta.Comp + c * JpegEncMeta.CompStride;
                    int h = 1;
                    int v = 1;
                    if (nc > 1)
                    {
                        h = meta[cb + JpegEncMeta.CH];
                        v = meta[cb + JpegEncMeta.CV];
                    }
                    int wib = meta[cb + JpegEncMeta.CWidthInBlocks];
                    int hib = meta[cb + JpegEncMeta.CHeightInBlocks];
                    int coefBase = meta[cb + JpegEncMeta.CCoefOffset];
                    int dcT = meta[cb + JpegEncMeta.CDcTbl];
                    int acT = 2 + meta[cb + JpegEncMeta.CAcTbl];
                    for (int by = 0; by < v; by++)
                    {
                        for (int bx = 0; bx < h; bx++)
                        {
                            int row = mcuRow * v + by;
                            int col = mcuCol * h + bx;
                            bool inside = (row < hib) & (col < wib);
                            if (inside)
                                EncodeBlock(ref st, dst, slotBase, coefs, coefBase + (row * wib + col) * 64, consts, dcT, acT, c);
                            else
                                EncodeDummyBlock(ref st, dst, slotBase, consts, dcT, acT);
                        }
                    }
                }
                mcu++;
            }
            go = (mcu < stop) & (st.Err == 0);
        }

        if (mcu >= last)
        {
            // End of the segment: pad the last byte with 1 bits (jchuff.c flush_bits).
            if (st.Cnt > 0)
            {
                int pad = 8 - st.Cnt;
                PutBits(ref st, dst, slotBase, (1 << pad) - 1, pad);
            }
            st.Cnt = 0;
        }
        if (st.Err != 0) meta[JpegEncMeta.Error] = st.Err;

        state[sbase + 0] = st.Pos;
        state[sbase + 1] = (int)st.Buf;
        state[sbase + 2] = st.Cnt;
        state[sbase + 3] = st.Pred0;
        state[sbase + 4] = st.Pred1;
        state[sbase + 5] = st.Pred2;
        state[sbase + 6] = st.Err;
        state[sbase + 7] = mcu;
        state[sbase + 9] = st.Pos;       // length of the segment's bytes once complete
    }

    // =================================================================================================================
    // 4/5. Assembly
    // =================================================================================================================

    private static void PrefixKernel(Index1D _, ArrayView<byte> output, ArrayView<int> meta, ArrayView<int> state)
    {
        int segs = meta[JpegEncMeta.SegCount];
        int off = meta[JpegEncMeta.HeaderLen];
        for (int s = 0; s < segs; s++)
        {
            if (s > 0) off += 2;                     // RSTn marker before the segment
            state[s * JpegEncMeta.StateStride + 10] = off;
            off += state[s * JpegEncMeta.StateStride + 9];
        }
        output[off] = 0xFF;
        output[off + 1] = 0xD9;
        meta[JpegEncMeta.TotalLen] = off + 2;
    }

    private static void AssembleKernel(Index1D index, ArrayView<byte> scratch, ArrayView<byte> output, ArrayView<int> meta, ArrayView<int> state)
    {
        int i = index.X;
        int slot = meta[JpegEncMeta.SlotSize];
        int seg = i / slot;
        int j = i - seg * slot;
        int len = state[seg * JpegEncMeta.StateStride + 9];
        int off = state[seg * JpegEncMeta.StateStride + 10];
        if (j < len) output[off + j] = scratch[i];
        if (j == 0)
        {
            if (seg > 0)
            {
                output[off - 2] = 0xFF;
                output[off - 1] = (byte)(0xD0 + ((seg - 1) & 7));
            }
        }
    }

    /// <summary>No-op (kernels are owned by the accelerator).</summary>
    public void Dispose() { }
}
