// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Kernels for the optimized-table and progressive JPEG encode paths. They follow libjpeg's jcphuff.c / jchuff.c
// two-pass structure, per scan:
//
//   1. ClearHist    zero the symbol-frequency tables
//   2. Scan(gather) the entropy coder in statistics mode: counts every symbol it would emit (atomic adds)
//   3. GenTables    jpeg_gen_optimal_table: optimal length-limited Huffman tables from the counts (4 threads)
//   4. Place        writes this scan's DHT segments (from the generated tables) and the host-built SOS blob
//   5. Scan(emit)   the same coder in output mode, writing entropy-coded bytes (directly into the output for one
//                   segment, into per-segment scratch slots when restart markers split the scan)
//   6. Finish       segment offsets / running output position
//   7. Assemble     (with restart markers) copies the segments into the output with RSTn markers
//
// Progressive coding is jcphuff.c ported function by function: DC first / refine, AC first with EOB runs, AC
// refinement with buffered correction bits. Coefficients are read from the zigzag-ordered block buffer produced by
// the FDCT kernel, so "natural_order[k]" of libjpeg is simply k here.
//
// Needs atomics (the symbol counters), so unlike the rest of the codec it does not run on the WebGL backend.

using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.Codecs.Images.Jpeg;

/// <summary>Bit-writer and run-length state of one restart segment of an optimized / progressive scan.</summary>
internal struct PhuffState
{
    public int Pos;
    public uint Buf;
    public int Cnt;
    public int Pred0, Pred1, Pred2;
    public int Bit0, Bit1, Bit2;
    public int EobRun;
    public int Be;
    public int Err;
    public int Gather;
}

/// <summary>Loads and launches the optimized-table / progressive encode kernels.</summary>
public sealed class JpegScanEncoderKernels : IDisposable
{
    private readonly Action<Index1D, ArrayView<int>> _clearHist;
    private readonly Action<Index1D, ArrayView<short>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>,
        ArrayView<int>, ArrayView<byte>, int, int, int, int> _scan;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>> _gen;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, int> _place;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, int> _finish;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>> _eoi;

    /// <summary>Compile the kernels.</summary>
    public JpegScanEncoderKernels(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _clearHist = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>>(ClearHistKernel);
        _scan = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<short>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>,
            ArrayView<int>, ArrayView<byte>, int, int, int, int>(ScanKernel);
        _gen = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>>(GenTablesKernel);
        _place = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, int>(PlaceKernel);
        _finish = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, int>(FinishKernel);
        _eoi = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<int>>(EoiKernel);
    }

    /// <summary>Zero the four symbol-frequency tables.</summary>
    public void ClearHist(ArrayView<int> hist) => _clearHist(4 * JpegEncMeta.HistStride, hist);

    /// <summary>Run (or advance) the entropy coder of scan <paramref name="scan"/>, counting symbols when <paramref name="gather"/> is set.</summary>
    public void Scan(ArrayView<short> coefs, ArrayView<byte> dst, ArrayView<int> meta, ArrayView<int> tabs,
        ArrayView<int> hist, ArrayView<int> state, ArrayView<byte> bitbuf, int segmentCount, int scan, bool gather,
        int budget, int round)
    {
        _scan(segmentCount, coefs, dst, meta, tabs, hist, state, bitbuf, scan, gather ? 1 : 0, budget, round);
    }

    /// <summary>Generate optimal Huffman tables from the counts.</summary>
    public void GenTables(ArrayView<int> hist, ArrayView<int> tabs) => _gen(4, hist, tabs);

    /// <summary>Write DHT + SOS for the scan, set the scan's output geometry.</summary>
    public void Place(ArrayView<byte> output, ArrayView<byte> sosBlob, ArrayView<int> meta, ArrayView<int> tabs, int scan)
        => _place(1, output, sosBlob, meta, tabs, scan);

    /// <summary>Segment offsets and the new output position after the scan.</summary>
    public void Finish(ArrayView<int> meta, ArrayView<int> state, int scan) => _finish(1, meta, state, scan);

    /// <summary>Write the EOI marker and the total length.</summary>
    public void Eoi(ArrayView<byte> output, ArrayView<int> meta) => _eoi(1, output, meta);

    // =================================================================================================================

    private static void ClearHistKernel(Index1D index, ArrayView<int> hist)
    {
        hist[index.X] = 0;
    }

    private static void PutBits(ref PhuffState st, ArrayView<byte> dst, int slotBase, int code, int size)
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

    private static void EmitBits(ref PhuffState st, ArrayView<byte> dst, int slotBase, int code, int size)
    {
        if (st.Gather == 0) PutBits(ref st, dst, slotBase, code, size);
    }

    private static void EmitSymbol(ref PhuffState st, ArrayView<byte> dst, int slotBase, ArrayView<int> tabs,
        ArrayView<int> hist, int table, int symbol)
    {
        if (st.Gather != 0)
        {
            Atomic.Add(ref hist[table * JpegEncMeta.HistStride + symbol], 1);
        }
        else
        {
            int e = tabs[table * JpegEncMeta.TabStride + JpegEncMeta.TabCodes + symbol];
            PutBits(ref st, dst, slotBase, e & 0xFFFF, e >> 16);
        }
    }

    private static void EmitBuffered(ref PhuffState st, ArrayView<byte> dst, int slotBase, ArrayView<byte> bitbuf, int start, int count)
    {
        if (st.Gather == 0)
        {
            for (int i = 0; i < count; i++) PutBits(ref st, dst, slotBase, bitbuf[start + i], 1);
        }
    }

    private static void EmitEobRun(ref PhuffState st, ArrayView<byte> dst, int slotBase, ArrayView<int> tabs,
        ArrayView<int> hist, ArrayView<byte> bitbuf, int bb, int acTable)
    {
        if (st.EobRun > 0)
        {
            int temp = st.EobRun >> 1;
            int nbits = 0;
            bool go = temp != 0;
            while (go)
            {
                nbits++;
                temp >>= 1;
                go = temp != 0;
            }
            if (nbits > 14) st.Err = 2;
            EmitSymbol(ref st, dst, slotBase, tabs, hist, acTable, nbits << 4);
            if (nbits != 0) EmitBits(ref st, dst, slotBase, st.EobRun, nbits);
            st.EobRun = 0;
            EmitBuffered(ref st, dst, slotBase, bitbuf, bb, st.Be);
            st.Be = 0;
        }
    }

    private static int GetPred(ref PhuffState st, int k)
    {
        int p = st.Pred0;
        if (k == 1) p = st.Pred1;
        if (k == 2) p = st.Pred2;
        return p;
    }

    private static void SetPred(ref PhuffState st, int k, int v)
    {
        if (k == 0) st.Pred0 = v;
        if (k == 1) st.Pred1 = v;
        if (k == 2) st.Pred2 = v;
    }

    private static int GetBit(ref PhuffState st, int k)
    {
        int p = st.Bit0;
        if (k == 1) p = st.Bit1;
        if (k == 2) p = st.Bit2;
        return p;
    }

    private static void SetBit(ref PhuffState st, int k, int v)
    {
        if (k == 0) st.Bit0 = v;
        if (k == 1) st.Bit1 = v;
        if (k == 2) st.Bit2 = v;
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

    /// <summary>Sequential (optimized baseline) block, jchuff.c encode_one_block.</summary>
    private static void SequentialBlock(ref PhuffState st, ArrayView<byte> dst, int slotBase, ArrayView<short> coefs, int off,
        ArrayView<int> tabs, ArrayView<int> hist, int dcTable, int acTable, int comp)
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
        EmitSymbol(ref st, dst, slotBase, tabs, hist, dcTable, nbits);
        if (nbits != 0) EmitBits(ref st, dst, slotBase, temp2, nbits);

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
                    EmitSymbol(ref st, dst, slotBase, tabs, hist, acTable, 0xF0);
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
                EmitSymbol(ref st, dst, slotBase, tabs, hist, acTable, (run << 4) + nb);
                EmitBits(ref st, dst, slotBase, t2, nb);
                run = 0;
            }
        }
        if (run > 0) EmitSymbol(ref st, dst, slotBase, tabs, hist, acTable, 0);
    }

    /// <summary>jcphuff.c encode_mcu_DC_first for one block (point transform Al).</summary>
    private static void DcFirstBlock(ref PhuffState st, ArrayView<byte> dst, int slotBase, ArrayView<short> coefs, int off,
        ArrayView<int> tabs, ArrayView<int> hist, int dcTable, int comp, int al, bool dummy)
    {
        int temp2 = 0;
        if (dummy) temp2 = GetPred(ref st, comp);          // dummy blocks repeat the previous DC: difference 0
        else temp2 = coefs[off] >> al;
        int temp = temp2 - GetPred(ref st, comp);
        SetPred(ref st, comp, temp2);
        temp2 = temp;
        if (temp < 0)
        {
            temp = -temp;
            temp2--;
        }
        int nbits = BitLength(temp);
        EmitSymbol(ref st, dst, slotBase, tabs, hist, dcTable, nbits);
        if (nbits != 0) EmitBits(ref st, dst, slotBase, temp2, nbits);
    }

    /// <summary>jcphuff.c encode_mcu_DC_refine for one block: the Al'th bit of the DC coefficient.</summary>
    private static void DcRefineBlock(ref PhuffState st, ArrayView<byte> dst, int slotBase, ArrayView<short> coefs, int off,
        int comp, int al, bool dummy)
    {
        int bit = 0;
        if (dummy) bit = GetBit(ref st, comp);             // dummy blocks repeat the previous block's bit
        else bit = (coefs[off] >> al) & 1;
        SetBit(ref st, comp, bit);
        EmitBits(ref st, dst, slotBase, bit, 1);
    }

    /// <summary>jcphuff.c encode_mcu_AC_first for one block.</summary>
    private static void AcFirstBlock(ref PhuffState st, ArrayView<byte> dst, int slotBase, ArrayView<short> coefs, int off,
        ArrayView<int> tabs, ArrayView<int> hist, ArrayView<byte> bitbuf, int bb, int acTable, int ss, int se, int al)
    {
        int r = 0;
        for (int k = ss; k <= se; k++)
        {
            int temp = coefs[off + k];
            if (temp == 0)
            {
                r++;
            }
            else
            {
                int temp2;
                if (temp < 0)
                {
                    temp = -temp;
                    temp >>= al;
                    temp2 = ~temp;
                }
                else
                {
                    temp >>= al;
                    temp2 = temp;
                }
                if (temp == 0)
                {
                    r++;
                }
                else
                {
                    if (st.EobRun > 0) EmitEobRun(ref st, dst, slotBase, tabs, hist, bitbuf, bb, acTable);
                    bool zrl = r > 15;
                    while (zrl)
                    {
                        EmitSymbol(ref st, dst, slotBase, tabs, hist, acTable, 0xF0);
                        r -= 16;
                        zrl = r > 15;
                    }
                    int nbits = BitLength(temp);
                    if (nbits > 14) st.Err = 3;
                    EmitSymbol(ref st, dst, slotBase, tabs, hist, acTable, (r << 4) + nbits);
                    EmitBits(ref st, dst, slotBase, temp2, nbits);
                    r = 0;
                }
            }
        }
        if (r > 0)
        {
            st.EobRun++;
            if (st.EobRun == 0x7FFF) EmitEobRun(ref st, dst, slotBase, tabs, hist, bitbuf, bb, acTable);
        }
    }

    /// <summary>jcphuff.c encode_mcu_AC_refine for one block.</summary>
    private static void AcRefineBlock(ref PhuffState st, ArrayView<byte> dst, int slotBase, ArrayView<short> coefs, int off,
        ArrayView<int> tabs, ArrayView<int> hist, ArrayView<byte> bitbuf, int bb, int acTable, int ss, int se, int al)
    {
        // Pre-pass: index of the last newly-nonzero coefficient.
        int eob = 0;
        for (int k = ss; k <= se; k++)
        {
            int t = coefs[off + k];
            if (t < 0) t = -t;
            t >>= al;
            if (t == 1) eob = k;
        }
        int r = 0;
        int br = 0;
        int brStart = st.Be;                                  // BR_buffer = bit_buffer + BE
        for (int k = ss; k <= se; k++)
        {
            int raw = coefs[off + k];
            int temp = raw;
            if (temp < 0) temp = -temp;
            temp >>= al;
            if (temp == 0)
            {
                r++;
            }
            else
            {
                // Emit any required ZRLs, but not if they can be folded into EOB.
                bool zrl = (r > 15) & (k <= eob);
                while (zrl)
                {
                    EmitEobRun(ref st, dst, slotBase, tabs, hist, bitbuf, bb, acTable);
                    EmitSymbol(ref st, dst, slotBase, tabs, hist, acTable, 0xF0);
                    r -= 16;
                    EmitBuffered(ref st, dst, slotBase, bitbuf, bb + brStart, br);
                    brStart = 0;
                    br = 0;
                    zrl = (r > 15) & (k <= eob);
                }
                if (temp > 1)
                {
                    // Previously nonzero: only a (buffered) correction bit.
                    bitbuf[bb + brStart + br] = (byte)(temp & 1);
                    br++;
                }
                else
                {
                    EmitEobRun(ref st, dst, slotBase, tabs, hist, bitbuf, bb, acTable);
                    EmitSymbol(ref st, dst, slotBase, tabs, hist, acTable, (r << 4) + 1);
                    int bit = raw < 0 ? 0 : 1;
                    EmitBits(ref st, dst, slotBase, bit, 1);
                    EmitBuffered(ref st, dst, slotBase, bitbuf, bb + brStart, br);
                    brStart = 0;
                    br = 0;
                    r = 0;
                }
            }
        }
        bool tail = (r > 0) | (br > 0);
        if (tail)
        {
            st.EobRun++;
            st.Be += br;
            bool force = (st.EobRun == 0x7FFF) | (st.Be > 1000 - 64 + 1);
            if (force) EmitEobRun(ref st, dst, slotBase, tabs, hist, bitbuf, bb, acTable);
        }
    }

    // =================================================================================================================
    // The scan kernel: one thread per restart segment
    // =================================================================================================================

    private static void ScanKernel(
        Index1D index, ArrayView<short> coefs, ArrayView<byte> dst, ArrayView<int> meta, ArrayView<int> tabs,
        ArrayView<int> hist, ArrayView<int> state, ArrayView<byte> bitbuf, int scan, int gather, int budget, int round)
    {
        int seg = index.X;
        int sb = JpegEncMeta.Scans + scan * JpegEncMeta.ScanStride;
        int kind = meta[sb + JpegEncMeta.SKind];
        int ns = meta[sb + JpegEncMeta.SNs];
        int ss = meta[sb + JpegEncMeta.SSs];
        int se = meta[sb + JpegEncMeta.SSe];
        int al = meta[sb + JpegEncMeta.SAl];
        int total = meta[sb + JpegEncMeta.STotalMcus];
        int mcusX = meta[sb + JpegEncMeta.SMcusX];
        int ri = meta[JpegEncMeta.RestartInterval];
        int slot = meta[JpegEncMeta.SlotSize];
        int slotBase = meta[JpegEncMeta.DstBase] + seg * slot;
        int bb = seg * 1024;
        int first = 0;
        int last = total;
        if (ri > 0)
        {
            first = seg * ri;
            last = first + ri;
            if (last > total) last = total;
        }

        var st = new PhuffState();
        st.Gather = gather;
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
            st.EobRun = state[sbase + 11];
            st.Be = state[sbase + 12];
            st.Bit0 = state[sbase + 13];
            st.Bit1 = state[sbase + 14];
            st.Bit2 = state[sbase + 15];
        }

        int stop = mcu + budget;
        if (stop > last) stop = last;
        int mcuBytes = meta[sb + JpegEncMeta.SBlocksPerMcu] * (meta[sb + JpegEncMeta.SBand] * 7 + 48);
        int acTableOfScan = 2;                                // AC scans: the single component's table
        if (ns == 1)
        {
            int c0 = meta[sb + JpegEncMeta.SComp];
            if (c0 != 0) acTableOfScan = 3;
        }
        bool go = (mcu < stop) & (st.Err == 0);
        while (go)
        {
            if (st.Gather == 0)
            {
                if (st.Pos + mcuBytes + 16 > slot) st.Err = 1;
            }
            if (st.Err == 0)
            {
                int mcuRow = mcu / mcusX;
                int mcuCol = mcu - mcuRow * mcusX;
                for (int k = 0; k < ns; k++)
                {
                    int ci = meta[sb + JpegEncMeta.SComp + k];
                    int cb = JpegEncMeta.Comp + ci * JpegEncMeta.CompStride;
                    int h = 1;
                    int v = 1;
                    if (ns > 1)
                    {
                        h = meta[cb + JpegEncMeta.CH];
                        v = meta[cb + JpegEncMeta.CV];
                    }
                    int wib = meta[cb + JpegEncMeta.CWidthInBlocks];
                    int hib = meta[cb + JpegEncMeta.CHeightInBlocks];
                    int coefBase = meta[cb + JpegEncMeta.CCoefOffset];
                    int dcT = ci == 0 ? 0 : 1;
                    int acT = ci == 0 ? 2 : 3;
                    for (int by = 0; by < v; by++)
                    {
                        for (int bx = 0; bx < h; bx++)
                        {
                            int row = mcuRow * v + by;
                            int col = mcuCol * h + bx;
                            bool inside = (row < hib) & (col < wib);
                            int off = coefBase + (row * wib + col) * 64;
                            if (kind == 0)
                            {
                                if (inside)
                                {
                                    SequentialBlock(ref st, dst, slotBase, coefs, off, tabs, hist, dcT, acT, k);
                                }
                                else
                                {
                                    EmitSymbol(ref st, dst, slotBase, tabs, hist, dcT, 0);
                                    EmitSymbol(ref st, dst, slotBase, tabs, hist, acT, 0);
                                }
                            }
                            else if (kind == 1)
                            {
                                DcFirstBlock(ref st, dst, slotBase, coefs, off, tabs, hist, dcT, k, al, !inside);
                            }
                            else if (kind == 2)
                            {
                                DcRefineBlock(ref st, dst, slotBase, coefs, off, k, al, !inside);
                            }
                            else if (kind == 3)
                            {
                                AcFirstBlock(ref st, dst, slotBase, coefs, off, tabs, hist, bitbuf, bb, acTableOfScan, ss, se, al);
                            }
                            else
                            {
                                AcRefineBlock(ref st, dst, slotBase, coefs, off, tabs, hist, bitbuf, bb, acTableOfScan, ss, se, al);
                            }
                        }
                    }
                }
                mcu++;
            }
            go = (mcu < stop) & (st.Err == 0);
        }

        bool finished = (mcu >= last) & (st.Err == 0);
        if (finished)
        {
            // End of the segment: flush a pending EOB run, then pad the last byte with 1 bits.
            EmitEobRun(ref st, dst, slotBase, tabs, hist, bitbuf, bb, acTableOfScan);
            if (st.Gather == 0)
            {
                if (st.Cnt > 0)
                {
                    int pad = 8 - st.Cnt;
                    PutBits(ref st, dst, slotBase, (1 << pad) - 1, pad);
                }
                st.Cnt = 0;
            }
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
        state[sbase + 9] = st.Pos;
        state[sbase + 11] = st.EobRun;
        state[sbase + 12] = st.Be;
        state[sbase + 13] = st.Bit0;
        state[sbase + 14] = st.Bit1;
        state[sbase + 15] = st.Bit2;
    }

    // =================================================================================================================
    // Optimal Huffman tables (jchuff.c jpeg_gen_optimal_table)
    // =================================================================================================================

    private static void GenTablesKernel(Index1D index, ArrayView<int> hist, ArrayView<int> tabs)
    {
        int slot = index.X;
        int t = slot * JpegEncMeta.TabStride;
        int h = slot * JpegEncMeta.HistStride;
        int freq = t + JpegEncMeta.TabFreq;
        int csz = t + JpegEncMeta.TabCodeSize;
        int oth = t + JpegEncMeta.TabOthers;
        int b32 = t + JpegEncMeta.TabBits32;
        for (int i = 0; i < 256; i++) tabs[freq + i] = hist[h + i];
        tabs[freq + 256] = 1;                                  // reserve the all-ones code
        for (int i = 0; i <= 256; i++)
        {
            tabs[csz + i] = 0;
            tabs[oth + i] = -1;
        }
        for (int i = 0; i < 34; i++) tabs[b32 + i] = 0;

        bool go = true;
        while (go)
        {
            int c1 = -1;
            int v = 1000000000;
            for (int i = 0; i <= 256; i++)
            {
                bool take = (tabs[freq + i] != 0) & (tabs[freq + i] <= v);
                if (take)
                {
                    v = tabs[freq + i];
                    c1 = i;
                }
            }
            int c2 = -1;
            v = 1000000000;
            for (int i = 0; i <= 256; i++)
            {
                bool take = (tabs[freq + i] != 0) & (tabs[freq + i] <= v) & (i != c1);
                if (take)
                {
                    v = tabs[freq + i];
                    c2 = i;
                }
            }
            if (c2 < 0)
            {
                go = false;
            }
            else
            {
                tabs[freq + c1] = tabs[freq + c1] + tabs[freq + c2];
                tabs[freq + c2] = 0;
                tabs[csz + c1] = tabs[csz + c1] + 1;
                bool chain = tabs[oth + c1] >= 0;
                while (chain)
                {
                    c1 = tabs[oth + c1];
                    tabs[csz + c1] = tabs[csz + c1] + 1;
                    chain = tabs[oth + c1] >= 0;
                }
                tabs[oth + c1] = c2;
                tabs[csz + c2] = tabs[csz + c2] + 1;
                chain = tabs[oth + c2] >= 0;
                while (chain)
                {
                    c2 = tabs[oth + c2];
                    tabs[csz + c2] = tabs[csz + c2] + 1;
                    chain = tabs[oth + c2] >= 0;
                }
            }
        }

        // Count the number of symbols of each code length.
        for (int i = 0; i <= 256; i++)
        {
            int cs = tabs[csz + i];
            if (cs != 0)
            {
                if (cs > 32) cs = 32;
                tabs[b32 + cs] = tabs[b32 + cs] + 1;
            }
        }
        // Limit the code lengths to 16 bits (JPEG K.2 figure K.3).
        for (int i = 32; i > 16; i--)
        {
            bool more = tabs[b32 + i] > 0;
            while (more)
            {
                int j = i - 2;
                bool seek = tabs[b32 + j] == 0;
                while (seek)
                {
                    j--;
                    seek = tabs[b32 + j] == 0;
                }
                tabs[b32 + i] = tabs[b32 + i] - 2;
                tabs[b32 + i - 1] = tabs[b32 + i - 1] + 1;
                tabs[b32 + j + 1] = tabs[b32 + j + 1] + 2;
                tabs[b32 + j] = tabs[b32 + j] - 1;
                more = tabs[b32 + i] > 0;
            }
        }
        // Remove the reserved all-ones symbol (the longest code).
        int top = 16;
        bool findTop = (top > 0) & (tabs[b32 + top] == 0);
        while (findTop)
        {
            top--;
            findTop = (top > 0) & (tabs[b32 + top] == 0);
        }
        if (top > 0) tabs[b32 + top] = tabs[b32 + top] - 1;
        for (int i = 1; i <= 16; i++) tabs[t + JpegEncMeta.TabBits + i] = tabs[b32 + i];

        // Symbols sorted by code length.
        int p = 0;
        for (int len = 1; len <= 32; len++)
        {
            for (int j = 0; j < 256; j++)
            {
                if (tabs[csz + j] == len)
                {
                    tabs[t + JpegEncMeta.TabVals + p] = j;
                    p++;
                }
            }
        }
        tabs[t + JpegEncMeta.TabCount] = p;

        // Encode table: canonical codes in order of increasing length (jpeg_make_c_derived_tbl).
        for (int i = 0; i < 256; i++) tabs[t + JpegEncMeta.TabCodes + i] = 0;
        int code = 0;
        int k = 0;
        for (int len = 1; len <= 16; len++)
        {
            int n = tabs[t + JpegEncMeta.TabBits + len];
            for (int i = 0; i < n; i++)
            {
                int sym = tabs[t + JpegEncMeta.TabVals + k];
                tabs[t + JpegEncMeta.TabCodes + sym] = code | (len << 16);
                code++;
                k++;
            }
            code <<= 1;
        }
    }

    // =================================================================================================================
    // Scan headers and assembly
    // =================================================================================================================

    private static void PlaceKernel(Index1D _, ArrayView<byte> output, ArrayView<byte> sosBlob, ArrayView<int> meta, ArrayView<int> tabs, int scan)
    {
        int sb = JpegEncMeta.Scans + scan * JpegEncMeta.ScanStride;
        int pos = meta[JpegEncMeta.OutPos];
        int nDht = meta[sb + JpegEncMeta.SDhtCount];
        for (int d = 0; d < nDht; d++)
        {
            int slot = meta[sb + JpegEncMeta.SDht + d];
            int t = slot * JpegEncMeta.TabStride;
            int count = tabs[t + JpegEncMeta.TabCount];
            int len = 2 + 1 + 16 + count;
            output[pos] = 0xFF;
            output[pos + 1] = 0xC4;
            output[pos + 2] = (byte)(len >> 8);
            output[pos + 3] = (byte)len;
            int cls = slot >= 2 ? 1 : 0;
            output[pos + 4] = (byte)((cls << 4) | (slot & 1));
            for (int i = 1; i <= 16; i++) output[pos + 4 + i] = (byte)tabs[t + JpegEncMeta.TabBits + i];
            for (int i = 0; i < count; i++) output[pos + 21 + i] = (byte)tabs[t + JpegEncMeta.TabVals + i];
            pos += 2 + len;
        }
        int sosOff = meta[sb + JpegEncMeta.SSosOff];
        int sosLen = meta[sb + JpegEncMeta.SSosLen];
        for (int i = 0; i < sosLen; i++) output[pos + i] = sosBlob[sosOff + i];
        pos += sosLen;
        meta[JpegEncMeta.OutPos] = pos;
        meta[JpegEncMeta.ScanDataStart] = pos;
        int segs = meta[sb + JpegEncMeta.SSegCount];
        if (segs == 1)
        {
            meta[JpegEncMeta.DstBase] = pos;
            meta[JpegEncMeta.SlotSize] = meta[JpegEncMeta.OutBound] - pos - 2;
        }
        else
        {
            meta[JpegEncMeta.DstBase] = 0;
            meta[JpegEncMeta.SlotSize] = meta[sb + JpegEncMeta.SSlot];
        }
    }

    private static void FinishKernel(Index1D _, ArrayView<int> meta, ArrayView<int> state, int scan)
    {
        int sb = JpegEncMeta.Scans + scan * JpegEncMeta.ScanStride;
        int segs = meta[sb + JpegEncMeta.SSegCount];
        int off = meta[JpegEncMeta.ScanDataStart];
        for (int s = 0; s < segs; s++)
        {
            if (s > 0) off += 2;
            state[s * JpegEncMeta.StateStride + 10] = off;
            off += state[s * JpegEncMeta.StateStride + 9];
        }
        meta[JpegEncMeta.OutPos] = off;
    }

    private static void EoiKernel(Index1D _, ArrayView<byte> output, ArrayView<int> meta)
    {
        int pos = meta[JpegEncMeta.OutPos];
        output[pos] = 0xFF;
        output[pos + 1] = 0xD9;
        meta[JpegEncMeta.TotalLen] = pos + 2;
    }

    /// <summary>No-op (kernels are owned by the accelerator).</summary>
    public void Dispose() { }
}
