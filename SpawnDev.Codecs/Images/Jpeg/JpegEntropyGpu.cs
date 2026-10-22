// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Device-side helpers shared by the JPEG decode kernels: the entropy bit reader (byte-stuffing aware, zero-fill
// past the end of a segment with a pad counter so a decoder can tell "consumed real bits" from "consumed fake
// bits"), Huffman symbol decode against the derived tables built by JpegKernels.BuildHuff, and the three block
// decoders (sequential, progressive DC, progressive AC first / refine).
//
// The state struct is passed by ref into static helpers (same pattern as Vp8BoolDecoderGpuState / InflateState).
// A 32-bit bit buffer is used on purpose: 64-bit integers are emulated on WebGPU / WebGL.

using ILGPU;

namespace SpawnDev.Codecs.Images.Jpeg;

/// <summary>Bit-reader and decoder state of one restart segment.</summary>
internal struct JpegBitState
{
    /// <summary>Next byte of the entropy-coded segment.</summary>
    public int Pos;
    /// <summary>One past the last byte of the segment.</summary>
    public int End;
    /// <summary>Bit buffer: the low <see cref="Cnt"/> bits are valid, newest bit lowest.</summary>
    public uint Buf;
    /// <summary>Number of valid bits in <see cref="Buf"/>.</summary>
    public int Cnt;
    /// <summary>Fake (zero) bits appended after the end of the segment, in multiples of 8.</summary>
    public int Pad;
    /// <summary><see cref="JpegError"/> raised by the decoder (0 = none).</summary>
    public int Err;
    /// <summary>Progressive AC: blocks still covered by an end-of-band run.</summary>
    public int EobRun;
    /// <summary>DC predictor of scan component 0.</summary>
    public int Pred0;
    /// <summary>DC predictor of scan component 1.</summary>
    public int Pred1;
    /// <summary>DC predictor of scan component 2.</summary>
    public int Pred2;
    /// <summary>DC predictor of scan component 3.</summary>
    public int Pred3;
}

internal static class JpegEntropyGpu
{
    /// <summary>Top the bit buffer up to more than 24 bits. Past the segment end zero bytes are appended and counted.</summary>
    public static void Fill(ref JpegBitState st, ArrayView<byte> file)
    {
        while (st.Cnt <= 24)
        {
            int b = 0;
            if (st.Pos < st.End)
            {
                b = file[st.Pos];
                if (b == 0xFF)
                {
                    // Inside a segment every 0xFF is followed by a stuffed 0x00; anything else ends the data.
                    int stuffed = 0;
                    if (st.Pos + 1 < st.End)
                    {
                        if (file[st.Pos + 1] == 0) stuffed = 1;
                    }
                    if (stuffed != 0)
                    {
                        st.Pos += 2;
                    }
                    else
                    {
                        st.Pos = st.End;
                        b = 0;
                        st.Pad += 8;
                    }
                }
                else
                {
                    st.Pos++;
                }
            }
            else
            {
                st.Pad += 8;
            }
            st.Buf = (st.Buf << 8) | (uint)b;
            st.Cnt += 8;
        }
    }

    /// <summary>Read <paramref name="n"/> (0..16) bits, most significant first.</summary>
    public static int GetBits(ref JpegBitState st, ArrayView<byte> file, int n)
    {
        if (st.Cnt < n) Fill(ref st, file);
        st.Cnt -= n;
        return (int)((st.Buf >> st.Cnt) & ((1u << n) - 1u));
    }

    /// <summary>JPEG "extend": the n-bit value v as a signed number (F.2.2.1).</summary>
    public static int Extend(int v, int n)
    {
        int r = v;
        if (n > 0)
        {
            int half = 1 << (n - 1);
            if (v < half) r = v - (1 << n) + 1;
        }
        return r;
    }

    /// <summary>
    /// Decode one Huffman symbol with the derived table at <paramref name="tb"/> in <paramref name="huff"/>.
    /// Sets <see cref="JpegBitState.Err"/> and returns 0 on a bit pattern that is no code of the table.
    /// </summary>
    public static int DecodeHuff(ref JpegBitState st, ArrayView<byte> file, ArrayView<int> huff, int tb)
    {
        if (st.Cnt < 16) Fill(ref st, file);
        int look = (int)((st.Buf >> (st.Cnt - 9)) & 0x1FFu);
        int e = huff[tb + JpegMeta.HuffLookup + look];
        int sym = 0;
        if (e != 0)
        {
            st.Cnt -= e >> 8;
            sym = e & 0xFF;
        }
        else
        {
            int p16 = (int)((st.Buf >> (st.Cnt - 16)) & 0xFFFFu);
            int found = 0;
            for (int l = 10; l <= 16; l++)
            {
                if (found == 0)
                {
                    int c = p16 >> (16 - l);
                    if (c <= huff[tb + JpegMeta.HuffMaxCode + l])
                    {
                        sym = huff[tb + JpegMeta.HuffSymbols + huff[tb + JpegMeta.HuffValOffset + l] + c];
                        st.Cnt -= l;
                        found = 1;
                    }
                }
            }
            if (found == 0) st.Err = (int)JpegError.BadHuffmanCode;
        }
        return sym;
    }

    private static int GetPred(ref JpegBitState st, int k)
    {
        int p = st.Pred0;
        if (k == 1) p = st.Pred1;
        if (k == 2) p = st.Pred2;
        if (k == 3) p = st.Pred3;
        return p;
    }

    private static void SetPred(ref JpegBitState st, int k, int v)
    {
        if (k == 0) st.Pred0 = v;
        if (k == 1) st.Pred1 = v;
        if (k == 2) st.Pred2 = v;
        if (k == 3) st.Pred3 = v;
    }

    /// <summary>Sequential (baseline / extended) block: DC difference then run-length coded AC coefficients.</summary>
    public static void DecodeBlockSequential(
        ref JpegBitState st, ArrayView<byte> file, ArrayView<short> coef, ArrayView<int> huff, ArrayView<int> consts,
        int off, int dcTb, int acTb, int comp)
    {
        int t = DecodeHuff(ref st, file, huff, dcTb);
        int diff = 0;
        bool hasBits = (t != 0) & (st.Err == 0);
        if (hasBits) diff = Extend(GetBits(ref st, file, t), t);
        int pred = GetPred(ref st, comp) + diff;
        SetPred(ref st, comp, pred);
        coef[off] = (short)pred;

        int k = 1;
        bool go1 = (k < 64) & (st.Err == 0);
        while (go1)
        {
            int rs = DecodeHuff(ref st, file, huff, acTb);
            int r = rs >> 4;
            int s = rs & 15;
            if (s != 0)
            {
                k += r;
                if (k > 63)
                {
                    st.Err = (int)JpegError.BadCoefficientRun;
                }
                else
                {
                    int v = Extend(GetBits(ref st, file, s), s);
                    coef[off + consts[k]] = (short)v;
                    k++;
                }
            }
            else
            {
                if (r == 15)
                {
                    k += 16;
                    // A zero run that leaves the block (more than one past the last coefficient) is corrupt.
                    if (k > 64) st.Err = (int)JpegError.BadCoefficientRun;
                }
                else
                {
                    k = 64;
                }
            }
            go1 = (k < 64) & (st.Err == 0);
        }
    }

    /// <summary>Progressive DC, first scan: coefficient 0 = (predictor + difference) &lt;&lt; Al.</summary>
    public static void DecodeBlockDcFirst(
        ref JpegBitState st, ArrayView<byte> file, ArrayView<short> coef, ArrayView<int> huff,
        int off, int dcTb, int comp, int al)
    {
        int t = DecodeHuff(ref st, file, huff, dcTb);
        int diff = 0;
        bool hasBits = (t != 0) & (st.Err == 0);
        if (hasBits) diff = Extend(GetBits(ref st, file, t), t);
        int pred = GetPred(ref st, comp) + diff;
        SetPred(ref st, comp, pred);
        coef[off] = (short)(pred << al);
    }

    /// <summary>Progressive DC refinement: one more bit of coefficient 0.</summary>
    public static void DecodeBlockDcRefine(
        ref JpegBitState st, ArrayView<byte> file, ArrayView<short> coef, int off, int al)
    {
        int bit = GetBits(ref st, file, 1);
        if (bit != 0) coef[off] = (short)(coef[off] | (1 << al));
    }

    /// <summary>Progressive AC, first scan of a band (single component).</summary>
    public static void DecodeBlockAcFirst(
        ref JpegBitState st, ArrayView<byte> file, ArrayView<short> coef, ArrayView<int> huff, ArrayView<int> consts,
        int off, int acTb, int ss, int se, int al)
    {
        if (st.EobRun > 0)
        {
            st.EobRun--;
        }
        else
        {
            int k = ss;
            bool go2 = (k <= se) & (st.Err == 0);
            while (go2)
            {
                int rs = DecodeHuff(ref st, file, huff, acTb);
                int r = rs >> 4;
                int s = rs & 15;
                if (s != 0)
                {
                    k += r;
                    if (k > se)
                    {
                        st.Err = (int)JpegError.BadCoefficientRun;
                    }
                    else
                    {
                        int v = Extend(GetBits(ref st, file, s), s);
                        coef[off + consts[k]] = (short)(v << al);
                        k++;
                    }
                }
                else
                {
                    if (r == 15)
                    {
                        k += 16;
                        if (k > se + 1) st.Err = (int)JpegError.BadCoefficientRun;
                    }
                    else
                    {
                        int run = 1 << r;
                        if (r != 0) run += GetBits(ref st, file, r);
                        st.EobRun = run - 1;
                        k = 64;
                    }
                }
                go2 = (k <= se) & (st.Err == 0);
            }
        }
    }

    /// <summary>Progressive AC refinement scan of a band (single component).</summary>
    public static void DecodeBlockAcRefine(
        ref JpegBitState st, ArrayView<byte> file, ArrayView<short> coef, ArrayView<int> huff, ArrayView<int> consts,
        int off, int acTb, int ss, int se, int al)
    {
        int p1 = 1 << al;
        int m1 = -1 << al;
        int k = ss;
        if (st.EobRun == 0)
        {
            int more = 1;
            bool go3 = (k <= se) & (more != 0) & (st.Err == 0);
            while (go3)
            {
                int rs = DecodeHuff(ref st, file, huff, acTb);
                int r = rs >> 4;
                int s = rs & 15;
                int newVal = 0;
                if (s != 0)
                {
                    // Only a magnitude of 1 is legal here (the new coefficient is +-1 at this bit position).
                    if (GetBits(ref st, file, 1) != 0) newVal = p1; else newVal = m1;
                }
                else if (r != 15)
                {
                    int run = 1 << r;
                    if (r != 0) run += GetBits(ref st, file, r);
                    st.EobRun = run;
                    more = 0;
                }
                if (more != 0)
                {
                    // Skip over already-nonzero coefficients (appending a correction bit to each) and r zero ones.
                    int advance = 1;
                    bool go4 = (advance != 0) & (k <= se) & (st.Err == 0);
                    while (go4)
                    {
                        int pos = off + consts[k];
                        int cur = coef[pos];
                        if (cur != 0)
                        {
                            if (GetBits(ref st, file, 1) != 0)
                            {
                                if ((cur & p1) == 0)
                                {
                                    if (cur >= 0) coef[pos] = (short)(cur + p1);
                                    else coef[pos] = (short)(cur + m1);
                                }
                            }
                            k++;
                        }
                        else
                        {
                            r--;
                            if (r < 0) advance = 0;
                            else k++;
                        }
                        go4 = (advance != 0) & (k <= se) & (st.Err == 0);
                    }
                    if (s != 0)
                    {
                        if (k > se) st.Err = (int)JpegError.BadCoefficientRun;
                        else coef[off + consts[k]] = (short)newVal;
                    }
                    k++;
                }
                go3 = (k <= se) & (more != 0) & (st.Err == 0);
            }
        }
        bool c5 = (st.EobRun > 0) & (st.Err == 0);
        if (c5)
        {
            // Remaining positions of the band after the end-of-band: only correction bits for nonzero coefficients.
            while (k <= se)
            {
                int pos = off + consts[k];
                int cur = coef[pos];
                if (cur != 0)
                {
                    if (GetBits(ref st, file, 1) != 0)
                    {
                        if ((cur & p1) == 0)
                        {
                            if (cur >= 0) coef[pos] = (short)(cur + p1);
                            else coef[pos] = (short)(cur + m1);
                        }
                    }
                }
                k++;
            }
            st.EobRun--;
        }
    }
}
