// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// The GIF decode kernels. Every stage that touches the file bytes or the pixels runs on the accelerator; the
// host only dispatches and reads back the metadata block, the frame table and per-frame status ints.
//
//   1. ScanKernel      (1 thread)            header, logical screen, global colour table, then every block:
//                                            Graphic Control Extension, Netscape loop extension, image
//                                            descriptors (frame table); other extensions are skipped safely.
//   2. LzwKernel       (1 thread per frame)  variable-width LZW (2..12 bits, clear / end codes, deferred clear,
//                                            full 4096-entry table) reading the data sub-blocks in place into a
//                                            palette-index buffer. LZW is a sequential dependency chain, so a
//                                            frame is one thread; frames of a batch decode in parallel.
//   3. CompositeKernel (1 thread per pixel)  one STEP of the compositing plan the host built from the frame table:
//                                            fill a rectangle (or the whole canvas) with a colour, save / restore a
//                                            frame rectangle (disposal), or draw a frame (de-interlace, palette
//                                            lookup, transparent index). The step carries every rule that differs
//                                            between the two compositing modes, so the kernel has no mode switch.
//
// Compositing modes (GifCompositing; the host plans the steps, see GifDecoderGpu):
//   Pillow (default): bit-exact with Pillow 13 (GifImagePlugin, default LOADING_STRATEGY, every frame read with
//     Image.seek + convert("RGBA")). Pillow's rules, all reproduced:
//       * the first frame is an indexed canvas: filled with its transparent index (else index 0), so the part a
//         smaller first frame does not cover is palette colour 0 (opaque) or transparent; an indexed pixel equal to
//         the first frame's transparent index has alpha 0 but KEEPS its palette colour (rgb is not zeroed);
//       * from the second frame on the canvas is RGBA (if the first frame had a transparent index) or RGB (opaque)
//         and a pixel equal to the current frame's transparent index leaves the canvas unchanged;
//       * the disposal method is STICKY: a frame whose graphic control extension says 0, or that has none, keeps
//         the last non-zero method seen; 4..7 act like 3;
//       * disposal 2 fills the frame rectangle with the frame's transparent index colour (alpha 0) if it has one,
//         else with the logical screen background colour (opaque; index 0 when there is no global table);
//         disposal 3 restores the rectangle, except on the FIRST frame, which is left in place unless it has a
//         transparent index (then the rectangle becomes transparent);
//       * a colour table that is the identity grayscale ramp is dropped by Pillow (mode "L"): an index is then its
//         own gray level, even beyond the table; a palette index beyond a real table is black.
//     Not reproduced (Pillow grows the canvas, ours is fixed): frames that extend beyond the logical screen are
//     clipped. A first frame with an identity palette followed by a frame with a real palette is erratic in Pillow
//     (it errors or returns black); here every frame uses its own palette.
//   Web: what browsers show. The canvas starts transparent (0, 0, 0, 0); disposal 2 clears the rectangle to
//     transparent (the background colour index is ignored); a pixel equal to the transparent index leaves the
//     canvas unchanged, every other pixel is opaque; disposal 3 restores the rectangle (also on the first frame,
//     where the canvas was transparent); 4..7 behave like 0; an index beyond the table is opaque black.
// Common to both: frame pixels outside the logical screen are clipped.
//
// Loop conditions are bool locals built with non-short-circuit & / | (a compound `while (a && b)` hung the
// WebGPU device).

using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.Codecs.Images.Gif;

/// <summary>Running state of one LZW bit reader over the data sub-blocks.</summary>
internal struct LzwBits
{
    /// <summary>Next file byte to read (a sub-block length byte when <see cref="BlockLeft"/> is 0).</summary>
    public int Pos;
    /// <summary>Payload bytes left in the current sub-block.</summary>
    public int BlockLeft;
    /// <summary>Bit buffer (low bits are the next bits of the stream).</summary>
    public uint Buf;
    /// <summary>Valid bits in <see cref="Buf"/>.</summary>
    public int Cnt;
    /// <summary>1 once the block terminator was reached with no more data.</summary>
    public int Err;
}

/// <summary>Loads and launches the GIF decode kernels.</summary>
public sealed class GifKernels : IDisposable
{
    /// <summary>Ints of scratch one LZW thread needs (the 4096-entry code table).</summary>
    public const int LzwScratchInts = 4096;
    /// <summary>Ints per LZW plan entry: frame, index offset, status, pixels produced.</summary>
    public const int PlanStride = 4;

    /// <summary>LZW status: decoded all pixels.</summary>
    public const int LzwOk = 0;
    /// <summary>LZW status: a code was not in the table.</summary>
    public const int LzwBadCode = 1;
    /// <summary>LZW status: the end code arrived before all pixels were produced.</summary>
    public const int LzwEarlyEnd = 2;
    /// <summary>LZW status: the data sub-blocks ended before all pixels were produced.</summary>
    public const int LzwDataEnded = 3;

    /// <summary>Ints per compositing step.</summary>
    public const int StepStride = 10;

    // ---- step fields (relative to step * StepStride) ----
    /// <summary>Frame whose rectangle, palette and interlacing the step uses.</summary>
    public const int StepFrame = 0;
    /// <summary>One of the <c>Op*</c> constants.</summary>
    public const int StepOp = 1;
    /// <summary>OpDraw: where the frame's palette indices start in the index buffer.</summary>
    public const int StepIndexOffset = 2;
    /// <summary>OpDraw: an index equal to this leaves the canvas pixel alone (-1 = none).</summary>
    public const int StepSkip = 3;
    /// <summary>OpDraw: an index equal to this is written with alpha 0 (-1 = none).</summary>
    public const int StepAlphaZero = 4;
    /// <summary>OpFill*: the palette index whose colour fills the area.</summary>
    public const int StepFillIdx = 5;
    /// <summary>OpFill*: one of the <c>Fill*</c> rules.</summary>
    public const int StepFillRule = 6;
    /// <summary>OpFill*: 1 = the fill has alpha 0 (its colour is kept), else alpha 255.</summary>
    public const int StepFillAlphaZero = 7;
    /// <summary>1 = the canvas is opaque RGB: every written pixel gets alpha 255.</summary>
    public const int StepOpaque = 8;
    /// <summary>1 = a frame whose colour table is the identity ramp maps an index to its own gray level.</summary>
    public const int StepGrayRule = 9;

    /// <summary>Op: fill the frame rectangle (clipped to the canvas).</summary>
    public const int OpFillRect = 0;
    /// <summary>Op: copy the frame rectangle of the canvas to the saved copy.</summary>
    public const int OpSave = 1;
    /// <summary>Op: copy the frame rectangle of the saved copy back to the canvas.</summary>
    public const int OpRestore = 2;
    /// <summary>Op: draw the frame.</summary>
    public const int OpDraw = 3;
    /// <summary>Op: fill the whole canvas.</summary>
    public const int OpFillCanvas = 4;

    /// <summary>Fill rule: the index is looked up in the frame's table; beyond the table is black.</summary>
    public const int FillThroughPalette = 0;
    /// <summary>Fill rule: like <see cref="FillThroughPalette"/> but an index beyond the table means entry 0 (Pillow's _rgb).</summary>
    public const int FillEntryZeroBeyond = 1;
    /// <summary>Fill rule: transparent black (0, 0, 0, 0).</summary>
    public const int FillTransparent = 2;

    private readonly Action<Index1D, ArrayView<byte>, int, ArrayView<int>> _scan;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<byte>, ArrayView<int>> _lzw;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, int> _composite;

    /// <summary>Compile the kernels.</summary>
    public GifKernels(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _scan = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, int, ArrayView<int>>(ScanKernel);
        _lzw = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<byte>, ArrayView<int>>(LzwKernel);
        _composite = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, int>(CompositeKernel);
    }

    /// <summary>Scan the whole block structure into <paramref name="meta"/> (header block + frame table).</summary>
    public void Scan(ArrayView<byte> file, int fileLength, ArrayView<int> meta)
    {
        _scan(1, file, fileLength, meta);
    }

    /// <summary>
    /// Decode <paramref name="frameCount"/> frames' LZW data in parallel. <paramref name="plan"/> holds
    /// <see cref="PlanStride"/> ints per frame: frame index, offset of the frame's first index byte in
    /// <paramref name="indices"/>; the kernel writes the status and the pixels produced.
    /// </summary>
    public void Lzw(ArrayView<byte> file, ArrayView<int> meta, ArrayView<int> plan, ArrayView<byte> indices,
        ArrayView<int> scratch, int frameCount)
    {
        _lzw(frameCount, file, meta, plan, indices, scratch);
    }

    /// <summary>
    /// Run compositing step <paramref name="step"/> of <paramref name="steps"/> (<see cref="StepStride"/> ints each).
    /// <paramref name="threads"/> = the step's rectangle clipped to the canvas (width x height), or the whole canvas
    /// for <see cref="OpFillCanvas"/>.
    /// </summary>
    public void Composite(ArrayView<byte> file, ArrayView<int> meta, ArrayView<int> steps, ArrayView<byte> indices,
        ArrayView<byte> canvas, ArrayView<byte> saved, int step, int threads)
    {
        _composite(threads, file, meta, steps, indices, canvas, saved, step);
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    private static int Le16(ArrayView<byte> f, int p) => f[p] | (f[p + 1] << 8);

    /// <summary>
    /// Position just after the data sub-block chain that starts at <paramref name="pos"/> (the length byte of
    /// the first sub-block), or -1 when the chain runs past the file.
    /// </summary>
    private static int SkipSubBlocks(ArrayView<byte> file, int fileLen, int pos)
    {
        int p = pos;
        int result = -1;
        bool go = true;
        while (go)
        {
            if (p >= fileLen)
            {
                go = false;
            }
            else
            {
                int len = file[p];
                p += 1 + len;
                if (p > fileLen)
                {
                    go = false;
                }
                else if (len == 0)
                {
                    result = p;
                    go = false;
                }
            }
        }
        return result;
    }

    /// <summary>Total payload bytes of the (already validated) sub-block chain starting at <paramref name="pos"/>.</summary>
    private static int PayloadBytes(ArrayView<byte> file, int fileLen, int pos)
    {
        int p = pos;
        int total = 0;
        bool go = true;
        while (go)
        {
            int len = file[p];
            total += len;
            p += 1 + len;
            go = (len != 0) & (p < fileLen);
        }
        return total;
    }

    /// <summary>1 when entry i of the <paramref name="count"/>-entry table at <paramref name="off"/> is (i, i, i) for every i.</summary>
    private static int IsIdentityPalette(ArrayView<byte> file, int off, int count)
    {
        int bad = 0;
        for (int i = 0; i < count; i++)
        {
            int e = off + i * 3;
            bad |= file[e] ^ i;
            bad |= file[e + 1] ^ i;
            bad |= file[e + 2] ^ i;
        }
        return bad == 0 ? 1 : 0;
    }

    private static void SetFrameFlag(ArrayView<int> meta, int fe, int bit)
    {
        meta[fe + GifMeta.FrameFlags] = meta[fe + GifMeta.FrameFlags] | bit;
    }

    // ------------------------------------------------------------------
    // 1. Scan
    // ------------------------------------------------------------------

    private static void ScanKernel(Index1D _, ArrayView<byte> file, int fileLen, ArrayView<int> meta)
    {
        for (int i = 0; i < GifMeta.Size; i++) meta[i] = 0;
        meta[GifMeta.GlobalPaletteOffset] = -1;
        meta[GifMeta.LoopCount] = -1;
        meta[GifMeta.FileLength] = fileLen;

        int err = 0;
        if (fileLen < 6)
        {
            err = (int)GifError.BadSignature;
        }
        else
        {
            int bad = file[0] ^ 0x47;
            bad |= file[1] ^ 0x49;
            bad |= file[2] ^ 0x46;
            bad |= file[3] ^ 0x38;
            bad |= file[5] ^ 0x61;
            bool v = (file[4] == 0x37) | (file[4] == 0x39);
            if ((bad != 0) | !v) err = (int)GifError.BadSignature;
        }
        if ((err == 0) & (fileLen < 13)) err = (int)GifError.Truncated;

        int pos = 13;
        if (err == 0)
        {
            meta[GifMeta.Width] = Le16(file, 6);
            meta[GifMeta.Height] = Le16(file, 8);
            int packed = file[10];
            meta[GifMeta.BackgroundIndex] = file[11];
            if ((packed & 0x80) != 0)
            {
                int count = 2 << (packed & 7);
                if (pos + 3 * count > fileLen)
                {
                    err = (int)GifError.Truncated;
                    meta[GifMeta.ErrorOffset] = pos;
                }
                else
                {
                    meta[GifMeta.GlobalPaletteOffset] = pos;
                    meta[GifMeta.GlobalPaletteCount] = count;
                    pos += 3 * count;
                }
            }
        }

        // Graphic Control Extension state waiting for the next image.
        bool hasGce = false;
        int gceDisposal = 0;
        int gceTransFlag = 0;
        int gceTransIndex = 0;
        int gceDelay = 0;
        int frames = 0;
        bool go = err == 0;
        while (go)
        {
            if (pos >= fileLen)
            {
                // A missing trailer is tolerated once at least one frame is complete.
                if (frames == 0)
                {
                    err = (int)GifError.Truncated;
                    meta[GifMeta.ErrorOffset] = pos;
                }
                go = false;
            }
            else
            {
                int b = file[pos];
                if (b == 0x3B)
                {
                    go = false;
                }
                else if (b == 0x21)
                {
                    if (pos + 2 > fileLen)
                    {
                        err = (int)GifError.Truncated;
                        meta[GifMeta.ErrorOffset] = pos;
                        go = false;
                    }
                    else
                    {
                        int label = file[pos + 1];
                        int first = pos + 2;     // first sub-block length byte
                        int end = SkipSubBlocks(file, fileLen, first);
                        if (end < 0)
                        {
                            err = (int)GifError.Truncated;
                            meta[GifMeta.ErrorOffset] = pos;
                            go = false;
                        }
                        else
                        {
                            int len0 = file[first];
                            if (label == 0xF9)
                            {
                                // GCE: packed, delay (LE16), transparent index. Two GCEs before one image (never
                                // written by an encoder) merge the way Pillow merges them: a later "unspecified"
                                // disposal never overrides a specified one, and a later GCE cannot clear the
                                // transparent index an earlier one set.
                                if (len0 >= 4)
                                {
                                    int gp = file[first + 1];
                                    hasGce = true;
                                    int dbits = (gp >> 2) & 7;
                                    if (dbits != 0) gceDisposal = dbits;
                                    if ((gp & 1) != 0)
                                    {
                                        gceTransFlag = 1;
                                        gceTransIndex = file[first + 4];
                                    }
                                    gceDelay = Le16(file, first + 2);
                                }
                            }
                            else if (label == 0xFF)
                            {
                                // Application extension "NETSCAPE2.0" (or "ANIMEXTS1.0") before the first image
                                // carries the loop count in the next sub-block: [3, 1, count LE16].
                                if ((len0 >= 11) & (frames == 0))
                                {
                                    bool netscape = (file[first + 1] == 0x4E) & (file[first + 2] == 0x45) & (file[first + 3] == 0x54)
                                                  & (file[first + 4] == 0x53) & (file[first + 5] == 0x43) & (file[first + 6] == 0x41)
                                                  & (file[first + 7] == 0x50) & (file[first + 8] == 0x45) & (file[first + 9] == 0x32)
                                                  & (file[first + 10] == 0x2E) & (file[first + 11] == 0x30);
                                    bool animexts = (file[first + 1] == 0x41) & (file[first + 2] == 0x4E) & (file[first + 3] == 0x49)
                                                  & (file[first + 4] == 0x4D) & (file[first + 5] == 0x45) & (file[first + 6] == 0x58)
                                                  & (file[first + 7] == 0x54) & (file[first + 8] == 0x53) & (file[first + 9] == 0x31)
                                                  & (file[first + 10] == 0x2E) & (file[first + 11] == 0x30);
                                    int second = first + 1 + len0;
                                    bool room = second + 4 <= fileLen;
                                    if ((netscape | animexts) & room)
                                    {
                                        bool isLoop = (file[second] >= 3) & (file[second + 1] == 1);
                                        if (isLoop) meta[GifMeta.LoopCount] = Le16(file, second + 2);
                                    }
                                }
                            }
                            pos = end;
                        }
                    }
                }
                else if (b == 0x2C)
                {
                    if (pos + 10 > fileLen)
                    {
                        err = (int)GifError.Truncated;
                        meta[GifMeta.ErrorOffset] = pos;
                        go = false;
                    }
                    else
                    {
                        int fx = Le16(file, pos + 1);
                        int fy = Le16(file, pos + 3);
                        int fw = Le16(file, pos + 5);
                        int fh = Le16(file, pos + 7);
                        int fpacked = file[pos + 9];
                        int p = pos + 10;
                        int palOff = meta[GifMeta.GlobalPaletteOffset];
                        int palCount = meta[GifMeta.GlobalPaletteCount];
                        bool ok = true;
                        if ((fpacked & 0x80) != 0)
                        {
                            int count = 2 << (fpacked & 7);
                            if (p + 3 * count > fileLen)
                            {
                                ok = false;
                            }
                            else
                            {
                                palOff = p;
                                palCount = count;
                                p += 3 * count;
                            }
                        }
                        // LZW minimum code size byte, then the data sub-blocks.
                        int dataEnd = -1;
                        if (ok & (p < fileLen)) dataEnd = SkipSubBlocks(file, fileLen, p + 1);
                        if (dataEnd < 0)
                        {
                            err = (int)GifError.Truncated;
                            meta[GifMeta.ErrorOffset] = pos;
                            go = false;
                        }
                        else if (frames >= GifMeta.MaxFrames)
                        {
                            err = (int)GifError.TooManyFrames;
                            meta[GifMeta.ErrorOffset] = pos;
                            go = false;
                        }
                        else
                        {
                            int fe = GifMeta.FrameTable + frames * GifMeta.FrameStride;
                            meta[fe + GifMeta.FrameX] = fx;
                            meta[fe + GifMeta.FrameY] = fy;
                            meta[fe + GifMeta.FrameW] = fw;
                            meta[fe + GifMeta.FrameH] = fh;
                            meta[fe + GifMeta.FrameFlags] = 0;
                            if ((fpacked & 0x40) != 0) SetFrameFlag(meta, fe, 1);
                            if (hasGce & (gceTransFlag != 0)) SetFrameFlag(meta, fe, 2);
                            meta[fe + GifMeta.FrameTransparentIndex] = gceTransIndex;
                            meta[fe + GifMeta.FrameDelayCs] = gceDelay;
                            meta[fe + GifMeta.FrameDisposal] = gceDisposal;
                            meta[fe + GifMeta.FramePaletteOffset] = palOff;
                            meta[fe + GifMeta.FramePaletteCount] = palCount;
                            meta[fe + GifMeta.FrameMinCodeSize] = file[p];
                            meta[fe + GifMeta.FrameDataOffset] = p + 1;
                            meta[fe + GifMeta.FramePaletteIdentity] = 0;
                            if (palOff >= 0) meta[fe + GifMeta.FramePaletteIdentity] = IsIdentityPalette(file, palOff, palCount);
                            meta[fe + GifMeta.FrameDataBytes] = PayloadBytes(file, fileLen, p + 1);
                            frames++;
                            hasGce = false;
                            gceDisposal = 0;
                            gceTransFlag = 0;
                            gceTransIndex = 0;
                            gceDelay = 0;
                            pos = dataEnd;
                        }
                    }
                }
                else
                {
                    err = (int)GifError.BadBlock;
                    meta[GifMeta.ErrorOffset] = pos;
                    go = false;
                }
            }
        }
        meta[GifMeta.FrameCount] = frames;
        meta[GifMeta.Error] = err;
    }

    // ------------------------------------------------------------------
    // 2. LZW
    // ------------------------------------------------------------------

    /// <summary>Next <paramref name="width"/>-bit code (LSB first), or -1 when the sub-blocks ran out.</summary>
    private static int NextCode(ref LzwBits st, ArrayView<byte> file, int width)
    {
        bool need = (st.Cnt < width) & (st.Err == 0);
        while (need)
        {
            if (st.BlockLeft == 0)
            {
                int len = file[st.Pos];
                st.Pos++;
                st.BlockLeft = len;
                if (len == 0) st.Err = 1;
            }
            if (st.Err == 0)
            {
                st.Buf |= (uint)file[st.Pos] << st.Cnt;
                st.Pos++;
                st.BlockLeft--;
                st.Cnt += 8;
            }
            need = (st.Cnt < width) & (st.Err == 0);
        }
        int code = -1;
        if (st.Err == 0)
        {
            code = (int)(st.Buf & ((1u << width) - 1u));
            st.Buf >>= width;
            st.Cnt -= width;
        }
        return code;
    }

    private static void LzwKernel(
        Index1D index, ArrayView<byte> file, ArrayView<int> meta, ArrayView<int> plan,
        ArrayView<byte> indices, ArrayView<int> scratch)
    {
        int t = index.X;
        int frame = plan[t * PlanStride];
        int outBase = plan[t * PlanStride + 1];
        int fe = GifMeta.FrameTable + frame * GifMeta.FrameStride;
        int total = meta[fe + GifMeta.FrameW] * meta[fe + GifMeta.FrameH];
        int minCode = meta[fe + GifMeta.FrameMinCodeSize];
        int sb = t * LzwScratchInts;

        int clear = 1 << minCode;
        int eoi = clear + 1;
        int width = minCode + 1;
        int next = clear + 2;
        int prev = -1;
        int firstChar = 0;
        int produced = 0;
        int status = LzwOk;
        var st = new LzwBits { Pos = meta[fe + GifMeta.FrameDataOffset], BlockLeft = 0, Buf = 0, Cnt = 0, Err = 0 };

        bool go = total > 0;
        while (go)
        {
            int code = NextCode(ref st, file, width);
            if (code < 0)
            {
                status = LzwDataEnded;
            }
            else if (code == clear)
            {
                width = minCode + 1;
                next = clear + 2;
                prev = -1;
            }
            else if (code == eoi)
            {
                status = LzwEarlyEnd;
            }
            else
            {
                bool bad = false;
                if (prev < 0)
                {
                    if (code >= clear) bad = true;
                }
                else if (code > next)
                {
                    bad = true;
                }
                if (bad)
                {
                    status = LzwBadCode;
                }
                else if (prev < 0)
                {
                    // First code after a clear: a literal.
                    indices[outBase + produced] = (byte)code;
                    produced++;
                    firstChar = code;
                    prev = code;
                }
                else
                {
                    int prevLen = 1;
                    if (prev >= clear) prevLen = (int)((uint)scratch[sb + prev] >> 20);
                    // code == next (KwKwK): the new entry is prev + the first character of prev.
                    if (code == next)
                        scratch[sb + next] = prev | (firstChar << 12) | ((prevLen + 1) << 20);

                    // Write string(code), last character first, walking the prefix chain.
                    int len = 1;
                    if (code >= clear) len = (int)((uint)scratch[sb + code] >> 20);
                    int c = code;
                    int k = len - 1;
                    int ch = 0;
                    bool walk = true;
                    while (walk)
                    {
                        if (c < clear)
                        {
                            ch = c;
                        }
                        else
                        {
                            int e = scratch[sb + c];
                            ch = (e >> 12) & 0xFF;
                            c = e & 0xFFF;
                        }
                        int dst = produced + k;
                        if (dst < total) indices[outBase + dst] = (byte)ch;
                        k--;
                        walk = k >= 0;
                    }
                    int room = total - produced;
                    produced += len < room ? len : room;

                    if (next < 4096)
                    {
                        if (code != next)
                            scratch[sb + next] = prev | (ch << 12) | ((prevLen + 1) << 20);
                        next++;
                        if ((next == (1 << width)) & (width < 12)) width++;
                    }
                    firstChar = ch;
                    prev = code;
                }
            }
            go = (status == LzwOk) & (produced < total);
        }
        plan[t * PlanStride + 2] = status;
        plan[t * PlanStride + 3] = produced;
    }

    // ------------------------------------------------------------------
    // 3. Compositing
    // ------------------------------------------------------------------

    /// <summary>Source row (in the stored, interlaced order) of output row <paramref name="y"/> of an <paramref name="h"/> row frame.</summary>
    private static int InterlacedSourceRow(int y, int h)
    {
        int n1 = (h + 7) >> 3;
        int n2 = 0;
        if (h > 4) n2 = (h - 4 + 7) >> 3;
        int n3 = 0;
        if (h > 2) n3 = (h - 2 + 3) >> 2;
        int m = y & 7;
        int r;
        if (m == 0) r = y >> 3;
        else if (m == 4) r = n1 + ((y - 4) >> 3);
        else if ((y & 3) == 2) r = n1 + n2 + ((y - 2) >> 2);
        else r = n1 + n2 + n3 + ((y - 1) >> 1);
        return r;
    }

    /// <summary>
    /// Colour (0xRRGGBB) of palette index <paramref name="idx"/> in the table of frame entry <paramref name="fe"/>.
    /// With <paramref name="gray"/> set, an identity-ramp table maps an index to its own gray level (even beyond the
    /// table); otherwise an index beyond the table is black, or entry 0 when <paramref name="beyondIsEntryZero"/>.
    /// </summary>
    private static int PaletteColor(ArrayView<byte> file, ArrayView<int> meta, int fe, int idx, int gray, bool beyondIsEntryZero)
    {
        int count = meta[fe + GifMeta.FramePaletteCount];
        int color = 0;
        bool identity = (gray != 0) & (meta[fe + GifMeta.FramePaletteIdentity] != 0);
        if (identity)
        {
            color = idx * 0x010101;
        }
        else
        {
            int e = idx;
            if ((idx >= count) & beyondIsEntryZero) e = 0;
            if (e < count)
            {
                int o = meta[fe + GifMeta.FramePaletteOffset] + e * 3;
                color = (file[o] << 16) | (file[o + 1] << 8) | file[o + 2];
            }
        }
        return color;
    }

    private static void CompositeKernel(
        Index1D index, ArrayView<byte> file, ArrayView<int> meta, ArrayView<int> steps, ArrayView<byte> indices,
        ArrayView<byte> canvas, ArrayView<byte> saved, int step)
    {
        int sb = step * StepStride;
        int op = steps[sb + StepOp];
        int fe = GifMeta.FrameTable + steps[sb + StepFrame] * GifMeta.FrameStride;
        int fw = meta[fe + GifMeta.FrameW];
        int fh = meta[fe + GifMeta.FrameH];
        int cw = meta[GifMeta.Width];

        int i = index.X;
        int fy = 0;
        int fx = 0;
        int p = i * 4;
        if (op != OpFillCanvas)
        {
            // The dispatch covers the frame rectangle clipped to the canvas.
            int x0 = meta[fe + GifMeta.FrameX];
            int y0 = meta[fe + GifMeta.FrameY];
            int clipW = fw;
            if (x0 + clipW > cw) clipW = cw - x0;
            fy = i / clipW;
            fx = i - fy * clipW;
            p = ((y0 + fy) * cw + x0 + fx) * 4;
        }

        if (op == OpSave)
        {
            saved[p] = canvas[p]; saved[p + 1] = canvas[p + 1]; saved[p + 2] = canvas[p + 2]; saved[p + 3] = canvas[p + 3];
        }
        else if (op == OpRestore)
        {
            canvas[p] = saved[p]; canvas[p + 1] = saved[p + 1]; canvas[p + 2] = saved[p + 2]; canvas[p + 3] = saved[p + 3];
        }
        else if (op == OpDraw)
        {
            int row = fy;
            if ((meta[fe + GifMeta.FrameFlags] & 1) != 0) row = InterlacedSourceRow(fy, fh);
            int idx = indices[steps[sb + StepIndexOffset] + row * fw + fx];
            if (idx != steps[sb + StepSkip])
            {
                int color = PaletteColor(file, meta, fe, idx, steps[sb + StepGrayRule], false);
                int a = 255;
                if ((idx == steps[sb + StepAlphaZero]) & (steps[sb + StepOpaque] == 0)) a = 0;
                canvas[p] = (byte)(color >> 16);
                canvas[p + 1] = (byte)(color >> 8);
                canvas[p + 2] = (byte)color;
                canvas[p + 3] = (byte)a;
            }
        }
        else
        {
            // OpFillRect / OpFillCanvas
            int rule = steps[sb + StepFillRule];
            int color = 0;
            int a = 0;
            if (rule != FillTransparent)
            {
                color = PaletteColor(file, meta, fe, steps[sb + StepFillIdx], steps[sb + StepGrayRule], rule == FillEntryZeroBeyond);
                a = 255;
                if ((steps[sb + StepFillAlphaZero] != 0) & (steps[sb + StepOpaque] == 0)) a = 0;
            }
            canvas[p] = (byte)(color >> 16);
            canvas[p + 1] = (byte)(color >> 8);
            canvas[p + 2] = (byte)color;
            canvas[p + 3] = (byte)a;
        }
    }

    /// <summary>No-op (kernels are owned by the accelerator).</summary>
    public void Dispose() { }
}
