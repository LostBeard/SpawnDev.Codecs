// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt). See NOTICE.md for upstream attributions.
//
// VP9 inter frame encoder, the parallel kernels (Plans/PLAN-VP9-Encoder-Inter-Frames.md).
//
// The encoder writes the full decoder's own layouts - mode info records + grid (Vp9Mi), the per-superblock block table
// and eobs (Vp9PLayout), dequantized coefficients (dq) and the frame buffer pool - and runs the decoder's reconstruction
// kernels on them (Vp9ReconKernels.InterPredictKernel, InterResidualKernel). The frame the encoder keeps as its
// reference is therefore what any decoder reconstructs from the bitstream, through the same kernels.
//
// Every block of an inter frame here is an inter block, which reads only the reference frame: no block depends on
// another, so nothing in this file needs a wavefront. One thread per 16x16 position of the picture (a "leaf": the
// whole BLOCK_16X16, or at the picture's edge the part inside it, as the keyframe encoder codes it: the top
// BLOCK_16X8, the left BLOCK_8X16, or the top-left BLOCK_8X8).
//
//   SetupKernel    mode info record + grid + block table entry of each leaf (its MV from the MV buffer)
//   [Vp9ReconKernels.InterPredictKernel: the prediction, into the new frame buffer]
//   ForwardKernel  source - prediction, forward DCT, quantize: levels (the entropy coder's input), dequantized
//                  coefficients and eobs (the decoder's layout), the leaf's skip flag
//   [Vp9ReconKernels.InterResidualKernel: the residual, added in place = the reconstruction]

using ILGPU;
using SpawnDev.Codecs.Video.Vp9.Decoder;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// The encoder's frame arguments: the decoder's reconstruction arguments (<see cref="Vp9RArg"/>, read by its kernels)
/// followed by the encoder's own.
/// </summary>
internal static class Vp9IArg
{
    /// <summary>Source planes Y, U, V: absolute offsets in the source buffer (strides = the current frame's).</summary>
    public const int SrcY = Vp9RArg.Count, SrcU = SrcY + 1, SrcV = SrcY + 2;
    /// <summary>Quantizer steps: Y DC, Y AC, UV DC, UV AC.</summary>
    public const int DqYDc = SrcY + 3, DqYAc = SrcY + 4, DqUvDc = SrcY + 5, DqUvAc = SrcY + 6;
    public const int Count = SrcY + 7;
}

internal static class Vp9InterEncodeKernels
{
    /// <summary>
    /// The leaf at 16x16 position (miRow, miCol): 0 BLOCK_16X16, 1 the top BLOCK_16X8 (bottom edge), 2 the left
    /// BLOCK_8X16 (right edge), 3 the top-left BLOCK_8X8 (the corner).
    /// </summary>
    internal static int LeafShape(int miRow, int miCol, int miRows, int miCols)
    {
        bool hasRows = miRow + 1 < miRows, hasCols = miCol + 1 < miCols;
        int shape = 3;
        if (hasRows & hasCols) shape = 0;
        else if (hasCols) shape = 1;
        else if (hasRows) shape = 2;
        return shape;
    }

    /// <summary>The leaf's slot in its superblock's block table: its z-order index among the 16 positions.</summary>
    internal static int LeafSlot(int miRow, int miCol) =>
        ((((miRow >> 2) & 1) * 2 + ((miCol >> 2) & 1)) << 2) + ((miRow >> 1) & 1) * 2 + ((miCol >> 1) & 1);

    /// <summary>
    /// One thread per 16x16 position inside the picture (launch <c>((miCols + 1) / 2) * ((miRows + 1) / 2)</c>): the
    /// leaf's mode info record (single reference LAST, EIGHTTAP, its MV from <paramref name="mvs"/>: one packed int_mv
    /// per leaf, 1/8 pixel, even components), its grid entries and its block table entry. The table slots are fixed
    /// (the z-order index, <see cref="LeafSlot"/>), not packed in parse order: only the inter kernels read them here.
    /// </summary>
    public static void SetupKernel(Index1D index, ArrayView<int> a, ArrayView<int> mi, ArrayView<int> sbBuf, ArrayView<int> mvs)
    {
        int miCols = a[Vp9RArg.MiCols], miRows = a[Vp9RArg.MiRows];
        int cols16 = (miCols + 1) >> 1;
        int r16 = index / cols16, c16 = index - r16 * cols16;
        int miRow = r16 * 2, miCol = c16 * 2;
        int shape = LeafShape(miRow, miCol, miRows, miCols);
        int wide = shape == 0 | shape == 1 ? 2 : 1, high = shape == 0 | shape == 2 ? 2 : 1;   // in mi (8 pixels)
        int rec = miRow * miCols + miCol, b = rec * Vp9Mi.Stride;
        int grid = a[Vp9RArg.GridOffset];
        for (int y = 0; y < high; y++)
            for (int x = 0; x < wide; x++) mi[grid + rec + y * miCols + x] = rec;
        for (int k = 0; k < Vp9Mi.Stride; k++) mi[b + k] = 0;
        // BLOCK_8X8 3, 8X16 4, 16X8 5, 16X16 6
        mi[b + Vp9Mi.SbType] = 3 + (high - 1) + 2 * (wide - 1);
        mi[b + Vp9Mi.Mode] = Vp9ParseKernels.ZeroMv;   // the entropy pass writes the mode it codes
        mi[b + Vp9Mi.TxSize] = shape == 0 ? 2 : 1;     // TX_16X16, at the edge TX_8X8 (tx_mode ALLOW_32X32)
        mi[b + Vp9Mi.Ref0] = Vp9Mi.LastFrame;
        mi[b + Vp9Mi.Ref1] = Vp9Mi.NoRefFrame;
        mi[b + Vp9Mi.Interp] = (int)Vp9Interp.EightTap;
        mi[b + Vp9Mi.Mv0] = mvs[index];
        int slot = LeafSlot(miRow, miCol);
        mi[b + Vp9Mi.TableSlot] = slot;

        int sb = (miRow >> 3) * a[Vp9RArg.SbCols] + (miCol >> 3);
        int tb = sb * Vp9PLayout.SbStride;
        if (slot == 0) sbBuf[tb + Vp9PLayout.SbBlockCount] = 16;
        int e = tb + Vp9PLayout.SbBlocks + slot * Vp9PLayout.BlkStride;
        int dqSb = sb * a[Vp9RArg.DqStride];
        sbBuf[e + Vp9PLayout.BlkRec] = rec;
        sbBuf[e + Vp9PLayout.BlkBwl] = wide;   // log2 of the width in 4x4 units: 16 pixels 2, 8 pixels 1
        sbBuf[e + Vp9PLayout.BlkBhl] = high;
        // a leaf's coefficients: 256 luma, 64 per chroma plane; its eobs: up to 2 transforms per plane
        sbBuf[e + Vp9PLayout.BlkDqY] = dqSb + slot * 256;
        sbBuf[e + Vp9PLayout.BlkDqU] = dqSb + a[Vp9RArg.DqU] + slot * 64;
        sbBuf[e + Vp9PLayout.BlkDqV] = dqSb + a[Vp9RArg.DqV] + slot * 64;
        sbBuf[e + Vp9PLayout.BlkEobY] = slot * 4;
        sbBuf[e + Vp9PLayout.BlkEobU] = slot * 4;
        sbBuf[e + Vp9PLayout.BlkEobV] = slot * 4;
    }

    /// <summary>
    /// One thread per 16x16 position inside the picture, after the prediction is in the new frame buffer: per transform
    /// block of the leaf (the decoder's order: luma, U, V; two per plane for the 16X8 / 8X16 leaves) the residual
    /// source - prediction, the forward DCT and the quantizer. Writes the levels (raster order, at the decoder's
    /// coefficient positions) for the entropy coder, the dequantized coefficients and the eobs for
    /// <see cref="Vp9ReconKernels.InterResidualKernel"/>, and the leaf's skip flag (no coefficient at all).
    /// One call site per transform size (Chrome's D3D compiler inlines every call).
    /// </summary>
    public static void ForwardKernel(Index1D index, ArrayView<byte> pool, ArrayView<byte> src, ArrayView<int> a, ArrayView<int> mi,
        ArrayView<int> sbBuf, ArrayView<int> dq, ArrayView<short> levels, ArrayView<ushort> scans)
    {
        int miCols = a[Vp9RArg.MiCols], miRows = a[Vp9RArg.MiRows];
        int cols16 = (miCols + 1) >> 1;
        int r16 = index / cols16, c16 = index - r16 * cols16;
        int miRow = r16 * 2, miCol = c16 * 2;
        int shape = LeafShape(miRow, miCol, miRows, miCols);
        int rec = miRow * miCols + miCol, b = rec * Vp9Mi.Stride;
        int sb = (miRow >> 3) * a[Vp9RArg.SbCols] + (miCol >> 3);
        int tb = sb * Vp9PLayout.SbStride;
        int e = tb + Vp9PLayout.SbBlocks + LeafSlot(miRow, miCol) * Vp9PLayout.BlkStride;

        // 2560 bytes of function-scope arrays per thread: under the 4096 that SpawnDev.ILGPU's NVIDIA flush starts at
        // (SpawnDev.ILGPU issue 10; the device is lost from 6608)
        var residual = LocalMemory.Allocate<short>(256);
        var coefs = LocalMemory.Allocate<int>(256);
        var scratch = LocalMemory.Allocate<int>(256);

        bool two = shape == 1 | shape == 2;
        int jobs = two ? 6 : 3;
        int any = 0;
        for (int j = 0; j < jobs; j++)
        {
            int plane = two ? j >> 1 : j;
            int half = two ? j & 1 : 0;
            int across = shape == 1 ? half : 0, down = shape == 2 ? half : 0;
            int s = plane == 0 ? 0 : 1;
            int tx = (shape == 0 ? 2 : 1) - s;   // luma TX_16X16 / TX_8X8, chroma TX_8X8 / TX_4X4
            int n = 4 << tx, count = n * n;
            int stride = a[Vp9RArg.Cur + plane * 4 + 1];
            int at = (((miRow * 8) >> s) + down * n) * stride + ((miCol * 8) >> s) + across * n;
            int predAt = a[Vp9RArg.Cur + plane * 4] + at, srcAt = a[Vp9IArg.SrcY + plane] + at;
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                    residual[r * n + c] = (short)(src[srcAt + r * stride + c] - pool[predAt + r * stride + c]);
            int scanAt;
            if (tx == 2)
            {
                Vp9ForwardDct16x16Gpu.Forward16x16(residual, 0, 16, coefs, 0, scratch);
                scanAt = Vp9KeyframeConstantsGpu.Scan16x16Offset;
            }
            else if (tx == 1)
            {
                Vp9ForwardDct8x8Gpu.Forward8x8(residual, 0, 8, coefs, 0, scratch);
                scanAt = Vp9KeyframeConstantsGpu.Scan8x8Offset;
            }
            else
            {
                Vp9ForwardDct4x4Gpu.Transform(residual, 0, 4, coefs, 0, scratch, 0);
                scanAt = Vp9KeyframeConstantsGpu.Scan4x4Offset;
            }
            int dcQ = a[plane == 0 ? Vp9IArg.DqYDc : Vp9IArg.DqUvDc], acQ = a[plane == 0 ? Vp9IArg.DqYAc : Vp9IArg.DqUvAc];
            Vp9ForwardQuantizerGpu.QuantizeBlock(coefs, 0, count, dcQ, acQ);
            // the half-th transform of the plane: its coefficients follow the first one's (16 << 2 tx = count each)
            int dqAt = sbBuf[e + Vp9PLayout.BlkDqY + plane] + half * count;
            for (int i = 0; i < count; i++)
            {
                int level = coefs[i];
                levels[dqAt + i] = (short)level;
                // the decoder's dequantization (vp9_decode_block_tokens: (val * dqv) >> dq_shift, 0 below 32x32)
                dq[dqAt + i] = (short)(level * (i == 0 ? dcQ : acQ));
            }
            // eob = the tokens the entropy coder writes: up to the last nonzero level in scan order
            int eob = 0;
            for (int i = 0; i < count; i++)
                if (coefs[(int)scans[scanAt + i]] != 0) eob = i + 1;
            sbBuf[tb + Vp9PLayout.SbEobY + plane * 256 + sbBuf[e + Vp9PLayout.BlkEobY + plane] + half] = eob;
            any |= eob;
        }
        // libvpx: an inter block of 8x8 or more without coefficients is skipped (the skip flag, coded by the entropy pass)
        mi[b + Vp9Mi.Skip] = any == 0 ? 1 : 0;
    }
}
