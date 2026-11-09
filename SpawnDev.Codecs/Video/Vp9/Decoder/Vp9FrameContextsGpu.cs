// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt). See NOTICE.md for upstream attributions.
//
// VP9 frame contexts on the accelerator + the compressed header. A port of libvpx v1.17.0 (BSD-3-Clause):
//   vp9/decoder/vp9_decodeframe.c read_compressed_header (+ read_tx_mode, read_tx_mode_probs, read_coef_probs,
//     read_switchable_interp_probs, read_inter_mode_probs, read_frame_reference_mode(_probs), read_mv_probs)
//   vp9/decoder/vp9_dsubexp.c     vp9_diff_update_prob (decode_term_subexp, inv_remap_prob)
//   vp9/common/vp9_pred_common.c  vp9_compound_reference_allowed, vp9_setup_compound_reference_mode
//   vp9/common/vp9_entropymode.c  the context resets of vp9_setup_past_independence
// The bool decoder is Vp8BoolDecoderGpu: VP8's split 1 + (((range - 1) * p) >> 8) equals libvpx VP9's
// (range * p + 256 - p) >> 8, and both read zeros past the end of the data.
//
// Storage: 5 contexts of Vp9Ctx.Bytes probabilities, one int each (no sub-word stores from parallel copies):
// slots 0..3 = the saved frame_contexts[4], slot 4 = the current context (libvpx cm->fc).

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Vp8;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp9.Decoder;

/// <summary>Offsets of libvpx FRAME_CONTEXT's probability arrays (struct order, one entry per vpx_prob).</summary>
internal static class Vp9Ctx
{
    public const int Bytes = 2039;
    public const int YMode = 0;              // [4 BLOCK_SIZE_GROUPS][9]
    public const int UvMode = 36;            // [10][9]
    public const int Partition = 126;        // [16][3]
    public const int Coef = 174;             // [4 tx][2 plane][2 ref][6 band][6 ctx][3]
    public const int SwitchableInterp = 1902;// [4][2]
    public const int InterMode = 1910;       // [7][3]
    public const int IntraInter = 1931;      // [4]
    public const int CompInter = 1935;       // [5]
    public const int SingleRef = 1940;       // [5][2]
    public const int CompRef = 1950;         // [5]
    public const int Tx32 = 1955;            // [2][3]
    public const int Tx16 = 1961;            // [2][2]
    public const int Tx8 = 1965;             // [2][1]
    public const int Skip = 1967;            // [3]
    public const int MvJoints = 1970;        // [3]
    public const int MvComp = 1973;          // [2] x 33: sign, classes[10], class0[1], bits[10], class0_fp[2][3], fp[3], class0_hp, hp
    public const int MvCompStride = 33;
    public const int MvSign = 0, MvClasses = 1, MvClass0 = 11, MvBits = 12, MvClass0Fp = 22, MvFp = 28, MvClass0Hp = 31, MvHp = 32;
    /// <summary>Slot of the current context (libvpx cm->fc).</summary>
    public const int Current = 4;
    public const int Slots = 5;
}

/// <summary>Indices into the compressed header kernel's argument and result ints.</summary>
internal static class Vp9CompArgs
{
    public const int Offset = 0, Size = 1, Lossless = 2, IntraOnly = 3, Switchable = 4, AllowHp = 5, SignBias = 6;   // 6..8 = LAST, GOLDEN, ALTREF
    public const int Count = 9;
    public const int RTxMode = 0, RRefMode = 1, RCompFixed = 2, RCompVar0 = 3, RCompVar1 = 4, RError = 5;
    public const int ResultCount = 6;
}

/// <summary>Kernels for the frame contexts and the compressed header.</summary>
internal static class Vp9CompressedHeaderKernels
{
    private const int DiffUpdateProb = 252, MvUpdateProb = 252;

    /// <summary>One thread per probability: slot <paramref name="dst"/> = defaults (<paramref name="src"/> &lt; 0) or slot <paramref name="src"/>.</summary>
    public static void CopyContextKernel(Index1D i, ArrayView<int> ctx, ArrayView<int> defaults, int dst, int src)
    {
        ctx[dst * Vp9Ctx.Bytes + i] = src < 0 ? defaults[i] : ctx[src * Vp9Ctx.Bytes + i];
    }

    /// <summary>One thread: read_compressed_header into the current context (slot 4).</summary>
    public static void CompressedHeaderKernel(Index1D index, ArrayView<byte> frame, ArrayView<int> ctx, ArrayView<int> invMap,
        ArrayView<int> args, ArrayView<int> results)
    {
        int fc = Vp9Ctx.Current * Vp9Ctx.Bytes;
        var r = Vp8BoolDecoderGpu.Init(frame, args[Vp9CompArgs.Offset], args[Vp9CompArgs.Size]);
        int error = Vp8BoolDecoderGpu.DecodeBool(ref r, frame, 128);   // marker bit (vpx_reader_init fails when set)

        int txMode = 0;
        if (args[Vp9CompArgs.Lossless] == 0)
        {
            txMode = ReadLiteral(ref r, frame, 2);
            if (txMode == 3) txMode += ReadBit(ref r, frame);   // ALLOW_32X32 -> TX_MODE_SELECT
        }
        if (txMode == 4)
        {
            for (int i = 0; i < 2; i++) DiffUpdate(ref r, frame, ctx, fc + Vp9Ctx.Tx8 + i, invMap);
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 2; j++) DiffUpdate(ref r, frame, ctx, fc + Vp9Ctx.Tx16 + i * 2 + j, invMap);
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 3; j++) DiffUpdate(ref r, frame, ctx, fc + Vp9Ctx.Tx32 + i * 3 + j, invMap);
        }
        int maxTx = txMode < 3 ? txMode : 3;   // tx_mode_to_biggest_tx_size
        for (int tx = 0; tx <= maxTx; tx++)
        {
            if (ReadBit(ref r, frame) == 0) continue;
            for (int plane = 0; plane < 2; plane++)
                for (int refType = 0; refType < 2; refType++)
                    for (int band = 0; band < 6; band++)
                    {
                        int contexts = band == 0 ? 3 : 6;   // BAND_COEFF_CONTEXTS
                        for (int c = 0; c < contexts; c++)
                            for (int node = 0; node < 3; node++)
                                DiffUpdate(ref r, frame, ctx,
                                    fc + Vp9Ctx.Coef + ((((tx * 2 + plane) * 2 + refType) * 6 + band) * 6 + c) * 3 + node, invMap);
                    }
        }
        for (int k = 0; k < 3; k++) DiffUpdate(ref r, frame, ctx, fc + Vp9Ctx.Skip + k, invMap);

        int refMode = 0, compFixed = -1, compVar0 = -1, compVar1 = -1;
        if (args[Vp9CompArgs.IntraOnly] == 0)
        {
            for (int i = 0; i < 7; i++)
                for (int j = 0; j < 3; j++) DiffUpdate(ref r, frame, ctx, fc + Vp9Ctx.InterMode + i * 3 + j, invMap);
            if (args[Vp9CompArgs.Switchable] != 0)
                for (int j = 0; j < 4; j++)
                    for (int i = 0; i < 2; i++) DiffUpdate(ref r, frame, ctx, fc + Vp9Ctx.SwitchableInterp + j * 2 + i, invMap);
            for (int i = 0; i < 4; i++) DiffUpdate(ref r, frame, ctx, fc + Vp9Ctx.IntraInter + i, invMap);

            int biasLast = args[Vp9CompArgs.SignBias], biasGolden = args[Vp9CompArgs.SignBias + 1], biasAlt = args[Vp9CompArgs.SignBias + 2];
            bool compoundAllowed = biasGolden != biasLast || biasAlt != biasLast;
            if (compoundAllowed)
                refMode = ReadBit(ref r, frame) != 0 ? (ReadBit(ref r, frame) != 0 ? 2 : 1) : 0;   // SINGLE / COMPOUND / SELECT
            if (refMode != 0)
            {
                // vp9_setup_compound_reference_mode (LAST 1, GOLDEN 2, ALTREF 3)
                if (biasLast == biasGolden) { compFixed = 3; compVar0 = 1; compVar1 = 2; }
                else if (biasLast == biasAlt) { compFixed = 2; compVar0 = 1; compVar1 = 3; }
                else { compFixed = 1; compVar0 = 2; compVar1 = 3; }
            }
            if (refMode == 2)
                for (int i = 0; i < 5; i++) DiffUpdate(ref r, frame, ctx, fc + Vp9Ctx.CompInter + i, invMap);
            if (refMode != 1)
                for (int i = 0; i < 5; i++)
                {
                    DiffUpdate(ref r, frame, ctx, fc + Vp9Ctx.SingleRef + i * 2, invMap);
                    DiffUpdate(ref r, frame, ctx, fc + Vp9Ctx.SingleRef + i * 2 + 1, invMap);
                }
            if (refMode != 0)
                for (int i = 0; i < 5; i++) DiffUpdate(ref r, frame, ctx, fc + Vp9Ctx.CompRef + i, invMap);
            for (int j = 0; j < 4; j++)
                for (int i = 0; i < 9; i++) DiffUpdate(ref r, frame, ctx, fc + Vp9Ctx.YMode + j * 9 + i, invMap);
            for (int j = 0; j < 16; j++)
                for (int i = 0; i < 3; i++) DiffUpdate(ref r, frame, ctx, fc + Vp9Ctx.Partition + j * 3 + i, invMap);

            // read_mv_probs
            MvUpdate(ref r, frame, ctx, fc + Vp9Ctx.MvJoints, 3);
            for (int c = 0; c < 2; c++)
            {
                int comp = fc + Vp9Ctx.MvComp + c * Vp9Ctx.MvCompStride;
                MvUpdate(ref r, frame, ctx, comp + Vp9Ctx.MvSign, 1);
                MvUpdate(ref r, frame, ctx, comp + Vp9Ctx.MvClasses, 10);
                MvUpdate(ref r, frame, ctx, comp + Vp9Ctx.MvClass0, 1);
                MvUpdate(ref r, frame, ctx, comp + Vp9Ctx.MvBits, 10);
            }
            for (int c = 0; c < 2; c++)
            {
                int comp = fc + Vp9Ctx.MvComp + c * Vp9Ctx.MvCompStride;
                for (int j = 0; j < 2; j++) MvUpdate(ref r, frame, ctx, comp + Vp9Ctx.MvClass0Fp + j * 3, 3);
                MvUpdate(ref r, frame, ctx, comp + Vp9Ctx.MvFp, 3);
            }
            if (args[Vp9CompArgs.AllowHp] != 0)
                for (int c = 0; c < 2; c++)
                {
                    int comp = fc + Vp9Ctx.MvComp + c * Vp9Ctx.MvCompStride;
                    MvUpdate(ref r, frame, ctx, comp + Vp9Ctx.MvClass0Hp, 1);
                    MvUpdate(ref r, frame, ctx, comp + Vp9Ctx.MvHp, 1);
                }
        }
        // vpx_reader_has_error: bits were read past the end of the partition
        if (r.Count > Vp8BoolDecoderGpu.BdValueSize && r.Count < Vp8BoolDecoderGpu.LotsOfBits) error = 1;
        results[Vp9CompArgs.RTxMode] = txMode;
        results[Vp9CompArgs.RRefMode] = refMode;
        results[Vp9CompArgs.RCompFixed] = compFixed;
        results[Vp9CompArgs.RCompVar0] = compVar0;
        results[Vp9CompArgs.RCompVar1] = compVar1;
        results[Vp9CompArgs.RError] = error;
    }

    private static int ReadBit(ref Vp8BoolDecoderGpuState r, ArrayView<byte> buf) => Vp8BoolDecoderGpu.DecodeBool(ref r, buf, 128);

    private static int ReadLiteral(ref Vp8BoolDecoderGpuState r, ArrayView<byte> buf, int bits)
    {
        int v = 0;
        for (int b = bits - 1; b >= 0; b--) v |= ReadBit(ref r, buf) << b;
        return v;
    }

    private static void MvUpdate(ref Vp8BoolDecoderGpuState r, ArrayView<byte> buf, ArrayView<int> ctx, int at, int n)
    {
        for (int i = 0; i < n; i++)
            if (Vp8BoolDecoderGpu.DecodeBool(ref r, buf, MvUpdateProb) != 0) ctx[at + i] = (ReadLiteral(ref r, buf, 7) << 1) | 1;
    }

    /// <summary>vp9_diff_update_prob.</summary>
    private static void DiffUpdate(ref Vp8BoolDecoderGpuState r, ArrayView<byte> buf, ArrayView<int> ctx, int at, ArrayView<int> invMap)
    {
        if (Vp8BoolDecoderGpu.DecodeBool(ref r, buf, DiffUpdateProb) == 0) return;
        // decode_term_subexp
        int delp;
        if (ReadBit(ref r, buf) == 0) delp = ReadLiteral(ref r, buf, 4);
        else if (ReadBit(ref r, buf) == 0) delp = ReadLiteral(ref r, buf, 4) + 16;
        else if (ReadBit(ref r, buf) == 0) delp = ReadLiteral(ref r, buf, 5) + 32;
        else
        {
            // decode_uniform: l = 8, m = 256 - 191 = 65
            int v = ReadLiteral(ref r, buf, 7);
            delp = (v < 65 ? v : (v << 1) - 65 + ReadBit(ref r, buf)) + 64;
        }
        // inv_remap_prob(delp, p)
        int vv = invMap[delp];
        int m = ctx[at] - 1;
        int p;
        if ((m << 1) <= 255) p = 1 + InvRecenterNonneg(vv, m);
        else p = 255 - InvRecenterNonneg(vv, 255 - 1 - m);
        ctx[at] = p;
    }

    private static int InvRecenterNonneg(int v, int m)
    {
        if (v > 2 * m) return v;
        return (v & 1) != 0 ? m - ((v + 1) >> 1) : m + (v >> 1);
    }
}

/// <summary>Result of <see cref="Vp9FrameContextsGpu.ReadCompressedHeader"/> (libvpx cm fields it sets).</summary>
public readonly record struct Vp9CompressedHeaderResult(int TxMode, int ReferenceMode, int CompFixedRef, int CompVarRef0, int CompVarRef1, bool Error);

/// <summary>The VP9 frame contexts on the accelerator: the four saved contexts, the current one, and the compressed header.</summary>
public sealed class Vp9FrameContextsGpu : IDisposable
{
    private readonly Accelerator _acc;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _ctx, _defaults, _invMap, _args, _results;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, int, int> _copy;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>> _compressed;

    /// <summary>Contexts on <paramref name="accelerator"/>, all slots at libvpx's defaults.</summary>
    public Vp9FrameContextsGpu(Accelerator accelerator)
    {
        _acc = accelerator ?? throw new ArgumentNullException(nameof(accelerator));
        _ctx = _acc.Allocate1D<int>(Vp9Ctx.Slots * Vp9Ctx.Bytes);
        _defaults = _acc.Allocate1D(Vp9DecoderTables.DefaultFrameContext.Select(b => (int)b).ToArray());
        _invMap = _acc.Allocate1D(Vp9DecoderTables.InvMapTable.Select(b => (int)b).ToArray());
        _args = _acc.Allocate1D<int>(Vp9CompArgs.Count);
        _results = _acc.Allocate1D<int>(Vp9CompArgs.ResultCount);
        _copy = _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, int, int>(Vp9CompressedHeaderKernels.CopyContextKernel);
        _compressed = _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>>(
            Vp9CompressedHeaderKernels.CompressedHeaderKernel);
        for (int s = 0; s < Vp9Ctx.Slots; s++) _copy(Vp9Ctx.Bytes, _ctx.View, _defaults.View, s, -1);
    }

    /// <summary>All contexts: 5 x <see cref="Vp9Ctx.Bytes"/> ints (slots 0..3 saved, 4 current).</summary>
    public ArrayView<int> Contexts => _ctx.View;

    /// <summary>
    /// Start of a frame (vp9_setup_past_independence's context resets, then cm->fc = frame_contexts[frame_context_idx]).
    /// </summary>
    public void BeginFrame(Vp9FrameInfo h)
    {
        if (h.ShowExistingFrame) return;
        if (h.ContextReset == Vp9ContextReset.All)
            for (int s = 0; s < 4; s++) _copy(Vp9Ctx.Bytes, _ctx.View, _defaults.View, s, -1);
        else if (h.ContextReset == Vp9ContextReset.One)
            _copy(Vp9Ctx.Bytes, _ctx.View, _defaults.View, h.ContextResetIdx, -1);
        _copy(Vp9Ctx.Bytes, _ctx.View, _defaults.View, Vp9Ctx.Current, h.FrameContextIdx);
    }

    /// <summary>Decode the compressed header of the frame in <paramref name="frame"/> into the current context.</summary>
    /// <param name="frame">The frame's bytes on the accelerator.</param>
    /// <param name="frameOffset">Where the frame starts in <paramref name="frame"/>.</param>
    /// <returns>The results stay on the accelerator (<see cref="Results"/>); read them with <see cref="ReadResultsAsync"/>.</returns>
    public void ReadCompressedHeader(ArrayView<byte> frame, int frameOffset, Vp9FrameInfo h)
    {
        var args = new int[Vp9CompArgs.Count];
        args[Vp9CompArgs.Offset] = frameOffset + h.UncompressedHeaderBytes;
        args[Vp9CompArgs.Size] = h.FirstPartitionSize;
        args[Vp9CompArgs.Lossless] = h.Lossless ? 1 : 0;
        args[Vp9CompArgs.IntraOnly] = h.IsIntraOnlyFrame ? 1 : 0;
        args[Vp9CompArgs.Switchable] = h.InterpFilter == Vp9Interp.Switchable ? 1 : 0;
        args[Vp9CompArgs.AllowHp] = h.AllowHighPrecisionMv ? 1 : 0;
        args[Vp9CompArgs.SignBias] = h.SignBias[1];
        args[Vp9CompArgs.SignBias + 1] = h.SignBias[2];
        args[Vp9CompArgs.SignBias + 2] = h.SignBias[3];
        _args.View.CopyFromCPU(args);   // nine scalars of frame metadata
        _compressed(1, frame, _ctx.View, _invMap.View, _args.View, _results.View);
    }

    /// <summary>The compressed header's results (tx mode, reference mode, compound refs, error) on the accelerator.</summary>
    public ArrayView<int> Results => _results.View;

    /// <summary>Read the six result ints (a tiny scalar readback).</summary>
    public async Task<Vp9CompressedHeaderResult> ReadResultsAsync()
    {
        var r = await _results.View.CopyToHostAsync();
        return new Vp9CompressedHeaderResult(r[0], r[1], r[2], r[3], r[4], r[5] != 0);
    }

    /// <summary>End of a frame: frame_contexts[frame_context_idx] = cm->fc when refresh_frame_context.</summary>
    public void EndFrame(Vp9FrameInfo h)
    {
        if (!h.ShowExistingFrame && h.RefreshFrameContext)
            _copy(Vp9Ctx.Bytes, _ctx.View, _defaults.View, h.FrameContextIdx, Vp9Ctx.Current);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _ctx.Dispose();
        _defaults.Dispose();
        _invMap.Dispose();
        _args.Dispose();
        _results.Dispose();
    }
}
