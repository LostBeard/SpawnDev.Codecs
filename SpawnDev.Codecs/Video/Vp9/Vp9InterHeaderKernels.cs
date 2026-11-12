// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt). See NOTICE.md for upstream attributions.
//
// VP9 inter frame encoder, the two frame headers (Plans/PLAN-VP9-Encoder-Inter-Frames.md), written as the decoder
// reads them (Decoder/Vp9HeaderReader.ReadFrame, Vp9CompressedHeaderKernels: libvpx v1.17.0 vp9_decodeframe.c
// read_uncompressed_header / read_compressed_header).
//
// The frames this encoder writes: non-key, shown, error_resilient_mode = 1 (so: every probability at libvpx's
// defaults, no backward adaptation, no previous-frame MV candidates), every reference slot refreshed, the three
// references all slot 0 without sign bias (single reference LAST), the frame size taken from the reference, quarter-pel
// MVs, the EIGHTTAP filter for every block, no loop filter, no segmentation, one tile.

using ILGPU;
using SpawnDev.Codecs.Video.Vp8;

namespace SpawnDev.Codecs.Video.Vp9;

internal static class Vp9InterHeaderKernels
{
    /// <summary>
    /// The compressed header's "no update" bools after the coefficient gates, each at probability 252 (libvpx
    /// DIFF_UPDATE_PROB / MV_UPDATE_PROB): skip 3, inter mode 21, intra / inter 4, single reference 10, y mode 36,
    /// partition 48, MV joints 3, then per MV component sign 1 + classes 10 + class0 1 + bits 10, then per component
    /// class0_fp 6 + fp 3. (No switchable filter probabilities: the filter is fixed. No reference mode bits and no
    /// compound probabilities: without sign bias compound prediction is not allowed. No hp probabilities: high precision
    /// is off.)
    /// </summary>
    internal const int NoUpdateBools = 3 + 21 + 4 + 10 + 36 + 48 + 3 + 2 * 22 + 2 * 9;

    /// <summary>One thread: the compressed header (every probability stays at its default); its length to <paramref name="outLen"/>[0].</summary>
    public static void CompressedHeaderKernel(Index1D index, ArrayView<byte> outBuf, ArrayView<long> outLen)
    {
        var state = Vp8BoolEncoderGpu.Init();
        Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 0, 128);   // marker bit
        // read_tx_mode: the literal 3 (ALLOW_32X32), then one more bit: 0 = not TX_MODE_SELECT
        Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 1, 128);
        Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 1, 128);
        // then read_coef_probs: one "no update" bit per transform size up to 32x32 (so 5 zero bits at 128 in all)
        for (int i = 0; i < 5; i++) Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 0, 128);
        for (int i = 0; i < NoUpdateBools; i++) Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 0, 252);
        Vp8BoolEncoderGpu.StopVp9(ref state, outBuf);
        outLen[0] = state.OutLen;
    }

    /// <summary>
    /// One thread: the uncompressed header of an inter frame; <paramref name="firstPartitionSize"/>[0] = the compressed
    /// header's length (on the accelerator, from <see cref="CompressedHeaderKernel"/>). Its length to
    /// <paramref name="outLen"/>[0].
    /// </summary>
    public static void UncompressedHeaderKernel(Index1D index, ArrayView<byte> outBuf, ArrayView<long> outLen, ArrayView<long> firstPartitionSize,
        int width, int baseQIndex)
    {
        var bw = Vp9BitWriterGpu.Init();
        Vp9BitWriterGpu.WriteBits(ref bw, outBuf, 0b10u, 2);   // frame_marker
        Vp9BitWriterGpu.WriteBits(ref bw, outBuf, 0u, 2);      // profile 0 (low bit, high bit)
        Vp9BitWriterGpu.WriteBits(ref bw, outBuf, 0u, 1);      // show_existing_frame
        Vp9BitWriterGpu.WriteBits(ref bw, outBuf, 1u, 1);      // frame_type: non-key
        Vp9BitWriterGpu.WriteBits(ref bw, outBuf, 1u, 1);      // show_frame (so no intra_only bit)
        Vp9BitWriterGpu.WriteBits(ref bw, outBuf, 1u, 1);      // error_resilient_mode (so no reset_frame_context)
        Vp9BitWriterGpu.WriteBits(ref bw, outBuf, 0xFFu, 8);   // refresh_frame_flags: every slot
        // ref_frame_idx (3 bits) + sign_bias (1 bit) for LAST, GOLDEN, ALTREF: slot 0, no bias
        Vp9BitWriterGpu.WriteBits(ref bw, outBuf, 0u, 12);
        // frame_size_with_refs: found_ref for the first reference (the size is the reference's), then
        // render_and_frame_size_different = 0
        Vp9BitWriterGpu.WriteBits(ref bw, outBuf, 1u, 1);
        Vp9BitWriterGpu.WriteBits(ref bw, outBuf, 0u, 1);
        Vp9BitWriterGpu.WriteBits(ref bw, outBuf, 0u, 1);      // allow_high_precision_mv
        Vp9BitWriterGpu.WriteBits(ref bw, outBuf, 0u, 1);      // is_filter_switchable
        Vp9BitWriterGpu.WriteBits(ref bw, outBuf, 1u, 2);      // raw_interpolation_filter: 1 = EIGHTTAP
        // (error resilient: no refresh_frame_context / frame_parallel_decoding_mode)
        Vp9BitWriterGpu.WriteBits(ref bw, outBuf, 0u, 2);      // frame_context_idx
        Vp9BitWriterGpu.WriteBits(ref bw, outBuf, 0u, 6);      // loop filter level
        Vp9BitWriterGpu.WriteBits(ref bw, outBuf, 0u, 3);      // sharpness
        Vp9BitWriterGpu.WriteBits(ref bw, outBuf, 0u, 1);      // mode_ref_delta_enabled
        Vp9BitWriterGpu.WriteBits(ref bw, outBuf, (uint)(baseQIndex & 0xFF), 8);
        Vp9BitWriterGpu.WriteBits(ref bw, outBuf, 0u, 3);      // no y_dc / uv_dc / uv_ac delta
        Vp9BitWriterGpu.WriteBits(ref bw, outBuf, 0u, 1);      // segmentation off
        // tile_info: the fewest tile columns (one 0 bit when more are possible), one tile row
        int sbCols = (((width + 7) >> 3) + 7) >> 3;
        int minLog2 = 0;
        while ((64 << minLog2) < sbCols) minLog2++;
        int maxLog2 = 1;
        while ((sbCols >> maxLog2) >= 4) maxLog2++;
        maxLog2--;
        if (maxLog2 > minLog2) Vp9BitWriterGpu.WriteBits(ref bw, outBuf, 0u, 1);
        Vp9BitWriterGpu.WriteBits(ref bw, outBuf, 0u, 1);
        Vp9BitWriterGpu.WriteBits(ref bw, outBuf, (uint)((int)firstPartitionSize[0] & 0xFFFF), 16);
        Vp9BitWriterGpu.PadToByte(ref bw, outBuf);
        outLen[0] = bw.OutLen;
    }
}
