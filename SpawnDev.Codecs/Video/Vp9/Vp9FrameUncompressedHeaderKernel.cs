// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 uncompressed frame header emit kernel. Single-thread dispatch;
// emits the bit-exact uncompressed header for a v1 keyframe (DC_PRED-
// only, single tile, LF disabled, default probs, baseQIndex sweep).
//
// The uncompressed header is RAW BITS (MSB-first byte packing), not
// bool-coded, so we use Vp9BitWriterGpu instead of the bool encoder.
// The compressed header that follows starts on a byte boundary because
// the uncompressed header pads to byte alignment before terminating;
// the bool-coded compressed header simply picks up at outLen bytes.
//
// Layout matches Vp9KeyframeEncoder.BuildUncompressedHeader exactly.
//
// Tile-info width-handling: VP9 spec sec 6.2.14 computes min/max
// log2_tile_cols from mi_cols. We inline that derivation rather than
// porting Vp9TileInfoParser.GetTileNBits, keeping the kernel self-
// contained. The constants come from Vp9TileInfo:
//   MiBlockSizeLog2  = 3
//   MinTileWidthSb64 = 4
//   MaxTileWidthSb64 = 64
//
// Frame metadata is fixed for v1 keyframes (Profile 0 / Bt601 color
// space / no LF / no segmentation / studio range), so the kernel
// hard-codes those bit patterns. Only width, height, baseQIndex, and
// firstPartitionSize vary per frame.

using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// VP9 uncompressed frame header emit kernel. Writes the bit-exact
/// uncompressed header for a v1 keyframe to a GPU output buffer.
/// </summary>
public sealed class Vp9FrameUncompressedHeaderKernel : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<long>, int, int, int, int> _kernel;

    /// <summary>Compile.</summary>
    public Vp9FrameUncompressedHeaderKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<long>, int, int, int, int>(EmitKernel);
    }

    /// <summary>
    /// Emit the uncompressed header. <paramref name="outLen"/> is
    /// written with the number of bytes used.
    /// </summary>
    public void Run(
        ArrayView<byte> outBuf,
        ArrayView<long> outLen,
        int width, int height,
        int baseQIndex,
        int firstPartitionSize)
    {
        if (outLen.Length < 1) throw new ArgumentException("outLen must hold 1 entry.", nameof(outLen));
        // Worst case for v1 keyframe is well under 32 bytes; require at least that.
        if (outBuf.Length < 32)
            throw new ArgumentException("outBuf must hold at least 32 bytes for the v1 keyframe header.", nameof(outBuf));
        _kernel(1, outBuf, outLen, width, height, baseQIndex, firstPartitionSize);
    }

    private static void EmitKernel(
        Index1D _,
        ArrayView<byte> outBuf,
        ArrayView<long> outLenOut,
        int width, int height,
        int baseQIndex,
        int firstPartitionSize)
    {
        var bw = Vp9BitWriterGpu.Init();

        // frame_marker f(2) = 0b10
        Vp9BitWriterGpu.WriteBits(ref bw, outBuf, 0b10u, 2);

        // profile = 0 -> two bits (low, high) = (0, 0)
        Vp9BitWriterGpu.WriteBits(ref bw, outBuf, 0u, 1);
        Vp9BitWriterGpu.WriteBits(ref bw, outBuf, 0u, 1);

        // show_existing_frame = 0
        Vp9BitWriterGpu.WriteBits(ref bw, outBuf, 0u, 1);

        // frame_type = KEY_FRAME = 0
        Vp9BitWriterGpu.WriteBits(ref bw, outBuf, 0u, 1);

        // show_frame = 1
        Vp9BitWriterGpu.WriteBits(ref bw, outBuf, 1u, 1);

        // error_resilient_mode = 0
        Vp9BitWriterGpu.WriteBits(ref bw, outBuf, 0u, 1);

        // sync_code 0x49 0x83 0x42 (Vp9SyncCode.Byte0/1/2)
        Vp9BitWriterGpu.WriteBits(ref bw, outBuf, 0x49u, 8);
        Vp9BitWriterGpu.WriteBits(ref bw, outBuf, 0x83u, 8);
        Vp9BitWriterGpu.WriteBits(ref bw, outBuf, 0x42u, 8);

        // color_config (profile 0): color_space(3) = Bt601(1), color_range(1) = 0
        Vp9BitWriterGpu.WriteBits(ref bw, outBuf, 1u, 3); // Vp9ColorSpace.Bt601 = 1
        Vp9BitWriterGpu.WriteBits(ref bw, outBuf, 0u, 1); // color_range = 0

        // frame_width_minus_1 f(16), frame_height_minus_1 f(16)
        Vp9BitWriterGpu.WriteBits(ref bw, outBuf, (uint)(width - 1) & 0xFFFFu, 16);
        Vp9BitWriterGpu.WriteBits(ref bw, outBuf, (uint)(height - 1) & 0xFFFFu, 16);

        // render_and_frame_size_different = 0
        Vp9BitWriterGpu.WriteBits(ref bw, outBuf, 0u, 1);

        // refresh_frame_context = 0, frame_parallel_decoding_mode = 0,
        // frame_context_idx f(2) = 0
        Vp9BitWriterGpu.WriteBits(ref bw, outBuf, 0u, 1);
        Vp9BitWriterGpu.WriteBits(ref bw, outBuf, 0u, 1);
        Vp9BitWriterGpu.WriteBits(ref bw, outBuf, 0u, 2);

        // loop_filter_params: filter_level(6) + sharpness_level(3) +
        // mode_ref_delta_enabled(1)
        Vp9BitWriterGpu.WriteBits(ref bw, outBuf, 0u, 6);
        Vp9BitWriterGpu.WriteBits(ref bw, outBuf, 0u, 3);
        Vp9BitWriterGpu.WriteBits(ref bw, outBuf, 0u, 1);

        // quantization_params: base_q_idx(8) + 3 delta-present flags = 0
        Vp9BitWriterGpu.WriteBits(ref bw, outBuf, (uint)(baseQIndex & 0xFF), 8);
        Vp9BitWriterGpu.WriteBits(ref bw, outBuf, 0u, 1);
        Vp9BitWriterGpu.WriteBits(ref bw, outBuf, 0u, 1);
        Vp9BitWriterGpu.WriteBits(ref bw, outBuf, 0u, 1);

        // segmentation_params: enabled f(1) = 0
        Vp9BitWriterGpu.WriteBits(ref bw, outBuf, 0u, 1);

        // tile_info: tile_cols_log2 = MIN, tile_rows_log2 = 0.
        // Inline GetTileNBits: sb_cols = AlignUp(miCols, 8) >> 3.
        int miCols = (width + 7) >> 3;
        int sbCols = ((miCols + 7) & ~7) >> 3;

        // minLog2Cols: smallest n where (MaxTileWidthSb64 << n) >= sbCols.
        int minLog2 = 0;
        while ((64 << minLog2) < sbCols) minLog2++;

        // maxLog2Cols: largest n where (sbCols >> n) >= MinTileWidthSb64,
        // computed as (n starting at 1, advance while predicate holds, then -1).
        int maxLog2 = 1;
        while ((sbCols >> maxLog2) >= 4) maxLog2++;
        maxLog2--;

        // Stay at minLog2: emit one 0 bit if there are increment bits.
        if (maxLog2 > minLog2)
            Vp9BitWriterGpu.WriteBits(ref bw, outBuf, 0u, 1);
        // tile_rows_log2 = 0: write a single 0 bit.
        Vp9BitWriterGpu.WriteBits(ref bw, outBuf, 0u, 1);

        // first_partition_size f(16): byte length of compressed header.
        Vp9BitWriterGpu.WriteBits(ref bw, outBuf, (uint)(firstPartitionSize & 0xFFFF), 16);

        // Byte-align so the compressed header starts on a byte boundary.
        Vp9BitWriterGpu.PadToByte(ref bw, outBuf);

        outLenOut[0] = bw.OutLen;
    }

    /// <summary>Release kernel resources.</summary>
    public void Dispose() { /* auto-grouped */ }
}
