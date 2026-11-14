// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 tile_info in the uncompressed header (spec section 6.2.13), the encoder side, for the key frame and inter frame
// header kernels. The emission follows branch vp9-multi-tile-entropy (2026-05-05): tile_cols_log2 is coded as
// increments over the minimum the width allows, each a 1 bit, then a 0 bit unless the maximum is reached (the decoder
// stops reading at the maximum); tile_rows_log2 0 / 1 / 2 is coded 0 / 1 0 / 1 1. The minimum and maximum are
// calc_min_log2_tile_cols / calc_max_log2_tile_cols (Vp9TileInfoParser.GetTileNBits). The tile boundaries themselves
// are libvpx get_tile_offset (Vp9TileGrid), NOT that branch's GetTileColRange, which is off by one superblock.

using ILGPU;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 tile_info written by the header kernels.</summary>
internal static class Vp9TileHeaderGpu
{
    /// <summary>
    /// Write tile_info for a picture <paramref name="width"/> pixels wide. <paramref name="tileLog2"/> = requested
    /// log2 tile columns | (log2 tile rows &lt;&lt; 8); each is clamped to its legal range (columns to the width's
    /// [min, max], rows to [0, 2]), so 0 writes the fewest tile columns and one tile row.
    /// </summary>
    public static void Write(ref Vp9BitWriterGpuState bw, ArrayView<byte> outBuf, int width, int tileLog2)
    {
        int sbCols = (((width + 7) >> 3) + 7) >> 3;
        int minLog2 = 0;
        while ((64 << minLog2) < sbCols) minLog2++;
        int maxLog2 = 1;
        while ((sbCols >> maxLog2) >= 4) maxLog2++;
        maxLog2--;
        int cols = tileLog2 & 0xFF;
        if (cols < minLog2) cols = minLog2;
        if (cols > maxLog2) cols = maxLog2;
        for (int i = minLog2; i < cols; i++) Vp9BitWriterGpu.WriteBits(ref bw, outBuf, 1u, 1);
        if (cols < maxLog2) Vp9BitWriterGpu.WriteBits(ref bw, outBuf, 0u, 1);
        int rows = (tileLog2 >> 8) & 0xFF;
        if (rows > 2) rows = 2;
        if (rows == 0) Vp9BitWriterGpu.WriteBits(ref bw, outBuf, 0u, 1);
        else Vp9BitWriterGpu.WriteBits(ref bw, outBuf, rows == 1 ? 2u : 3u, 2);
    }
}
