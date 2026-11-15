// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt). See NOTICE.md for upstream attributions.
//
// VP9 loop filter scheduling on the accelerator. libvpx v1.17.0 filters superblocks in raster order (loop_filter_rows);
// its multi-threaded path lets superblock (r, c) run once (r, c - 1), (r - 1, c) and (r - 1, c + 1) are done
// (vp9_thread_common.c sync_read). Wavefront step k = 2r + c meets exactly that: (r, c - 1) is on step k - 1,
// (r - 1, c + 1) on step k - 1, (r - 1, c) on step k - 2. Each step is one launch, one thread group of
// Vp9Lpf.GroupLanes lanes per superblock on it (Vp9Lpf.FilterSuperblockGroup: a lane per pixel row, then per column).

using ILGPU;

namespace SpawnDev.Codecs.Video.Vp9.Decoder;

/// <summary>Per-frame int arguments of <see cref="Vp9LoopFilterKernels.WavefrontKernel"/>.</summary>
internal static class Vp9LArg
{
    public const int MiRows = 0, MiCols = 1, SsX = 2, SsY = 3, GridOffset = 4, Y = 5, U = 6, V = 7, YStride = 8, UvStride = 9;
    public const int Count = 10;
}

internal static class Vp9LoopFilterKernels
{
    /// <summary>One thread: the frame's filter level table (vp9_loop_filter_frame_init).</summary>
    public static void FrameInitKernel(Index1D i, ArrayView<int> hdr, ArrayView<int> lf) => Vp9Lpf.FrameInit(hdr, 0, lf, 0);

    /// <summary>
    /// Thread s: the masks of superblock s (raster order) into <paramref name="masks"/>[s * Vp9Lpf.GroupSharedWords..]
    /// (<see cref="Vp9Lpf.BuildMasksInto"/>), for every superblock of the frame at once, before the wavefront. Nothing when
    /// the frame's filter level is 0.
    /// </summary>
    public static void MaskKernel(Index1D s, ArrayView<int> a, ArrayView<int> mi, ArrayView<int> lf, ArrayView<uint> masks)
    {
        int miRows = a[Vp9LArg.MiRows], miCols = a[Vp9LArg.MiCols];
        int sbCols = (miCols + 7) >> 3, sbRows = (miRows + 7) >> 3;
        if ((s >= sbRows * sbCols) | (lf[Vp9LfTable.FrameLevel] == 0)) return;
        int r = s / sbCols, c = s - r * sbCols;
        Vp9Lpf.BuildMasksInto(masks, s * Vp9Lpf.GroupSharedWords, mi, a[Vp9LArg.GridOffset], a[Vp9LArg.SsX], a[Vp9LArg.SsY],
            miRows, miCols, r * 8, c * 8, lf, 0);
    }

    /// <summary>
    /// Explicitly grouped, <see cref="Vp9Lpf.GroupLanes"/> lanes per group: group r filters superblock (r, k - 2r) of
    /// wavefront step <paramref name="k"/> (= 2 * row + col), with the masks <see cref="MaskKernel"/> stored. Launch rows
    /// 0 .. min(sbRows - 1, k / 2); groups whose column falls outside the frame filter nothing.
    /// </summary>
    public static void WavefrontKernel(ArrayView<byte> pool, ArrayView<int> a, ArrayView<uint> masks, ArrayView<int> lf, int k)
    {
        int r = Grid.IdxX, c = k - 2 * r;
        int miRows = a[Vp9LArg.MiRows], miCols = a[Vp9LArg.MiCols];
        int sbCols = (miCols + 7) >> 3;
        bool active = c >= 0 && c < sbCols;   // the same for every lane of the group
        int miRow = r * 8, miCol = c * 8;
        int ssX = a[Vp9LArg.SsX], ssY = a[Vp9LArg.SsY];
        int yStride = a[Vp9LArg.YStride], uvStride = a[Vp9LArg.UvStride];
        int yAt = a[Vp9LArg.Y] + miRow * 8 * yStride + miCol * 8;
        int uvRow = ((miRow * 8) >> ssY) * uvStride + ((miCol * 8) >> ssX);
        // inactive groups read no masks (the index would be outside this frame's)
        int maskAt = active ? (r * sbCols + c) * Vp9Lpf.GroupSharedWords : 0;
        Vp9Lpf.FilterSuperblockGroup(masks, maskAt, active, pool, yAt, yStride, a[Vp9LArg.U] + uvRow, a[Vp9LArg.V] + uvRow, uvStride,
            ssX, ssY, miRows, miRow, miCol, lf, 0);
    }
}
