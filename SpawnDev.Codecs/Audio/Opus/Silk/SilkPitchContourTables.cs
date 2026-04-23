// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Structural port of libopus silk/pitch_est_tables.c pitch-contour codebooks.
// These 2D tables map (subframe, contour_index) -> a small signed delta added
// to the frame's coarse pitch lag, yielding per-subframe pitch lags.
//
// Upstream Copyright (c) 2006-2011 Skype Limited. BSD 3-Clause. See NOTICE.md.

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// Pitch contour codebooks as flat row-major arrays. For each table, the layout
/// is <c>[subframe][contour_index]</c> - element index is
/// <c>subframe * codebookSize + contour_index</c>.
/// </summary>
internal static class SilkPitchContourTables
{
    /// <summary>
    /// <c>silk_CB_lags_stage2[4][11]</c>: stage-2 pitch contour codebook for NB 20 ms frames.
    /// </summary>
    internal static readonly sbyte[] Stage2 =
    {
         0,  2, -1, -1, -1,  0,  0,  1,  1,  0,  1,   // subframe 0
         0,  1,  0,  0,  0,  0,  0,  1,  0,  0,  0,   // subframe 1
         0,  0,  1,  0,  0,  0,  1,  0,  0,  0,  0,   // subframe 2
         0, -1,  2,  1,  0,  1,  1,  0,  0, -1, -1,   // subframe 3
    };

    /// <summary>
    /// <c>silk_CB_lags_stage2_10_ms[2][3]</c>: stage-2 pitch contour codebook for NB 10 ms frames.
    /// </summary>
    internal static readonly sbyte[] Stage210Ms =
    {
        0, 1, 0,
        0, 0, 1,
    };

    /// <summary>
    /// <c>silk_CB_lags_stage3[4][34]</c>: stage-3 pitch contour codebook for non-NB 20 ms frames.
    /// </summary>
    internal static readonly sbyte[] Stage3 =
    {
         0,  0,  1, -1,  0,  1, -1,  0, -1,  1, -2,  2, -2, -2,  2, -3,  2,  3, -3, -4,  3, -4,  4,  4, -5,  5, -6, -5,  6, -7,  6,  5,  8, -9,
         0,  0,  1,  0,  0,  0,  0,  0,  0,  0, -1,  1,  0,  0,  1, -1,  0,  1, -1, -1,  1, -1,  2,  1, -1,  2, -2, -2,  2, -2,  2,  2,  3, -3,
         0,  1,  0,  0,  0,  0,  0,  0,  1,  0,  1,  0,  0,  1, -1,  1,  0,  0,  2,  1, -1,  2, -1, -1,  2, -1,  2,  2, -1,  3, -2, -2, -2,  3,
         0,  1,  0,  0,  1,  0,  1, -1,  2, -1,  2, -1,  2,  3, -2,  3, -2, -2,  4,  4, -3,  5, -3, -4,  6, -4,  6,  5, -5,  8, -6, -5, -7,  9,
    };

    /// <summary>
    /// <c>silk_CB_lags_stage3_10_ms[2][12]</c>: stage-3 pitch contour codebook for non-NB 10 ms frames.
    /// </summary>
    internal static readonly sbyte[] Stage310Ms =
    {
        0, 0, 1, -1, 1, -1, 2, -2, 2, -2, 3, -3,
        0, 1, 0,  1, -1, 2, -1, 2, -2, 3, -2, 3,
    };
}
