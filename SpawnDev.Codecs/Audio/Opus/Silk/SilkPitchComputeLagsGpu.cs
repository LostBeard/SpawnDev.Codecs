// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-callable port of SilkPitchDecoder.ComputeLags. Expands a decoded
// (lagIndex, contourIndex) pair into nbSubfr per-subframe pitch lags
// using the appropriate pitch-contour codebook (selected by caller
// based on fs_kHz and nbSubfr) + clamping to [PE_MIN_LAG_MS * fsKHz,
// PE_MAX_LAG_MS * fsKHz].
//
// Pure GPU compute - no range decoder. Used by SilkParametersDecoderGpu
// during the per-frame parameter dequantization step.
//
// Upstream Copyright (c) 2006-2011 Skype Limited. BSD 3-Clause.

using ILGPU;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// GPU-callable per-subframe pitch lag expansion. Mirror of
/// `SilkPitchDecoder.ComputeLags`.
/// </summary>
public static class SilkPitchComputeLagsGpu
{
    /// <summary>SilkConstants.PE_MIN_LAG_MS = 2.</summary>
    public const int PeMinLagMs = 2;
    /// <summary>SilkConstants.PE_MAX_LAG_MS = 18.</summary>
    public const int PeMaxLagMs = 18;

    /// <summary>
    /// Expand (lagIndex, contourIndex) into <paramref name="nbSubfr"/>
    /// per-subframe pitch lags clamped to <c>[2*fsKHz, 18*fsKHz]</c>.
    /// </summary>
    /// <param name="contourCb">Caller-resolved (fs_kHz, nbSubfr)-specific
    /// pitch-contour codebook (sbyte values; layout
    /// <c>cb[k * cbSize + contourIndex]</c>).</param>
    /// <param name="contourCbBase">Offset into <paramref name="contourCb"/>.</param>
    /// <param name="cbSize">Codebook size for the resolved (fsKHz, nbSubfr)
    /// pair: 11/3/34/12 for NB-20/NB-10/non-NB-20/non-NB-10 respectively.</param>
    /// <param name="lagIndex">Decoded coarse lag index (0..N-1).</param>
    /// <param name="contourIndex">Decoded contour index (0..cbSize-1).</param>
    /// <param name="fsKHz">Internal SILK sample rate (8, 12, or 16).</param>
    /// <param name="nbSubfr">Subframe count (2 or 4).</param>
    /// <param name="pitchLagsOut">Output ArrayView&lt;int&gt; of length &gt;= nbSubfr.</param>
    /// <param name="pitchLagsBase">Offset into <paramref name="pitchLagsOut"/>.</param>
    public static void ComputeLags(
        ArrayView<sbyte> contourCb, long contourCbBase, int cbSize,
        int lagIndex, int contourIndex,
        int fsKHz, int nbSubfr,
        ArrayView<int> pitchLagsOut, long pitchLagsBase)
    {
        int minLag = PeMinLagMs * fsKHz;
        int maxLag = PeMaxLagMs * fsKHz;
        int baseLag = minLag + lagIndex;

        for (int k = 0; k < nbSubfr; k++)
        {
            int lag = baseLag + contourCb[contourCbBase + (long)k * cbSize + contourIndex];
            if (lag < minLag) lag = minLag;
            else if (lag > maxLag) lag = maxLag;
            pitchLagsOut[pitchLagsBase + k] = lag;
        }
    }
}
