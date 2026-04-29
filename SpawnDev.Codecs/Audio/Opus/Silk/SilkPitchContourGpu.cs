// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-callable SILK pitch contour expansion. Mirror of
// SilkPitchDecoder.ComputeLags (libopus silk/decode_pitch.c). Expands the
// decoded (lagIndex, contourIndex) pair into per-subframe pitch lags
// using one of four contour codebooks selected by (fsKHz, nbSubfr).
//
// Per-subframe parallel: thread k reads cb[k * cbSize + contourIndex],
// adds the base lag, clamps to [minLag, maxLag] and writes pitchLags[k].
// True parallel-per-subframe across all 6 ILGPU backends.
//
// All silk macros (no special ones - just int arithmetic + clamp).

using ILGPU;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// GPU-callable SILK pitch contour expansion. Mirror of
/// <see cref="SilkPitchDecoder"/>.ComputeLags.
/// </summary>
public static class SilkPitchContourGpu
{
    private const int PE_MIN_LAG_MS = 2;
    private const int PE_MAX_LAG_MS = 18;

    /// <summary>
    /// Compute one subframe's pitch lag at index <paramref name="subfrIdx"/>.
    /// Bit-exact vs the CPU SilkPitchDecoder.ComputeLags.
    /// </summary>
    /// <param name="pitchLags">Output: per-subframe pitch lags. Length nbSubfr.</param>
    /// <param name="lagsBase">Base offset.</param>
    /// <param name="contourCb">Contour codebook (caller selects between Stage2 / Stage210Ms /
    /// Stage3 / Stage310Ms). Layout: nbSubfr rows of cbSize entries each, row-major.</param>
    /// <param name="cbBase">Base offset.</param>
    /// <param name="lagIndex">Decoded coarse lag index.</param>
    /// <param name="contourIndex">Decoded contour index.</param>
    /// <param name="cbSize">Codebook column count (number of entries per row).</param>
    /// <param name="fsKHz">Internal SILK sample rate (8, 12, or 16).</param>
    /// <param name="subfrIdx">Subframe index in [0, nbSubfr).</param>
    public static void ComputeLagAt(
        ArrayView<int> pitchLags, long lagsBase,
        ArrayView<sbyte> contourCb, long cbBase,
        int lagIndex, int contourIndex, int cbSize, int fsKHz, int subfrIdx)
    {
        int minLag = PE_MIN_LAG_MS * fsKHz;
        int maxLag = PE_MAX_LAG_MS * fsKHz;
        int baseLag = minLag + lagIndex;

        long cbIdx = cbBase + (long)subfrIdx * cbSize + contourIndex;
        int delta = contourCb[cbIdx];
        int lag = baseLag + delta;

        if (lag < minLag) lag = minLag;
        else if (lag > maxLag) lag = maxLag;

        pitchLags[lagsBase + subfrIdx] = lag;
    }
}
