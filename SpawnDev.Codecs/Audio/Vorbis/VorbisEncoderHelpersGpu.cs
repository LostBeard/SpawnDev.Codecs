// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-callable encoder-side helpers for Vorbis. Mirrors the
// MagnitudeToFloorY + QuantiseResidueValue scalar helpers on the
// CPU encoder (VorbisAudioEncoder.cs) so they can be invoked
// per-sample from a GPU encode kernel.

using ILGPU;
using ILGPU.Algorithms;

namespace SpawnDev.Codecs.Audio.Vorbis;

/// <summary>
/// GPU-callable Vorbis encoder helpers (per-sample scalar math).
/// </summary>
public static class VorbisEncoderHelpersGpu
{
    /// <summary>
    /// Map a non-negative spectrum magnitude to a Floor 1 posterior Y
    /// in [0, 255] via binary search on the 256-entry inverse-dB table.
    /// Returns the smallest Y such that <c>inverseDb[Y] &gt;= magnitude</c>
    /// (ceiling semantics, so residue = spectrum/floor stays in [-1, +1]).
    /// Avoids the Log10 intrinsic which is unsupported on some ILGPU
    /// backends without EnableAlgorithms.
    /// </summary>
    /// <param name="magnitude">Spectrum magnitude (already non-negative).</param>
    /// <param name="inverseDbTable">256-entry inverse-dB lookup (uploaded once per accelerator).</param>
    /// <param name="inverseDbBase">Base offset.</param>
    /// <returns>Floor Y index in [0, 255].</returns>
    public static int MagnitudeToFloorY(
        float magnitude,
        ArrayView<float> inverseDbTable, long inverseDbBase)
    {
        // Binary search on the inverse-dB table (avoids the Log10
        // intrinsic which is unsupported on some ILGPU backends without
        // EnableAlgorithms). Returns the smallest Y such that
        // inverseDb[Y] &gt;= magnitude (ceiling semantics, so residue =
        // spectrum/floor stays in [-1, +1]).
        if (!(magnitude > 0)) return 0;       // catches NaN, 0, negative
        if (magnitude >= 1.0f) return 255;
        int lo = 0;
        int hi = 255;
        while (lo < hi)
        {
            int mid = (lo + hi) >> 1;
            if (inverseDbTable[inverseDbBase + mid] < magnitude) lo = mid + 1;
            else hi = mid;
        }
        return lo;
    }

    /// <summary>
    /// Per-bin residue divide + quantize at index <paramref name="i"/>:
    /// <c>residueQ[outBase + i] = QuantiseResidueValue(spectrum[i] / max(floorCurve[i], 1e-12))</c>.
    /// Per-bin parallel; one thread per output bin. Used by the future
    /// Vorbis GPU encoder's residue stage.
    /// </summary>
    public static void DivideQuantizeAt(
        ArrayView<float> spectrum, long spectrumBase,
        ArrayView<float> floorCurve, long floorBase,
        ArrayView<int> residueQ, long residueQBase,
        int i, float residueRange, int bookEntries)
    {
        float floor = floorCurve[floorBase + i];
        if (floor < 1e-12f) floor = 1e-12f;
        float r = spectrum[spectrumBase + i] / floor;
        residueQ[residueQBase + i] = QuantiseResidueValue(r, residueRange, bookEntries);
    }

    /// <summary>
    /// Quantize a residue sample (already divided by the floor curve)
    /// into the residue codebook entry index. Codebook is anchored:
    /// entry N/2 decodes to 0, entry i decodes to (i - N/2) * step
    /// where step = 2R/N.
    /// Mirror of VorbisAudioEncoder.QuantiseResidueValue.
    /// </summary>
    /// <param name="v">Residue value (spectrum / floor, expected in [-residueRange, +residueRange]).</param>
    /// <param name="residueRange">Half-range R (the codebook covers [-R, +R)).</param>
    /// <param name="bookEntries">Total entries in the residue codebook.</param>
    /// <returns>Codebook entry index in [0, bookEntries-1].</returns>
    public static int QuantiseResidueValue(float v, float residueRange, int bookEntries)
    {
        float step = 2f * residueRange / bookEntries;
        int half = bookEntries / 2;
        // Inline banker's-equivalent away-from-zero rounding to avoid
        // XMath.Round / Math.Round which require EnableAlgorithms on
        // some ILGPU backends. Math.Round defaults to ToEven; for the
        // residue quantizer we mirror exactly Math.Round's banker's
        // semantics by using IntegerRound below.
        int idx = IntegerRound(v / step) + half;
        if (idx < 0) idx = 0;
        if (idx >= bookEntries) idx = bookEntries - 1;
        return idx;
    }

    /// <summary>
    /// Banker's rounding (round half to even) on a float, returning int.
    /// Matches .NET's Math.Round default mode bit-exactly without invoking
    /// the ILGPU XMath.Round intrinsic.
    /// </summary>
    private static int IntegerRound(float v)
    {
        // floor(v + 0.5) for positive, ceil(v - 0.5) for negative is
        // away-from-zero. For half-to-even (banker's), check fractional
        // part exactly: if frac == 0.5, round to even.
        float floor = v >= 0 ? (int)v : (int)v - (v != (int)v ? 1 : 0);
        // Cast above doesn't handle truncate-toward-zero correctly for
        // negatives. Use a more robust path:
        int truncated = (int)v;
        float frac = v - truncated;
        if (v < 0)
        {
            // For negative, truncated is the ceil; we want floor.
            if (frac != 0) truncated -= 1;
            frac = v - truncated; // now in [0, 1)
        }
        // frac in [0, 1).
        if (frac < 0.5f) return truncated;
        if (frac > 0.5f) return truncated + 1;
        // Exactly 0.5 -> round to even.
        return (truncated & 1) == 0 ? truncated : truncated + 1;
    }
}
