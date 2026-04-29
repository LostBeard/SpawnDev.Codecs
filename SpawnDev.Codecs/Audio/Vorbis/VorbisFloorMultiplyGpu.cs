// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-callable per-bin floor curve x residue multiplier. Mirror of
// VorbisAudioDecoder's "spectrum = floor * residue" step (Vorbis I
// sec 4.3 step 5). One thread per output bin - cleanly parallel.

using ILGPU;

namespace SpawnDev.Codecs.Audio.Vorbis;

/// <summary>
/// GPU-callable Vorbis floor x residue multiplier.
/// </summary>
public static class VorbisFloorMultiplyGpu
{
    /// <summary>
    /// Compute the spectral coefficient at bin <paramref name="i"/>:
    /// <c>spectrum[outBase + i] = floorCurve[floorBase + i] * residue[residueBase + i]</c>.
    /// Per-bin independent; one thread per output bin.
    /// </summary>
    public static void MultiplyAt(
        ArrayView<float> floorCurve, long floorBase,
        ArrayView<float> residue, long residueBase,
        ArrayView<float> spectrum, long outBase,
        int i)
    {
        spectrum[outBase + i] = floorCurve[floorBase + i] * residue[residueBase + i];
    }

    /// <summary>
    /// Same as <see cref="MultiplyAt"/> but writes 0 unconditionally
    /// (used when the per-channel floor was silent / not OK).
    /// </summary>
    public static void ZeroAt(
        ArrayView<float> spectrum, long outBase, int i)
    {
        spectrum[outBase + i] = 0f;
    }
}
