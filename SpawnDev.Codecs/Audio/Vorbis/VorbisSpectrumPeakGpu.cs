// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-callable per-half-band absolute-max reducer for Vorbis encoder
// floor endpoint selection. Mirror of the half-band peak loops in
// VorbisAudioEncoder.EncodeAudioPacket.

using ILGPU;

namespace SpawnDev.Codecs.Audio.Vorbis;

/// <summary>
/// GPU-callable per-half-band peak finder for Vorbis encoder floor
/// endpoint selection.
/// </summary>
public static class VorbisSpectrumPeakGpu
{
    /// <summary>
    /// Compute the absolute-value peak across the lower half-band
    /// (bins [0, halfBlock/2)) and upper half-band ([halfBlock/2,
    /// halfBlock)) of a Vorbis spectrum. Writes the two peaks into
    /// <paramref name="peaksOut"/>[0] (low) and [1] (high).
    /// </summary>
    /// <param name="spectrum">Spectrum buffer (post-MDCT, post-scale).</param>
    /// <param name="spectrumBase">Base offset.</param>
    /// <param name="halfBlock">Spectrum length (= blockSize / 2).</param>
    /// <param name="peaksOut">Output buffer of at least 2 floats.</param>
    /// <param name="peaksBase">Base offset (peaksOut[+0] = low, [+1] = high).</param>
    public static void ComputeHalfBandPeaks(
        ArrayView<float> spectrum, long spectrumBase, int halfBlock,
        ArrayView<float> peaksOut, long peaksBase)
    {
        int split = halfBlock >> 1;
        float lowPeak = 0f;
        float highPeak = 0f;

        for (int i = 0; i < split; i++)
        {
            float v = spectrum[spectrumBase + i];
            float a = v < 0 ? -v : v;
            if (a > lowPeak) lowPeak = a;
        }
        for (int i = split; i < halfBlock; i++)
        {
            float v = spectrum[spectrumBase + i];
            float a = v < 0 ? -v : v;
            if (a > highPeak) highPeak = a;
        }

        peaksOut[peaksBase + 0] = lowPeak;
        peaksOut[peaksBase + 1] = highPeak;
    }
}
