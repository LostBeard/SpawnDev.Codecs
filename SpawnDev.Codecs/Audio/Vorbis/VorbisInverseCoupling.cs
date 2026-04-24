// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Inverse channel coupling per Vorbis I Section 4.3.8. Applied to the
// post-residue spectral coefficients before IMDCT. Undoes the encoder's
// per-coupling-step magnitude/angle correlation.

namespace SpawnDev.Codecs.Audio.Vorbis;

internal static class VorbisInverseCoupling
{
    /// <summary>
    /// Apply each coupling step in reverse order (LAST -> FIRST) per Vorbis I
    /// Section 4.3.8.2. For each step: take the magnitude and angle channels'
    /// coefficients and reconstruct the two original channel coefficients.
    /// </summary>
    /// <param name="spectra">Per-channel spectral buffers. Each is mutated in place.</param>
    /// <param name="mapping">Mapping config that drove the coupling; its coupling
    /// pairs are iterated in reverse.</param>
    internal static void Apply(Span<float[]> spectra, VorbisMappingConfig mapping)
    {
        int steps = mapping.CouplingMagnitudeChannels.Length;
        for (int step = steps - 1; step >= 0; step--)
        {
            int magCh = mapping.CouplingMagnitudeChannels[step];
            int angCh = mapping.CouplingAngleChannels[step];
            float[] magBuf = spectra[magCh];
            float[] angBuf = spectra[angCh];
            int n = Math.Min(magBuf.Length, angBuf.Length);
            for (int i = 0; i < n; i++)
            {
                float mag = magBuf[i];
                float ang = angBuf[i];
                float newM, newA;
                if (mag > 0)
                {
                    if (ang > 0) { newM = mag; newA = mag - ang; }
                    else { newA = mag; newM = mag + ang; }
                }
                else
                {
                    if (ang > 0) { newM = mag; newA = mag + ang; }
                    else { newA = mag; newM = mag - ang; }
                }
                magBuf[i] = newM;
                angBuf[i] = newA;
            }
        }
    }
}
