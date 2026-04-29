// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Vorbis I Section 4.3.8 inverse channel coupling, GPU-callable form.
// Bit-exact mirror of VorbisInverseCoupling.Apply for in-kernel use
// by the upcoming Vorbis decoder pipeline.
//
// Each coupling step processes one (magnitude, angle) channel pair
// across N coefficients. Within a step the per-coefficient
// reconstruction is independent, so the natural parallelization is
// one thread per coefficient. The encoder applies coupling steps in
// order; the decoder must apply them in REVERSE - the host
// orchestrates the per-step kernel dispatches in reverse-step order.

using ILGPU;

namespace SpawnDev.Codecs.Audio.Vorbis;

/// <summary>
/// GPU-callable Vorbis inverse channel coupling per coefficient.
/// Each invocation processes one coefficient of one (mag, ang)
/// channel pair.
/// </summary>
public static class VorbisInverseCouplingGpu
{
    /// <summary>
    /// Apply the inverse coupling reconstruction at coefficient index
    /// <paramref name="i"/> for the (magnitude, angle) channel pair.
    /// Reads from <paramref name="magBuf"/>[<paramref name="magBase"/>+i]
    /// and <paramref name="angBuf"/>[<paramref name="angBase"/>+i];
    /// writes back to the same positions.
    /// </summary>
    public static void ApplyAtCoefficient(
        ArrayView<float> magBuf, long magBase,
        ArrayView<float> angBuf, long angBase,
        int i)
    {
        float mag = magBuf[magBase + i];
        float ang = angBuf[angBase + i];
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
        magBuf[magBase + i] = newM;
        angBuf[angBase + i] = newA;
    }
}
