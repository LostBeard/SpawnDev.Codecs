// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-callable forward MDCT + 4/N normalization in one shot. Mirror
// of the Vorbis encoder's pre-floor MDCT step (libvorbis convention -
// the encoder applies 4/N on the forward transform; the decoder leaves
// the inverse unscaled). One thread per output bin - cleanly per-bin
// parallel.

using ILGPU;
using SpawnDev.Codecs.Audio.Transforms;

namespace SpawnDev.Codecs.Audio.Vorbis;

/// <summary>
/// GPU-callable Vorbis-convention forward MDCT (O(N^2) reference) +
/// 4/N normalization, one bin per call.
/// </summary>
public static class VorbisFwdMdctScaledGpu
{
    /// <summary>
    /// Compute one Vorbis-scaled forward MDCT output bin:
    /// <c>output[outBase + k] = (4f/n) * Sum_{i=0..2N-1} input[i] * cos(pi/N * (i + 0.5 + N/2) * (k + 0.5))</c>.
    /// Per-bin independent; one thread per output k.
    /// Float-precision XMath.Cos accumulator (the CPU MdctReference uses
    /// double-precision Math.Cos; the float-vs-double drift over 2N
    /// cosines means non-silent Vorbis encoder output isn't bit-exact
    /// across CPU + GPU - decoded PCM is acoustically identical, but
    /// individual bits in the bitstream may differ at floor-Y boundaries).
    /// </summary>
    public static void ForwardScaledAt(
        ArrayView<float> timeDomain, long inBase,
        ArrayView<float> output, long outBase,
        int n, int k)
    {
        float coef = MdctReferenceGpu.Coefficient(timeDomain, inBase, n, k);
        float scale = 4f / n;
        output[outBase + k] = coef * scale;
    }
}
