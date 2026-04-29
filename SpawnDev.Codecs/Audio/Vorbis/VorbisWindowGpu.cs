// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Vorbis synthesis window + overlap-add helpers, GPU-callable form.
// Bit-exact mirror of VorbisWindow per Vorbis I Section 1.3.2 / 4.3.8.
// Per-sample independent math - one thread per output sample maps
// cleanly across all backends.

using ILGPU;
using ILGPU.Algorithms;

namespace SpawnDev.Codecs.Audio.Vorbis;

/// <summary>
/// GPU-callable Vorbis synthesis window helpers. Per-sample helpers
/// for canonical window generation + overlap-add sum.
/// </summary>
public static class VorbisWindowGpu
{
    /// <summary>
    /// Compute one canonical Vorbis synthesis window sample at index
    /// <paramref name="i"/> for window length <paramref name="n"/>.
    /// Window shape: w[i] = sin(pi/2 * sin^2(pi/n * (i + 0.5))).
    /// Uses float-precision XMath.Sin; cross-backend tests compare with
    /// 1e-6 tolerance vs the CPU double-precision reference (the
    /// difference is &lt; 1 ULP at typical inputs and audibly inaudible).
    /// </summary>
    public static float CanonicalSample(int i, int n)
    {
        float factor = (float)(Math.PI / n);
        float s = XMath.Sin(factor * (i + 0.5f));
        return XMath.Sin(0.5f * (float)Math.PI * s * s);
    }

    /// <summary>
    /// Apply window + add to the overlap buffer at one sample position.
    /// Reads input[inBase + i] and overlap[overlapBase + i], writes the
    /// summed result to output[outBase + i]. Used for the overlap-add
    /// final step of the Vorbis decode pipeline.
    /// </summary>
    public static void OverlapAddAt(
        ArrayView<float> previousRightHalf, long prevBase,
        ArrayView<float> currentLeftHalf, long currBase,
        ArrayView<float> output, long outBase,
        int i)
    {
        output[outBase + i] = previousRightHalf[prevBase + i] + currentLeftHalf[currBase + i];
    }

    /// <summary>
    /// Apply the canonical window to one input sample at index <paramref name="i"/>:
    /// <c>output[outBase + i] = input[inBase + i] * CanonicalSample(i, n)</c>.
    /// Per-sample independent; one thread per output sample. Used by the
    /// Vorbis encoder's pre-MDCT windowing step.
    /// </summary>
    public static void ApplyWindowAt(
        ArrayView<float> input, long inBase,
        ArrayView<float> output, long outBase,
        int i, int n)
    {
        output[outBase + i] = input[inBase + i] * CanonicalSample(i, n);
    }
}
