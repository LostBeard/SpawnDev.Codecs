// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-callable Vorbis overlap-add. Mirror of VorbisWindow.OverlapAdd
// (Vorbis I spec section 4.3.7). Sums the previous block's windowed
// right-half samples with the current block's windowed left-half
// samples to produce one half-block of finalised PCM output.
//
// Per-sample independent: output[i] = previousRightHalf[i] + currentLeftHalf[i].
// One thread per output sample - true parallel-per-element across all
// 6 ILGPU backends.

using ILGPU;

namespace SpawnDev.Codecs.Audio.Vorbis;

/// <summary>
/// GPU-callable Vorbis overlap-add. Mirror of
/// <see cref="VorbisWindow"/>.OverlapAdd.
/// </summary>
public static class VorbisOverlapAddGpu
{
    /// <summary>
    /// Compute one finalised PCM sample at index <paramref name="i"/>:
    /// <c>output[i] = previousRightHalf[i] + currentLeftHalf[i]</c>. Caller
    /// dispatches halfBlockSize threads to fill the output span.
    /// </summary>
    public static void AddAt(
        ArrayView<float> previousRightHalf, long prevBase,
        ArrayView<float> currentLeftHalf, long curBase,
        ArrayView<float> output, long outBase,
        int i)
    {
        output[outBase + i] = previousRightHalf[prevBase + i] + currentLeftHalf[curBase + i];
    }
}
