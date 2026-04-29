// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-callable Vorbis post-IMDCT processing. Composite of the three
// per-sample steps inside VorbisAudioDecoder.DecodePacket between the
// IMDCT and the interleaved output stage:
//   1. Apply the canonical sine window to the time-domain output.
//   2. Overlap-add against the previous packet's right half.
//   3. Save the new right half for the next packet's overlap-add.
//
// This composite emits one PCM output sample per (channel, n) pair while
// also writing the new right half for the next call. Per-sample-parallel:
// caller dispatches halfBlockSize threads per channel.
//
// Used after IMDCT in the Vorbis decoder pipeline. Composes with
// VorbisWindowGpu for the canonical-window generation (the window itself
// is unchanged across frames at the same blockSize).

using ILGPU;

namespace SpawnDev.Codecs.Audio.Vorbis;

/// <summary>
/// GPU-callable Vorbis post-IMDCT processing (window apply + overlap-add
/// + right-half save). Mirror of the per-sample loop inside
/// <see cref="VorbisAudioDecoder"/> after the IMDCT step.
/// </summary>
public static class VorbisPostImdctGpu
{
    /// <summary>
    /// Compute one output PCM sample at half-block index <paramref name="i"/>
    /// for one channel: window the IMDCT output, overlap-add with the
    /// stored previous right half, write the PCM sample, and save the
    /// new right half for the next packet.
    /// </summary>
    /// <param name="td">In/out IMDCT time-domain output (length blockSize).</param>
    /// <param name="tdBase">Base offset.</param>
    /// <param name="window">Canonical sine window (length blockSize).</param>
    /// <param name="windowBase">Base offset.</param>
    /// <param name="previousRightHalf">Previous packet's stored right-half samples (length halfBlockSize).</param>
    /// <param name="prevBase">Base offset.</param>
    /// <param name="newRightHalfOut">Output: this packet's right half for the next call (length halfBlockSize).</param>
    /// <param name="newRightBase">Base offset.</param>
    /// <param name="pcmOut">Output PCM (length halfBlockSize).</param>
    /// <param name="pcmBase">Base offset.</param>
    /// <param name="halfBlockSize">Half of blockSize.</param>
    /// <param name="i">Half-block sample index in [0, halfBlockSize).</param>
    public static void ProcessAt(
        ArrayView<float> td, long tdBase,
        ArrayView<float> window, long windowBase,
        ArrayView<float> previousRightHalf, long prevBase,
        ArrayView<float> newRightHalfOut, long newRightBase,
        ArrayView<float> pcmOut, long pcmBase,
        int halfBlockSize, int i)
    {
        // Apply window to the left and right halves of the IMDCT output.
        float leftWindowed = td[tdBase + i] * window[windowBase + i];
        float rightWindowed = td[tdBase + halfBlockSize + i]
                            * window[windowBase + halfBlockSize + i];

        // Overlap-add: left half of current + previous packet's right half.
        pcmOut[pcmBase + i] = leftWindowed + previousRightHalf[prevBase + i];

        // Save right half of current as the previous-right-half for next call.
        newRightHalfOut[newRightBase + i] = rightWindowed;
    }
}
