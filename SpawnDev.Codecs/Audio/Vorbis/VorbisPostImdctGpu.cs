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

    /// <summary>
    /// Block-switching overlap-add (Vorbis I sec 4.3.8) at flat index <paramref name="i"/>, for all channels in
    /// one dispatch of <c>channels * max(outFrames, blockSize / 2)</c> threads. A packet returns the frames from
    /// the previous block's centre to this block's centre, <c>outFrames = prevBlockSize / 4 + blockSize / 4</c>
    /// (none for the first packet, <paramref name="prevBlockSize"/> = 0): output frame t is the previous block's
    /// windowed sample prevBlockSize / 2 + t (its stored right half, zero past it) plus this block's windowed
    /// sample t - prevBlockSize / 4 + blockSize / 4 (zero before it). Written interleaved. Also stores this
    /// block's windowed right half for the next packet (into a DIFFERENT buffer than the one read).
    /// </summary>
    /// <param name="td">IMDCT outputs, channel-major rows of blockSize.</param>
    /// <param name="window">This block's window (blockSize, shaped by its neighbours: <see cref="VorbisWindow.Generate"/>).</param>
    /// <param name="previousRightHalf">Previous block's windowed right halves, rows of <paramref name="rowStride"/>.</param>
    /// <param name="newRightHalfOut">This block's windowed right halves, rows of <paramref name="rowStride"/>.</param>
    /// <param name="pcmOut">Interleaved output frames (channels * outFrames).</param>
    /// <param name="channels">Channel count.</param>
    /// <param name="blockSize">This block's size.</param>
    /// <param name="prevBlockSize">Previous block's size, 0 for the first packet.</param>
    /// <param name="rowStride">Row stride of the right-half buffers (at least blockSize1 / 2).</param>
    /// <param name="i">Flat thread index.</param>
    public static void OverlapSwitchingAt(
        ArrayView<float> td, ArrayView<float> window,
        ArrayView<float> previousRightHalf, ArrayView<float> newRightHalfOut,
        ArrayView<float> pcmOut,
        int channels, int blockSize, int prevBlockSize, int rowStride, int i)
    {
        int half = blockSize / 2;
        int outFrames = prevBlockSize > 0 ? prevBlockSize / 4 + blockSize / 4 : 0;
        if (i < channels * outFrames)
        {
            int t = i / channels;
            int ch = i - t * channels;
            int c = t - (prevBlockSize / 4 - blockSize / 4);
            float v = t < prevBlockSize / 2 ? previousRightHalf[(long)ch * rowStride + t] : 0f;
            if (c >= 0) v += td[(long)ch * blockSize + c] * window[c];
            pcmOut[i] = v;
        }
        if (i < channels * half)
        {
            int ch = i / half;
            int k = i - ch * half;
            newRightHalfOut[(long)ch * rowStride + k] = td[(long)ch * blockSize + half + k] * window[half + k];
        }
    }
}
