// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-callable per-sample interleaved -> channel-major de-interleave for the
// FLAC encoder's PCM input stage. Mirror of FlacInterleaveOutputGpu (which
// runs the channel-major -> interleaved direction on the decoder side).
//
// Per-(channel, sample) parallel: each thread reads one source value
// from the interleaved buffer and writes it to the corresponding
// position in the channel-major output. True parallel-per-element across
// all 6 ILGPU backends.
//
// Caller dispatches numFrames * channels threads.

using ILGPU;

namespace SpawnDev.Codecs.Audio.Flac;

/// <summary>
/// GPU-callable FLAC input de-interleave (int samples). Mirror of
/// the per-sample de-interleave loop that previously ran on the host
/// inside <see cref="FlacEncoderGpu"/>.EncodeStreamAsync.
/// </summary>
public static class FlacDeinterleaveInputGpu
{
    /// <summary>
    /// Compute one output sample at thread index <paramref name="threadIdx"/>
    /// in [0, numFrames * channels). Maps n = threadIdx / channels and
    /// ch = threadIdx % channels.
    /// </summary>
    /// <param name="interleaved">Per-sample interleaved PCM samples
    /// (length numFrames * channels), sample n's data at indices
    /// [n * channels, (n+1) * channels).</param>
    /// <param name="inBase">Base offset.</param>
    /// <param name="channelMajor">Output PCM (length channels * numFrames),
    /// channel ch's samples at indices [ch * numFrames, (ch+1) * numFrames).</param>
    /// <param name="cmBase">Base offset.</param>
    /// <param name="channels">Channel count.</param>
    /// <param name="numFrames">Per-channel frame count.</param>
    /// <param name="threadIdx">Linear thread index in [0, numFrames * channels).</param>
    public static void DeinterleaveAt(
        ArrayView<int> interleaved, long inBase,
        ArrayView<int> channelMajor, long cmBase,
        int channels, int numFrames, int threadIdx)
    {
        int n = threadIdx / channels;
        int ch = threadIdx - n * channels;

        long src = inBase + (long)n * channels + ch;
        long dst = cmBase + (long)ch * numFrames + n;
        channelMajor[dst] = interleaved[src];
    }
}
