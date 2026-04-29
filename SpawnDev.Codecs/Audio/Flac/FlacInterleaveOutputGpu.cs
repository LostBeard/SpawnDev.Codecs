// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-callable per-sample channel-major -> sample-major interleave for the
// FLAC decoder's PCM output stage. Mirror of the per-sample interleave
// loop that previously ran on the host inside FlacDecoderGpu.DecodeStreamAsync.
//
// Per-(channel, sample) parallel: each thread reads one source value
// from the channel-major buffer and writes it to the corresponding
// position in the interleaved output. True parallel-per-element across
// all 6 ILGPU backends.
//
// Caller dispatches numFrames * channels threads.

using ILGPU;

namespace SpawnDev.Codecs.Audio.Flac;

/// <summary>
/// GPU-callable FLAC output interleave (int samples). Mirror of the
/// per-sample interleave loop that previously ran on the host inside
/// <see cref="FlacDecoderGpu"/>.DecodeStreamAsync.
/// </summary>
public static class FlacInterleaveOutputGpu
{
    /// <summary>
    /// Compute one output sample at thread index <paramref name="threadIdx"/>
    /// in [0, numFrames * channels). Maps n = threadIdx / channels and
    /// ch = threadIdx % channels.
    /// </summary>
    /// <param name="channelMajor">Per-channel PCM samples (length channels * numFrames),
    /// channel ch's samples at indices [ch * numFrames, (ch+1) * numFrames).</param>
    /// <param name="cmBase">Base offset.</param>
    /// <param name="interleavedOut">Output PCM (length numFrames * channels).</param>
    /// <param name="outBase">Base offset.</param>
    /// <param name="channels">Channel count.</param>
    /// <param name="numFrames">Per-channel frame count.</param>
    /// <param name="threadIdx">Linear thread index in [0, numFrames * channels).</param>
    public static void InterleaveAt(
        ArrayView<int> channelMajor, long cmBase,
        ArrayView<int> interleavedOut, long outBase,
        int channels, int numFrames, int threadIdx)
    {
        int n = threadIdx / channels;
        int ch = threadIdx - n * channels;

        long src = cmBase + (long)ch * numFrames + n;
        long dst = outBase + (long)n * channels + ch;
        interleavedOut[dst] = channelMajor[src];
    }
}
