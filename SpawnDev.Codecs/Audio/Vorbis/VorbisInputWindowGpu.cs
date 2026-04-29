// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-callable per-sample windowed-copy with zero-pad for the Vorbis
// encoder's per-packet input prep stage. Mirror of the per-sample loop
// that previously ran on the host inside VorbisAudioEncoderGpu.EncodeStreamAsync
// to slice a sliding window of source PCM (with leading zero-pad on the
// first overlap-priming packet) into the per-packet input buffer.
//
// Per-sample parallel: each thread reads one source value (or 0 when the
// source index falls outside the source range) and writes to the
// corresponding position in the per-packet input. True parallel-per-element
// across all 6 ILGPU backends.
//
// Caller dispatches blockSize threads per packet.

using ILGPU;

namespace SpawnDev.Codecs.Audio.Vorbis;

/// <summary>
/// GPU-callable Vorbis per-packet windowed input prep. Computes one sample
/// of the per-packet input buffer at thread index <paramref name="threadIdx"/>:
/// reads from <paramref name="srcMono"/>[<paramref name="srcStart"/> +
/// threadIdx] when that index is in [0, totalSamples), or writes 0 otherwise.
/// </summary>
public static class VorbisInputWindowGpu
{
    /// <summary>
    /// Compute one output sample at thread index <paramref name="threadIdx"/>
    /// in [0, blockSize).
    /// </summary>
    /// <param name="srcMono">Full source mono PCM buffer.</param>
    /// <param name="srcMonoBase">Base offset into <paramref name="srcMono"/>.</param>
    /// <param name="totalSamples">Length of source mono PCM (in samples).</param>
    /// <param name="srcStart">Starting source index for this packet (may be negative
    /// on the first overlap-priming packet to request zero-pad).</param>
    /// <param name="dst">Output per-packet input buffer (length blockSize).</param>
    /// <param name="dstBase">Base offset into <paramref name="dst"/>.</param>
    /// <param name="threadIdx">Linear thread index in [0, blockSize).</param>
    public static void WindowedCopyAt(
        ArrayView<float> srcMono, long srcMonoBase, int totalSamples,
        int srcStart,
        ArrayView<float> dst, long dstBase,
        int threadIdx)
    {
        int srcIdx = srcStart + threadIdx;
        float sample = (srcIdx >= 0 && srcIdx < totalSamples)
            ? srcMono[srcMonoBase + srcIdx]
            : 0f;
        dst[dstBase + threadIdx] = sample;
    }
}
