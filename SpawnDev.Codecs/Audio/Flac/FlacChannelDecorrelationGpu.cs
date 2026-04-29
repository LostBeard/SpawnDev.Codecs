// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-callable FLAC stereo channel decorrelation. Mirror of the
// per-sample channel-decorrelation step inside FlacFrameDecoder.Decode
// (RFC 9639 Section 8.1.5 inter-channel decorrelation). Expands the
// decoded LeftSide / RightSide / MidSide stereo encodings back into
// independent L/R samples in place.
//
// Per-sample independent: each thread reads samples[n] (channel 0) and
// samples[blockSize + n] (channel 1), computes the L/R pair according
// to the assignment mode, and writes both back to the same indices.
// True parallel-per-sample across all 6 ILGPU backends.
//
// Mode is the FlacChannelAssignment value:
//   8  LeftSide  - ch0 = L, ch1 = L - R          -> R = ch0 - ch1
//   9  RightSide - ch0 = L - R, ch1 = R          -> L = ch0 + ch1
//   10 MidSide   - ch0 = mid, ch1 = side         -> libFLAC mid-scale recovery
// Other modes (Independent / 3-7 channel layouts) are no-ops by construction.

using ILGPU;

namespace SpawnDev.Codecs.Audio.Flac;

/// <summary>
/// GPU-callable FLAC stereo decorrelation. Mirror of the per-sample loop
/// inside <see cref="FlacFrameDecoder"/>.Decode that converts side-encoded
/// stereo back to L/R.
/// </summary>
public static class FlacChannelDecorrelationGpu
{
    /// <summary>FlacChannelAssignment.LeftSide.</summary>
    public const int MODE_LEFT_SIDE = 8;
    /// <summary>FlacChannelAssignment.RightSide.</summary>
    public const int MODE_RIGHT_SIDE = 9;
    /// <summary>FlacChannelAssignment.MidSide.</summary>
    public const int MODE_MID_SIDE = 10;

    /// <summary>
    /// Convert one stereo sample pair at index <paramref name="n"/> from
    /// the encoded form to L/R in place. samples buffer is laid out as
    /// [ch0_block, ch1_block] with ch0 at samples[0..blockSize) and ch1
    /// at samples[blockSize..2*blockSize). Bit-exact vs the CPU
    /// FlacFrameDecoder.Decode decorrelation step.
    /// </summary>
    /// <param name="samples">In/out interleaved-block stereo samples (length &gt;= 2*blockSize).</param>
    /// <param name="samplesBase">Base offset.</param>
    /// <param name="blockSize">Per-channel block size in samples.</param>
    /// <param name="mode">FlacChannelAssignment numeric value (8/9/10).</param>
    /// <param name="n">Sample index in [0, blockSize).</param>
    public static void DecorrelateAt(
        ArrayView<int> samples, long samplesBase,
        int blockSize, int mode, int n)
    {
        long aIdx = samplesBase + n;
        long bIdx = samplesBase + blockSize + n;
        int a = samples[aIdx];
        int b = samples[bIdx];

        if (mode == MODE_LEFT_SIDE)
        {
            // ch0 = L (unchanged), ch1 = L - R -> R = L - side = a - b
            samples[bIdx] = a - b;
        }
        else if (mode == MODE_RIGHT_SIDE)
        {
            // ch0 = L - R, ch1 = R (unchanged) -> L = side + R = a + b
            samples[aIdx] = a + b;
        }
        else if (mode == MODE_MID_SIDE)
        {
            // libFLAC: mid_scaled = (mid << 1) | (side & 1)
            //          L = (mid_scaled + side) >> 1
            //          R = (mid_scaled - side) >> 1
            int mid = a;
            int side = b;
            int midScaled = (mid << 1) | (side & 1);
            samples[aIdx] = (midScaled + side) >> 1;
            samples[bIdx] = (midScaled - side) >> 1;
        }
        // else: no-op (Independent or unsupported mode).
    }
}
