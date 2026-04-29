// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-callable SILK AR2 IIR pre-filter. Mirror of the private
// silk_resampler_private_AR2 inside libopus silk/resampler_private_AR2.c.
// Used as the IIR stage of the downsample-FIR chain (down_FIR variants);
// produces a Q8 output stream from an int16 input stream + 2 Q14 coefs.
//
// Sequential per-stream: each output sample depends on the prior IIR
// state in S[0..1]. Per the cardinal rule, one-thread-per-stream on the
// GPU. Multiple independent streams parallelize cleanly across threads.
//
// All silk macros (LSHIFT, ADD_LSHIFT32, SMULWB, SMLAWB) inlined here.

using ILGPU;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// GPU-callable SILK AR2 IIR pre-filter. Mirror of libopus
/// <c>silk_resampler_private_AR2</c>.
/// </summary>
public static class SilkResamplerAr2Gpu
{
    /// <summary>
    /// Run the AR2 IIR filter for <paramref name="len"/> input samples,
    /// producing <paramref name="len"/> Q8 outputs into
    /// <paramref name="outQ8"/>. Updates <paramref name="state"/> in place.
    /// Bit-exact vs the CPU SilkResampler.Ar2.
    /// </summary>
    public static void ApplyAt(
        ArrayView<int> state, long stateBase,
        ArrayView<int> outQ8, long outBase,
        ArrayView<short> input, long inBase,
        ArrayView<short> aQ14, long aBase,
        int len)
    {
        int s0 = state[stateBase + 0];
        int s1 = state[stateBase + 1];
        short a0 = aQ14[aBase + 0];
        short a1 = aQ14[aBase + 1];

        for (int k = 0; k < len; k++)
        {
            // out32 = S[0] + (input[k] << 8)
            int out32 = s0 + ((int)input[inBase + k] << 8);
            outQ8[outBase + k] = out32;

            // out32 <<= 2 (Q8 -> Q10)
            out32 <<= 2;

            // S[0] = S[1] + SMULWB(out32, a0)
            s0 = s1 + (int)((long)out32 * a0 >> 16);
            // S[1] = SMULWB(out32, a1)
            s1 = (int)((long)out32 * a1 >> 16);
        }

        state[stateBase + 0] = s0;
        state[stateBase + 1] = s1;
    }
}
