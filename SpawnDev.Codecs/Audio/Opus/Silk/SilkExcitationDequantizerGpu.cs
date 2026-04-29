// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-callable SILK excitation dequantizer. Mirror of
// SilkExcitationDequantizer.Dequantize (libopus silk/decode_core.c
// excitation loop). Converts decoded pulse magnitudes into the Q14
// excitation signal that drives the LTP + LPC synthesis chain.
//
// Sequential per-stream because the PRNG state evolves sample-by-sample:
//   randSeed = RAND(randSeed)
//   randSeed = ADD_OVFLW(randSeed, pulses[i])
// One-thread-per-stream on the GPU. Multiple independent SILK streams
// (multi-channel decode) parallelize cleanly across threads.
//
// All silk macros (LSHIFT, ADD32_ovflw, RAND) inlined here.

using ILGPU;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// GPU-callable SILK excitation dequantizer. Mirror of
/// <see cref="SilkExcitationDequantizer"/>.Dequantize.
/// </summary>
public static class SilkExcitationDequantizerGpu
{
    private const int QUANT_LEVEL_ADJUST_Q10 = 80;
    private const int RAND_INCREMENT = 907633515;
    private const int RAND_MULTIPLIER = 196314165;

    /// <summary>
    /// Dequantize <paramref name="frameLength"/> pulse samples into the Q14
    /// excitation buffer. Bit-exact vs the CPU SilkExcitationDequantizer.Dequantize.
    /// </summary>
    /// <param name="excQ14">Output: excitation in Q14, length &gt;= frameLength.</param>
    /// <param name="excBase">Base offset.</param>
    /// <param name="pulses">Decoded pulse magnitudes (signed).</param>
    /// <param name="pulsesBase">Base offset.</param>
    /// <param name="signalType">SILK signal type (0 inactive, 1 unvoiced, 2 voiced).</param>
    /// <param name="quantOffsetType">SILK quantizer offset type (0 low, 1 high).</param>
    /// <param name="seed">Initial PRNG seed.</param>
    /// <param name="frameLength">Frame length in samples.</param>
    public static void DequantizeAt(
        ArrayView<int> excQ14, long excBase,
        ArrayView<short> pulses, long pulsesBase,
        int signalType, int quantOffsetType, int seed, int frameLength)
    {
        // QUANTIZATION_OFFSETS_Q10[signalType >> 1, quantOffsetType] = either 100/240 (UV) or 32/100 (V).
        int row = signalType >> 1;
        int offsetQ10 = row == 0
            ? (quantOffsetType == 0 ? 100 : 240)
            : (quantOffsetType == 0 ? 32 : 100);

        int quantAdjustQ14 = QUANT_LEVEL_ADJUST_Q10 << 4;
        int offsetQ14 = offsetQ10 << 4;

        int randSeed = seed;
        for (int i = 0; i < frameLength; i++)
        {
            // silk_RAND(seed) = RAND_INCREMENT + seed * RAND_MULTIPLIER (overflow-wrapping).
            // Use long arithmetic for portable modular semantics across all 6 ILGPU
            // backends. Cast seed via (uint) first to zero-extend negative values
            // correctly into the unsigned domain, then long-multiply, add the
            // increment, truncate to low 32 bits and re-cast as int.
            long stateExt = (long)unchecked((uint)randSeed);
            long mulFull = stateExt * RAND_MULTIPLIER;
            randSeed = unchecked((int)(mulFull + RAND_INCREMENT));

            int pulse = pulses[pulsesBase + i];
            int exc = pulse << 14;
            if (exc > 0) exc -= quantAdjustQ14;
            else if (exc < 0) exc += quantAdjustQ14;
            exc += offsetQ14;

            if (randSeed < 0) exc = -exc;

            excQ14[excBase + i] = exc;

            // silk_ADD32_ovflw(randSeed, pulses[i]) - low-32-bit add. unchecked(a + b)
            // gives the same bit pattern as the (int)((uint)a + (uint)b) form on every backend.
            randSeed = unchecked(randSeed + pulse);
        }
    }
}
