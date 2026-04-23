using SpawnDev.Codecs.Audio.Opus.Silk;
using SpawnDev.Codecs.EntropyCoders;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// End-to-end round-trip tests for the full SILK excitation (pulses) decoder.
/// Combines the rate-level selection, per-block pulse-count decode (including
/// optional LSB-extension for large pulses), shell coder for per-sample
/// magnitudes, LSB bit extension, and sign decoding - and verifies via its
/// companion encoder that the whole pipeline is bit-exact.
/// </summary>
public abstract partial class CodecsTestBase
{
    // -------- Baseline: all-zero excitation --------

    [TestMethod]
    public void PulsesDecoder_AllZero_RoundTrip_NbFrame()
    {
        int frameLength = 160; // NB 20 ms = 10 shell blocks x 16 samples
        short[] pulses = new short[frameLength];

        var enc = new OpusRangeEncoder(256);
        SilkPulsesDecoder.Encode(enc, pulses, signalType: 1, quantOffsetType: 0,
            frameLength: frameLength, rateLevelIndex: 4);
        enc.Done();

        short[] decoded = new short[frameLength];
        var dec = new OpusRangeDecoder(enc.ToArray());
        SilkPulsesDecoder.Decode(decoded, dec, signalType: 1, quantOffsetType: 0, frameLength: frameLength);

        for (int i = 0; i < frameLength; i++) Equal((short)0, decoded[i], $"pos {i}");
    }

    // -------- Single non-zero pulse --------

    [TestMethod]
    public void PulsesDecoder_SinglePositivePulse_RoundTrip()
    {
        // One pulse in block 2, position 5, magnitude +3, for a WB 20 ms frame (320 samples).
        int frameLength = 320;
        short[] pulses = new short[frameLength];
        pulses[2 * 16 + 5] = 3;

        var enc = new OpusRangeEncoder(256);
        SilkPulsesDecoder.Encode(enc, pulses, signalType: 2, quantOffsetType: 1,
            frameLength: frameLength, rateLevelIndex: 3);
        enc.Done();

        short[] decoded = new short[frameLength];
        var dec = new OpusRangeDecoder(enc.ToArray());
        SilkPulsesDecoder.Decode(decoded, dec, signalType: 2, quantOffsetType: 1, frameLength: frameLength);

        for (int i = 0; i < frameLength; i++) Equal(pulses[i], decoded[i], $"pos {i}");
    }

    [TestMethod]
    public void PulsesDecoder_SingleNegativePulse_RoundTrip()
    {
        int frameLength = 160;
        short[] pulses = new short[frameLength];
        pulses[7 * 16 + 12] = -4;

        var enc = new OpusRangeEncoder(256);
        SilkPulsesDecoder.Encode(enc, pulses, signalType: 2, quantOffsetType: 0,
            frameLength: frameLength, rateLevelIndex: 5);
        enc.Done();

        short[] decoded = new short[frameLength];
        var dec = new OpusRangeDecoder(enc.ToArray());
        SilkPulsesDecoder.Decode(decoded, dec, signalType: 2, quantOffsetType: 0, frameLength: frameLength);

        for (int i = 0; i < frameLength; i++) Equal(pulses[i], decoded[i], $"pos {i}");
    }

    // -------- Distributed pulses across blocks --------

    [TestMethod]
    public void PulsesDecoder_DistributedAcrossBlocks_RoundTrip()
    {
        int frameLength = 320;
        short[] pulses = new short[frameLength];
        // Scatter positive and negative pulses across several blocks.
        pulses[0 * 16 + 3] = 2;
        pulses[0 * 16 + 11] = -1;
        pulses[3 * 16 + 0] = 1;
        pulses[3 * 16 + 15] = -3;
        pulses[10 * 16 + 7] = 5;
        pulses[19 * 16 + 15] = -2;

        var enc = new OpusRangeEncoder(512);
        SilkPulsesDecoder.Encode(enc, pulses, signalType: 2, quantOffsetType: 1,
            frameLength: frameLength, rateLevelIndex: 2);
        enc.Done();

        short[] decoded = new short[frameLength];
        var dec = new OpusRangeDecoder(enc.ToArray());
        SilkPulsesDecoder.Decode(decoded, dec, signalType: 2, quantOffsetType: 1, frameLength: frameLength);

        for (int i = 0; i < frameLength; i++) Equal(pulses[i], decoded[i], $"pos {i}");
    }

    // -------- LSB-extension path (large pulses) --------

    [TestMethod]
    public void PulsesDecoder_LargePulseRequiringLsbExtension_RoundTrip()
    {
        // One block with a lot of pulses - sum > SILK_MAX_PULSES (16) triggers the
        // escape path with nLshifts >= 1. Choose magnitudes that cleanly shift-right.
        int frameLength = 160;
        short[] pulses = new short[frameLength];
        // Block 2: put two pulses of magnitude 16 (sum of abs = 32, > SILK_MAX_PULSES).
        // After one lshift, sum is 16 (at the limit). Values 16 round-trip cleanly (no LSB remainder).
        pulses[2 * 16 + 4] = 16;
        pulses[2 * 16 + 11] = -16;

        var enc = new OpusRangeEncoder(512);
        SilkPulsesDecoder.Encode(enc, pulses, signalType: 2, quantOffsetType: 1,
            frameLength: frameLength, rateLevelIndex: 7);
        enc.Done();

        short[] decoded = new short[frameLength];
        var dec = new OpusRangeDecoder(enc.ToArray());
        SilkPulsesDecoder.Decode(decoded, dec, signalType: 2, quantOffsetType: 1, frameLength: frameLength);

        for (int i = 0; i < frameLength; i++) Equal(pulses[i], decoded[i], $"pos {i}");
    }

    // -------- Randomized small-magnitude distributions --------

    [TestMethod]
    public void PulsesDecoder_RandomSmallDistributions_RoundTrip()
    {
        int frameLength = 160;
        var rng = new Random(0x12345678);

        for (int trial = 0; trial < 30; trial++)
        {
            short[] pulses = new short[frameLength];
            // Distribute up to SILK_MAX_PULSES absolute units per block, with random signs.
            for (int block = 0; block < 10; block++)
            {
                int remaining = rng.Next(0, SilkConstants.SILK_MAX_PULSES + 1);
                while (remaining > 0)
                {
                    int pos = rng.Next(0, 16);
                    int take = rng.Next(1, Math.Min(4, remaining) + 1);
                    int sign = rng.Next(0, 2) == 0 ? -1 : 1;
                    pulses[block * 16 + pos] = (short)(pulses[block * 16 + pos] + sign * take);
                    // Handle the rare case where signs cancel - keep summing abs values only
                    // approximately (we'll validate after).
                    remaining -= take;
                }
            }

            // Enforce per-block abs-sum <= SILK_MAX_PULSES so the encoder does not need LSB extension.
            int[] blockSums = new int[10];
            for (int k = 0; k < frameLength; k++) blockSums[k / 16] += Math.Abs((int)pulses[k]);
            bool needsAdjust = false;
            for (int block = 0; block < 10; block++)
            {
                while (blockSums[block] > SilkConstants.SILK_MAX_PULSES)
                {
                    // Trim the first non-zero pulse in the block.
                    for (int p = 0; p < 16; p++)
                    {
                        int idx = block * 16 + p;
                        if (pulses[idx] != 0)
                        {
                            int abs = Math.Abs((int)pulses[idx]);
                            int reduce = Math.Min(abs, blockSums[block] - SilkConstants.SILK_MAX_PULSES);
                            pulses[idx] = (short)((pulses[idx] > 0 ? 1 : -1) * (abs - reduce));
                            blockSums[block] -= reduce;
                            needsAdjust = true;
                            if (blockSums[block] <= SilkConstants.SILK_MAX_PULSES) break;
                        }
                    }
                }
            }

            int rateLevel = rng.Next(0, 9);
            var enc = new OpusRangeEncoder(512);
            SilkPulsesDecoder.Encode(enc, pulses, signalType: 2, quantOffsetType: 0,
                frameLength: frameLength, rateLevelIndex: rateLevel);
            enc.Done();

            short[] decoded = new short[frameLength];
            var dec = new OpusRangeDecoder(enc.ToArray());
            SilkPulsesDecoder.Decode(decoded, dec, signalType: 2, quantOffsetType: 0, frameLength: frameLength);

            for (int i = 0; i < frameLength; i++)
            {
                if (pulses[i] != decoded[i])
                {
                    throw new Exception(
                        $"Trial {trial} rateLevel={rateLevel} adjust={needsAdjust} pos {i}: " +
                        $"expected {pulses[i]}, got {decoded[i]}");
                }
            }
        }
    }

    // -------- Argument validation --------

    [TestMethod]
    public void PulsesDecoder_NullRangeDecoder_Throws()
    {
        short[] pulses = new short[160];
        Throws<ArgumentNullException>(() =>
            SilkPulsesDecoder.Decode(pulses, null!, 1, 0, 160));
    }

    [TestMethod]
    public void PulsesDecoder_InvalidSignalType_Throws()
    {
        var enc = new OpusRangeEncoder(32);
        enc.Done();
        var dec = new OpusRangeDecoder(enc.ToArray());
        short[] pulses = new short[160];
        Throws<ArgumentOutOfRangeException>(() =>
            SilkPulsesDecoder.Decode(pulses, dec, signalType: 3, quantOffsetType: 0, frameLength: 160));
    }

    [TestMethod]
    public void PulsesDecoder_InvalidQuantOffset_Throws()
    {
        var enc = new OpusRangeEncoder(32);
        enc.Done();
        var dec = new OpusRangeDecoder(enc.ToArray());
        short[] pulses = new short[160];
        Throws<ArgumentOutOfRangeException>(() =>
            SilkPulsesDecoder.Decode(pulses, dec, signalType: 1, quantOffsetType: 2, frameLength: 160));
    }
}
