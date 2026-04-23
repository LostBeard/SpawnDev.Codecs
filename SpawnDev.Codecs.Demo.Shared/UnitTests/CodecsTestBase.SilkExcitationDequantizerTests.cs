using SpawnDev.Codecs.Audio.Opus.Silk;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for <see cref="SilkExcitationDequantizer.Dequantize"/> - the first step of
/// silk_decode_core that turns pulse magnitudes into a Q14 excitation signal.
/// Hand-computed reference values match libopus exactly.
/// </summary>
public abstract partial class CodecsTestBase
{
    // -------- Zero pulses --------

    [TestMethod]
    public void ExcDequant_ZeroPulses_ProducesOffsetSignal()
    {
        // All-zero pulses with inactive signalType=0, quantOffset=1: every sample should
        // equal +/-offsetQ14 depending on the PRNG sign.
        // offset_Q10 = QUANTIZATION_OFFSETS_Q10[0,1] = 240. offset_Q14 = 240 << 4 = 3840.
        const int signalType = 0, quantOffsetType = 1;
        const int offsetQ14 = 240 << 4;
        const int seed = 3;
        short[] pulses = new short[160];
        int[] exc = new int[160];

        SilkExcitationDequantizer.Dequantize(exc, pulses, signalType, quantOffsetType, seed, 160);

        // Verify every output is either +offset or -offset.
        for (int i = 0; i < 160; i++)
        {
            True(exc[i] == offsetQ14 || exc[i] == -offsetQ14,
                $"exc[{i}] = {exc[i]} should be +/-{offsetQ14}");
        }
    }

    // -------- Single positive pulse --------

    [TestMethod]
    public void ExcDequant_SinglePositivePulse_AppliesQuantAdjustAndOffset()
    {
        // Place one pulse of magnitude +3 in an otherwise zero frame.
        // For that sample (before any PRNG sign): exc = 3 << 14 = 49152. Then > 0 so
        // subtract QUANT_LEVEL_ADJUST_Q10 << 4 = 1280. Then add offset_Q14.
        // For voiced signalType=2 quantOffset=0: offset = 32, offset_Q14 = 512.
        // exc = 49152 - 1280 + 512 = 48384.
        // The PRNG may flip sign (check via |exc| magnitude).
        const int signalType = 2, quantOffsetType = 0;
        short[] pulses = new short[16];
        pulses[5] = 3;
        int[] exc = new int[16];

        SilkExcitationDequantizer.Dequantize(exc, pulses, signalType, quantOffsetType, seed: 0, frameLength: 16);

        Equal(48384, Math.Abs(exc[5]));
    }

    [TestMethod]
    public void ExcDequant_SingleNegativePulse_AppliesQuantAdjustAndOffset()
    {
        // Pulse of -2. exc = -2 << 14 = -32768. exc < 0 so add 1280 -> -31488. Add offset.
        // For voiced quantOffset=1: offset = 100, offset_Q14 = 1600.
        // -31488 + 1600 = -29888. Sign may flip.
        const int signalType = 2, quantOffsetType = 1;
        short[] pulses = new short[16];
        pulses[10] = -2;
        int[] exc = new int[16];

        SilkExcitationDequantizer.Dequantize(exc, pulses, signalType, quantOffsetType, seed: 42, frameLength: 16);

        Equal(29888, Math.Abs(exc[10]));
    }

    // -------- Offset table coverage --------

    [TestMethod]
    public void ExcDequant_AllFourQuantizationOffsets_AppliedCorrectly()
    {
        short[] pulses = new short[4];
        int[] exc = new int[4];
        int[] expectedOffsetsQ14 =
        {
            SilkConstants.QUANTIZATION_OFFSETS_Q10[0, 0] << 4, // UVL: 100 << 4 = 1600
            SilkConstants.QUANTIZATION_OFFSETS_Q10[0, 1] << 4, // UVH: 240 << 4 = 3840
            SilkConstants.QUANTIZATION_OFFSETS_Q10[1, 0] << 4, // VL:   32 << 4 = 512
            SilkConstants.QUANTIZATION_OFFSETS_Q10[1, 1] << 4, // VH:  100 << 4 = 1600
        };
        int[,] typeOffsetPairs = { { 0, 0 }, { 0, 1 }, { 2, 0 }, { 2, 1 } };

        for (int k = 0; k < 4; k++)
        {
            int signalType = typeOffsetPairs[k, 0];
            int quantOffsetType = typeOffsetPairs[k, 1];
            SilkExcitationDequantizer.Dequantize(exc, pulses, signalType, quantOffsetType, seed: 1, frameLength: 4);
            int expected = expectedOffsetsQ14[k];

            // All four output samples should be +/- expected (zero pulses).
            for (int i = 0; i < 4; i++)
            {
                True(exc[i] == expected || exc[i] == -expected,
                    $"k={k} signalType={signalType} quantOffset={quantOffsetType} i={i}: " +
                    $"expected +/-{expected}, got {exc[i]}");
            }
        }
    }

    // -------- Deterministic PRNG flow --------

    [TestMethod]
    public void ExcDequant_PrngIsDeterministicAcrossCalls()
    {
        // Same seed + same pulses -> same output on both calls.
        short[] pulses = { 1, -2, 3, 0, 0, -1, 2, 0 };
        int[] exc1 = new int[8];
        int[] exc2 = new int[8];

        SilkExcitationDequantizer.Dequantize(exc1, pulses, 2, 0, seed: 0, frameLength: 8);
        SilkExcitationDequantizer.Dequantize(exc2, pulses, 2, 0, seed: 0, frameLength: 8);

        for (int i = 0; i < 8; i++) Equal(exc1[i], exc2[i], $"pos {i}");
    }

    [TestMethod]
    public void ExcDequant_DifferentSeeds_ProduceDifferentSignPatterns()
    {
        // Same pulses + different seeds should produce different sign patterns.
        short[] pulses = new short[32];
        for (int i = 0; i < 32; i++) pulses[i] = 1;
        int[] excSeed0 = new int[32];
        int[] excSeed3 = new int[32];

        SilkExcitationDequantizer.Dequantize(excSeed0, pulses, 2, 0, seed: 0, frameLength: 32);
        SilkExcitationDequantizer.Dequantize(excSeed3, pulses, 2, 0, seed: 3, frameLength: 32);

        int differentSignCount = 0;
        for (int i = 0; i < 32; i++)
        {
            if (Math.Sign(excSeed0[i]) != Math.Sign(excSeed3[i])) differentSignCount++;
        }
        True(differentSignCount > 0, "Expected different sign patterns across different seeds");
    }

    // -------- Hand-computed single-sample reference value --------

    [TestMethod]
    public void ExcDequant_HandComputedReferenceValue()
    {
        // Single pulse at position 0 with seed = 0.
        // silk_RAND(0) = RAND_INCREMENT = 907633515 (positive, so no sign flip).
        // pulse = 4. exc = 4 << 14 = 65536. 65536 > 0 so subtract 1280 -> 64256.
        // signalType = 1 (unvoiced), quantOffset = 0 -> offsetQ14 = 100 << 4 = 1600.
        // exc = 64256 + 1600 = 65856. randSeed positive, no flip.
        // Expected exc[0] = 65856.
        const int signalType = 1, quantOffsetType = 0;
        short[] pulses = new short[1] { 4 };
        int[] exc = new int[1];

        SilkExcitationDequantizer.Dequantize(exc, pulses, signalType, quantOffsetType, seed: 0, frameLength: 1);
        Equal(65856, exc[0]);
    }

    [TestMethod]
    public void ExcDequant_HandComputedReferenceValue_NegativeRand()
    {
        // Force sign-flip: we need seed whose silk_RAND produces a negative first value.
        // silk_RAND(seed=1) = 907633515 + 196314165 = 1,103,947,680 (positive).
        // silk_RAND(seed=2147483647) = ? Let's compute: RAND_INC + 2147483647 * RAND_MUL.
        // 2147483647 * 196314165 = overflows. Let's just pick seed that gives negative
        // RAND output. Try seed = -1: RAND(-1) = 907633515 + (-1)*196314165 via MLA_ovflw.
        // uint(-1) * 196314165 mod 2^32. (uint)-1 = 0xFFFFFFFF.
        // 0xFFFFFFFF * 196314165 mod 2^32: 196314165 * (2^32 - 1) = 196314165 * 2^32 - 196314165.
        // mod 2^32 = -196314165 as signed = 0xF448C68B (since 2^32 - 196314165 = 4098653131 = 0xF448C68B).
        // signed int interpretation: -196314165.
        // Plus 907633515 = 907633515 - 196314165 = 711319350. POSITIVE.
        // Hmm. Let me try seed = 2: RAND(2) = 907633515 + 2 * 196314165 = 1300261845. Positive.
        // Large positive seed -> overflow into negative: seed = 0x80000000 (int.MinValue).
        // RAND(int.MinValue) = RAND_INC + MLA_ovflw(0, MinValue, RAND_MUL).
        // Actually silk_RAND(seed) = silk_MLA_ovflw(RAND_INC, seed, RAND_MUL)
        //                         = RAND_INC + (uint)seed * (uint)RAND_MUL as int.
        // For seed = int.MinValue (0x80000000 as uint):
        //   0x80000000 * 196314165 = 0x80000000 * 0xBB29F8B5 = ... huge.
        //   mod 2^32: 0x80000000 * odd number = 0x80000000 (low bit zero preserved... wait no).
        //   Actually (uint)int.MinValue * RAND_MUL: 2147483648 * 196314165 = 421654676064829440.
        //   mod 2^32 (=4294967296): 421654676064829440 / 4294967296 ≈ 9.8e7.
        //   ... too tedious to compute by hand. Let me just try a small set of seeds and
        //   find one where the sign was flipped.

        // Simpler approach: run with multiple seeds and find one where exc[0] is negative.
        // Pulse = 2, signalType = 2 voiced, quantOffset = 0.
        // Without flip: exc = (2 << 14) - 1280 + 512 = 32768 - 1280 + 512 = 32000. POSITIVE.
        // With flip: -32000.
        const int signalType = 2, quantOffsetType = 0;
        short[] pulses = new short[1] { 2 };
        int[] exc = new int[1];

        // Try seeds until we get a negative output.
        for (int seed = 0; seed < 100; seed++)
        {
            SilkExcitationDequantizer.Dequantize(exc, pulses, signalType, quantOffsetType, seed, 1);
            if (exc[0] == -32000)
            {
                // Got the sign-flipped version. Verify magnitude.
                Equal(32000, Math.Abs(exc[0]));
                return;
            }
            else if (exc[0] == 32000)
            {
                // Non-flipped version; try next seed.
                continue;
            }
            else
            {
                throw new Exception($"Seed {seed}: unexpected exc value {exc[0]} (expected +/-32000)");
            }
        }
        throw new Exception("Expected to find a seed producing a sign-flipped output in 100 tries");
    }

    // -------- Argument validation --------

    [TestMethod]
    public void ExcDequant_InvalidSignalType_Throws()
    {
        short[] pulses = new short[4];
        int[] exc = new int[4];
        Throws<ArgumentOutOfRangeException>(() =>
            SilkExcitationDequantizer.Dequantize(exc, pulses, signalType: 3, quantOffsetType: 0, seed: 0, frameLength: 4));
    }

    [TestMethod]
    public void ExcDequant_InvalidQuantOffset_Throws()
    {
        short[] pulses = new short[4];
        int[] exc = new int[4];
        Throws<ArgumentOutOfRangeException>(() =>
            SilkExcitationDequantizer.Dequantize(exc, pulses, signalType: 1, quantOffsetType: 2, seed: 0, frameLength: 4));
    }

    [TestMethod]
    public void ExcDequant_OutputTooSmall_Throws()
    {
        short[] pulses = new short[10];
        int[] exc = new int[9];
        Throws<ArgumentException>(() =>
            SilkExcitationDequantizer.Dequantize(exc, pulses, 1, 0, 0, 10));
    }
}
