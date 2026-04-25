// Tests for Vp9QuantizationParamsParser (slice 187).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9QuantizationParams_AllZero_IsLossless()
    {
        // base_q_idx=0, no deltas (3 zero flags).
        var data = BitsToBytes(
            (0, 8),
            (0, 1), (0, 1), (0, 1));

        var q = Vp9QuantizationParamsParser.Parse(data);

        Equal(0, q.BaseQIndex);
        Equal(0, q.YDcDeltaQ);
        Equal(0, q.UvDcDeltaQ);
        Equal(0, q.UvAcDeltaQ);
        Equal(true, q.Lossless);
    }

    [TestMethod]
    public void Vp9QuantizationParams_NonZeroBaseQIndex_NotLossless()
    {
        var data = BitsToBytes(
            (100, 8),
            (0, 1), (0, 1), (0, 1));

        var q = Vp9QuantizationParamsParser.Parse(data);

        Equal(100, q.BaseQIndex);
        Equal(false, q.Lossless);
    }

    [TestMethod]
    public void Vp9QuantizationParams_AllDeltas_PositiveValues()
    {
        // base_q_idx=64, y_dc=+3, uv_dc=+5, uv_ac=+7.
        var data = BitsToBytes(
            (64, 8),
            (1, 1), (3, 4), (0, 1),  // y_dc=+3
            (1, 1), (5, 4), (0, 1),  // uv_dc=+5
            (1, 1), (7, 4), (0, 1)); // uv_ac=+7

        var q = Vp9QuantizationParamsParser.Parse(data);

        Equal(64, q.BaseQIndex);
        Equal(3, q.YDcDeltaQ);
        Equal(5, q.UvDcDeltaQ);
        Equal(7, q.UvAcDeltaQ);
        Equal(false, q.Lossless);
    }

    [TestMethod]
    public void Vp9QuantizationParams_NegativeDelta_HandledViaSignBit()
    {
        var data = BitsToBytes(
            (50, 8),
            (1, 1), (4, 4), (1, 1),  // y_dc=-4 (mag=4, sign=1)
            (0, 1),                  // uv_dc skipped
            (1, 1), (2, 4), (1, 1)); // uv_ac=-2

        var q = Vp9QuantizationParamsParser.Parse(data);

        Equal(50, q.BaseQIndex);
        Equal(-4, q.YDcDeltaQ);
        Equal(0, q.UvDcDeltaQ);
        Equal(-2, q.UvAcDeltaQ);
    }

    [TestMethod]
    public void Vp9QuantizationParams_BaseQIdxMaxValue()
    {
        var data = BitsToBytes(
            (255, 8),
            (0, 1), (0, 1), (0, 1));

        var q = Vp9QuantizationParamsParser.Parse(data);

        Equal(255, q.BaseQIndex);
        Equal(false, q.Lossless);
    }

    [TestMethod]
    public void Vp9QuantizationParams_OnlySomeDeltasPresent()
    {
        // y_dc skipped, uv_dc=+1, uv_ac skipped.
        var data = BitsToBytes(
            (5, 8),
            (0, 1),                  // y_dc skip
            (1, 1), (1, 4), (0, 1),  // uv_dc=+1
            (0, 1));                 // uv_ac skip

        var q = Vp9QuantizationParamsParser.Parse(data);

        Equal(5, q.BaseQIndex);
        Equal(0, q.YDcDeltaQ);
        Equal(1, q.UvDcDeltaQ);
        Equal(0, q.UvAcDeltaQ);
    }
}
