// AV1 dequantization table tests. Spot-check known libaom values
// across qindex range + bit depth.

using SpawnDev.Codecs.Video.Av1;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Av1DequantTables_HaveCorrectLengths()
    {
        Equal(256, Av1DequantTables.DcLookup8.Length);
        Equal(256, Av1DequantTables.DcLookup10.Length);
        Equal(256, Av1DequantTables.DcLookup12.Length);
        Equal(256, Av1DequantTables.AcLookup8.Length);
        Equal(256, Av1DequantTables.AcLookup10.Length);
        Equal(256, Av1DequantTables.AcLookup12.Length);
    }

    [TestMethod]
    public void Av1DequantTables_DcQuantQtx_MatchesLibaomValues()
    {
        // Spot-check known libaom values from the source table:
        Equal(4, Av1DequantTables.DcQuantQtx(0, 0, 8));
        Equal(8, Av1DequantTables.DcQuantQtx(1, 0, 8));
        Equal(138, Av1DequantTables.DcQuantQtx(127, 0, 8)); // dc_qlookup_QTX[127] = 138
        Equal(1336, Av1DequantTables.DcQuantQtx(255, 0, 8));
        Equal(4, Av1DequantTables.DcQuantQtx(0, 0, 10));
        Equal(5347, Av1DequantTables.DcQuantQtx(255, 0, 10));
        Equal(21387, Av1DequantTables.DcQuantQtx(255, 0, 12));
    }

    [TestMethod]
    public void Av1DequantTables_AcQuantQtx_MatchesLibaomValues()
    {
        Equal(4, Av1DequantTables.AcQuantQtx(0, 0, 8));
        Equal(8, Av1DequantTables.AcQuantQtx(1, 0, 8));
        Equal(1828, Av1DequantTables.AcQuantQtx(255, 0, 8));
        Equal(7312, Av1DequantTables.AcQuantQtx(255, 0, 10));
        Equal(29247, Av1DequantTables.AcQuantQtx(255, 0, 12));
    }

    [TestMethod]
    public void Av1DequantTables_DeltaShiftsQindex()
    {
        // qindex 100 + delta -10 -> lookup at 90
        // libaom: dc_qlookup_QTX[90] = 81 (counted from table)
        Equal(81, Av1DequantTables.DcQuantQtx(100, -10, 8));
    }

    [TestMethod]
    public void Av1DequantTables_DeltaClampsToRange()
    {
        // qindex 200 + delta 100 -> clamp to 255 -> 1336
        Equal(1336, Av1DequantTables.DcQuantQtx(200, 100, 8));
        // qindex 50 + delta -100 -> clamp to 0 -> 4
        Equal(4, Av1DequantTables.DcQuantQtx(50, -100, 8));
    }

    [TestMethod]
    public void Av1SmoothWeights_HaveExpectedLayout()
    {
        // Total length should be 4+8+16+32+64 = 124
        Equal(124, Av1SmoothWeights.Weights.Length);
        // First entry of each block size = 255 (max)
        Equal((byte)255, Av1SmoothWeights.Weights[0]);   // bs=4
        Equal((byte)255, Av1SmoothWeights.Weights[4]);   // bs=8
        Equal((byte)255, Av1SmoothWeights.Weights[12]);  // bs=16
        Equal((byte)255, Av1SmoothWeights.Weights[28]);  // bs=32
        Equal((byte)255, Av1SmoothWeights.Weights[60]);  // bs=64
    }

    [TestMethod]
    public void Av1SmoothWeights_GetWeights_SlicesCorrectRange()
    {
        Equal(4, Av1SmoothWeights.GetWeights(4).Length);
        Equal(8, Av1SmoothWeights.GetWeights(8).Length);
        Equal(16, Av1SmoothWeights.GetWeights(16).Length);
        Equal(32, Av1SmoothWeights.GetWeights(32).Length);
        Equal(64, Av1SmoothWeights.GetWeights(64).Length);
        // Spot check: bs=4 weights = {255, 149, 85, 64}
        var w4 = Av1SmoothWeights.GetWeights(4);
        Equal((byte)255, w4[0]);
        Equal((byte)149, w4[1]);
        Equal((byte)85, w4[2]);
        Equal((byte)64, w4[3]);
        // bs=8 last weight = 32
        Equal((byte)32, Av1SmoothWeights.GetWeights(8)[7]);
    }
}
