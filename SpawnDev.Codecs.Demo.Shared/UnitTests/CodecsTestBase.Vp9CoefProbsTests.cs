// Tests for Vp9CoefProbs (slice 140). Shape + pinned-value checks
// against libvpx vp9_entropy.c.

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9CoefProbs_CatProbs_HaveCorrectLengthsAndPinnedValues()
    {
        // Lengths from libvpx (cat<n>_prob has n entries, except cat6
        // which has 14 for 8-bit / 18 for 12-bit).
        Equal(1,  Vp9CoefProbs.Cat1Prob.Length);
        Equal(2,  Vp9CoefProbs.Cat2Prob.Length);
        Equal(3,  Vp9CoefProbs.Cat3Prob.Length);
        Equal(4,  Vp9CoefProbs.Cat4Prob.Length);
        Equal(5,  Vp9CoefProbs.Cat5Prob.Length);
        Equal(14, Vp9CoefProbs.Cat6Prob.Length);
        Equal(18, Vp9CoefProbs.Cat6ProbHigh12.Length);

        // Pinned values from libvpx.
        Equal((byte)159, Vp9CoefProbs.Cat1Prob[0]);
        Equal((byte)165, Vp9CoefProbs.Cat2Prob[0]);
        Equal((byte)145, Vp9CoefProbs.Cat2Prob[1]);
        Equal((byte)254, Vp9CoefProbs.Cat6Prob[0]);
        Equal((byte)129, Vp9CoefProbs.Cat6Prob[13]);
        Equal((byte)255, Vp9CoefProbs.Cat6ProbHigh12[0]);
        Equal((byte)129, Vp9CoefProbs.Cat6ProbHigh12[17]);
    }

    [TestMethod]
    public void Vp9CoefProbs_Pareto8Full_HasCorrectShape()
    {
        // 255 x 8 byte table from libvpx.
        Equal(255, Vp9CoefProbs.Pareto8Full.GetLength(0));
        Equal(8,   Vp9CoefProbs.Pareto8Full.GetLength(1));
    }

    [TestMethod]
    public void Vp9CoefProbs_Pareto8Full_PinnedFirstAndLastRows()
    {
        // First row: { 3, 86, 128, 6, 86, 23, 88, 29 }
        Equal((byte)3,   Vp9CoefProbs.Pareto8Full[0, 0]);
        Equal((byte)86,  Vp9CoefProbs.Pareto8Full[0, 1]);
        Equal((byte)128, Vp9CoefProbs.Pareto8Full[0, 2]);
        Equal((byte)6,   Vp9CoefProbs.Pareto8Full[0, 3]);
        Equal((byte)86,  Vp9CoefProbs.Pareto8Full[0, 4]);
        Equal((byte)23,  Vp9CoefProbs.Pareto8Full[0, 5]);
        Equal((byte)88,  Vp9CoefProbs.Pareto8Full[0, 6]);
        Equal((byte)29,  Vp9CoefProbs.Pareto8Full[0, 7]);

        // Last row (row 254): { 255, 246, 247, 255, 239, 255, 253, 255 }
        Equal((byte)255, Vp9CoefProbs.Pareto8Full[254, 0]);
        Equal((byte)246, Vp9CoefProbs.Pareto8Full[254, 1]);
        Equal((byte)247, Vp9CoefProbs.Pareto8Full[254, 2]);
        Equal((byte)255, Vp9CoefProbs.Pareto8Full[254, 3]);
        Equal((byte)239, Vp9CoefProbs.Pareto8Full[254, 4]);
        Equal((byte)255, Vp9CoefProbs.Pareto8Full[254, 5]);
        Equal((byte)253, Vp9CoefProbs.Pareto8Full[254, 6]);
        Equal((byte)255, Vp9CoefProbs.Pareto8Full[254, 7]);
    }

    [TestMethod]
    public void Vp9CoefProbs_Pareto8Full_AllValuesInValidRange()
    {
        // VP9 probabilities are uint8 values in [1, 255] - 0 is reserved
        // for "the probability tree is dead" and never appears in a
        // legitimate prob table (libvpx clamps to MIN_PROB = 1 at every
        // update site). Verify the entire 2040-entry table respects this.
        for (int row = 0; row < 255; row++)
        for (int col = 0; col < 8; col++)
        {
            byte v = Vp9CoefProbs.Pareto8Full[row, col];
            True(v >= 1, $"Pareto8Full[{row},{col}] = {v} below MIN_PROB");
        }
    }
}
