// Tests for Vp8DefaultCoefProbs - sample-based bit-exact verification
// against libvpx default_coef_probs values. Catches transcription
// errors in the 4D probability table.

using SpawnDev.Codecs.Video.Vp8;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp8DefaultCoefProbs_Dimensions_MatchLibvpx()
    {
        Equal(4, Vp8DefaultCoefProbs.BlockTypes);
        Equal(8, Vp8DefaultCoefProbs.CoefBands);
        Equal(3, Vp8DefaultCoefProbs.PrevCoefContexts);
        Equal(11, Vp8DefaultCoefProbs.EntropyNodes);
        Equal(4, Vp8DefaultCoefProbs.DefaultProbs.GetLength(0));
        Equal(8, Vp8DefaultCoefProbs.DefaultProbs.GetLength(1));
        Equal(3, Vp8DefaultCoefProbs.DefaultProbs.GetLength(2));
        Equal(11, Vp8DefaultCoefProbs.DefaultProbs.GetLength(3));
    }

    [TestMethod]
    public void Vp8DefaultCoefProbs_Type0Band0_AllNeutral128()
    {
        // Block Type 0 (Y after Y2), Band 0 (DC) - all probs initialize to 128.
        // libvpx default_coef_probs[0][0][*][*] = 128 for all 33 entries.
        for (int ctx = 0; ctx < 3; ctx++)
            for (int node = 0; node < 11; node++)
                Equal((byte)128, Vp8DefaultCoefProbs.DefaultProbs[0, 0, ctx, node]);
    }

    [TestMethod]
    public void Vp8DefaultCoefProbs_Type0Band1_Context0_MatchesLibvpx()
    {
        // libvpx: { 253, 136, 254, 255, 228, 219, 128, 128, 128, 128, 128 }
        byte[] expected = { 253, 136, 254, 255, 228, 219, 128, 128, 128, 128, 128 };
        for (int n = 0; n < 11; n++)
            Equal(expected[n], Vp8DefaultCoefProbs.DefaultProbs[0, 1, 0, n]);
    }

    [TestMethod]
    public void Vp8DefaultCoefProbs_Type1Band0_Context2_MatchesLibvpx()
    {
        // libvpx: { 68, 47, 146, 208, 149, 167, 221, 162, 255, 223, 128 }
        byte[] expected = { 68, 47, 146, 208, 149, 167, 221, 162, 255, 223, 128 };
        for (int n = 0; n < 11; n++)
            Equal(expected[n], Vp8DefaultCoefProbs.DefaultProbs[1, 0, 2, n]);
    }

    [TestMethod]
    public void Vp8DefaultCoefProbs_Type3Band0_Context0_MatchesLibvpx()
    {
        // Y2 (Block Type 3), Band 0, Context 0: highest energy class normally,
        // libvpx: { 202, 24, 213, 235, 186, 191, 220, 160, 240, 175, 255 }
        byte[] expected = { 202, 24, 213, 235, 186, 191, 220, 160, 240, 175, 255 };
        for (int n = 0; n < 11; n++)
            Equal(expected[n], Vp8DefaultCoefProbs.DefaultProbs[3, 0, 0, n]);
    }

    [TestMethod]
    public void Vp8DefaultCoefProbs_Type2Band7_AllNeutral128()
    {
        // UV (Block Type 2), Band 7: all probs 128 in libvpx.
        for (int ctx = 0; ctx < 3; ctx++)
            for (int node = 0; node < 11; node++)
                Equal((byte)128, Vp8DefaultCoefProbs.DefaultProbs[2, 7, ctx, node]);
    }
}
