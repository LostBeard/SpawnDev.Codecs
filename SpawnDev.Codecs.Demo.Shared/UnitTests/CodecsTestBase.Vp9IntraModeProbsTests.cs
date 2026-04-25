// Tests for Vp9IntraModeProbs (slice 156). Length, pinned values,
// dispatcher slice correctness.

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9IntraModeProbs_AllArraysHaveExpectedLengths()
    {
        Equal(90, Vp9IntraModeProbs.KfUvModeProbs.Length);   // 10 x 9
        Equal(36, Vp9IntraModeProbs.DefaultIfYProbs.Length); // 4 x 9
        Equal(90, Vp9IntraModeProbs.DefaultIfUvProbs.Length); // 10 x 9
        Equal(9, Vp9IntraModeProbs.ProbsPerMode);
        Equal(4, Vp9IntraModeProbs.BlockSizeGroups);
    }

    [TestMethod]
    public void Vp9IntraModeProbs_KfUvModeProbs_FirstAndLastRowsMatchLibvpx()
    {
        // y = DcPred row 0: { 144, 11, 54, 157, 195, 130, 46, 58, 108 }
        Equal((byte)144, Vp9IntraModeProbs.KfUvModeProbs[0]);
        Equal((byte)11,  Vp9IntraModeProbs.KfUvModeProbs[1]);
        Equal((byte)108, Vp9IntraModeProbs.KfUvModeProbs[8]);

        // y = TmPred row 9: { 102, 19, 66, 162, 182, 122, 35, 59, 128 }
        Equal((byte)102, Vp9IntraModeProbs.KfUvModeProbs[81]);
        Equal((byte)19,  Vp9IntraModeProbs.KfUvModeProbs[82]);
        Equal((byte)128, Vp9IntraModeProbs.KfUvModeProbs[89]);
    }

    [TestMethod]
    public void Vp9IntraModeProbs_DefaultIfYProbs_FirstAndLastRowsMatchLibvpx()
    {
        // block_size < 8x8: { 65, 32, 18, 144, 162, 194, 41, 51, 98 }
        Equal((byte)65, Vp9IntraModeProbs.DefaultIfYProbs[0]);
        Equal((byte)32, Vp9IntraModeProbs.DefaultIfYProbs[1]);
        Equal((byte)98, Vp9IntraModeProbs.DefaultIfYProbs[8]);

        // block_size >= 32x32: { 221, 135, 38, 194, 248, 121, 96, 85, 29 }
        Equal((byte)221, Vp9IntraModeProbs.DefaultIfYProbs[27]);
        Equal((byte)135, Vp9IntraModeProbs.DefaultIfYProbs[28]);
        Equal((byte)29,  Vp9IntraModeProbs.DefaultIfYProbs[35]);
    }

    [TestMethod]
    public void Vp9IntraModeProbs_DefaultIfUvProbs_FirstAndLastRowsMatchLibvpx()
    {
        // y = DcPred row 0: { 120, 7, 76, 176, 208, 126, 28, 54, 103 }
        Equal((byte)120, Vp9IntraModeProbs.DefaultIfUvProbs[0]);
        Equal((byte)7,   Vp9IntraModeProbs.DefaultIfUvProbs[1]);
        Equal((byte)103, Vp9IntraModeProbs.DefaultIfUvProbs[8]);

        // y = TmPred row 9: { 101, 21, 107, 181, 192, 103, 19, 67, 125 }
        Equal((byte)101, Vp9IntraModeProbs.DefaultIfUvProbs[81]);
        Equal((byte)21,  Vp9IntraModeProbs.DefaultIfUvProbs[82]);
        Equal((byte)125, Vp9IntraModeProbs.DefaultIfUvProbs[89]);
    }

    [TestMethod]
    public void Vp9IntraModeProbs_KeyframeUvProbsHelper_ReturnsCorrectSliceForEveryYMode()
    {
        // Each Y mode (0..9) yields a 9-byte slice. Verify by spot-
        // checking the first byte against the array layout.
        for (int y = 0; y < 10; y++)
        {
            var slice = Vp9IntraModeProbs.KeyframeUvProbs((Vp9IntraMode)y);
            Equal(9, slice.Length);
            Equal(Vp9IntraModeProbs.KfUvModeProbs[y * 9], slice[0]);
            Equal(Vp9IntraModeProbs.KfUvModeProbs[y * 9 + 8], slice[8]);
        }
    }

    [TestMethod]
    public void Vp9IntraModeProbs_InterFrameYProbsHelper_ReturnsCorrectSliceForEveryGroup()
    {
        for (int g = 0; g < 4; g++)
        {
            var slice = Vp9IntraModeProbs.InterFrameYProbs(g);
            Equal(9, slice.Length);
            Equal(Vp9IntraModeProbs.DefaultIfYProbs[g * 9], slice[0]);
            Equal(Vp9IntraModeProbs.DefaultIfYProbs[g * 9 + 8], slice[8]);
        }
    }

    [TestMethod]
    public void Vp9IntraModeProbs_InterFrameUvProbsHelper_ReturnsCorrectSliceForEveryYMode()
    {
        for (int y = 0; y < 10; y++)
        {
            var slice = Vp9IntraModeProbs.InterFrameUvProbs((Vp9IntraMode)y);
            Equal(9, slice.Length);
            Equal(Vp9IntraModeProbs.DefaultIfUvProbs[y * 9], slice[0]);
            Equal(Vp9IntraModeProbs.DefaultIfUvProbs[y * 9 + 8], slice[8]);
        }
    }

    [TestMethod]
    public void Vp9IntraModeProbs_Helpers_RejectOutOfRangeArgs()
    {
        Throws<ArgumentOutOfRangeException>(() =>
            Vp9IntraModeProbs.KeyframeUvProbs((Vp9IntraMode)10));
        Throws<ArgumentOutOfRangeException>(() =>
            Vp9IntraModeProbs.InterFrameUvProbs((Vp9IntraMode)10));
        Throws<ArgumentOutOfRangeException>(() =>
            Vp9IntraModeProbs.InterFrameYProbs(4));
        Throws<ArgumentOutOfRangeException>(() =>
            Vp9IntraModeProbs.InterFrameYProbs(-1));
    }

    [TestMethod]
    public void Vp9IntraModeProbs_DriveProbsIntoIntraModeTree_DecodesDcPredOnFirstZeroBit()
    {
        // End-to-end: pull a real probability slice from this slice's
        // tables and decode through slice 153's Vp9IntraModeTree.Decode.
        // First bit = 0 -> DcPred regardless of which probs slice we picked.
        var probs = Vp9IntraModeProbs.KeyframeUvProbs(Vp9IntraMode.DcPred);
        var bits = new int[] { 0 };
        int idx = 0;
        var m = Vp9IntraModeTree.Decode(_ => bits[idx++], probs);
        Equal(Vp9IntraMode.DcPred, m);
    }
}
