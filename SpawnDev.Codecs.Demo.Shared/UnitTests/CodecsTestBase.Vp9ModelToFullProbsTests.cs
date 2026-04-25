// Tests for Vp9CoefProbs.ModelToFullProbs (slice 144). Verifies the
// pareto8 expansion produces the libvpx-spec full 11-entry probability
// vector from the 3-entry stored model.

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9ModelToFullProbs_ConstantsMatchLibvpx()
    {
        Equal(3,  Vp9CoefProbs.UnconstrainedNodes);
        Equal(8,  Vp9CoefProbs.ModelNodes);
        Equal(11, Vp9CoefProbs.EntropyNodes);
        Equal(2,  Vp9CoefProbs.PivotNode);
    }

    [TestMethod]
    public void Vp9ModelToFullProbs_CopiesModelThenAppendsPareto8Row()
    {
        // For pivot = 1, pareto8_full[0] = { 3, 86, 128, 6, 86, 23, 88, 29 }.
        var model = new byte[] { 100, 50, 1 };
        var full = new byte[11];
        Vp9CoefProbs.ModelToFullProbs(model, full);

        // First 3 entries copied from the model.
        Equal((byte)100, full[0]);
        Equal((byte)50,  full[1]);
        Equal((byte)1,   full[2]);

        // Next 8 entries come from pareto8_full[pivot - 1 = 0].
        Equal((byte)3,   full[3]);
        Equal((byte)86,  full[4]);
        Equal((byte)128, full[5]);
        Equal((byte)6,   full[6]);
        Equal((byte)86,  full[7]);
        Equal((byte)23,  full[8]);
        Equal((byte)88,  full[9]);
        Equal((byte)29,  full[10]);
    }

    [TestMethod]
    public void Vp9ModelToFullProbs_PivotEqualsTwoIndexesSecondPareto8Row()
    {
        // pareto8_full[1] = { 6, 86, 128, 11, 87, 42, 91, 52 }.
        var model = new byte[] { 200, 150, 2 };
        var full = new byte[11];
        Vp9CoefProbs.ModelToFullProbs(model, full);
        Equal((byte)200, full[0]);
        Equal((byte)6,   full[3]);
        Equal((byte)86,  full[4]);
        Equal((byte)128, full[5]);
        Equal((byte)52,  full[10]);
    }

    [TestMethod]
    public void Vp9ModelToFullProbs_PivotEquals255IndexesLastPareto8Row()
    {
        // pareto8_full[254] = { 255, 246, 247, 255, 239, 255, 253, 255 }.
        var model = new byte[] { 1, 1, 255 };
        var full = new byte[11];
        Vp9CoefProbs.ModelToFullProbs(model, full);
        Equal((byte)255, full[3]);
        Equal((byte)246, full[4]);
        Equal((byte)247, full[5]);
        Equal((byte)255, full[6]);
        Equal((byte)239, full[7]);
        Equal((byte)255, full[8]);
        Equal((byte)253, full[9]);
        Equal((byte)255, full[10]);
    }

    [TestMethod]
    public void Vp9ModelToFullProbs_PivotZero_ThrowsInvalidData()
    {
        // Pivot probability of 0 would index Pareto8Full at row -1 -
        // libvpx asserts the same condition. A correct VP9 bitstream
        // never produces a zero pivot for a coefficient that's actually
        // being decoded, so this is a hard error rather than a silent
        // out-of-bounds.
        var model = new byte[] { 100, 50, 0 };
        var full = new byte[11];
        Throws<InvalidDataException>(() => Vp9CoefProbs.ModelToFullProbs(model, full));
    }

    [TestMethod]
    public void Vp9ModelToFullProbs_RejectsTooSmallBuffers()
    {
        Throws<ArgumentException>(() => Vp9CoefProbs.ModelToFullProbs(new byte[2], new byte[11]));
        Throws<ArgumentException>(() => Vp9CoefProbs.ModelToFullProbs(new byte[3], new byte[10]));
    }

    [TestMethod]
    public void Vp9ModelToFullProbs_DrivenByDefaultCoefProbs4x4_ProducesValidFullVector()
    {
        // End-to-end: pull a stored 3-entry model out of slice 142's
        // 4x4 default prob table, expand it, and verify the result has
        // a non-zero entry at every position. This sanity-checks that
        // the stored prob -> pareto8 expansion path is wired correctly
        // for the actual production data.
        int idx = Vp9CoefProbs.Index4x4(plane: 0, refType: 0, band: 0, ctx: 0, node: 0);
        var model = new byte[]
        {
            Vp9CoefProbs.DefaultCoefProbs4x4[idx + 0],
            Vp9CoefProbs.DefaultCoefProbs4x4[idx + 1],
            Vp9CoefProbs.DefaultCoefProbs4x4[idx + 2],
        };
        // Y/Intra/Band 0 ctx 0 model = { 195, 29, 183 } per slice 142.
        Equal((byte)195, model[0]);
        Equal((byte)29,  model[1]);
        Equal((byte)183, model[2]);

        var full = new byte[11];
        Vp9CoefProbs.ModelToFullProbs(model, full);

        for (int i = 0; i < 11; i++)
            True(full[i] >= 1, $"full[{i}] = {full[i]} below MIN_PROB");
    }
}
