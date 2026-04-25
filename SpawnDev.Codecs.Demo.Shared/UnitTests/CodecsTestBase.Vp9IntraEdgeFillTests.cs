// Tests for Vp9IntraEdgeFill (slice 172). Verifies the libvpx 127 /
// 129 boundary-fill convention plus right-edge replication.

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9IntraEdgeFill_Constants_MatchLibvpx()
    {
        Equal((byte)127, Vp9IntraEdgeFill.AboveFill);
        Equal((byte)129, Vp9IntraEdgeFill.LeftFill);
        Equal((byte)127, Vp9IntraEdgeFill.CornerFillNoAbove);
        Equal((byte)129, Vp9IntraEdgeFill.CornerFillAboveOnly);
    }

    [TestMethod]
    public void Vp9IntraEdgeFill_FillMissingAbove_FillsNSamplesWith127()
    {
        var above = new byte[8];  // pad past 4
        for (int i = 0; i < 8; i++) above[i] = 99;  // sentinel
        Vp9IntraEdgeFill.FillMissingAbove(above, n: 4);

        for (int i = 0; i < 4; i++) Equal((byte)127, above[i]);
        // Past-N slots untouched.
        for (int i = 4; i < 8; i++) Equal((byte)99, above[i]);
    }

    [TestMethod]
    public void Vp9IntraEdgeFill_FillMissingAbove_With2N_FillsExtension()
    {
        var above = new byte[16];
        for (int i = 0; i < 16; i++) above[i] = 99;
        Vp9IntraEdgeFill.FillMissingAbove(above, n: 8, needAboveRight: true);

        for (int i = 0; i < 16; i++) Equal((byte)127, above[i]);
    }

    [TestMethod]
    public void Vp9IntraEdgeFill_FillMissingLeft_FillsNSamplesWith129()
    {
        var left = new byte[6];
        for (int i = 0; i < 6; i++) left[i] = 99;
        Vp9IntraEdgeFill.FillMissingLeft(left, n: 4);

        for (int i = 0; i < 4; i++) Equal((byte)129, left[i]);
        for (int i = 4; i < 6; i++) Equal((byte)99, left[i]);
    }

    [TestMethod]
    public void Vp9IntraEdgeFill_ResolveCorner_CoversAllBranches()
    {
        // No above -> 127 regardless of left or refValue.
        Equal((byte)127, Vp9IntraEdgeFill.ResolveCorner(hasAbove: false, hasLeft: false, refValue: 200));
        Equal((byte)127, Vp9IntraEdgeFill.ResolveCorner(hasAbove: false, hasLeft: true, refValue: 200));

        // Above but no left -> 129.
        Equal((byte)129, Vp9IntraEdgeFill.ResolveCorner(hasAbove: true, hasLeft: false, refValue: 200));

        // Both available -> caller's ref value passes through.
        Equal((byte)200, Vp9IntraEdgeFill.ResolveCorner(hasAbove: true, hasLeft: true, refValue: 200));
        Equal((byte)5, Vp9IntraEdgeFill.ResolveCorner(hasAbove: true, hasLeft: true, refValue: 5));
    }

    [TestMethod]
    public void Vp9IntraEdgeFill_ReplicateAboveRight_FillsExtensionWithLastInBlockSample()
    {
        var above = new byte[8];
        // First 4 samples are real; last 4 should become above[3] = 50 after replication.
        above[0] = 10; above[1] = 20; above[2] = 30; above[3] = 50;
        for (int i = 4; i < 8; i++) above[i] = 99;

        Vp9IntraEdgeFill.ReplicateAboveRight(above, n: 4);

        // First N samples untouched.
        Equal((byte)10, above[0]); Equal((byte)20, above[1]);
        Equal((byte)30, above[2]); Equal((byte)50, above[3]);
        // Extension replicated.
        for (int i = 4; i < 8; i++) Equal((byte)50, above[i]);
    }

    [TestMethod]
    public void Vp9IntraEdgeFill_ReplicateAboveRight_AllSizes()
    {
        // Sanity: 8, 16, 32 all run without throwing and replicate above[N-1].
        foreach (int n in new[] { 8, 16, 32 })
        {
            var above = new byte[2 * n];
            above[n - 1] = 77;
            Vp9IntraEdgeFill.ReplicateAboveRight(above, n);
            for (int i = n; i < 2 * n; i++) Equal((byte)77, above[i]);
        }
    }

    [TestMethod]
    public void Vp9IntraEdgeFill_FullPipeline_EdgesMissing_ProducesDcCorner()
    {
        // Simulate "no neighbors" case at the top-left of the frame.
        // The top-left block has no above, no left, and no corner.
        // After fill: above = 127s, left = 129s, corner = 127.
        var above = new byte[8];  // 2N for D45/D63 if needed
        var left = new byte[4];

        Vp9IntraEdgeFill.FillMissingAbove(above, n: 4, needAboveRight: true);
        Vp9IntraEdgeFill.FillMissingLeft(left, n: 4);
        byte corner = Vp9IntraEdgeFill.ResolveCorner(hasAbove: false, hasLeft: false, refValue: 0);

        for (int i = 0; i < 8; i++) Equal((byte)127, above[i]);
        for (int i = 0; i < 4; i++) Equal((byte)129, left[i]);
        Equal((byte)127, corner);
    }

    [TestMethod]
    public void Vp9IntraEdgeFill_RejectsInvalidArgs()
    {
        Throws<ArgumentOutOfRangeException>(() =>
            Vp9IntraEdgeFill.FillMissingAbove(new byte[5], n: 5));
        Throws<ArgumentOutOfRangeException>(() =>
            Vp9IntraEdgeFill.FillMissingLeft(new byte[5], n: 5));
        Throws<ArgumentOutOfRangeException>(() =>
            Vp9IntraEdgeFill.ReplicateAboveRight(new byte[10], n: 5));

        // Buffer too small.
        Throws<ArgumentException>(() =>
            Vp9IntraEdgeFill.FillMissingAbove(new byte[3], n: 4));
        Throws<ArgumentException>(() =>
            Vp9IntraEdgeFill.FillMissingAbove(new byte[7], n: 4, needAboveRight: true));
        Throws<ArgumentException>(() =>
            Vp9IntraEdgeFill.FillMissingLeft(new byte[3], n: 4));
        Throws<ArgumentException>(() =>
            Vp9IntraEdgeFill.ReplicateAboveRight(new byte[7], n: 4));
    }
}
