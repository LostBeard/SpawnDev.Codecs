// Tests for Vp9NeighborContexts (slice 230).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9NeighborContexts_Skip_BothMissing()
    {
        Equal(0, Vp9NeighborContexts.GetSkipContext(null, null));
    }

    [TestMethod]
    public void Vp9NeighborContexts_Skip_OnlyOneSide_Skipped()
    {
        Equal(1, Vp9NeighborContexts.GetSkipContext(true, null));
        Equal(1, Vp9NeighborContexts.GetSkipContext(null, true));
    }

    [TestMethod]
    public void Vp9NeighborContexts_Skip_BothPresent_NeitherSkipped()
    {
        Equal(0, Vp9NeighborContexts.GetSkipContext(false, false));
    }

    [TestMethod]
    public void Vp9NeighborContexts_Skip_BothPresent_OneSkipped()
    {
        Equal(1, Vp9NeighborContexts.GetSkipContext(true, false));
        Equal(1, Vp9NeighborContexts.GetSkipContext(false, true));
    }

    [TestMethod]
    public void Vp9NeighborContexts_Skip_BothPresent_BothSkipped()
    {
        Equal(2, Vp9NeighborContexts.GetSkipContext(true, true));
    }

    [TestMethod]
    public void Vp9NeighborContexts_IntraInter_BothMissing()
    {
        Equal(0, Vp9NeighborContexts.GetIntraInterContext(null, null));
    }

    [TestMethod]
    public void Vp9NeighborContexts_IntraInter_OneEdge_Inter()
    {
        Equal(0, Vp9NeighborContexts.GetIntraInterContext(false, null));
        Equal(0, Vp9NeighborContexts.GetIntraInterContext(null, false));
    }

    [TestMethod]
    public void Vp9NeighborContexts_IntraInter_OneEdge_Intra()
    {
        Equal(2, Vp9NeighborContexts.GetIntraInterContext(true, null));
        Equal(2, Vp9NeighborContexts.GetIntraInterContext(null, true));
    }

    [TestMethod]
    public void Vp9NeighborContexts_IntraInter_BothInter()
    {
        Equal(0, Vp9NeighborContexts.GetIntraInterContext(false, false));
    }

    [TestMethod]
    public void Vp9NeighborContexts_IntraInter_OneIntraOneInter()
    {
        Equal(1, Vp9NeighborContexts.GetIntraInterContext(true, false));
        Equal(1, Vp9NeighborContexts.GetIntraInterContext(false, true));
    }

    [TestMethod]
    public void Vp9NeighborContexts_IntraInter_BothIntra()
    {
        Equal(3, Vp9NeighborContexts.GetIntraInterContext(true, true));
    }

    [TestMethod]
    public void Vp9NeighborContexts_TxSize_BothMissing_ReturnsZero()
    {
        // No neighbors -> both ctx default to max -> sum = 2*max,
        // > max -> context 1. (libvpx behavior at frame top-left.)
        Equal(1, Vp9NeighborContexts.GetTxSizeContext(
            Vp9BlockSize.Block16x16, null, null));
    }

    [TestMethod]
    public void Vp9NeighborContexts_TxSize_BothSkippedNeighbors_ReturnsContext1()
    {
        // Skipped neighbors contribute max_tx_size each; sum = 2*max,
        // > max -> context 1.
        Equal(1, Vp9NeighborContexts.GetTxSizeContext(
            Vp9BlockSize.Block16x16,
            (Vp9TxSize.Tx4x4, true),
            (Vp9TxSize.Tx4x4, true)));
    }

    [TestMethod]
    public void Vp9NeighborContexts_TxSize_BothSmall_ReturnsZero()
    {
        // Block16x16 max_tx = 16 (=2). Both neighbors at 4x4 (=0).
        // sum = 0, < 2 -> context 0.
        Equal(0, Vp9NeighborContexts.GetTxSizeContext(
            Vp9BlockSize.Block16x16,
            (Vp9TxSize.Tx4x4, false),
            (Vp9TxSize.Tx4x4, false)));
    }

    [TestMethod]
    public void Vp9NeighborContexts_TxSize_NeighborSumEqualsMax_ReturnsZero()
    {
        // Block16x16 max_tx = 2. Neighbors both at 8x8 (=1) -> sum = 2.
        // 2 > 2 is false -> context 0.
        Equal(0, Vp9NeighborContexts.GetTxSizeContext(
            Vp9BlockSize.Block16x16,
            (Vp9TxSize.Tx8x8, false),
            (Vp9TxSize.Tx8x8, false)));
    }

    [TestMethod]
    public void Vp9NeighborContexts_TxSize_NeighborSumExceedsMax_ReturnsOne()
    {
        // Block16x16 max_tx = 2. Above 16x16 (=2) + left 8x8 (=1) = 3 > 2.
        Equal(1, Vp9NeighborContexts.GetTxSizeContext(
            Vp9BlockSize.Block16x16,
            (Vp9TxSize.Tx16x16, false),
            (Vp9TxSize.Tx8x8, false)));
    }

    [TestMethod]
    public void Vp9NeighborContexts_TxSize_OnlyOneNeighbor_InheritsToOther()
    {
        // Above present at 8x8 (=1, not skipped); no left.
        // Per libvpx: !has_left -> left_ctx = above_ctx = 1.
        // sum = 2, max = 2 -> 2 > 2 false -> 0.
        Equal(0, Vp9NeighborContexts.GetTxSizeContext(
            Vp9BlockSize.Block16x16,
            (Vp9TxSize.Tx8x8, false),
            null));
    }

    [TestMethod]
    public void Vp9NeighborContexts_TxSize_OnlyOneNeighborAt16x16_ReturnsOne()
    {
        // Left present at 16x16 (=2, not skipped); no above.
        // !has_above -> above_ctx = left_ctx = 2.
        // sum = 4, > 2 -> context 1.
        Equal(1, Vp9NeighborContexts.GetTxSizeContext(
            Vp9BlockSize.Block16x16,
            null,
            (Vp9TxSize.Tx16x16, false)));
    }
}
