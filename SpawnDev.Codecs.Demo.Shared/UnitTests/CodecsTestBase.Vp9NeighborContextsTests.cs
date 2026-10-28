// Tests for Vp9NeighborContexts (slice 230).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod(Category = "CpuOnly")]
    public void Vp9NeighborContexts_Skip_BothMissing()
    {
        CpuOnlyTest();
        Equal(0, Vp9NeighborContexts.GetSkipContext(null, null));
    }

    [TestMethod(Category = "CpuOnly")]
    public void Vp9NeighborContexts_Skip_OnlyOneSide_Skipped()
    {
        CpuOnlyTest();
        Equal(1, Vp9NeighborContexts.GetSkipContext(true, null));
        Equal(1, Vp9NeighborContexts.GetSkipContext(null, true));
    }

    [TestMethod(Category = "CpuOnly")]
    public void Vp9NeighborContexts_Skip_BothPresent_NeitherSkipped()
    {
        CpuOnlyTest();
        Equal(0, Vp9NeighborContexts.GetSkipContext(false, false));
    }

    [TestMethod(Category = "CpuOnly")]
    public void Vp9NeighborContexts_Skip_BothPresent_OneSkipped()
    {
        CpuOnlyTest();
        Equal(1, Vp9NeighborContexts.GetSkipContext(true, false));
        Equal(1, Vp9NeighborContexts.GetSkipContext(false, true));
    }

    [TestMethod(Category = "CpuOnly")]
    public void Vp9NeighborContexts_Skip_BothPresent_BothSkipped()
    {
        CpuOnlyTest();
        Equal(2, Vp9NeighborContexts.GetSkipContext(true, true));
    }

    [TestMethod(Category = "CpuOnly")]
    public void Vp9NeighborContexts_IntraInter_BothMissing()
    {
        CpuOnlyTest();
        Equal(0, Vp9NeighborContexts.GetIntraInterContext(null, null));
    }

    [TestMethod(Category = "CpuOnly")]
    public void Vp9NeighborContexts_IntraInter_OneEdge_Inter()
    {
        CpuOnlyTest();
        Equal(0, Vp9NeighborContexts.GetIntraInterContext(false, null));
        Equal(0, Vp9NeighborContexts.GetIntraInterContext(null, false));
    }

    [TestMethod(Category = "CpuOnly")]
    public void Vp9NeighborContexts_IntraInter_OneEdge_Intra()
    {
        CpuOnlyTest();
        Equal(2, Vp9NeighborContexts.GetIntraInterContext(true, null));
        Equal(2, Vp9NeighborContexts.GetIntraInterContext(null, true));
    }

    [TestMethod(Category = "CpuOnly")]
    public void Vp9NeighborContexts_IntraInter_BothInter()
    {
        CpuOnlyTest();
        Equal(0, Vp9NeighborContexts.GetIntraInterContext(false, false));
    }

    [TestMethod(Category = "CpuOnly")]
    public void Vp9NeighborContexts_IntraInter_OneIntraOneInter()
    {
        CpuOnlyTest();
        Equal(1, Vp9NeighborContexts.GetIntraInterContext(true, false));
        Equal(1, Vp9NeighborContexts.GetIntraInterContext(false, true));
    }

    [TestMethod(Category = "CpuOnly")]
    public void Vp9NeighborContexts_IntraInter_BothIntra()
    {
        CpuOnlyTest();
        Equal(3, Vp9NeighborContexts.GetIntraInterContext(true, true));
    }

    [TestMethod(Category = "CpuOnly")]
    public void Vp9NeighborContexts_TxSize_BothMissing_ReturnsZero()
    {
        CpuOnlyTest();
        // No neighbors -> both ctx default to max -> sum = 2*max,
        // > max -> context 1. (libvpx behavior at frame top-left.)
        Equal(1, Vp9NeighborContexts.GetTxSizeContext(
            Vp9BlockSize.Block16x16, null, null));
    }

    [TestMethod(Category = "CpuOnly")]
    public void Vp9NeighborContexts_TxSize_BothSkippedNeighbors_ReturnsContext1()
    {
        CpuOnlyTest();
        // Skipped neighbors contribute max_tx_size each; sum = 2*max,
        // > max -> context 1.
        Equal(1, Vp9NeighborContexts.GetTxSizeContext(
            Vp9BlockSize.Block16x16,
            (Vp9TxSize.Tx4x4, true),
            (Vp9TxSize.Tx4x4, true)));
    }

    [TestMethod(Category = "CpuOnly")]
    public void Vp9NeighborContexts_TxSize_BothSmall_ReturnsZero()
    {
        CpuOnlyTest();
        // Block16x16 max_tx = 16 (=2). Both neighbors at 4x4 (=0).
        // sum = 0, < 2 -> context 0.
        Equal(0, Vp9NeighborContexts.GetTxSizeContext(
            Vp9BlockSize.Block16x16,
            (Vp9TxSize.Tx4x4, false),
            (Vp9TxSize.Tx4x4, false)));
    }

    [TestMethod(Category = "CpuOnly")]
    public void Vp9NeighborContexts_TxSize_NeighborSumEqualsMax_ReturnsZero()
    {
        CpuOnlyTest();
        // Block16x16 max_tx = 2. Neighbors both at 8x8 (=1) -> sum = 2.
        // 2 > 2 is false -> context 0.
        Equal(0, Vp9NeighborContexts.GetTxSizeContext(
            Vp9BlockSize.Block16x16,
            (Vp9TxSize.Tx8x8, false),
            (Vp9TxSize.Tx8x8, false)));
    }

    [TestMethod(Category = "CpuOnly")]
    public void Vp9NeighborContexts_TxSize_NeighborSumExceedsMax_ReturnsOne()
    {
        CpuOnlyTest();
        // Block16x16 max_tx = 2. Above 16x16 (=2) + left 8x8 (=1) = 3 > 2.
        Equal(1, Vp9NeighborContexts.GetTxSizeContext(
            Vp9BlockSize.Block16x16,
            (Vp9TxSize.Tx16x16, false),
            (Vp9TxSize.Tx8x8, false)));
    }

    [TestMethod(Category = "CpuOnly")]
    public void Vp9NeighborContexts_TxSize_OnlyOneNeighbor_InheritsToOther()
    {
        CpuOnlyTest();
        // Above present at 8x8 (=1, not skipped); no left.
        // Per libvpx: !has_left -> left_ctx = above_ctx = 1.
        // sum = 2, max = 2 -> 2 > 2 false -> 0.
        Equal(0, Vp9NeighborContexts.GetTxSizeContext(
            Vp9BlockSize.Block16x16,
            (Vp9TxSize.Tx8x8, false),
            null));
    }

    [TestMethod(Category = "CpuOnly")]
    public void Vp9NeighborContexts_TxSize_OnlyOneNeighborAt16x16_ReturnsOne()
    {
        CpuOnlyTest();
        // Left present at 16x16 (=2, not skipped); no above.
        // !has_above -> above_ctx = left_ctx = 2.
        // sum = 4, > 2 -> context 1.
        Equal(1, Vp9NeighborContexts.GetTxSizeContext(
            Vp9BlockSize.Block16x16,
            null,
            (Vp9TxSize.Tx16x16, false)));
    }

    [TestMethod(Category = "CpuOnly")]
    public void Vp9NeighborContexts_PartitionPlane_Block8x8_BothZero()
    {
        CpuOnlyTest();
        // bsl = 0 (8x8). Both ctx bytes 0 -> bit 0 = 0. Context = 0.
        Equal(0, Vp9NeighborContexts.GetPartitionPlaneContext(
            0, 0, Vp9BlockSize.Block8x8));
    }

    [TestMethod(Category = "CpuOnly")]
    public void Vp9NeighborContexts_PartitionPlane_Block8x8_AboveSplit()
    {
        CpuOnlyTest();
        // bsl = 0. above_ctx bit 0 = 1, left = 0. -> 0*2 + 1 = 1, +0*4 = 1.
        Equal(1, Vp9NeighborContexts.GetPartitionPlaneContext(
            0b00000001, 0, Vp9BlockSize.Block8x8));
    }

    [TestMethod(Category = "CpuOnly")]
    public void Vp9NeighborContexts_PartitionPlane_Block8x8_LeftSplit()
    {
        CpuOnlyTest();
        // bsl = 0. above = 0, left bit 0 = 1. -> 1*2 + 0 = 2.
        Equal(2, Vp9NeighborContexts.GetPartitionPlaneContext(
            0, 0b00000001, Vp9BlockSize.Block8x8));
    }

    [TestMethod(Category = "CpuOnly")]
    public void Vp9NeighborContexts_PartitionPlane_Block8x8_BothSplit()
    {
        CpuOnlyTest();
        // bsl = 0. both = 1 -> 1*2 + 1 = 3.
        Equal(3, Vp9NeighborContexts.GetPartitionPlaneContext(
            0b00000001, 0b00000001, Vp9BlockSize.Block8x8));
    }

    [TestMethod(Category = "CpuOnly")]
    public void Vp9NeighborContexts_PartitionPlane_Block16x16_GroupOffset()
    {
        CpuOnlyTest();
        // bsl = 1 (16x16). above bit 1 of 0b10 = 1, left = 0.
        // (0*2 + 1) + 1*4 = 5 (bsl=1 group, "above split, left not split").
        Equal(5, Vp9NeighborContexts.GetPartitionPlaneContext(
            0b00000010, 0, Vp9BlockSize.Block16x16));
    }

    [TestMethod(Category = "CpuOnly")]
    public void Vp9NeighborContexts_PartitionPlane_Block32x32_BaseAt8()
    {
        CpuOnlyTest();
        // bsl = 2 (32x32). Both ctx bytes 0 -> 0 + 2*4 = 8.
        Equal(8, Vp9NeighborContexts.GetPartitionPlaneContext(
            0, 0, Vp9BlockSize.Block32x32));
    }

    [TestMethod(Category = "CpuOnly")]
    public void Vp9NeighborContexts_PartitionPlane_Block64x64_BaseAt12()
    {
        CpuOnlyTest();
        // bsl = 3 (64x64). Both ctx bytes 0 -> 0 + 3*4 = 12.
        Equal(12, Vp9NeighborContexts.GetPartitionPlaneContext(
            0, 0, Vp9BlockSize.Block64x64));
    }

    [TestMethod(Category = "CpuOnly")]
    public void Vp9NeighborContexts_PartitionPlane_Block64x64_BothSplitMaxContext()
    {
        CpuOnlyTest();
        // bsl = 3. above bit 3 = 1 (0b1000), left bit 3 = 1.
        // (1*2 + 1) + 3*4 = 3 + 12 = 15 (the highest valid partition context).
        Equal(15, Vp9NeighborContexts.GetPartitionPlaneContext(
            0b00001000, 0b00001000, Vp9BlockSize.Block64x64));
    }

    [TestMethod(Category = "CpuOnly")]
    public void Vp9NeighborContexts_PartitionPlane_Block16x16_OnlyBit1Counts()
    {
        CpuOnlyTest();
        // bsl = 1. above_ctx 0b11111101 -> bit 1 = 0, same for left.
        // Other bits don't contribute. Split state = 0, base for bsl=1
        // group is 1*4 = 4.
        Equal(4, Vp9NeighborContexts.GetPartitionPlaneContext(
            0b11111101, 0b11111101, Vp9BlockSize.Block16x16));
    }

    [TestMethod(Category = "CpuOnly")]
    public void Vp9NeighborContexts_PartitionPlane_RejectsOutOfRangeBlock()
    {
        CpuOnlyTest();
        Throws<ArgumentOutOfRangeException>(() =>
            Vp9NeighborContexts.GetPartitionPlaneContext(0, 0, (Vp9BlockSize)99));
    }

    [TestMethod(Category = "CpuOnly")]
    public void Vp9NeighborContexts_SwitchableInterp_BothMissing_ReturnsSentinel()
    {
        CpuOnlyTest();
        // Both sides absent -> both sentinel -> return sentinel (3).
        Equal(3, Vp9NeighborContexts.GetSwitchableInterpContext(null, null));
    }

    [TestMethod(Category = "CpuOnly")]
    public void Vp9NeighborContexts_SwitchableInterp_BothIntra_ReturnsSentinel()
    {
        CpuOnlyTest();
        // Both present but intra -> both sentinel -> return sentinel.
        Equal(3, Vp9NeighborContexts.GetSwitchableInterpContext(
            (false, Vp9InterpFilter.EightTap),
            (false, Vp9InterpFilter.EightTap)));
    }

    [TestMethod(Category = "CpuOnly")]
    public void Vp9NeighborContexts_SwitchableInterp_BothMatchEightTap()
    {
        CpuOnlyTest();
        Equal((int)Vp9InterpFilter.EightTap,
            Vp9NeighborContexts.GetSwitchableInterpContext(
                (true, Vp9InterpFilter.EightTap),
                (true, Vp9InterpFilter.EightTap)));
    }

    [TestMethod(Category = "CpuOnly")]
    public void Vp9NeighborContexts_SwitchableInterp_BothMatchSmooth()
    {
        CpuOnlyTest();
        Equal((int)Vp9InterpFilter.EightTapSmooth,
            Vp9NeighborContexts.GetSwitchableInterpContext(
                (true, Vp9InterpFilter.EightTapSmooth),
                (true, Vp9InterpFilter.EightTapSmooth)));
    }

    [TestMethod(Category = "CpuOnly")]
    public void Vp9NeighborContexts_SwitchableInterp_OnlyAboveInter_ReturnsAboveFilter()
    {
        CpuOnlyTest();
        // Left intra/missing, above inter EightTapSharp.
        Equal((int)Vp9InterpFilter.EightTapSharp,
            Vp9NeighborContexts.GetSwitchableInterpContext(
                (true, Vp9InterpFilter.EightTapSharp),
                null));
        Equal((int)Vp9InterpFilter.EightTapSharp,
            Vp9NeighborContexts.GetSwitchableInterpContext(
                (true, Vp9InterpFilter.EightTapSharp),
                (false, Vp9InterpFilter.EightTap)));
    }

    [TestMethod(Category = "CpuOnly")]
    public void Vp9NeighborContexts_SwitchableInterp_OnlyLeftInter_ReturnsLeftFilter()
    {
        CpuOnlyTest();
        Equal((int)Vp9InterpFilter.EightTapSmooth,
            Vp9NeighborContexts.GetSwitchableInterpContext(
                null,
                (true, Vp9InterpFilter.EightTapSmooth)));
    }

    [TestMethod(Category = "CpuOnly")]
    public void Vp9NeighborContexts_SwitchableInterp_BothInterDifferent_ReturnsSentinel()
    {
        CpuOnlyTest();
        Equal(3, Vp9NeighborContexts.GetSwitchableInterpContext(
            (true, Vp9InterpFilter.EightTap),
            (true, Vp9InterpFilter.EightTapSharp)));
    }

    [TestMethod(Category = "CpuOnly")]
    public void Vp9NeighborContexts_SegId_BothMissing()
    {
        CpuOnlyTest();
        Equal(0, Vp9NeighborContexts.GetSegIdContext(null, null));
    }

    [TestMethod(Category = "CpuOnly")]
    public void Vp9NeighborContexts_SegId_OneSidePredicted()
    {
        CpuOnlyTest();
        Equal(1, Vp9NeighborContexts.GetSegIdContext(true, null));
        Equal(1, Vp9NeighborContexts.GetSegIdContext(null, true));
        Equal(1, Vp9NeighborContexts.GetSegIdContext(true, false));
        Equal(1, Vp9NeighborContexts.GetSegIdContext(false, true));
    }

    [TestMethod(Category = "CpuOnly")]
    public void Vp9NeighborContexts_SegId_BothPredicted()
    {
        CpuOnlyTest();
        Equal(2, Vp9NeighborContexts.GetSegIdContext(true, true));
    }

    [TestMethod(Category = "CpuOnly")]
    public void Vp9NeighborContexts_SegId_BothPresentNeitherPredicted()
    {
        CpuOnlyTest();
        Equal(0, Vp9NeighborContexts.GetSegIdContext(false, false));
    }
}
