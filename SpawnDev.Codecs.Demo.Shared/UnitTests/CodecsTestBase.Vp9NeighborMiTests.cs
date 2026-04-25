// Tests for Vp9NeighborMi (slice 278).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9NeighborMi_AboveAtTopRow_OutOfBounds()
    {
        var pos = Vp9NeighborMi.GetNeighbor(
            currentMiRow: 0, currentMiCol: 5,
            rowOffset: -1, colOffset: 0,
            frameMiRows: 32, frameMiCols: 32);
        Equal(false, pos.HasValue);
    }

    [TestMethod]
    public void Vp9NeighborMi_AboveAtNonTopRow_InBounds()
    {
        var pos = Vp9NeighborMi.GetNeighbor(
            currentMiRow: 5, currentMiCol: 5,
            rowOffset: -1, colOffset: 0,
            frameMiRows: 32, frameMiCols: 32);
        Equal(true, pos.HasValue);
        Equal(4, pos!.Value.Row);
        Equal(5, pos.Value.Col);
    }

    [TestMethod]
    public void Vp9NeighborMi_LeftAtCol0_OutOfBounds()
    {
        var pos = Vp9NeighborMi.GetNeighbor(
            currentMiRow: 5, currentMiCol: 0,
            rowOffset: 0, colOffset: -1,
            frameMiRows: 32, frameMiCols: 32);
        Equal(false, pos.HasValue);
    }

    [TestMethod]
    public void Vp9NeighborMi_LargerOffsets_PastEdge()
    {
        // Row -3 from row 1 -> -2, out of bounds.
        var pos = Vp9NeighborMi.GetNeighbor(
            currentMiRow: 1, currentMiCol: 5,
            rowOffset: -3, colOffset: 0,
            frameMiRows: 32, frameMiCols: 32);
        Equal(false, pos.HasValue);
    }

    [TestMethod]
    public void Vp9NeighborMi_RightOfFrame_OutOfBounds()
    {
        // Col offset = 6 from col 30 -> 36, past 32-col frame.
        var pos = Vp9NeighborMi.GetNeighbor(
            currentMiRow: 5, currentMiCol: 30,
            rowOffset: 0, colOffset: 6,
            frameMiRows: 32, frameMiCols: 32);
        Equal(false, pos.HasValue);
    }

    [TestMethod]
    public void Vp9NeighborMi_BelowFrame_OutOfBounds()
    {
        var pos = Vp9NeighborMi.GetNeighbor(
            currentMiRow: 31, currentMiCol: 5,
            rowOffset: 1, colOffset: 0,
            frameMiRows: 32, frameMiCols: 32);
        Equal(false, pos.HasValue);
    }

    [TestMethod]
    public void Vp9NeighborMi_IsInBounds_MatchesGetNeighbor()
    {
        Equal(true, Vp9NeighborMi.IsInBounds(5, 5, -1, 0, 32, 32));
        Equal(false, Vp9NeighborMi.IsInBounds(0, 0, -1, 0, 32, 32));
    }

    [TestMethod]
    public void Vp9NeighborMi_RejectsZeroOrNegativeFrameDimensions()
    {
        Throws<ArgumentOutOfRangeException>(() =>
            Vp9NeighborMi.GetNeighbor(0, 0, 0, 0, 0, 32));
        Throws<ArgumentOutOfRangeException>(() =>
            Vp9NeighborMi.GetNeighbor(0, 0, 0, 0, 32, 0));
    }
}
