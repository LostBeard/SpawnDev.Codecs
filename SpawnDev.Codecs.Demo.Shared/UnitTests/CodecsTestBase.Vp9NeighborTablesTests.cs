// Tests for Vp9NeighborTables. Each table must be the correct length,
// have (0,0) padding at scan position 0 (DC has no entropy-context
// neighbors), and reference only valid raster positions inside the
// block. The position-0 (0,0) check + length check + range check
// catches every plausible copy error short of an interior swap.

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private static void AssertNeighborTableShape(
        ushort[] neighbors, int blockSize, string label)
    {
        // Layout: (blockSize + 1) pairs of (n0, n1).
        Equal((blockSize + 1) * 2, neighbors.Length);
        // Scan position 0 is the DC slot; libvpx pads its neighbors
        // with (0, 0) so get_coef_context returns 0 at the boundary.
        Equal((ushort)0, neighbors[0]);
        Equal((ushort)0, neighbors[1]);
        // Last pair is the boundary marker, also (0, 0).
        Equal((ushort)0, neighbors[neighbors.Length - 2]);
        Equal((ushort)0, neighbors[neighbors.Length - 1]);
        // Every interior neighbor must be inside the block.
        for (int i = 0; i < neighbors.Length; i++)
        {
            True(neighbors[i] < blockSize,
                $"{label}[{i}] = {neighbors[i]} out of range [0,{blockSize})");
        }
    }

    [TestMethod]
    public void Vp9NeighborTables_4x4_AllThreeShapesValid()
    {
        AssertNeighborTableShape(Vp9NeighborTables.DefaultScan4x4Neighbors, 16, "DefaultScan4x4Neighbors");
        AssertNeighborTableShape(Vp9NeighborTables.RowScan4x4Neighbors,     16, "RowScan4x4Neighbors");
        AssertNeighborTableShape(Vp9NeighborTables.ColScan4x4Neighbors,     16, "ColScan4x4Neighbors");
    }

    [TestMethod]
    public void Vp9NeighborTables_8x8_AllThreeShapesValid()
    {
        AssertNeighborTableShape(Vp9NeighborTables.DefaultScan8x8Neighbors, 64, "DefaultScan8x8Neighbors");
        AssertNeighborTableShape(Vp9NeighborTables.RowScan8x8Neighbors,     64, "RowScan8x8Neighbors");
        AssertNeighborTableShape(Vp9NeighborTables.ColScan8x8Neighbors,     64, "ColScan8x8Neighbors");
    }

    [TestMethod]
    public void Vp9NeighborTables_4x4DefaultScan_PinnedFirstFewPairs()
    {
        // Pinned pairs from libvpx default_scan_4x4_neighbors:
        // (0,0)(0,0)(0,0)(1,4)(4,4)(1,1)(8,8)(5,8)...
        var t = Vp9NeighborTables.DefaultScan4x4Neighbors;
        Equal((ushort)0, t[0]);  Equal((ushort)0, t[1]);   // pos 0 -> (0,0)
        Equal((ushort)0, t[2]);  Equal((ushort)0, t[3]);   // pos 1 -> (0,0)
        Equal((ushort)0, t[4]);  Equal((ushort)0, t[5]);   // pos 2 -> (0,0)
        Equal((ushort)1, t[6]);  Equal((ushort)4, t[7]);   // pos 3 -> (1,4)
        Equal((ushort)4, t[8]);  Equal((ushort)4, t[9]);   // pos 4 -> (4,4)
    }

    [TestMethod]
    public void Vp9NeighborTables_4x4ColScan_PinnedFirstFewPairs()
    {
        // Distinct from default - col scan has a (4,4) pair already at
        // index pair 2 instead of three (0,0) pairs.
        var t = Vp9NeighborTables.ColScan4x4Neighbors;
        Equal((ushort)0, t[0]);  Equal((ushort)0, t[1]);   // pos 0 -> (0,0)
        Equal((ushort)0, t[2]);  Equal((ushort)0, t[3]);   // pos 1 -> (0,0)
        Equal((ushort)4, t[4]);  Equal((ushort)4, t[5]);   // pos 2 -> (4,4)
        Equal((ushort)0, t[6]);  Equal((ushort)0, t[7]);   // pos 3 -> (0,0)
        Equal((ushort)8, t[8]);  Equal((ushort)8, t[9]);   // pos 4 -> (8,8)
    }

    [TestMethod]
    public void Vp9NeighborTables_GetNeighbors4x4_DispatchesByScanType()
    {
        True(ReferenceEquals(Vp9NeighborTables.DefaultScan4x4Neighbors, Vp9NeighborTables.GetNeighbors4x4(Vp9ScanType.Default)));
        True(ReferenceEquals(Vp9NeighborTables.RowScan4x4Neighbors,     Vp9NeighborTables.GetNeighbors4x4(Vp9ScanType.Row)));
        True(ReferenceEquals(Vp9NeighborTables.ColScan4x4Neighbors,     Vp9NeighborTables.GetNeighbors4x4(Vp9ScanType.Col)));
    }

    [TestMethod]
    public void Vp9NeighborTables_GetNeighbors8x8_DispatchesByScanType()
    {
        True(ReferenceEquals(Vp9NeighborTables.DefaultScan8x8Neighbors, Vp9NeighborTables.GetNeighbors8x8(Vp9ScanType.Default)));
        True(ReferenceEquals(Vp9NeighborTables.RowScan8x8Neighbors,     Vp9NeighborTables.GetNeighbors8x8(Vp9ScanType.Row)));
        True(ReferenceEquals(Vp9NeighborTables.ColScan8x8Neighbors,     Vp9NeighborTables.GetNeighbors8x8(Vp9ScanType.Col)));
    }
}
