// Tests for Vp9ScanTables. Each scan table must (a) be the correct
// length, (b) start with raster-position 0 (DC is always at scan-pos
// 0), and (c) be a valid permutation of [0, N-1] - any duplicate or
// missing raster position would corrupt the inverse-scan path. The
// permutation invariant catches most table copy errors without
// hand-pinning every entry.

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private static void AssertValidScanPermutation(ushort[] scan, int expectedLength)
    {
        Equal(expectedLength, scan.Length);
        // DC (raster position 0) is always at scan position 0.
        Equal((ushort)0, scan[0]);
        var seen = new bool[expectedLength];
        for (int i = 0; i < expectedLength; i++)
        {
            int p = scan[i];
            True(p >= 0 && p < expectedLength,
                $"scan[{i}] = {p} out of range [0,{expectedLength})");
            True(!seen[p],
                $"raster position {p} appears twice in scan (at idx {i})");
            seen[p] = true;
        }
        // After the loop every raster position must have been visited.
        for (int p = 0; p < expectedLength; p++)
            True(seen[p], $"raster position {p} missing from scan");
    }

    [TestMethod]
    public void Vp9ScanTables_4x4_AllThreeScansAreValidPermutations()
    {
        AssertValidScanPermutation(Vp9ScanTables.DefaultScan4x4, 16);
        AssertValidScanPermutation(Vp9ScanTables.RowScan4x4, 16);
        AssertValidScanPermutation(Vp9ScanTables.ColScan4x4, 16);
    }

    [TestMethod]
    public void Vp9ScanTables_8x8_AllThreeScansAreValidPermutations()
    {
        AssertValidScanPermutation(Vp9ScanTables.DefaultScan8x8, 64);
        AssertValidScanPermutation(Vp9ScanTables.RowScan8x8, 64);
        AssertValidScanPermutation(Vp9ScanTables.ColScan8x8, 64);
    }

    [TestMethod]
    public void Vp9ScanTables_16x16_AllThreeScansAreValidPermutations()
    {
        AssertValidScanPermutation(Vp9ScanTables.DefaultScan16x16, 256);
        AssertValidScanPermutation(Vp9ScanTables.RowScan16x16, 256);
        AssertValidScanPermutation(Vp9ScanTables.ColScan16x16, 256);
    }

    [TestMethod]
    public void Vp9ScanTables_32x32_DefaultScanIsValidPermutation()
    {
        AssertValidScanPermutation(Vp9ScanTables.DefaultScan32x32, 1024);
    }

    [TestMethod]
    public void Vp9ScanTables_4x4DefaultScan_PinnedFirstAndLastEntries()
    {
        // Pinned vs libvpx default_scan_4x4: starts 0,4,1,5,8,...
        // and ends ...11,15.
        Equal((ushort)0,  Vp9ScanTables.DefaultScan4x4[0]);
        Equal((ushort)4,  Vp9ScanTables.DefaultScan4x4[1]);
        Equal((ushort)1,  Vp9ScanTables.DefaultScan4x4[2]);
        Equal((ushort)5,  Vp9ScanTables.DefaultScan4x4[3]);
        Equal((ushort)11, Vp9ScanTables.DefaultScan4x4[14]);
        Equal((ushort)15, Vp9ScanTables.DefaultScan4x4[15]);
    }

    [TestMethod]
    public void Vp9ScanTables_4x4RowScan_PinnedFirstAndLastEntries()
    {
        Equal((ushort)0,  Vp9ScanTables.RowScan4x4[0]);
        Equal((ushort)1,  Vp9ScanTables.RowScan4x4[1]);
        Equal((ushort)4,  Vp9ScanTables.RowScan4x4[2]);
        Equal((ushort)15, Vp9ScanTables.RowScan4x4[15]);
    }

    [TestMethod]
    public void Vp9ScanTables_4x4ColScan_PinnedFirstAndLastEntries()
    {
        Equal((ushort)0,  Vp9ScanTables.ColScan4x4[0]);
        Equal((ushort)4,  Vp9ScanTables.ColScan4x4[1]);
        Equal((ushort)8,  Vp9ScanTables.ColScan4x4[2]);
        Equal((ushort)15, Vp9ScanTables.ColScan4x4[15]);
    }

    [TestMethod]
    public void Vp9ScanTables_32x32DefaultScan_LastEntryIs1023()
    {
        // The 1024-entry table's terminal value must be the highest
        // raster position - the permutation invariant test catches
        // missing entries, but pinning the last value too gives a
        // direct cross-check against libvpx.
        Equal((ushort)1023, Vp9ScanTables.DefaultScan32x32[1023]);
    }

    [TestMethod]
    public void Vp9ScanTables_ScanTypeForTxType4x4_MapsCorrectly()
    {
        Equal(Vp9ScanType.Default, Vp9ScanTables.ScanTypeForTxType4x4(Vp9TxType4x4.DctDct));
        Equal(Vp9ScanType.Row,     Vp9ScanTables.ScanTypeForTxType4x4(Vp9TxType4x4.AdstDct));
        Equal(Vp9ScanType.Col,     Vp9ScanTables.ScanTypeForTxType4x4(Vp9TxType4x4.DctAdst));
        Equal(Vp9ScanType.Default, Vp9ScanTables.ScanTypeForTxType4x4(Vp9TxType4x4.AdstAdst));
    }

    [TestMethod]
    public void Vp9ScanTables_ScanTypeForTxType8x8_MapsCorrectly()
    {
        Equal(Vp9ScanType.Default, Vp9ScanTables.ScanTypeForTxType8x8(Vp9TxType8x8.DctDct));
        Equal(Vp9ScanType.Row,     Vp9ScanTables.ScanTypeForTxType8x8(Vp9TxType8x8.AdstDct));
        Equal(Vp9ScanType.Col,     Vp9ScanTables.ScanTypeForTxType8x8(Vp9TxType8x8.DctAdst));
        Equal(Vp9ScanType.Default, Vp9ScanTables.ScanTypeForTxType8x8(Vp9TxType8x8.AdstAdst));
    }

    [TestMethod]
    public void Vp9ScanTables_GetScan_DispatchesByTxSizeAndScanType()
    {
        // 4x4 / 8x8 / 16x16 cover all three flavors; 32x32 falls
        // back to default for any flavor.
        True(ReferenceEquals(Vp9ScanTables.DefaultScan4x4,   Vp9ScanTables.GetScan(Vp9TxSize.Tx4x4,   Vp9ScanType.Default)));
        True(ReferenceEquals(Vp9ScanTables.RowScan4x4,       Vp9ScanTables.GetScan(Vp9TxSize.Tx4x4,   Vp9ScanType.Row)));
        True(ReferenceEquals(Vp9ScanTables.ColScan4x4,       Vp9ScanTables.GetScan(Vp9TxSize.Tx4x4,   Vp9ScanType.Col)));
        True(ReferenceEquals(Vp9ScanTables.DefaultScan8x8,   Vp9ScanTables.GetScan(Vp9TxSize.Tx8x8,   Vp9ScanType.Default)));
        True(ReferenceEquals(Vp9ScanTables.RowScan8x8,       Vp9ScanTables.GetScan(Vp9TxSize.Tx8x8,   Vp9ScanType.Row)));
        True(ReferenceEquals(Vp9ScanTables.ColScan8x8,       Vp9ScanTables.GetScan(Vp9TxSize.Tx8x8,   Vp9ScanType.Col)));
        True(ReferenceEquals(Vp9ScanTables.DefaultScan16x16, Vp9ScanTables.GetScan(Vp9TxSize.Tx16x16, Vp9ScanType.Default)));
        True(ReferenceEquals(Vp9ScanTables.RowScan16x16,     Vp9ScanTables.GetScan(Vp9TxSize.Tx16x16, Vp9ScanType.Row)));
        True(ReferenceEquals(Vp9ScanTables.ColScan16x16,     Vp9ScanTables.GetScan(Vp9TxSize.Tx16x16, Vp9ScanType.Col)));
        True(ReferenceEquals(Vp9ScanTables.DefaultScan32x32, Vp9ScanTables.GetScan(Vp9TxSize.Tx32x32, Vp9ScanType.Default)));
        True(ReferenceEquals(Vp9ScanTables.DefaultScan32x32, Vp9ScanTables.GetScan(Vp9TxSize.Tx32x32, Vp9ScanType.Row)));
        True(ReferenceEquals(Vp9ScanTables.DefaultScan32x32, Vp9ScanTables.GetScan(Vp9TxSize.Tx32x32, Vp9ScanType.Col)));
    }
}
