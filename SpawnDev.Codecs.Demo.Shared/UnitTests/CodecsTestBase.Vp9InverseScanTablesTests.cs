// Tests for Vp9InverseScanTables. Each iscan table must (a) have the
// correct length and (b) be consistent with the matching forward scan
// from slice 135: iscan[scan[i]] should equal i + 1 for every scan
// position i. The consistency invariant catches transposed entries
// without needing to pin every value individually.

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private static void AssertIscanRoundTripsScan(
        ushort[] scan, ushort[] iscan, int n, string label)
    {
        Equal(n, scan.Length);
        Equal(n, iscan.Length);
        // For every scan position i, the forward scan gives raster
        // position scan[i]; libvpx iscan stores `scan_position + 1`
        // at that raster slot. The inverse round-trip must hold.
        for (int i = 0; i < n; i++)
        {
            int raster = scan[i];
            int recovered = iscan[raster];
            Equal(i + 1, recovered);
        }
    }

    [TestMethod]
    public void Vp9InverseScanTables_4x4_AllThreeFlavorsRoundTrip()
    {
        AssertIscanRoundTripsScan(Vp9ScanTables.DefaultScan4x4, Vp9InverseScanTables.DefaultIscan4x4, 16, "default 4x4");
        AssertIscanRoundTripsScan(Vp9ScanTables.RowScan4x4,     Vp9InverseScanTables.RowIscan4x4,     16, "row 4x4");
        AssertIscanRoundTripsScan(Vp9ScanTables.ColScan4x4,     Vp9InverseScanTables.ColIscan4x4,     16, "col 4x4");
    }

    [TestMethod]
    public void Vp9InverseScanTables_8x8_AllThreeFlavorsRoundTrip()
    {
        AssertIscanRoundTripsScan(Vp9ScanTables.DefaultScan8x8, Vp9InverseScanTables.DefaultIscan8x8, 64, "default 8x8");
        AssertIscanRoundTripsScan(Vp9ScanTables.RowScan8x8,     Vp9InverseScanTables.RowIscan8x8,     64, "row 8x8");
        AssertIscanRoundTripsScan(Vp9ScanTables.ColScan8x8,     Vp9InverseScanTables.ColIscan8x8,     64, "col 8x8");
    }

    [TestMethod]
    public void Vp9InverseScanTables_16x16_AllThreeFlavorsRoundTrip()
    {
        AssertIscanRoundTripsScan(Vp9ScanTables.DefaultScan16x16, Vp9InverseScanTables.DefaultIscan16x16, 256, "default 16x16");
        AssertIscanRoundTripsScan(Vp9ScanTables.RowScan16x16,     Vp9InverseScanTables.RowIscan16x16,     256, "row 16x16");
        AssertIscanRoundTripsScan(Vp9ScanTables.ColScan16x16,     Vp9InverseScanTables.ColIscan16x16,     256, "col 16x16");
    }

    [TestMethod]
    public void Vp9InverseScanTables_32x32_DefaultFlavorRoundTrips()
    {
        AssertIscanRoundTripsScan(Vp9ScanTables.DefaultScan32x32, Vp9InverseScanTables.DefaultIscan32x32, 1024, "default 32x32");
    }

    [TestMethod]
    public void Vp9InverseScanTables_GetIscan_DispatchesCorrectly()
    {
        True(ReferenceEquals(Vp9InverseScanTables.DefaultIscan4x4,  Vp9InverseScanTables.GetIscan(Vp9TxSize.Tx4x4,   Vp9ScanType.Default)));
        True(ReferenceEquals(Vp9InverseScanTables.RowIscan4x4,      Vp9InverseScanTables.GetIscan(Vp9TxSize.Tx4x4,   Vp9ScanType.Row)));
        True(ReferenceEquals(Vp9InverseScanTables.ColIscan4x4,      Vp9InverseScanTables.GetIscan(Vp9TxSize.Tx4x4,   Vp9ScanType.Col)));
        True(ReferenceEquals(Vp9InverseScanTables.DefaultIscan32x32, Vp9InverseScanTables.GetIscan(Vp9TxSize.Tx32x32, Vp9ScanType.Default)));
        True(ReferenceEquals(Vp9InverseScanTables.DefaultIscan32x32, Vp9InverseScanTables.GetIscan(Vp9TxSize.Tx32x32, Vp9ScanType.Row)));
        True(ReferenceEquals(Vp9InverseScanTables.DefaultIscan32x32, Vp9InverseScanTables.GetIscan(Vp9TxSize.Tx32x32, Vp9ScanType.Col)));
    }
}
