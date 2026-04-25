// Tests for Vp9ScanTables.ScanTypeForTxType + GetIntraScan (slice 234).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9ScanPicker_ScanTypeForTxType_DctDct_IsDefault()
    {
        Equal(Vp9ScanType.Default, Vp9ScanTables.ScanTypeForTxType(Vp9TxType.DctDct));
    }

    [TestMethod]
    public void Vp9ScanPicker_ScanTypeForTxType_AdstDct_IsRow()
    {
        Equal(Vp9ScanType.Row, Vp9ScanTables.ScanTypeForTxType(Vp9TxType.AdstDct));
    }

    [TestMethod]
    public void Vp9ScanPicker_ScanTypeForTxType_DctAdst_IsCol()
    {
        Equal(Vp9ScanType.Col, Vp9ScanTables.ScanTypeForTxType(Vp9TxType.DctAdst));
    }

    [TestMethod]
    public void Vp9ScanPicker_ScanTypeForTxType_AdstAdst_IsDefault()
    {
        Equal(Vp9ScanType.Default, Vp9ScanTables.ScanTypeForTxType(Vp9TxType.AdstAdst));
    }

    [TestMethod]
    public void Vp9ScanPicker_GetIntraScan_DcPred_4x4_DefaultScan()
    {
        // DC_PRED -> DCT_DCT -> Default scan.
        var scan = Vp9ScanTables.GetIntraScan(Vp9TxSize.Tx4x4, Vp9IntraMode.DcPred);
        Equal(true, ReferenceEquals(Vp9ScanTables.DefaultScan4x4, scan));
    }

    [TestMethod]
    public void Vp9ScanPicker_GetIntraScan_VPred_4x4_RowScan()
    {
        // V_PRED -> ADST_DCT -> Row scan.
        var scan = Vp9ScanTables.GetIntraScan(Vp9TxSize.Tx4x4, Vp9IntraMode.VPred);
        Equal(true, ReferenceEquals(Vp9ScanTables.RowScan4x4, scan));
    }

    [TestMethod]
    public void Vp9ScanPicker_GetIntraScan_HPred_4x4_ColScan()
    {
        // H_PRED -> DCT_ADST -> Col scan.
        var scan = Vp9ScanTables.GetIntraScan(Vp9TxSize.Tx4x4, Vp9IntraMode.HPred);
        Equal(true, ReferenceEquals(Vp9ScanTables.ColScan4x4, scan));
    }

    [TestMethod]
    public void Vp9ScanPicker_GetIntraScan_TmPred_8x8_DefaultScan()
    {
        // TM_PRED -> ADST_ADST -> Default scan.
        var scan = Vp9ScanTables.GetIntraScan(Vp9TxSize.Tx8x8, Vp9IntraMode.TmPred);
        Equal(true, ReferenceEquals(Vp9ScanTables.DefaultScan8x8, scan));
    }

    [TestMethod]
    public void Vp9ScanPicker_GetIntraScan_VPred_16x16_RowScan()
    {
        var scan = Vp9ScanTables.GetIntraScan(Vp9TxSize.Tx16x16, Vp9IntraMode.VPred);
        Equal(true, ReferenceEquals(Vp9ScanTables.RowScan16x16, scan));
    }

    [TestMethod]
    public void Vp9ScanPicker_GetIntraScan_32x32_AlwaysDefault()
    {
        // 32x32 always uses the default scan regardless of intra mode.
        Equal(true, ReferenceEquals(Vp9ScanTables.DefaultScan32x32,
            Vp9ScanTables.GetIntraScan(Vp9TxSize.Tx32x32, Vp9IntraMode.DcPred)));
        Equal(true, ReferenceEquals(Vp9ScanTables.DefaultScan32x32,
            Vp9ScanTables.GetIntraScan(Vp9TxSize.Tx32x32, Vp9IntraMode.VPred)));
        Equal(true, ReferenceEquals(Vp9ScanTables.DefaultScan32x32,
            Vp9ScanTables.GetIntraScan(Vp9TxSize.Tx32x32, Vp9IntraMode.HPred)));
        Equal(true, ReferenceEquals(Vp9ScanTables.DefaultScan32x32,
            Vp9ScanTables.GetIntraScan(Vp9TxSize.Tx32x32, Vp9IntraMode.TmPred)));
    }

    [TestMethod]
    public void Vp9ScanPicker_GetIntraScan_RejectsOutOfRangeMode()
    {
        // Out-of-range intra mode propagates from Vp9IntraTxType.ForMode.
        Throws<ArgumentOutOfRangeException>(() =>
            Vp9ScanTables.GetIntraScan(Vp9TxSize.Tx4x4, (Vp9IntraMode)99));
    }
}
