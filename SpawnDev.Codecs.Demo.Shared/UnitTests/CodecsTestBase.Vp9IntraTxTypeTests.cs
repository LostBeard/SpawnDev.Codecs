// Tests for Vp9IntraTxType (slice 173). Verifies the 10-entry
// intra_mode -> tx_type table matches libvpx vp9_blockd.h
// intra_mode_to_tx_type_lookup.

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9IntraTxType_Lookup_HasTenEntries()
    {
        Equal(10, Vp9IntraTxType.Lookup.Length);
    }

    [TestMethod]
    public void Vp9IntraTxType_DcMode_IsDctDct()
    {
        Equal(Vp9TxType.DctDct, Vp9IntraTxType.ForMode(Vp9IntraMode.DcPred));
    }

    [TestMethod]
    public void Vp9IntraTxType_VMode_IsAdstDct()
    {
        Equal(Vp9TxType.AdstDct, Vp9IntraTxType.ForMode(Vp9IntraMode.VPred));
    }

    [TestMethod]
    public void Vp9IntraTxType_HMode_IsDctAdst()
    {
        Equal(Vp9TxType.DctAdst, Vp9IntraTxType.ForMode(Vp9IntraMode.HPred));
    }

    [TestMethod]
    public void Vp9IntraTxType_D45Mode_IsDctDct()
    {
        Equal(Vp9TxType.DctDct, Vp9IntraTxType.ForMode(Vp9IntraMode.D45Pred));
    }

    [TestMethod]
    public void Vp9IntraTxType_D135Mode_IsAdstAdst()
    {
        Equal(Vp9TxType.AdstAdst, Vp9IntraTxType.ForMode(Vp9IntraMode.D135Pred));
    }

    [TestMethod]
    public void Vp9IntraTxType_D117Mode_IsAdstDct()
    {
        Equal(Vp9TxType.AdstDct, Vp9IntraTxType.ForMode(Vp9IntraMode.D117Pred));
    }

    [TestMethod]
    public void Vp9IntraTxType_D153Mode_IsDctAdst()
    {
        Equal(Vp9TxType.DctAdst, Vp9IntraTxType.ForMode(Vp9IntraMode.D153Pred));
    }

    [TestMethod]
    public void Vp9IntraTxType_D207Mode_IsDctAdst()
    {
        Equal(Vp9TxType.DctAdst, Vp9IntraTxType.ForMode(Vp9IntraMode.D207Pred));
    }

    [TestMethod]
    public void Vp9IntraTxType_D63Mode_IsAdstDct()
    {
        Equal(Vp9TxType.AdstDct, Vp9IntraTxType.ForMode(Vp9IntraMode.D63Pred));
    }

    [TestMethod]
    public void Vp9IntraTxType_TmMode_IsAdstAdst()
    {
        Equal(Vp9TxType.AdstAdst, Vp9IntraTxType.ForMode(Vp9IntraMode.TmPred));
    }

    [TestMethod]
    public void Vp9IntraTxType_RowAndColumnBitsMatchSemantics()
    {
        // Per libvpx: low bit = row is iADST, high bit = col is iADST.
        // Verify the enum values encode that.
        Equal(0, (int)Vp9TxType.DctDct);    // row=DCT, col=DCT
        Equal(1, (int)Vp9TxType.AdstDct);   // row=ADST, col=DCT
        Equal(2, (int)Vp9TxType.DctAdst);   // row=DCT, col=ADST
        Equal(3, (int)Vp9TxType.AdstAdst);  // row=ADST, col=ADST

        for (int v = 0; v < 4; v++)
        {
            bool rowAdst = (v & 1) != 0;
            bool colAdst = (v & 2) != 0;
            Vp9TxType tx = (Vp9TxType)v;
            // Round-trip the bit semantics.
            int recomposed = (rowAdst ? 1 : 0) | (colAdst ? 2 : 0);
            Equal(v, recomposed);
            Equal(tx, (Vp9TxType)recomposed);
        }
    }

    [TestMethod]
    public void Vp9IntraTxType_RejectsOutOfRangeMode()
    {
        Throws<ArgumentOutOfRangeException>(() =>
            Vp9IntraTxType.ForMode((Vp9IntraMode)99));
    }
}
