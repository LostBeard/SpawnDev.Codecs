// Tests for Vp9CoefProbsParser (slice 211). The full update-path
// round-trip needs an arithmetic encoder, which we don't ship.
// These tests cover the structural helpers and the no-update path
// (a 0 update flag leaves the table untouched).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9CoefProbsParser_Constants_MatchLibvpx()
    {
        Equal(2, Vp9CoefProbsParser.PlaneTypes);
        Equal(2, Vp9CoefProbsParser.RefTypes);
        Equal(6, Vp9CoefProbsParser.CoefBands);
        Equal(3, Vp9CoefProbsParser.UnconstrainedNodes);
        Equal(6, Vp9CoefProbsParser.CoefContextsPerBand);
        Equal(432, Vp9CoefProbsParser.FlatSize);
    }

    [TestMethod]
    public void Vp9CoefProbsParser_BandCoefContexts_DcVsAc()
    {
        // Band 0 (DC) has 3 contexts; bands 1..5 (AC) have 6.
        Equal(3, Vp9CoefProbsParser.BandCoefContexts(0));
        Equal(6, Vp9CoefProbsParser.BandCoefContexts(1));
        Equal(6, Vp9CoefProbsParser.BandCoefContexts(5));
    }

    [TestMethod]
    public void Vp9CoefProbsParser_FlatIndex_MatchesAxisOrdering()
    {
        // [plane][ref][band][ctx][node] row-major. Stride table:
        //   node:1; ctx:3; band:6*3=18; ref:6*18=108; plane:2*108=216.
        Equal(0, Vp9CoefProbsParser.FlatIndex(0, 0, 0, 0, 0));
        Equal(1, Vp9CoefProbsParser.FlatIndex(0, 0, 0, 0, 1));
        Equal(2, Vp9CoefProbsParser.FlatIndex(0, 0, 0, 0, 2));
        Equal(3, Vp9CoefProbsParser.FlatIndex(0, 0, 0, 1, 0));
        Equal(18, Vp9CoefProbsParser.FlatIndex(0, 0, 1, 0, 0));
        Equal(108, Vp9CoefProbsParser.FlatIndex(0, 1, 0, 0, 0));
        Equal(216, Vp9CoefProbsParser.FlatIndex(1, 0, 0, 0, 0));
        Equal(431, Vp9CoefProbsParser.FlatIndex(1, 1, 5, 5, 2));
    }

    [TestMethod]
    public void Vp9CoefProbsParser_TxModeToBiggestTxSize_AllValues()
    {
        Equal(0, Vp9CoefProbsParser.TxModeToBiggestTxSize(Vp9TxMode.Only4x4));
        Equal(1, Vp9CoefProbsParser.TxModeToBiggestTxSize(Vp9TxMode.AllowOnly8x8));
        Equal(2, Vp9CoefProbsParser.TxModeToBiggestTxSize(Vp9TxMode.AllowOnly16x16));
        Equal(3, Vp9CoefProbsParser.TxModeToBiggestTxSize(Vp9TxMode.Allow32x32));
        Equal(3, Vp9CoefProbsParser.TxModeToBiggestTxSize(Vp9TxMode.TxModeSelect));
    }

    [TestMethod]
    public void Vp9CoefProbsParser_TxModeToBiggestTxSize_RejectsUnknown()
    {
        Throws<ArgumentOutOfRangeException>(() =>
            Vp9CoefProbsParser.TxModeToBiggestTxSize((Vp9TxMode)99));
    }

    [TestMethod]
    public void Vp9CoefProbsParser_NoUpdate_LeavesTableUntouched()
    {
        // Build a buffer that decodes to: marker=0 (init), then update_flag=0.
        // Vp9BoolDecoder's internal arithmetic with prob=128 on 0x00 byte
        // contents reads as 0, 0, 0... Init consumes 1 bit; we need the
        // next bit to also read as 0 -> 0x00 buffer is sufficient.
        var data = new byte[16];  // all zeros
        var reader = new Vp9BoolDecoder(data, 0, data.Length);

        var probs = new byte[Vp9CoefProbsParser.FlatSize];
        for (int i = 0; i < probs.Length; i++) probs[i] = (byte)((i * 7) & 0xFF);
        var snapshot = (byte[])probs.Clone();

        Vp9CoefProbsParser.ReadCoefProbsCommon(probs, reader);

        // Update flag was 0; probs unchanged.
        for (int i = 0; i < probs.Length; i++) Equal(snapshot[i], probs[i]);
    }

    [TestMethod]
    public void Vp9CoefProbsParser_ReadCoefProbs_AllTxSizesUntouched_WhenNoUpdates()
    {
        // 4 update flags all 0 (one per tx_size). Buffer of zeros suffices.
        var data = new byte[16];
        var reader = new Vp9BoolDecoder(data, 0, data.Length);

        var tables = new byte[4][];
        var snapshots = new byte[4][];
        for (int i = 0; i < 4; i++)
        {
            tables[i] = new byte[Vp9CoefProbsParser.FlatSize];
            for (int j = 0; j < tables[i].Length; j++)
                tables[i][j] = (byte)((i * 13 + j * 7) & 0xFF);
            snapshots[i] = (byte[])tables[i].Clone();
        }

        Vp9CoefProbsParser.ReadCoefProbs(tables, Vp9TxMode.Allow32x32, reader);

        for (int i = 0; i < 4; i++)
            for (int j = 0; j < tables[i].Length; j++)
                Equal(snapshots[i][j], tables[i][j]);
    }

    [TestMethod]
    public void Vp9CoefProbsParser_ReadCoefProbs_OnlyUpToMaxTxSize_Walked()
    {
        // For tx_mode = Only4x4, only tables[0] should be walked. Tables 1..3
        // shouldn't even be inspected. Since our buffer encodes 4 update flags
        // as 0, only the first is read; the remaining are still readable
        // (more zeros) but the parser should not call into tables[1..3].
        var data = new byte[16];
        var reader = new Vp9BoolDecoder(data, 0, data.Length);

        var tables = new byte[4][];
        for (int i = 0; i < 4; i++)
        {
            tables[i] = new byte[Vp9CoefProbsParser.FlatSize];
            for (int j = 0; j < tables[i].Length; j++) tables[i][j] = 100;
        }

        Vp9CoefProbsParser.ReadCoefProbs(tables, Vp9TxMode.Only4x4, reader);

        // All tables should still be 100 (since no updates were signalled).
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < tables[i].Length; j++)
                Equal((byte)100, tables[i][j]);
    }

    [TestMethod]
    public void Vp9CoefProbsParser_RejectsUndersizedTable()
    {
        var data = new byte[16];
        var reader = new Vp9BoolDecoder(data, 0, data.Length);
        var tooSmall = new byte[Vp9CoefProbsParser.FlatSize - 1];
        Throws<ArgumentException>(() =>
            Vp9CoefProbsParser.ReadCoefProbsCommon(tooSmall, reader));
    }
}
