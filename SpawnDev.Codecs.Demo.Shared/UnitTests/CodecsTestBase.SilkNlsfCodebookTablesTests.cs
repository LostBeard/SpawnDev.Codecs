using SpawnDev.Codecs.Audio.Opus.Silk;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for <see cref="SilkNlsfCodebookTables"/> - the real SILK NLSF codebooks
/// (NB/MB order 10 and WB order 16). Verifies that each ported array has the correct
/// length (matching the libopus <c>[N]</c> declaration), that the quantizer step-size
/// constants resolve to the correct <c>SILK_FIX_CONST</c> integers, and that a handful
/// of specific entries match the upstream C source.
///
/// The full table content is NOT reproduced here (32 * 10 = 320 bytes for NB/MB,
/// 32 * 16 = 512 bytes for WB, plus a dozen auxiliary tables). Instead we sample
/// representative entries: first row, last row, and a middle entry per table.
/// If any of these drift, the whole table was corrupted during the port.
/// </summary>
public abstract partial class CodecsTestBase
{
    // -------- NB/MB codebook shape and constants --------

    [TestMethod]
    public void SilkNlsfCb_NbMb_HasCorrectShape()
    {
        var cb = SilkNlsfCodebookTables.NbMb;
        Equal(32, cb.NVectors);
        Equal(10, cb.Order);
        Equal(11796, cb.QuantStepSizeQ16);
        Equal(356, cb.InvQuantStepSizeQ6);

        Equal(320, cb.Cb1NlsfQ8.Length);   // NVectors * Order
        Equal(320, cb.Cb1WghtQ9.Length);
        Equal(64, cb.Cb1Icdf.Length);
        Equal(160, cb.EcSel.Length);        // NVectors * Order / 2 = 160 (one byte per coefficient pair)
        Equal(72, cb.EcIcdf.Length);
        Equal(72, cb.EcRatesQ5.Length);
        Equal(18, cb.PredQ8.Length);        // 2 * (Order - 1)
        Equal(11, cb.DeltaMinQ15.Length);   // Order + 1
    }

    [TestMethod]
    public void SilkNlsfCb_NbMb_SampleEntries_MatchLibopus()
    {
        var cb = SilkNlsfCodebookTables.NbMb;

        // First row of silk_NLSF_CB1_NB_MB_Q8.
        Equal((byte)12, cb.Cb1NlsfQ8[0]);
        Equal((byte)35, cb.Cb1NlsfQ8[1]);
        Equal((byte)228, cb.Cb1NlsfQ8[9]);

        // Last row (vector index 31, indices 310..319).
        Equal((byte)37, cb.Cb1NlsfQ8[310]);
        Equal((byte)230, cb.Cb1NlsfQ8[319]);

        // First and last weights (row 0, row 31).
        Equal((short)2897, cb.Cb1WghtQ9[0]);
        Equal((short)2181, cb.Cb1WghtQ9[319]);

        // Cb1Icdf: boundary values at the two halves' starts/ends.
        Equal((byte)212, cb.Cb1Icdf[0]);
        Equal((byte)0, cb.Cb1Icdf[31]);
        Equal((byte)255, cb.Cb1Icdf[32]);
        Equal((byte)0, cb.Cb1Icdf[63]);

        // PredQ8 first and last.
        Equal((byte)179, cb.PredQ8[0]);
        Equal((byte)92, cb.PredQ8[17]);

        // DeltaMinQ15: first (250), last (461).
        Equal((short)250, cb.DeltaMinQ15[0]);
        Equal((short)461, cb.DeltaMinQ15[10]);
    }

    // -------- WB codebook shape and constants --------

    [TestMethod]
    public void SilkNlsfCb_Wb_HasCorrectShape()
    {
        var cb = SilkNlsfCodebookTables.Wb;
        Equal(32, cb.NVectors);
        Equal(16, cb.Order);
        Equal(9830, cb.QuantStepSizeQ16);
        Equal(427, cb.InvQuantStepSizeQ6);

        Equal(512, cb.Cb1NlsfQ8.Length);   // 32 * 16
        Equal(512, cb.Cb1WghtQ9.Length);
        Equal(64, cb.Cb1Icdf.Length);
        Equal(256, cb.EcSel.Length);        // 32 * 16 / 2
        Equal(72, cb.EcIcdf.Length);
        Equal(72, cb.EcRatesQ5.Length);
        Equal(30, cb.PredQ8.Length);        // 2 * (16 - 1)
        Equal(17, cb.DeltaMinQ15.Length);   // Order + 1
    }

    [TestMethod]
    public void SilkNlsfCb_Wb_SampleEntries_MatchLibopus()
    {
        var cb = SilkNlsfCodebookTables.Wb;

        // First row of silk_NLSF_CB1_WB_Q8: starts at 7, 23, 38, ... ends at 239.
        Equal((byte)7, cb.Cb1NlsfQ8[0]);
        Equal((byte)23, cb.Cb1NlsfQ8[1]);
        Equal((byte)239, cb.Cb1NlsfQ8[15]);

        // Last row (vector index 31, indices 496..511): starts at 15, ends at 237.
        Equal((byte)15, cb.Cb1NlsfQ8[496]);
        Equal((byte)237, cb.Cb1NlsfQ8[511]);

        // Weight table first and last.
        Equal((short)3657, cb.Cb1WghtQ9[0]);
        Equal((short)2607, cb.Cb1WghtQ9[511]);

        // Cb1Icdf boundaries.
        Equal((byte)225, cb.Cb1Icdf[0]);
        Equal((byte)0, cb.Cb1Icdf[31]);
        Equal((byte)255, cb.Cb1Icdf[32]);
        Equal((byte)0, cb.Cb1Icdf[63]);

        // PredQ8 first and last.
        Equal((byte)175, cb.PredQ8[0]);
        Equal((byte)155, cb.PredQ8[29]);

        // DeltaMinQ15: first (100), last (347).
        Equal((short)100, cb.DeltaMinQ15[0]);
        Equal((short)347, cb.DeltaMinQ15[16]);
    }

    // -------- Cross-table sanity --------

    [TestMethod]
    public void SilkNlsfCb_Icdf_EndsWithZero()
    {
        // Each iCDF sub-table ends with 0 so the range coder can reach the terminal state.
        // NB/MB: two sub-tables of 32 entries each.
        var nbMb = SilkNlsfCodebookTables.NbMb;
        Equal((byte)0, nbMb.Cb1Icdf[31]);
        Equal((byte)0, nbMb.Cb1Icdf[63]);

        // WB: two sub-tables of 32 entries each.
        var wb = SilkNlsfCodebookTables.Wb;
        Equal((byte)0, wb.Cb1Icdf[31]);
        Equal((byte)0, wb.Cb1Icdf[63]);

        // Residual iCDFs: 8 sub-tables of 9 entries each; each sub-table ends with 0.
        for (int t = 0; t < 8; t++)
        {
            Equal((byte)0, nbMb.EcIcdf[9 * t + 8]);
            Equal((byte)0, wb.EcIcdf[9 * t + 8]);
        }
    }
}
