using SpawnDev.Codecs.Audio.Opus;
using SpawnDev.Codecs.Audio.Opus.Celt;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for the initial CELT scaffolding: constants, eband table, and the
/// <see cref="CeltMode.Create"/> factory. CELT synthesis itself (IMDCT, band
/// allocation, PVQ) lands in subsequent slices.
/// </summary>
public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void CeltConstants_Eband5Ms_MatchesLibopus()
    {
        // From libopus celt/modes.c: eband5ms[] has 22 entries for 21 bands.
        Equal(22, CeltConstants.Eband5Ms.Length);
        Equal(CeltConstants.NB_BANDS_FULLBAND, CeltConstants.Eband5Ms.Length - 1);

        // First entry is 0, last is 100 (corresponds to 20 kHz in Q200Hz bin-space).
        Equal((short)0, CeltConstants.Eband5Ms[0]);
        Equal((short)100, CeltConstants.Eband5Ms[21]);

        // Monotonically increasing.
        for (int i = 1; i < CeltConstants.Eband5Ms.Length; i++)
        {
            True(CeltConstants.Eband5Ms[i] > CeltConstants.Eband5Ms[i - 1],
                $"eband5ms[{i}] should be > eband5ms[{i - 1}]");
        }
    }

    [TestMethod]
    public void CeltConstants_BandCounts_MatchLibopus()
    {
        Equal(13, CeltConstants.NB_BANDS_NB);
        Equal(17, CeltConstants.NB_BANDS_WB);
        Equal(19, CeltConstants.NB_BANDS_SWB);
        Equal(21, CeltConstants.NB_BANDS_FULLBAND);
    }

    [TestMethod]
    public void CeltConstants_FrameSizes_48kHz()
    {
        Equal(120, CeltConstants.FRAME_SIZE_2_5MS);
        Equal(240, CeltConstants.FRAME_SIZE_5MS);
        Equal(480, CeltConstants.FRAME_SIZE_10MS);
        Equal(960, CeltConstants.FRAME_SIZE_20MS);
    }

    [TestMethod]
    public void CeltMode_Create_FullbandTwentyMs_PopulatesExpectedFields()
    {
        var mode = CeltMode.Create(CeltConstants.FRAME_SIZE_20MS, CeltConstants.NB_BANDS_FULLBAND);
        Equal(48000, mode.SampleRateHz);
        Equal(960, mode.FrameSize);
        Equal(21, mode.NbEBands);
        Equal(0, mode.StartBand);
        Equal(21, mode.EndBand);
        // Should reference the same eband table.
        if (!ReferenceEquals(mode.EBands, CeltConstants.Eband5Ms))
            throw new Exception("mode.EBands should reference CeltConstants.Eband5Ms");
    }

    [TestMethod]
    public void CeltMode_Create_NarrowbandTenMs_ClampsEndBand()
    {
        var mode = CeltMode.Create(CeltConstants.FRAME_SIZE_10MS, CeltConstants.NB_BANDS_NB);
        Equal(480, mode.FrameSize);
        Equal(13, mode.EndBand);
        // NbEBands is still 21 (the table dimension) even though only bands [0, 13) are used.
        Equal(21, mode.NbEBands);
    }

    [TestMethod]
    public void CeltMode_Create_InvalidFrameSize_Throws()
    {
        Throws<ArgumentException>(() => CeltMode.Create(100, 21));
        Throws<ArgumentException>(() => CeltMode.Create(500, 21));
    }

    [TestMethod]
    public void CeltMode_Create_InvalidEndBand_Throws()
    {
        Throws<ArgumentException>(() => CeltMode.Create(960, 0));
        Throws<ArgumentException>(() => CeltMode.Create(960, 22));
    }

    [TestMethod]
    public void CeltMode_EndBandForBandwidth_MapsCorrectly()
    {
        Equal(CeltConstants.NB_BANDS_NB, CeltMode.EndBandForBandwidth(OpusBandwidth.Narrowband));
        Equal(CeltConstants.NB_BANDS_WB, CeltMode.EndBandForBandwidth(OpusBandwidth.Wideband));
        Equal(CeltConstants.NB_BANDS_SWB, CeltMode.EndBandForBandwidth(OpusBandwidth.Superwideband));
        Equal(CeltConstants.NB_BANDS_FULLBAND, CeltMode.EndBandForBandwidth(OpusBandwidth.Fullband));
    }

    [TestMethod]
    public void CeltMode_EndBandForBandwidth_MediumbandThrows()
    {
        // CELT does not operate at MB (libopus uses SILK for MB).
        Throws<ArgumentException>(() => CeltMode.EndBandForBandwidth(OpusBandwidth.Mediumband));
    }

    [TestMethod]
    public void CeltMode_Create_PopulatesMdctGeometryFields()
    {
        // The new fields added for the future hand-port (overlap, maxLM,
        // nbShortMdcts, shortMdctSize) must match the libopus standard
        // 48 kHz CELT mode geometry.
        var mode = CeltMode.Create(CeltConstants.FRAME_SIZE_20MS, CeltConstants.NB_BANDS_FULLBAND);
        Equal(120, mode.Overlap);
        Equal(3, mode.MaxLM);
        Equal(8, mode.NbShortMdcts);
        Equal(120, mode.ShortMdctSize);
        Equal(21, mode.EffectiveEBands);
    }

    [TestMethod]
    public void CeltConstants_FixedPoint_MatchesLibopus()
    {
        // Spot-check that the canonical libopus fixed-point constants are
        // present and have their RFC-defined values.
        Equal(32767, CeltConstants.Q15ONE);
        Equal(12, CeltConstants.SIG_SHIFT);
        Equal(10, CeltConstants.DB_SHIFT);
        Equal(1024, CeltConstants.COMBFILTER_MAXPERIOD);
        Equal(15, CeltConstants.COMBFILTER_MINPERIOD);
        Equal(2048, CeltConstants.DECODE_BUFFER_SIZE);
        Equal(24, CeltConstants.LPC_ORDER);
        Equal(720, CeltConstants.PLC_PITCH_LAG_MAX);
        Equal(100, CeltConstants.PLC_PITCH_LAG_MIN);
        Equal(11, CeltConstants.BITALLOC_SIZE);
        Equal(40, CeltConstants.MAX_PSEUDO);
        Equal(128, CeltConstants.CELT_MAX_PULSES);
        Equal(8, CeltConstants.MAX_FINE_BITS);
        Equal(21, CeltConstants.FINE_OFFSET);
        Equal(4, CeltConstants.QTHETA_OFFSET);
    }
}
