using SpawnDev.Codecs.Audio.Opus.Silk;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for <see cref="SilkPitchDecoder.ComputeLags"/> - the expansion of the
/// decoded (lagIndex, contourIndex) pair into per-subframe pitch lag samples
/// via the four <see cref="SilkPitchContourTables"/>. Verifies clamping at
/// <c>[PE_MIN_LAG_MS * fsKHz, PE_MAX_LAG_MS * fsKHz]</c> and bit-exact lookup
/// against the libopus codebooks.
/// </summary>
public abstract partial class CodecsTestBase
{
    // -------- Stage-2 NB 20 ms --------

    [TestMethod]
    public void ComputeLags_Nb20Ms_ContourZero_AllSubframesEqualBaseLag()
    {
        // silk_CB_lags_stage2[k][0] = 0 for all k, so every subframe should be baseLag.
        int fsKHz = 8;
        short lagIndex = 10; // -> baseLag = 2*8 + 10 = 26
        int[] lags = new int[4];
        SilkPitchDecoder.ComputeLags(lags, lagIndex, contourIndex: 0, fsKHz, nbSubfr: 4);
        for (int k = 0; k < 4; k++) Equal(26, lags[k], $"subframe {k}");
    }

    [TestMethod]
    public void ComputeLags_Nb20Ms_ContourOne_AppliesExpectedDeltas()
    {
        // silk_CB_lags_stage2[*][1] = {2, 1, 0, -1}
        int fsKHz = 8;
        short lagIndex = 10;
        int[] lags = new int[4];
        SilkPitchDecoder.ComputeLags(lags, lagIndex, contourIndex: 1, fsKHz, nbSubfr: 4);
        Equal(26 + 2, lags[0]);
        Equal(26 + 1, lags[1]);
        Equal(26 + 0, lags[2]);
        Equal(26 - 1, lags[3]);
    }

    [TestMethod]
    public void ComputeLags_Nb20Ms_AllContours_WithinLagRange()
    {
        // For a mid-range lagIndex, every contour should produce lags strictly within [minLag, maxLag].
        int fsKHz = 8;
        int minLag = SilkConstants.PE_MIN_LAG_MS * fsKHz;
        int maxLag = SilkConstants.PE_MAX_LAG_MS * fsKHz;
        short lagIndex = 16; // baseLag = minLag + 16 = 32
        int[] lags = new int[4];
        for (int cb = 0; cb < SilkConstants.PE_NB_CBKS_STAGE2_EXT; cb++)
        {
            SilkPitchDecoder.ComputeLags(lags, lagIndex, (sbyte)cb, fsKHz, 4);
            for (int k = 0; k < 4; k++)
            {
                True(lags[k] >= minLag, $"contour {cb} subframe {k}: lag {lags[k]} < minLag {minLag}");
                True(lags[k] <= maxLag, $"contour {cb} subframe {k}: lag {lags[k]} > maxLag {maxLag}");
            }
        }
    }

    // -------- Stage-2 NB 10 ms --------

    [TestMethod]
    public void ComputeLags_Nb10Ms_ContourZero_BothSubframesEqualBaseLag()
    {
        // silk_CB_lags_stage2_10_ms[k][0] = 0 for both rows.
        int[] lags = new int[2];
        SilkPitchDecoder.ComputeLags(lags, lagIndex: 5, contourIndex: 0, fsKHz: 8, nbSubfr: 2);
        Equal(21, lags[0]);
        Equal(21, lags[1]);
    }

    [TestMethod]
    public void ComputeLags_Nb10Ms_ContourOne_FirstSubframeOffsetPlusOne()
    {
        // silk_CB_lags_stage2_10_ms[*][1] = {1, 0}
        int[] lags = new int[2];
        SilkPitchDecoder.ComputeLags(lags, lagIndex: 5, contourIndex: 1, fsKHz: 8, nbSubfr: 2);
        Equal(22, lags[0]);
        Equal(21, lags[1]);
    }

    [TestMethod]
    public void ComputeLags_Nb10Ms_ContourTwo_SecondSubframeOffsetPlusOne()
    {
        // silk_CB_lags_stage2_10_ms[*][2] = {0, 1}
        int[] lags = new int[2];
        SilkPitchDecoder.ComputeLags(lags, lagIndex: 5, contourIndex: 2, fsKHz: 8, nbSubfr: 2);
        Equal(21, lags[0]);
        Equal(22, lags[1]);
    }

    // -------- Stage-3 non-NB 20 ms --------

    [TestMethod]
    public void ComputeLags_Wb20Ms_ContourZero_AllSubframesEqualBaseLag()
    {
        // silk_CB_lags_stage3[k][0] = 0 for all k.
        int[] lags = new int[4];
        SilkPitchDecoder.ComputeLags(lags, lagIndex: 50, contourIndex: 0, fsKHz: 16, nbSubfr: 4);
        int expected = 50 + SilkConstants.PE_MIN_LAG_MS * 16; // 50 + 32 = 82
        for (int k = 0; k < 4; k++) Equal(expected, lags[k]);
    }

    [TestMethod]
    public void ComputeLags_Wb20Ms_LastContour_AppliesExpectedDeltas()
    {
        // silk_CB_lags_stage3[*][33] = {-9, -3,  3,  9}  (taking the last column of each row)
        int[] lags = new int[4];
        SilkPitchDecoder.ComputeLags(lags, lagIndex: 100, contourIndex: 33, fsKHz: 16, nbSubfr: 4);
        int baseLag = 100 + SilkConstants.PE_MIN_LAG_MS * 16; // 132
        Equal(baseLag - 9, lags[0]);
        Equal(baseLag - 3, lags[1]);
        Equal(baseLag + 3, lags[2]);
        Equal(baseLag + 9, lags[3]);
    }

    // -------- Stage-3 non-NB 10 ms --------

    [TestMethod]
    public void ComputeLags_Mb10Ms_LastContour_AppliesExpectedDeltas()
    {
        // silk_CB_lags_stage3_10_ms[*][11] = {-3, 3}
        int[] lags = new int[2];
        SilkPitchDecoder.ComputeLags(lags, lagIndex: 30, contourIndex: 11, fsKHz: 12, nbSubfr: 2);
        int baseLag = 30 + SilkConstants.PE_MIN_LAG_MS * 12; // 54
        Equal(baseLag - 3, lags[0]);
        Equal(baseLag + 3, lags[1]);
    }

    // -------- Clamping --------

    [TestMethod]
    public void ComputeLags_Nb20Ms_SmallLagWithNegativeDelta_ClampsToMinLag()
    {
        // lagIndex = 0 -> baseLag = minLag = 16 at fsKHz=8. Contour 3 has a -1 at subframe 3:
        // silk_CB_lags_stage2[3][3] = 1 -> no clamp, but contour 2 has stage2[0][2] = -1:
        // baseLag + (-1) = 15 < minLag=16 -> should clamp to 16.
        int fsKHz = 8;
        int minLag = SilkConstants.PE_MIN_LAG_MS * fsKHz;
        int[] lags = new int[4];
        SilkPitchDecoder.ComputeLags(lags, lagIndex: 0, contourIndex: 2, fsKHz, nbSubfr: 4);
        // stage2[*][2] = {-1, 0, 1, 2}
        Equal(minLag, lags[0]);     // clamped from 15
        Equal(minLag + 0, lags[1]);
        Equal(minLag + 1, lags[2]);
        Equal(minLag + 2, lags[3]);
    }

    [TestMethod]
    public void ComputeLags_Nb20Ms_LargeLagWithPositiveDelta_ClampsToMaxLag()
    {
        // lagIndex at the top of the iCDF (31) -> baseLag = minLag + 31 = 47. maxLag = 144.
        // Well within range; won't clamp. But if we artificially push lagIndex past the
        // coarse-iCDF range (library never does this in practice), we should clamp.
        // Here we just confirm the upper clamp works by using fsKHz=8 where lagIndex can
        // make baseLag approach maxLag. With stage2[3][1] = -1, a value at the top:
        int fsKHz = 8;
        int maxLag = SilkConstants.PE_MAX_LAG_MS * fsKHz;
        short aggressiveLag = (short)(maxLag - SilkConstants.PE_MIN_LAG_MS * fsKHz); // -> baseLag = maxLag
        int[] lags = new int[4];
        // Use contour 1 which has stage2[0][1] = 2 (would push baseLag + 2 > maxLag).
        SilkPitchDecoder.ComputeLags(lags, aggressiveLag, contourIndex: 1, fsKHz, nbSubfr: 4);
        Equal(maxLag, lags[0]); // clamped
    }

    // -------- Arg validation --------

    [TestMethod]
    public void ComputeLags_UnsupportedFsKHz_Throws()
    {
        int[] lags = new int[4];
        Throws<ArgumentException>(() =>
            SilkPitchDecoder.ComputeLags(lags, 0, 0, fsKHz: 11, nbSubfr: 4));
    }

    [TestMethod]
    public void ComputeLags_InvalidNbSubfr_Throws()
    {
        int[] lags = new int[4];
        Throws<ArgumentException>(() =>
            SilkPitchDecoder.ComputeLags(lags, 0, 0, fsKHz: 16, nbSubfr: 3));
    }

    [TestMethod]
    public void ComputeLags_ContourIndexOutOfRange_Throws()
    {
        int[] lags = new int[4];
        // fsKHz=16, nbSubfr=4 -> cbSize = PE_NB_CBKS_STAGE3_MAX = 34. 34 is out of range.
        Throws<ArgumentOutOfRangeException>(() =>
            SilkPitchDecoder.ComputeLags(lags, 0, 34, fsKHz: 16, nbSubfr: 4));
    }

    [TestMethod]
    public void ComputeLags_OutputTooSmall_Throws()
    {
        int[] small = new int[3];
        Throws<ArgumentException>(() =>
            SilkPitchDecoder.ComputeLags(small, 0, 0, fsKHz: 16, nbSubfr: 4));
    }
}
