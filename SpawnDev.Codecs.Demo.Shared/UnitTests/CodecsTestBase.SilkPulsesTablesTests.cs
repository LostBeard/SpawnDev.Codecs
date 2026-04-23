using SpawnDev.Codecs.Audio.Opus.Silk;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Sanity tests for the pulses-block iCDF tables ported from libopus
/// silk/tables_pulses_per_block.c + silk/tables_other.c. Verifies table shapes,
/// row offsets, and the row-selector helpers. Bit-exact content is checked by
/// spot-sampling first/last/mid values per row.
/// </summary>
public abstract partial class CodecsTestBase
{
    // -------- Shape and constants --------

    [TestMethod]
    public void PulsesTables_Constants_MatchLibopus()
    {
        Equal(4, SilkConstants.LOG2_SHELL_CODEC_FRAME_LENGTH);
        Equal(16, SilkConstants.SHELL_CODEC_FRAME_LENGTH);
        Equal(20, SilkConstants.MAX_NB_SHELL_BLOCKS);
        Equal(16, SilkConstants.SILK_MAX_PULSES);
        Equal(10, SilkConstants.N_RATE_LEVELS);
    }

    [TestMethod]
    public void PulsesTables_RateLevels_HasTwoRowsOfNine()
    {
        Equal(18, SilkIcdfTables.RateLevels.Length);
        Equal(9, SilkIcdfTables.RateLevelsEntriesPerType);

        // First row (non-voiced): starts 241, 190, ..., ends with 0.
        Equal((byte)241, SilkIcdfTables.RateLevels[0]);
        Equal((byte)190, SilkIcdfTables.RateLevels[1]);
        Equal((byte)0, SilkIcdfTables.RateLevels[8]);

        // Second row (voiced): starts 223, ends with 0.
        Equal((byte)223, SilkIcdfTables.RateLevels[9]);
        Equal((byte)0, SilkIcdfTables.RateLevels[17]);
    }

    [TestMethod]
    public void PulsesTables_RateLevels_OffsetMatchesSignalTypeShift()
    {
        // signalType >> 1 picks the row: 0 or 1 for non-voiced (inactive/unvoiced), 1 for voiced.
        Equal(0, SilkIcdfTables.RateLevelsOffset(0)); // inactive
        Equal(0, SilkIcdfTables.RateLevelsOffset(1)); // unvoiced
        Equal(9, SilkIcdfTables.RateLevelsOffset(2)); // voiced
    }

    [TestMethod]
    public void PulsesTables_PulsesPerBlock_HasTenRowsOfEighteen()
    {
        Equal(180, SilkIcdfTables.PulsesPerBlock.Length);
        Equal(18, SilkIcdfTables.PulsesPerBlockEntriesPerRow);
        Equal(SilkConstants.N_RATE_LEVELS,
              SilkIcdfTables.PulsesPerBlock.Length / SilkIcdfTables.PulsesPerBlockEntriesPerRow);

        // First row (rate level 0): first/last values.
        Equal((byte)125, SilkIcdfTables.PulsesPerBlock[0]);
        Equal((byte)0, SilkIcdfTables.PulsesPerBlock[17]);

        // Last row (rate level 9): first/last.
        Equal((byte)255, SilkIcdfTables.PulsesPerBlock[9 * 18]);
        Equal((byte)0, SilkIcdfTables.PulsesPerBlock[9 * 18 + 17]);

        // Middle row (rate level 5) starts 249, ends 0.
        Equal((byte)249, SilkIcdfTables.PulsesPerBlock[5 * 18]);
        Equal((byte)0, SilkIcdfTables.PulsesPerBlock[5 * 18 + 17]);
    }

    [TestMethod]
    public void PulsesTables_PulsesPerBlock_AllRowsEndInZero()
    {
        // Each iCDF must terminate in 0 so the range coder can reach its final state.
        for (int row = 0; row < SilkConstants.N_RATE_LEVELS; row++)
        {
            int offset = SilkIcdfTables.PulsesPerBlockOffset(row);
            Equal((byte)0, SilkIcdfTables.PulsesPerBlock[offset + 17], $"row {row} terminal");
        }
    }

    [TestMethod]
    public void PulsesTables_Sign_HasSixRowsOfSeven()
    {
        Equal(42, SilkIcdfTables.Sign.Length);
        Equal(7, SilkIcdfTables.SignEntriesPerRow);

        // First row starts 254, ends 99.
        Equal((byte)254, SilkIcdfTables.Sign[0]);
        Equal((byte)99, SilkIcdfTables.Sign[6]);

        // Last row starts 248, ends 102.
        Equal((byte)248, SilkIcdfTables.Sign[35]);
        Equal((byte)102, SilkIcdfTables.Sign[41]);
    }

    [TestMethod]
    public void PulsesTables_Sign_OffsetMatchesRowIndexFormula()
    {
        // row_index = quantOffsetType + 2 * signalType
        // row 0: signalType=0, quantOffsetType=0
        // row 1: signalType=0, quantOffsetType=1
        // row 2: signalType=1, quantOffsetType=0
        // row 3: signalType=1, quantOffsetType=1
        // row 4: signalType=2, quantOffsetType=0
        // row 5: signalType=2, quantOffsetType=1
        Equal(0, SilkIcdfTables.SignOffset(0, 0));
        Equal(7, SilkIcdfTables.SignOffset(0, 1));
        Equal(14, SilkIcdfTables.SignOffset(1, 0));
        Equal(21, SilkIcdfTables.SignOffset(1, 1));
        Equal(28, SilkIcdfTables.SignOffset(2, 0));
        Equal(35, SilkIcdfTables.SignOffset(2, 1));
    }

    [TestMethod]
    public void PulsesTables_Lsb_IsTwoSymbolBinaryIcdf()
    {
        Equal(2, SilkIcdfTables.Lsb.Length);
        Equal((byte)120, SilkIcdfTables.Lsb[0]);
        Equal((byte)0, SilkIcdfTables.Lsb[1]);
    }
}
