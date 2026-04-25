// Tests for Vp9SubPelFilters (slice 240).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9SubPelFilters_Constants_MatchLibvpx()
    {
        Equal(4, Vp9SubPelFilters.SubPelBits);
        Equal(16, Vp9SubPelFilters.SubPelShifts);
        Equal(8, Vp9SubPelFilters.SubPelTaps);
        Equal(7, Vp9SubPelFilters.FilterBits);
    }

    [TestMethod]
    public void Vp9SubPelFilters_TableLengths_AreUniform()
    {
        Equal(128, Vp9SubPelFilters.EightTap.Length);
        Equal(128, Vp9SubPelFilters.EightTapSmooth.Length);
        Equal(128, Vp9SubPelFilters.EightTapSharp.Length);
        Equal(128, Vp9SubPelFilters.Bilinear.Length);
    }

    [TestMethod]
    public void Vp9SubPelFilters_AllFilters_AtSubPel0_AreIdentity()
    {
        // Position 0 of every kernel is { 0, 0, 0, 128, 0, 0, 0, 0 }
        // (the pure integer-pel sample passes through).
        VerifyIdentity(Vp9SubPelFilters.EightTap);
        VerifyIdentity(Vp9SubPelFilters.EightTapSmooth);
        VerifyIdentity(Vp9SubPelFilters.EightTapSharp);
        VerifyIdentity(Vp9SubPelFilters.Bilinear);

        static void VerifyIdentity(short[] filter)
        {
            Equal((short)0, filter[0]);
            Equal((short)0, filter[1]);
            Equal((short)0, filter[2]);
            Equal((short)128, filter[3]);
            Equal((short)0, filter[4]);
            Equal((short)0, filter[5]);
            Equal((short)0, filter[6]);
            Equal((short)0, filter[7]);
        }
    }

    [TestMethod]
    public void Vp9SubPelFilters_AllFilters_AllRowsSumToFilterScale()
    {
        // Every 8-tap row sums to 128 (FILTER_BITS = 7 -> 2^7 = 128).
        VerifySumsTo128(Vp9SubPelFilters.EightTap);
        VerifySumsTo128(Vp9SubPelFilters.EightTapSmooth);
        VerifySumsTo128(Vp9SubPelFilters.EightTapSharp);
        VerifySumsTo128(Vp9SubPelFilters.Bilinear);

        static void VerifySumsTo128(short[] filter)
        {
            for (int p = 0; p < Vp9SubPelFilters.SubPelShifts; p++)
            {
                int sum = 0;
                for (int t = 0; t < Vp9SubPelFilters.SubPelTaps; t++)
                    sum += filter[p * Vp9SubPelFilters.SubPelTaps + t];
                Equal(128, sum);
            }
        }
    }

    [TestMethod]
    public void Vp9SubPelFilters_EightTap_SubPel8_IsCenterSymmetric()
    {
        // Sub-pel 8 is exactly halfway between integer samples; the
        // libvpx 8-tap regular filter is symmetric there.
        // Row: { -1,  6, -19,  78,  78, -19,  6, -1 }
        var row = Vp9SubPelFilters.GetRow(Vp9InterpFilter.EightTap, 8);
        Equal(8, row.Length);
        Equal((short)-1, row[0]); Equal((short)-1, row[7]);
        Equal((short)6,  row[1]); Equal((short)6,  row[6]);
        Equal((short)-19, row[2]); Equal((short)-19, row[5]);
        Equal((short)78, row[3]); Equal((short)78, row[4]);
    }

    [TestMethod]
    public void Vp9SubPelFilters_Bilinear_SubPel8_IsHalfHalf()
    {
        // Bilinear at half-pel: { 0, 0, -12, 76, 76, -12, 0, 0 }
        var row = Vp9SubPelFilters.GetRow(Vp9InterpFilter.Bilinear, 8);
        Equal((short)0, row[0]); Equal((short)0, row[1]);
        Equal((short)-12, row[2]); Equal((short)76, row[3]);
        Equal((short)76, row[4]); Equal((short)-12, row[5]);
        Equal((short)0, row[6]); Equal((short)0, row[7]);
    }

    [TestMethod]
    public void Vp9SubPelFilters_GetFilter_RejectsSwitchable()
    {
        // Switchable is a per-block selector at the frame level;
        // there is no tap table for it.
        Throws<ArgumentOutOfRangeException>(() =>
            Vp9SubPelFilters.GetFilter(Vp9InterpFilter.Switchable));
    }

    [TestMethod]
    public void Vp9SubPelFilters_GetRow_RejectsOutOfRangeSubPel()
    {
        Throws<ArgumentOutOfRangeException>(() =>
            Vp9SubPelFilters.GetRow(Vp9InterpFilter.EightTap, 16));
        Throws<ArgumentOutOfRangeException>(() =>
            Vp9SubPelFilters.GetRow(Vp9InterpFilter.EightTap, -1));
    }
}
