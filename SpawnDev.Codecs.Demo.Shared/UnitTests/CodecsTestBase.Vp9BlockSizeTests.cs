// Tests for Vp9BlockSize + Vp9BlockSizes (slice 224).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9BlockSize_Constants_MatchLibvpx()
    {
        Equal(13, Vp9BlockSizes.Count);
        Equal(Vp9BlockSize.Block64x64, Vp9BlockSizes.Largest);
    }

    [TestMethod]
    public void Vp9BlockSize_LookupArrayLengths()
    {
        Equal(13, Vp9BlockSizes.WidthPx.Length);
        Equal(13, Vp9BlockSizes.HeightPx.Length);
        Equal(13, Vp9BlockSizes.Num8x8Wide.Length);
        Equal(13, Vp9BlockSizes.Num8x8High.Length);
        Equal(13, Vp9BlockSizes.Num4x4Wide.Length);
        Equal(13, Vp9BlockSizes.Num4x4High.Length);
        Equal(13, Vp9BlockSizes.MiWidthLog2.Length);
        Equal(13, Vp9BlockSizes.MiHeightLog2.Length);
        Equal(13, Vp9BlockSizes.NumPelsLog2.Length);
    }

    [TestMethod]
    public void Vp9BlockSize_Width_AllSizes()
    {
        Equal(4, Vp9BlockSizes.Width(Vp9BlockSize.Block4x4));
        Equal(4, Vp9BlockSizes.Width(Vp9BlockSize.Block4x8));
        Equal(8, Vp9BlockSizes.Width(Vp9BlockSize.Block8x4));
        Equal(8, Vp9BlockSizes.Width(Vp9BlockSize.Block8x8));
        Equal(8, Vp9BlockSizes.Width(Vp9BlockSize.Block8x16));
        Equal(16, Vp9BlockSizes.Width(Vp9BlockSize.Block16x8));
        Equal(16, Vp9BlockSizes.Width(Vp9BlockSize.Block16x16));
        Equal(16, Vp9BlockSizes.Width(Vp9BlockSize.Block16x32));
        Equal(32, Vp9BlockSizes.Width(Vp9BlockSize.Block32x16));
        Equal(32, Vp9BlockSizes.Width(Vp9BlockSize.Block32x32));
        Equal(32, Vp9BlockSizes.Width(Vp9BlockSize.Block32x64));
        Equal(64, Vp9BlockSizes.Width(Vp9BlockSize.Block64x32));
        Equal(64, Vp9BlockSizes.Width(Vp9BlockSize.Block64x64));
    }

    [TestMethod]
    public void Vp9BlockSize_Height_AllSizes()
    {
        Equal(4, Vp9BlockSizes.Height(Vp9BlockSize.Block4x4));
        Equal(8, Vp9BlockSizes.Height(Vp9BlockSize.Block4x8));
        Equal(4, Vp9BlockSizes.Height(Vp9BlockSize.Block8x4));
        Equal(8, Vp9BlockSizes.Height(Vp9BlockSize.Block8x8));
        Equal(16, Vp9BlockSizes.Height(Vp9BlockSize.Block8x16));
        Equal(8, Vp9BlockSizes.Height(Vp9BlockSize.Block16x8));
        Equal(16, Vp9BlockSizes.Height(Vp9BlockSize.Block16x16));
        Equal(32, Vp9BlockSizes.Height(Vp9BlockSize.Block16x32));
        Equal(16, Vp9BlockSizes.Height(Vp9BlockSize.Block32x16));
        Equal(32, Vp9BlockSizes.Height(Vp9BlockSize.Block32x32));
        Equal(64, Vp9BlockSizes.Height(Vp9BlockSize.Block32x64));
        Equal(32, Vp9BlockSizes.Height(Vp9BlockSize.Block64x32));
        Equal(64, Vp9BlockSizes.Height(Vp9BlockSize.Block64x64));
    }

    [TestMethod]
    public void Vp9BlockSize_MiWidthHeight_64x64Is8x8()
    {
        Equal(8, Vp9BlockSizes.MiWidth(Vp9BlockSize.Block64x64));
        Equal(8, Vp9BlockSizes.MiHeight(Vp9BlockSize.Block64x64));
        Equal(1, Vp9BlockSizes.MiWidth(Vp9BlockSize.Block8x8));
        Equal(1, Vp9BlockSizes.MiHeight(Vp9BlockSize.Block8x8));
    }

    [TestMethod]
    public void Vp9BlockSize_B4x4WidthHeight_4x4Is1x1()
    {
        Equal(1, Vp9BlockSizes.B4x4Width(Vp9BlockSize.Block4x4));
        Equal(1, Vp9BlockSizes.B4x4Height(Vp9BlockSize.Block4x4));
        Equal(16, Vp9BlockSizes.B4x4Width(Vp9BlockSize.Block64x64));
        Equal(16, Vp9BlockSizes.B4x4Height(Vp9BlockSize.Block64x64));
    }

    [TestMethod]
    public void Vp9BlockSize_NumPelsLog2_64x64Is12()
    {
        // 64x64 = 4096 pels = 2^12.
        Equal(12, Vp9BlockSizes.NumPelsLog2[(int)Vp9BlockSize.Block64x64]);
        // 4x4 = 16 pels = 2^4.
        Equal(4, Vp9BlockSizes.NumPelsLog2[(int)Vp9BlockSize.Block4x4]);
        // 16x16 = 256 pels = 2^8.
        Equal(8, Vp9BlockSizes.NumPelsLog2[(int)Vp9BlockSize.Block16x16]);
        // 32x32 = 1024 = 2^10.
        Equal(10, Vp9BlockSizes.NumPelsLog2[(int)Vp9BlockSize.Block32x32]);
    }

    [TestMethod]
    public void Vp9BlockSize_IsSquare()
    {
        Equal(true, Vp9BlockSizes.IsSquare(Vp9BlockSize.Block4x4));
        Equal(true, Vp9BlockSizes.IsSquare(Vp9BlockSize.Block8x8));
        Equal(true, Vp9BlockSizes.IsSquare(Vp9BlockSize.Block16x16));
        Equal(true, Vp9BlockSizes.IsSquare(Vp9BlockSize.Block32x32));
        Equal(true, Vp9BlockSizes.IsSquare(Vp9BlockSize.Block64x64));
        Equal(false, Vp9BlockSizes.IsSquare(Vp9BlockSize.Block4x8));
        Equal(false, Vp9BlockSizes.IsSquare(Vp9BlockSize.Block8x4));
        Equal(false, Vp9BlockSizes.IsSquare(Vp9BlockSize.Block16x32));
        Equal(false, Vp9BlockSizes.IsSquare(Vp9BlockSize.Block64x32));
    }

    [TestMethod]
    public void Vp9BlockSize_NumPelsLog2_DerivedFromWidthHeight()
    {
        // Verify NumPelsLog2 is consistent with Width * Height = 2^NumPelsLog2.
        for (int i = 0; i < Vp9BlockSizes.Count; i++)
        {
            int pels = Vp9BlockSizes.WidthPx[i] * Vp9BlockSizes.HeightPx[i];
            int expected = 0;
            for (int p = pels; p > 1; p >>= 1) expected++;
            Equal(expected, Vp9BlockSizes.NumPelsLog2[i]);
        }
    }

    [TestMethod]
    public void Vp9BlockSize_B4x4WidthLog2_MatchesNum4x4Wide()
    {
        // b_width_log2_lookup[i] == log2(num_4x4_blocks_wide_lookup[i]).
        Equal(13, Vp9BlockSizes.B4x4WidthLog2.Length);
        for (int i = 0; i < Vp9BlockSizes.Count; i++)
        {
            int expected = 0;
            for (int n = Vp9BlockSizes.Num4x4Wide[i]; n > 1; n >>= 1) expected++;
            Equal(expected, Vp9BlockSizes.B4x4WidthLog2[i]);
        }
    }

    [TestMethod]
    public void Vp9BlockSize_B4x4HeightLog2_MatchesNum4x4High()
    {
        // b_height_log2_lookup[i] == log2(num_4x4_blocks_high_lookup[i]).
        Equal(13, Vp9BlockSizes.B4x4HeightLog2.Length);
        for (int i = 0; i < Vp9BlockSizes.Count; i++)
        {
            int expected = 0;
            for (int n = Vp9BlockSizes.Num4x4High[i]; n > 1; n >>= 1) expected++;
            Equal(expected, Vp9BlockSizes.B4x4HeightLog2[i]);
        }
    }

    [TestMethod]
    public void Vp9BlockSize_B4x4Log2_64x64Is4()
    {
        Equal(4, Vp9BlockSizes.B4x4WidthLog2[(int)Vp9BlockSize.Block64x64]);
        Equal(4, Vp9BlockSizes.B4x4HeightLog2[(int)Vp9BlockSize.Block64x64]);
        Equal(0, Vp9BlockSizes.B4x4WidthLog2[(int)Vp9BlockSize.Block4x4]);
        Equal(0, Vp9BlockSizes.B4x4HeightLog2[(int)Vp9BlockSize.Block4x4]);
    }
}
