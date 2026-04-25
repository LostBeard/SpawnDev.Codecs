// Tests for Vp9CoefBands. Each table must (a) have the correct length,
// (b) hit each of the six bands at least once (a structurally valid
// band table is a prefix-monotone non-decreasing function from scan
// position to band 0..5), and (c) match the libvpx-pinned size of
// each band exactly. The size-of-each-band test catches a wide class
// of band-boundary copy errors at low cost.

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9CoefBands_4x4_Has16EntriesAcrossSixBands()
    {
        var t = Vp9CoefBands.CoefBand4x4;
        Equal(16, t.Length);
        // Band 0 starts at position 0 (DC), every band index appears
        // and they appear in non-decreasing order.
        Equal((byte)0, t[0]);
        for (int i = 1; i < 16; i++)
            True(t[i] >= t[i - 1], $"4x4 band must be non-decreasing at {i}");
        Equal((byte)5, t[15]);
    }

    [TestMethod]
    public void Vp9CoefBands_4x4_BandSizesMatchLibvpx()
    {
        // libvpx layout: bands 0..5 = sizes [1, 2, 3, 4, 3, 3].
        var counts = new int[6];
        foreach (var b in Vp9CoefBands.CoefBand4x4)
            counts[b]++;
        EqualInts(new[] { 1, 2, 3, 4, 3, 3 }, counts);
    }

    [TestMethod]
    public void Vp9CoefBands_8x8Plus_Has1024EntriesAcrossSixBands()
    {
        var t = Vp9CoefBands.CoefBandTrans8x8Plus;
        Equal(1024, t.Length);
        Equal((byte)0, t[0]);
        for (int i = 1; i < 1024; i++)
            True(t[i] >= t[i - 1], $"8x8plus band must be non-decreasing at {i}");
        Equal((byte)5, t[1023]);
    }

    [TestMethod]
    public void Vp9CoefBands_8x8Plus_BandSizesMatchLibvpx()
    {
        // libvpx layout: bands 0..5 = sizes [1, 2, 3, 4, 11, 1003].
        var counts = new int[6];
        foreach (var b in Vp9CoefBands.CoefBandTrans8x8Plus)
            counts[b]++;
        EqualInts(new[] { 1, 2, 3, 4, 11, 1003 }, counts);
    }

    [TestMethod]
    public void Vp9CoefBands_GetBand_DispatchesBySize()
    {
        // 4x4 hits CoefBand4x4 directly.
        for (int i = 0; i < 16; i++)
            Equal(Vp9CoefBands.CoefBand4x4[i], Vp9CoefBands.GetBand(Vp9TxSize.Tx4x4, i));

        // 8x8 / 16x16 / 32x32 share CoefBandTrans8x8Plus, each using
        // only its prefix.
        for (int i = 0; i < 64; i++)
            Equal(Vp9CoefBands.CoefBandTrans8x8Plus[i], Vp9CoefBands.GetBand(Vp9TxSize.Tx8x8, i));
        for (int i = 0; i < 256; i++)
            Equal(Vp9CoefBands.CoefBandTrans8x8Plus[i], Vp9CoefBands.GetBand(Vp9TxSize.Tx16x16, i));
        // 32x32: spot-check at the boundaries to keep test fast.
        Equal(Vp9CoefBands.CoefBandTrans8x8Plus[0],    Vp9CoefBands.GetBand(Vp9TxSize.Tx32x32, 0));
        Equal(Vp9CoefBands.CoefBandTrans8x8Plus[20],   Vp9CoefBands.GetBand(Vp9TxSize.Tx32x32, 20));
        Equal(Vp9CoefBands.CoefBandTrans8x8Plus[21],   Vp9CoefBands.GetBand(Vp9TxSize.Tx32x32, 21));
        Equal(Vp9CoefBands.CoefBandTrans8x8Plus[1023], Vp9CoefBands.GetBand(Vp9TxSize.Tx32x32, 1023));
    }

    [TestMethod]
    public void Vp9CoefBands_GetBand_ThrowsOnOutOfRange()
    {
        Throws<ArgumentOutOfRangeException>(() => Vp9CoefBands.GetBand(Vp9TxSize.Tx4x4, 16));
        Throws<ArgumentOutOfRangeException>(() => Vp9CoefBands.GetBand(Vp9TxSize.Tx8x8, 64));
        Throws<ArgumentOutOfRangeException>(() => Vp9CoefBands.GetBand(Vp9TxSize.Tx16x16, 256));
        Throws<ArgumentOutOfRangeException>(() => Vp9CoefBands.GetBand(Vp9TxSize.Tx32x32, 1024));
        Throws<ArgumentOutOfRangeException>(() => Vp9CoefBands.GetBand(Vp9TxSize.Tx4x4, -1));
    }
}
