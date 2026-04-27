// Tests for Vp9CoefContext (slice 148). Verifies the pt_energy_class
// table matches libvpx and GetCoefContext produces the expected 0/1/2
// values for representative neighbor + token-cache states.

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9CoefContext_PtEnergyClass_HasCorrectLengthAndValues()
    {
        // libvpx vp9_pt_energy_class = { 0, 1, 2, 3, 3, 4, 4, 5, 5, 5, 5, 5 }.
        Equal(12, Vp9CoefContext.PtEnergyClass.Length);
        Equal((byte)0, Vp9CoefContext.PtEnergyClass[(int)Vp9CoefToken.Zero]);
        Equal((byte)1, Vp9CoefContext.PtEnergyClass[(int)Vp9CoefToken.One]);
        Equal((byte)2, Vp9CoefContext.PtEnergyClass[(int)Vp9CoefToken.Two]);
        Equal((byte)3, Vp9CoefContext.PtEnergyClass[(int)Vp9CoefToken.Three]);
        Equal((byte)3, Vp9CoefContext.PtEnergyClass[(int)Vp9CoefToken.Four]);
        Equal((byte)4, Vp9CoefContext.PtEnergyClass[(int)Vp9CoefToken.Category1]);
        Equal((byte)4, Vp9CoefContext.PtEnergyClass[(int)Vp9CoefToken.Category2]);
        Equal((byte)5, Vp9CoefContext.PtEnergyClass[(int)Vp9CoefToken.Category3]);
        Equal((byte)5, Vp9CoefContext.PtEnergyClass[(int)Vp9CoefToken.Category4]);
        Equal((byte)5, Vp9CoefContext.PtEnergyClass[(int)Vp9CoefToken.Category5]);
        Equal((byte)5, Vp9CoefContext.PtEnergyClass[(int)Vp9CoefToken.Category6]);
        Equal((byte)0, Vp9CoefContext.PtEnergyClass[(int)Vp9CoefToken.Eob]);
    }

    [TestMethod]
    public void Vp9CoefContext_GetCoefContext_ScanPos0HasAllZeroNeighbors_ReturnsZero()
    {
        // Default 4x4 scan: scan position 0 has neighbors (0, 0); the
        // token cache starts all-zeros so e0 + e1 = 0; context = (1 + 0
        // + 0) >> 1 = 0.
        var neighbors = Vp9NeighborTables.DefaultScan4x4Neighbors.AsSpan();
        Span<byte> cache = stackalloc byte[16];
        int ctx = Vp9CoefContext.GetCoefContext(neighbors, cache, scanPos: 0);
        Equal(0, ctx);
    }

    [TestMethod]
    public void Vp9CoefContext_GetCoefContext_OneEnergyClass1Neighbor_ReturnsOne()
    {
        // Default 4x4 scan: scan position 3 has neighbors (1, 4) per
        // libvpx default_scan_4x4_neighbors. If raster position 1 has
        // energy class 1 (One token) and raster position 4 is zero,
        // context = (1 + 1 + 0) >> 1 = 1.
        var neighbors = Vp9NeighborTables.DefaultScan4x4Neighbors.AsSpan();
        Span<byte> cache = stackalloc byte[16];
        cache[1] = 1; // raster pos 1 was a One token
        int ctx = Vp9CoefContext.GetCoefContext(neighbors, cache, scanPos: 3);
        Equal(1, ctx);
    }

    [TestMethod]
    public void Vp9CoefContext_GetCoefContext_BothNeighborsHighEnergy_ReturnsFive()
    {
        // Default 4x4 scan position 3 has neighbors (1, 4). With both
        // neighbors at energy class 5 (Cat3+), the raw context value is
        // (1 + 5 + 5) >> 1 = 5. libvpx vp9_scan.h get_coef_context does
        // NOT clamp; the prob table for bands 1..5 has 6 ctx columns
        // (BAND_COEFF_CONTEXTS(band) = (band==0 ? 3 : 6)) so ctx 0..5
        // are all valid. An earlier version of this test asserted a cap
        // at 2 - that cap was the AC variance under-decode bug.
        var neighbors = Vp9NeighborTables.DefaultScan4x4Neighbors.AsSpan();
        Span<byte> cache = stackalloc byte[16];
        cache[1] = 5;
        cache[4] = 5;
        int ctx = Vp9CoefContext.GetCoefContext(neighbors, cache, scanPos: 3);
        Equal(5, ctx);
    }

    [TestMethod]
    public void Vp9CoefContext_GetCoefContext_BothEnergyClass2_ReturnsContext2()
    {
        // (1 + 2 + 2) >> 1 = 2 - exact boundary case at the cap.
        var neighbors = Vp9NeighborTables.DefaultScan4x4Neighbors.AsSpan();
        Span<byte> cache = stackalloc byte[16];
        cache[1] = 2;
        cache[4] = 2;
        int ctx = Vp9CoefContext.GetCoefContext(neighbors, cache, scanPos: 3);
        Equal(2, ctx);
    }

    [TestMethod]
    public void Vp9CoefContext_GetCoefContext_BothEnergyClass1_ReturnsContext1()
    {
        // (1 + 1 + 1) >> 1 = 1 - just below the cap.
        var neighbors = Vp9NeighborTables.DefaultScan4x4Neighbors.AsSpan();
        Span<byte> cache = stackalloc byte[16];
        cache[1] = 1;
        cache[4] = 1;
        int ctx = Vp9CoefContext.GetCoefContext(neighbors, cache, scanPos: 3);
        Equal(1, ctx);
    }

    [TestMethod]
    public void Vp9CoefContext_GetCoefContext_RejectsOutOfRangeScanPos()
    {
        var neighbors = Vp9NeighborTables.DefaultScan4x4Neighbors.AsSpan();
        Span<byte> cache = stackalloc byte[16];
        // 4x4 neighbors has 17 pairs (16 scan pos + boundary), so
        // scanPos 17 is out of range.
        var exNeighbors = Vp9NeighborTables.DefaultScan4x4Neighbors;
        var exCache = new byte[16];
        Throws<ArgumentOutOfRangeException>(() =>
            Vp9CoefContext.GetCoefContext(exNeighbors, exCache, scanPos: 17));
        Throws<ArgumentOutOfRangeException>(() =>
            Vp9CoefContext.GetCoefContext(exNeighbors, exCache, scanPos: -1));
    }
}
