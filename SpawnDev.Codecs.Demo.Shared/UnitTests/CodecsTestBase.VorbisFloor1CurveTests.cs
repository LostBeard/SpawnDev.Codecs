using SpawnDev.Codecs.Audio.Vorbis;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for <see cref="VorbisFloor1Curve"/>. Cover the static helpers
/// (low_neighbour, high_neighbour, render_point) against the Vorbis I Section
/// 7.2.4 reference formulas, plus an end-to-end curve render producing a
/// non-negative float curve of the expected length.
/// </summary>
public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void VorbisFloor1Curve_LowNeighbour_TakesLargestStrictlyLessThan()
    {
        // x = [0, 128, 4, 64, 32, 16], i = 5 -> neighbours considered are j=0..4.
        // x[i] = 16. Values strictly less than 16 among x[0..4]: 0 (idx 0), 4 (idx 2).
        // Largest: 4 at index 2.
        int[] x = { 0, 128, 4, 64, 32, 16 };
        Equal(2, VorbisFloor1Curve.LowNeighbourOffset(x, 5));
    }

    [TestMethod]
    public void VorbisFloor1Curve_HighNeighbour_TakesSmallestStrictlyGreaterThan()
    {
        // Same array, same i. Values strictly > 16 among x[0..4]: 128 (idx 1),
        // 64 (idx 3), 32 (idx 4). Smallest = 32 at index 4.
        int[] x = { 0, 128, 4, 64, 32, 16 };
        Equal(4, VorbisFloor1Curve.HighNeighbourOffset(x, 5));
    }

    [TestMethod]
    public void VorbisFloor1Curve_RenderPoint_IntegerInterpolationAtHalf()
    {
        // Line from (0, 100) to (10, 200), sample at x=5.
        // dy=100, adx=10, ady=100, err=100*5=500, off=500/10=50 -> y = 100+50 = 150.
        Equal(150, VorbisFloor1Curve.RenderPoint(0, 100, 10, 200, 5));
    }

    [TestMethod]
    public void VorbisFloor1Curve_RenderPoint_NegativeSlope()
    {
        // Line (0, 200) to (10, 100), sample at x=5 -> y = 150 (symmetric).
        Equal(150, VorbisFloor1Curve.RenderPoint(0, 200, 10, 100, 5));
    }

    [TestMethod]
    public void VorbisFloor1Curve_RenderPoint_Endpoints()
    {
        // At x=0 and x=10, returns exact endpoint values.
        Equal(100, VorbisFloor1Curve.RenderPoint(0, 100, 10, 200, 0));
        Equal(200, VorbisFloor1Curve.RenderPoint(0, 100, 10, 200, 10));
    }

    [TestMethod]
    public void VorbisFloor1Curve_Render_FullCurve_ProducesPositiveFloats()
    {
        // Minimal floor: 1 partition class 0, dims 1, subclasses 0, one X at 10.
        // multiplier 1, rangeBits 4 -> XList = [0, 16, 10].
        // decodedY = [100, 50, 20]. The rest of the fields (Partitions, class
        // tables) don't matter for Render because it only uses XList +
        // Multiplier + decodedY.
        var cfg = new VorbisFloor1Config
        {
            Partitions = 1,
            PartitionClassList = new[] { 0 },
            ClassDimensions = new[] { 1 },
            ClassSubclasses = new[] { 0 },
            ClassMasterbooks = new[] { -1 },
            ClassSubclassBooks = new[] { new[] { 0 } },
            Multiplier = 1,
            RangeBits = 4,
            XList = new[] { 0, 16, 10 },
        };
        var decodedY = new[] { 100, 50, 20 };
        float[] curve = new float[16];
        VorbisFloor1Curve.Render(cfg, decodedY, 16, curve);
        // Every output must be non-negative and finite.
        for (int i = 0; i < curve.Length; i++)
        {
            True(curve[i] >= 0, $"curve[{i}] = {curve[i]} should be >= 0");
            True(float.IsFinite(curve[i]), $"curve[{i}] = {curve[i]} should be finite");
        }
        // First sample should correspond to the first-X-sorted point's amplitude.
        True(curve[0] > 0, "curve[0] should be > 0 (first point amplitude).");
    }

    [TestMethod]
    public void VorbisFloor1Curve_Render_LengthMismatch_Throws()
    {
        var cfg = new VorbisFloor1Config
        {
            Partitions = 0, PartitionClassList = Array.Empty<int>(),
            ClassDimensions = Array.Empty<int>(), ClassSubclasses = Array.Empty<int>(),
            ClassMasterbooks = Array.Empty<int>(), ClassSubclassBooks = Array.Empty<int[]>(),
            Multiplier = 1, RangeBits = 4, XList = new[] { 0, 16 },
        };
        bool threw = false;
        try
        {
            VorbisFloor1Curve.Render(cfg, new[] { 10, 20 }, 16, new float[10]);
        }
        catch (ArgumentException) { threw = true; }
        True(threw);
    }

    [TestMethod]
    public void VorbisFloor1Curve_Render_DifferentMultipliers_ProduceFiniteCurves()
    {
        foreach (int mult in new[] { 1, 2, 3, 4 })
        {
            var cfg = new VorbisFloor1Config
            {
                Partitions = 0, PartitionClassList = Array.Empty<int>(),
                ClassDimensions = Array.Empty<int>(), ClassSubclasses = Array.Empty<int>(),
                ClassMasterbooks = Array.Empty<int>(), ClassSubclassBooks = Array.Empty<int[]>(),
                Multiplier = mult, RangeBits = 4, XList = new[] { 0, 16 },
            };
            float[] curve = new float[16];
            VorbisFloor1Curve.Render(cfg, new[] { 20, 10 }, 16, curve);
            for (int i = 0; i < curve.Length; i++)
                True(float.IsFinite(curve[i]) && curve[i] >= 0, $"mult={mult} curve[{i}]={curve[i]}");
        }
    }
}
