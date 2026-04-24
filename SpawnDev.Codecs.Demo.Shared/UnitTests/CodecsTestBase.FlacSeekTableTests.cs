using SpawnDev.Codecs.Audio.Flac;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for <see cref="FlacSeekTableParser"/>. Each test hand-builds an
/// 18-bytes-per-point SEEKTABLE payload (big-endian) and parses it back.
/// </summary>
public abstract partial class CodecsTestBase
{
    private static byte[] BuildSeekTablePayload(params FlacSeekPoint[] points)
    {
        var bytes = new byte[points.Length * 18];
        int pos = 0;
        foreach (var p in points)
        {
            WriteUInt64Be(bytes, pos, p.SampleNumber); pos += 8;
            WriteUInt64Be(bytes, pos, p.StreamOffset); pos += 8;
            WriteUInt16Be(bytes, pos, p.FrameSamples); pos += 2;
        }
        return bytes;
    }

    private static void WriteUInt64Be(byte[] dest, int offset, ulong value)
    {
        for (int i = 0; i < 8; i++) dest[offset + i] = (byte)(value >> (56 - 8 * i));
    }

    private static void WriteUInt16Be(byte[] dest, int offset, ushort value)
    {
        dest[offset] = (byte)(value >> 8);
        dest[offset + 1] = (byte)value;
    }

    [TestMethod]
    public void FlacSeekTable_EmptyPayload_ParsesToZeroPoints()
    {
        var tbl = FlacSeekTableParser.Parse(Array.Empty<byte>());
        Equal(0, tbl.Points.Length);
    }

    [TestMethod]
    public void FlacSeekTable_SinglePoint_Parses()
    {
        var point = new FlacSeekPoint(1_000_000UL, 0x123456789ABCUL, 4096);
        var payload = BuildSeekTablePayload(point);
        var tbl = FlacSeekTableParser.Parse(payload);
        Equal(1, tbl.Points.Length);
        Equal(1_000_000UL, tbl.Points[0].SampleNumber);
        Equal(0x123456789ABCUL, tbl.Points[0].StreamOffset);
        Equal((ushort)4096, tbl.Points[0].FrameSamples);
        False(tbl.Points[0].IsPlaceholder);
    }

    [TestMethod]
    public void FlacSeekTable_PlaceholderPoint_FlaggedCorrectly()
    {
        var point = new FlacSeekPoint(FlacSeekPoint.PlaceholderSampleNumber, 0, 0);
        var tbl = FlacSeekTableParser.Parse(BuildSeekTablePayload(point));
        True(tbl.Points[0].IsPlaceholder);
    }

    [TestMethod]
    public void FlacSeekTable_MultiplePoints_PreservesOrder()
    {
        var points = new[]
        {
            new FlacSeekPoint(0, 0, 4096),
            new FlacSeekPoint(4096, 1000, 4096),
            new FlacSeekPoint(8192, 2100, 4096),
            new FlacSeekPoint(12288, 3333, 4096),
        };
        var tbl = FlacSeekTableParser.Parse(BuildSeekTablePayload(points));
        Equal(4, tbl.Points.Length);
        for (int i = 0; i < points.Length; i++)
        {
            Equal(points[i].SampleNumber, tbl.Points[i].SampleNumber);
            Equal(points[i].StreamOffset, tbl.Points[i].StreamOffset);
            Equal(points[i].FrameSamples, tbl.Points[i].FrameSamples);
        }
    }

    [TestMethod]
    public void FlacSeekTable_MalformedLength_Throws()
    {
        // 17 bytes is not a multiple of 18.
        var bad = new byte[17];
        bool threw = false;
        try { FlacSeekTableParser.Parse(bad); } catch (InvalidDataException) { threw = true; }
        True(threw);
    }

    [TestMethod]
    public void FlacSeekTable_FindNearest_PicksLargestNotExceedingTarget()
    {
        var points = new[]
        {
            new FlacSeekPoint(0, 0, 4096),
            new FlacSeekPoint(4096, 100, 4096),
            new FlacSeekPoint(8192, 200, 4096),
            new FlacSeekPoint(12288, 300, 4096),
        };
        var tbl = new FlacSeekTable { Points = points };
        // Target between point[1] and point[2] picks point[1].
        var nearest = FlacSeekTableParser.FindNearest(tbl, 7000);
        True(nearest.HasValue);
        Equal(4096UL, nearest!.Value.SampleNumber);
        // Target exactly on a point returns that point.
        nearest = FlacSeekTableParser.FindNearest(tbl, 8192);
        Equal(8192UL, nearest!.Value.SampleNumber);
        // Target beyond last point returns the last point.
        nearest = FlacSeekTableParser.FindNearest(tbl, 99999);
        Equal(12288UL, nearest!.Value.SampleNumber);
    }

    [TestMethod]
    public void FlacSeekTable_FindNearest_SkipsPlaceholders()
    {
        var points = new[]
        {
            new FlacSeekPoint(0, 0, 4096),
            new FlacSeekPoint(FlacSeekPoint.PlaceholderSampleNumber, 0, 0),
            new FlacSeekPoint(8192, 200, 4096),
        };
        var tbl = new FlacSeekTable { Points = points };
        var nearest = FlacSeekTableParser.FindNearest(tbl, 4096);
        // Placeholder skipped - answer is the 0-sample point.
        True(nearest.HasValue);
        Equal(0UL, nearest!.Value.SampleNumber);
    }

    [TestMethod]
    public void FlacSeekTable_FindNearest_AllPlaceholders_ReturnsNull()
    {
        var tbl = new FlacSeekTable
        {
            Points = new[] { new FlacSeekPoint(FlacSeekPoint.PlaceholderSampleNumber, 0, 0) },
        };
        var nearest = FlacSeekTableParser.FindNearest(tbl, 1000);
        True(nearest is null);
    }
}
