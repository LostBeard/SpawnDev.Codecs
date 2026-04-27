// IvfWriter + IvfReader round-trip tests. Verify that frames written
// with IvfWriter are read back bit-exact by IvfReader on a wide range
// of payload sizes and frame counts. This is fundamental container
// correctness that every video codec in the library depends on.

using SpawnDev.Codecs.Container.Ivf;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private static byte[] BuildIvf(string fourCc, int width, int height,
        IReadOnlyList<(byte[] payload, long pts)> frames,
        uint frameRate = 30, uint timeScale = 1)
    {
        using var ms = new MemoryStream();
        var writer = new IvfWriter(ms, fourCc, width, height,
                                   frameRate, timeScale,
                                   numFrames: 0,
                                   leaveOpen: true);
        foreach (var f in frames) writer.WriteFrame(f.payload, f.pts);
        writer.Finish();
        return ms.ToArray();
    }

    [TestMethod]
    public void IvfRoundTrip_EmptyStream_ReadsHeaderOnly()
    {
        var bytes = BuildIvf("VP80", 320, 240, Array.Empty<(byte[], long)>());
        var hdr = IvfReader.ParseHeader(bytes);
        Equal("VP80", hdr.FourCc);
        Equal(320, hdr.Width);
        Equal(240, hdr.Height);
        Equal(0u, hdr.NumFrames);

        int frameCount = 0;
        foreach (var _ in IvfReader.EnumerateFrames(bytes)) frameCount++;
        Equal(0, frameCount);
    }

    [TestMethod]
    public void IvfRoundTrip_SingleFrame_RecoversBytes()
    {
        var payload = new byte[] { 0x01, 0x02, 0x03, 0x04, 0x05 };
        var bytes = BuildIvf("VP90", 640, 480, new[] { (payload, 0L) });

        var hdr = IvfReader.ParseHeader(bytes);
        Equal("VP90", hdr.FourCc);
        Equal(640, hdr.Width);
        Equal(480, hdr.Height);
        Equal(1u, hdr.NumFrames);

        var frames = IvfReader.EnumerateFrames(bytes).ToList();
        Equal(1, frames.Count);
        Equal(0L, frames[0].Pts);
        True(frames[0].Data.Span.SequenceEqual(payload),
            $"payload mismatch: expected {payload.Length}B, got {frames[0].Data.Length}B");
    }

    [TestMethod]
    public void IvfRoundTrip_MultipleFrames_PreservesPtsOrder()
    {
        var rng = new Random(0x1F1F);
        var sources = new List<(byte[] payload, long pts)>();
        for (int i = 0; i < 32; i++)
        {
            int len = 16 + rng.Next(256);
            var p = new byte[len];
            rng.NextBytes(p);
            sources.Add((p, i * 1000L));
        }

        var bytes = BuildIvf("AV01", 320, 180, sources);
        var hdr = IvfReader.ParseHeader(bytes);
        Equal((uint)sources.Count, hdr.NumFrames);

        var frames = IvfReader.EnumerateFrames(bytes).ToList();
        Equal(sources.Count, frames.Count);
        for (int i = 0; i < sources.Count; i++)
        {
            Equal(sources[i].pts, frames[i].Pts);
            True(frames[i].Data.Span.SequenceEqual(sources[i].payload),
                $"payload mismatch at frame {i}");
        }
    }

    [TestMethod]
    public void IvfRoundTrip_LargeFrame_RecoversBytes()
    {
        // 2 MiB frame (above the typical 1 KiB stack threshold).
        var payload = new byte[2 * 1024 * 1024];
        for (int i = 0; i < payload.Length; i++) payload[i] = (byte)(i & 0xFF);
        var bytes = BuildIvf("VP90", 1920, 1080, new[] { (payload, 0L) });

        var frames = IvfReader.EnumerateFrames(bytes).ToList();
        Equal(1, frames.Count);
        Equal(payload.Length, frames[0].Data.Length);
        True(frames[0].Data.Span.SequenceEqual(payload), "large payload mismatch");
    }

    [TestMethod]
    public void IvfRoundTrip_ZeroLengthFrame_ReadsAsZeroBytes()
    {
        // VP9 super-frames sometimes produce zero-length packets; the
        // writer must encode them and the reader must enumerate them
        // without skipping or crashing.
        var bytes = BuildIvf("VP90", 320, 240, new[]
        {
            (new byte[] { 0xAA, 0xBB }, 0L),
            (Array.Empty<byte>(), 1L),
            (new byte[] { 0xCC, 0xDD }, 2L),
        });

        var frames = IvfReader.EnumerateFrames(bytes).ToList();
        Equal(3, frames.Count);
        Equal(2, frames[0].Data.Length);
        Equal(0, frames[1].Data.Length);
        Equal(2, frames[2].Data.Length);
    }

    [TestMethod]
    public void IvfRoundTrip_FourCcVariants_AllAccepted()
    {
        // Cover every codec that ships in this library: VP80, VP90, AV01.
        foreach (var fourCc in new[] { "VP80", "VP90", "AV01" })
        {
            var bytes = BuildIvf(fourCc, 100, 50, new[] { (new byte[] { 0x01 }, 0L) });
            var hdr = IvfReader.ParseHeader(bytes);
            Equal(fourCc, hdr.FourCc);
            Equal(100, hdr.Width);
            Equal(50, hdr.Height);
        }
    }

    [TestMethod]
    public void IvfRoundTrip_FrameRateAndTimeScale_Preserved()
    {
        var bytes = BuildIvf("VP90", 320, 240, Array.Empty<(byte[], long)>(),
            frameRate: 60, timeScale: 1001);
        var hdr = IvfReader.ParseHeader(bytes);
        Equal(60u, hdr.FrameRate);
        Equal(1001u, hdr.TimeScale);
    }

    [TestMethod]
    public void IvfRoundTrip_FrameCountPatchedByFinish()
    {
        // Construct without calling Finish: num_frames stays 0.
        // Construct with Finish: num_frames = actual count.
        using var ms1 = new MemoryStream();
        var w1 = new IvfWriter(ms1, "VP90", 320, 240, leaveOpen: true);
        w1.WriteFrame(new byte[] { 1 }, 0);
        w1.WriteFrame(new byte[] { 2 }, 1);
        // Skip Finish().
        var hdr1 = IvfReader.ParseHeader(ms1.ToArray());
        Equal(0u, hdr1.NumFrames);

        using var ms2 = new MemoryStream();
        var w2 = new IvfWriter(ms2, "VP90", 320, 240, leaveOpen: true);
        w2.WriteFrame(new byte[] { 1 }, 0);
        w2.WriteFrame(new byte[] { 2 }, 1);
        w2.Finish();
        var hdr2 = IvfReader.ParseHeader(ms2.ToArray());
        Equal(2u, hdr2.NumFrames);
    }
}
