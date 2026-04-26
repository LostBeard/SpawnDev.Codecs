// Vp9IvfRemuxer self-roundtrip tests. Constructs a VP9 IVF from BBB.webm
// packets, then drives that IVF through the remuxer and verifies the
// output is byte-identical.

using SpawnDev.Codecs.Container.Ivf;
using SpawnDev.Codecs.Container.Matroska;
using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private static byte[] BuildBbbVp9Ivf()
    {
        using var stream = LoadBigBuckBunnyWebM();
        var container = new MatroskaContainer(stream);
        var video = container.Tracks.First(t => t.IsVideo);

        using var ms = new MemoryStream();
        var ivfOut = new IvfWriter(ms, "VP90", 320, 180, frameRate: 30, timeScale: 1);
        long pts = 0;
        foreach (var pkt in container.Frames.Where(f => f.TrackNumber == video.TrackNumber))
        {
            ivfOut.WriteFrame(pkt.Data.ToArray(), pts++);
        }
        ivfOut.Finish();
        return ms.ToArray();
    }

    [TestMethod]
    public void Vp9IvfRemuxer_BbbAsIvf_RemuxIsByteIdentical()
    {
        var bbbIvf = BuildBbbVp9Ivf();
        var remuxed = Vp9IvfRemuxer.RemuxToBytes(bbbIvf);
        Equal(bbbIvf.Length, remuxed.Length);
        for (int i = 0; i < bbbIvf.Length; i++)
        {
            if (bbbIvf[i] != remuxed[i])
                throw new Exception(
                    $"Byte {i}: source 0x{bbbIvf[i]:X2} vs remux 0x{remuxed[i]:X2}");
        }
    }

    [TestMethod]
    public void Vp9IvfRemuxer_BbbAsIvf_HeaderPreserved()
    {
        var bbbIvf = BuildBbbVp9Ivf();
        var remuxed = Vp9IvfRemuxer.RemuxToBytes(bbbIvf);

        var srcHeader = IvfReader.ParseHeader(bbbIvf);
        var rmxHeader = IvfReader.ParseHeader(remuxed);
        Equal(srcHeader.FourCc, rmxHeader.FourCc);
        Equal(srcHeader.Width, rmxHeader.Width);
        Equal(srcHeader.Height, rmxHeader.Height);
    }
}
