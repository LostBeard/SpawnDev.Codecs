using SpawnDev.Codecs.Audio.Vorbis;
using SpawnDev.Codecs.Container.Ogg;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Parity of both Vorbis decoders with LIBVORBIS on real libvorbis streams (TestData/Vorbis, made by
/// scripts/audio/make_vorbis_fixtures.py): the reference PCM is libsndfile's decode of the same .ogg. Unlike the
/// self round trips, these streams have two floors with block switching, coupled stereo (residue type 2) and several
/// quality setups. The reference is trimmed to the last granule position; the decoders are not, so the first
/// <c>frames</c> sample frames are compared (both start at the second packet: the first only primes the overlap).
/// </summary>
public abstract partial class CodecsTestBase
{
    private const string VorbisFixturePrefix = "SpawnDev.Codecs.Demo.Shared.TestData.Vorbis.";

    /// <summary>
    /// Largest |ours - libvorbis| accepted per sample. Both are float decoders of the same bitstream with different
    /// transform implementations, so they differ in the last bits only; a wrong codebook, floor or residue is off by
    /// orders of magnitude more.
    /// </summary>
    private const float VorbisParityTolerance = 1e-4f;

    private sealed record VorbisFixture(string Name, int Channels, int Rate, int Frames, byte[] Ogg, float[] Reference);

    private static List<VorbisFixture> LoadVorbisFixtures()
    {
        var list = new List<VorbisFixture>();
        string manifest = System.Text.Encoding.UTF8.GetString(ReadEmbedded(VorbisFixturePrefix + "manifest.tsv"));
        foreach (var line in manifest.Split('\n'))
        {
            if (line.Length == 0 || line[0] == '#') continue;
            var f = line.Split('\t');
            byte[] pcm = Gunzip(ReadEmbedded(VorbisFixturePrefix + f[0] + ".f32.gz"));
            var reference = new float[pcm.Length / 4];
            Buffer.BlockCopy(pcm, 0, reference, 0, pcm.Length);
            list.Add(new VorbisFixture(f[0], int.Parse(f[1]), int.Parse(f[2]), int.Parse(f[3]),
                ReadEmbedded(VorbisFixturePrefix + f[0] + ".ogg"), reference));
        }
        return list;
    }

    /// <summary>The headers and audio packets of a single-stream Ogg-Vorbis file.</summary>
    private static (VorbisIdentificationHeader Ident, VorbisSetupHeader Setup, byte[][] Audio) ReadVorbisPackets(byte[] ogg)
    {
        var pages = OggPageReader.EnumeratePages(ogg).ToArray();
        var packets = OggPacketReader.AssemblePackets(pages.Where(p => p.BitstreamSerial == pages[0].BitstreamSerial).ToArray()).ToArray();
        var ident = VorbisIdentificationHeaderParser.Parse(packets[0].Data);
        var setup = VorbisSetupHeaderParser.Parse(packets[2].Data, ident.AudioChannels);
        return (ident, setup, packets.Skip(3).Select(p => p.Data).ToArray());
    }

    /// <summary>Null when the first <c>Frames</c> frames of <paramref name="ours"/> match the reference within the tolerance.</summary>
    private static string? CompareVorbisPcm(VorbisFixture fx, float[] ours, out float maxError)
    {
        maxError = 0;
        int n = fx.Frames * fx.Channels;
        if (ours.Length < n) return $"{fx.Name}: {ours.Length / fx.Channels} frames decoded, libvorbis has {fx.Frames}";
        int worst = -1;
        for (int i = 0; i < n; i++)
        {
            float e = Math.Abs(ours[i] - fx.Reference[i]);
            if (!(e <= maxError)) { maxError = e; worst = i; }
        }
        if (!(maxError <= VorbisParityTolerance))
            return $"{fx.Name}: max |error| {maxError:G4} at frame {worst / fx.Channels} channel {worst % fx.Channels} " +
                $"(ours {ours[worst]:G6}, libvorbis {fx.Reference[worst]:G6})";
        return null;
    }

    /// <summary>A short summary of what a setup exercises (floors, residue types, block sizes).</summary>
    private static string DescribeVorbisSetup(VorbisIdentificationHeader ident, VorbisSetupHeader setup)
        => $"{ident.AudioChannels} ch {ident.SampleRateHz} Hz, blocks {ident.BlockSize0}/{ident.BlockSize1}, " +
            $"{setup.Floors.Length} floors, residue types [{string.Join(",", setup.Residues.Select(r => (int)r.Type))}], " +
            $"{setup.Codebooks.Length} books";

    [TestMethod]
    public void VorbisAudioDecoder_LibvorbisStreams_MatchLibvorbis()
    {
        CpuOnlyTest();
        var failures = new List<string>();
        foreach (var fx in LoadVorbisFixtures())
        {
            var decoded = VorbisOggDecoder.Decode(fx.Ogg);
            string? diff = CompareVorbisPcm(fx, decoded.InterleavedSamples, out float err);
            Console.WriteLine($"{fx.Name}: {DescribeVorbisSetup(decoded.Identification, decoded.Setup)}, max |error| {err:G4}");
            if (diff != null) failures.Add(diff);
        }
        True(failures.Count == 0, string.Join("\n", failures));
    }

    [TestMethod(Timeout = 600000)]
    public async Task VorbisAudioDecoderGpu_LibvorbisStreams_MatchLibvorbis()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        var failures = new List<string>();
        foreach (var fx in LoadVorbisFixtures())
        {
            var (ident, setup, audio) = ReadVorbisPackets(fx.Ogg);
            using var dec = new VorbisAudioDecoderGpu(acc, ident, setup);
            var ours = new List<float>(fx.Frames * fx.Channels + ident.BlockSize1 * fx.Channels);
            foreach (var packet in audio) ours.AddRange(await dec.DecodePacketAsync(packet));
            string? diff = CompareVorbisPcm(fx, ours.ToArray(), out float err);
            Console.WriteLine($"{fx.Name}: {DescribeVorbisSetup(ident, setup)}, max |error| {err:G4}");
            if (diff != null) failures.Add(diff);
        }
        True(failures.Count == 0, string.Join("\n", failures));
    }
}
