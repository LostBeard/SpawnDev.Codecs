// Tests for Mp4Writer (Container/Mp4), the AV1 + Opus MP4 muxer.
//
// Fixtures (scripts/video/make_mp4_mux_fixtures.py): the ffmpeg command line tool (a black box) encodes short synthetic
// clips (libaom-av1, libopus) and muxes them into MP4 with -fflags +bitexact; each <case>.mp4 is the truth and
// <case>.json lists the tracks (codec configuration, timescale) and the packets (decode time, duration, key flag, bytes as
// offsets into the truth file). Cases: AV1 alone (with and without non-key frames), AV1 + Opus, Opus alone at 48 kHz
// stereo and from 44.1 kHz mono, AV1 at 24, 25, 30000/1001 and 16 fps (the timescale rule), and 12 s of AV1 + Opus
// (interleaving into many chunks).
//
// What must hold: the same bytes as ffmpeg, whether the packets arrive in file order or track by track.

using System.Text.Json;
using SpawnDev.Codecs.Container.Mp4;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private const string Mp4MuxPrefix = "SpawnDev.Codecs.Demo.Shared.TestData.Mp4Mux.";

    private static byte[] Mp4Mux(byte[] truth, JsonElement doc, bool trackByTrack)
    {
        var stream = new MemoryStream();
        using (var w = new Mp4Writer(stream, leaveOpen: true))
        {
            foreach (var t in doc.GetProperty("tracks").EnumerateArray())
            {
                var extradata = Convert.FromHexString(t.GetProperty("extradata").GetString()!);
                int timescale = int.Parse(t.GetProperty("timebase").GetString()!.Split('/')[1]);
                if (t.GetProperty("type").GetString() == "video")
                    w.AddVideoTrack(new Mp4Av1Track { Av1C = extradata, Width = t.GetProperty("width").GetInt32(), Height = t.GetProperty("height").GetInt32(), Timescale = timescale });
                else
                    w.AddAudioTrack(new Mp4OpusTrack { OpusHead = extradata });
            }
            var packets = doc.GetProperty("packets").EnumerateArray().ToList();
            if (trackByTrack) packets = packets.OrderBy(p => p.GetProperty("track").GetInt32()).ToList();   // stable
            foreach (var p in packets)
            {
                var d = p.GetProperty("data");
                w.WriteFrame(p.GetProperty("track").GetInt32() + 1, truth.AsSpan(d[0].GetInt32(), d[1].GetInt32()),
                    p.GetProperty("dts").GetInt64(), p.GetProperty("duration").GetInt64(), p.GetProperty("key").GetBoolean());
            }
            w.Finish();
        }
        return stream.ToArray();
    }

    [TestMethod(Category = "CpuOnly")]
    public async Task Mp4Writer_ByteIdenticalWithFfmpeg()
    {
        await Task.Yield();
        var cases = System.Text.Encoding.UTF8.GetString(ReadEmbedded(Mp4MuxPrefix + "manifest.txt"))
            .Split('\n').Select(l => l.Trim()).Where(l => l.Length > 0).ToList();
        var failures = new List<string>();
        int checkedOrders = 0;
        foreach (var name in cases)
        {
            var truth = ReadEmbedded(Mp4MuxPrefix + name + ".mp4");
            using var json = JsonDocument.Parse(ReadEmbedded(Mp4MuxPrefix + name + ".json"));
            foreach (bool byTrack in new[] { false, true })
            {
                try
                {
                    var ours = Mp4Mux(truth, json.RootElement, byTrack);
                    checkedOrders++;
                    if (!OperatingSystem.IsBrowser() && Environment.GetEnvironmentVariable("CODECS_DUMP_DIR") is { Length: > 0 } dump)
                        File.WriteAllBytes(Path.Combine(dump, name + (byTrack ? ".bytrack" : "") + ".mp4"), ours);
                    if (ours.AsSpan().SequenceEqual(truth)) continue;
                    int first = 0, lim = Math.Min(ours.Length, truth.Length);
                    while (first < lim && ours[first] == truth[first]) first++;
                    failures.Add($"{name}{(byTrack ? " (track by track)" : "")}: {ours.Length} bytes vs ffmpeg {truth.Length}, first difference at byte {first}");
                }
                catch (Exception ex)
                {
                    failures.Add($"{name}: {ex.GetType().Name}: {ex.Message}");
                }
            }
        }
        True(cases.Count >= 10, $"only {cases.Count} fixtures");
        True(failures.Count == 0, $"{failures.Count} of {checkedOrders} differ:\n" + string.Join("\n", failures));
    }

    /// <summary>
    /// Av1Isobmff on every AV1 fixture: BuildAv1C of ffmpeg's first video sample (it keeps the sequence header in key
    /// samples) equals the av1C ffmpeg wrote, and ToSample removes a temporal delimiter put in front of each sample (ffmpeg
    /// samples carry none) and changes nothing else.
    /// </summary>
    [TestMethod(Category = "CpuOnly")]
    public async Task Av1Isobmff_Av1CAndSamples_MatchFfmpeg()
    {
        await Task.Yield();
        var cases = System.Text.Encoding.UTF8.GetString(ReadEmbedded(Mp4MuxPrefix + "manifest.txt"))
            .Split('\n').Select(l => l.Trim()).Where(l => l.Length > 0).ToList();
        int records = 0, samples = 0;
        foreach (var name in cases)
        {
            var file = ReadEmbedded(Mp4MuxPrefix + name + ".mp4");
            using var json = JsonDocument.Parse(ReadEmbedded(Mp4MuxPrefix + name + ".json"));
            foreach (var t in json.RootElement.GetProperty("tracks").EnumerateArray())
            {
                if (t.GetProperty("type").GetString() != "video") continue;
                int index = t.GetProperty("index").GetInt32();
                var packets = json.RootElement.GetProperty("packets").EnumerateArray().Where(p => p.GetProperty("track").GetInt32() == index).ToList();
                byte[] Bytes(JsonElement p) { var d = p.GetProperty("data"); return file.AsSpan(d[0].GetInt32(), d[1].GetInt32()).ToArray(); }
                var av1c = Av1Isobmff.BuildAv1C(Bytes(packets[0]));
                Equal(t.GetProperty("extradata").GetString()!.ToUpperInvariant(), Convert.ToHexString(av1c), $"{name}: av1C");
                records++;
                foreach (var p in packets)
                {
                    var s = Bytes(p);
                    var tu = new byte[] { 0x12, 0x00 }.Concat(s).ToArray();   // temporal delimiter OBU (type 2, size 0)
                    True(Av1Isobmff.ToSample(tu).AsSpan().SequenceEqual(s), $"{name}: ToSample changed a sample");
                    samples++;
                }
            }
        }
        True(records >= 7, $"only {records} AV1 tracks");
        Console.WriteLine($"{records} av1C records == ffmpeg, {samples} samples");
    }

    /// <summary>ffmpeg's video timescale: the rate's numerator scaled by the smallest power of two reaching 10000.</summary>
    [TestMethod(Category = "CpuOnly")]
    public void Mp4Writer_VideoTimescale_MatchesFfmpeg()
    {
        Equal(10240, Mp4Writer.VideoTimescale(10, 1));
        Equal(12288, Mp4Writer.VideoTimescale(24, 1));
        Equal(12800, Mp4Writer.VideoTimescale(25, 1));
        Equal(16384, Mp4Writer.VideoTimescale(16, 1));
        Equal(30000, Mp4Writer.VideoTimescale(30000, 1001));
    }
}
