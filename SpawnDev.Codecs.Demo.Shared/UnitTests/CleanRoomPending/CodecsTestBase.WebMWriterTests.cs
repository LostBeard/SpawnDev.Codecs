// Tests for WebMWriter (Container/Matroska), the port of FFmpeg 8.1's matroskaenc in WebM mode.
//
// Fixtures (scripts/video/make_webm_mux_fixtures.py): ffmpeg encodes short synthetic clips and muxes them with
// -fflags +bitexact; each <case>.webm is the truth and <case>.json lists what the muxer was given (tracks, packets as
// offsets into the truth file). Cases: VP9 alone; VP9 + Opus (codec delay, interleaving, the held-back audio packet,
// DiscardPadding at the end); VP9 with alpha (BlockGroup + BlockAdditions, AlphaMode); 12 s VP9 + Opus (5 s clusters,
// Cues); Opus alone at 48 and 44.1 kHz (cues per cluster without video); VP8; AV1 (CodecPrivate av1C); title + global and
// track tags; and a non-seekable (pipe) output (SeekHead in the reserved room, no Duration / Cues, 1 s clusters).
//
// What must hold: the same bytes as ffmpeg, whether the packets arrive in file order or track by track (the writer
// interleaves them as libavformat does).

using System.Text.Json;
using SpawnDev.Codecs.Container.Matroska;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private const string WebMMuxPrefix = "SpawnDev.Codecs.Demo.Shared.TestData.WebMMux.";

    /// <summary>A write-only stream that cannot seek (ffmpeg writing to a pipe).</summary>
    private sealed class NonSeekableStream : Stream
    {
        public readonly MemoryStream Inner = new();
        public override bool CanRead => false;
        public override bool CanSeek => false;
        public override bool CanWrite => true;
        public override long Length => throw new NotSupportedException();
        public override long Position { get => throw new NotSupportedException(); set => throw new NotSupportedException(); }
        public override void Flush() { }
        public override int Read(byte[] buffer, int offset, int count) => throw new NotSupportedException();
        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
        public override void SetLength(long value) => throw new NotSupportedException();
        public override void Write(byte[] buffer, int offset, int count) => Inner.Write(buffer, offset, count);
        public override void Write(ReadOnlySpan<byte> buffer) => Inner.Write(buffer);
    }

    private static List<KeyValuePair<string, string>> WebMTags(JsonElement arr) =>
        arr.EnumerateArray().Select(t => new KeyValuePair<string, string>(t[0].GetString()!, t[1].GetString()!)).ToList();

    private static byte[] WebMMux(byte[] truth, JsonElement doc, bool trackByTrack)
    {
        var info = doc.GetProperty("info");
        bool live = doc.GetProperty("live").GetBoolean();
        var options = new WebMWriterOptions
        {
            MuxingApp = info.GetProperty("muxing").GetString()!,
            WritingApp = info.GetProperty("writing").GetString()!,
            Title = info.GetProperty("title").ValueKind == JsonValueKind.String ? info.GetProperty("title").GetString() : null,
            Tags = WebMTags(doc.GetProperty("globalTags")),
        };
        Stream stream = live ? new NonSeekableStream() : new MemoryStream();
        using var w = new WebMWriter(stream, options, leaveOpen: true);
        foreach (var t in doc.GetProperty("tracks").EnumerateArray())
        {
            string codec = t.GetProperty("codec").GetString()!;
            byte[]? cp = t.TryGetProperty("codecPrivate", out var cpv) ? Convert.FromHexString(cpv.GetString()!) : null;
            int reserve = t.TryGetProperty("codecPrivateReserve", out var rv) ? rv.GetInt32() : 0;
            long? dd = t.TryGetProperty("defaultDurationNs", out var ddv) ? ddv.GetInt64() : null;
            var tags = WebMTags(t.GetProperty("tags"));
            string? lang = t.TryGetProperty("language", out var lv) ? lv.GetString() : null;
            string? name = t.TryGetProperty("name", out var nv) ? nv.GetString() : null;
            bool isDefault = t.GetProperty("default").GetBoolean(), forced = t.GetProperty("forced").GetBoolean();
            ulong uid = t.GetProperty("uid").GetUInt64();
            if (t.GetProperty("type").GetInt32() == 1)
            {
                WebMColour? colour = null;
                if (t.TryGetProperty("colour", out var c))
                {
                    int? Get(string k) => c.TryGetProperty(k, out var v) ? v.GetInt32() : null;
                    colour = new WebMColour
                    {
                        MatrixCoefficients = Get("matrix"), Range = Get("range"), TransferCharacteristics = Get("transfer"),
                        Primaries = Get("primaries"), ChromaSitingHorz = Get("sitingH"), ChromaSitingVert = Get("sitingV"),
                    };
                }
                w.AddVideoTrack(new WebMVideoTrack
                {
                    CodecId = codec, CodecPrivate = cp, CodecPrivateReserve = reserve, Uid = uid, Name = name, Language = lang, IsDefault = isDefault, IsForced = forced,
                    DefaultDurationNs = dd, Tags = tags,
                    Width = t.GetProperty("width").GetInt32(), Height = t.GetProperty("height").GetInt32(),
                    Interlaced = t.TryGetProperty("interlaced", out var il) ? il.GetBoolean() : null,
                    AlphaMode = t.TryGetProperty("alphaMode", out var am) && am.GetBoolean(),
                    DisplaySize = t.TryGetProperty("displayWidth", out var dw) ? (dw.GetInt32(), t.GetProperty("displayHeight").GetInt32()) : null,
                    Colour = colour,
                });
            }
            else
            {
                w.AddAudioTrack(new WebMAudioTrack
                {
                    CodecId = codec, CodecPrivate = cp, CodecPrivateReserve = reserve, Uid = uid, Name = name, Language = lang, IsDefault = isDefault, IsForced = forced,
                    DefaultDurationNs = dd, Tags = tags,
                    SampleRate = t.GetProperty("sampleRate").GetDouble(), Channels = t.GetProperty("channels").GetInt32(),
                    BitDepth = t.TryGetProperty("bitDepth", out var bd) ? bd.GetInt32() : null,
                    InitialPaddingSamples = t.TryGetProperty("initialPaddingSamples", out var ip) ? ip.GetInt32() : 0,
                });
            }
        }
        var packets = doc.GetProperty("packets").EnumerateArray().ToList();
        if (trackByTrack) packets = packets.OrderBy(p => p.GetProperty("track").GetInt32()).ToList();   // stable: per-track order kept
        foreach (var p in packets)
        {
            var d = p.GetProperty("data");
            var data = truth.AsSpan(d[0].GetInt32(), d[1].GetInt32());
            ReadOnlySpan<byte> alpha = default;
            if (p.GetProperty("add").ValueKind == JsonValueKind.Array)
            {
                var a = p.GetProperty("add");
                alpha = truth.AsSpan(a[0].GetInt32(), a[1].GetInt32());
            }
            w.WriteFrame(p.GetProperty("track").GetInt32(), data, p.GetProperty("ts").GetInt64(), p.GetProperty("key").GetBoolean(),
                p.GetProperty("duration").GetInt64(), alpha, p.GetProperty("discardNs").GetInt64());
        }
        w.Finish();
        return live ? ((NonSeekableStream)stream).Inner.ToArray() : ((MemoryStream)stream).ToArray();
    }

    [TestMethod(Category = "CpuOnly")]
    public async Task WebMWriter_ByteIdenticalWithFfmpeg()
    {
        await Task.Yield();
        var cases = System.Text.Encoding.UTF8.GetString(ReadEmbedded(WebMMuxPrefix + "manifest.txt"))
            .Split('\n').Select(l => l.Trim()).Where(l => l.Length > 0).ToList();
        var failures = new List<string>();
        int checkedOrders = 0;
        foreach (var name in cases)
        {
            var truth = ReadEmbedded(WebMMuxPrefix + name + ".webm");
            using var json = JsonDocument.Parse(ReadEmbedded(WebMMuxPrefix + name + ".json"));
            long span = json.RootElement.GetProperty("packets").EnumerateArray().Max(p => p.GetProperty("ts").GetInt64());
            foreach (bool byTrack in span < 9000 ? new[] { false, true } : new[] { false })
            {
                try
                {
                    var ours = WebMMux(truth, json.RootElement, byTrack);
                    checkedOrders++;
                    if (!OperatingSystem.IsBrowser() && Environment.GetEnvironmentVariable("CODECS_DUMP_DIR") is { Length: > 0 } dump)
                        File.WriteAllBytes(Path.Combine(dump, name + (byTrack ? ".bytrack" : "") + ".webm"), ours);
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
}
