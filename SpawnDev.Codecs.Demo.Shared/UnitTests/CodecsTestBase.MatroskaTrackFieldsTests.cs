// MatroskaContainer reads the track header fields and per-block DiscardPadding a decoder needs (CodecPrivate,
// DefaultDuration, CodecDelay, PixelWidth / Height, AlphaMode, SamplingFrequency, Channels, BitDepth).
//
// Oracle: the WebMMux fixtures (scripts/video/make_webm_mux_fixtures.py): ffmpeg (a black box) wrote each <case>.webm
// and <case>.json records what that file carries per track and per packet (data as offsets into the file). The reader,
// given the file, must report the same track fields and the same packet bytes and discard padding, track by track.

using System.Text.Json;
using SpawnDev.Codecs.Container.Matroska;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod(Category = "CpuOnly")]
    public async Task MatroskaContainer_TrackFieldsAndDiscardPadding_MatchFfmpegFixtures()
    {
        await Task.Yield();
        var cases = System.Text.Encoding.UTF8.GetString(ReadEmbedded(WebMMuxPrefix + "manifest.txt"))
            .Split('\n').Select(l => l.Trim()).Where(l => l.Length > 0).ToList();
        var failures = new List<string>();
        int tracksChecked = 0, packetsChecked = 0, discards = 0;
        foreach (var name in cases)
        {
            var file = ReadEmbedded(WebMMuxPrefix + name + ".webm");
            using var json = JsonDocument.Parse(ReadEmbedded(WebMMuxPrefix + name + ".json"));
            var doc = json.RootElement;
            var mkv = new MatroskaContainer(new MemoryStream(file));
            var tracks = mkv.Tracks.ToDictionary(t => (int)t.TrackNumber);
            void Check(string what, object? want, object? got)
            {
                if (!Equals(want, got)) failures.Add($"{name}: {what}: want {want ?? "null"}, got {got ?? "null"}");
            }
            foreach (var t in doc.GetProperty("tracks").EnumerateArray())
            {
                int number = t.GetProperty("number").GetInt32();
                if (!tracks.TryGetValue(number, out var tr)) { failures.Add($"{name}: track {number} missing"); continue; }
                tracksChecked++;
                string p = $"track {number} ";
                Check(p + "codec", t.GetProperty("codec").GetString(), tr.CodecId);
                Check(p + "type", (ulong)t.GetProperty("type").GetInt32(), tr.TrackType);
                Check(p + "CodecPrivate", t.TryGetProperty("codecPrivate", out var cp) ? cp.GetString()!.ToUpperInvariant() : null,
                    tr.CodecPrivate == null ? null : Convert.ToHexString(tr.CodecPrivate));
                Check(p + "DefaultDuration", t.TryGetProperty("defaultDurationNs", out var dd) ? dd.GetUInt64() : (ulong?)null, tr.DefaultDurationNs);
                Check(p + "CodecDelay", t.TryGetProperty("codecDelayNs", out var cd) ? cd.GetUInt64() : (ulong?)null, tr.CodecDelayNs);
                if (t.GetProperty("type").GetInt32() == 1)
                {
                    Check(p + "PixelWidth", t.GetProperty("width").GetUInt64(), tr.PixelWidth);
                    Check(p + "PixelHeight", t.GetProperty("height").GetUInt64(), tr.PixelHeight);
                    bool alpha = t.TryGetProperty("alphaMode", out var am) && am.GetBoolean();
                    Check(p + "AlphaMode", alpha ? 1UL : (ulong?)null, tr.AlphaMode);
                }
                else
                {
                    Check(p + "SamplingFrequency", t.GetProperty("sampleRate").GetDouble(), tr.SamplingFrequency);
                    Check(p + "Channels", t.GetProperty("channels").GetUInt64(), tr.Channels);
                    Check(p + "BitDepth", t.TryGetProperty("bitDepth", out var bd) ? bd.GetUInt64() : (ulong?)null, tr.BitDepth);
                }
            }
            // packets, per track in order: same bytes, same discard padding
            var frames = mkv.Frames.GroupBy(f => (int)f.TrackNumber).ToDictionary(g => g.Key, g => g.ToList());
            foreach (var group in doc.GetProperty("packets").EnumerateArray().GroupBy(pk => pk.GetProperty("track").GetInt32()))
            {
                var want = group.ToList();
                var got = frames.TryGetValue(group.Key, out var g) ? g : new List<MatroskaFrame>();
                if (want.Count != got.Count) { failures.Add($"{name}: track {group.Key}: {got.Count} packets, want {want.Count}"); continue; }
                for (int k = 0; k < want.Count; k++)
                {
                    var d = want[k].GetProperty("data");
                    if (!file.AsSpan(d[0].GetInt32(), d[1].GetInt32()).SequenceEqual(got[k].Data))
                    {
                        failures.Add($"{name}: track {group.Key} packet {k}: bytes differ");
                        break;
                    }
                    long discard = want[k].GetProperty("discardNs").GetInt64();
                    if (discard != 0) discards++;
                    if ((got[k].DiscardPaddingNs ?? 0) != discard)
                        failures.Add($"{name}: track {group.Key} packet {k}: DiscardPadding want {discard}, got {got[k].DiscardPaddingNs?.ToString() ?? "null"}");
                    packetsChecked++;
                }
            }
        }
        True(cases.Count >= 21, $"only {cases.Count} fixtures");
        True(discards > 0, "no fixture packet carries DiscardPadding: the discard check is vacuous");
        True(failures.Count == 0, $"{failures.Count} mismatches ({tracksChecked} tracks, {packetsChecked} packets):\n" + string.Join("\n", failures.Take(40)));
    }
}
