// Tests for the full VP9 decoder (Plans/PLAN-VP9-Decoder-Full.md) against libvpx's profile-0 conformance vectors and
// ComfyUI SaveWEBM files (TestData/Vp9Conformance/SOURCES.md). Every vector comes with libvpx's MD5 of every displayed
// frame.

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.Codecs.Video.Vp9.Decoder;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private const string Vp9VectorPrefix = "SpawnDev.Codecs.Demo.Shared.TestData.Vp9Conformance.";
    private const string Vp9ComfyPrefix = "SpawnDev.Codecs.Demo.Shared.TestData.Vp9ComfyUI.";
    private const string Vp9SynthPrefix = "SpawnDev.Codecs.Demo.Shared.TestData.Vp9Synthetic.";
    private static readonly string[] Vp9Prefixes = { Vp9VectorPrefix, Vp9ComfyPrefix, Vp9SynthPrefix };

    /// <summary>A VP9 test file and libvpx's MD5 per displayed frame.</summary>
    private sealed record Vp9Vector(string Name, byte[] File, string[] FrameMd5);

    /// <summary>
    /// Every VP9 test file: the embedded vectors, ComfyUI files and synthetic header fixtures
    /// (scripts/video/make_vp9_header_fixtures.py), plus (desktop only) the four large vectors read from
    /// TestData/Vp9ConformanceLarge next to the runner.
    /// </summary>
    private static IEnumerable<Vp9Vector> Vp9Vectors()
    {
        var asm = typeof(CodecsTestBase).Assembly;
        foreach (string res in asm.GetManifestResourceNames().Order(StringComparer.Ordinal))
        {
            string? prefix = Vp9Prefixes.FirstOrDefault(p => res.StartsWith(p));
            if (prefix is null || !(res.EndsWith(".webm") || res.EndsWith(".ivf"))) continue;
            string name = res[prefix.Length..];
            yield return new Vp9Vector(name, ReadEmbedded(res), Md5Lines(ReadEmbedded(res + ".md5")));
        }
        if (OperatingSystem.IsBrowser()) yield break;
        string dir = Path.Combine(AppContext.BaseDirectory, "TestData", "Vp9ConformanceLarge");
        if (!Directory.Exists(dir)) throw new DirectoryNotFoundException($"Large VP9 vectors missing: {dir}");
        foreach (string path in Directory.GetFiles(dir).Where(p => p.EndsWith(".webm") || p.EndsWith(".ivf")).Order(StringComparer.Ordinal))
            yield return new Vp9Vector(Path.GetFileName(path), File.ReadAllBytes(path), Md5Lines(File.ReadAllBytes(path + ".md5")));
    }

    private static string[] Md5Lines(byte[] md5File) =>
        System.Text.Encoding.ASCII.GetString(md5File).Split('\n').Select(l => l.Trim()).Where(l => l.Length > 0)
            .Select(l => l.Split(' ')[0]).ToArray();

    /// <summary>
    /// S0: our demux (WebM / IVF) + <see cref="Vp9Packets.Split"/> turn every vector into frames, and the packets
    /// holding a shown frame (show_frame, or show_existing_frame) are exactly the frames libvpx outputs (its .md5 line
    /// count; libvpx outputs one frame per packet).
    /// </summary>
    [TestMethod(Timeout = 600000)]
    public async Task Vp9Packets_EveryVector_ShownFramesMatchLibvpxOutputCount()
    {
        await Task.Yield();
        var failures = new List<string>();
        int vectors = 0, frames = 0, superframes = 0;
        foreach (var v in Vp9Vectors())
        {
            vectors++;
            int shown = 0;
            try
            {
                foreach (byte[] packet in Vp9Packets.ReadFile(v.File))
                {
                    var parts = Vp9Packets.Split(packet);
                    if (parts.Count > 1) superframes++;
                    bool packetShows = false;
                    foreach (var (offset, length) in parts)
                    {
                        frames++;
                        var h = Vp9FrameHeaderParser.Parse(packet.AsSpan(offset, length));
                        packetShows |= h.ShowExistingFrame || h.ShowFrame;
                    }
                    // libvpx outputs at most one frame per packet, the last shown one (vp9_get_raw_frame after the
                    // whole packet): the SVC vectors show every spatial layer, only the top one comes out.
                    if (packetShows) shown++;
                }
            }
            catch (Exception e) { failures.Add($"{v.Name}: {e.GetType().Name}: {e.Message}"); continue; }
            if (shown != v.FrameMd5.Length) failures.Add($"{v.Name}: {shown} shown frames, libvpx outputs {v.FrameMd5.Length}");
        }
        Console.WriteLine($"{vectors} files, {frames} frames, {superframes} superframes");
        True(vectors >= (OperatingSystem.IsBrowser() ? 307 : 311), $"only {vectors} VP9 test files found");
        True(failures.Count == 0, string.Join("\n", failures));
    }

    /// <summary>libvpx_ref -h's header dump of every frame of every VP9 test file: file name -> "key=value" maps in decode order.</summary>
    private static Dictionary<string, List<Dictionary<string, string>>> LibvpxHeaders()
    {
        using var gz = new System.IO.Compression.GZipStream(new MemoryStream(ReadEmbedded(Vp9VectorPrefix + "libvpx_headers.tsv.gz")),
            System.IO.Compression.CompressionMode.Decompress);
        using var reader = new StreamReader(gz);
        var all = new Dictionary<string, List<Dictionary<string, string>>>();
        string? line;
        while ((line = reader.ReadLine()) != null)
        {
            int tab = line.IndexOf('\t');
            string file = line[..tab];
            var fields = new Dictionary<string, string>();
            foreach (string kv in line[(tab + 1)..].Split(' ').Skip(3))   // "H <packet> <frame>"
            {
                int eq = kv.IndexOf('=');
                if (eq > 0) fields[kv[..eq]] = kv[(eq + 1)..];
            }
            if (!all.TryGetValue(file, out var list)) all[file] = list = new();
            list.Add(fields);
        }
        return all;
    }

    /// <summary>Our header fields in libvpx_ref -h's format (uncompressed header + derived state; not the compressed header).</summary>
    private static Dictionary<string, string> Vp9HeaderFields(Vp9FrameInfo h)
    {
        var f = new Dictionary<string, string> { ["show_existing"] = h.ShowExistingFrame ? "1" : "0" };
        if (h.ShowExistingFrame) return f;
        var lf = h.LoopFilter;
        var seg = h.Segmentation;
        f["profile"] = h.Profile.ToString();
        f["bit_depth"] = h.BitDepth.ToString();
        f["frame_type"] = h.FrameType.ToString();
        f["show"] = h.ShowFrame ? "1" : "0";
        f["intra_only"] = h.IntraOnly ? "1" : "0";
        f["error_res"] = h.ErrorResilient ? "1" : "0";
        f["size"] = $"{h.Width}x{h.Height}";
        f["render"] = $"{h.RenderWidth}x{h.RenderHeight}";
        f["ss"] = $"{h.SubsamplingX},{h.SubsamplingY}";
        if (h.FrameType != 0) f["reset_ctx"] = h.ResetFrameContext.ToString();   // libvpx leaves it stale on key frames
        f["refresh"] = $"0x{h.RefreshFrameFlags:x2}";
        f["sign_bias"] = $"{h.SignBias[1]},{h.SignBias[2]},{h.SignBias[3]}";
        if (!h.IsIntraOnlyFrame)   // read only by inter frames; libvpx leaves them stale otherwise
        {
            f["allow_hp"] = h.AllowHighPrecisionMv ? "1" : "0";
            f["interp"] = ((int)h.InterpFilter).ToString();
            f["ref_sizes"] = string.Join(",", h.RefSizes.Select(s => $"{s.Width}x{s.Height}"));
        }
        f["refresh_ctx"] = h.RefreshFrameContext ? "1" : "0";
        f["frame_parallel"] = h.FrameParallel ? "1" : "0";
        f["ctx_idx"] = h.FrameContextIdx.ToString();
        f["lf_level"] = lf.Level.ToString();
        f["sharpness"] = lf.Sharpness.ToString();
        f["lf_delta_enabled"] = lf.DeltaEnabled ? "1" : "0";
        f["ref_deltas"] = string.Join(",", lf.RefDeltas);
        f["mode_deltas"] = string.Join(",", lf.ModeDeltas);
        f["base_q"] = h.BaseQIndex.ToString();
        f["dq"] = $"{h.YDcDeltaQ},{h.UvDcDeltaQ},{h.UvAcDeltaQ}";
        f["seg"] = seg.Enabled ? "1" : "0";
        f["seg_update_map"] = seg.UpdateMap ? "1" : "0";
        f["seg_temporal"] = seg.TemporalUpdate ? "1" : "0";
        f["seg_abs"] = seg.AbsDelta ? "1" : "0";
        f["tile_log2"] = $"{h.Log2TileCols},{h.Log2TileRows}";
        f["mi"] = $"{h.MiCols}x{h.MiRows}";
        f["use_prev_mvs"] = h.UsePrevFrameMvs ? "1" : "0";
        f["lossless"] = h.Lossless ? "1" : "0";
        f["dequant"] = string.Join(";", Enumerable.Range(0, h.Dequant.GetLength(0))
            .Select(s => $"{h.Dequant[s, 0]},{h.Dequant[s, 1]},{h.Dequant[s, 2]},{h.Dequant[s, 3]}"));
        if (seg.Enabled)
        {
            f["seg_tree"] = string.Join(",", seg.TreeProbs);
            f["seg_pred"] = string.Join(",", seg.PredProbs);
            f["seg_features"] = string.Join(";", Enumerable.Range(0, 8)
                .Select(s => $"{seg.FeatureMask[s]:x},{seg.FeatureData[s, 0]},{seg.FeatureData[s, 1]},{seg.FeatureData[s, 2]},{seg.FeatureData[s, 3]}"));
        }
        return f;
    }

    /// <summary>
    /// S1: <see cref="Vp9HeaderReader"/> reads every frame of every VP9 test file to libvpx's parsed values (uncompressed
    /// header, loop filter / segmentation state carried between frames, reference frame sizes through the reference
    /// map, use_prev_frame_mvs, lossless, per-segment dequantizers).
    /// </summary>
    [TestMethod(Timeout = 600000)]
    public async Task Vp9HeaderReader_EveryFrame_MatchesLibvpx()
    {
        await Task.Yield();
        var reference = LibvpxHeaders();
        var failures = new List<string>();
        int frames = 0;
        foreach (var v in Vp9Vectors())
        {
            if (!reference.TryGetValue(v.Name, out var want)) { failures.Add($"{v.Name}: no libvpx header dump"); continue; }
            var reader = new Vp9HeaderReader();
            int n = 0;
            try
            {
                foreach (byte[] packet in Vp9Packets.ReadFile(v.File))
                    foreach (var (offset, length) in Vp9Packets.Split(packet))
                    {
                        var h = reader.ReadFrame(packet.AsSpan(offset, length));
                        reader.FinishFrame();
                        if (n >= want.Count) { failures.Add($"{v.Name}: more frames than libvpx's {want.Count}"); break; }
                        var bad = Vp9HeaderFields(h).Where(kv => !want[n].TryGetValue(kv.Key, out var w) || w != kv.Value)
                            .Select(kv => $"{kv.Key}={kv.Value} (libvpx {want[n].GetValueOrDefault(kv.Key, "-")})").ToList();
                        if (bad.Count > 0) failures.Add($"{v.Name} frame {n + 1}: {string.Join(", ", bad.Take(6))}");
                        n++;
                        frames++;
                    }
            }
            catch (Exception e) { failures.Add($"{v.Name} frame {n + 1}: {e.GetType().Name}: {e.Message}"); }
            if (n != want.Count && failures.Count == 0) failures.Add($"{v.Name}: {n} frames, libvpx {want.Count}");
        }
        Console.WriteLine($"{frames} frame headers compared");
        True(failures.Count == 0, $"{failures.Count} mismatches:\n" + string.Join("\n", failures.Take(40)));
    }
}
