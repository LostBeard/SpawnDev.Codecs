// Tests for the full VP9 decoder (Plans/PLAN-VP9-Decoder-Full.md) against libvpx's profile-0 conformance vectors and
// ComfyUI SaveWEBM files (TestData/Vp9Conformance/SOURCES.md). Every vector comes with libvpx's MD5 of every displayed
// frame.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.ILGPU;
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
        // diagnostics: VP9_FILE_FILTER=substring limits every VP9 test to the matching files (desktop)
        string? only = OperatingSystem.IsBrowser() ? null : Environment.GetEnvironmentVariable("VP9_FILE_FILTER");
        return only == null ? Vp9VectorsAll() : Vp9VectorsAll().Where(v => v.Name.Contains(only));
    }

    private static IEnumerable<Vp9Vector> Vp9VectorsAll()
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

    /// <summary>MD5 (hex) of <paramref name="bytes"/>, on every platform (FlacMd5: System.Security.Cryptography.MD5 is not in WASM).</summary>
    private static string Md5Hex(byte[] bytes)
    {
        var samples = new int[bytes.Length];
        for (int i = 0; i < bytes.Length; i++) samples[i] = (sbyte)bytes[i];
        return Convert.ToHexString(SpawnDev.Codecs.Audio.Flac.FlacMd5.Compute(samples, 8)).ToLowerInvariant();
    }

    /// <summary>One decoded frame in libvpx_modeinfo.tsv.gz: mode info MD5, counts MD5 (or "-"), context after adaptation,
    /// reconstructed frame before (Pre) and after (Post) the loop filter.</summary>
    private sealed class Vp9FrameRef
    {
        public string Mi = "", Counts = "", Adapted = "", Pre = "", Post = "";
    }

    /// <summary>libvpx_modeinfo.tsv.gz: file name -> per decoded frame (M / A / P / L lines of libvpx_ref -m).</summary>
    private static Dictionary<string, List<Vp9FrameRef>> LibvpxModeInfo()
    {
        using var gz = new System.IO.Compression.GZipStream(new MemoryStream(ReadEmbedded(Vp9VectorPrefix + "libvpx_modeinfo.tsv.gz")),
            System.IO.Compression.CompressionMode.Decompress);
        using var reader = new StreamReader(gz);
        var all = new Dictionary<string, List<Vp9FrameRef>>();
        string? line;
        while ((line = reader.ReadLine()) != null)
        {
            int tab = line.IndexOf('\t');
            string file = line[..tab];
            var parts = line[(tab + 1)..].Split(' ');
            if (!all.TryGetValue(file, out var list)) all[file] = list = new();
            int n = int.Parse(parts[1]);   // 1-based decoded frame number
            while (list.Count < n) list.Add(new Vp9FrameRef());
            var f = list[n - 1];
            switch (parts[0])
            {
                case "M": f.Mi = parts[2]["mi=".Length..]; f.Counts = parts[3]["counts=".Length..]; break;
                case "A": f.Adapted = parts[2]; break;
                case "P": f.Pre = parts[2]; break;
                case "L": f.Post = parts[2]; break;
            }
        }
        return all;
    }

    /// <summary>Our mode info grid serialized as libvpx_ref.c mi_hook does (see there), as bytes.</summary>
    private static byte[] Vp9ModeInfoBytes(int[] mi, int miCols, int miRows, int gridOffset)
    {
        var v = new List<int>(miCols * miRows * 12);
        for (int r = 0; r < miRows; r++)
            for (int c = 0; c < miCols; c++)
            {
                int b = mi[gridOffset + r * miCols + c] * Vp9Mi.Stride;
                bool inter = mi[b + Vp9Mi.Ref0] > 0;
                v.Add(mi[b + Vp9Mi.SbType]); v.Add(mi[b + Vp9Mi.SegmentId]); v.Add(mi[b + Vp9Mi.Skip]); v.Add(mi[b + Vp9Mi.TxSize]);
                v.Add(inter ? 1 : 0);
                if (!inter)
                {
                    v.Add(mi[b + Vp9Mi.Mode]); v.Add(mi[b + Vp9Mi.UvMode]);
                    if (mi[b + Vp9Mi.SbType] < 3) for (int k = 0; k < 4; k++) v.Add(mi[b + Vp9Mi.BmiMode + k]);
                }
                else
                {
                    v.Add(mi[b + Vp9Mi.Mode]); v.Add(mi[b + Vp9Mi.Ref0]); v.Add(mi[b + Vp9Mi.Ref1]); v.Add(mi[b + Vp9Mi.Interp]);
                    bool compound = mi[b + Vp9Mi.Ref1] > 0;   // second MVs only for compound blocks (undefined otherwise)
                    v.Add(mi[b + Vp9Mi.Mv0]);
                    if (compound) v.Add(mi[b + Vp9Mi.Mv1]);
                    if (mi[b + Vp9Mi.SbType] < 3)
                        for (int k = 0; k < 4; k++) { v.Add(mi[b + Vp9Mi.BmiMv0 + k]); if (compound) v.Add(mi[b + Vp9Mi.BmiMv1 + k]); }
                }
            }
        var bytes = new byte[v.Count * 4];
        Buffer.BlockCopy(v.ToArray(), 0, bytes, 0, bytes.Length);
        return bytes;
    }

    /// <summary>
    /// S2: the tile parse kernel (partitions, mode info, MVs, tokens, counts) gives libvpx's mode info grid (MD5 over the
    /// fields each block type defines) on every frame of every VP9 test file, and libvpx's FRAME_COUNTS on frames with
    /// backward adaptation. Each frame starts from libvpx's loaded context (backward adaptation is S4); everything else
    /// (header state, compressed header, segmentation maps, previous-frame MVs) is ours.
    /// </summary>
    [TestMethod(Timeout = 3600000, Category = "NotOnWebGL")]
    public async Task Vp9Parse_EveryFrame_MatchesLibvpxModeInfo()
    {
        var contextsRef = LibvpxContexts();
        var modeInfoRef = LibvpxModeInfo();
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var contexts = new Vp9FrameContextsGpu(acc);
            using var parser = new Vp9TileParserGpu(acc);
            await contexts.LoadKernelsAsync();
            await parser.LoadKernelsAsync();
            var failures = new List<string>();
            int frames = 0, countsChecked = 0;
            foreach (var v in Vp9Vectors())
            {
                if (!contextsRef.TryGetValue(v.Name, out var recs) || !modeInfoRef.TryGetValue(v.Name, out var want))
                { failures.Add($"{v.Name}: no libvpx reference"); continue; }
                var reader = new Vp9HeaderReader();
                int rec = 0, decoded = 0, prevBuffer = -1;
                foreach (byte[] packet in Vp9Packets.ReadFile(v.File))
                {
                    using var buf = acc.Allocate1D(packet);
                    foreach (var (offset, length) in Vp9Packets.Split(packet))
                    {
                        var h = reader.ReadFrame(packet.AsSpan(offset, length));
                        if (h.ShowExistingFrame) { reader.FinishFrame(); continue; }
                        contexts.BeginFrame(h);
                        var cur = contexts.Contexts.SubView(Vp9Ctx.Current * Vp9Ctx.Bytes, Vp9Ctx.Bytes);
                        // our own context chain (resets, loads, adaptation, stores) must load what libvpx loads
                        var loaded = await cur.CopyToHostAsync();
                        int ld = Enumerable.Range(0, Vp9Ctx.Bytes).FirstOrDefault(k => loaded[k] != recs[rec].Context[k], -1);
                        if (ld >= 0) failures.Add($"{v.Name} frame {decoded + 1}: loaded context differs at {ld}: {loaded[ld]} vs libvpx {recs[rec].Context[ld]}");
                        rec += 2;
                        contexts.ReadCompressedHeader(buf.View, offset, h);
                        parser.Parse(buf.View, offset, packet.AsSpan(offset, length), h, contexts, prevBuffer);
                        var status = await parser.ReadStatusAsync();
                        var mi = await parser.ModeInfo.CopyToHostAsync();
                        var miBytes = Vp9ModeInfoBytes(mi, parser.MiCols, parser.MiRows, parser.GridOffset);
                        string miMd5 = Md5Hex(miBytes);
                        // diagnostics: VP9_MI_DUMP=dir and VP9_MI_FILE=name write ours_<n>.bin (compare with libvpx_ref -d)
                        string? dumpDir = OperatingSystem.IsBrowser() ? null : Environment.GetEnvironmentVariable("VP9_MI_DUMP");
                        if (dumpDir != null && v.Name == Environment.GetEnvironmentVariable("VP9_MI_FILE"))
                            File.WriteAllBytes(Path.Combine(dumpDir, $"ours_{decoded + 1}.bin"), miBytes);
                        var w = want[decoded];
                        if (status.Any(s => s != 0)) failures.Add($"{v.Name} frame {decoded + 1}: parse status {string.Join(",", status)}");
                        else if (miMd5 != w.Mi) failures.Add($"{v.Name} frame {decoded + 1}: mode info differs ({h.Width}x{h.Height}, type {h.FrameType}, intra-only {h.IntraOnly})");
                        if (w.Counts != "-")
                        {
                            var counts = await parser.Counts.SubView(0, (long)parser.TileCols * Vp9Cnt.Size).CopyToHostAsync();
                            var sum = new int[Vp9Cnt.Size];
                            for (int tc = 0; tc < parser.TileCols; tc++)
                                for (int k = 0; k < Vp9Cnt.Size; k++) sum[k] += counts[tc * Vp9Cnt.Size + k];
                            var bytes = new byte[sum.Length * 4];
                            Buffer.BlockCopy(sum, 0, bytes, 0, bytes.Length);
                            if (Md5Hex(bytes) != w.Counts) failures.Add($"{v.Name} frame {decoded + 1}: counts differ");
                            countsChecked++;
                        }
                        // backward adaptation (S4): the current context afterwards == libvpx's (A line)
                        contexts.Adapt(h, parser.Counts, parser.TileCols);
                        var adapted = await cur.CopyToHostAsync();
                        if (Md5Hex(adapted.Select(x => (byte)x).ToArray()) != w.Adapted)
                            failures.Add($"{v.Name} frame {decoded + 1}: context after adaptation differs (adaptation {(!h.ErrorResilient && !h.FrameParallel ? "on" : "off")})");
                        contexts.EndFrame(h);
                        parser.EndFrame(h);
                        reader.FinishFrame();
                        prevBuffer = h.NewBuffer;
                        decoded++;
                        frames++;
                        if (failures.Count > 40) goto done;
                    }
                }
            }
        done:
            Console.WriteLine($"{acc.AcceleratorType}: {frames} frames parsed, {countsChecked} counts checked");
            True(failures.Count == 0, $"{failures.Count} mismatches:\n" + string.Join("\n", failures.Take(40)));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    /// <summary>
    /// S3/S4 reconstruction: the decode pipeline's frame before the loop filter equals libvpx's (P line, cropped planes) on
    /// every frame whose references are not loop-filtered yet (all earlier frames of the stream had filter level 0; the
    /// first frame of every stream qualifies). Covers intra prediction, inverse transforms, inter prediction and their
    /// orchestration; the loop filter extends this to every frame.
    /// </summary>
    [TestMethod(Timeout = 3600000, Category = "NotOnWebGL")]
    public async Task Vp9Decode_EveryFrame_MatchesLibvpx()
    {
        var modeInfoRef = LibvpxModeInfo();
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            var failures = new List<string>();
            int frames = 0, outputs = 0, files = 0;
            foreach (var v in Vp9Vectors())
            {
                if (!modeInfoRef.TryGetValue(v.Name, out var want)) { failures.Add($"{v.Name}: no libvpx reference"); continue; }
                files++;
                using var pipe = new Vp9DecodePipeline(acc);
                if (!OperatingSystem.IsBrowser() && int.TryParse(Environment.GetEnvironmentVariable("VP9_DEBUG_FLAGS"), out int dbg)) pipe.DebugFlags = dbg;
                int decoded = 0, shown = 0;
                foreach (byte[] packet in Vp9Packets.ReadFile(v.File))
                {
                    using var buf = acc.Allocate1D(packet);
                    int show = -1;   // libvpx outputs the packet's last shown frame
                    foreach (var (offset, length) in Vp9Packets.Split(packet))
                    {
                        int n = decoded;
                        var h = await pipe.DecodeFrameAsync(packet.AsMemory(offset, length), buf.View, offset, async fh =>
                        {
                            var pre = await pipe.ReadPlanesAsync(fh.NewBuffer);
                            // diagnostics: VP9_DUMP_PLANES=dir writes <backend>_<file>_<n>.yuv (pre loop filter)
                            string? dump = OperatingSystem.IsBrowser() ? null : Environment.GetEnvironmentVariable("VP9_DUMP_PLANES");
                            if (dump != null) File.WriteAllBytes(Path.Combine(dump, $"{acc.AcceleratorType}_{v.Name}_{n + 1}.yuv"), pre);
                            if (Md5Hex(pre) != want[n].Pre)
                                failures.Add($"{v.Name} frame {n + 1}: reconstruction differs ({fh.Width}x{fh.Height}, type {fh.FrameType}, intra-only {fh.IntraOnly}, lossless {fh.Lossless})");
                        });
                        if (h.ShowExistingFrame) { show = h.FrameToShowBuffer; continue; }
                        if (Md5Hex(await pipe.ReadPlanesAsync(h.NewBuffer)) != want[n].Post)
                            failures.Add($"{v.Name} frame {n + 1}: loop-filtered frame differs (filter level {h.LoopFilter.Level}, sharpness {h.LoopFilter.Sharpness})");
                        if (h.ShowFrame) show = h.NewBuffer;
                        decoded++;
                        frames++;
                        if (failures.Count > 40) goto done;
                    }
                    if (show >= 0)
                    {
                        if (shown >= v.FrameMd5.Length) failures.Add($"{v.Name}: more output frames than libvpx's {v.FrameMd5.Length}");
                        else if (Md5Hex(await pipe.ReadPlanesAsync(show)) != v.FrameMd5[shown])
                            failures.Add($"{v.Name} output frame {shown + 1}: MD5 differs from the vector's .md5");
                        shown++;
                        outputs++;
                    }
                }
                if (shown != v.FrameMd5.Length) failures.Add($"{v.Name}: {shown} output frames, libvpx {v.FrameMd5.Length}");
            }
        done:
            Console.WriteLine($"{acc.AcceleratorType}: {files} files, {frames} frames decoded, {outputs} output frames == libvpx .md5");
            True(failures.Count == 0, $"{failures.Count} mismatches:\n" + string.Join("\n", failures.Take(40)));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    /// <summary>
    /// The public decoder (<see cref="Vp9DecoderGpu.DecodeFileAsync"/> and the ArrayView overload) gives libvpx's pictures:
    /// the ComfyUI SaveWEBM files and the vectors with the trickiest output rules (show-existing, intra-only, resizing,
    /// big superframes, odd sizes).
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "NotOnWebGL")]
    public async Task Vp9DecoderGpu_PublicApi_MatchesLibvpx()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            string[] picks = { "vp90-2-17-show-existing-frame.webm", "vp90-2-16-intra-only.webm", "vp90-2-05-resize.ivf",
                "vp90-2-20-big_superframe-01.webm", "vp90-2-02-size-66x66.webm", "vp90-2-03-size-226x198.webm" };
            var failures = new List<string>();
            int pictures = 0;
            foreach (var v in Vp9Vectors().Where(x => x.Name.StartsWith("comfyui_") || picks.Contains(x.Name)))
            {
                using var dec = new Vp9DecoderGpu(acc);
                int n = 0;
                await foreach (var frame in dec.DecodeFileAsync(v.File))
                {
                    if (n >= v.FrameMd5.Length) { failures.Add($"{v.Name}: more pictures than libvpx"); break; }
                    if (frame.Planes.LongLength != frame.Format.PlanarBytes) failures.Add($"{v.Name} picture {n + 1}: {frame.Planes.Length} bytes");
                    else if (Md5Hex(frame.Planes) != v.FrameMd5[n]) failures.Add($"{v.Name} picture {n + 1}: MD5 differs ({frame.Format})");
                    n++;
                    pictures++;
                }
                if (n != v.FrameMd5.Length) failures.Add($"{v.Name}: {n} pictures, libvpx {v.FrameMd5.Length}");
                // the ArrayView overload on the same stream: pictures stay on the accelerator
                using var dec2 = new Vp9DecoderGpu(acc);
                using var dst = acc.Allocate1D<byte>(64L * 1024 * 1024 / 16);
                int k = 0;
                foreach (byte[] packet in Vp9Packets.ReadFile(v.File))
                {
                    var format = await dec2.DecodePacketAsync(packet, dst.View);
                    if (format is not { } f) continue;
                    if (k < v.FrameMd5.Length && Md5Hex(await dst.View.SubView(0, f.PlanarBytes).CopyToHostAsync()) != v.FrameMd5[k])
                        failures.Add($"{v.Name} picture {k + 1}: ArrayView overload differs");
                    k++;
                }
                if (k != v.FrameMd5.Length) failures.Add($"{v.Name}: ArrayView overload gave {k} pictures, libvpx {v.FrameMd5.Length}");
            }
            Console.WriteLine($"{acc.AcceleratorType}: {pictures} pictures == libvpx");
            True(pictures > 300, $"only {pictures} pictures");
            True(failures.Count == 0, $"{failures.Count} mismatches:\n" + string.Join("\n", failures.Take(40)));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    /// <summary>
    /// WebM alpha (ComfyUI SaveWEBM with RGBA input: AlphaMode 1, a second VP9 stream in BlockAdditional): the alpha planes
    /// <see cref="Vp9DecoderGpu.DecodeFileAsync"/> returns equal libvpx's (FFmpeg's libvpx-vp9 decoder, yuva420p A plane,
    /// comfyui_vp9_rgba_512x320_crf32.webm.alpha.md5), and the colour pictures still equal the .md5.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "NotOnWebGL")]
    public async Task Vp9DecoderGpu_WebMAlpha_MatchesLibvpx()
    {
        const string name = "comfyui_vp9_rgba_512x320_crf32.webm";
        byte[] file = ReadEmbedded(Vp9ComfyPrefix + name);
        string[] colour = Md5Lines(ReadEmbedded(Vp9ComfyPrefix + name + ".md5"));
        string[] alpha = Md5Lines(ReadEmbedded(Vp9ComfyPrefix + name + ".alpha.md5"));
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var dec = new Vp9DecoderGpu(acc);
            var failures = new List<string>();
            int n = 0;
            await foreach (var frame in dec.DecodeFileAsync(file))
            {
                if (n >= colour.Length) { failures.Add("more pictures than libvpx"); break; }
                if (Md5Hex(frame.Planes) != colour[n]) failures.Add($"picture {n + 1}: colour differs");
                if (frame.Alpha == null) failures.Add($"picture {n + 1}: no alpha");
                else if (frame.Alpha.LongLength != (long)frame.Format.Width * frame.Format.Height) failures.Add($"picture {n + 1}: alpha has {frame.Alpha.Length} bytes");
                else if (Md5Hex(frame.Alpha) != alpha[n]) failures.Add($"picture {n + 1}: alpha differs");
                n++;
            }
            if (n != colour.Length) failures.Add($"{n} pictures, libvpx {colour.Length}");
            Console.WriteLine($"{acc.AcceleratorType}: {n} pictures with alpha");
            True(failures.Count == 0, string.Join("\n", failures));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    /// <summary>One libvpx_ref -c record: stage 0 = the context a frame loads, stage 1 = after its compressed header
    /// (+ tx_mode, reference_mode, comp_fixed_ref, comp_var_ref[0..1]).</summary>
    private sealed record Vp9CtxRecord(int Stage, byte[] Context, byte[] Extra);

    /// <summary>TestData/Vp9Conformance/libvpx_contexts.gz (scripts/video/libvpx_ref/pack_contexts.py): file name -> records.</summary>
    private static Dictionary<string, List<Vp9CtxRecord>> LibvpxContexts()
    {
        const int ctxBytes = 2040;
        using var gz = new System.IO.Compression.GZipStream(new MemoryStream(ReadEmbedded(Vp9VectorPrefix + "libvpx_contexts.gz")),
            System.IO.Compression.CompressionMode.Decompress);
        var ms = new MemoryStream();
        gz.CopyTo(ms);
        byte[] d = ms.ToArray();
        var all = new Dictionary<string, List<Vp9CtxRecord>>();
        int at = 0;
        while (at < d.Length)
        {
            int nameLen = d[at] | (d[at + 1] << 8);
            string name = System.Text.Encoding.ASCII.GetString(d, at + 2, nameLen);
            at += 2 + nameLen;
            int count = BitConverter.ToInt32(d, at);
            at += 4;
            var list = new List<Vp9CtxRecord>(count);
            var prev = new byte[ctxBytes];
            for (int i = 0; i < count; i++)
            {
                int stage = d[at++];
                var ctxBytesArr = new byte[ctxBytes];
                for (int k = 0; k < ctxBytes; k++) ctxBytesArr[k] = (byte)(d[at + k] ^ prev[k]);
                at += ctxBytes;
                byte[] extra = Array.Empty<byte>();
                if (stage == 1) { extra = d.AsSpan(at, 5).ToArray(); at += 5; }
                list.Add(new Vp9CtxRecord(stage, ctxBytesArr, extra));
                prev = ctxBytesArr;
            }
            all[name] = list;
        }
        return all;
    }

    /// <summary>
    /// S1 (accelerator half): for every frame of every VP9 test file, the compressed header kernel turns libvpx's loaded
    /// context into libvpx's context after the compressed header, byte for byte, with the same tx_mode, reference_mode
    /// and compound references; and while no earlier frame of the stream used backward adaptation (S4), our own context
    /// handling (past-independence resets, load, store) loads exactly the context libvpx loads.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "NotOnWebGL")]
    public async Task Vp9CompressedHeader_EveryFrame_MatchesLibvpx()
    {
        var reference = LibvpxContexts();
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var contexts = new Vp9FrameContextsGpu(acc);
            await contexts.LoadKernelsAsync();
            var failures = new List<string>();
            int frames = 0, managedChecked = 0;
            foreach (var v in Vp9Vectors())
            {
                if (!reference.TryGetValue(v.Name, out var recs)) { failures.Add($"{v.Name}: no libvpx context dump"); continue; }
                var reader = new Vp9HeaderReader();
                int rec = 0, n = 0;
                bool adaptationFree = true;
                foreach (byte[] packet in Vp9Packets.ReadFile(v.File))
                {
                    using var buf = acc.Allocate1D(packet);
                    foreach (var (offset, length) in Vp9Packets.Split(packet))
                    {
                        n++;
                        var h = reader.ReadFrame(packet.AsSpan(offset, length));
                        if (h.ShowExistingFrame) { reader.FinishFrame(); continue; }
                        if (rec + 1 >= recs.Count || recs[rec].Stage != 0 || recs[rec + 1].Stage != 1)
                        { failures.Add($"{v.Name} frame {n}: context records out of step"); goto nextFile; }
                        var loaded = recs[rec].Context;
                        var after = recs[rec + 1];
                        rec += 2;

                        contexts.BeginFrame(h);
                        var cur = contexts.Contexts.SubView(Vp9Ctx.Current * Vp9Ctx.Bytes, Vp9Ctx.Bytes);
                        if (adaptationFree)
                        {
                            var ours = await cur.CopyToHostAsync();
                            int bad = Enumerable.Range(0, Vp9Ctx.Bytes).FirstOrDefault(k => ours[k] != loaded[k], -1);
                            if (bad >= 0) failures.Add($"{v.Name} frame {n}: loaded context differs at {bad}: {ours[bad]} vs libvpx {loaded[bad]}");
                            managedChecked++;
                        }
                        cur.CopyFromCPU(loaded.Take(Vp9Ctx.Bytes).Select(b => (int)b).ToArray());   // test: start from libvpx's
                        contexts.ReadCompressedHeader(buf.View, offset, h);
                        var res = await contexts.ReadResultsAsync();
                        var got = await cur.CopyToHostAsync();
                        int diff = Enumerable.Range(0, Vp9Ctx.Bytes).FirstOrDefault(k => got[k] != after.Context[k], -1);
                        if (diff >= 0) failures.Add($"{v.Name} frame {n}: context after the compressed header differs at {diff}: {got[diff]} vs libvpx {after.Context[diff]}");
                        if (res.TxMode != after.Extra[0] || res.ReferenceMode != after.Extra[1] || res.Error)
                            failures.Add($"{v.Name} frame {n}: tx_mode {res.TxMode} ref_mode {res.ReferenceMode} error {res.Error} vs libvpx {after.Extra[0]} {after.Extra[1]}");
                        if (res.ReferenceMode != 0 && (res.CompFixedRef != after.Extra[2] || res.CompVarRef0 != after.Extra[3] || res.CompVarRef1 != after.Extra[4]))
                            failures.Add($"{v.Name} frame {n}: compound refs {res.CompFixedRef},{res.CompVarRef0},{res.CompVarRef1} vs libvpx {after.Extra[2]},{after.Extra[3]},{after.Extra[4]}");
                        contexts.EndFrame(h);
                        reader.FinishFrame();
                        if (!h.ErrorResilient && !h.FrameParallel) adaptationFree = false;
                        frames++;
                        if (failures.Count > 40) goto done;
                    }
                }
                if (rec != recs.Count) failures.Add($"{v.Name}: {rec} of {recs.Count} context records used");
            nextFile:;
            }
        done:
            Console.WriteLine($"{acc.AcceleratorType}: {frames} compressed headers compared, {managedChecked} loaded contexts checked");
            True(failures.Count == 0, $"{failures.Count} mismatches:\n" + string.Join("\n", failures.Take(40)));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }
}
