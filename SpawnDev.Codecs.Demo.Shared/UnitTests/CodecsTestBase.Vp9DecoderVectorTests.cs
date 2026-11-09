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

    /// <summary>A VP9 test file and libvpx's MD5 per displayed frame.</summary>
    private sealed record Vp9Vector(string Name, byte[] File, string[] FrameMd5);

    /// <summary>
    /// Every VP9 test file: the embedded vectors and ComfyUI files, plus (desktop only) the four large vectors read from
    /// TestData/Vp9ConformanceLarge next to the runner.
    /// </summary>
    private static IEnumerable<Vp9Vector> Vp9Vectors()
    {
        var asm = typeof(CodecsTestBase).Assembly;
        foreach (string res in asm.GetManifestResourceNames().Order(StringComparer.Ordinal))
        {
            if (!(res.StartsWith(Vp9VectorPrefix) || res.StartsWith(Vp9ComfyPrefix))) continue;
            if (!(res.EndsWith(".webm") || res.EndsWith(".ivf"))) continue;
            string name = res[(res.StartsWith(Vp9VectorPrefix) ? Vp9VectorPrefix : Vp9ComfyPrefix).Length..];
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
        True(vectors >= (OperatingSystem.IsBrowser() ? 300 : 304), $"only {vectors} VP9 test files found");
        True(failures.Count == 0, string.Join("\n", failures));
    }
}
