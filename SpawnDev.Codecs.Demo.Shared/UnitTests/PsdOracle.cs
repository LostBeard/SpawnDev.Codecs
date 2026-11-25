// psd-tools (MIT licence) as a black-box reader of the PSD writer's files (desktop only): runs scripts/images/psd_layers.py in the oracle Python.
// (Pillow's PSD plugin is not used: it takes each layer's channels as R, G, B, A in file order, so it misreads real layouts, which put alpha first.)

using System.Diagnostics;
using System.Runtime.Versioning;
using System.Text.Json;
using SpawnDev.UnitTesting;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

internal static class PsdOracle
{
    internal sealed record Layer(string Name, int Left, int Top, int Right, int Bottom, bool Visible, int Opacity, string Blend, int Width, int Height);
    internal sealed record Info(int Width, int Height, List<Layer> Layers, int CompositeWidth, int CompositeHeight);

    /// <summary>First statement of an oracle test: skip when no oracle Python is set (or in a browser).</summary>
    public static void RequireAvailable()
    {
        if (OperatingSystem.IsBrowser()) throw new UnsupportedTestException("The psd-tools oracle needs a desktop process.");
        if (OraclePython.Find() == null) throw new UnsupportedTestException($"Set {OraclePython.Variables} to a Python with psd-tools to run the PSD oracle test.");
    }

    private static string FindScript()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir != null)
        {
            string candidate = Path.Combine(dir.FullName, "scripts", "images", "psd_layers.py");
            if (File.Exists(candidate)) return candidate;
            dir = dir.Parent;
        }
        throw new UnsupportedTestException("scripts/images/psd_layers.py was not found above " + AppContext.BaseDirectory);
    }

    /// <summary>
    /// What psd-tools reads: the document, the merged image (RGBA8, canvas size) and each layer's own pixels (RGBA8, the layer's rectangle).
    /// Throws <see cref="UnsupportedTestException"/> when the oracle Python has no psd-tools.
    /// </summary>
    [UnsupportedOSPlatform("browser")]
    public static (Info Info, byte[] Composite, byte[][] Layers) Read(byte[] file)
    {
        string python = OraclePython.Find()!;
        string work = Path.Combine(Path.GetTempPath(), "spawndev-psd-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(work);
        try
        {
            string input = Path.Combine(work, "input.psd"), outDir = Path.Combine(work, "out");
            File.WriteAllBytes(input, file);
            if (Environment.GetEnvironmentVariable("PSD_TEST_DUMP") is { } dump) File.WriteAllBytes(Path.Combine(dump, "last.psd"), file);
            var psi = new ProcessStartInfo(python) { RedirectStandardOutput = true, RedirectStandardError = true, UseShellExecute = false, CreateNoWindow = true };
            psi.ArgumentList.Add("-I");
            psi.ArgumentList.Add(FindScript());
            psi.ArgumentList.Add(input);
            psi.ArgumentList.Add(outDir);
            using var proc = Process.Start(psi) ?? throw new InvalidOperationException("could not start python");
            string stdout = proc.StandardOutput.ReadToEnd(), stderr = proc.StandardError.ReadToEnd();
            if (!proc.WaitForExit(120_000)) { proc.Kill(true); throw new Exception("the PSD oracle timed out"); }
            if (proc.ExitCode == 3) throw new UnsupportedTestException("the oracle Python has no psd-tools (pip install psd-tools)");
            if (proc.ExitCode != 0) throw new Exception($"psd-tools could not read the PSD: {stdout.Trim()} {stderr.Trim()}");
            using var json = JsonDocument.Parse(File.ReadAllText(Path.Combine(outDir, "layers.json"), System.Text.Encoding.UTF8));
            var root = json.RootElement;
            var layers = root.GetProperty("layers").EnumerateArray().Select(l => new Layer(l.GetProperty("name").GetString() ?? "",
                l.GetProperty("left").GetInt32(), l.GetProperty("top").GetInt32(), l.GetProperty("right").GetInt32(), l.GetProperty("bottom").GetInt32(),
                l.GetProperty("visible").GetBoolean(), l.GetProperty("opacity").GetInt32(), l.GetProperty("blend").GetString() ?? "",
                l.GetProperty("width").GetInt32(), l.GetProperty("height").GetInt32())).ToList();
            var layerPixels = layers.Select((_, i) => File.ReadAllBytes(Path.Combine(outDir, $"layer{i}.rgba"))).ToArray();
            var c = root.GetProperty("composite");
            return (new Info(root.GetProperty("width").GetInt32(), root.GetProperty("height").GetInt32(), layers, c[0].GetInt32(), c[1].GetInt32()),
                File.ReadAllBytes(Path.Combine(outDir, "composite.rgba")), layerPixels);
        }
        finally
        {
            try { Directory.Delete(work, true); } catch { /* temp cleanup is best effort */ }
        }
    }
}
