// Test support: Pillow as an independent oracle for the image ENCODER tests. The encoder under test writes a
// file; Pillow (libjpeg / libpng / giflib-class decoders, not our code) opens it and reports what it sees.
//
// Desktop lanes only: needs a Python with Pillow + numpy, named by the SPAWNDEV_IMG_PYTHON environment
// variable. When that is not set (or in a browser lane) the calling test is reported as Unsupported, never as
// passed. The script is scripts/images/pillow_decode.py.

using System.Diagnostics;
using System.Runtime.Versioning;
using System.Text.Json;
using SpawnDev.UnitTesting;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>What Pillow saw in a file.</summary>
internal sealed record PillowResult(
    string Format, int Width, int Height, string Mode, int Frames, int?[] Durations, int? Loop, byte[][] FramesRgba);

internal static class PillowOracle
{
    private static string FindScript()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir != null)
        {
            string candidate = Path.Combine(dir.FullName, "scripts", "images", "pillow_decode.py");
            if (File.Exists(candidate)) return candidate;
            dir = dir.Parent;
        }
        throw new UnsupportedTestException("scripts/images/pillow_decode.py was not found above " + AppContext.BaseDirectory);
    }

    /// <summary>
    /// First statement of a Pillow-oracle test: skip BEFORE any encoding work when the oracle cannot run (browser lanes,
    /// no SPAWNDEV_IMG_PYTHON), instead of encoding everything and then skipping.
    /// </summary>
    public static void RequireAvailable()
    {
        if (OperatingSystem.IsBrowser())
            throw new UnsupportedTestException("The Pillow oracle needs a desktop process.");
        string? python = Environment.GetEnvironmentVariable("SPAWNDEV_IMG_PYTHON");
        if (string.IsNullOrEmpty(python) || !File.Exists(python))
            throw new UnsupportedTestException("Set SPAWNDEV_IMG_PYTHON to a Python with Pillow and numpy to run the Pillow oracle tests.");
    }

    /// <summary>Open <paramref name="file"/> with Pillow and return what it sees (every frame as RGBA).</summary>
    [UnsupportedOSPlatform("browser")]
    public static PillowResult Decode(byte[] file, string extension)
    {
        RequireAvailable();
        string? python = Environment.GetEnvironmentVariable("SPAWNDEV_IMG_PYTHON");
        string script = FindScript();

        string work = Path.Combine(Path.GetTempPath(), "spawndev-pillow-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(work);
        try
        {
            string input = Path.Combine(work, "input" + extension);
            File.WriteAllBytes(input, file);
            string outDir = Path.Combine(work, "out");
            var psi = new ProcessStartInfo(python)
            {
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true,
            };
            psi.ArgumentList.Add("-I");
            psi.ArgumentList.Add(script);
            psi.ArgumentList.Add(input);
            psi.ArgumentList.Add(outDir);
            using var proc = Process.Start(psi) ?? throw new InvalidOperationException("could not start python");
            string stdout = proc.StandardOutput.ReadToEnd();
            string stderr = proc.StandardError.ReadToEnd();
            if (!proc.WaitForExit(120_000)) { proc.Kill(true); throw new Exception("Pillow oracle timed out"); }
            if (proc.ExitCode != 0)
                throw new Exception($"Pillow could not open the file: {stdout.Trim()} {stderr.Trim()}");

            using var json = JsonDocument.Parse(File.ReadAllText(Path.Combine(outDir, "input" + extension + ".json")));
            var root = json.RootElement;
            int frames = root.GetProperty("frames").GetInt32();
            var durations = new int?[frames];
            int k = 0;
            foreach (var d in root.GetProperty("durations").EnumerateArray())
                durations[k++] = d.ValueKind == JsonValueKind.Number ? d.GetInt32() : null;
            int? loop = root.GetProperty("loop").ValueKind == JsonValueKind.Number ? root.GetProperty("loop").GetInt32() : null;
            var rgba = new byte[frames][];
            for (int f = 0; f < frames; f++)
                rgba[f] = File.ReadAllBytes(Path.Combine(outDir, $"input{extension}.f{f}.rgba8"));
            return new PillowResult(
                root.GetProperty("format").GetString() ?? "",
                root.GetProperty("width").GetInt32(), root.GetProperty("height").GetInt32(),
                root.GetProperty("mode").GetString() ?? "", frames, durations, loop, rgba);
        }
        finally
        {
            try { Directory.Delete(work, true); } catch { /* temp cleanup is best effort */ }
        }
    }
}
