// Test support: libFLAC (through Python soundfile / libsndfile) as an independent oracle for the FLAC encoder. The
// encoder under test writes a file; libFLAC decodes it (checking every frame CRC) and the samples come back.
//
// Desktop lanes only: needs the Python named by SPAWNDEV_IMG_PYTHON with numpy + soundfile. Without it (or in a browser)
// the calling test reports Unsupported, never passed. The script is scripts/audio/flac_decode.py.

using System.Diagnostics;
using System.Runtime.Versioning;
using System.Text.Json;
using SpawnDev.UnitTesting;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

internal static class FlacOracle
{
    private static string FindScript()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir != null)
        {
            string candidate = Path.Combine(dir.FullName, "scripts", "audio", "flac_decode.py");
            if (File.Exists(candidate)) return candidate;
            dir = dir.Parent;
        }
        throw new UnsupportedTestException("scripts/audio/flac_decode.py was not found above " + AppContext.BaseDirectory);
    }

    /// <summary>Skip (Unsupported) up front when libFLAC cannot run here.</summary>
    public static void RequireAvailable()
    {
        if (OperatingSystem.IsBrowser())
            throw new UnsupportedTestException("The libFLAC oracle needs a desktop process.");
        string? python = Environment.GetEnvironmentVariable("SPAWNDEV_IMG_PYTHON");
        if (string.IsNullOrEmpty(python) || !File.Exists(python))
            throw new UnsupportedTestException("Set SPAWNDEV_IMG_PYTHON to a Python with numpy + soundfile to run the libFLAC oracle tests.");
    }

    /// <summary>libFLAC's decode of <paramref name="file"/>: interleaved samples (in the file's bit depth), channels, rate.</summary>
    [UnsupportedOSPlatform("browser")]
    public static (int[] Samples, int Channels, int Rate) Decode(byte[] file)
    {
        RequireAvailable();
        string python = Environment.GetEnvironmentVariable("SPAWNDEV_IMG_PYTHON")!;
        string work = Path.Combine(Path.GetTempPath(), "spawndev-flac-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(work);
        try
        {
            string input = Path.Combine(work, "in.flac"), output = Path.Combine(work, "out.raw");
            File.WriteAllBytes(input, file);
            var psi = new ProcessStartInfo(python)
            {
                RedirectStandardOutput = true, RedirectStandardError = true, UseShellExecute = false, CreateNoWindow = true,
            };
            psi.ArgumentList.Add("-I");
            psi.ArgumentList.Add(FindScript());
            psi.ArgumentList.Add(input);
            psi.ArgumentList.Add(output);
            using var proc = Process.Start(psi) ?? throw new InvalidOperationException("could not start python");
            string stdout = proc.StandardOutput.ReadToEnd();
            string stderr = proc.StandardError.ReadToEnd();
            if (!proc.WaitForExit(120_000)) { proc.Kill(true); throw new Exception("libFLAC oracle timed out"); }
            if (proc.ExitCode != 0) throw new Exception($"libFLAC could not decode the file: {stdout.Trim()} {stderr.Trim()}");
            using var json = JsonDocument.Parse(stdout);
            var raw = File.ReadAllBytes(output);
            var samples = new int[raw.Length / 4];
            Buffer.BlockCopy(raw, 0, samples, 0, raw.Length);
            return (samples, json.RootElement.GetProperty("channels").GetInt32(), json.RootElement.GetProperty("rate").GetInt32());
        }
        finally
        {
            try { Directory.Delete(work, true); } catch { /* best effort */ }
        }
    }
}
