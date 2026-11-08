// Test support: FFmpeg (on PATH) as an independent decoder for the MP3 encoder: it decodes our file with error reporting
// on, any message on stderr fails the test. Desktop lanes only; without ffmpeg (or in a browser) the calling test reports
// Unsupported, never passed.

using System.Diagnostics;
using System.Runtime.Versioning;
using SpawnDev.UnitTesting;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

internal static class FfmpegOracle
{
    /// <summary>Skip (Unsupported) up front when ffmpeg cannot run here.</summary>
    public static void RequireAvailable()
    {
        if (OperatingSystem.IsBrowser()) throw new UnsupportedTestException("The FFmpeg oracle needs a desktop process.");
        try
        {
            using var p = Process.Start(new ProcessStartInfo("ffmpeg", "-hide_banner -version")
            {
                RedirectStandardOutput = true, RedirectStandardError = true, UseShellExecute = false, CreateNoWindow = true,
            });
            p!.StandardOutput.ReadToEnd();
            p.WaitForExit(30_000);
        }
        catch (Exception e)
        {
            throw new UnsupportedTestException("ffmpeg is not on PATH: " + e.Message);
        }
    }

    /// <summary>FFmpeg's decode of <paramref name="file"/> as interleaved float32 (<paramref name="ext"/> names the format).
    /// <paramref name="strict"/> (the default) turns on FFmpeg's error detection and fails on any message; without it
    /// FFmpeg decodes as a player does (a third-party file it warns about still decodes) and only fatal errors fail.</summary>
    [UnsupportedOSPlatform("browser")]
    public static float[] DecodeFloat(byte[] file, string ext, bool strict = true)
    {
        RequireAvailable();
        string work = Path.Combine(Path.GetTempPath(), "spawndev-ff-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(work);
        try
        {
            string input = Path.Combine(work, "in" + ext), output = Path.Combine(work, "out.f32");
            File.WriteAllBytes(input, file);
            var psi = new ProcessStartInfo("ffmpeg")
            {
                RedirectStandardOutput = true, RedirectStandardError = true, UseShellExecute = false, CreateNoWindow = true,
            };
            foreach (var arg in strict
                         ? new[] { "-hide_banner", "-v", "error", "-err_detect", "crccheck+bitstream+buffer+explode", "-i", input, "-f", "f32le", "-y", output }
                         : new[] { "-hide_banner", "-v", "fatal", "-i", input, "-f", "f32le", "-y", output })
                psi.ArgumentList.Add(arg);
            using var proc = Process.Start(psi) ?? throw new InvalidOperationException("could not start ffmpeg");
            string stderr = proc.StandardError.ReadToEnd();
            proc.StandardOutput.ReadToEnd();
            if (!proc.WaitForExit(120_000)) { proc.Kill(true); throw new Exception("ffmpeg timed out"); }
            if (proc.ExitCode != 0 || stderr.Trim().Length > 0) throw new Exception($"ffmpeg reported: {stderr.Trim()} (exit {proc.ExitCode})");
            var raw = File.ReadAllBytes(output);
            var f = new float[raw.Length / 4];
            Buffer.BlockCopy(raw, 0, f, 0, raw.Length);
            return f;
        }
        finally
        {
            try { Directory.Delete(work, true); } catch { /* best effort */ }
        }
    }

    /// <summary>Run an FFmpeg tool (ffprobe, ffmpeg) on a file and return its stdout; stderr output fails.</summary>
    [UnsupportedOSPlatform("browser")]
    public static string Probe(byte[] file, string ext, params string[] args)
    {
        RequireAvailable();
        string work = Path.Combine(Path.GetTempPath(), "spawndev-ffp-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(work);
        try
        {
            string input = Path.Combine(work, "in" + ext);
            File.WriteAllBytes(input, file);
            var psi = new ProcessStartInfo("ffprobe")
            {
                RedirectStandardOutput = true, RedirectStandardError = true, UseShellExecute = false, CreateNoWindow = true,
            };
            psi.ArgumentList.Add("-v");
            psi.ArgumentList.Add("error");
            foreach (var a in args) psi.ArgumentList.Add(a);
            psi.ArgumentList.Add(input);
            using var proc = Process.Start(psi) ?? throw new InvalidOperationException("could not start ffprobe");
            string stdout = proc.StandardOutput.ReadToEnd();
            string stderr = proc.StandardError.ReadToEnd();
            if (!proc.WaitForExit(60_000)) { proc.Kill(true); throw new Exception("ffprobe timed out"); }
            if (proc.ExitCode != 0 || stderr.Trim().Length > 0) throw new Exception($"ffprobe reported: {stderr.Trim()}");
            return stdout;
        }
        finally
        {
            try { Directory.Delete(work, true); } catch { /* best effort */ }
        }
    }
}
