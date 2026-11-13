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

    /// <summary>FFmpeg's decode of the video in <paramref name="file"/> (<paramref name="ext"/> names the format) as one tight
    /// I420 array per frame (Y, then U, then V, chroma (w+1)/2 x (h+1)/2). Every frame is kept (no frame dropping or
    /// duplication); any ffmpeg error output, a non-zero exit or a partial frame throws.</summary>
    [UnsupportedOSPlatform("browser")]
    public static List<byte[]> DecodeVideoI420(byte[] file, string ext, int width, int height)
    {
        RequireAvailable();
        string work = Path.Combine(Path.GetTempPath(), "spawndev-ffv-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(work);
        try
        {
            string input = Path.Combine(work, "in" + ext), output = Path.Combine(work, "out.yuv");
            File.WriteAllBytes(input, file);
            var psi = new ProcessStartInfo("ffmpeg")
            {
                RedirectStandardOutput = true, RedirectStandardError = true, UseShellExecute = false, CreateNoWindow = true,
            };
            foreach (var arg in new[] { "-hide_banner", "-v", "error", "-i", input, "-f", "rawvideo", "-pix_fmt", "yuv420p", "-vsync", "passthrough", "-y", output })
                psi.ArgumentList.Add(arg);
            using var proc = Process.Start(psi) ?? throw new InvalidOperationException("could not start ffmpeg");
            string stderr = proc.StandardError.ReadToEnd();
            proc.StandardOutput.ReadToEnd();
            if (!proc.WaitForExit(120_000)) { proc.Kill(true); throw new Exception("ffmpeg timed out"); }
            if (proc.ExitCode != 0 || stderr.Trim().Length > 0) throw new Exception($"ffmpeg reported: {stderr.Trim()} (exit {proc.ExitCode})");
            var raw = File.ReadAllBytes(output);
            int frameBytes = width * height + 2 * ((width + 1) / 2) * ((height + 1) / 2);
            if (raw.Length % frameBytes != 0)
                throw new Exception($"ffmpeg wrote {raw.Length} bytes, not a whole number of {width}x{height} I420 frames ({frameBytes} bytes each)");
            var frames = new List<byte[]>(raw.Length / frameBytes);
            for (int at = 0; at < raw.Length; at += frameBytes)
                frames.Add(raw.AsSpan(at, frameBytes).ToArray());
            return frames;
        }
        finally
        {
            try { Directory.Delete(work, true); } catch { /* best effort */ }
        }
    }

    /// <summary>
    /// Like <see cref="DecodeVideoI420(byte[], string, int, int)"/> with FFmpeg input options that select the decoder
    /// (<c>-c:v av1_cuvid</c>, <c>-hwaccel d3d11va -c:v av1</c> ...). Made for decoders that can fail badly: the error
    /// output is kept to its first 2000 characters and the process is killed at <paramref name="timeoutMs"/> (NVDEC on
    /// a frame it does not support repeats one error line without end: 7 GB in ten minutes, 2026-11-13).
    /// </summary>
    [UnsupportedOSPlatform("browser")]
    public static List<byte[]> DecodeVideoI420(byte[] file, string ext, int width, int height, string[] inputOptions, int timeoutMs = 60_000)
    {
        RequireAvailable();
        string work = Path.Combine(Path.GetTempPath(), "spawndev-ffh-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(work);
        try
        {
            string input = Path.Combine(work, "in" + ext), output = Path.Combine(work, "out.yuv");
            File.WriteAllBytes(input, file);
            var psi = new ProcessStartInfo("ffmpeg")
            {
                RedirectStandardOutput = true, RedirectStandardError = true, UseShellExecute = false, CreateNoWindow = true,
            };
            foreach (var arg in new[] { "-hide_banner", "-v", "error" }) psi.ArgumentList.Add(arg);
            foreach (var arg in inputOptions) psi.ArgumentList.Add(arg);
            foreach (var arg in new[] { "-i", input, "-f", "rawvideo", "-pix_fmt", "yuv420p", "-vsync", "passthrough", "-y", output })
                psi.ArgumentList.Add(arg);
            var stderr = new System.Text.StringBuilder();
            using var proc = new Process { StartInfo = psi };
            proc.ErrorDataReceived += (_, e) =>
            {
                if (e.Data == null) return;
                lock (stderr) { if (stderr.Length < 2000) stderr.AppendLine(e.Data); }
            };
            proc.OutputDataReceived += (_, _) => { };
            if (!proc.Start()) throw new InvalidOperationException("could not start ffmpeg");
            proc.BeginErrorReadLine();
            proc.BeginOutputReadLine();
            if (!proc.WaitForExit(timeoutMs))
            {
                proc.Kill(true);
                proc.WaitForExit(10_000);
                string head;
                lock (stderr) head = stderr.ToString().Trim();
                throw new Exception($"ffmpeg did not finish in {timeoutMs} ms and was stopped: {(head.Length > 300 ? head.Substring(0, 300) : head)}");
            }
            proc.WaitForExit();   // drain the redirected streams
            string err;
            lock (stderr) err = stderr.ToString().Trim();
            if (proc.ExitCode != 0 || err.Length > 0) throw new Exception($"ffmpeg reported: {(err.Length > 600 ? err.Substring(0, 600) : err)} (exit {proc.ExitCode})");
            var raw = File.ReadAllBytes(output);
            int frameBytes = width * height + 2 * ((width + 1) / 2) * ((height + 1) / 2);
            if (raw.Length == 0 || raw.Length % frameBytes != 0)
                throw new Exception($"ffmpeg wrote {raw.Length} bytes, not a whole number of {width}x{height} I420 frames ({frameBytes} bytes each)");
            var frames = new List<byte[]>(raw.Length / frameBytes);
            for (int at = 0; at < raw.Length; at += frameBytes)
                frames.Add(raw.AsSpan(at, frameBytes).ToArray());
            return frames;
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
