// One-stop benchmark runner: invokes every individual benchmark in
// sequence and writes the combined output to a single timestamped report.
// Useful for "give me a perf snapshot" without remembering each script's
// command line.
//
// Usage: dotnet run scripts/benchmarks/benchmark_all.cs

#:project ../../SpawnDev.Codecs/SpawnDev.Codecs.csproj
using System;
using System.Diagnostics;
using System.IO;
using System.Text;

string outDir = Path.Combine(Path.GetTempPath(), "spawndev_benchmarks");
Directory.CreateDirectory(outDir);
string reportPath = Path.Combine(outDir, $"snapshot_{DateTime.UtcNow:yyyyMMdd_HHmmss}.txt");
var report = new StringBuilder();
report.AppendLine("============================================================");
report.AppendLine($"  SpawnDev.Codecs Benchmark Snapshot");
report.AppendLine($"  {DateTime.UtcNow:yyyy-MM-dd HH:mm:ss}Z");
report.AppendLine("============================================================");
report.AppendLine();

string[] scripts = {
    "scripts/benchmarks/benchmark_all_codecs.cs",   // throughput across enc + dec
    "scripts/benchmarks/benchmark_vs_ffmpeg.cs",    // side-by-side vs ffmpeg
    "scripts/benchmarks/benchmark_audio_quality.cs", // SNR vs source on real audio
    "scripts/benchmarks/benchmark_video_psnr.cs",   // PSNR vs source on real video
};

foreach (var script in scripts)
{
    Console.WriteLine($"\n>>> Running {script} ...");
    report.AppendLine($"### {script}");
    report.AppendLine();
    var sw = Stopwatch.StartNew();
    var psi = new ProcessStartInfo("dotnet", $"run {script}")
    {
        RedirectStandardOutput = true,
        RedirectStandardError = true,
        UseShellExecute = false,
        WorkingDirectory = Directory.GetCurrentDirectory(),
        CreateNoWindow = true,
    };
    var p = Process.Start(psi)!;
    string stdout = p.StandardOutput.ReadToEnd();
    string stderr = p.StandardError.ReadToEnd();
    p.WaitForExit();
    sw.Stop();
    report.AppendLine(stdout);
    if (p.ExitCode != 0)
    {
        report.AppendLine($"[exit {p.ExitCode}]");
        if (!string.IsNullOrWhiteSpace(stderr)) report.AppendLine(stderr);
    }
    Console.WriteLine($"    done in {sw.Elapsed.TotalSeconds:F1}s, exit {p.ExitCode}");
    report.AppendLine();
    report.AppendLine($"(elapsed: {sw.Elapsed.TotalSeconds:F1}s)");
    report.AppendLine();
    report.AppendLine("---");
    report.AppendLine();
}

File.WriteAllText(reportPath, report.ToString());
Console.WriteLine();
Console.WriteLine($"Snapshot written to: {reportPath}");
