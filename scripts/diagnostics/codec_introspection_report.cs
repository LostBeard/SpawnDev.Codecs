// Codec introspection demo: AV1 + VP9 streams driven through the
// SpawnDev.Codecs analyzer + validator APIs in one place.
//
// Demonstrates the consumer-facing API surface for stream metadata
// extraction and bitstream QA. No ffmpeg dependency - pure SpawnDev
// parsing + validation.

#:project ../../SpawnDev.Codecs/SpawnDev.Codecs.csproj
using System;
using System.IO;
using System.Linq;
using SpawnDev.Codecs.Container.Matroska;
using SpawnDev.Codecs.Video.Av1;
using SpawnDev.Codecs.Video.Vp9;

string testDataDir = "SpawnDev.Codecs.Demo.Shared/TestData";

Console.WriteLine();
Console.WriteLine("============================================================");
Console.WriteLine("  Codec introspection report: AV1 + VP9");
Console.WriteLine("============================================================");

// AV1 - bbb_180_2s.ivf
{
    string ivfPath = Path.Combine(testDataDir, "bbb_180_2s.ivf");
    var bytes = File.ReadAllBytes(ivfPath);

    Console.WriteLine();
    Console.WriteLine($"=== AV1: {ivfPath} ===");
    Console.WriteLine();
    var summary = Av1StreamAnalyzer.Analyze(bytes);
    Console.WriteLine(summary.ToReport());
    Console.WriteLine();

    Console.WriteLine("Validation:");
    var validation = Av1StreamValidator.Validate(bytes);
    Console.WriteLine($"  IsValid: {validation.IsValid}");
    Console.WriteLine($"  Errors: {validation.CountBy(Av1ValidationSeverity.Error)}, "
        + $"Warnings: {validation.CountBy(Av1ValidationSeverity.Warning)}, "
        + $"Info: {validation.CountBy(Av1ValidationSeverity.Info)}");
    if (validation.Findings.Count > 0)
    {
        foreach (var f in validation.Findings.Take(5))
            Console.WriteLine($"    [{f.Severity}] {f.Message}");
    }
}

// VP9 - Big_Buck_Bunny_180_10s.webm
{
    string webmPath = Path.Combine(testDataDir, "Big_Buck_Bunny_180_10s.webm");
    using var stream = File.OpenRead(webmPath);
    var container = new MatroskaContainer(stream);
    var video = container.Tracks.First(t => t.IsVideo);

    Console.WriteLine();
    Console.WriteLine($"=== VP9: {webmPath} ===");
    Console.WriteLine();
    Console.WriteLine($"Container: {container.DocType} (WebM); Codec: {video.CodecId}");
    Console.WriteLine();

    var packets = container.Frames
        .Where(f => f.TrackNumber == video.TrackNumber)
        .Select(f => (ReadOnlyMemory<byte>)f.Data)
        .ToList(); // Materialize for both analyzer + validator passes.

    var summary = Vp9StreamAnalyzer.Analyze(packets);
    Console.WriteLine(summary.ToReport());
    Console.WriteLine();

    Console.WriteLine("Validation:");
    var validation = Vp9StreamValidator.Validate(packets);
    Console.WriteLine($"  IsValid: {validation.IsValid}");
    Console.WriteLine($"  Errors: {validation.CountBy(Av1ValidationSeverity.Error)}, "
        + $"Warnings: {validation.CountBy(Av1ValidationSeverity.Warning)}");
    if (validation.Findings.Count > 0)
    {
        foreach (var f in validation.Findings.Take(5))
            Console.WriteLine($"    [{f.Severity}] {f.Message}");
    }
}

Console.WriteLine();
Console.WriteLine("============================================================");
Console.WriteLine("  Both AV1 + VP9 streams introspected via consumer APIs.");
Console.WriteLine("  No ffmpeg dependency - pure SpawnDev.Codecs parsing.");
Console.WriteLine("============================================================");
