// Runs the new AV1 OBU writer tests via the Console runner.

using System.Diagnostics;

string[] tests = new[]
{
    "Av1ObuWriter_Leb128_RoundTripCanonicalValues",
    "Av1ObuWriter_Leb128_KnownEncodings",
    "Av1ObuWriter_BbbFirstFrame_RoundTripsBitExact",
    "Av1ObuWriter_BbbAllFrames_RoundTripsBitExact",
    "Av1ObuWriter_BbbFirstFrame_ConcatenatedRoundTripMatchesSource",
    "Av1SequenceHeaderWriter_BbbConfig_RoundTripsThroughParser",
    "Av1SequenceHeaderWriter_HighBitDepth10_RoundTrips",
    "Av1SequenceHeaderWriter_4kFrame_RoundTrips",
    "Av1SequenceHeaderWriter_Monochrome_RoundTrips",
    "Av1SequenceHeaderWriter_WrappedAsObu_StreamHasValidShape",
    "Av1SequenceHeaderWriter_RejectsInvalidConfigs",
};

int total = 0, passed = 0;

foreach (var test in tests)
{
    string fullName = $"CpuCodecsTests.{test}";
    var psi = new ProcessStartInfo("dotnet",
        $"run --project SpawnDev.Codecs.DemoConsole --no-build -- \"{fullName}\"")
    {
        RedirectStandardOutput = true,
        RedirectStandardError = true,
        UseShellExecute = false,
    };
    using var p = Process.Start(psi)!;
    string stdout = p.StandardOutput.ReadToEnd();
    p.WaitForExit();
    total++;
    bool ok = stdout.Contains("\"ResultText\":\"Success\"");
    if (ok) passed++;
    string status = ok ? "PASS" : "FAIL";
    Console.WriteLine($"  [{status}] {test}");
    if (!ok)
    {
        // Show error info from JSON
        int idx = stdout.IndexOf("\"Error\":\"");
        if (idx >= 0)
        {
            int end = stdout.IndexOf("\",\"StackTrace\"", idx);
            string err = end > idx ? stdout.Substring(idx + 9, end - idx - 9) : stdout.Substring(idx + 9, Math.Min(200, stdout.Length - idx - 9));
            Console.WriteLine($"      {err}");
        }
    }
}

Console.WriteLine();
Console.WriteLine($"Av1ObuWriter tests: {passed}/{total} passed");
