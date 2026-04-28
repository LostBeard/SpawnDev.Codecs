// Round-trip test for Vp9BoolEncoder + Vp9BoolDecoder.
// Now uses the proper Vp9BoolDecoder pair after the encoder was fixed
// to emit the leading marker bit per libvpx vpx_start_encode.

#:project ../../SpawnDev.Codecs/SpawnDev.Codecs.csproj
using System;
using SpawnDev.Codecs.Video.Vp9;

int totalChecks = 0;
int totalFails = 0;

// Test 1: 1000 bools at varied probabilities.
{
    var enc = new Vp9BoolEncoder();
    int[] probs = { 128, 64, 200, 32, 240, 16, 250, 6 };
    var rng = new Random(2024);
    var symbols = new int[1000];
    for (int i = 0; i < symbols.Length; i++) symbols[i] = rng.Next(2);
    for (int i = 0; i < symbols.Length; i++)
        enc.Write(symbols[i], probs[i % probs.Length]);
    var bytes = enc.Stop();
    Console.WriteLine($"Test 1: {symbols.Length} bools encoded into {bytes.Length} bytes");

    var dec = new Vp9BoolDecoder(bytes, 0, bytes.Length);
    for (int i = 0; i < symbols.Length; i++)
    {
        int got = dec.Read(probs[i % probs.Length]);
        totalChecks++;
        if (got != symbols[i]) totalFails++;
    }
}

// Test 2: literals.
{
    var enc = new Vp9BoolEncoder();
    int[] values = { 0xA5, 0x12, 0xFF, 0x00, 0x7F, 0x3C, 0xBA, 0x10 };
    foreach (var v in values) enc.WriteLiteral(v, 8);
    var bytes = enc.Stop();
    Console.WriteLine($"Test 2: {values.Length} literals encoded into {bytes.Length} bytes");

    var dec = new Vp9BoolDecoder(bytes, 0, bytes.Length);
    for (int i = 0; i < values.Length; i++)
    {
        int got = (int)dec.ReadLiteral(8);
        totalChecks++;
        if (got != values[i]) totalFails++;
    }
}

Console.WriteLine();
Console.WriteLine($"VP9 bool coder round-trip: {totalChecks - totalFails}/{totalChecks} pass");
if (totalFails == 0)
    Console.WriteLine("VP9 bool encoder + decoder pair round-trip BIT-EXACT.");
else
    Environment.Exit(1);
