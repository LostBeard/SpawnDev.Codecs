// Minimal repro: 40 alternating bools at prob 128, no raw values.
#:project ../../SpawnDev.Codecs/SpawnDev.Codecs.csproj
using System;
using SpawnDev.Codecs.Video.Vp8;

var enc = new Vp8BoolEncoder();
var symbols = new int[40];
for (int i = 0; i < symbols.Length; i++) symbols[i] = i & 1;
foreach (var s in symbols) enc.EncodeBool(s, 128);
var bytes = enc.Stop();
Console.WriteLine($"Encoded {symbols.Length} alternating bools at prob 128 into {bytes.Length} bytes");
Console.WriteLine($"Bytes: {string.Join(" ", Array.ConvertAll(bytes, b => b.ToString("X2")))}");

var dec = new Vp8BoolDecoder(bytes);
int fails = 0;
for (int i = 0; i < symbols.Length; i++)
{
    int got = dec.DecodeBool(128);
    if (got != symbols[i]) { Console.WriteLine($"  FAIL idx {i}: got {got}, expected {symbols[i]}"); fails++; }
}
Console.WriteLine($"{symbols.Length - fails}/{symbols.Length} pass");

// Also test 200 ops at varied probs
Console.WriteLine();
var enc2 = new Vp8BoolEncoder();
var probs = new[] { 128, 200, 64 };
var rng = new Random(7);
var sym2 = new int[200];
for (int i = 0; i < sym2.Length; i++) sym2[i] = rng.Next(2);
for (int i = 0; i < sym2.Length; i++) enc2.EncodeBool(sym2[i], probs[i % probs.Length]);
var bytes2 = enc2.Stop();
var dec2 = new Vp8BoolDecoder(bytes2);
int fails2 = 0;
for (int i = 0; i < sym2.Length; i++)
{
    int got = dec2.DecodeBool(probs[i % probs.Length]);
    if (got != sym2[i]) { fails2++; if (fails2 <= 5) Console.WriteLine($"  FAIL idx {i}: got {got}, expected {sym2[i]} prob={probs[i % probs.Length]}"); }
}
Console.WriteLine($"200 mixed-prob bools: {sym2.Length - fails2}/{sym2.Length} pass");

// And 1000 bools at varied probs (stress)
Console.WriteLine();
var enc3 = new Vp8BoolEncoder();
var sym3 = new int[1000];
for (int i = 0; i < sym3.Length; i++) sym3[i] = rng.Next(2);
for (int i = 0; i < sym3.Length; i++) enc3.EncodeBool(sym3[i], probs[i % probs.Length]);
var bytes3 = enc3.Stop();
var dec3 = new Vp8BoolDecoder(bytes3);
int fails3 = 0;
for (int i = 0; i < sym3.Length; i++)
{
    int got = dec3.DecodeBool(probs[i % probs.Length]);
    if (got != sym3[i]) { fails3++; if (fails3 <= 5) Console.WriteLine($"  FAIL idx {i}: got {got}, expected {sym3[i]} prob={probs[i % probs.Length]}"); }
}
Console.WriteLine($"1000 stress bools: {sym3.Length - fails3}/{sym3.Length} pass");
