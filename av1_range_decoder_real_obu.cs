// Integration test: open BBB AV1 IVF, locate the first frame's tile data
// (skip TD + SH + Frame header), initialize Av1RangeDecoder on the
// entropy-coded portion, decode some symbols, verify it doesn't crash
// and that Tell advances by a reasonable amount.
//
// This is a smoke test, NOT a bit-exact verification - we don't have
// libaom CDFs wired yet so we can't predict the exact symbols. What it
// proves: the range coder initializes correctly on real AV1 bytes,
// renormalize/refill don't crash, Tell reports a plausible progression.

#:project SpawnDev.Codecs/SpawnDev.Codecs.csproj
using System;
using System.IO;
using SpawnDev.Codecs.Container.Ivf;
using SpawnDev.Codecs.EntropyCoders;
using SpawnDev.Codecs.Video.Av1;

string ivfPath = "SpawnDev.Codecs.Demo.Shared/TestData/bbb_180_2s.ivf";
if (!File.Exists(ivfPath))
{
    Console.WriteLine($"FAIL: {ivfPath} not found");
    Environment.Exit(1);
}

var ivfBytes = File.ReadAllBytes(ivfPath);
var ivfHeader = IvfReader.ParseHeader(ivfBytes);
Console.WriteLine($"IVF header: {ivfHeader.FourCc} {ivfHeader.Width}x{ivfHeader.Height} (frames declared: {ivfHeader.NumFrames})");

if (ivfHeader.FourCc != "AV01")
{
    Console.WriteLine($"FAIL: expected AV01 fourcc, got {ivfHeader.FourCc}");
    Environment.Exit(1);
}

var firstFrame = IvfReader.EnumerateFrames(ivfBytes).First();
var frameBytes = firstFrame.Data.ToArray();
Console.WriteLine($"First AV1 frame: {frameBytes.Length} bytes total.");

// Walk OBUs in the first frame, find the Frame OBU (or TileGroup).
var obus = Av1ObuParser.EnumerateObus(frameBytes).ToArray();
Console.WriteLine($"OBUs in first frame: {obus.Length}");
foreach (var obu in obus)
    Console.WriteLine($"  OBU type={obu.Type}, payload offset={obu.PayloadOffset}, length={obu.PayloadLength}");

// Find the Frame OBU (combined frame header + tile group).
var frameObu = obus.FirstOrDefault(o => o.Type == Av1ObuType.Frame || o.Type == Av1ObuType.TileGroup);
if (frameObu.Type != Av1ObuType.Frame && frameObu.Type != Av1ObuType.TileGroup)
{
    Console.WriteLine("FAIL: no Frame or TileGroup OBU found");
    Environment.Exit(1);
}

Console.WriteLine();
Console.WriteLine($"Selected Frame/TileGroup OBU: payload {frameObu.PayloadLength} bytes");

// Initialize range decoder on the LAST half of the OBU payload as a
// rough proxy for "past the frame header bits". We don't yet have a
// full AV1 frame header parser to compute the exact tile data offset,
// so this is a smoke test - we just verify the range coder initializes
// + decodes symbols + Tell advances without crashing.
var payload = new byte[frameObu.PayloadLength];
Buffer.BlockCopy(frameBytes, frameObu.PayloadOffset, payload, 0, frameObu.PayloadLength);
int splitPoint = payload.Length / 2;

var dec = new Av1RangeDecoder(payload, splitPoint, payload.Length - splitPoint);
int initialTell = dec.Tell;
Console.WriteLine($"Range decoder initialized on {payload.Length - splitPoint} bytes starting at offset {splitPoint}.");
Console.WriteLine($"  Initial Tell: {initialTell}");

// Decode 100 booleans at varied probabilities.
int decoded = 0, ones = 0;
ushort[] icdf2 = new ushort[] { 16384, 0 }; // 2-symbol uniform CDF
for (int i = 0; i < 100; i++)
{
    uint f = (uint)((i * 311 + 4099) % 32760 + 4); // varied prob in [4, 32763]
    int bit = dec.DecodeBoolQ15(f);
    decoded++;
    if (bit == 1) ones++;
}

int afterTell = dec.Tell;
Console.WriteLine($"  Decoded {decoded} bools at varied probabilities; got {ones} ones.");
Console.WriteLine($"  Tell after 100 decodes: {afterTell} (advanced by {afterTell - initialTell} bits)");

// Sanity check: Tell must advance and stay positive
if (afterTell <= initialTell)
{
    Console.WriteLine($"FAIL: Tell did not advance ({initialTell} -> {afterTell})");
    Environment.Exit(1);
}
if (afterTell - initialTell > decoded * 32)
{
    Console.WriteLine($"FAIL: Tell advanced by impossibly large amount ({afterTell - initialTell} for {decoded} bool decodes)");
    Environment.Exit(1);
}

// Decode 50 4-symbol CDF values to exercise the CDF path.
ushort[] icdf4 = new ushort[] { 24576, 16384, 8192, 0 };
int beforeCdf = dec.Tell;
for (int i = 0; i < 50; i++)
{
    int sym = dec.DecodeCdfQ15(icdf4, 4);
    if (sym < 0 || sym >= 4)
    {
        Console.WriteLine($"FAIL: invalid CDF symbol {sym}");
        Environment.Exit(1);
    }
}
int afterCdf = dec.Tell;
Console.WriteLine($"  Decoded 50 4-sym CDF symbols; Tell advanced by {afterCdf - beforeCdf} bits.");

Console.WriteLine();
Console.WriteLine("=== AV1 RANGE DECODER ON REAL AV1 BYTES: PASS ===");
Console.WriteLine($"Range coder initializes + decodes 150 symbols on real BBB AV1");
Console.WriteLine($"OBU bytes without crashing. Bit cursor advances reasonably.");
Console.WriteLine($"Bit-exact symbol verification follows once AV1 CDF tables land.");
