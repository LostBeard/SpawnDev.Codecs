// Round-trip the Vp8BoolEncoder + Vp8BoolDecoder pair. Encodes mixed
// boolean + raw-value sequences and verifies decode equality.

#:project SpawnDev.Codecs/SpawnDev.Codecs.csproj
using System;
using SpawnDev.Codecs.Video.Vp8;

int totalChecks = 0;
int totalFails = 0;
void Check(bool ok, string label)
{
    totalChecks++;
    if (!ok) { totalFails++; Console.WriteLine($"  FAIL: {label}"); }
}

// Test 1: Bools with varying probability
{
    var enc = new Vp8BoolEncoder();
    var probs = new[] { 128, 64, 200, 32, 240, 16, 250, 6 };
    var rng = new Random(42);
    var symbols = new int[200];
    for (int i = 0; i < symbols.Length; i++) symbols[i] = rng.Next(2);
    for (int i = 0; i < symbols.Length; i++)
        enc.EncodeBool(symbols[i], probs[i % probs.Length]);
    var bytes = enc.Stop();
    Console.WriteLine($"Test 1: {symbols.Length} bools encoded into {bytes.Length} bytes");
    var dec = new Vp8BoolDecoder(bytes);
    for (int i = 0; i < symbols.Length; i++)
    {
        int got = dec.DecodeBool(probs[i % probs.Length]);
        Check(got == symbols[i], $"bool[{i}]: got {got}, expected {symbols[i]} (prob={probs[i % probs.Length]})");
    }
}

// Test 2: Raw values
{
    var enc = new Vp8BoolEncoder();
    var values = new[] { 0xA5, 0x12, 0xFF, 0x00, 0x7F, 0x3C, 0xBA, 0x10 };
    foreach (var v in values) enc.EncodeValue(v, 8);
    var bytes = enc.Stop();
    Console.WriteLine($"Test 2: {values.Length} raw 8-bit values encoded into {bytes.Length} bytes");
    var dec = new Vp8BoolDecoder(bytes);
    for (int i = 0; i < values.Length; i++)
    {
        int got = dec.DecodeValue(8);
        Check(got == values[i], $"value[{i}]: got 0x{got:X2}, expected 0x{values[i]:X2}");
    }
}

// Test 3: Mixed - 500 random ops with bools and raw values at varied widths
{
    var enc = new Vp8BoolEncoder();
    var rng = new Random(2024);
    var ops = new (int kind, int sym, int param)[500];
    for (int i = 0; i < ops.Length; i++)
    {
        int kind = rng.Next(2);
        int sym = kind == 0 ? rng.Next(2) : rng.Next(1 << (1 + rng.Next(8)));
        int param = kind == 0 ? rng.Next(1, 255) : 0;
        ops[i] = (kind, sym, param);
    }
    for (int i = 0; i < ops.Length; i++)
    {
        var op = ops[i];
        if (op.kind == 0) enc.EncodeBool(op.sym, op.param);
        else
        {
            int bits = (int)Math.Ceiling(Math.Log2(Math.Max(2, op.sym + 1)));
            // Encode as 8-bit value for simplicity
            enc.EncodeValue(op.sym, 8);
        }
    }
    var bytes = enc.Stop();
    Console.WriteLine($"Test 3: 500 mixed ops encoded into {bytes.Length} bytes");
    var dec = new Vp8BoolDecoder(bytes);
    for (int i = 0; i < ops.Length; i++)
    {
        var op = ops[i];
        int got;
        if (op.kind == 0) got = dec.DecodeBool(op.param);
        else got = dec.DecodeValue(8);
        Check(got == op.sym, $"mixed[{i}] kind={op.kind}: got {got}, expected {op.sym}");
    }
}

Console.WriteLine();
Console.WriteLine($"VP8 bool coder round-trip: {totalChecks - totalFails}/{totalChecks} pass, {totalFails} fail");
if (totalFails == 0)
{
    Console.WriteLine("VP8 bool coder pair is BIT-EXACT round-trip consistent.");
}
else
{
    Environment.Exit(1);
}
