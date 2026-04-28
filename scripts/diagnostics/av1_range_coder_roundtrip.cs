// Round-trip test for Av1RangeEncoder + Av1RangeDecoder. Encodes a known
// sequence of binary + CDF + raw-bit symbols, then decodes and verifies
// equality. Proves the pair is consistent. (Cross-verification against
// libaom on a real AV1 frame comes later when block decode lands.)

#:project ../../SpawnDev.Codecs/SpawnDev.Codecs.csproj
using System;
using SpawnDev.Codecs.EntropyCoders;

int totalChecks = 0;
int totalFails = 0;

void Check(bool ok, string label)
{
    totalChecks++;
    if (!ok) { totalFails++; Console.WriteLine($"  FAIL: {label}"); }
}

// ---- Test 1: 32 alternating booleans at varying f ----
{
    var enc = new Av1RangeEncoder();
    var probs = new uint[] { 16384, 8192, 24576, 4096, 28672, 2048, 30720 };
    var symbols = new int[64];
    var rng = new Random(42);
    for (int i = 0; i < symbols.Length; i++) symbols[i] = rng.Next(2);

    for (int i = 0; i < symbols.Length; i++)
        enc.EncodeBoolQ15(symbols[i], probs[i % probs.Length]);

    var bytes = enc.Done();
    Console.WriteLine($"Test 1: {symbols.Length} bools encoded into {bytes.Length} bytes");

    var dec = new Av1RangeDecoder(bytes);
    for (int i = 0; i < symbols.Length; i++)
    {
        int got = dec.DecodeBoolQ15(probs[i % probs.Length]);
        Check(got == symbols[i], $"bool[{i}]: got {got}, expected {symbols[i]} (f={probs[i % probs.Length]})");
    }
}

// ---- Test 2: CDF symbols (4-symbol alphabet) ----
{
    // ICDF: monotonically NON-INCREASING with 0 at the end.
    // Symbol 0 spans [0, CDF_TOP - icdf[0])
    // Symbol 1 spans [CDF_TOP - icdf[0], CDF_TOP - icdf[1])
    // ...
    // Symbol 3 spans [CDF_TOP - icdf[2], CDF_TOP - icdf[3]=0) -- so icdf[3] = 0
    // For 4 symbols of equal probability: cumprob = [8192, 16384, 24576, 32768]
    // icdf = [32768-8192, 32768-16384, 32768-24576, 32768-32768] = [24576, 16384, 8192, 0]
    var icdf = new ushort[] { 24576, 16384, 8192, 0 };
    var symbols = new int[40];
    var rng = new Random(123);
    for (int i = 0; i < symbols.Length; i++) symbols[i] = rng.Next(4);

    var enc = new Av1RangeEncoder();
    for (int i = 0; i < symbols.Length; i++) enc.EncodeCdfQ15(symbols[i], icdf, 4);
    var bytes = enc.Done();
    Console.WriteLine($"Test 2: {symbols.Length} CDF symbols encoded into {bytes.Length} bytes");

    var dec = new Av1RangeDecoder(bytes);
    for (int i = 0; i < symbols.Length; i++)
    {
        int got = dec.DecodeCdfQ15(icdf, 4);
        Check(got == symbols[i], $"cdf[{i}]: got {got}, expected {symbols[i]}");
    }
}

// ---- Test 3: Raw bits ----
{
    var values = new uint[] { 0xA5, 0x12, 0xFF, 0x00, 0x7F };
    var enc = new Av1RangeEncoder();
    foreach (var v in values) enc.EncodeBits(v, 8);
    var bytes = enc.Done();
    Console.WriteLine($"Test 3: {values.Length} raw 8-bit values encoded into {bytes.Length} bytes");
    var dec = new Av1RangeDecoder(bytes);
    for (int i = 0; i < values.Length; i++)
    {
        uint got = dec.DecodeBits(8);
        Check(got == values[i], $"raw[{i}]: got 0x{got:X2}, expected 0x{values[i]:X2}");
    }
}

// ---- Test 4: Mixed - 200 random binary + CDF + raw ----
{
    var icdf4 = new ushort[] { 24576, 16384, 8192, 0 };
    var icdf8 = new ushort[] { 28672, 24576, 20480, 16384, 12288, 8192, 4096, 0 };
    var rng = new Random(2024);
    var ops = new (int kind, int sym, uint param)[200];
    for (int i = 0; i < ops.Length; i++)
    {
        int kind = rng.Next(3);
        int sym;
        uint param;
        switch (kind)
        {
            case 0: sym = rng.Next(2); param = (uint)rng.Next(1, 32768); break;
            case 1: sym = rng.Next(4); param = 4; break; // 4-sym CDF
            case 2: sym = rng.Next(256); param = 8; break; // 8 raw bits
            default: sym = 0; param = 0; break;
        }
        ops[i] = (kind, sym, param);
    }

    var enc = new Av1RangeEncoder();
    foreach (var op in ops)
    {
        switch (op.kind)
        {
            case 0: enc.EncodeBoolQ15(op.sym, op.param); break;
            case 1: enc.EncodeCdfQ15(op.sym, icdf4, 4); break;
            case 2: enc.EncodeBits((uint)op.sym, 8); break;
        }
    }
    var bytes = enc.Done();
    Console.WriteLine($"Test 4: 200 mixed ops encoded into {bytes.Length} bytes");

    var dec = new Av1RangeDecoder(bytes);
    for (int i = 0; i < ops.Length; i++)
    {
        var op = ops[i];
        int got;
        switch (op.kind)
        {
            case 0: got = dec.DecodeBoolQ15(op.param); break;
            case 1: got = dec.DecodeCdfQ15(icdf4, 4); break;
            case 2: got = (int)dec.DecodeBits(8); break;
            default: got = -1; break;
        }
        Check(got == op.sym, $"mixed[{i}] kind={op.kind}: got {got}, expected {op.sym}");
    }
}

Console.WriteLine();
Console.WriteLine($"Round-trip checks: {totalChecks - totalFails}/{totalChecks} pass, {totalFails} fail");
if (totalFails == 0)
{
    Console.WriteLine("AV1 range coder pair is BIT-EXACT round-trip consistent.");
}
else
{
    Environment.Exit(1);
}
