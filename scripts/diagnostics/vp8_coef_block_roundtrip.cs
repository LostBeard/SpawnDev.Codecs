// Round-trip test for Vp8CoefBlockEncoder + Vp8CoefBlockDecoder.
// Encode known coefficient blocks via the encoder, then decode them
// back through the bool decoder + decoder pair, verify exact match.

#:project ../../SpawnDev.Codecs/SpawnDev.Codecs.csproj
using System;
using SpawnDev.Codecs.Video.Vp8;

int totalChecks = 0;
int totalFails = 0;

void RunCase(string name, short[] coefs, int firstCoef = 0, int initialCtx = 0)
{
    var probs = Vp8DefaultCoefProbs.DefaultProbs;
    // Use block type 0 (Y after Y2) for these tests - shape is [8 bands, 3 ctx, 11 nodes]
    // Vp8DefaultCoefProbs.DefaultProbs is 4D [block][band][ctx][node].
    // Vp8CoefBlockDecoder/Encoder takes a 3D [band][ctx][node] slice.
    var sliced = new byte[8, 3, 11];
    for (int band = 0; band < 8; band++)
        for (int ctx = 0; ctx < 3; ctx++)
            for (int node = 0; node < 11; node++)
                sliced[band, ctx, node] = probs[0, band, ctx, node];

    var enc = new Vp8BoolEncoder();
    int writtenEob = Vp8CoefBlockEncoder.Encode(enc, sliced, initialCtx, firstCoef, coefs);
    var bytes = enc.Stop();

    var dec = new Vp8BoolDecoder(bytes);
    Span<short> decoded = stackalloc short[16];
    int readEob = Vp8CoefBlockDecoder.Decode(dec, sliced, initialCtx, firstCoef, decoded);

    bool ok = true;
    for (int i = 0; i < 16; i++)
    {
        if (coefs[i] != decoded[i]) { ok = false; break; }
    }
    totalChecks++;
    if (!ok)
    {
        totalFails++;
        Console.Write($"  FAIL {name,-25}: encoded[");
        for (int i = 0; i < 16; i++) Console.Write($"{coefs[i]} ");
        Console.Write("] -> decoded[");
        for (int i = 0; i < 16; i++) Console.Write($"{decoded[i]} ");
        Console.WriteLine("]");
    }
    else
    {
        Console.WriteLine($"  OK   {name,-25}: eob_w={writtenEob} eob_r={readEob} bytes={bytes.Length}");
    }
}

// Test 1: all zero
RunCase("AllZero", new short[16]);

// Test 2: one DC only
RunCase("DcOnly_+5", new short[16] { 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 });
RunCase("DcOnly_-5", new short[16] { -5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 });

// Test 3: small AC
RunCase("Ac1", new short[16] { 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 });
RunCase("Ac2", new short[16] { 2, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 });
RunCase("Ac4", new short[16] { 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 });

// Test 4: category tokens
RunCase("Cat1_+6", new short[16] { 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 });
RunCase("Cat2_+10", new short[16] { 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 });
RunCase("Cat3_+18", new short[16] { 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 });
RunCase("Cat4_+30", new short[16] { 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 });
RunCase("Cat5_+50", new short[16] { 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 });
RunCase("Cat6_+200", new short[16] { 200, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 });

// Test 5: dense block
RunCase("Dense1", new short[16] { -10, 5, 3, 2, 1, -2, 4, 0, 1, 0, 1, 0, 0, 1, 0, 1 });

// Test 6: trailing zeros / EOB in middle
RunCase("EobMid", new short[16] { -8, 4, -2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 });

// Test 7: scattered non-zero
RunCase("Scatter", new short[16] { 0, -3, 0, 0, 5, 0, 0, 0, 0, -7, 0, 0, 0, 0, 11, 0 });

// Test 8: full block of mixed-sign values
{
    var rng = new Random(7);
    var coefs = new short[16];
    for (int i = 0; i < 16; i++) coefs[i] = (short)rng.Next(-50, 51);
    RunCase("Random_Mixed_50", coefs);
}

// Test 9: firstCoef=1 (Y_after_Y2 case)
RunCase("Y_after_Y2", new short[16] { 0, 5, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, firstCoef: 1);

Console.WriteLine();
Console.WriteLine($"VP8 coef block round-trip: {totalChecks - totalFails}/{totalChecks} pass");
if (totalFails == 0)
    Console.WriteLine("VP8 coefficient block encoder + decoder pair round-trip BIT-EXACT.");
else
    Environment.Exit(1);
