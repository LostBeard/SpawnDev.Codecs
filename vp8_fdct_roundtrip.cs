// Round-trip test for Vp8ForwardTransform + Vp8InverseTransform.
// Encode a 4x4 byte block as residual, forward DCT it, inverse DCT
// against zero prediction, verify the result matches the original input
// within fixed-point rounding error.

#:project SpawnDev.Codecs/SpawnDev.Codecs.csproj
using System;
using SpawnDev.Codecs.Video.Vp8;

int totalChecks = 0;
int totalFails = 0;
int maxDelta = 0;

void Test(string name, byte[] input)
{
    // Build the residual: input byte - 0 prediction = input value as short.
    Span<short> residual = stackalloc short[16];
    for (int i = 0; i < 16; i++) residual[i] = input[i];

    Span<short> coefs = stackalloc short[16];
    Vp8ForwardTransform.ShortFdct4x4(residual, 4, coefs);

    // Inverse with zero prediction.
    Span<byte> pred = stackalloc byte[16];
    Span<byte> output = stackalloc byte[16];
    Vp8InverseTransform.ShortIdct4x4Llm(coefs, pred, 4, output, 4);

    int worst = 0;
    for (int i = 0; i < 16; i++)
    {
        int delta = Math.Abs(input[i] - output[i]);
        if (delta > worst) worst = delta;
    }
    totalChecks++;
    if (worst > 2) totalFails++;
    if (worst > maxDelta) maxDelta = worst;

    Console.WriteLine($"  {name,-20}: max_delta={worst}");
}

// Test 1: all zero
Test("AllZero", new byte[16]);

// Test 2: constant 128
{
    var input = new byte[16];
    Array.Fill(input, (byte)128);
    Test("Constant128", input);
}

// Test 3: vertical gradient
{
    var input = new byte[16];
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++)
            input[r * 4 + c] = (byte)(r * 30 + 100);
    Test("VertGradient", input);
}

// Test 4: horizontal gradient
{
    var input = new byte[16];
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++)
            input[r * 4 + c] = (byte)(c * 30 + 100);
    Test("HorzGradient", input);
}

// Test 5: random pattern
{
    var rng = new Random(42);
    var input = new byte[16];
    for (int i = 0; i < 16; i++) input[i] = (byte)rng.Next(80, 180);
    Test("Random80to180", input);
}

// Test 6: high-frequency pattern (checkerboard)
{
    var input = new byte[16];
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++)
            input[r * 4 + c] = (byte)(((r + c) & 1) == 0 ? 50 : 200);
    Test("Checkerboard50_200", input);
}

Console.WriteLine();
Console.WriteLine($"VP8 FDCT round-trip: {totalChecks - totalFails}/{totalChecks} pass (delta <= 2), max delta seen = {maxDelta}");
if (totalFails == 0)
{
    Console.WriteLine("VP8 forward DCT + inverse DCT pair round-trips within fixed-point precision.");
}
else
{
    Environment.Exit(1);
}
