// Pure-C# smoke test for Vp9ForwardDct8x8Kernel logic vs Vp9ForwardDct8x8 reference.
// dotnet run can't init ILGPU; this script runs the kernel butterfly inline.

#:project ../../SpawnDev.Codecs/SpawnDev.Codecs.csproj

using SpawnDev.Codecs.Video.Vp9;

const int BlockCount = 64;
const int CosPi4_64  = 16069;
const int CosPi8_64  = 15137;
const int CosPi12_64 = 13623;
const int CosPi16_64 = 11585;
const int CosPi20_64 = 9102;
const int CosPi24_64 = 6270;
const int CosPi28_64 = 3196;
const int DctConstBits = 14;
const int DctConstRounding = 1 << (DctConstBits - 1);

static int RoundShift(long v) => (int)((v + DctConstRounding) >> DctConstBits);

static void Butterfly(
    int s0, int s1, int s2, int s3, int s4, int s5, int s6, int s7,
    out int o0, out int o1, out int o2, out int o3,
    out int o4, out int o5, out int o6, out int o7)
{
    int x0 = s0 + s3, x1 = s1 + s2, x2 = s1 - s2, x3 = s0 - s3;
    long t0 = (long)(x0 + x1) * CosPi16_64;
    long t1 = (long)(x0 - x1) * CosPi16_64;
    long t2 = (long)x2 * CosPi24_64 + (long)x3 * CosPi8_64;
    long t3 = (long)(-x2) * CosPi8_64 + (long)x3 * CosPi24_64;
    o0 = RoundShift(t0);
    o2 = RoundShift(t2);
    o4 = RoundShift(t1);
    o6 = RoundShift(t3);

    long u0 = (long)(s6 - s5) * CosPi16_64;
    long u1 = (long)(s6 + s5) * CosPi16_64;
    int v2 = RoundShift(u0);
    int v3 = RoundShift(u1);

    int y0 = s4 + v2, y1 = s4 - v2, y2 = s7 - v3, y3 = s7 + v3;
    long w0 = (long)y0 * CosPi28_64 + (long)y3 * CosPi4_64;
    long w1 = (long)y1 * CosPi12_64 + (long)y2 * CosPi20_64;
    long w2 = (long)y2 * CosPi12_64 + (long)y1 * (-CosPi20_64);
    long w3 = (long)y3 * CosPi28_64 + (long)y0 * (-CosPi4_64);
    o1 = RoundShift(w0);
    o3 = RoundShift(w2);
    o5 = RoundShift(w1);
    o7 = RoundShift(w3);
}

static void KernelBlock(short[] input, int inBase, int[] output, int outBase)
{
    int[] tmp = new int[64];
    // Pass 1.
    for (int col = 0; col < 8; col++)
    {
        int s0 = (input[inBase + col + 0 * 8] + input[inBase + col + 7 * 8]) * 4;
        int s1 = (input[inBase + col + 1 * 8] + input[inBase + col + 6 * 8]) * 4;
        int s2 = (input[inBase + col + 2 * 8] + input[inBase + col + 5 * 8]) * 4;
        int s3 = (input[inBase + col + 3 * 8] + input[inBase + col + 4 * 8]) * 4;
        int s4 = (input[inBase + col + 3 * 8] - input[inBase + col + 4 * 8]) * 4;
        int s5 = (input[inBase + col + 2 * 8] - input[inBase + col + 5 * 8]) * 4;
        int s6 = (input[inBase + col + 1 * 8] - input[inBase + col + 6 * 8]) * 4;
        int s7 = (input[inBase + col + 0 * 8] - input[inBase + col + 7 * 8]) * 4;
        Butterfly(s0, s1, s2, s3, s4, s5, s6, s7,
            out int o0, out int o1, out int o2, out int o3,
            out int o4, out int o5, out int o6, out int o7);
        tmp[col * 8 + 0] = o0; tmp[col * 8 + 1] = o1; tmp[col * 8 + 2] = o2; tmp[col * 8 + 3] = o3;
        tmp[col * 8 + 4] = o4; tmp[col * 8 + 5] = o5; tmp[col * 8 + 6] = o6; tmp[col * 8 + 7] = o7;
    }
    // Pass 2.
    for (int col = 0; col < 8; col++)
    {
        int s0 = tmp[col + 0 * 8] + tmp[col + 7 * 8];
        int s1 = tmp[col + 1 * 8] + tmp[col + 6 * 8];
        int s2 = tmp[col + 2 * 8] + tmp[col + 5 * 8];
        int s3 = tmp[col + 3 * 8] + tmp[col + 4 * 8];
        int s4 = tmp[col + 3 * 8] - tmp[col + 4 * 8];
        int s5 = tmp[col + 2 * 8] - tmp[col + 5 * 8];
        int s6 = tmp[col + 1 * 8] - tmp[col + 6 * 8];
        int s7 = tmp[col + 0 * 8] - tmp[col + 7 * 8];
        Butterfly(s0, s1, s2, s3, s4, s5, s6, s7,
            out int o0, out int o1, out int o2, out int o3,
            out int o4, out int o5, out int o6, out int o7);
        output[outBase + col * 8 + 0] = o0 / 2;
        output[outBase + col * 8 + 1] = o1 / 2;
        output[outBase + col * 8 + 2] = o2 / 2;
        output[outBase + col * 8 + 3] = o3 / 2;
        output[outBase + col * 8 + 4] = o4 / 2;
        output[outBase + col * 8 + 5] = o5 / 2;
        output[outBase + col * 8 + 6] = o6 / 2;
        output[outBase + col * 8 + 7] = o7 / 2;
    }
}

// Use the same seed as the PMT failing test (Vp9ForwardDct8x8KernelTests random).
var rng = new Random(unchecked((int)0x9F8D8808u));
const int TestBlockCount = 32;
var input = new short[TestBlockCount * 64];
for (int i = 0; i < input.Length; i++) input[i] = (short)rng.Next(-1024, 1024);

var cpuOut = new int[TestBlockCount * 64];
for (int b = 0; b < TestBlockCount; b++)
    Vp9ForwardDct8x8.Transform(input.AsSpan(b * 64, 64), 8, cpuOut.AsSpan(b * 64, 64));

var kernelOut = new int[TestBlockCount * 64];
for (int b = 0; b < TestBlockCount; b++)
    KernelBlock(input, b * 64, kernelOut, b * 64);

int mismatches = 0; int firstBad = -1;
for (int i = 0; i < cpuOut.Length; i++)
    if (cpuOut[i] != kernelOut[i])
    {
        if (firstBad < 0) firstBad = i;
        mismatches++;
    }

Console.WriteLine($"Vp9ForwardDct8x8Kernel: {TestBlockCount} blocks, {mismatches} mismatches.");
if (mismatches > 0)
{
    int b = firstBad / 64, idx = firstBad % 64;
    Console.WriteLine($"First mismatched block #{b}, index {idx}: cpu={cpuOut[firstBad]}, kernel={kernelOut[firstBad]}");
}
else
    Console.WriteLine("PASS - bit-exact match.");
