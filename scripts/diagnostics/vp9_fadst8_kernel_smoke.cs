// Smoke test for Vp9ForwardAdst8Kernel logic vs Vp9ForwardAdst8 reference.

#:project ../../SpawnDev.Codecs/SpawnDev.Codecs.csproj

using SpawnDev.Codecs.Video.Vp9;

const int BlockCount = 256;
const int CosPi2_64 = 16305, CosPi6_64 = 15679, CosPi8_64 = 15137;
const int CosPi10_64 = 14449, CosPi14_64 = 12665, CosPi16_64 = 11585;
const int CosPi18_64 = 10394, CosPi22_64 = 7723, CosPi24_64 = 6270;
const int CosPi26_64 = 4756, CosPi30_64 = 1606;
const int DctConstBits = 14;
const int DctConstRounding = 1 << (DctConstBits - 1);

static long RS(long v) => (v + DctConstRounding) >> DctConstBits;

static void KernelBlock(int[] input, int[] output)
{
    long x0 = input[7], x1 = input[0], x2 = input[5], x3 = input[2];
    long x4 = input[3], x5 = input[4], x6 = input[1], x7 = input[6];

    long s0 = (long)CosPi2_64 * x0 + (long)CosPi30_64 * x1;
    long s1 = (long)CosPi30_64 * x0 - (long)CosPi2_64 * x1;
    long s2 = (long)CosPi10_64 * x2 + (long)CosPi22_64 * x3;
    long s3 = (long)CosPi22_64 * x2 - (long)CosPi10_64 * x3;
    long s4 = (long)CosPi18_64 * x4 + (long)CosPi14_64 * x5;
    long s5 = (long)CosPi14_64 * x4 - (long)CosPi18_64 * x5;
    long s6 = (long)CosPi26_64 * x6 + (long)CosPi6_64 * x7;
    long s7 = (long)CosPi6_64 * x6 - (long)CosPi26_64 * x7;

    x0 = RS(s0 + s4); x1 = RS(s1 + s5); x2 = RS(s2 + s6); x3 = RS(s3 + s7);
    x4 = RS(s0 - s4); x5 = RS(s1 - s5); x6 = RS(s2 - s6); x7 = RS(s3 - s7);

    long t0 = x0, t1 = x1, t2 = x2, t3 = x3;
    long t4 = (long)CosPi8_64 * x4 + (long)CosPi24_64 * x5;
    long t5 = (long)CosPi24_64 * x4 - (long)CosPi8_64 * x5;
    long t6 = -(long)CosPi24_64 * x6 + (long)CosPi8_64 * x7;
    long t7 = (long)CosPi8_64 * x6 + (long)CosPi24_64 * x7;

    x0 = t0 + t2; x1 = t1 + t3; x2 = t0 - t2; x3 = t1 - t3;
    x4 = RS(t4 + t6); x5 = RS(t5 + t7); x6 = RS(t4 - t6); x7 = RS(t5 - t7);

    long u2 = (long)CosPi16_64 * (x2 + x3);
    long u3 = (long)CosPi16_64 * (x2 - x3);
    long u6 = (long)CosPi16_64 * (x6 + x7);
    long u7 = (long)CosPi16_64 * (x6 - x7);
    x2 = RS(u2); x3 = RS(u3); x6 = RS(u6); x7 = RS(u7);

    output[0] = (int)x0;
    output[1] = (int)-x4;
    output[2] = (int)x6;
    output[3] = (int)-x2;
    output[4] = (int)x3;
    output[5] = (int)-x7;
    output[6] = (int)x5;
    output[7] = (int)-x1;
}

var rng = new Random(42);
int mismatches = 0;
for (int b = 0; b < BlockCount; b++)
{
    var input = new int[8];
    for (int i = 0; i < 8; i++) input[i] = rng.Next(-2048, 2048);

    var cpuOut = new int[8];
    Vp9ForwardAdst8.Transform(input, cpuOut);

    var kernelOut = new int[8];
    KernelBlock(input, kernelOut);

    for (int i = 0; i < 8; i++) if (cpuOut[i] != kernelOut[i]) mismatches++;
}

Console.WriteLine($"Vp9ForwardAdst8Kernel: {BlockCount} blocks, {mismatches} mismatches.");
Console.WriteLine(mismatches == 0 ? "PASS - bit-exact match." : "FAIL - mismatches found.");
