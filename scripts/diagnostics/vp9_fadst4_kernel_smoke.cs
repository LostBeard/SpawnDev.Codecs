// Smoke test for Vp9ForwardAdst4Kernel logic vs Vp9ForwardAdst4 reference.

#:project ../../SpawnDev.Codecs/SpawnDev.Codecs.csproj

using SpawnDev.Codecs.Video.Vp9;

const int BlockCount = 256;
const int Sinpi1_9 = 5283, Sinpi2_9 = 9929, Sinpi3_9 = 13377, Sinpi4_9 = 15212;
const int DctConstBits = 14;
const int DctConstRounding = 1 << (DctConstBits - 1);

static void KernelBlock(int x0, int x1, int x2, int x3, out int o0, out int o1, out int o2, out int o3)
{
    long s0 = (long)Sinpi1_9 * x0;
    long s1 = (long)Sinpi4_9 * x0;
    long s2 = (long)Sinpi2_9 * x1;
    long s3 = (long)Sinpi1_9 * x1;
    long s4 = (long)Sinpi3_9 * x2;
    long s5 = (long)Sinpi4_9 * x3;
    long s6 = (long)Sinpi2_9 * x3;
    long s7 = x0 + x1 - x3;

    long y0 = s0 + s2 + s5;
    long y1 = (long)Sinpi3_9 * s7;
    long y2 = s1 - s3 + s6;
    long y3 = s4;

    long t0 = y0 + y3;
    long t1 = y1;
    long t2 = y2 - y3;
    long t3 = y2 - y0 + y3;

    o0 = (int)((t0 + DctConstRounding) >> DctConstBits);
    o1 = (int)((t1 + DctConstRounding) >> DctConstBits);
    o2 = (int)((t2 + DctConstRounding) >> DctConstBits);
    o3 = (int)((t3 + DctConstRounding) >> DctConstBits);
}

var rng = new Random(42);
int mismatches = 0;
for (int b = 0; b < BlockCount; b++)
{
    int x0 = rng.Next(-2048, 2048);
    int x1 = rng.Next(-2048, 2048);
    int x2 = rng.Next(-2048, 2048);
    int x3 = rng.Next(-2048, 2048);

    int[] input = { x0, x1, x2, x3 };
    int[] cpuOut = new int[4];
    Vp9ForwardAdst4.Transform(input, cpuOut);

    KernelBlock(x0, x1, x2, x3, out int k0, out int k1, out int k2, out int k3);
    int[] kernelOut = { k0, k1, k2, k3 };

    for (int i = 0; i < 4; i++) if (cpuOut[i] != kernelOut[i]) mismatches++;
}

Console.WriteLine($"Vp9ForwardAdst4Kernel: {BlockCount} blocks, {mismatches} mismatches.");
Console.WriteLine(mismatches == 0 ? "PASS - bit-exact match." : "FAIL - mismatches found.");
