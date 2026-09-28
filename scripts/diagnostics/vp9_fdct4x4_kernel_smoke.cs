// Pure-C# (no ILGPU) smoke test for the Vp9ForwardDct4x4Kernel logic vs
// Vp9ForwardDct4x4 reference. Verifies bit-exact match on N random blocks.
// dotnet run can't JIT-init ILGPU in a single-file script, so this script
// reimplements the kernel butterfly inline and compares to the reference.

#:project ../../SpawnDev.Codecs/SpawnDev.Codecs.csproj

using SpawnDev.Codecs.Video.Vp9;

const int BlockCount = 64;

const int CosPi8_64 = 15137;
const int CosPi16_64 = 11585;
const int CosPi24_64 = 6270;
const int DctConstBits = 14;
const int DctConstRounding = 1 << (DctConstBits - 1);

static void FdctRow(
    int s0, int s1, int s2, int s3, bool addOneIfNonZero,
    out int o0, out int o1, out int o2, out int o3)
{
    if (addOneIfNonZero && s0 != 0) s0++;

    int x0 = s0 + s3;
    int x1 = s1 + s2;
    int x2 = s1 - s2;
    int x3 = s0 - s3;

    o0 = (int)(((long)(x0 + x1) * CosPi16_64 + DctConstRounding) >> DctConstBits);
    o2 = (int)(((long)(x0 - x1) * CosPi16_64 + DctConstRounding) >> DctConstBits);
    o1 = (int)(((long)x2 * CosPi24_64 + (long)x3 * CosPi8_64 + DctConstRounding) >> DctConstBits);
    o3 = (int)(((long)(-x2) * CosPi8_64 + (long)x3 * CosPi24_64 + DctConstRounding) >> DctConstBits);
}

static void KernelBlock(short[] input, int inBase, int[] output, int outBase)
{
    short i00 = input[inBase + 0],  i01 = input[inBase + 1],
          i02 = input[inBase + 2],  i03 = input[inBase + 3];
    short i10 = input[inBase + 4],  i11 = input[inBase + 5],
          i12 = input[inBase + 6],  i13 = input[inBase + 7];
    short i20 = input[inBase + 8],  i21 = input[inBase + 9],
          i22 = input[inBase + 10], i23 = input[inBase + 11];
    short i30 = input[inBase + 12], i31 = input[inBase + 13],
          i32 = input[inBase + 14], i33 = input[inBase + 15];

    FdctRow(i00 * 16, i10 * 16, i20 * 16, i30 * 16, true,
            out int t00_p1, out int t01_p1, out int t02_p1, out int t03_p1);
    FdctRow(i01 * 16, i11 * 16, i21 * 16, i31 * 16, false,
            out int t10_p1, out int t11_p1, out int t12_p1, out int t13_p1);
    FdctRow(i02 * 16, i12 * 16, i22 * 16, i32 * 16, false,
            out int t20_p1, out int t21_p1, out int t22_p1, out int t23_p1);
    FdctRow(i03 * 16, i13 * 16, i23 * 16, i33 * 16, false,
            out int t30_p1, out int t31_p1, out int t32_p1, out int t33_p1);

    FdctRow(t00_p1, t10_p1, t20_p1, t30_p1, false,
            out int r0_o0, out int r0_o1, out int r0_o2, out int r0_o3);
    FdctRow(t01_p1, t11_p1, t21_p1, t31_p1, false,
            out int r1_o0, out int r1_o1, out int r1_o2, out int r1_o3);
    FdctRow(t02_p1, t12_p1, t22_p1, t32_p1, false,
            out int r2_o0, out int r2_o1, out int r2_o2, out int r2_o3);
    FdctRow(t03_p1, t13_p1, t23_p1, t33_p1, false,
            out int r3_o0, out int r3_o1, out int r3_o2, out int r3_o3);

    output[outBase + 0]  = (r0_o0 + 1) >> 2;
    output[outBase + 1]  = (r0_o1 + 1) >> 2;
    output[outBase + 2]  = (r0_o2 + 1) >> 2;
    output[outBase + 3]  = (r0_o3 + 1) >> 2;
    output[outBase + 4]  = (r1_o0 + 1) >> 2;
    output[outBase + 5]  = (r1_o1 + 1) >> 2;
    output[outBase + 6]  = (r1_o2 + 1) >> 2;
    output[outBase + 7]  = (r1_o3 + 1) >> 2;
    output[outBase + 8]  = (r2_o0 + 1) >> 2;
    output[outBase + 9]  = (r2_o1 + 1) >> 2;
    output[outBase + 10] = (r2_o2 + 1) >> 2;
    output[outBase + 11] = (r2_o3 + 1) >> 2;
    output[outBase + 12] = (r3_o0 + 1) >> 2;
    output[outBase + 13] = (r3_o1 + 1) >> 2;
    output[outBase + 14] = (r3_o2 + 1) >> 2;
    output[outBase + 15] = (r3_o3 + 1) >> 2;
}

var rng = new Random(42);
var input = new short[BlockCount * 16];
for (int i = 0; i < input.Length; i++) input[i] = (short)rng.Next(-1024, 1024);

var cpuOut = new int[BlockCount * 16];
for (int b = 0; b < BlockCount; b++)
    Vp9ForwardDct4x4.Transform(input.AsSpan(b * 16, 16), 4, cpuOut.AsSpan(b * 16, 16));

var kernelOut = new int[BlockCount * 16];
for (int b = 0; b < BlockCount; b++)
    KernelBlock(input, b * 16, kernelOut, b * 16);

int mismatches = 0;
int firstBad = -1;
for (int i = 0; i < cpuOut.Length; i++)
{
    if (cpuOut[i] != kernelOut[i])
    {
        if (firstBad < 0) firstBad = i;
        mismatches++;
    }
}

Console.WriteLine($"Vp9ForwardDct4x4Kernel: {BlockCount} blocks, {mismatches} mismatches.");
if (mismatches > 0)
{
    int b = firstBad / 16;
    Console.WriteLine($"First mismatched block #{b}, first index {firstBad}:");
    Console.Write("  input  : ");
    for (int i = 0; i < 16; i++) Console.Write($"{input[b * 16 + i],6} ");
    Console.WriteLine();
    Console.Write("  cpuOut : ");
    for (int i = 0; i < 16; i++) Console.Write($"{cpuOut[b * 16 + i],6} ");
    Console.WriteLine();
    Console.Write("  kernel : ");
    for (int i = 0; i < 16; i++) Console.Write($"{kernelOut[b * 16 + i],6} ");
    Console.WriteLine();
}
else
{
    Console.WriteLine("PASS - bit-exact match.");
}
