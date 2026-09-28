// Pure-C# smoke test for Vp9ForwardDct32x32Kernel logic vs reference.
// The kernel uses the same Fdct32 sequence as the reference; this test
// runs a span-based port of the kernel's harness + Fdct32 inline and
// compares against the reference's Transform across N random blocks.

#:project ../../SpawnDev.Codecs/SpawnDev.Codecs.csproj

using SpawnDev.Codecs.Video.Vp9;

const int BlockCount = 4;
const int CosPi1_64 = 16364, CosPi2_64 = 16305, CosPi3_64 = 16207;
const int CosPi5_64 = 15893, CosPi7_64 = 15426, CosPi9_64 = 14811;
const int CosPi11_64 = 14053, CosPi13_64 = 13160, CosPi15_64 = 12140;
const int CosPi17_64 = 11003, CosPi19_64 = 9760, CosPi21_64 = 8423;
const int CosPi23_64 = 7005, CosPi25_64 = 5520, CosPi27_64 = 3981;
const int CosPi29_64 = 2404, CosPi31_64 = 804;
const int CosPi4_64 = 16069, CosPi6_64 = 15679, CosPi8_64 = 15137;
const int CosPi10_64 = 14449, CosPi12_64 = 13623, CosPi14_64 = 12665;
const int CosPi16_64 = 11585, CosPi18_64 = 10394, CosPi20_64 = 9102;
const int CosPi22_64 = 7723, CosPi24_64 = 6270, CosPi26_64 = 4756;
const int CosPi28_64 = 3196, CosPi30_64 = 1606;

const int DctConstBits = 14;
const int DctConstRounding = 1 << (DctConstBits - 1);

static long DctRound(long v) => (v + DctConstRounding) >> DctConstBits;
static long PositiveBiasShift(long v) => (v + 1 + (v > 0 ? 1 : 0)) >> 2;
static long HalfRoundShift(long v) => (v + 1 + (v < 0 ? 1 : 0)) >> 2;

static void Fdct32(Span<long> input, Span<long> output, Span<long> step)
{
    long c16 = CosPi16_64, c8 = CosPi8_64, c24 = CosPi24_64;

    step[0] = input[0] + input[31]; step[1] = input[1] + input[30];
    step[2] = input[2] + input[29]; step[3] = input[3] + input[28];
    step[4] = input[4] + input[27]; step[5] = input[5] + input[26];
    step[6] = input[6] + input[25]; step[7] = input[7] + input[24];
    step[8] = input[8] + input[23]; step[9] = input[9] + input[22];
    step[10] = input[10] + input[21]; step[11] = input[11] + input[20];
    step[12] = input[12] + input[19]; step[13] = input[13] + input[18];
    step[14] = input[14] + input[17]; step[15] = input[15] + input[16];
    step[16] = -input[16] + input[15]; step[17] = -input[17] + input[14];
    step[18] = -input[18] + input[13]; step[19] = -input[19] + input[12];
    step[20] = -input[20] + input[11]; step[21] = -input[21] + input[10];
    step[22] = -input[22] + input[9]; step[23] = -input[23] + input[8];
    step[24] = -input[24] + input[7]; step[25] = -input[25] + input[6];
    step[26] = -input[26] + input[5]; step[27] = -input[27] + input[4];
    step[28] = -input[28] + input[3]; step[29] = -input[29] + input[2];
    step[30] = -input[30] + input[1]; step[31] = -input[31] + input[0];

    output[0] = step[0] + step[15]; output[1] = step[1] + step[14];
    output[2] = step[2] + step[13]; output[3] = step[3] + step[12];
    output[4] = step[4] + step[11]; output[5] = step[5] + step[10];
    output[6] = step[6] + step[9]; output[7] = step[7] + step[8];
    output[8] = -step[8] + step[7]; output[9] = -step[9] + step[6];
    output[10] = -step[10] + step[5]; output[11] = -step[11] + step[4];
    output[12] = -step[12] + step[3]; output[13] = -step[13] + step[2];
    output[14] = -step[14] + step[1]; output[15] = -step[15] + step[0];

    output[16] = step[16]; output[17] = step[17];
    output[18] = step[18]; output[19] = step[19];
    output[20] = DctRound((-step[20] + step[27]) * c16);
    output[21] = DctRound((-step[21] + step[26]) * c16);
    output[22] = DctRound((-step[22] + step[25]) * c16);
    output[23] = DctRound((-step[23] + step[24]) * c16);
    output[24] = DctRound((step[24] + step[23]) * c16);
    output[25] = DctRound((step[25] + step[22]) * c16);
    output[26] = DctRound((step[26] + step[21]) * c16);
    output[27] = DctRound((step[27] + step[20]) * c16);
    output[28] = step[28]; output[29] = step[29];
    output[30] = step[30]; output[31] = step[31];

    step[0] = output[0] + output[7]; step[1] = output[1] + output[6];
    step[2] = output[2] + output[5]; step[3] = output[3] + output[4];
    step[4] = -output[4] + output[3]; step[5] = -output[5] + output[2];
    step[6] = -output[6] + output[1]; step[7] = -output[7] + output[0];
    step[8] = output[8]; step[9] = output[9];
    step[10] = DctRound((-output[10] + output[13]) * c16);
    step[11] = DctRound((-output[11] + output[12]) * c16);
    step[12] = DctRound((output[12] + output[11]) * c16);
    step[13] = DctRound((output[13] + output[10]) * c16);
    step[14] = output[14]; step[15] = output[15];

    step[16] = output[16] + output[23]; step[17] = output[17] + output[22];
    step[18] = output[18] + output[21]; step[19] = output[19] + output[20];
    step[20] = -output[20] + output[19]; step[21] = -output[21] + output[18];
    step[22] = -output[22] + output[17]; step[23] = -output[23] + output[16];
    step[24] = -output[24] + output[31]; step[25] = -output[25] + output[30];
    step[26] = -output[26] + output[29]; step[27] = -output[27] + output[28];
    step[28] = output[28] + output[27]; step[29] = output[29] + output[26];
    step[30] = output[30] + output[25]; step[31] = output[31] + output[24];

    output[0] = step[0] + step[3]; output[1] = step[1] + step[2];
    output[2] = -step[2] + step[1]; output[3] = -step[3] + step[0];
    output[4] = step[4]; output[5] = DctRound((-step[5] + step[6]) * c16);
    output[6] = DctRound((step[6] + step[5]) * c16); output[7] = step[7];
    output[8] = step[8] + step[11]; output[9] = step[9] + step[10];
    output[10] = -step[10] + step[9]; output[11] = -step[11] + step[8];
    output[12] = -step[12] + step[15]; output[13] = -step[13] + step[14];
    output[14] = step[14] + step[13]; output[15] = step[15] + step[12];

    output[16] = step[16]; output[17] = step[17];
    output[18] = DctRound(step[18] * -c8 + step[29] * c24);
    output[19] = DctRound(step[19] * -c8 + step[28] * c24);
    output[20] = DctRound(step[20] * -c24 + step[27] * -c8);
    output[21] = DctRound(step[21] * -c24 + step[26] * -c8);
    output[22] = step[22]; output[23] = step[23]; output[24] = step[24]; output[25] = step[25];
    output[26] = DctRound(step[26] * c24 + step[21] * -c8);
    output[27] = DctRound(step[27] * c24 + step[20] * -c8);
    output[28] = DctRound(step[28] * c8 + step[19] * c24);
    output[29] = DctRound(step[29] * c8 + step[18] * c24);
    output[30] = step[30]; output[31] = step[31];

    step[0] = DctRound((output[0] + output[1]) * c16);
    step[1] = DctRound((-output[1] + output[0]) * c16);
    step[2] = DctRound(output[2] * c24 + output[3] * c8);
    step[3] = DctRound(output[3] * c24 - output[2] * c8);
    step[4] = output[4] + output[5]; step[5] = -output[5] + output[4];
    step[6] = -output[6] + output[7]; step[7] = output[7] + output[6];
    step[8] = output[8];
    step[9] = DctRound(output[9] * -c8 + output[14] * c24);
    step[10] = DctRound(output[10] * -c24 + output[13] * -c8);
    step[11] = output[11]; step[12] = output[12];
    step[13] = DctRound(output[13] * c24 + output[10] * -c8);
    step[14] = DctRound(output[14] * c8 + output[9] * c24);
    step[15] = output[15];

    step[16] = output[16] + output[19]; step[17] = output[17] + output[18];
    step[18] = -output[18] + output[17]; step[19] = -output[19] + output[16];
    step[20] = -output[20] + output[23]; step[21] = -output[21] + output[22];
    step[22] = output[22] + output[21]; step[23] = output[23] + output[20];
    step[24] = output[24] + output[27]; step[25] = output[25] + output[26];
    step[26] = -output[26] + output[25]; step[27] = -output[27] + output[24];
    step[28] = -output[28] + output[31]; step[29] = -output[29] + output[30];
    step[30] = output[30] + output[29]; step[31] = output[31] + output[28];

    long c4 = CosPi4_64, c28 = CosPi28_64, c12 = CosPi12_64, c20 = CosPi20_64;

    output[0] = step[0]; output[1] = step[1];
    output[2] = step[2]; output[3] = step[3];
    output[4] = DctRound(step[4] * c28 + step[7] * c4);
    output[5] = DctRound(step[5] * c12 + step[6] * c20);
    output[6] = DctRound(step[6] * c12 + step[5] * -c20);
    output[7] = DctRound(step[7] * c28 + step[4] * -c4);
    output[8] = step[8] + step[9]; output[9] = -step[9] + step[8];
    output[10] = -step[10] + step[11]; output[11] = step[11] + step[10];
    output[12] = step[12] + step[13]; output[13] = -step[13] + step[12];
    output[14] = -step[14] + step[15]; output[15] = step[15] + step[14];

    output[16] = step[16];
    output[17] = DctRound(step[17] * -c4 + step[30] * c28);
    output[18] = DctRound(step[18] * -c28 + step[29] * -c4);
    output[19] = step[19]; output[20] = step[20];
    output[21] = DctRound(step[21] * -c20 + step[26] * c12);
    output[22] = DctRound(step[22] * -c12 + step[25] * -c20);
    output[23] = step[23]; output[24] = step[24];
    output[25] = DctRound(step[25] * c12 + step[22] * -c20);
    output[26] = DctRound(step[26] * c20 + step[21] * c12);
    output[27] = step[27]; output[28] = step[28];
    output[29] = DctRound(step[29] * c28 + step[18] * -c4);
    output[30] = DctRound(step[30] * c4 + step[17] * c28);
    output[31] = step[31];

    long c2 = CosPi2_64, c30 = CosPi30_64, c14 = CosPi14_64, c18 = CosPi18_64;
    long c10 = CosPi10_64, c22 = CosPi22_64, c26 = CosPi26_64, c6 = CosPi6_64;

    step[0] = output[0]; step[1] = output[1]; step[2] = output[2]; step[3] = output[3];
    step[4] = output[4]; step[5] = output[5]; step[6] = output[6]; step[7] = output[7];
    step[8] = DctRound(output[8] * c30 + output[15] * c2);
    step[9] = DctRound(output[9] * c14 + output[14] * c18);
    step[10] = DctRound(output[10] * c22 + output[13] * c10);
    step[11] = DctRound(output[11] * c6 + output[12] * c26);
    step[12] = DctRound(output[12] * c6 + output[11] * -c26);
    step[13] = DctRound(output[13] * c22 + output[10] * -c10);
    step[14] = DctRound(output[14] * c14 + output[9] * -c18);
    step[15] = DctRound(output[15] * c30 + output[8] * -c2);

    step[16] = output[16] + output[17]; step[17] = -output[17] + output[16];
    step[18] = -output[18] + output[19]; step[19] = output[19] + output[18];
    step[20] = output[20] + output[21]; step[21] = -output[21] + output[20];
    step[22] = -output[22] + output[23]; step[23] = output[23] + output[22];
    step[24] = output[24] + output[25]; step[25] = -output[25] + output[24];
    step[26] = -output[26] + output[27]; step[27] = output[27] + output[26];
    step[28] = output[28] + output[29]; step[29] = -output[29] + output[28];
    step[30] = -output[30] + output[31]; step[31] = output[31] + output[30];

    long c1 = CosPi1_64, c31 = CosPi31_64, c15 = CosPi15_64, c17 = CosPi17_64;
    long c23 = CosPi23_64, c9 = CosPi9_64, c7 = CosPi7_64, c25 = CosPi25_64;
    long c27 = CosPi27_64, c5 = CosPi5_64, c11 = CosPi11_64, c21 = CosPi21_64;
    long c19 = CosPi19_64, c13 = CosPi13_64, c3 = CosPi3_64, c29 = CosPi29_64;

    output[0] = step[0]; output[16] = step[1]; output[8] = step[2]; output[24] = step[3];
    output[4] = step[4]; output[20] = step[5]; output[12] = step[6]; output[28] = step[7];
    output[2] = step[8]; output[18] = step[9]; output[10] = step[10]; output[26] = step[11];
    output[6] = step[12]; output[22] = step[13]; output[14] = step[14]; output[30] = step[15];

    output[1] = DctRound(step[16] * c31 + step[31] * c1);
    output[17] = DctRound(step[17] * c15 + step[30] * c17);
    output[9] = DctRound(step[18] * c23 + step[29] * c9);
    output[25] = DctRound(step[19] * c7 + step[28] * c25);
    output[5] = DctRound(step[20] * c27 + step[27] * c5);
    output[21] = DctRound(step[21] * c11 + step[26] * c21);
    output[13] = DctRound(step[22] * c19 + step[25] * c13);
    output[29] = DctRound(step[23] * c3 + step[24] * c29);
    output[3] = DctRound(step[24] * c3 + step[23] * -c29);
    output[19] = DctRound(step[25] * c19 + step[22] * -c13);
    output[11] = DctRound(step[26] * c11 + step[21] * -c21);
    output[27] = DctRound(step[27] * c27 + step[20] * -c5);
    output[7] = DctRound(step[28] * c7 + step[19] * -c25);
    output[23] = DctRound(step[29] * c23 + step[18] * -c9);
    output[15] = DctRound(step[30] * c15 + step[17] * -c17);
    output[31] = DctRound(step[31] * c31 + step[16] * -c1);
}

static void KernelBlock(short[] input, int inBase, int[] output, int outBase)
{
    var intermediate = new long[1024];
    var tempIn = new long[32];
    var tempOut = new long[32];
    var step = new long[32];

    for (int i = 0; i < 32; i++)
    {
        for (int j = 0; j < 32; j++) tempIn[j] = (long)input[inBase + j * 32 + i] * 4L;
        Fdct32(tempIn, tempOut, step);
        for (int j = 0; j < 32; j++) intermediate[j * 32 + i] = PositiveBiasShift(tempOut[j]);
    }

    for (int i = 0; i < 32; i++)
    {
        for (int j = 0; j < 32; j++) tempIn[j] = intermediate[j + i * 32];
        Fdct32(tempIn, tempOut, step);
        for (int j = 0; j < 32; j++) output[outBase + j + i * 32] = (int)HalfRoundShift(tempOut[j]);
    }
}

var rng = new Random(42);
var input = new short[BlockCount * 1024];
for (int i = 0; i < input.Length; i++) input[i] = (short)rng.Next(-1024, 1024);

var refOut = new int[BlockCount * 1024];
for (int b = 0; b < BlockCount; b++)
    Vp9ForwardDct32x32.Transform(input.AsSpan(b * 1024, 1024), 32, refOut.AsSpan(b * 1024, 1024));

var kernelOut = new int[BlockCount * 1024];
for (int b = 0; b < BlockCount; b++)
    KernelBlock(input, b * 1024, kernelOut, b * 1024);

int mismatches = 0; int firstBad = -1;
for (int i = 0; i < refOut.Length; i++)
    if (refOut[i] != kernelOut[i])
    {
        if (firstBad < 0) firstBad = i;
        mismatches++;
    }

Console.WriteLine($"Vp9ForwardDct32x32Kernel: {BlockCount} blocks, {mismatches} mismatches.");
if (mismatches > 0)
{
    int b = firstBad / 1024, idx = firstBad % 1024;
    Console.WriteLine($"First mismatched block #{b}, index {idx}: ref={refOut[firstBad]}, kernel={kernelOut[firstBad]}");
}
else
    Console.WriteLine("PASS - bit-exact match.");
