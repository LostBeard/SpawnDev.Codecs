// Pure-C# smoke test for Vp9ForwardDct16x16Kernel logic vs reference.

#:project ../../SpawnDev.Codecs/SpawnDev.Codecs.csproj

using SpawnDev.Codecs.Video.Vp9;

const int BlockCount = 16;
const int CosPi2_64 = 16305, CosPi4_64 = 16069, CosPi6_64 = 15679;
const int CosPi8_64 = 15137, CosPi10_64 = 14449, CosPi12_64 = 13623;
const int CosPi14_64 = 12665, CosPi16_64 = 11585, CosPi18_64 = 10394;
const int CosPi20_64 = 9102, CosPi22_64 = 7723, CosPi24_64 = 6270;
const int CosPi26_64 = 4756, CosPi28_64 = 3196, CosPi30_64 = 1606;
const int DctConstBits = 14;
const int DctConstRounding = 1 << (DctConstBits - 1);

static int RS(long v) => (int)((v + DctConstRounding) >> DctConstBits);

static void Butterfly(
    int ih0, int ih1, int ih2, int ih3, int ih4, int ih5, int ih6, int ih7,
    int s10, int s11, int s12, int s13, int s14, int s15, int s16, int s17,
    out int o0, out int o1, out int o2, out int o3,
    out int o4, out int o5, out int o6, out int o7,
    out int o8, out int o9, out int o10, out int o11,
    out int o12, out int o13, out int o14, out int o15)
{
    int s0 = ih0 + ih7, s1 = ih1 + ih6, s2 = ih2 + ih5, s3 = ih3 + ih4;
    int s4 = ih3 - ih4, s5 = ih2 - ih5, s6 = ih1 - ih6, s7 = ih0 - ih7;
    int x0 = s0 + s3, x1 = s1 + s2, x2 = s1 - s2, x3 = s0 - s3;
    long t0 = (long)(x0 + x1) * CosPi16_64;
    long t1 = (long)(x0 - x1) * CosPi16_64;
    long t2 = (long)x3 * CosPi8_64 + (long)x2 * CosPi24_64;
    long t3 = (long)x3 * CosPi24_64 - (long)x2 * CosPi8_64;
    o0 = RS(t0); o4 = RS(t2); o8 = RS(t1); o12 = RS(t3);
    long u0 = (long)(s6 - s5) * CosPi16_64;
    long u1 = (long)(s6 + s5) * CosPi16_64;
    int v2 = RS(u0), v3 = RS(u1);
    int y0 = s4 + v2, y1 = s4 - v2, y2 = s7 - v3, y3 = s7 + v3;
    long w0 = (long)y0 * CosPi28_64 + (long)y3 * CosPi4_64;
    long w1 = (long)y1 * CosPi12_64 + (long)y2 * CosPi20_64;
    long w2 = (long)y2 * CosPi12_64 + (long)y1 * (-CosPi20_64);
    long w3 = (long)y3 * CosPi28_64 + (long)y0 * (-CosPi4_64);
    o2 = RS(w0); o6 = RS(w2); o10 = RS(w1); o14 = RS(w3);

    long temp1 = (long)(s15 - s12) * CosPi16_64;
    long temp2 = (long)(s14 - s13) * CosPi16_64;
    int sa2 = RS(temp1), sa3 = RS(temp2);
    temp1 = (long)(s14 + s13) * CosPi16_64;
    temp2 = (long)(s15 + s12) * CosPi16_64;
    int sa4 = RS(temp1), sa5 = RS(temp2);

    int sb0 = s10 + sa3, sb1 = s11 + sa2, sb2 = s11 - sa2, sb3 = s10 - sa3;
    int sb4 = s17 - sa4, sb5 = s16 - sa5, sb6 = s16 + sa5, sb7 = s17 + sa4;

    temp1 = (long)sb1 * (-CosPi8_64) + (long)sb6 * CosPi24_64;
    temp2 = (long)sb2 * CosPi24_64 + (long)sb5 * CosPi8_64;
    int sc1 = RS(temp1), sc2 = RS(temp2);
    temp1 = (long)sb2 * CosPi8_64 - (long)sb5 * CosPi24_64;
    temp2 = (long)sb1 * CosPi24_64 + (long)sb6 * CosPi8_64;
    int sc5 = RS(temp1), sc6 = RS(temp2);

    int sd0 = sb0 + sc1, sd1 = sb0 - sc1;
    int sd2 = sb3 + sc2, sd3 = sb3 - sc2;
    int sd4 = sb4 - sc5, sd5 = sb4 + sc5;
    int sd6 = sb7 - sc6, sd7 = sb7 + sc6;

    temp1 = (long)sd0 * CosPi30_64 + (long)sd7 * CosPi2_64;
    temp2 = (long)sd1 * CosPi14_64 + (long)sd6 * CosPi18_64;
    o1 = RS(temp1); o9 = RS(temp2);
    temp1 = (long)sd2 * CosPi22_64 + (long)sd5 * CosPi10_64;
    temp2 = (long)sd3 * CosPi6_64 + (long)sd4 * CosPi26_64;
    o5 = RS(temp1); o13 = RS(temp2);
    temp1 = (long)sd3 * (-CosPi26_64) + (long)sd4 * CosPi6_64;
    temp2 = (long)sd2 * (-CosPi10_64) + (long)sd5 * CosPi22_64;
    o3 = RS(temp1); o11 = RS(temp2);
    temp1 = (long)sd1 * (-CosPi18_64) + (long)sd6 * CosPi14_64;
    temp2 = (long)sd0 * (-CosPi2_64) + (long)sd7 * CosPi30_64;
    o7 = RS(temp1); o15 = RS(temp2);
}

static void KernelBlock(short[] input, int inBase, int[] output, int outBase)
{
    int[] tmp = new int[256];
    for (int col = 0; col < 16; col++)
    {
        int ih0 = (input[inBase + col + 0 * 16] + input[inBase + col + 15 * 16]) * 4;
        int ih1 = (input[inBase + col + 1 * 16] + input[inBase + col + 14 * 16]) * 4;
        int ih2 = (input[inBase + col + 2 * 16] + input[inBase + col + 13 * 16]) * 4;
        int ih3 = (input[inBase + col + 3 * 16] + input[inBase + col + 12 * 16]) * 4;
        int ih4 = (input[inBase + col + 4 * 16] + input[inBase + col + 11 * 16]) * 4;
        int ih5 = (input[inBase + col + 5 * 16] + input[inBase + col + 10 * 16]) * 4;
        int ih6 = (input[inBase + col + 6 * 16] + input[inBase + col + 9 * 16]) * 4;
        int ih7 = (input[inBase + col + 7 * 16] + input[inBase + col + 8 * 16]) * 4;
        int s10 = (input[inBase + col + 7 * 16] - input[inBase + col + 8 * 16]) * 4;
        int s11 = (input[inBase + col + 6 * 16] - input[inBase + col + 9 * 16]) * 4;
        int s12 = (input[inBase + col + 5 * 16] - input[inBase + col + 10 * 16]) * 4;
        int s13 = (input[inBase + col + 4 * 16] - input[inBase + col + 11 * 16]) * 4;
        int s14 = (input[inBase + col + 3 * 16] - input[inBase + col + 12 * 16]) * 4;
        int s15 = (input[inBase + col + 2 * 16] - input[inBase + col + 13 * 16]) * 4;
        int s16 = (input[inBase + col + 1 * 16] - input[inBase + col + 14 * 16]) * 4;
        int s17 = (input[inBase + col + 0 * 16] - input[inBase + col + 15 * 16]) * 4;
        Butterfly(ih0, ih1, ih2, ih3, ih4, ih5, ih6, ih7, s10, s11, s12, s13, s14, s15, s16, s17,
            out int o0, out int o1, out int o2, out int o3, out int o4, out int o5, out int o6, out int o7,
            out int o8, out int o9, out int o10, out int o11, out int o12, out int o13, out int o14, out int o15);
        int b = col * 16;
        tmp[b + 0] = o0; tmp[b + 1] = o1; tmp[b + 2] = o2; tmp[b + 3] = o3;
        tmp[b + 4] = o4; tmp[b + 5] = o5; tmp[b + 6] = o6; tmp[b + 7] = o7;
        tmp[b + 8] = o8; tmp[b + 9] = o9; tmp[b + 10] = o10; tmp[b + 11] = o11;
        tmp[b + 12] = o12; tmp[b + 13] = o13; tmp[b + 14] = o14; tmp[b + 15] = o15;
    }
    for (int col = 0; col < 16; col++)
    {
        int ih0 = ((tmp[col + 0 * 16] + 1) >> 2) + ((tmp[col + 15 * 16] + 1) >> 2);
        int ih1 = ((tmp[col + 1 * 16] + 1) >> 2) + ((tmp[col + 14 * 16] + 1) >> 2);
        int ih2 = ((tmp[col + 2 * 16] + 1) >> 2) + ((tmp[col + 13 * 16] + 1) >> 2);
        int ih3 = ((tmp[col + 3 * 16] + 1) >> 2) + ((tmp[col + 12 * 16] + 1) >> 2);
        int ih4 = ((tmp[col + 4 * 16] + 1) >> 2) + ((tmp[col + 11 * 16] + 1) >> 2);
        int ih5 = ((tmp[col + 5 * 16] + 1) >> 2) + ((tmp[col + 10 * 16] + 1) >> 2);
        int ih6 = ((tmp[col + 6 * 16] + 1) >> 2) + ((tmp[col + 9 * 16] + 1) >> 2);
        int ih7 = ((tmp[col + 7 * 16] + 1) >> 2) + ((tmp[col + 8 * 16] + 1) >> 2);
        int s10 = ((tmp[col + 7 * 16] + 1) >> 2) - ((tmp[col + 8 * 16] + 1) >> 2);
        int s11 = ((tmp[col + 6 * 16] + 1) >> 2) - ((tmp[col + 9 * 16] + 1) >> 2);
        int s12 = ((tmp[col + 5 * 16] + 1) >> 2) - ((tmp[col + 10 * 16] + 1) >> 2);
        int s13 = ((tmp[col + 4 * 16] + 1) >> 2) - ((tmp[col + 11 * 16] + 1) >> 2);
        int s14 = ((tmp[col + 3 * 16] + 1) >> 2) - ((tmp[col + 12 * 16] + 1) >> 2);
        int s15 = ((tmp[col + 2 * 16] + 1) >> 2) - ((tmp[col + 13 * 16] + 1) >> 2);
        int s16 = ((tmp[col + 1 * 16] + 1) >> 2) - ((tmp[col + 14 * 16] + 1) >> 2);
        int s17 = ((tmp[col + 0 * 16] + 1) >> 2) - ((tmp[col + 15 * 16] + 1) >> 2);
        Butterfly(ih0, ih1, ih2, ih3, ih4, ih5, ih6, ih7, s10, s11, s12, s13, s14, s15, s16, s17,
            out int o0, out int o1, out int o2, out int o3, out int o4, out int o5, out int o6, out int o7,
            out int o8, out int o9, out int o10, out int o11, out int o12, out int o13, out int o14, out int o15);
        int b = outBase + col * 16;
        output[b + 0] = o0; output[b + 1] = o1; output[b + 2] = o2; output[b + 3] = o3;
        output[b + 4] = o4; output[b + 5] = o5; output[b + 6] = o6; output[b + 7] = o7;
        output[b + 8] = o8; output[b + 9] = o9; output[b + 10] = o10; output[b + 11] = o11;
        output[b + 12] = o12; output[b + 13] = o13; output[b + 14] = o14; output[b + 15] = o15;
    }
}

var rng = new Random(42);
var input = new short[BlockCount * 256];
for (int i = 0; i < input.Length; i++) input[i] = (short)rng.Next(-1024, 1024);

var cpuOut = new int[BlockCount * 256];
for (int b = 0; b < BlockCount; b++)
    Vp9ForwardDct16x16.Transform(input.AsSpan(b * 256, 256), 16, cpuOut.AsSpan(b * 256, 256));

var kernelOut = new int[BlockCount * 256];
for (int b = 0; b < BlockCount; b++)
    KernelBlock(input, b * 256, kernelOut, b * 256);

int mismatches = 0; int firstBad = -1;
for (int i = 0; i < cpuOut.Length; i++)
    if (cpuOut[i] != kernelOut[i])
    {
        if (firstBad < 0) firstBad = i;
        mismatches++;
    }

Console.WriteLine($"Vp9ForwardDct16x16Kernel: {BlockCount} blocks, {mismatches} mismatches.");
if (mismatches > 0)
{
    int b = firstBad / 256, idx = firstBad % 256;
    Console.WriteLine($"First mismatched block #{b}, index {idx}: cpu={cpuOut[firstBad]}, kernel={kernelOut[firstBad]}");
}
else
    Console.WriteLine("PASS - bit-exact match.");
