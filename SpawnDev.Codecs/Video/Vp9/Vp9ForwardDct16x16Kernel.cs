// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// ILGPU kernel for the VP9 forward DCT 16x16. Bit-exact mirror of
// Vp9ForwardDct16x16.Transform (the libvpx vpx_fdct16x16_c port). Batched:
// one thread per 16x16 block.
//
// Pass 1: column DCT (input *= 4). Pass 2: row DCT (input rounded via
// `(x + 1) >> 2` - libvpx half_round_shift on the intermediate buffer).
// Intermediate layout: tmp[col*16 + j] = pass1_col-c, output-j. Pass 2
// reads tmp[col + r*16] for r=0..15 (column-major read of intermediate).
//
// Mirrors slice 123 (iDCT 16x16 kernel) - LocalMemory<int>(256) for the
// row-pass scratch, MethodImpl.NoInlining on the butterfly to keep WGSL
// shader size sane. 256 ints x 4 bytes = 1024 bytes per thread.

using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>Batched ILGPU kernel for the VP9 forward DCT 16x16.</summary>
public sealed class Vp9ForwardDct16x16Kernel : IDisposable
{
    private const int CosPi2_64  = 16305;
    private const int CosPi4_64  = 16069;
    private const int CosPi6_64  = 15679;
    private const int CosPi8_64  = 15137;
    private const int CosPi10_64 = 14449;
    private const int CosPi12_64 = 13623;
    private const int CosPi14_64 = 12665;
    private const int CosPi16_64 = 11585;
    private const int CosPi18_64 = 10394;
    private const int CosPi20_64 = 9102;
    private const int CosPi22_64 = 7723;
    private const int CosPi24_64 = 6270;
    private const int CosPi26_64 = 4756;
    private const int CosPi28_64 = 3196;
    private const int CosPi30_64 = 1606;

    private const int DctConstBits = 14;
    private const int DctConstRounding = 1 << (DctConstBits - 1);

    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<short>, ArrayView<int>, int> _kernel;

    /// <summary>Compile the kernel onto <paramref name="accelerator"/>.</summary>
    public Vp9ForwardDct16x16Kernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<short>, ArrayView<int>, int>(FdctKernel);
    }

    /// <summary>
    /// Run the FDCT on <paramref name="blockCount"/> 16x16 blocks. Each
    /// block: 256 contiguous shorts in / 256 contiguous ints out.
    /// </summary>
    public void Run(ArrayView<short> input, ArrayView<int> output, int blockCount)
    {
        if (blockCount < 0) throw new ArgumentOutOfRangeException(nameof(blockCount));
        if (blockCount == 0) return;
        if (input.Length < blockCount * 256L)
            throw new ArgumentException($"input must hold at least blockCount*256 shorts.", nameof(input));
        if (output.Length < blockCount * 256L)
            throw new ArgumentException($"output must hold at least blockCount*256 ints.", nameof(output));
        _kernel(blockCount, input, output, blockCount);
    }

    /// <summary>
    /// Convenience: allocate temporary GPU buffers, run, copy back.
    /// </summary>
    public async Task RunAsync(
        ReadOnlyMemory<short> input, Memory<int> output, int blockCount)
    {
        if (blockCount <= 0) return;
        using var dIn = _accelerator.Allocate1D<short>(blockCount * 256);
        using var dOut = _accelerator.Allocate1D<int>(blockCount * 256);
        dIn.View.CopyFromCPU(input.Span.ToArray());
        _kernel(blockCount, dIn.View, dOut.View, blockCount);
        await _accelerator.SynchronizeAsync();
        var readBack = await dOut.CopyToHostAsync();
        readBack.AsSpan(0, blockCount * 256).CopyTo(output.Span);
    }

    private static void FdctKernel(
        Index1D blockIdx,
        ArrayView<short> input,
        ArrayView<int> output,
        int blockCount)
    {
        int idx = blockIdx;
        if (idx >= blockCount) return;

        long inBase = (long)idx * 256;
        long outBase = (long)idx * 256;

        // tmp[col*16 + j] holds pass-1 column-c, output-j (intermediate).
        var tmp = LocalMemory.Allocate<int>(256);

        // Pass 1: 16 column DCTs (input * 4).
        for (int col = 0; col < 16; col++)
        {
            int ih0 = (input[inBase + col +  0 * 16] + input[inBase + col + 15 * 16]) * 4;
            int ih1 = (input[inBase + col +  1 * 16] + input[inBase + col + 14 * 16]) * 4;
            int ih2 = (input[inBase + col +  2 * 16] + input[inBase + col + 13 * 16]) * 4;
            int ih3 = (input[inBase + col +  3 * 16] + input[inBase + col + 12 * 16]) * 4;
            int ih4 = (input[inBase + col +  4 * 16] + input[inBase + col + 11 * 16]) * 4;
            int ih5 = (input[inBase + col +  5 * 16] + input[inBase + col + 10 * 16]) * 4;
            int ih6 = (input[inBase + col +  6 * 16] + input[inBase + col +  9 * 16]) * 4;
            int ih7 = (input[inBase + col +  7 * 16] + input[inBase + col +  8 * 16]) * 4;
            int s10 = (input[inBase + col +  7 * 16] - input[inBase + col +  8 * 16]) * 4;
            int s11 = (input[inBase + col +  6 * 16] - input[inBase + col +  9 * 16]) * 4;
            int s12 = (input[inBase + col +  5 * 16] - input[inBase + col + 10 * 16]) * 4;
            int s13 = (input[inBase + col +  4 * 16] - input[inBase + col + 11 * 16]) * 4;
            int s14 = (input[inBase + col +  3 * 16] - input[inBase + col + 12 * 16]) * 4;
            int s15 = (input[inBase + col +  2 * 16] - input[inBase + col + 13 * 16]) * 4;
            int s16 = (input[inBase + col +  1 * 16] - input[inBase + col + 14 * 16]) * 4;
            int s17 = (input[inBase + col +  0 * 16] - input[inBase + col + 15 * 16]) * 4;

            ButterflyAndStore(
                ih0, ih1, ih2, ih3, ih4, ih5, ih6, ih7,
                s10, s11, s12, s13, s14, s15, s16, s17,
                out int o0,  out int o1,  out int o2,  out int o3,
                out int o4,  out int o5,  out int o6,  out int o7,
                out int o8,  out int o9,  out int o10, out int o11,
                out int o12, out int o13, out int o14, out int o15);

            int b = col * 16;
            tmp[b +  0] = o0;  tmp[b +  1] = o1;  tmp[b +  2] = o2;  tmp[b +  3] = o3;
            tmp[b +  4] = o4;  tmp[b +  5] = o5;  tmp[b +  6] = o6;  tmp[b +  7] = o7;
            tmp[b +  8] = o8;  tmp[b +  9] = o9;  tmp[b + 10] = o10; tmp[b + 11] = o11;
            tmp[b + 12] = o12; tmp[b + 13] = o13; tmp[b + 14] = o14; tmp[b + 15] = o15;
        }

        // Pass 2: 16 row DCTs - input rounded via (x + 1) >> 2.
        // Read tmp[col + r*16] for r=0..15 to traverse one column of
        // intermediate = pass1_col-r, output-col.
        for (int col = 0; col < 16; col++)
        {
            int ih0 = ((tmp[col +  0 * 16] + 1) >> 2) + ((tmp[col + 15 * 16] + 1) >> 2);
            int ih1 = ((tmp[col +  1 * 16] + 1) >> 2) + ((tmp[col + 14 * 16] + 1) >> 2);
            int ih2 = ((tmp[col +  2 * 16] + 1) >> 2) + ((tmp[col + 13 * 16] + 1) >> 2);
            int ih3 = ((tmp[col +  3 * 16] + 1) >> 2) + ((tmp[col + 12 * 16] + 1) >> 2);
            int ih4 = ((tmp[col +  4 * 16] + 1) >> 2) + ((tmp[col + 11 * 16] + 1) >> 2);
            int ih5 = ((tmp[col +  5 * 16] + 1) >> 2) + ((tmp[col + 10 * 16] + 1) >> 2);
            int ih6 = ((tmp[col +  6 * 16] + 1) >> 2) + ((tmp[col +  9 * 16] + 1) >> 2);
            int ih7 = ((tmp[col +  7 * 16] + 1) >> 2) + ((tmp[col +  8 * 16] + 1) >> 2);
            int s10 = ((tmp[col +  7 * 16] + 1) >> 2) - ((tmp[col +  8 * 16] + 1) >> 2);
            int s11 = ((tmp[col +  6 * 16] + 1) >> 2) - ((tmp[col +  9 * 16] + 1) >> 2);
            int s12 = ((tmp[col +  5 * 16] + 1) >> 2) - ((tmp[col + 10 * 16] + 1) >> 2);
            int s13 = ((tmp[col +  4 * 16] + 1) >> 2) - ((tmp[col + 11 * 16] + 1) >> 2);
            int s14 = ((tmp[col +  3 * 16] + 1) >> 2) - ((tmp[col + 12 * 16] + 1) >> 2);
            int s15 = ((tmp[col +  2 * 16] + 1) >> 2) - ((tmp[col + 13 * 16] + 1) >> 2);
            int s16 = ((tmp[col +  1 * 16] + 1) >> 2) - ((tmp[col + 14 * 16] + 1) >> 2);
            int s17 = ((tmp[col +  0 * 16] + 1) >> 2) - ((tmp[col + 15 * 16] + 1) >> 2);

            ButterflyAndStore(
                ih0, ih1, ih2, ih3, ih4, ih5, ih6, ih7,
                s10, s11, s12, s13, s14, s15, s16, s17,
                out int o0,  out int o1,  out int o2,  out int o3,
                out int o4,  out int o5,  out int o6,  out int o7,
                out int o8,  out int o9,  out int o10, out int o11,
                out int o12, out int o13, out int o14, out int o15);

            int b = (int)outBase + col * 16;
            output[b +  0] = o0;  output[b +  1] = o1;  output[b +  2] = o2;  output[b +  3] = o3;
            output[b +  4] = o4;  output[b +  5] = o5;  output[b +  6] = o6;  output[b +  7] = o7;
            output[b +  8] = o8;  output[b +  9] = o9;  output[b + 10] = o10; output[b + 11] = o11;
            output[b + 12] = o12; output[b + 13] = o13; output[b + 14] = o14; output[b + 15] = o15;
        }
    }

    /// <summary>
    /// 16-point 1D forward DCT butterfly. Bit-exact against the reference's
    /// inner loop. NoInlining keeps WGSL shader size manageable - same
    /// reasoning as Vp9Idct16x16Kernel.Idct16Row.
    /// </summary>
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void ButterflyAndStore(
        int ih0, int ih1, int ih2, int ih3, int ih4, int ih5, int ih6, int ih7,
        int s10, int s11, int s12, int s13, int s14, int s15, int s16, int s17,
        out int o0,  out int o1,  out int o2,  out int o3,
        out int o4,  out int o5,  out int o6,  out int o7,
        out int o8,  out int o9,  out int o10, out int o11,
        out int o12, out int o13, out int o14, out int o15)
    {
        // Even half: fdct8 on inHigh[0..7] (locals ih0..ih7).
        int s0 = ih0 + ih7;
        int s1 = ih1 + ih6;
        int s2 = ih2 + ih5;
        int s3 = ih3 + ih4;
        int s4 = ih3 - ih4;
        int s5 = ih2 - ih5;
        int s6 = ih1 - ih6;
        int s7 = ih0 - ih7;

        int x0 = s0 + s3, x1 = s1 + s2, x2 = s1 - s2, x3 = s0 - s3;
        long t0 = (long)(x0 + x1) * CosPi16_64;
        long t1 = (long)(x0 - x1) * CosPi16_64;
        long t2 = (long)x3 * CosPi8_64 + (long)x2 * CosPi24_64;
        long t3 = (long)x3 * CosPi24_64 - (long)x2 * CosPi8_64;
        o0  = RoundShift(t0);
        o4  = RoundShift(t2);
        o8  = RoundShift(t1);
        o12 = RoundShift(t3);

        long u0 = (long)(s6 - s5) * CosPi16_64;
        long u1 = (long)(s6 + s5) * CosPi16_64;
        int v2 = RoundShift(u0);
        int v3 = RoundShift(u1);

        int y0 = s4 + v2, y1 = s4 - v2, y2 = s7 - v3, y3 = s7 + v3;
        long w0 = (long)y0 * CosPi28_64 + (long)y3 * CosPi4_64;
        long w1 = (long)y1 * CosPi12_64 + (long)y2 * CosPi20_64;
        long w2 = (long)y2 * CosPi12_64 + (long)y1 * (-CosPi20_64);
        long w3 = (long)y3 * CosPi28_64 + (long)y0 * (-CosPi4_64);
        o2  = RoundShift(w0);
        o6  = RoundShift(w2);
        o10 = RoundShift(w1);
        o14 = RoundShift(w3);

        // Odd half. step1[0..7] = locals s10..s17 (s1X where X is the index).
        // Stage 2.
        //   step2[2] = RS((step1[5] - step1[2]) * c16) = RS((s15 - s12) * c16)
        //   step2[3] = RS((step1[4] - step1[3]) * c16) = RS((s14 - s13) * c16)
        //   step2[4] = RS((step1[4] + step1[3]) * c16) = RS((s14 + s13) * c16)
        //   step2[5] = RS((step1[5] + step1[2]) * c16) = RS((s15 + s12) * c16)
        long temp1 = (long)(s15 - s12) * CosPi16_64;
        long temp2 = (long)(s14 - s13) * CosPi16_64;
        int sa2 = RoundShift(temp1);  // step2[2]
        int sa3 = RoundShift(temp2);  // step2[3]
        temp1 = (long)(s14 + s13) * CosPi16_64;
        temp2 = (long)(s15 + s12) * CosPi16_64;
        int sa4 = RoundShift(temp1);  // step2[4]
        int sa5 = RoundShift(temp2);  // step2[5]

        // Stage 3.
        //   step3[0] = step1[0] + step2[3] = s10 + sa3
        //   step3[1] = step1[1] + step2[2] = s11 + sa2
        //   step3[2] = step1[1] - step2[2] = s11 - sa2
        //   step3[3] = step1[0] - step2[3] = s10 - sa3
        //   step3[4] = step1[7] - step2[4] = s17 - sa4
        //   step3[5] = step1[6] - step2[5] = s16 - sa5
        //   step3[6] = step1[6] + step2[5] = s16 + sa5
        //   step3[7] = step1[7] + step2[4] = s17 + sa4
        int sb0 = s10 + sa3;          // step3[0]
        int sb1 = s11 + sa2;          // step3[1]
        int sb2 = s11 - sa2;          // step3[2]
        int sb3 = s10 - sa3;          // step3[3]
        int sb4 = s17 - sa4;          // step3[4]
        int sb5 = s16 - sa5;          // step3[5]
        int sb6 = s16 + sa5;          // step3[6]
        int sb7 = s17 + sa4;          // step3[7]

        // Stage 4.
        temp1 = (long)sb1 * (-CosPi8_64) + (long)sb6 * CosPi24_64;
        temp2 = (long)sb2 * CosPi24_64 + (long)sb5 * CosPi8_64;
        int sc1 = RoundShift(temp1);  // step2[1]
        int sc2 = RoundShift(temp2);  // step2[2]
        temp1 = (long)sb2 * CosPi8_64 - (long)sb5 * CosPi24_64;
        temp2 = (long)sb1 * CosPi24_64 + (long)sb6 * CosPi8_64;
        int sc5 = RoundShift(temp1);  // step2[5]
        int sc6 = RoundShift(temp2);  // step2[6]

        // Stage 5. Restore step1 layout.
        int sd0 = sb0 + sc1;
        int sd1 = sb0 - sc1;
        int sd2 = sb3 + sc2;
        int sd3 = sb3 - sc2;
        int sd4 = sb4 - sc5;
        int sd5 = sb4 + sc5;
        int sd6 = sb7 - sc6;
        int sd7 = sb7 + sc6;

        // Stage 6.
        temp1 = (long)sd0 * CosPi30_64 + (long)sd7 * CosPi2_64;
        temp2 = (long)sd1 * CosPi14_64 + (long)sd6 * CosPi18_64;
        o1 = RoundShift(temp1);
        o9 = RoundShift(temp2);
        temp1 = (long)sd2 * CosPi22_64 + (long)sd5 * CosPi10_64;
        temp2 = (long)sd3 * CosPi6_64 + (long)sd4 * CosPi26_64;
        o5  = RoundShift(temp1);
        o13 = RoundShift(temp2);
        temp1 = (long)sd3 * (-CosPi26_64) + (long)sd4 * CosPi6_64;
        temp2 = (long)sd2 * (-CosPi10_64) + (long)sd5 * CosPi22_64;
        o3  = RoundShift(temp1);
        o11 = RoundShift(temp2);
        temp1 = (long)sd1 * (-CosPi18_64) + (long)sd6 * CosPi14_64;
        temp2 = (long)sd0 * (-CosPi2_64) + (long)sd7 * CosPi30_64;
        o7  = RoundShift(temp1);
        o15 = RoundShift(temp2);
    }

    private static int RoundShift(long input) =>
        (int)((input + DctConstRounding) >> DctConstBits);

    /// <summary>Release kernel resources. Does NOT dispose the accelerator.</summary>
    public void Dispose() { /* auto-grouped */ }
}
