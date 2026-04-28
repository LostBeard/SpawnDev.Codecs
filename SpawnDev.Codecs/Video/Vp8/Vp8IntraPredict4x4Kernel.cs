// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// ILGPU kernel for VP8 4x4 intra prediction. Bit-exact mirror of
// Vp8IntraPredictor4x4.Predict for all 10 modes (RFC 6386 sec 12.3).
// One thread per 4x4 block.
//
// Memory layout per block (packed):
//   above  : 9 bytes - above[-1] at index 0, above[0..7] at index 1..8
//   left   :  4 bytes
//   mode   :  1 byte (Vp8IntraMode4x4)
//   dst    : 16 bytes (4x4 packed, dstStride=4)
//
// Caller scatters dst into the frame buffer after the kernel completes.
//
// All 10 modes share a single switch in the kernel body; divergent
// blocks within a warp serialize through unused arms. For best
// throughput, sort blocks by mode and run mode-coherent batches.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp8;

/// <summary>
/// Batched ILGPU kernel for VP8 4x4 intra prediction. One thread per
/// 4x4 block. Bit-exact mirror of <see cref="Vp8IntraPredictor4x4"/>.
/// </summary>
public sealed class Vp8IntraPredict4x4Kernel : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, int> _kernel;

    /// <summary>Compile the kernel onto <paramref name="accelerator"/>.</summary>
    public Vp8IntraPredict4x4Kernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, int>(PredictKernel);
    }

    /// <summary>Run on N blocks. above=9b/block, left=4b/block, mode=1b/block, dst=16b/block.</summary>
    public void Run(
        ArrayView<byte> above,
        ArrayView<byte> left,
        ArrayView<byte> modes,
        ArrayView<byte> dst,
        int blockCount)
    {
        if (blockCount < 0) throw new ArgumentOutOfRangeException(nameof(blockCount));
        if (blockCount == 0) return;
        if (above.Length < blockCount * 9L) throw new ArgumentException("above must hold blockCount*9 bytes.", nameof(above));
        if (left.Length < blockCount * 4L) throw new ArgumentException("left must hold blockCount*4 bytes.", nameof(left));
        if (modes.Length < blockCount) throw new ArgumentException("modes must hold blockCount bytes.", nameof(modes));
        if (dst.Length < blockCount * 16L) throw new ArgumentException("dst must hold blockCount*16 bytes.", nameof(dst));
        _kernel(blockCount, above, left, modes, dst, blockCount);
    }

    private static int Avg3(int a, int b, int c) => (a + 2 * b + c + 2) >> 2;
    private static int Avg2(int a, int b) => (a + b + 1) >> 1;
    private static int Clamp255(int v) => v < 0 ? 0 : v > 255 ? 255 : v;

    private static void PredictKernel(
        Index1D blockIdx,
        ArrayView<byte> above,
        ArrayView<byte> left,
        ArrayView<byte> modes,
        ArrayView<byte> dst,
        int blockCount)
    {
        int idx = blockIdx;
        if (idx >= blockCount) return;
        long aBase = (long)idx * 9;   // above[-1] at aBase + 0; above[0] at aBase + 1
        long lBase = (long)idx * 4;
        long dBase = (long)idx * 16;
        byte mode = modes[idx];

        // Read corners + neighbours into registers.
        int X = above[aBase + 0];                         // above[-1]
        int A = above[aBase + 1], B = above[aBase + 2];   // above[0..7]
        int C = above[aBase + 3], D = above[aBase + 4];
        int E = above[aBase + 5], F = above[aBase + 6];
        int G = above[aBase + 7], H = above[aBase + 8];
        int I = left[lBase + 0], J = left[lBase + 1];
        int K = left[lBase + 2], L = left[lBase + 3];

        // Stack-resident 4x4 destination registers.
        int p00 = 0, p01 = 0, p02 = 0, p03 = 0;
        int p10 = 0, p11 = 0, p12 = 0, p13 = 0;
        int p20 = 0, p21 = 0, p22 = 0, p23 = 0;
        int p30 = 0, p31 = 0, p32 = 0, p33 = 0;

        switch (mode)
        {
            case 0: // BDcPred
            {
                int sum = A + B + C + D + I + J + K + L;
                int dc = (sum + 4) >> 3;
                p00 = p01 = p02 = p03 =
                p10 = p11 = p12 = p13 =
                p20 = p21 = p22 = p23 =
                p30 = p31 = p32 = p33 = dc;
                break;
            }
            case 1: // BTmPred: dst[r,c] = clamp(left[r] + above[c] - X)
            {
                p00 = Clamp255(I + A - X); p01 = Clamp255(I + B - X); p02 = Clamp255(I + C - X); p03 = Clamp255(I + D - X);
                p10 = Clamp255(J + A - X); p11 = Clamp255(J + B - X); p12 = Clamp255(J + C - X); p13 = Clamp255(J + D - X);
                p20 = Clamp255(K + A - X); p21 = Clamp255(K + B - X); p22 = Clamp255(K + C - X); p23 = Clamp255(K + D - X);
                p30 = Clamp255(L + A - X); p31 = Clamp255(L + B - X); p32 = Clamp255(L + C - X); p33 = Clamp255(L + D - X);
                break;
            }
            case 2: // BVePred: filtered vertical
            {
                int v0 = Avg3(X, A, B);
                int v1 = Avg3(A, B, C);
                int v2 = Avg3(B, C, D);
                int v3 = Avg3(C, D, E);
                p00 = p10 = p20 = p30 = v0;
                p01 = p11 = p21 = p31 = v1;
                p02 = p12 = p22 = p32 = v2;
                p03 = p13 = p23 = p33 = v3;
                break;
            }
            case 3: // BHePred: filtered horizontal
            {
                int r0 = Avg3(X, I, J);
                int r1 = Avg3(I, J, K);
                int r2 = Avg3(J, K, L);
                int r3 = Avg3(K, L, L);
                p00 = p01 = p02 = p03 = r0;
                p10 = p11 = p12 = p13 = r1;
                p20 = p21 = p22 = p23 = r2;
                p30 = p31 = p32 = p33 = r3;
                break;
            }
            case 4: // BLdPred: D45e down-left
            {
                p00 = Avg3(A, B, C);
                p01 = p10 = Avg3(B, C, D);
                p02 = p11 = p20 = Avg3(C, D, E);
                p03 = p12 = p21 = p30 = Avg3(D, E, F);
                p13 = p22 = p31 = Avg3(E, F, G);
                p23 = p32 = Avg3(F, G, H);
                p33 = Avg3(G, H, H);
                break;
            }
            case 5: // BRdPred: D135 right-down
            {
                p03 = Avg3(D, C, B);
                p02 = p13 = Avg3(C, B, A);
                p01 = p12 = p23 = Avg3(B, A, X);
                p00 = p11 = p22 = p33 = Avg3(A, X, I);
                p10 = p21 = p32 = Avg3(X, I, J);
                p20 = p31 = Avg3(I, J, K);
                p30 = Avg3(J, K, L);
                break;
            }
            case 6: // BVrPred: D117 vertical-right
            {
                p00 = p21 = Avg2(X, A);
                p01 = p22 = Avg2(A, B);
                p02 = p23 = Avg2(B, C);
                p03      = Avg2(C, D);
                p10 = p31 = Avg3(I, X, A);
                p11 = p32 = Avg3(X, A, B);
                p12 = p33 = Avg3(A, B, C);
                p13      = Avg3(B, C, D);
                p20      = Avg3(J, I, X);
                p30      = Avg3(K, J, I);
                break;
            }
            case 7: // BVlPred: D63e vertical-left
            {
                p00      = Avg2(A, B);
                p01 = p20 = Avg2(B, C);
                p02 = p21 = Avg2(C, D);
                p03 = p22 = Avg2(D, E);
                p23      = Avg3(E, F, G);
                p10      = Avg3(A, B, C);
                p11 = p30 = Avg3(B, C, D);
                p12 = p31 = Avg3(C, D, E);
                p13 = p32 = Avg3(D, E, F);
                p33      = Avg3(E, F, G);
                break;
            }
            case 8: // BHdPred: D153 horizontal-down
            {
                p00 = p12 = Avg2(I, X);
                p10 = p22 = Avg2(J, I);
                p20 = p32 = Avg2(K, J);
                p30      = Avg2(L, K);
                p03      = Avg3(A, B, C);
                p02      = Avg3(X, A, B);
                p01 = p13 = Avg3(I, X, A);
                p11 = p23 = Avg3(J, I, X);
                p21 = p33 = Avg3(K, J, I);
                p31      = Avg3(L, K, J);
                break;
            }
            default: // case 9: BHuPred: D207 horizontal-up
            {
                p00      = Avg2(I, J);
                p02 = p10 = Avg2(J, K);
                p12 = p20 = Avg2(K, L);
                p01      = Avg3(I, J, K);
                p11 = p03 = Avg3(J, K, L);
                p21 = p13 = Avg3(K, L, L);
                p22 = p23 = p30 = p31 = p32 = p33 = L;
                break;
            }
        }

        dst[dBase +  0] = (byte)p00; dst[dBase +  1] = (byte)p01; dst[dBase +  2] = (byte)p02; dst[dBase +  3] = (byte)p03;
        dst[dBase +  4] = (byte)p10; dst[dBase +  5] = (byte)p11; dst[dBase +  6] = (byte)p12; dst[dBase +  7] = (byte)p13;
        dst[dBase +  8] = (byte)p20; dst[dBase +  9] = (byte)p21; dst[dBase + 10] = (byte)p22; dst[dBase + 11] = (byte)p23;
        dst[dBase + 12] = (byte)p30; dst[dBase + 13] = (byte)p31; dst[dBase + 14] = (byte)p32; dst[dBase + 15] = (byte)p33;
    }

    /// <summary>Release kernel resources. Does NOT dispose the accelerator.</summary>
    public void Dispose() { /* auto-grouped */ }
}
