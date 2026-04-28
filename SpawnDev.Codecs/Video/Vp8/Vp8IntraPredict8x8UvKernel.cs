// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// ILGPU kernel for VP8 8x8 chroma intra prediction. Bit-exact mirror
// of Vp8IntraPredictor8x8.Predict for all 4 modes (DC, V, H, TM).
// One thread per 8x8 block. Used for both U and V planes (caller
// dispatches twice per macroblock - once per plane).
//
// Memory layout per block (packed):
//   above        :  8 bytes
//   left         :  8 bytes
//   topLeft      :  1 byte
//   modeAndFlags :  1 byte (mode in bits 0-3,
//                           haveAbove in bit 4,
//                           haveLeft  in bit 5)
//   dst          : 64 bytes (8x8 packed, dstStride=8)
//
// Caller scatters dst into the frame buffer after the kernel completes.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp8;

/// <summary>
/// Batched ILGPU kernel for VP8 8x8 chroma intra prediction. One thread
/// per 8x8 block. Bit-exact mirror of <see cref="Vp8IntraPredictor8x8"/>.
/// </summary>
public sealed class Vp8IntraPredict8x8UvKernel : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, int> _kernel;

    /// <summary>Compile the kernel onto <paramref name="accelerator"/>.</summary>
    public Vp8IntraPredict8x8UvKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, int>(PredictKernel);
    }

    /// <summary>Run on N blocks. above=8b, left=8b, topLeft=1b, modeAndFlags=1b, dst=64b.</summary>
    public void Run(
        ArrayView<byte> above,
        ArrayView<byte> left,
        ArrayView<byte> topLeft,
        ArrayView<byte> modeAndFlags,
        ArrayView<byte> dst,
        int blockCount)
    {
        if (blockCount < 0) throw new ArgumentOutOfRangeException(nameof(blockCount));
        if (blockCount == 0) return;
        if (above.Length < blockCount * 8L) throw new ArgumentException("above must hold blockCount*8 bytes.", nameof(above));
        if (left.Length < blockCount * 8L) throw new ArgumentException("left must hold blockCount*8 bytes.", nameof(left));
        if (topLeft.Length < blockCount) throw new ArgumentException("topLeft must hold blockCount bytes.", nameof(topLeft));
        if (modeAndFlags.Length < blockCount) throw new ArgumentException("modeAndFlags must hold blockCount bytes.", nameof(modeAndFlags));
        if (dst.Length < blockCount * 64L) throw new ArgumentException("dst must hold blockCount*64 bytes.", nameof(dst));
        _kernel(blockCount, above, left, topLeft, modeAndFlags, dst, blockCount);
    }

    private static int Clamp255(int v) => v < 0 ? 0 : v > 255 ? 255 : v;

    private static void PredictKernel(
        Index1D blockIdx,
        ArrayView<byte> above,
        ArrayView<byte> left,
        ArrayView<byte> topLeft,
        ArrayView<byte> modeAndFlags,
        ArrayView<byte> dst,
        int blockCount)
    {
        int idx = blockIdx;
        if (idx >= blockCount) return;
        long aBase = (long)idx * 8;
        long lBase = (long)idx * 8;
        long dBase = (long)idx * 64;

        byte mf = modeAndFlags[idx];
        int mode = mf & 0x0F;
        bool haveAbove = (mf & 0x10) != 0;
        bool haveLeft = (mf & 0x20) != 0;
        int tl = topLeft[idx];

        switch (mode)
        {
            case 0: // DcPred
            {
                int dc;
                if (haveAbove && haveLeft)
                {
                    int sum = 0;
                    for (int i = 0; i < 8; i++) sum += above[aBase + i] + left[lBase + i];
                    dc = (sum + 8) >> 4;
                }
                else if (haveAbove)
                {
                    int sum = 0;
                    for (int i = 0; i < 8; i++) sum += above[aBase + i];
                    dc = (sum + 4) >> 3;
                }
                else if (haveLeft)
                {
                    int sum = 0;
                    for (int i = 0; i < 8; i++) sum += left[lBase + i];
                    dc = (sum + 4) >> 3;
                }
                else
                {
                    dc = 128;
                }

                byte dcByte = (byte)dc;
                for (int i = 0; i < 64; i++) dst[dBase + i] = dcByte;
                break;
            }
            case 1: // VPred
            {
                for (int r = 0; r < 8; r++)
                {
                    long row = dBase + r * 8;
                    for (int c = 0; c < 8; c++) dst[row + c] = above[aBase + c];
                }
                break;
            }
            case 2: // HPred
            {
                for (int r = 0; r < 8; r++)
                {
                    byte v = left[lBase + r];
                    long row = dBase + r * 8;
                    for (int c = 0; c < 8; c++) dst[row + c] = v;
                }
                break;
            }
            default: // case 3: TmPred
            {
                for (int r = 0; r < 8; r++)
                {
                    int leftR = left[lBase + r];
                    long row = dBase + r * 8;
                    for (int c = 0; c < 8; c++)
                    {
                        int p = leftR + above[aBase + c] - tl;
                        dst[row + c] = (byte)Clamp255(p);
                    }
                }
                break;
            }
        }
    }

    /// <summary>Release kernel resources. Does NOT dispose the accelerator.</summary>
    public void Dispose() { /* auto-grouped */ }
}
