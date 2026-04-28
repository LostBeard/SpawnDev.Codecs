// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// ILGPU kernel for VP8 16x16 intra prediction. Bit-exact mirror of
// Vp8IntraPredictor16x16.Predict for all 4 modes (DC, V, H, TM).
// One thread per 16x16 macroblock.
//
// Memory layout per block (packed):
//   above        :  16 bytes
//   left         :  16 bytes
//   topLeft      :   1 byte
//   modeAndFlags :   1 byte (mode in bits 0-3,
//                            haveAbove in bit 4,
//                            haveLeft  in bit 5)
//   dst          : 256 bytes (16x16 packed, dstStride=16)
//
// Caller scatters dst into the frame buffer after the kernel completes.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp8;

/// <summary>
/// Batched ILGPU kernel for VP8 16x16 intra prediction. One thread per
/// 16x16 macroblock. Bit-exact mirror of <see cref="Vp8IntraPredictor16x16"/>.
/// </summary>
public sealed class Vp8IntraPredict16x16Kernel : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, int> _kernel;

    /// <summary>Compile the kernel onto <paramref name="accelerator"/>.</summary>
    public Vp8IntraPredict16x16Kernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, int>(PredictKernel);
    }

    /// <summary>Run on N macroblocks. above=16b, left=16b, topLeft=1b, modeAndFlags=1b, dst=256b.</summary>
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
        if (above.Length < blockCount * 16L) throw new ArgumentException("above must hold blockCount*16 bytes.", nameof(above));
        if (left.Length < blockCount * 16L) throw new ArgumentException("left must hold blockCount*16 bytes.", nameof(left));
        if (topLeft.Length < blockCount) throw new ArgumentException("topLeft must hold blockCount bytes.", nameof(topLeft));
        if (modeAndFlags.Length < blockCount) throw new ArgumentException("modeAndFlags must hold blockCount bytes.", nameof(modeAndFlags));
        if (dst.Length < blockCount * 256L) throw new ArgumentException("dst must hold blockCount*256 bytes.", nameof(dst));
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
        long aBase = (long)idx * 16;
        long lBase = (long)idx * 16;
        long dBase = (long)idx * 256;

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
                    for (int i = 0; i < 16; i++) sum += above[aBase + i] + left[lBase + i];
                    dc = (sum + 16) >> 5;
                }
                else if (haveAbove)
                {
                    int sum = 0;
                    for (int i = 0; i < 16; i++) sum += above[aBase + i];
                    dc = (sum + 8) >> 4;
                }
                else if (haveLeft)
                {
                    int sum = 0;
                    for (int i = 0; i < 16; i++) sum += left[lBase + i];
                    dc = (sum + 8) >> 4;
                }
                else
                {
                    dc = 128;
                }

                byte dcByte = (byte)dc;
                for (int i = 0; i < 256; i++) dst[dBase + i] = dcByte;
                break;
            }
            case 1: // VPred
            {
                for (int r = 0; r < 16; r++)
                {
                    long row = dBase + r * 16;
                    for (int c = 0; c < 16; c++) dst[row + c] = above[aBase + c];
                }
                break;
            }
            case 2: // HPred
            {
                for (int r = 0; r < 16; r++)
                {
                    byte v = left[lBase + r];
                    long row = dBase + r * 16;
                    for (int c = 0; c < 16; c++) dst[row + c] = v;
                }
                break;
            }
            default: // case 3: TmPred
            {
                for (int r = 0; r < 16; r++)
                {
                    int leftR = left[lBase + r];
                    long row = dBase + r * 16;
                    for (int c = 0; c < 16; c++)
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
