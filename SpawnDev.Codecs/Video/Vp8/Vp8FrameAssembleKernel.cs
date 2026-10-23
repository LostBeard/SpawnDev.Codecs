// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP8 frame assembly kernel. Writes the 10-byte uncompressed frame
// tag + start code + size code at output[0..10], then copies
// partition0[0..p0Len] and tokenP0[0..tp0Len] into output. Computes
// the final encoded length and stores it in outLen[0] (-1 when a partition
// length exceeds its buffer: the stream overflowed and nothing is assembled).
//
// One thread per output byte (the batch form adds a frame dimension): a
// single thread copying the whole frame grew with the bitstream. Replaces
// the CPU-side Buffer.BlockCopy concatenation in the v2 integration class
// so the entire encoder hot path is GPU-resident.
//
// VP8 keyframe tag layout (RFC 6386 sec 9.1 / 19.1):
//   bytes [0..3]   3-byte uncompressed tag (frame_type, version,
//                  show_frame, first_partition_size)
//   bytes [3..6]   3-byte VP8 start code 0x9D 0x01 0x2A
//   bytes [6..8]   horiz_size_code = width | (horizScale << 14) (2 bytes LE)
//   bytes [8..10]  vert_size_code = height | (vertScale << 14) (2 bytes LE)
//
// v1 simplifications: ShowFrame=true, IsKeyFrame=true, Version=Bicubic=0,
// horizScale=vertScale=0.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp8;

/// <summary>
/// VP8 frame assembly kernel. Writes tag + start code + size code +
/// partition0 + tokenP0 into a single output buffer; stores final
/// length in outLen[0]. One thread per output byte.
/// </summary>
public sealed class Vp8FrameAssembleKernel : IDisposable
{
    private const byte StartCode0 = 0x9D;
    private const byte StartCode1 = 0x01;
    private const byte StartCode2 = 0x2A;

    private readonly Accelerator _accelerator;
    private readonly Action<
        Index1D,
        ArrayView<byte>, ArrayView<byte>, ArrayView<long>,
        ArrayView<byte>, ArrayView<int>,
        int, int> _kernel;

    private readonly Action<
        Index2D,
        ArrayView<byte>, ArrayView<byte>, ArrayView<long>,
        ArrayView<byte>, ArrayView<int>,
        int, int, int, int, int> _batchKernel;

    /// <summary>Compile.</summary>
    public Vp8FrameAssembleKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<byte>, ArrayView<byte>, ArrayView<long>,
            ArrayView<byte>, ArrayView<int>,
            int, int>(Vp8AssembleKernel);
        _batchKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index2D,
            ArrayView<byte>, ArrayView<byte>, ArrayView<long>,
            ArrayView<byte>, ArrayView<int>,
            int, int, int, int, int>(Vp8AssembleBatchKernel);
    }

    /// <summary>Run the assembly. partLens = [partition0Len, tokenP0Len].</summary>
    public void Run(
        ArrayView<byte> partition0,
        ArrayView<byte> tokenP0,
        ArrayView<long> partLens,
        ArrayView<byte> output,
        ArrayView<int> outLen,
        int width, int height)
    {
        if (width <= 0 || height <= 0) throw new ArgumentOutOfRangeException();
        if (partLens.Length < 2)
            throw new ArgumentException("partLens must hold 2 longs.", nameof(partLens));
        if (outLen.Length < 1)
            throw new ArgumentException("outLen must hold at least 1 int.", nameof(outLen));
        // Upper bound of the frame length: the tag plus both partitions at their capacity.
        long maxBytes = Math.Min(output.Length, 10 + partition0.Length + tokenP0.Length);
        _kernel(checked((int)maxBytes), partition0, tokenP0, partLens, output, outLen, width, height);
    }

    /// <summary>
    /// Batch assemble: one thread per (output byte, frame). outLens layout: 2 longs per frame (p0Len, tpLen).
    /// output layout: outputStride bytes per frame.
    /// </summary>
    public void RunBatch(
        ArrayView<byte> partition0,
        ArrayView<byte> tokenP0,
        ArrayView<long> partLens,
        ArrayView<byte> output,
        ArrayView<int> outLen,
        int width, int height,
        int frameCount, int p0Stride, int tp0Stride, int outputStride)
    {
        if (frameCount <= 0) throw new ArgumentOutOfRangeException(nameof(frameCount));
        int maxBytes = Math.Min(outputStride, 10 + p0Stride + tp0Stride);
        _batchKernel(new Index2D(maxBytes, frameCount), partition0, tokenP0, partLens, output, outLen,
            width, height, p0Stride, tp0Stride, outputStride);
    }

    private static void Vp8AssembleBatchKernel(
        Index2D idx,
        ArrayView<byte> partition0, ArrayView<byte> tokenP0, ArrayView<long> partLens,
        ArrayView<byte> output, ArrayView<int> outLen,
        int width, int height, int p0Stride, int tp0Stride, int outputStride)
    {
        int f = idx.Y;
        var fP0 = partition0.SubView((long)f * p0Stride, p0Stride);
        var fTp = tokenP0.SubView((long)f * tp0Stride, tp0Stride);
        var fPartLens = partLens.SubView((long)f * 2, 2);
        var fOut = output.SubView((long)f * outputStride, outputStride);
        var fOutLen = outLen.SubView(f, 1);
        AssembleByte(idx.X, fP0, fTp, fPartLens, fOut, fOutLen, width, height);
    }

    private static void Vp8AssembleKernel(
        Index1D idx,
        ArrayView<byte> partition0,
        ArrayView<byte> tokenP0,
        ArrayView<long> partLens,
        ArrayView<byte> output,
        ArrayView<int> outLen,
        int width, int height)
    {
        AssembleByte(idx.X, partition0, tokenP0, partLens, output, outLen, width, height);
    }

    /// <summary>Write output byte <paramref name="i"/> of the frame (thread 0 also writes the length).</summary>
    private static void AssembleByte(
        int i,
        ArrayView<byte> partition0,
        ArrayView<byte> tokenP0,
        ArrayView<long> partLens,
        ArrayView<byte> output,
        ArrayView<int> outLen,
        int width, int height)
    {
        long p0Len = partLens[0];
        long tp0Len = partLens[1];
        // A partition that outgrew its buffer (the bool encoder counted on but stopped writing): report -1 and
        // assemble nothing.
        bool overflow = (p0Len > partition0.Length) | (tp0Len > tokenP0.Length);
        if (overflow)
        {
            if (i == 0) outLen[0] = -1;
            return;
        }
        long total = 10 + p0Len + tp0Len;
        if (i == 0) outLen[0] = (int)total;
        if (i >= total) return;

        if (i >= 10)
        {
            long j = i - 10;
            if (j < p0Len) output[i] = partition0[j];
            else output[i] = tokenP0[j - p0Len];
            return;
        }

        // Build the 24-bit tag. v1: IsKeyFrame=true (bit 0 = 0),
        // Version=Bicubic=0 (bits 1..3 all zero), ShowFrame=true
        // (bit 4 = 1), FirstPartitionSize = p0Len (bits 5..23).
        uint tagBits = 0u;
        tagBits |= 0x10u;             // show_frame = 1
        tagBits |= ((uint)p0Len & 0x7FFFFu) << 5;

        // Frame tag (3 bytes LE), start code (3 bytes), size codes (2 + 2 bytes LE; v1 horizScale =
        // vertScale = 0).
        int value;
        if (i < 3) value = (int)((tagBits >> (8 * i)) & 0xFF);
        else if (i == 3) value = StartCode0;
        else if (i == 4) value = StartCode1;
        else if (i == 5) value = StartCode2;
        else if (i == 6) value = width & 0xFF;
        else if (i == 7) value = (width >> 8) & 0xFF;
        else if (i == 8) value = height & 0xFF;
        else value = (height >> 8) & 0xFF;
        output[i] = (byte)value;
    }

    /// <summary>Release kernel resources.</summary>
    public void Dispose() { /* auto-grouped */ }
}
