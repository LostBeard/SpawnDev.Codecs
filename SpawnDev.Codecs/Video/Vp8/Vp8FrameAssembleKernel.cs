// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP8 frame assembly kernel. Writes the 10-byte uncompressed frame
// tag + start code + size code at output[0..10], then copies
// partition0[0..p0Len] and tokenP0[0..tp0Len] into output. Computes
// the final encoded length and stores it in outLen[0].
//
// Single-thread per frame. Replaces the CPU-side Buffer.BlockCopy
// concatenation in the v2 integration class so the entire encoder
// hot path is GPU-resident.
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
/// length in outLen[0]. Single thread per frame.
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
        Index1D,
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
            int, int>(AssembleKernel);
        _batchKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<byte>, ArrayView<byte>, ArrayView<long>,
            ArrayView<byte>, ArrayView<int>,
            int, int, int, int, int>(AssembleBatchKernel);
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
        _kernel(1, partition0, tokenP0, partLens, output, outLen, width, height);
    }

    /// <summary>
    /// Batch assemble: extent=N, each thread assembles one frame's slot.
    /// outLens layout: 2 longs per frame (p0Len, tpLen). output layout:
    /// outputStride bytes per frame.
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
        _batchKernel(frameCount, partition0, tokenP0, partLens, output, outLen,
            width, height, p0Stride, tp0Stride, outputStride);
    }

    private static void AssembleBatchKernel(
        Index1D idx,
        ArrayView<byte> partition0, ArrayView<byte> tokenP0, ArrayView<long> partLens,
        ArrayView<byte> output, ArrayView<int> outLen,
        int width, int height, int p0Stride, int tp0Stride, int outputStride)
    {
        int f = idx.X;
        var fP0 = partition0.SubView((long)f * p0Stride, p0Stride);
        var fTp = tokenP0.SubView((long)f * tp0Stride, tp0Stride);
        var fPartLens = partLens.SubView((long)f * 2, 2);
        var fOut = output.SubView((long)f * outputStride, outputStride);
        var fOutLen = outLen.SubView(f, 1);
        AssembleBody(fP0, fTp, fPartLens, fOut, fOutLen, width, height);
    }

    private static void AssembleKernel(
        Index1D _,
        ArrayView<byte> partition0,
        ArrayView<byte> tokenP0,
        ArrayView<long> partLens,
        ArrayView<byte> output,
        ArrayView<int> outLen,
        int width, int height)
    {
        AssembleBody(partition0, tokenP0, partLens, output, outLen, width, height);
    }

    private static void AssembleBody(
        ArrayView<byte> partition0,
        ArrayView<byte> tokenP0,
        ArrayView<long> partLens,
        ArrayView<byte> output,
        ArrayView<int> outLen,
        int width, int height)
    {
        long p0Len = partLens[0];
        long tp0Len = partLens[1];

        // Build the 24-bit tag. v1: IsKeyFrame=true (bit 0 = 0),
        // Version=Bicubic=0 (bits 1..3 all zero), ShowFrame=true
        // (bit 4 = 1), FirstPartitionSize = p0Len (bits 5..23).
        uint tagBits = 0u;
        tagBits |= 0x10u;             // show_frame = 1
        tagBits |= ((uint)p0Len & 0x7FFFFu) << 5;

        // Frame tag (3 bytes LE).
        output[0] = (byte)(tagBits & 0xFF);
        output[1] = (byte)((tagBits >> 8) & 0xFF);
        output[2] = (byte)((tagBits >> 16) & 0xFF);

        // Start code (3 bytes).
        output[3] = StartCode0;
        output[4] = StartCode1;
        output[5] = StartCode2;

        // Size codes (2 + 2 bytes LE). v1 horizScale = vertScale = 0.
        int horizSizeCode = width;
        int vertSizeCode = height;
        output[6] = (byte)(horizSizeCode & 0xFF);
        output[7] = (byte)((horizSizeCode >> 8) & 0xFF);
        output[8] = (byte)(vertSizeCode & 0xFF);
        output[9] = (byte)((vertSizeCode >> 8) & 0xFF);

        // Copy partition0 bytes.
        for (long i = 0; i < p0Len; i++)
            output[10 + i] = partition0[i];

        // Copy tokenP0 bytes.
        long tp0Base = 10 + p0Len;
        for (long i = 0; i < tp0Len; i++)
            output[tp0Base + i] = tokenP0[i];

        // Final length.
        outLen[0] = (int)(10 + p0Len + tp0Len);
    }

    /// <summary>Release kernel resources.</summary>
    public void Dispose() { /* auto-grouped */ }
}
