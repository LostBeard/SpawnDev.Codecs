// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Multi-frame ILGPU driver for FlacFrameWriterGpu. Encodes frameCount frames into a contiguous strided output
// buffer (one per-frame slot of `outBufStride` bytes each). FLAC frames are independent of each other in the spec,
// and every byte of a VERBATIM frame sits at an offset known up front, so no thread walks a whole frame (a frame
// is up to 8 channels x 4096 samples; one GPU thread per frame used to hold the GPU for the whole block):
//
//   FrameHeader     1 thread per frame            header bits + CRC-8 (at most 12 bytes)
//   FramePack       1 thread per subframe byte    FlacVerbatimSubframeGpu.PackByteAt (8 bits each)
//   FrameCrcPiece   1 thread per (frame, piece)   CRC-16 register of FlacCrcGpu.PieceBytes (4096) bytes
//   FrameCrcCombine 1 thread per frame            folds the pieces (a frame is at most ~513 pieces), writes the
//                                                 CRC-16 footer and the frame length
//
// Frames go in groups of at most LaunchBytes (4 MiB) of subframe bytes per launch, so the launch count is known on
// the host (no readback) and every launch is short whatever the stream length. The host paces the launches with
// LaunchPacer (the browser's command encoder must not collect a long queue).

using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.Codecs.Audio.Flac;

/// <summary>
/// Multi-frame FLAC encoder kernels. Every frame is written into its own slot of the strided output buffer by
/// bounded parallel launches (see the file header).
/// </summary>
public sealed class FlacFrameWriterGpuKernel : IDisposable
{
    /// <summary>Subframe bytes (pack threads) of one launch group of frames.</summary>
    public const int LaunchBytes = 1 << 22;

    private readonly Accelerator _accelerator;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _tables;
    private readonly Action<Index1D, ArrayView<byte>, int, ulong, int, int> _headerKernel;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<byte>, FlacFrameShape, int, long> _packKernel;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, FlacFrameShape, int, int> _pieceKernel;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<long>, ArrayView<int>, ArrayView<int>, FlacFrameShape, int, int, int> _combineKernel;

    /// <summary>Compile.</summary>
    public FlacFrameWriterGpuKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _tables = accelerator.Allocate1D<int>(FlacCrcGpu.TableInts);
        _tables.View.CopyFromCPU(FlacCrcGpu.BuildTables());
        _headerKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, int, ulong, int, int>(FlacFrameHeaderKernel);
        _packKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>, ArrayView<byte>, FlacFrameShape, int, long>(FlacFramePackKernel);
        _pieceKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, FlacFrameShape, int, int>(FlacFrameCrcPieceKernel);
        _combineKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<long>, ArrayView<int>, ArrayView<int>, FlacFrameShape, int, int, int>(FlacFrameCrcCombineKernel);
    }

    /// <summary>
    /// Encode one FLAC frame on the accelerator (single-frame call). Completes when the frame is written.
    /// </summary>
    public Task RunAsync(
        ArrayView<int> samples,
        ArrayView<byte> outBuf,
        ArrayView<long> outLen,
        int blockSize, int channels, int bps,
        ulong frameNumber)
        // frameCount=1, samplesStride=0, outBufStride=0 -> single-frame mode.
        => RunBatchAsync(samples, outBuf, outLen, blockSize, channels, bps, frameNumber, 1, 0, 0);

    /// <summary>
    /// Batch-encode <paramref name="frameCount"/> FLAC frames. Frame <c>i</c> reads from
    /// <c>samples[i*samplesStride..]</c> (channel-major, samplesStride samples per frame) and writes to
    /// <c>outBuf[i*outBufStride..]</c> with the per-frame byte length stored in <c>outLen[i]</c>. Completes when
    /// every frame is written.
    /// </summary>
    public async Task RunBatchAsync(
        ArrayView<int> samples,
        ArrayView<byte> outBuf,
        ArrayView<long> outLen,
        int blockSize, int channels, int bps,
        ulong startFrameNumber,
        int frameCount,
        int samplesStride,
        int outBufStride)
    {
        if (frameCount <= 0) throw new ArgumentOutOfRangeException(nameof(frameCount));
        var shape = new FlacFrameShape
        {
            BlockSize = blockSize, Channels = channels, Bps = bps, StartFrameNumber = startFrameNumber,
            SamplesStride = samplesStride, OutBufStride = outBufStride,
        };
        long subBytes = FlacVerbatimSubframeGpu.SubframesBytes(blockSize, channels, bps);
        long maxPre = FlacFrameWriterGpu.PreFooterBytes(blockSize, channels, bps, startFrameNumber + (ulong)(frameCount - 1));
        int maxPieces = FlacCrcGpu.PiecesOf(maxPre);
        int combineSteps = FlacCrcGpu.CombineSteps(maxPieces);
        int framesPerLaunch = (int)Math.Max(1, Math.Min(frameCount, LaunchBytes / Math.Max(1, subBytes)));

        using var pieces = _accelerator.Allocate1D<int>((long)frameCount * maxPieces);
        var pacer = new LaunchPacer(_accelerator);
        for (int f0 = 0; f0 < frameCount; f0 += framesPerLaunch)
        {
            int count = Math.Min(framesPerLaunch, frameCount - f0);
            _headerKernel(count, outBuf, channels, startFrameNumber + (ulong)f0, outBufStride, f0);
            await pacer.LaunchedAsync();
            _packKernel(checked((int)(count * subBytes)), samples, outBuf, shape, f0, subBytes);
            await pacer.LaunchedAsync();
            _pieceKernel(count * maxPieces, outBuf, pieces.View, _tables.View, shape, f0, maxPieces);
            await pacer.LaunchedAsync();
            for (int step = 0; step < combineSteps; step++)
            {
                _combineKernel(count, outBuf, outLen, pieces.View, _tables.View, shape, f0, maxPieces, step);
                await pacer.LaunchedAsync();
            }
        }
        await _accelerator.SynchronizeAsync();
    }

    /// <summary>Frame <c>f0 + i</c>: header bits + CRC-8 at the start of its slot.</summary>
    private static void FlacFrameHeaderKernel(
        Index1D i, ArrayView<byte> outBuf, int channels, ulong firstFrameNumber, int outBufStride, int f0)
    {
        long outOff = (long)(f0 + i.X) * outBufStride;
        FlacFrameWriterGpu.WriteHeader(outBuf, outOff, channels, firstFrameNumber + (ulong)i.X);
    }

    /// <summary>Subframe byte <c>t % subBytes</c> of frame <c>f0 + t / subBytes</c>.</summary>
    private static void FlacFramePackKernel(
        Index1D t, ArrayView<int> samples, ArrayView<byte> outBuf, FlacFrameShape s, int f0, long subBytes)
    {
        long fi = t.X / subBytes;
        long j = t.X - fi * subBytes;
        long f = f0 + fi;
        ulong frameNumber = s.StartFrameNumber + (ulong)f;
        long outOff = f * s.OutBufStride + FlacFrameWriterGpu.SubframesStart(frameNumber);
        outBuf[outOff + j] = FlacVerbatimSubframeGpu.PackByteAt(
            samples, f * s.SamplesStride, s.BlockSize, s.BlockSize, s.Channels, s.Bps, j);
    }

    /// <summary>CRC-16 piece <c>t % maxPieces</c> of the pre-footer bytes of frame <c>f0 + t / maxPieces</c>.</summary>
    private static void FlacFrameCrcPieceKernel(
        Index1D t, ArrayView<byte> outBuf, ArrayView<int> pieces, ArrayView<int> tables, FlacFrameShape s,
        int f0, int maxPieces)
    {
        int fi = t.X / maxPieces;
        int piece = t.X - fi * maxPieces;
        long f = f0 + fi;
        long pre = FlacFrameWriterGpu.PreFooterBytes(s.BlockSize, s.Channels, s.Bps, s.StartFrameNumber + (ulong)f);
        if (piece < FlacCrcGpu.PiecesOf(pre))
        {
            FlacCrcGpu.PieceSpan(pre, piece, out long start, out int count);
            pieces[f * maxPieces + piece] = FlacCrcGpu.Piece16(outBuf, f * s.OutBufStride + start, count, tables);
        }
    }

    /// <summary>
    /// Combine launch <paramref name="step"/> of frame <c>f0 + i</c>: folds its CRC-16 pieces; the launch that folds
    /// the last one writes the big-endian footer and the frame length.
    /// </summary>
    private static void FlacFrameCrcCombineKernel(
        Index1D i, ArrayView<byte> outBuf, ArrayView<long> outLen, ArrayView<int> pieces, ArrayView<int> tables,
        FlacFrameShape s, int f0, int maxPieces, int step)
    {
        long f = f0 + i.X;
        long pre = FlacFrameWriterGpu.PreFooterBytes(s.BlockSize, s.Channels, s.Bps, s.StartFrameNumber + (ulong)f);
        long at = f * maxPieces;
        int finished = FlacCrcGpu.CombineStep16(pieces, at, FlacCrcGpu.PiecesOf(pre), step, tables);
        if (finished != 0)
        {
            int crc16 = pieces[at];
            long outOff = f * s.OutBufStride;
            outBuf[outOff + pre] = (byte)(crc16 >> 8);
            outBuf[outOff + pre + 1] = (byte)(crc16 & 0xFF);
            outLen[f] = pre + 2;
        }
    }

    /// <summary>Release kernel resources.</summary>
    public void Dispose() => _tables.Dispose();
}

/// <summary>Block shape and slot strides of a batch of VERBATIM frames (kernel argument).</summary>
public struct FlacFrameShape
{
    /// <summary>Samples per channel.</summary>
    public int BlockSize;
    /// <summary>Channel count.</summary>
    public int Channels;
    /// <summary>Bits per sample.</summary>
    public int Bps;
    /// <summary>Frame number of frame 0 of the batch.</summary>
    public ulong StartFrameNumber;
    /// <summary>Samples per frame slot of the input (channel-major inside the slot).</summary>
    public int SamplesStride;
    /// <summary>Bytes per frame slot of the output.</summary>
    public int OutBufStride;
}
