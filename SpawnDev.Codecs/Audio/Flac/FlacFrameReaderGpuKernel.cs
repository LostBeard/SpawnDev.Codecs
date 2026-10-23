// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Single-frame ILGPU driver for FlacFrameReaderGpu. Decodes one VERBATIM FLAC frame from the input bytes; writes
// decoded samples to the output buffer. No thread walks the frame (a frame is up to 8 channels x 65535 samples):
//
//   FrameParse         1 thread               header, CRC-8, subframe headers (FlacFrameReaderGpu.ParseHeader)
//   FrameUnpack        1 thread per sample    FlacVerbatimSubframeGpu.UnpackSampleAt, LaunchSamples per launch
//   FrameVerifyPiece   1 thread per piece     CRC-16 register of FlacCrcGpu.PieceBytes (4096) frame bytes
//   FrameVerifyCombine 1 thread               folds the pieces (at most ~513), checks the footer, frame length
//
// The launch count follows from the block shape alone (no readback). Status precedence matches the serial reader:
// a header failure (1, 2, 4) stops the later stages; the CRC-16 (3) is checked last.

using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.Codecs.Audio.Flac;

/// <summary>
/// Single-frame FLAC decoder kernels (see the file header).
/// </summary>
public sealed class FlacFrameReaderGpuKernel : IDisposable
{
    /// <summary>Samples (threads) of one unpack launch.</summary>
    public const int LaunchSamples = 1 << 22;

    private readonly Accelerator _accelerator;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _tables;
    private MemoryBuffer1D<int, Stride1D.Dense>? _pieces;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<long>, long, int, int, int, int> _parseKernel;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<long>, long, int, int, int> _unpackKernel;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<long>, ArrayView<int>, long, long> _pieceKernel;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<long>, ArrayView<int>, ArrayView<int>, long, long, int> _combineKernel;

    /// <summary>Compile.</summary>
    public FlacFrameReaderGpuKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _tables = accelerator.Allocate1D<int>(FlacCrcGpu.TableInts);
        _tables.View.CopyFromCPU(FlacCrcGpu.BuildTables());
        _parseKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<long>, long, int, int, int, int>(FlacFrameParseKernel);
        _unpackKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<long>, long, int, int, int>(FlacFrameUnpackKernel);
        _pieceKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<long>, ArrayView<int>, long, long>(FlacFrameVerifyPieceKernel);
        _combineKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<long>, ArrayView<int>, ArrayView<int>, long, long, int>(FlacFrameVerifyCombineKernel);
    }

    /// <summary>
    /// Decode one FLAC frame on the accelerator; completes when the frame is decoded. Outputs:
    /// <c>samples</c> = decoded samples (channel-major);
    /// <c>statusOut</c>[0] = 0 success or non-zero error code;
    /// <c>frameLen</c>[0] = bytes consumed (0 on error).
    /// </summary>
    public async Task RunAsync(
        ArrayView<byte> data,
        ArrayView<int> samples,
        ArrayView<int> statusOut,
        ArrayView<long> frameLen,
        long frameBase, int frameLength,
        int blockSize, int channels, int bps)
    {
        long subBytes = FlacVerbatimSubframeGpu.SubframesBytes(blockSize, channels, bps);
        int maxPieces = FlacCrcGpu.PiecesOf(FlacFrameReaderGpu.MaxSubframesStart + subBytes);
        // RunAsync completes its work before returning, so a smaller scratch from an earlier frame is idle here.
        if (_pieces is null || _pieces.Length < maxPieces)
        {
            _pieces?.Dispose();
            _pieces = _accelerator.Allocate1D<int>(maxPieces);
        }
        var pacer = new LaunchPacer(_accelerator);
        _parseKernel(1, data, statusOut, frameLen, frameBase, frameLength, blockSize, channels, bps);
        await pacer.LaunchedAsync();
        long total = (long)blockSize * channels;
        for (long first = 0; first < total; first += LaunchSamples)
        {
            _unpackKernel((int)Math.Min(LaunchSamples, total - first), data, samples, statusOut, frameLen,
                frameBase, blockSize, bps, (int)first);
            await pacer.LaunchedAsync();
        }
        _pieceKernel(maxPieces, data, _pieces.View, statusOut, frameLen, _tables.View, frameBase, subBytes);
        await pacer.LaunchedAsync();
        int steps = FlacCrcGpu.CombineSteps(maxPieces);
        for (int step = 0; step < steps; step++)
        {
            _combineKernel(1, data, statusOut, frameLen, _pieces.View, _tables.View, frameBase, subBytes, step);
            await pacer.LaunchedAsync();
        }
        await _accelerator.SynchronizeAsync();
    }

    /// <summary>Header parse; leaves the subframe start in frameLen[0] (0 on failure) for the later stages.</summary>
    private static void FlacFrameParseKernel(
        Index1D _,
        ArrayView<byte> data, ArrayView<int> statusOut, ArrayView<long> frameLen,
        long frameBase, int frameLength,
        int blockSize, int channels, int bps)
    {
        frameLen[0] = FlacFrameReaderGpu.ParseHeader(
            data, frameBase, frameLength, blockSize, channels, bps, statusOut);
    }

    /// <summary>Sample <c>first + t</c> of the channel-major frame (channel = index / blockSize).</summary>
    private static void FlacFrameUnpackKernel(
        Index1D t,
        ArrayView<byte> data, ArrayView<int> samples, ArrayView<int> statusOut, ArrayView<long> frameLen,
        long frameBase, int blockSize, int bps, int first)
    {
        if (statusOut[0] != 0) return;
        int s = first + t.X;
        int ch = s / blockSize;
        int i = s - ch * blockSize;
        samples[s] = FlacVerbatimSubframeGpu.UnpackSampleAt(data, frameBase + frameLen[0], blockSize, bps, ch, i);
    }

    /// <summary>CRC-16 piece <paramref name="piece"/> of the frame bytes before the footer.</summary>
    private static void FlacFrameVerifyPieceKernel(
        Index1D piece,
        ArrayView<byte> data, ArrayView<int> pieces, ArrayView<int> statusOut, ArrayView<long> frameLen,
        ArrayView<int> tables, long frameBase, long subBytes)
    {
        if (statusOut[0] != 0) return;
        long pre = frameLen[0] + subBytes;
        if (piece.X < FlacCrcGpu.PiecesOf(pre))
        {
            FlacCrcGpu.PieceSpan(pre, piece.X, out long start, out int count);
            pieces[piece.X] = FlacCrcGpu.Piece16(data, frameBase + start, count, tables);
        }
    }

    /// <summary>
    /// Combine launch <paramref name="step"/>: folds the pieces; the launch that folds the last one checks the
    /// footer (status 3 on mismatch) and writes the frame length (0 on any failure).
    /// </summary>
    private static void FlacFrameVerifyCombineKernel(
        Index1D _,
        ArrayView<byte> data, ArrayView<int> statusOut, ArrayView<long> frameLen,
        ArrayView<int> pieces, ArrayView<int> tables, long frameBase, long subBytes, int step)
    {
        if (statusOut[0] != 0)
        {
            frameLen[0] = 0;
            return;
        }
        long pre = frameLen[0] + subBytes;
        int finished = FlacCrcGpu.CombineStep16(pieces, 0, FlacCrcGpu.PiecesOf(pre), step, tables);
        if (finished != 0)
        {
            int expectedCrc16 = pieces[0];
            int actualCrc16 = (data[frameBase + pre] << 8) | data[frameBase + pre + 1];
            if (expectedCrc16 != actualCrc16)
            {
                statusOut[0] = 3;
                frameLen[0] = 0;
            }
            else
            {
                frameLen[0] = pre + 2;
            }
        }
    }

    /// <summary>Release kernel resources.</summary>
    public void Dispose()
    {
        _tables.Dispose();
        _pieces?.Dispose();
    }
}
