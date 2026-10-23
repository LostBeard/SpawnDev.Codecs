// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 frame assembly kernel. One thread per output byte (the batch form
// adds a frame dimension; a single thread copying the whole frame grew
// with the bitstream); concatenates the three pre-computed byte streams
// of a v1 keyframe into the final output buffer:
//
//   [uncompressed header bytes]
//   [compressed header bytes]
//   [tile data bytes]
//
// V1 uses a single tile (Log2NumTiles = 0, Log2TileRows = 0). Per
// VP9 spec sec 6.3.1 the last tile in the frame omits the per-tile
// size prefix because it spans to end-of-frame, so a single-tile
// frame has zero tile-size prefix bytes. That means the assembly
// is a pure 3-stream concatenation - no length headers between the
// runs, no per-tile size words.
//
// The uncompressed header already encodes
// first_partition_size = compressed.Length, so the decoder can
// locate the boundary between compressed header and tile data
// without seeing it in the bytestream.

using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>Per-frame strides for Vp9FrameAssembleKernel batch dispatch.</summary>
public struct Vp9AssembleBatchStrides
{
    /// <summary>Uncompressed-header bytes per frame.</summary>
    public int UhStride;
    /// <summary>Compressed-header bytes per frame.</summary>
    public int ChStride;
    /// <summary>Tile bytes per frame (worst case).</summary>
    public int TileStride;
    /// <summary>Output bytes per frame (worst case).</summary>
    public int OutStride;
}

/// <summary>
/// VP9 frame assembly kernel. Concatenates the uncompressed header,
/// compressed header, and tile data byte streams into the final
/// frame output buffer, one thread per output byte.
/// </summary>
public sealed class Vp9FrameAssembleKernel : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Action<
        Index1D,
        ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>,
        ArrayView<long>,
        ArrayView<long>, ArrayView<long>, ArrayView<long>> _kernel;

    private readonly Action<
        Index2D,
        ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>,
        ArrayView<long>,
        ArrayView<long>, ArrayView<long>, ArrayView<long>,
        Vp9AssembleBatchStrides> _batchKernel;

    /// <summary>Compile.</summary>
    public Vp9FrameAssembleKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>,
            ArrayView<long>,
            ArrayView<long>, ArrayView<long>, ArrayView<long>>(Vp9AssembleKernel);
        _batchKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index2D,
            ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>,
            ArrayView<long>,
            ArrayView<long>, ArrayView<long>, ArrayView<long>,
            Vp9AssembleBatchStrides>(Vp9AssembleBatchKernel);
    }

    /// <summary>Batch assemble: one thread per (output byte, frame).</summary>
    public void RunBatch(
        ArrayView<byte> uncompressedHeader, ArrayView<byte> compressedHeader,
        ArrayView<byte> tileBytes, ArrayView<byte> outBuf, ArrayView<long> outLen,
        ArrayView<long> uhLen, ArrayView<long> chLen, ArrayView<long> tileLen,
        int frameCount, Vp9AssembleBatchStrides strides)
    {
        if (frameCount <= 0) throw new ArgumentOutOfRangeException(nameof(frameCount));
        long maxBytes = Math.Min(strides.OutStride, (long)strides.UhStride + strides.ChStride + strides.TileStride);
        _batchKernel(new Index2D(checked((int)maxBytes), frameCount),
            uncompressedHeader, compressedHeader, tileBytes, outBuf, outLen,
            uhLen, chLen, tileLen, strides);
    }

    private static void Vp9AssembleBatchKernel(
        Index2D idx,
        ArrayView<byte> uncompressedHeader, ArrayView<byte> compressedHeader,
        ArrayView<byte> tileBytes, ArrayView<byte> outBuf, ArrayView<long> outLenOut,
        ArrayView<long> uhLenView, ArrayView<long> chLenView, ArrayView<long> tileLenView,
        Vp9AssembleBatchStrides s)
    {
        int f = idx.Y;
        var fUH = uncompressedHeader.SubView((long)f * s.UhStride, s.UhStride);
        var fCH = compressedHeader.SubView((long)f * s.ChStride, s.ChStride);
        var fTile = tileBytes.SubView((long)f * s.TileStride, s.TileStride);
        var fOut = outBuf.SubView((long)f * s.OutStride, s.OutStride);
        var fOutLen = outLenOut.SubView(f, 1);
        var fUhLen = uhLenView.SubView(f, 1);
        var fChLen = chLenView.SubView(f, 1);
        var fTileLen = tileLenView.SubView(f, 1);
        AssembleByte(idx.X, fUH, fCH, fTile, fOut, fOutLen, fUhLen, fChLen, fTileLen);
    }

    /// <summary>
    /// GPU-resident path: read the three lengths from their views (which
    /// are the same buffers the upstream kernels wrote them to). The
    /// host does not need to sync + read them back to launch this kernel.
    /// </summary>
    public void Run(
        ArrayView<byte> uncompressedHeader,
        ArrayView<byte> compressedHeader,
        ArrayView<byte> tileBytes,
        ArrayView<byte> outBuf,
        ArrayView<long> outLen,
        ArrayView<long> uncompressedLenView,
        ArrayView<long> compressedLenView,
        ArrayView<long> tileLenView)
    {
        if (outLen.Length < 1) throw new ArgumentException("outLen must hold 1 entry.", nameof(outLen));
        if (uncompressedLenView.Length < 1) throw new ArgumentException("uncompressedLenView must hold 1 entry.", nameof(uncompressedLenView));
        if (compressedLenView.Length < 1) throw new ArgumentException("compressedLenView must hold 1 entry.", nameof(compressedLenView));
        if (tileLenView.Length < 1) throw new ArgumentException("tileLenView must hold 1 entry.", nameof(tileLenView));
        // Upper bound of the frame length: the three streams at their capacity. At least one thread, which
        // writes the length even for an empty frame.
        long maxBytes = Math.Min(outBuf.Length, uncompressedHeader.Length + compressedHeader.Length + tileBytes.Length);
        if (maxBytes < 1) maxBytes = 1;
        _kernel(checked((int)maxBytes), uncompressedHeader, compressedHeader, tileBytes, outBuf, outLen,
                uncompressedLenView, compressedLenView, tileLenView);
    }

    /// <summary>
    /// Convenience overload for tests / standalone callers that already have
    /// the lengths on the host. Allocates 3 single-element scratch views,
    /// uploads, dispatches.
    /// </summary>
    public async Task RunAsync(
        ArrayView<byte> uncompressedHeader,
        ArrayView<byte> compressedHeader,
        ArrayView<byte> tileBytes,
        ArrayView<byte> outBuf,
        ArrayView<long> outLen,
        int uncompressedLen,
        int compressedLen,
        int tileLen)
    {
        if (uncompressedLen < 0) throw new ArgumentOutOfRangeException(nameof(uncompressedLen));
        if (compressedLen < 0) throw new ArgumentOutOfRangeException(nameof(compressedLen));
        if (tileLen < 0) throw new ArgumentOutOfRangeException(nameof(tileLen));
        long total = (long)uncompressedLen + compressedLen + tileLen;
        if (outBuf.Length < total)
            throw new ArgumentException(
                $"outBuf too short ({outBuf.Length}) for total payload ({total}).",
                nameof(outBuf));
        using var sU = _accelerator.Allocate1D<long>(1);
        using var sC = _accelerator.Allocate1D<long>(1);
        using var sT = _accelerator.Allocate1D<long>(1);
        sU.View.CopyFromCPU(new[] { (long)uncompressedLen });
        sC.View.CopyFromCPU(new[] { (long)compressedLen });
        sT.View.CopyFromCPU(new[] { (long)tileLen });
        Run(uncompressedHeader, compressedHeader, tileBytes, outBuf, outLen,
            sU.View, sC.View, sT.View);
        await _accelerator.SynchronizeAsync();
    }

    private static void Vp9AssembleKernel(
        Index1D idx,
        ArrayView<byte> uncompressedHeader,
        ArrayView<byte> compressedHeader,
        ArrayView<byte> tileBytes,
        ArrayView<byte> outBuf,
        ArrayView<long> outLenOut,
        ArrayView<long> uncompressedLenView,
        ArrayView<long> compressedLenView,
        ArrayView<long> tileLenView)
    {
        AssembleByte(idx.X, uncompressedHeader, compressedHeader, tileBytes,
            outBuf, outLenOut,
            uncompressedLenView, compressedLenView, tileLenView);
    }

    /// <summary>Write output byte <paramref name="i"/> (thread 0 also writes the length).</summary>
    private static void AssembleByte(
        int i,
        ArrayView<byte> uncompressedHeader,
        ArrayView<byte> compressedHeader,
        ArrayView<byte> tileBytes,
        ArrayView<byte> outBuf,
        ArrayView<long> outLenOut,
        ArrayView<long> uncompressedLenView,
        ArrayView<long> compressedLenView,
        ArrayView<long> tileLenView)
    {
        int uncompressedLen = (int)uncompressedLenView[0];
        int compressedLen = (int)compressedLenView[0];
        int tileLen = (int)tileLenView[0];
        // A stream that outgrew its buffer (the bool encoder counted on but stopped writing): report -1 and
        // assemble nothing.
        bool overflow = (uncompressedLen > uncompressedHeader.Length) | (compressedLen > compressedHeader.Length)
            | (tileLen > tileBytes.Length);
        if (overflow)
        {
            if (i == 0) outLenOut[0] = -1;
            return;
        }
        long total = (long)uncompressedLen + compressedLen + tileLen;
        if (i == 0) outLenOut[0] = total;
        if (i >= total) return;
        if (i < uncompressedLen) outBuf[i] = uncompressedHeader[i];
        else if (i < uncompressedLen + compressedLen) outBuf[i] = compressedHeader[i - uncompressedLen];
        else outBuf[i] = tileBytes[i - uncompressedLen - compressedLen];
    }

    /// <summary>Release kernel resources.</summary>
    public void Dispose() { /* auto-grouped */ }
}
