// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 frame assembly kernel. Single-thread dispatch; concatenates
// the three pre-computed byte streams of a v1 keyframe into the
// final output buffer:
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
/// frame output buffer.
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
        Index1D,
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
            ArrayView<long>, ArrayView<long>, ArrayView<long>>(AssembleKernel);
        _batchKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>,
            ArrayView<long>,
            ArrayView<long>, ArrayView<long>, ArrayView<long>,
            Vp9AssembleBatchStrides>(AssembleBatchKernel);
    }

    /// <summary>Batch assemble: extent=N, one thread per frame.</summary>
    public void RunBatch(
        ArrayView<byte> uncompressedHeader, ArrayView<byte> compressedHeader,
        ArrayView<byte> tileBytes, ArrayView<byte> outBuf, ArrayView<long> outLen,
        ArrayView<long> uhLen, ArrayView<long> chLen, ArrayView<long> tileLen,
        int frameCount, Vp9AssembleBatchStrides strides)
    {
        if (frameCount <= 0) throw new ArgumentOutOfRangeException(nameof(frameCount));
        _batchKernel(frameCount,
            uncompressedHeader, compressedHeader, tileBytes, outBuf, outLen,
            uhLen, chLen, tileLen, strides);
    }

    private static void AssembleBatchKernel(
        Index1D idx,
        ArrayView<byte> uncompressedHeader, ArrayView<byte> compressedHeader,
        ArrayView<byte> tileBytes, ArrayView<byte> outBuf, ArrayView<long> outLenOut,
        ArrayView<long> uhLenView, ArrayView<long> chLenView, ArrayView<long> tileLenView,
        Vp9AssembleBatchStrides s)
    {
        int f = idx.X;
        var fUH = uncompressedHeader.SubView((long)f * s.UhStride, s.UhStride);
        var fCH = compressedHeader.SubView((long)f * s.ChStride, s.ChStride);
        var fTile = tileBytes.SubView((long)f * s.TileStride, s.TileStride);
        var fOut = outBuf.SubView((long)f * s.OutStride, s.OutStride);
        var fOutLen = outLenOut.SubView(f, 1);
        var fUhLen = uhLenView.SubView(f, 1);
        var fChLen = chLenView.SubView(f, 1);
        var fTileLen = tileLenView.SubView(f, 1);
        AssembleBody(fUH, fCH, fTile, fOut, fOutLen, fUhLen, fChLen, fTileLen);
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
        _kernel(1, uncompressedHeader, compressedHeader, tileBytes, outBuf, outLen,
                uncompressedLenView, compressedLenView, tileLenView);
    }

    /// <summary>
    /// Convenience overload for tests / standalone callers that already have
    /// the lengths on the host. Allocates 3 single-element scratch views,
    /// uploads, dispatches.
    /// </summary>
    public void Run(
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
        _accelerator.Synchronize();
    }

    private static void AssembleKernel(
        Index1D _,
        ArrayView<byte> uncompressedHeader,
        ArrayView<byte> compressedHeader,
        ArrayView<byte> tileBytes,
        ArrayView<byte> outBuf,
        ArrayView<long> outLenOut,
        ArrayView<long> uncompressedLenView,
        ArrayView<long> compressedLenView,
        ArrayView<long> tileLenView)
    {
        AssembleBody(uncompressedHeader, compressedHeader, tileBytes,
            outBuf, outLenOut,
            uncompressedLenView, compressedLenView, tileLenView);
    }

    private static void AssembleBody(
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
        long pos = 0;
        for (int i = 0; i < uncompressedLen; i++) outBuf[pos++] = uncompressedHeader[i];
        for (int i = 0; i < compressedLen; i++)   outBuf[pos++] = compressedHeader[i];
        for (int i = 0; i < tileLen; i++)         outBuf[pos++] = tileBytes[i];
        outLenOut[0] = pos;
    }

    /// <summary>Release kernel resources.</summary>
    public void Dispose() { /* auto-grouped */ }
}
