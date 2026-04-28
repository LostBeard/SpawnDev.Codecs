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
        int, int, int> _kernel;

    /// <summary>Compile.</summary>
    public Vp9FrameAssembleKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>,
            ArrayView<long>,
            int, int, int>(AssembleKernel);
    }

    /// <summary>
    /// Concatenate the three byte streams into <paramref name="outBuf"/>.
    /// <paramref name="outLen"/> receives the total byte count.
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
        if (outLen.Length < 1) throw new ArgumentException("outLen must hold 1 entry.", nameof(outLen));
        if (uncompressedLen < 0) throw new ArgumentOutOfRangeException(nameof(uncompressedLen));
        if (compressedLen < 0) throw new ArgumentOutOfRangeException(nameof(compressedLen));
        if (tileLen < 0) throw new ArgumentOutOfRangeException(nameof(tileLen));
        long total = (long)uncompressedLen + compressedLen + tileLen;
        if (outBuf.Length < total)
            throw new ArgumentException(
                $"outBuf too short ({outBuf.Length}) for total payload ({total}).",
                nameof(outBuf));
        _kernel(1, uncompressedHeader, compressedHeader, tileBytes, outBuf, outLen,
                uncompressedLen, compressedLen, tileLen);
    }

    private static void AssembleKernel(
        Index1D _,
        ArrayView<byte> uncompressedHeader,
        ArrayView<byte> compressedHeader,
        ArrayView<byte> tileBytes,
        ArrayView<byte> outBuf,
        ArrayView<long> outLenOut,
        int uncompressedLen,
        int compressedLen,
        int tileLen)
    {
        long pos = 0;
        for (int i = 0; i < uncompressedLen; i++) outBuf[pos++] = uncompressedHeader[i];
        for (int i = 0; i < compressedLen; i++)   outBuf[pos++] = compressedHeader[i];
        for (int i = 0; i < tileLen; i++)         outBuf[pos++] = tileBytes[i];
        outLenOut[0] = pos;
    }

    /// <summary>Release kernel resources.</summary>
    public void Dispose() { /* auto-grouped */ }
}
