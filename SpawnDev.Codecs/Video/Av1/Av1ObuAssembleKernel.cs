// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 OBU assembly on the accelerator. A keyframe temporal unit is
//
//     [Temporal Delimiter OBU][Sequence Header OBU][Frame OBU = header byte, leb128(size), uncompressed header, tile data]
//
// Everything except the tile data and the Frame OBU's size field depends only on (width, height, baseQIndex), so the
// host builds that small fixed blob (TD + SH OBUs, then the uncompressed frame header: well under 128 bytes of
// metadata, the "metadata struct setup" the cardinal rule allows) and uploads it once per frame. This kernel then does
// the data-dependent part: it writes the leb128 size (which depends on the GPU-produced tile length) and copies the
// blob and the tile bytes into the caller's output, one thread per output byte. The tile bytes never visit the host.

using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>Assembles an AV1 keyframe temporal unit from a host-built header blob and GPU-produced tile bytes.</summary>
public sealed class Av1ObuAssembleKernel : IDisposable
{
    /// <summary>The Frame OBU header byte: obu_type 6 (OBU_FRAME), no extension, has_size_field = 1.</summary>
    public const int FrameObuHeaderByte = (6 << 3) | 0x02;

    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<long>, ArrayView<byte>, ArrayView<int>, int, int> _kernel;

    /// <summary>Compile.</summary>
    public Av1ObuAssembleKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<long>, ArrayView<byte>, ArrayView<int>, int, int>(AssembleKernel);
    }

    /// <summary>
    /// Dispatch <paramref name="maxBytes"/> threads. The output is <c>blob[0..aLen)</c> (TD + SH OBUs), the Frame OBU
    /// header byte, leb128(bLen + tileLen), <c>blob[aLen..aLen+bLen)</c> (uncompressed header) and the tile bytes;
    /// its length is written to <c>outLen[0]</c>.
    /// </summary>
    /// <param name="blob">Host-built TD + SH OBUs followed by the uncompressed frame header.</param>
    /// <param name="tile">GPU-produced tile bytes.</param>
    /// <param name="tileLen">One long: the tile length in bytes.</param>
    /// <param name="output">Receives the temporal unit.</param>
    /// <param name="outLen">One int: receives the output length.</param>
    /// <param name="aLen">Bytes of TD + SH OBUs at the start of <paramref name="blob"/>.</param>
    /// <param name="bLen">Bytes of the uncompressed frame header following them.</param>
    /// <param name="maxBytes">Upper bound of the output length (threads to dispatch).</param>
    public void Run(
        ArrayView<byte> blob, ArrayView<byte> tile, ArrayView<long> tileLen,
        ArrayView<byte> output, ArrayView<int> outLen, int aLen, int bLen, int maxBytes)
    {
        if (aLen < 0 || bLen < 0 || blob.Length < aLen + bLen) throw new ArgumentException("blob is shorter than aLen + bLen.", nameof(blob));
        if (outLen.Length < 1 || tileLen.Length < 1) throw new ArgumentException("outLen / tileLen need one element.");
        if (output.Length < maxBytes) throw new ArgumentException("output is shorter than maxBytes.", nameof(output));
        _kernel(maxBytes, blob, tile, tileLen, output, outLen, aLen, bLen);
    }

    private static void AssembleKernel(
        Index1D idx, ArrayView<byte> blob, ArrayView<byte> tile, ArrayView<long> tileLenView,
        ArrayView<byte> output, ArrayView<int> outLen, int aLen, int bLen)
    {
        int tileLen = (int)tileLenView[0];
        int payload = bLen + tileLen;
        // leb128 byte count of the payload size.
        int lebLen = 1;
        int rest = payload >> 7;
        while (rest > 0)
        {
            lebLen++;
            rest >>= 7;
        }
        int total = aLen + 1 + lebLen + payload;
        int i = idx.X;
        if (i == 0) outLen[0] = total;
        if (i >= total) return;

        int value;
        if (i < aLen)
        {
            value = blob[i];
        }
        else
        {
            int j = i - aLen;
            if (j == 0)
            {
                value = FrameObuHeaderByte;
            }
            else if (j <= lebLen)
            {
                int k = j - 1;
                int part = (payload >> (7 * k)) & 0x7F;
                if (k < lebLen - 1) part |= 0x80;
                value = part;
            }
            else
            {
                int m = j - 1 - lebLen;
                if (m < bLen) value = blob[aLen + m];
                else value = tile[m - bLen];
            }
        }
        output[i] = (byte)value;
    }

    /// <summary>No-op (kernels are owned by the accelerator).</summary>
    public void Dispose() { }
}
