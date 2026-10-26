// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Shuffle filters on the accelerator: byte shuffle (c-blosc / numcodecs Shuffle) and bitshuffle (kiyo-masui/bitshuffle,
// the HDF5 / Blosc BITSHUFFLE filter). Neither compresses anything; they regroup the bytes (or bits) of fixed-size
// elements so an entropy coder or LZ stage downstream sees long runs of similar values (the exponent bytes of a float
// tensor, the high bytes of positions, ...). Both are exact permutations, bit-identical with the reference libraries.
//
// Byte shuffle of n = length / E elements of E bytes: out[j * n + i] = in[i * E + j]; a trailing partial element
// (length % E bytes) is copied raw (c-blosc2 shuffle_generic_inline).
//
// Bitshuffle works on blocks of B elements (B a multiple of 8; default bshuf_default_block_size: 8192 / E rounded down
// to a multiple of 8, at least 128). Full blocks first, then one last block of (n % B) rounded down to a multiple of 8,
// then the leftover n % 8 elements copied raw. Inside a block of m elements the reference transposes bytes, then bits
// within bytes (TRANS_BIT_8X8), then rows of bits; composed, that is: bit row r = 8 * j + b (byte j of the element,
// bit b) is m / 8 bytes long, and bit k of its byte q is bit b of byte j of element 8 * q + k. So every output byte is a
// gather of 8 input bits, and the inverse is a gather too: one thread per output byte, no thread writes anywhere but
// its own byte (which also keeps WebGL, without scatter stores, on the same kernels).
//
// No launch grows with the input: a call is cut into launches of LaunchBytes output bytes, paced by LaunchPacer.
// Lengths are int (2^31 - 1 bytes per call); a larger tensor is filtered in slices the caller cuts on element (and,
// for bitshuffle, block) boundaries.
//
// API shape: the same three overloads as the codecs (ArrayView1D / managed / browser TypedArray). The ArrayView1D
// overloads only queue the launches (chain the next stage on the same accelerator without a wait); the managed and
// TypedArray overloads stage, run, and read the result back.

using System.Runtime.Versioning;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.SpawnJS.JSObjects;

namespace SpawnDev.Codecs.Compression;

/// <summary>Byte shuffle and bitshuffle filters (and their inverses) on the accelerator.</summary>
public sealed class ShuffleFilterGpu : IDisposable
{
    /// <summary>Output bytes one launch writes (one thread each).</summary>
    public const int LaunchBytes = 1 << 22;

    /// <summary>Bitshuffle block sizes must be a multiple of this many elements (BSHUF_BLOCKED_MULT).</summary>
    public const int BitShuffleBlockMultiple = 8;

    private enum Op { Shuffle, Unshuffle, BitShuffle, BitUnshuffle }

    private readonly Accelerator _accelerator;
    private readonly LaunchPacer _pacer;
    // (src, dst, element size, bitshuffle block elements, length, launch offset); at most 7 kernel parameters in all.
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, int, int, int, int> _shuffle, _unshuffle, _bitShuffle, _bitUnshuffle;
    private readonly StageBuffer<byte> _stageIn, _stageOut;

    /// <summary>Compiles the filter kernels for <paramref name="accelerator"/>.</summary>
    public ShuffleFilterGpu(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _pacer = new LaunchPacer(accelerator);
        _shuffle = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<byte>, int, int, int, int>(ShuffleKernel);
        _unshuffle = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<byte>, int, int, int, int>(UnshuffleKernel);
        _bitShuffle = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<byte>, int, int, int, int>(BitShuffleKernel);
        _bitUnshuffle = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<byte>, int, int, int, int>(BitUnshuffleKernel);
        _stageIn = new StageBuffer<byte>(accelerator);
        _stageOut = new StageBuffer<byte>(accelerator);
    }

    /// <summary>The reference library's default bitshuffle block (in elements) for <paramref name="elementSize"/>-byte elements.</summary>
    public static int DefaultBitShuffleBlock(int elementSize)
    {
        if (elementSize < 1) throw new ArgumentOutOfRangeException(nameof(elementSize));
        int block = 8192 / elementSize / BitShuffleBlockMultiple * BitShuffleBlockMultiple;
        return Math.Max(block, 128);
    }

    // ------------------------------------------------------------------
    // ArrayView1D overloads (queue only)
    // ------------------------------------------------------------------

    /// <summary>Byte shuffle of the first <paramref name="length"/> bytes of <paramref name="src"/> into <paramref name="dst"/> (must not overlap).</summary>
    public ValueTask ShuffleAsync(ArrayView1D<byte, Stride1D.Dense> src, ArrayView1D<byte, Stride1D.Dense> dst, int length, int elementSize)
        => RunAsync(Op.Shuffle, src, dst, length, elementSize, 0);

    /// <summary>Inverse of <see cref="ShuffleAsync(ArrayView1D{byte, Stride1D.Dense}, ArrayView1D{byte, Stride1D.Dense}, int, int)"/>.</summary>
    public ValueTask UnshuffleAsync(ArrayView1D<byte, Stride1D.Dense> src, ArrayView1D<byte, Stride1D.Dense> dst, int length, int elementSize)
        => RunAsync(Op.Unshuffle, src, dst, length, elementSize, 0);

    /// <summary>
    /// Bitshuffle of the first <paramref name="length"/> bytes of <paramref name="src"/> (whole elements plus any trailing
    /// partial element, which is copied raw) into <paramref name="dst"/> (must not overlap). <paramref name="blockElements"/>
    /// 0 = <see cref="DefaultBitShuffleBlock"/>.
    /// </summary>
    public ValueTask BitShuffleAsync(ArrayView1D<byte, Stride1D.Dense> src, ArrayView1D<byte, Stride1D.Dense> dst, int length, int elementSize, int blockElements = 0)
        => RunAsync(Op.BitShuffle, src, dst, length, elementSize, blockElements);

    /// <summary>Inverse of <see cref="BitShuffleAsync(ArrayView1D{byte, Stride1D.Dense}, ArrayView1D{byte, Stride1D.Dense}, int, int, int)"/> (same block size).</summary>
    public ValueTask BitUnshuffleAsync(ArrayView1D<byte, Stride1D.Dense> src, ArrayView1D<byte, Stride1D.Dense> dst, int length, int elementSize, int blockElements = 0)
        => RunAsync(Op.BitUnshuffle, src, dst, length, elementSize, blockElements);

    // ------------------------------------------------------------------
    // Managed overloads
    // ------------------------------------------------------------------

    /// <summary>Byte shuffle of <paramref name="data"/>.</summary>
    public Task<byte[]> ShuffleAsync(ReadOnlyMemory<byte> data, int elementSize) => RunManagedAsync(Op.Shuffle, data, elementSize, 0);

    /// <summary>Inverse byte shuffle of <paramref name="data"/>.</summary>
    public Task<byte[]> UnshuffleAsync(ReadOnlyMemory<byte> data, int elementSize) => RunManagedAsync(Op.Unshuffle, data, elementSize, 0);

    /// <summary>Bitshuffle of <paramref name="data"/>.</summary>
    public Task<byte[]> BitShuffleAsync(ReadOnlyMemory<byte> data, int elementSize, int blockElements = 0) => RunManagedAsync(Op.BitShuffle, data, elementSize, blockElements);

    /// <summary>Inverse bitshuffle of <paramref name="data"/>.</summary>
    public Task<byte[]> BitUnshuffleAsync(ReadOnlyMemory<byte> data, int elementSize, int blockElements = 0) => RunManagedAsync(Op.BitUnshuffle, data, elementSize, blockElements);

    // ------------------------------------------------------------------
    // Browser TypedArray overloads (the bytes never enter the .NET heap)
    // ------------------------------------------------------------------

    /// <summary>Byte shuffle of <paramref name="data"/> (its bytes); the caller disposes the result.</summary>
    [SupportedOSPlatform("browser")]
    public Task<Uint8Array> ShuffleAsync(TypedArray data, int elementSize) => RunTypedAsync(Op.Shuffle, data, elementSize, 0);

    /// <summary>Inverse byte shuffle of <paramref name="data"/>; the caller disposes the result.</summary>
    [SupportedOSPlatform("browser")]
    public Task<Uint8Array> UnshuffleAsync(TypedArray data, int elementSize) => RunTypedAsync(Op.Unshuffle, data, elementSize, 0);

    /// <summary>Bitshuffle of <paramref name="data"/>; the caller disposes the result.</summary>
    [SupportedOSPlatform("browser")]
    public Task<Uint8Array> BitShuffleAsync(TypedArray data, int elementSize, int blockElements = 0) => RunTypedAsync(Op.BitShuffle, data, elementSize, blockElements);

    /// <summary>Inverse bitshuffle of <paramref name="data"/>; the caller disposes the result.</summary>
    [SupportedOSPlatform("browser")]
    public Task<Uint8Array> BitUnshuffleAsync(TypedArray data, int elementSize, int blockElements = 0) => RunTypedAsync(Op.BitUnshuffle, data, elementSize, blockElements);

    // ------------------------------------------------------------------

    private async Task<byte[]> RunManagedAsync(Op op, ReadOnlyMemory<byte> data, int elementSize, int blockElements)
    {
        int length = data.Length;
        var src = _stageIn.Ensure(length);
        var dst = _stageOut.Ensure(length);
        if (length > 0) DeviceTransfer.Upload(src, data.Span);
        await RunAsync(op, src, dst, length, elementSize, blockElements);
        await _accelerator.SynchronizeAsync();
        return await DeviceTransfer.DownloadAsync(dst, length);
    }

    [SupportedOSPlatform("browser")]
    private async Task<Uint8Array> RunTypedAsync(Op op, TypedArray data, int elementSize, int blockElements)
    {
        ArgumentNullException.ThrowIfNull(data);
        long byteLength = data.ByteLength;
        if (byteLength > int.MaxValue) throw new ArgumentOutOfRangeException(nameof(data), "At most 2^31 - 1 bytes per call.");
        int length = (int)byteLength;
        var src = _stageIn.Ensure(length);
        var dst = _stageOut.Ensure(length);
        if (length > 0) DeviceTransfer.Upload(src, data);
        await RunAsync(op, src, dst, length, elementSize, blockElements);
        await _accelerator.SynchronizeAsync();
        return await DeviceTransfer.DownloadUint8ArrayAsync(dst, length);
    }

    private async ValueTask RunAsync(Op op, ArrayView1D<byte, Stride1D.Dense> src, ArrayView1D<byte, Stride1D.Dense> dst, int length, int elementSize, int blockElements)
    {
        if (elementSize < 1) throw new ArgumentOutOfRangeException(nameof(elementSize), "Element size must be at least 1 byte.");
        if (length < 0 || length > src.Length) throw new ArgumentOutOfRangeException(nameof(length), $"{length} bytes; the source holds {src.Length}.");
        if (length > dst.Length) throw new ArgumentException($"The destination holds {dst.Length} bytes; {length} are needed.", nameof(dst));
        if (length == 0) return;

        int block = 0;
        if (op == Op.BitShuffle || op == Op.BitUnshuffle)
        {
            block = blockElements == 0 ? DefaultBitShuffleBlock(elementSize) : blockElements;
            if (block <= 0 || block % BitShuffleBlockMultiple != 0)
                throw new ArgumentOutOfRangeException(nameof(blockElements), $"Block size {blockElements} is not a positive multiple of {BitShuffleBlockMultiple} elements.");
            if ((long)block * elementSize > int.MaxValue) throw new ArgumentOutOfRangeException(nameof(blockElements));
        }

        for (int offset = 0; offset < length; offset += LaunchBytes)
        {
            int count = Math.Min(LaunchBytes, length - offset);
            // Each launch gets the destination from its own first byte: thread idx stores dst[idx], its own slot
            // (WebGL's Transform Feedback captures a thread's store only there).
            var part = dst.SubView(offset, count);
            switch (op)
            {
                case Op.Shuffle: _shuffle(count, src, part, elementSize, block, length, offset); break;
                case Op.Unshuffle: _unshuffle(count, src, part, elementSize, block, length, offset); break;
                case Op.BitShuffle: _bitShuffle(count, src, part, elementSize, block, length, offset); break;
                default: _bitUnshuffle(count, src, part, elementSize, block, length, offset); break;
            }
            await _pacer.LaunchedAsync();
        }
    }

    // ------------------------------------------------------------------
    // Kernels: one thread per output byte o = offset + idx (a gather). dst is the launch's part of the output (it starts at
    // byte offset), so every thread stores only dst[idx]: its own slot on every backend, WebGL included.
    // ------------------------------------------------------------------

    private static void ShuffleKernel(Index1D idx, ArrayView<byte> src, ArrayView<byte> dst, int elementSize, int block, int length, int offset)
    {
        int o = offset + idx.X;
        if (o >= length) return;
        int elements = length / elementSize;
        int s = o;
        if (o < elements * elementSize)
        {
            int j = o / elements;
            int i = o - j * elements;
            s = i * elementSize + j;
        }
        dst[idx.X] = src[s];
    }

    private static void UnshuffleKernel(Index1D idx, ArrayView<byte> src, ArrayView<byte> dst, int elementSize, int block, int length, int offset)
    {
        int o = offset + idx.X;
        if (o >= length) return;
        int elements = length / elementSize;
        int s = o;
        if (o < elements * elementSize)
        {
            int i = o / elementSize;
            int j = o - i * elementSize;
            s = j * elements + i;
        }
        dst[idx.X] = src[s];
    }

    /// <summary>Byte offset of the bitshuffle block (full, or the last rounded-down one) holding output byte <paramref name="o"/>.</summary>
    private static int BitBlockBase(int o, int elementSize, int block, int length)
    {
        int blockBytes = block * elementSize;
        int fullBytes = length / elementSize / block * blockBytes;
        int baseByte = fullBytes;
        if (o < fullBytes) baseByte = o / blockBytes * blockBytes;
        return baseByte;
    }

    /// <summary>
    /// Elements (a multiple of 8) of the bitshuffle block at <paramref name="baseByte"/> when output byte <paramref name="o"/>
    /// lies inside it; 0 when o lies in the raw tail (the n % 8 leftover elements and any trailing partial element).
    /// </summary>
    private static int BitBlockElements(int o, int baseByte, int elementSize, int block, int length)
    {
        int elements = length / elementSize;
        int fullBlocks = elements / block;
        int n = block;
        if (baseByte >= fullBlocks * block * elementSize)
        {
            n = elements - fullBlocks * block;
            n -= n & 7;
        }
        if (o - baseByte >= n * elementSize) n = 0;
        return n;
    }

    private static void BitShuffleKernel(Index1D idx, ArrayView<byte> src, ArrayView<byte> dst, int elementSize, int block, int length, int offset)
    {
        int o = offset + idx.X;
        if (o >= length) return;
        int baseByte = BitBlockBase(o, elementSize, block, length);
        int n = BitBlockElements(o, baseByte, elementSize, block, length);
        int v = src[o];
        if (n > 0)
        {
            // Row r = 8 * j + b of n / 8 bytes; bit k of its byte q = bit b of byte j of element 8 * q + k.
            int rowBytes = n >> 3;
            int local = o - baseByte;
            int r = local / rowBytes;
            int q = local - r * rowBytes;
            int j = r >> 3;
            int b = r & 7;
            int at = baseByte + (q << 3) * elementSize + j;
            v = 0;
            for (int k = 0; k < 8; k++)
                v |= ((src[at + k * elementSize] >> b) & 1) << k;
        }
        dst[idx.X] = (byte)v;
    }

    private static void BitUnshuffleKernel(Index1D idx, ArrayView<byte> src, ArrayView<byte> dst, int elementSize, int block, int length, int offset)
    {
        int o = offset + idx.X;
        if (o >= length) return;
        int baseByte = BitBlockBase(o, elementSize, block, length);
        int n = BitBlockElements(o, baseByte, elementSize, block, length);
        int v = src[o];
        if (n > 0)
        {
            // Byte j of element e of the block: bit b comes from row 8 * j + b, byte e / 8, bit e % 8.
            int rowBytes = n >> 3;
            int local = o - baseByte;
            int e = local / elementSize;
            int j = local - e * elementSize;
            int at = baseByte + (j << 3) * rowBytes + (e >> 3);
            int k = e & 7;
            v = 0;
            for (int b = 0; b < 8; b++)
                v |= ((src[at + b * rowBytes] >> k) & 1) << b;
        }
        dst[idx.X] = (byte)v;
    }

    public void Dispose()
    {
        _stageIn.Dispose();
        _stageOut.Dispose();
    }
}
