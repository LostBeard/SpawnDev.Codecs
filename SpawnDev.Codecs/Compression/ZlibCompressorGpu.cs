// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// zlib (RFC 1950) compressor on the accelerator. The input is cut into independent blocks; one GPU thread
// compresses one block (DeflateGpu.CompressBlock) into a fixed-size slot, then two small kernels lay the
// slots out back to back (a byte copy: every non-final block ends byte-aligned) and write the 2-byte zlib
// header and the Adler-32 trailer. Neither the input nor the output visits the host; the host reads back
// one int (the compressed size).
//
// API shape: the same three overloads as the codecs (ArrayView1D / managed / browser TypedArray).

using System.Runtime.Versioning;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.SpawnJS.JSObjects;

namespace SpawnDev.Codecs.Compression;

/// <summary>Launches the block-parallel DEFLATE compressor kernels.</summary>
internal sealed class DeflateKernels : IDisposable
{
    // prm: [0] input length, [1] block size, [2] level, [3] slot capacity, [4] block count, [5] scratch ints per thread
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int> _block;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<byte>, ArrayView<int>, ArrayView<int>> _layout;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>> _copy;

    public DeflateKernels(Accelerator accelerator)
    {
        _block = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int>(BlockKernel);
        _layout = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>, ArrayView<byte>, ArrayView<int>, ArrayView<int>>(LayoutKernel);
        _copy = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>>(CopyKernel);
    }

    /// <summary>Compress blocks [waveBase, waveBase + count) (one thread each).</summary>
    public void RunBlocks(
        ArrayView<byte> data, ArrayView<byte> slots, ArrayView<int> scratch, ArrayView<int> res, ArrayView<int> prm,
        int waveBase, int count)
    {
        _block(count, data, slots, scratch, res, prm, waveBase);
    }

    /// <summary>One thread: per-block offsets, zlib header, combined Adler-32 trailer, total size in layout[blockCount].</summary>
    public void RunLayout(ArrayView<int> res, ArrayView<byte> stream, ArrayView<int> prm, ArrayView<int> layout)
    {
        _layout(1, res, stream, prm, layout);
    }

    /// <summary>Copy every block's bytes from its slot to its place in the stream (one thread per slot byte).</summary>
    public void RunCopy(
        ArrayView<byte> slots, ArrayView<byte> stream, ArrayView<int> res, ArrayView<int> layout, ArrayView<int> prm,
        int slotBytes)
    {
        _copy(slotBytes, slots, stream, res, layout, prm);
    }

    private static void BlockKernel(
        Index1D idx, ArrayView<byte> data, ArrayView<byte> slots, ArrayView<int> scratch,
        ArrayView<int> res, ArrayView<int> prm, int waveBase)
    {
        int block = waveBase + idx.X;
        int blockCount = prm[4];
        if (block >= blockCount) return;
        int total = prm[0];
        int bs = prm[1];
        int start = block * bs;
        int len = total - start;
        if (len > bs) len = bs;
        bool isFinal = block == blockCount - 1;
        int slotCap = prm[3];
        int outLen = DeflateGpu.CompressBlock(
            data, start, len, isFinal, prm[2], slots, block * slotCap, slotCap,
            scratch, idx.X * prm[5], bs);
        uint adler = DeflateGpu.AdlerOfRange(data, start, len);
        res[block * 4 + 0] = outLen;
        res[block * 4 + 1] = (int)(adler & 0xFFFFu);
        res[block * 4 + 2] = (int)(adler >> 16);
        res[block * 4 + 3] = len;
    }

    private static void LayoutKernel(
        Index1D _, ArrayView<int> res, ArrayView<byte> stream, ArrayView<int> prm, ArrayView<int> layout)
    {
        int blockCount = prm[4];
        const uint Base = 65521u;
        uint a = 1;
        uint b = 0;
        int pos = 2;
        int failed = 0;
        for (int i = 0; i < blockCount; i++)
        {
            int outLen = res[i * 4 + 0];
            if (outLen < 0) failed = 1;
            layout[i] = pos;
            if (outLen > 0) pos += outLen;
            // Adler-32 combine: blocks were summed from the initial state a = 1, b = 0.
            uint a2 = (uint)res[i * 4 + 1];
            uint b2 = (uint)res[i * 4 + 2];
            uint len2 = (uint)res[i * 4 + 3] % Base;
            b = (b + b2 + len2 * ((a + Base - 1u) % Base)) % Base;
            a = (a + a2 + Base - 1u) % Base;
        }
        stream[0] = 0x78;
        stream[1] = 0x01;
        uint adler = (b << 16) | a;
        stream[pos + 0] = (byte)(adler >> 24);
        stream[pos + 1] = (byte)(adler >> 16);
        stream[pos + 2] = (byte)(adler >> 8);
        stream[pos + 3] = (byte)adler;
        layout[blockCount] = failed != 0 ? -1 : pos + 4;
    }

    private static void CopyKernel(
        Index1D idx, ArrayView<byte> slots, ArrayView<byte> stream, ArrayView<int> res,
        ArrayView<int> layout, ArrayView<int> prm)
    {
        int slotCap = prm[3];
        int i = idx.X;
        int block = i / slotCap;
        int off = i - block * slotCap;
        if (block >= prm[4]) return;
        if (off < res[block * 4 + 0])
            stream[layout[block] + off] = slots[i];
    }

    public void Dispose() { }
}

/// <summary>
/// Owns the block-compression working memory and enqueues the per-block kernels. Shared by the zlib compressor
/// and the PNG encoder (which frames the same blocks as IDAT chunks instead of one zlib stream).
/// </summary>
internal sealed class DeflateBlockRunner : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly int _wave;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _prm;
    private readonly StageBuffer<byte> _slots;
    private readonly StageBuffer<int> _res;
    private readonly StageBuffer<int> _scratch;

    public DeflateKernels Kernels { get; }
    public int BlockSize { get; }
    public int BlockCount { get; private set; }
    public int SlotCapacity { get; private set; }
    public long SlotBytes { get; private set; }
    public ArrayView1D<byte, Stride1D.Dense> Slots => _slots.View;
    public ArrayView1D<int, Stride1D.Dense> Res => _res.View;
    public ArrayView1D<int, Stride1D.Dense> Prm => _prm.View;

    public DeflateBlockRunner(Accelerator accelerator, int blockSize, int threadsPerWave)
    {
        if (blockSize < 1024 || blockSize > DeflateGpu.MaxBlockSize) throw new ArgumentOutOfRangeException(nameof(blockSize));
        if (threadsPerWave < 1) throw new ArgumentOutOfRangeException(nameof(threadsPerWave));
        _accelerator = accelerator;
        BlockSize = blockSize;
        _wave = threadsPerWave;
        Kernels = new DeflateKernels(accelerator);
        _prm = accelerator.Allocate1D<int>(8);
        _slots = new StageBuffer<byte>(accelerator);
        _res = new StageBuffer<int>(accelerator);
        _scratch = new StageBuffer<int>(accelerator);
    }

    public static int BlocksFor(int length, int blockSize) => Math.Max(1, (length + blockSize - 1) / blockSize);

    /// <summary>Enqueue compression of the first <paramref name="length"/> bytes of <paramref name="data"/>. No synchronization.</summary>
    public void Run(ArrayView<byte> data, int length, int level)
    {
        int blocks = BlocksFor(length, BlockSize);
        int slotCap = DeflateGpu.MaxBlockOutput(BlockSize);
        int scratchPer = DeflateGpu.ScratchInts(BlockSize);
        long slotBytes = (long)blocks * slotCap;
        if (slotBytes > int.MaxValue) throw new ArgumentOutOfRangeException(nameof(length), "input too large for one call");
        BlockCount = blocks;
        SlotCapacity = slotCap;
        SlotBytes = slotBytes;

        _slots.Ensure(slotBytes);
        _res.Ensure((long)blocks * 4);
        int wave = Math.Min(_wave, blocks);
        _scratch.Ensure((long)wave * scratchPer);
        var prm = new[] { length, BlockSize, level, slotCap, blocks, scratchPer, 0, 0 };
        _prm.View.CopyFromCPU(ref prm[0], prm.Length);
        for (int first = 0; first < blocks; first += wave)
            Kernels.RunBlocks(data, _slots.View, _scratch.View, _res.View, _prm.View, first, Math.Min(wave, blocks - first));
    }

    public void Dispose()
    {
        Kernels.Dispose();
        _prm.Dispose();
        _slots.Dispose();
        _res.Dispose();
        _scratch.Dispose();
    }
}

/// <summary>zlib compressor running on an ILGPU accelerator.</summary>
/// <remarks>Not thread-safe: one in-flight call per instance.</remarks>
public sealed class ZlibCompressorGpu : IDisposable
{
    /// <summary>Default input bytes per independent block.</summary>
    public const int DefaultBlockSize = 65536;

    private readonly Accelerator _accelerator;
    private readonly DeflateKernels _kernels;
    private readonly int _blockSize;
    private readonly int _wave;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _prm;
    private readonly StageBuffer<byte> _slots;
    private readonly StageBuffer<int> _res;
    private readonly StageBuffer<int> _layout;
    private readonly StageBuffer<int> _scratch;
    private readonly StageBuffer<byte> _stageIn;
    private readonly StageBuffer<byte> _stageOut;

    /// <summary>Compile the kernels.</summary>
    /// <param name="accelerator">The accelerator.</param>
    /// <param name="blockSize">Input bytes per independent block (1 KiB .. 1 MiB). Larger blocks compress better, smaller ones parallelize more.</param>
    /// <param name="threadsPerWave">Blocks compressed concurrently (scratch memory scales with this).</param>
    public ZlibCompressorGpu(Accelerator accelerator, int blockSize = DefaultBlockSize, int threadsPerWave = 32)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        if (blockSize < 1024 || blockSize > DeflateGpu.MaxBlockSize) throw new ArgumentOutOfRangeException(nameof(blockSize));
        if (threadsPerWave < 1) throw new ArgumentOutOfRangeException(nameof(threadsPerWave));
        _accelerator = accelerator;
        _blockSize = blockSize;
        _wave = threadsPerWave;
        _kernels = new DeflateKernels(accelerator);
        _prm = accelerator.Allocate1D<int>(8);
        _slots = new StageBuffer<byte>(accelerator);
        _res = new StageBuffer<int>(accelerator);
        _layout = new StageBuffer<int>(accelerator);
        _scratch = new StageBuffer<int>(accelerator);
        _stageIn = new StageBuffer<byte>(accelerator);
        _stageOut = new StageBuffer<byte>(accelerator);
    }

    private static int BlockCount(int length, int blockSize) => Math.Max(1, (length + blockSize - 1) / blockSize);

    /// <summary>Upper bound of the compressed size of <paramref name="inputLength"/> bytes (the output view must be at least this long).</summary>
    public static long GetMaxCompressedSize(int inputLength, int blockSize = DefaultBlockSize) =>
        6L + (long)BlockCount(inputLength, blockSize) * DeflateGpu.MaxBlockOutput(blockSize);

    /// <summary>
    /// Compress <paramref name="length"/> bytes that are on the accelerator into a zlib stream in
    /// <paramref name="output"/> (at least <see cref="GetMaxCompressedSize"/> bytes). Returns the stream length.
    /// </summary>
    /// <param name="level">0 = stored, 1 = fast, 2 = default, 3 = best (lazy matching).</param>
    public async Task<int> CompressAsync(
        ArrayView1D<byte, Stride1D.Dense> data, int length,
        ArrayView1D<byte, Stride1D.Dense> output, int level = 2)
    {
        if (length < 0 || length > data.Length) throw new ArgumentOutOfRangeException(nameof(length));
        if (level < 0 || level > 3) throw new ArgumentOutOfRangeException(nameof(level));
        long maxOut = GetMaxCompressedSize(length, _blockSize);
        if (output.Length < maxOut)
            throw new ArgumentException($"output must hold {maxOut} bytes.", nameof(output));

        int blocks = BlockCount(length, _blockSize);
        int slotCap = DeflateGpu.MaxBlockOutput(_blockSize);
        int scratchPer = DeflateGpu.ScratchInts(_blockSize);
        long slotBytes = (long)blocks * slotCap;
        if (slotBytes > int.MaxValue) throw new ArgumentOutOfRangeException(nameof(length), "input too large for one call");

        _slots.Ensure(slotBytes);
        _res.Ensure((long)blocks * 4);
        _layout.Ensure(blocks + 1);
        int wave = Math.Min(_wave, blocks);
        _scratch.Ensure((long)wave * scratchPer);
        var prm = new[] { length, _blockSize, level, slotCap, blocks, scratchPer, 0, 0 };
        _prm.View.CopyFromCPU(ref prm[0], prm.Length);

        for (int first = 0; first < blocks; first += wave)
            _kernels.RunBlocks(data, _slots.View, _scratch.View, _res.View, _prm.View, first, Math.Min(wave, blocks - first));
        _kernels.RunLayout(_res.View, output, _prm.View, _layout.View);
        _kernels.RunCopy(_slots.View, output, _res.View, _layout.View, _prm.View, (int)slotBytes);
        await _accelerator.SynchronizeAsync();

        int total = (await _layout.View.SubView(blocks, 1).CopyToHostAsync())[0];
        if (total < 0)
            throw new InvalidOperationException("zlib compression overflowed a block slot (internal error).");
        return total;
    }

    /// <summary>Compress managed bytes; returns the zlib stream as a managed array.</summary>
    public async Task<byte[]> CompressAsync(ReadOnlyMemory<byte> data, int level = 2)
    {
        _stageIn.Ensure(Math.Max(1, data.Length));
        if (data.Length > 0) DeviceTransfer.Upload(_stageIn.View, data.Span);
        _stageOut.Ensure(GetMaxCompressedSize(data.Length, _blockSize));
        int n = await CompressAsync(_stageIn.View, data.Length, _stageOut.View, level);
        return await DeviceTransfer.DownloadAsync(_stageOut.View, n);
    }

    /// <summary>Compress a JS typed array (browser accelerators only); returns the zlib stream as a JS <see cref="Uint8Array"/>.</summary>
    [SupportedOSPlatform("browser")]
    public async Task<Uint8Array> CompressAsync(TypedArray data, int level = 2)
    {
        ArgumentNullException.ThrowIfNull(data);
        long length = data.ByteLength;
        _stageIn.Ensure(Math.Max(1, length));
        DeviceTransfer.Upload(_stageIn.View, data);
        _stageOut.Ensure(GetMaxCompressedSize(checked((int)length), _blockSize));
        int n = await CompressAsync(_stageIn.View, (int)length, _stageOut.View, level);
        return await DeviceTransfer.DownloadUint8ArrayAsync(_stageOut.View, n);
    }

    /// <summary>Release kernel resources and buffers.</summary>
    public void Dispose()
    {
        _kernels.Dispose();
        _prm.Dispose();
        _slots.Dispose();
        _res.Dispose();
        _layout.Dispose();
        _scratch.Dispose();
        _stageIn.Dispose();
        _stageOut.Dispose();
    }
}
