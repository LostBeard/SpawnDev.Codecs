// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// BCn texture block encoder on the accelerator: RGBA8 to BC1, BC3, BC4 and BC5 blocks, the blocks etcpak writes (its
// portable DXTC compressors, ProcessDxtc.cpp, the paths the etcpak 0.9.15 Windows wheel runs: its BC4 blocks match this
// arithmetic 256 / 256, the SSE4.1 variant's 246 / 256; BSD 3-clause, see NOTICE.md):
//   colour (BC1, BC3's colour): a block whose pixels agree in their 5:6:5 bits is solid (c0 = 0, c1 = the colour, every
//     index 1). Otherwise the per-channel min / max, the summed RGB range picks a reciprocal (EtcpakDxtcTables.DivTable),
//     every pixel's summed distance from the min times it, >> 16, is its position 0..3 between min and max; the endpoints
//     are min / max moved in by range / 16 per channel (c0 = max, c1 = min, 5:6:5 by truncation) and positions map to
//     indices {1, 3, 2, 0}. Alpha takes no part (BC1 keeps no alpha; BC3 codes it apart).
//   one channel (BC3 alpha, BC4 red, BC5 red and green): solid -> (v, 0, indices 0); else a0 = max, a1 = min, position
//     ((v - min) * ((8 << 13) / (1 + max - min))) >> 13 mapped to indices {1, 7, 6, 5, 4, 3, 2, 0}.
// Pixels outside the picture (a size that is not a multiple of 4) repeat the nearest edge pixel. One thread per output
// 32-bit word (it encodes its block, or the half of it the word belongs to, and writes its 4 bytes), so the kernel also
// runs where a thread may store only its own elements (WebGL). Nothing returns to the host.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.SpawnJS.JSObjects;

namespace SpawnDev.Codecs.Images.Texture;

/// <summary>Encodes RGBA8 pictures to BCn texture blocks on an ILGPU accelerator.</summary>
public sealed class BcnEncoderGpu : IDisposable
{
    private const int DivBase = 0, TableInts = DivBase + 766;

    private readonly MemoryBuffer1D<int, Stride1D.Dense> _tables;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, int, int, int> _encode;
    private readonly StageBuffer<byte> _stageIn;
    private readonly StageBuffer<byte> _stageOut;

    /// <summary>Compile the kernel and upload etcpak's division table.</summary>
    public BcnEncoderGpu(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _tables = accelerator.Allocate1D<int>(TableInts);
        var t = new int[TableInts];
        for (int i = 0; i < 766; i++) t[DivBase + i] = EtcpakDxtcTables.DivTable[i];
        _tables.View.CopyFromCPU(t);
        _encode = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, int, int, int>(EncodeKernel);
        _stageIn = new StageBuffer<byte>(accelerator);
        _stageOut = new StageBuffer<byte>(accelerator);
    }

    /// <summary>
    /// Queue the encode of <paramref name="rgba"/> (width x height pixels of 4 bytes, tightly packed) into
    /// <paramref name="blocks"/> (<see cref="BcnDecoderGpu.BlocksSize"/> bytes, row-major 4x4 blocks). BC1, BC3, BC4, BC5.
    /// </summary>
    public void Encode(ArrayView1D<byte, Stride1D.Dense> rgba, int width, int height, BcnFormat format, ArrayView1D<byte, Stride1D.Dense> blocks)
    {
        Validate(format, width, height);
        if (rgba.Length < 4L * width * height) throw new ArgumentException("rgba too short.", nameof(rgba));
        long size = BcnDecoderGpu.BlocksSize(format, width, height);
        if (blocks.Length < size) throw new ArgumentException("blocks too short.", nameof(blocks));
        _encode((int)(size / 4), rgba, blocks, _tables.View, (int)format, width, height);
    }

    /// <summary>Encode a picture held by the host; returns the blocks.</summary>
    public async Task<byte[]> EncodeAsync(ReadOnlyMemory<byte> rgba, int width, int height, BcnFormat format)
    {
        Validate(format, width, height);
        long n = 4L * width * height;
        if (rgba.Length < n) throw new ArgumentException("rgba too short.", nameof(rgba));
        long size = BcnDecoderGpu.BlocksSize(format, width, height);
        _stageIn.Ensure(n);
        _stageOut.Ensure(size);
        DeviceTransfer.Upload(_stageIn.View, rgba.Span.Slice(0, (int)n));
        _encode((int)(size / 4), _stageIn.View, _stageOut.View, _tables.View, (int)format, width, height);
        return await DeviceTransfer.DownloadAsync(_stageOut.View, size);
    }

    /// <summary>Encode a picture held in a browser TypedArray; returns the blocks as a Uint8Array.</summary>
    [System.Runtime.Versioning.SupportedOSPlatform("browser")]
    public async Task<Uint8Array> EncodeAsync(TypedArray rgba, int width, int height, BcnFormat format)
    {
        Validate(format, width, height);
        if (rgba.ByteLength < 4L * width * height) throw new ArgumentException("rgba too short.", nameof(rgba));
        long size = BcnDecoderGpu.BlocksSize(format, width, height);
        _stageIn.Ensure(rgba.ByteLength);
        _stageOut.Ensure(size);
        DeviceTransfer.Upload(_stageIn.View, rgba);
        _encode((int)(size / 4), _stageIn.View, _stageOut.View, _tables.View, (int)format, width, height);
        return await DeviceTransfer.DownloadUint8ArrayAsync(_stageOut.View, size);
    }

    private static void Validate(BcnFormat format, int width, int height)
    {
        if (format is not (BcnFormat.Bc1 or BcnFormat.Bc3 or BcnFormat.Bc4 or BcnFormat.Bc5))
            throw new ArgumentOutOfRangeException(nameof(format), "BC1, BC3, BC4 and BC5 (BC7 encoding is not written yet).");
        if (width <= 0) throw new ArgumentOutOfRangeException(nameof(width));
        if (height <= 0) throw new ArgumentOutOfRangeException(nameof(height));
        if ((long)width * height > int.MaxValue / 4) throw new ArgumentOutOfRangeException(nameof(width), "picture too large");
    }

    // ------------------------------------------------------------------
    // Kernel
    // ------------------------------------------------------------------

    private static void EncodeKernel(Index1D word, ArrayView<byte> rgba, ArrayView<byte> blocks, ArrayView<int> t, int format, int width, int height)
    {
        int wordsPerBlock = format == (int)BcnFormat.Bc1 || format == (int)BcnFormat.Bc4 ? 2 : 4;
        int block = word / wordsPerBlock;
        int wi = word - block * wordsPerBlock;
        int bw = (width + 3) >> 2;
        int x0 = (block % bw) * 4, y0 = (block / bw) * 4;
        uint v;
        if (format == (int)BcnFormat.Bc1) v = ColourWord(rgba, t, x0, y0, width, height, wi);
        else if (format == (int)BcnFormat.Bc3)
            v = wi < 2 ? ChannelWord(rgba, t, x0, y0, width, height, 3, wi) : ColourWord(rgba, t, x0, y0, width, height, wi - 2);
        else if (format == (int)BcnFormat.Bc4) v = ChannelWord(rgba, t, x0, y0, width, height, 0, wi);
        else v = ChannelWord(rgba, t, x0, y0, width, height, wi < 2 ? 0 : 1, wi & 1);
        int o = word * 4;
        blocks[o] = (byte)v;
        blocks[o + 1] = (byte)(v >> 8);
        blocks[o + 2] = (byte)(v >> 16);
        blocks[o + 3] = (byte)(v >> 24);
    }

    /// <summary>Byte <paramref name="c"/> of block pixel <paramref name="p"/> (row-major 4x4), the nearest edge pixel outside.</summary>
    private static int Px(ArrayView<byte> rgba, int x0, int y0, int width, int height, int p, int c)
    {
        int x = x0 + (p & 3), y = y0 + (p >> 2);
        if (x >= width) x = width - 1;
        if (y >= height) y = height - 1;
        return rgba[(y * width + x) * 4 + c];
    }

    private static int To565(int r, int g, int b) => ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3);

    /// <summary>Word <paramref name="wi"/> (0: c0 | c1 &lt;&lt; 16, 1: the indices) of the block's colour half.</summary>
    private static uint ColourWord(ArrayView<byte> rgba, ArrayView<int> t, int x0, int y0, int width, int height, int wi)
    {
        int r0 = Px(rgba, x0, y0, width, height, 0, 0), g0 = Px(rgba, x0, y0, width, height, 0, 1), b0 = Px(rgba, x0, y0, width, height, 0, 2);
        int minR = r0, minG = g0, minB = b0, maxR = r0, maxG = g0, maxB = b0;
        bool solid = true;
        for (int p = 1; p < 16; p++)
        {
            int r = Px(rgba, x0, y0, width, height, p, 0), g = Px(rgba, x0, y0, width, height, p, 1), b = Px(rgba, x0, y0, width, height, p, 2);
            solid &= ((r & 0xF8) == (r0 & 0xF8)) & ((g & 0xFC) == (g0 & 0xFC)) & ((b & 0xF8) == (b0 & 0xF8));
            minR = r < minR ? r : minR; maxR = r > maxR ? r : maxR;
            minG = g < minG ? g : minG; maxG = g > maxG ? g : maxG;
            minB = b < minB ? b : minB; maxB = b > maxB ? b : maxB;
        }
        if (solid) return wi == 0 ? (uint)To565(r0, g0, b0) << 16 : 0x55555555u;
        int dr = maxR - minR, dg = maxG - minG, db = maxB - minB;
        if (wi == 0)
        {
            int c0 = To565(maxR - (dr >> 4), maxG - (dg >> 4), maxB - (db >> 4));
            int c1 = To565(minR + (dr >> 4), minG + (dg >> 4), minB + (db >> 4));
            return (uint)c0 | ((uint)c1 << 16);
        }
        int range = t[DivBase + dr + dg + db];
        uint idx = 0;
        for (int p = 0; p < 16; p++)
        {
            int s = Px(rgba, x0, y0, width, height, p, 0) - minR + Px(rgba, x0, y0, width, height, p, 1) - minG
                + Px(rgba, x0, y0, width, height, p, 2) - minB;
            int q = (s * range) >> 16;
            // Position 0 (nearest min) .. 3 (nearest max) to index: min is c1 (1), max is c0 (0).
            int i = q == 0 ? 1 : (q == 1 ? 3 : (q == 2 ? 2 : 0));
            idx |= (uint)i << (2 * p);
        }
        return idx;
    }

    /// <summary>Word <paramref name="wi"/> of the one-channel block of channel <paramref name="c"/> (0 red, 1 green, 3 alpha).</summary>
    private static uint ChannelWord(ArrayView<byte> rgba, ArrayView<int> t, int x0, int y0, int width, int height, int c, int wi)
    {
        int v0 = Px(rgba, x0, y0, width, height, 0, c);
        int min = v0, max = v0;
        bool solid = true;
        for (int p = 1; p < 16; p++)
        {
            int v = Px(rgba, x0, y0, width, height, p, c);
            solid &= v == v0;
            min = v < min ? v : min;
            max = v > max ? v : max;
        }
        if (solid) return wi == 0 ? (uint)v0 : 0u;
        int range = (8 << 13) / (1 + max - min);
        // 16 3-bit indices from bit 16 of the 64-bit block: pixels 0-4 and the low bit of 5 in word 0, the rest in word 1.
        ulong data = 0;
        for (int p = 0; p < 16; p++)
        {
            int q = ((Px(rgba, x0, y0, width, height, p, c) - min) * range) >> 13;
            int i = q == 0 ? 1 : (q == 7 ? 0 : 8 - q);
            data |= (ulong)i << (3 * p);
        }
        return wi == 0 ? (uint)max | ((uint)min << 8) | ((uint)data << 16) : (uint)(data >> 16);
    }

    /// <summary>Release the table and staging buffers.</summary>
    public void Dispose()
    {
        _tables.Dispose();
        _stageIn.Dispose();
        _stageOut.Dispose();
    }
}
