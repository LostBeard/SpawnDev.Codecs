// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// BCn texture block decoder on the accelerator: BC1, BC3, BC4, BC5 and BC7 blocks to RGBA8, exactly as the reference
// decoders read them (texture2ddecoder and Pillow, cross-checked by scripts/texture/make_bcn_fixtures.py):
//   BC1  RGB 5:6:5 endpoints; c0 > c1: 4 colours (2 c0 + c1) / 3, (c0 + 2 c1) / 3; else 3 colours (c0 + c1) / 2 and
//        index 3 = transparent black. 5:6:5 expands by bit replication.
//   BC3  BC4-style alpha block + a BC1 colour block read in 4-colour mode whatever c0 vs c1 (the D3D spec).
//   BC4  one channel: a0 > a1: 8 values ((7 - i) a0 + i a1) / 7; else 6 values ((5 - i) a0 + i a1) / 5, then 0, 255.
//        Output (r, 0, 0, 255).
//   BC5  two BC4 blocks (red, green). Output (r, g, 0, 255).
//   BC7  the BPTC modes 0-7 (subsets, partitions from Bc7Tables, p-bits, rotation, index selection); interpolation
//        ((64 - w) e0 + w e1 + 32) >> 6; the reserved mode (no mode bit) decodes to (0, 0, 0, 0).
// One thread per output pixel (gather form: a thread reads its block and decodes only what its pixel needs and writes
// only its own pixel's 4 bytes), so the kernel also runs where a thread may store only its own elements (WebGL). The
// host uploads the BC7 tables once and dispatches; nothing returns to the host.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.SpawnJS.JSObjects;

namespace SpawnDev.Codecs.Images.Texture;

/// <summary>A BCn block format.</summary>
public enum BcnFormat
{
    /// <summary>BC1 (DXT1): RGB + 1-bit alpha, 8 bytes per 4x4 block.</summary>
    Bc1 = 1,
    /// <summary>BC3 (DXT5): RGBA, 16 bytes per block.</summary>
    Bc3 = 3,
    /// <summary>BC4: one channel, 8 bytes per block (decoded to red).</summary>
    Bc4 = 4,
    /// <summary>BC5: two channels, 16 bytes per block (decoded to red and green).</summary>
    Bc5 = 5,
    /// <summary>BC7 (BPTC): RGBA, 16 bytes per block.</summary>
    Bc7 = 7,
}

/// <summary>Decodes BCn texture blocks to RGBA8 on an ILGPU accelerator.</summary>
public sealed class BcnDecoderGpu : IDisposable
{
    // Table layout (ints): BC7 partitions and anchors, then the interpolation weights.
    private const int P2Base = 0, A2Base = P2Base + 1024, P3Base = A2Base + 64, A3bBase = P3Base + 1024, A3cBase = A3bBase + 64;
    private const int W2Base = A3cBase + 64, W3Base = W2Base + 4, W4Base = W3Base + 8, TableInts = W4Base + 16;

    private readonly Accelerator _accelerator;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _tables;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, int, int, int> _decode;
    private readonly StageBuffer<byte> _stageIn;
    private readonly StageBuffer<byte> _stageOut;

    /// <summary>Compile the kernel and upload the BC7 tables.</summary>
    public BcnDecoderGpu(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _tables = accelerator.Allocate1D<int>(TableInts);
        _tables.View.CopyFromCPU(BuildTables());
        _decode = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, int, int, int>(DecodeKernel);
        _stageIn = new StageBuffer<byte>(accelerator);
        _stageOut = new StageBuffer<byte>(accelerator);
    }

    private static int[] BuildTables()
    {
        var t = new int[TableInts];
        for (int i = 0; i < 1024; i++) t[P2Base + i] = Bc7Tables.Partition2[i];
        for (int i = 0; i < 64; i++) t[A2Base + i] = Bc7Tables.Anchor2[i];
        for (int i = 0; i < 1024; i++) t[P3Base + i] = Bc7Tables.Partition3[i];
        for (int i = 0; i < 64; i++) t[A3bBase + i] = Bc7Tables.Anchor3Second[i];
        for (int i = 0; i < 64; i++) t[A3cBase + i] = Bc7Tables.Anchor3Third[i];
        int[] w2 = { 0, 21, 43, 64 };
        int[] w3 = { 0, 9, 18, 27, 37, 46, 55, 64 };
        int[] w4 = { 0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64 };
        w2.CopyTo(t, W2Base);
        w3.CopyTo(t, W3Base);
        w4.CopyTo(t, W4Base);
        return t;
    }

    /// <summary>Bytes of one 4x4 block of <paramref name="format"/>.</summary>
    public static int BlockBytes(BcnFormat format) => format is BcnFormat.Bc1 or BcnFormat.Bc4 ? 8 : 16;

    /// <summary>Bytes of the blocks of a <paramref name="width"/> x <paramref name="height"/> picture (whole 4x4 blocks).</summary>
    public static long BlocksSize(BcnFormat format, int width, int height) =>
        (long)((width + 3) / 4) * ((height + 3) / 4) * BlockBytes(format);

    /// <summary>
    /// Queue the decode of <paramref name="blocks"/> (row-major 4x4 blocks) into <paramref name="rgba"/>: width x height
    /// pixels of 4 bytes (R, G, B, A), tightly packed. Synchronize before reading it.
    /// </summary>
    public void Decode(ArrayView1D<byte, Stride1D.Dense> blocks, BcnFormat format, int width, int height, ArrayView1D<byte, Stride1D.Dense> rgba)
    {
        Validate(format, width, height);
        if (blocks.Length < BlocksSize(format, width, height)) throw new ArgumentException("blocks too short.", nameof(blocks));
        if (rgba.Length < 4L * width * height) throw new ArgumentException("rgba too short.", nameof(rgba));
        _decode(width * height, blocks, rgba, _tables.View, (int)format, width, height);
    }

    /// <summary>Decode blocks held by the host; returns width x height RGBA8 pixels.</summary>
    public async Task<byte[]> DecodeAsync(ReadOnlyMemory<byte> blocks, BcnFormat format, int width, int height)
    {
        Validate(format, width, height);
        long need = BlocksSize(format, width, height);
        if (blocks.Length < need) throw new ArgumentException("blocks too short.", nameof(blocks));
        _stageIn.Ensure(need);
        _stageOut.Ensure(4L * width * height);
        DeviceTransfer.Upload(_stageIn.View, blocks.Span.Slice(0, (int)need));
        _decode(width * height, _stageIn.View, _stageOut.View, _tables.View, (int)format, width, height);
        return await DeviceTransfer.DownloadAsync(_stageOut.View, 4L * width * height);
    }

    /// <summary>Decode blocks held in a browser TypedArray; returns the RGBA8 pixels as a Uint8Array.</summary>
    [System.Runtime.Versioning.SupportedOSPlatform("browser")]
    public async Task<Uint8Array> DecodeAsync(TypedArray blocks, BcnFormat format, int width, int height)
    {
        Validate(format, width, height);
        long need = BlocksSize(format, width, height);
        if (blocks.ByteLength < need) throw new ArgumentException("blocks too short.", nameof(blocks));
        _stageIn.Ensure(blocks.ByteLength);
        _stageOut.Ensure(4L * width * height);
        DeviceTransfer.Upload(_stageIn.View, blocks);
        _decode(width * height, _stageIn.View, _stageOut.View, _tables.View, (int)format, width, height);
        return await DeviceTransfer.DownloadUint8ArrayAsync(_stageOut.View, 4L * width * height);
    }

    private static void Validate(BcnFormat format, int width, int height)
    {
        if (format is not (BcnFormat.Bc1 or BcnFormat.Bc3 or BcnFormat.Bc4 or BcnFormat.Bc5 or BcnFormat.Bc7))
            throw new ArgumentOutOfRangeException(nameof(format));
        if (width <= 0) throw new ArgumentOutOfRangeException(nameof(width));
        if (height <= 0) throw new ArgumentOutOfRangeException(nameof(height));
        if ((long)width * height > int.MaxValue / 4) throw new ArgumentOutOfRangeException(nameof(width), "picture too large");
    }

    // ------------------------------------------------------------------
    // Kernel
    // ------------------------------------------------------------------

    private static void DecodeKernel(Index1D index, ArrayView<byte> blocks, ArrayView<byte> rgba, ArrayView<int> t, int format, int width, int height)
    {
        int x = index % width, y = index / width;
        int bw = (width + 3) >> 2;
        int block = (y >> 2) * bw + (x >> 2);
        int p = ((y & 3) << 2) | (x & 3);
        uint px;
        if (format == (int)BcnFormat.Bc1)
            px = Bc1Colour(Word(blocks, block * 8), Word(blocks, block * 8 + 4), p, false);
        else if (format == (int)BcnFormat.Bc3)
        {
            int at = block * 16;
            uint rgb = Bc1Colour(Word(blocks, at + 8), Word(blocks, at + 12), p, true) & 0x00FFFFFFu;
            px = rgb | ((uint)Bc4Value(Word(blocks, at), Word(blocks, at + 4), p) << 24);
        }
        else if (format == (int)BcnFormat.Bc4)
            px = (uint)Bc4Value(Word(blocks, block * 8), Word(blocks, block * 8 + 4), p) | 0xFF000000u;
        else if (format == (int)BcnFormat.Bc5)
        {
            int at = block * 16;
            px = (uint)Bc4Value(Word(blocks, at), Word(blocks, at + 4), p)
                | ((uint)Bc4Value(Word(blocks, at + 8), Word(blocks, at + 12), p) << 8) | 0xFF000000u;
        }
        else
        {
            int at = block * 16;
            px = Bc7Pixel(Word(blocks, at), Word(blocks, at + 4), Word(blocks, at + 8), Word(blocks, at + 12), p, t);
        }
        int o = index * 4;
        rgba[o] = (byte)px;
        rgba[o + 1] = (byte)(px >> 8);
        rgba[o + 2] = (byte)(px >> 16);
        rgba[o + 3] = (byte)(px >> 24);
    }

    /// <summary>The little-endian 32-bit word at byte <paramref name="at"/>.</summary>
    private static uint Word(ArrayView<byte> d, int at) =>
        d[at] | ((uint)d[at + 1] << 8) | ((uint)d[at + 2] << 16) | ((uint)d[at + 3] << 24);

    /// <summary>BC1 colour of pixel <paramref name="p"/> (packed RGBA); <paramref name="fourColour"/> forces 4-colour mode (BC3).</summary>
    private static uint Bc1Colour(uint ends, uint indices, int p, bool fourColour)
    {
        uint c0 = ends & 0xFFFFu, c1 = ends >> 16;
        int idx = (int)((indices >> (2 * p)) & 3u);
        int r0 = Expand5((int)(c0 >> 11)), g0 = Expand6((int)((c0 >> 5) & 63u)), b0 = Expand5((int)(c0 & 31u));
        int r1 = Expand5((int)(c1 >> 11)), g1 = Expand6((int)((c1 >> 5) & 63u)), b1 = Expand5((int)(c1 & 31u));
        int r, g, b, a = 255;
        bool four = fourColour | (c0 > c1);
        if (idx == 0) { r = r0; g = g0; b = b0; }
        else if (idx == 1) { r = r1; g = g1; b = b1; }
        else if (four)
        {
            if (idx == 2) { r = (2 * r0 + r1) / 3; g = (2 * g0 + g1) / 3; b = (2 * b0 + b1) / 3; }
            else { r = (r0 + 2 * r1) / 3; g = (g0 + 2 * g1) / 3; b = (b0 + 2 * b1) / 3; }
        }
        else if (idx == 2) { r = (r0 + r1) / 2; g = (g0 + g1) / 2; b = (b0 + b1) / 2; }
        else { r = 0; g = 0; b = 0; a = 0; }
        return (uint)r | ((uint)g << 8) | ((uint)b << 16) | ((uint)a << 24);
    }

    private static int Expand5(int v) => (v << 3) | (v >> 2);
    private static int Expand6(int v) => (v << 2) | (v >> 4);

    /// <summary>BC4 value of pixel <paramref name="p"/> from the block's two words (a0, a1, then 16 3-bit indices).</summary>
    private static int Bc4Value(uint lo, uint hi, int p)
    {
        int a0 = (int)(lo & 0xFFu), a1 = (int)((lo >> 8) & 0xFFu);
        // Index bits start at bit 16 of the 64-bit block: 3 bits per pixel, possibly straddling the two words.
        int bit = 16 + 3 * p;
        uint idxBits;
        if (bit + 3 <= 32) idxBits = lo >> bit;
        else if (bit >= 32) idxBits = hi >> (bit - 32);
        else idxBits = (lo >> bit) | (hi << (32 - bit));
        int i = (int)(idxBits & 7u);
        int v;
        if (i == 0) v = a0;
        else if (i == 1) v = a1;
        else if (a0 > a1) v = ((8 - i) * a0 + (i - 1) * a1) / 7;
        else if (i == 6) v = 0;
        else if (i == 7) v = 255;
        else v = ((6 - i) * a0 + (i - 1) * a1) / 5;
        return v;
    }

    /// <summary><paramref name="count"/> (at most 8) bits of the 128-bit block at <paramref name="bit"/>.</summary>
    private static int Bits(uint w0, uint w1, uint w2, uint w3, int bit, int count)
    {
        int wi = bit >> 5, sh = bit & 31;
        uint lo = wi == 0 ? w0 : (wi == 1 ? w1 : (wi == 2 ? w2 : w3));
        uint hi = wi == 0 ? w1 : (wi == 1 ? w2 : (wi == 2 ? w3 : 0u));
        uint v = lo >> sh;
        if (sh + count > 32) v |= hi << (32 - sh);
        return (int)(v & ((1u << count) - 1u));
    }

    /// <summary>An n-bit endpoint value (p-bit included) expanded to 8 bits by bit replication.</summary>
    private static int Unquantize(int v, int n) => n >= 8 ? v : (v << (8 - n)) | (v >> (2 * n - 8));

    private static int Interpolate(int e0, int e1, int w) => ((64 - w) * e0 + w * e1 + 32) >> 6;

    private static int Weight(ArrayView<int> t, int bits, int i) =>
        bits == 2 ? t[W2Base + i] : (bits == 3 ? t[W3Base + i] : t[W4Base + i]);

    /// <summary>The BC7 colour of pixel <paramref name="p"/> (packed RGBA).</summary>
    private static uint Bc7Pixel(uint w0, uint w1, uint w2, uint w3, int p, ArrayView<int> t)
    {
        int mode = 0;
        while ((mode < 8) & (((w0 >> mode) & 1u) == 0)) mode++;
        if (mode == 8) return 0u;   // reserved: every channel 0

        // Mode table: subsets, partition bits, rotation bits, index-selection bit, colour bits, alpha bits,
        // endpoint p-bits (1 = one per endpoint), shared p-bits (1 = one per subset), index bits, second index bits.
        int ns = 1, pb = 0, rb = 0, isb = 0, cb = 0, ab = 0, epb = 0, spb = 0, ib = 0, ib2 = 0;
        if (mode == 0) { ns = 3; pb = 4; cb = 4; epb = 1; ib = 3; }
        else if (mode == 1) { ns = 2; pb = 6; cb = 6; spb = 1; ib = 3; }
        else if (mode == 2) { ns = 3; pb = 6; cb = 5; ib = 2; }
        else if (mode == 3) { ns = 2; pb = 6; cb = 7; epb = 1; ib = 2; }
        else if (mode == 4) { rb = 2; isb = 1; cb = 5; ab = 6; ib = 2; ib2 = 3; }
        else if (mode == 5) { rb = 2; cb = 7; ab = 8; ib = 2; ib2 = 2; }
        else if (mode == 6) { cb = 7; ab = 7; epb = 1; ib = 4; }
        else { ns = 2; pb = 6; cb = 5; ab = 5; epb = 1; ib = 2; }

        int bit = mode + 1;
        int partition = Bits(w0, w1, w2, w3, bit, pb); bit += pb;
        int rotation = Bits(w0, w1, w2, w3, bit, rb); bit += rb;
        int idxSel = Bits(w0, w1, w2, w3, bit, isb); bit += isb;

        // This pixel's subset and the anchors of the partition.
        int subset = 0, anchor1 = 16, anchor2 = 16;
        if (ns == 2)
        {
            subset = t[P2Base + partition * 16 + p];
            anchor1 = t[A2Base + partition];
        }
        else if (ns == 3)
        {
            subset = t[P3Base + partition * 16 + p];
            anchor1 = t[A3bBase + partition];
            anchor2 = t[A3cBase + partition];
        }

        // Endpoints are stored channel by channel (R of every subset's e0, e1, then G, B, then A), then the p-bits.
        int colourStart = bit;
        int alphaStart = colourStart + 3 * 2 * ns * cb;
        int pbitStart = alphaStart + 2 * ns * ab;
        int e0r = Bits(w0, w1, w2, w3, colourStart + (0 * 2 * ns + subset * 2) * cb, cb);
        int e1r = Bits(w0, w1, w2, w3, colourStart + (0 * 2 * ns + subset * 2 + 1) * cb, cb);
        int e0g = Bits(w0, w1, w2, w3, colourStart + (1 * 2 * ns + subset * 2) * cb, cb);
        int e1g = Bits(w0, w1, w2, w3, colourStart + (1 * 2 * ns + subset * 2 + 1) * cb, cb);
        int e0b = Bits(w0, w1, w2, w3, colourStart + (2 * 2 * ns + subset * 2) * cb, cb);
        int e1b = Bits(w0, w1, w2, w3, colourStart + (2 * 2 * ns + subset * 2 + 1) * cb, cb);
        int e0a = 255, e1a = 255;
        if (ab > 0)
        {
            e0a = Bits(w0, w1, w2, w3, alphaStart + (subset * 2) * ab, ab);
            e1a = Bits(w0, w1, w2, w3, alphaStart + (subset * 2 + 1) * ab, ab);
        }
        int cn = cb, an = ab;
        int indexStart = pbitStart;
        if (epb != 0)
        {
            int p0 = Bits(w0, w1, w2, w3, pbitStart + subset * 2, 1);
            int p1 = Bits(w0, w1, w2, w3, pbitStart + subset * 2 + 1, 1);
            e0r = (e0r << 1) | p0; e1r = (e1r << 1) | p1;
            e0g = (e0g << 1) | p0; e1g = (e1g << 1) | p1;
            e0b = (e0b << 1) | p0; e1b = (e1b << 1) | p1;
            if (ab > 0) { e0a = (e0a << 1) | p0; e1a = (e1a << 1) | p1; }
            cn = cb + 1;
            an = ab + 1;
            indexStart = pbitStart + 2 * ns;
        }
        else if (spb != 0)
        {
            int ps = Bits(w0, w1, w2, w3, pbitStart + subset, 1);
            e0r = (e0r << 1) | ps; e1r = (e1r << 1) | ps;
            e0g = (e0g << 1) | ps; e1g = (e1g << 1) | ps;
            e0b = (e0b << 1) | ps; e1b = (e1b << 1) | ps;
            cn = cb + 1;
            indexStart = pbitStart + ns;
        }
        e0r = Unquantize(e0r, cn); e1r = Unquantize(e1r, cn);
        e0g = Unquantize(e0g, cn); e1g = Unquantize(e1g, cn);
        e0b = Unquantize(e0b, cn); e1b = Unquantize(e1b, cn);
        if (ab > 0) { e0a = Unquantize(e0a, an); e1a = Unquantize(e1a, an); }

        // Primary indices: ib bits each, anchors (pixel 0 and the other subsets' anchors) one bit fewer.
        int anchorsBelow = (p > 0 ? 1 : 0) + (anchor1 < p ? 1 : 0) + (anchor2 < p ? 1 : 0);
        bool isAnchor = (p == 0) | (p == anchor1) | (p == anchor2);
        int i1 = Bits(w0, w1, w2, w3, indexStart + p * ib - anchorsBelow, isAnchor ? ib - 1 : ib);
        int colourIdx = i1, colourBits = ib, alphaIdx = i1, alphaBits = ib;
        if (ib2 > 0)
        {
            int secondStart = indexStart + 16 * ib - 1;
            int i2 = Bits(w0, w1, w2, w3, secondStart + p * ib2 - (p > 0 ? 1 : 0), p == 0 ? ib2 - 1 : ib2);
            if (idxSel == 0) { alphaIdx = i2; alphaBits = ib2; }
            else { colourIdx = i2; colourBits = ib2; alphaIdx = i1; alphaBits = ib; }
        }
        int cw = Weight(t, colourBits, colourIdx);
        int r = Interpolate(e0r, e1r, cw), g = Interpolate(e0g, e1g, cw), b = Interpolate(e0b, e1b, cw);
        int a = 255;
        if (ab > 0) a = Interpolate(e0a, e1a, Weight(t, alphaBits, alphaIdx));
        if (rotation == 1) { int s = a; a = r; r = s; }
        else if (rotation == 2) { int s = a; a = g; g = s; }
        else if (rotation == 3) { int s = a; a = b; b = s; }
        return (uint)r | ((uint)g << 8) | ((uint)b << 16) | ((uint)a << 24);
    }

    /// <summary>Release the table and staging buffers.</summary>
    public void Dispose()
    {
        _tables.Dispose();
        _stageIn.Dispose();
        _stageOut.Dispose();
    }
}
