// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Baseline JPEG encoder, 100% accelerator-resident. The host is a pure coordinator: it builds the tiny header blob
// (< 1 KB: SOI, JFIF, optional EXIF orientation, DQT, SOF0, DHT, DRI, SOS), uploads the per-call constants
// (quantization divisors and Huffman code tables, a few KB), dispatches the JpegEncoderKernels chain, and reads
// back ONE scalar (the file length). Pixels, planes, coefficients and entropy-coded bytes never touch the host
// unless the caller asks for managed output.
//
// API shape (same as PngDecoderGpu / the video codecs): three overloads, all funnelling into the ArrayView1D one.
//   - ArrayView1D : RGBA already on the accelerator, file written into a caller-owned view (at least
//                   GetMaxEncodedSize bytes), returns the length.
//   - TypedArray  : browser. JS bytes in via CopyFromJS, JS Uint8Array out; nothing enters the .NET heap.
//   - ReadOnlyMemory<byte> : managed. Desktop and small inputs; the file comes back as byte[].
//
// The output is byte-identical to libjpeg-turbo (Pillow) for the same pixels, quality, subsampling and restart
// interval: libjpeg quality scaling of the Annex K quantization tables, the standard Annex K Huffman tables,
// jccolor.c RGB -> YCbCr fixed point, jcsample.c downsampling with its alternating bias, jfdctint.c ISLOW forward DCT.
// Alpha is ignored (JPEG has none); the alpha bytes of the RGBA input are never read.

using System.Runtime.Versioning;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.SpawnJS.JSObjects;

namespace SpawnDev.Codecs.Images.Jpeg;

/// <summary>Chroma subsampling of an encoded JPEG.</summary>
public enum JpegSubsampling
{
    /// <summary>YCbCr, no subsampling (each component at full resolution).</summary>
    Yuv444 = 0,
    /// <summary>YCbCr, chroma halved horizontally (2x1 luma blocks per chroma sample).</summary>
    Yuv422 = 1,
    /// <summary>YCbCr, chroma halved in both directions (the libjpeg default).</summary>
    Yuv420 = 2,
    /// <summary>Single component (luminance only).</summary>
    Gray = 3,
}

/// <summary>Options of <see cref="JpegEncoderGpu"/>.</summary>
public sealed class JpegEncodeOptions
{
    /// <summary>libjpeg quality 1..100 (scales the Annex K quantization tables). Default 90.</summary>
    public int Quality { get; init; } = 90;

    /// <summary>Chroma subsampling. Default <see cref="JpegSubsampling.Yuv420"/>.</summary>
    public JpegSubsampling Subsampling { get; init; } = JpegSubsampling.Yuv420;

    /// <summary>Write a progressive file. Not implemented yet: must be false.</summary>
    public bool Progressive { get; init; }

    /// <summary>MCUs between restart markers (0 = no restart markers). Independent segments are encoded by parallel GPU threads.</summary>
    public int RestartInterval { get; init; }

    /// <summary>EXIF orientation 1..8 to embed as an APP1 segment (1 = write no EXIF segment).</summary>
    public int ExifOrientation { get; init; } = 1;
}

/// <summary>Baseline JPEG encoder running on an ILGPU accelerator.</summary>
/// <remarks>Not thread-safe: one in-flight encode per instance.</remarks>
public sealed class JpegEncoderGpu : IDisposable
{
    /// <summary>Blocks one entropy dispatch advances per restart segment (bounds the run time of a single launch).</summary>
    internal const int BlocksPerDispatch = 8192;

    /// <summary>Worst-case bytes of entropy-coded data per 8x8 block (incl. 0xFF stuffing).</summary>
    internal const int MaxBytesPerBlock = 400;

    private const int HeaderReserve = 1024;

    private readonly Accelerator _accelerator;
    private readonly JpegEncoderKernels _kernels;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _meta;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _consts;
    private readonly StageBuffer<byte> _planes;
    private readonly StageBuffer<short> _coefs;
    private readonly StageBuffer<byte> _scratch;
    private readonly StageBuffer<int> _state;
    private readonly StageBuffer<byte> _stageIn;
    private readonly StageBuffer<byte> _stageOut;

    /// <summary>Compile the kernels.</summary>
    public JpegEncoderGpu(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernels = new JpegEncoderKernels(accelerator);
        _meta = accelerator.Allocate1D<int>(JpegEncMeta.Size);
        _consts = accelerator.Allocate1D<int>(JpegEncMeta.KSize);
        _planes = new StageBuffer<byte>(accelerator);
        _coefs = new StageBuffer<short>(accelerator);
        _scratch = new StageBuffer<byte>(accelerator);
        _state = new StageBuffer<int>(accelerator);
        _stageIn = new StageBuffer<byte>(accelerator);
        _stageOut = new StageBuffer<byte>(accelerator);
    }

    // ------------------------------------------------------------------
    // Geometry
    // ------------------------------------------------------------------

    private sealed class Geometry
    {
        public int Width, Height, NumComps, MaxH, MaxV, McusX, McusY, BlocksPerMcu, RestartInterval, SegmentCount;
        public long TotalMcus, TotalSamples, TotalBlocks, TotalCoefs;
        public int[] H = new int[3], V = new int[3], Wib = new int[3], Hib = new int[3], DsRows = new int[3];
        public long[] PlaneOffset = new long[3], CoefOffset = new long[3];
        public int[] BlockPrefix = new int[3];
        public int SlotSize;
        public byte[] Header = System.Array.Empty<byte>();
    }

    private static int CeilDiv(int a, int b) => (a + b - 1) / b;

    private static void Validate(int width, int height, JpegEncodeOptions options)
    {
        if (width <= 0 || height <= 0 || width > 65535 || height > 65535)
            throw new ArgumentOutOfRangeException(nameof(width), $"JPEG dimensions must be 1..65535 (got {width}x{height}).");
        if (options.Quality < 1 || options.Quality > 100)
            throw new ArgumentOutOfRangeException(nameof(options), "Quality must be 1..100.");
        if (options.RestartInterval < 0 || options.RestartInterval > 65535)
            throw new ArgumentOutOfRangeException(nameof(options), "RestartInterval must be 0..65535.");
        if (options.ExifOrientation < 1 || options.ExifOrientation > 8)
            throw new ArgumentOutOfRangeException(nameof(options), "ExifOrientation must be 1..8.");
        if (options.Progressive)
            throw new NotSupportedException("Progressive JPEG encoding is not implemented; set Progressive = false.");
        if (!Enum.IsDefined(options.Subsampling))
            throw new ArgumentOutOfRangeException(nameof(options), "Unknown subsampling.");
        if ((long)width * height * 4 > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(width), "Image is too large (more than 2^31 bytes of RGBA).");
    }

    private static Geometry ComputeGeometry(int width, int height, JpegEncodeOptions options)
    {
        var g = new Geometry { Width = width, Height = height };
        switch (options.Subsampling)
        {
            case JpegSubsampling.Gray: g.NumComps = 1; g.H[0] = 1; g.V[0] = 1; break;
            case JpegSubsampling.Yuv444: g.NumComps = 3; g.H[0] = 1; g.V[0] = 1; break;
            case JpegSubsampling.Yuv422: g.NumComps = 3; g.H[0] = 2; g.V[0] = 1; break;
            default: g.NumComps = 3; g.H[0] = 2; g.V[0] = 2; break;
        }
        for (int c = 1; c < g.NumComps; c++) { g.H[c] = 1; g.V[c] = 1; }
        g.MaxH = g.H[0];
        g.MaxV = g.V[0];
        g.McusX = CeilDiv(width, 8 * g.MaxH);
        g.McusY = CeilDiv(height, 8 * g.MaxV);
        long planeOffset = 0, coefOffset = 0;
        int blockPrefix = 0;
        for (int c = 0; c < g.NumComps; c++)
        {
            int hf = g.MaxH / g.H[c];
            int vf = g.MaxV / g.V[c];
            int dsW = CeilDiv(width, hf);
            int dsH = CeilDiv(height, vf);
            g.Wib[c] = CeilDiv(dsW, 8);
            g.Hib[c] = CeilDiv(dsH, 8);
            g.DsRows[c] = dsH;
            g.PlaneOffset[c] = planeOffset;
            g.CoefOffset[c] = coefOffset;
            g.BlockPrefix[c] = blockPrefix;
            planeOffset += (long)g.Wib[c] * 8 * g.Hib[c] * 8;
            coefOffset += (long)g.Wib[c] * g.Hib[c] * 64;
            blockPrefix += g.Wib[c] * g.Hib[c];
        }
        g.TotalSamples = planeOffset;
        g.TotalCoefs = coefOffset;
        g.TotalBlocks = blockPrefix;
        if (g.NumComps == 1)
        {
            g.BlocksPerMcu = 1;
            g.TotalMcus = (long)g.Wib[0] * g.Hib[0];
        }
        else
        {
            for (int c = 0; c < g.NumComps; c++) g.BlocksPerMcu += g.H[c] * g.V[c];
            g.TotalMcus = (long)g.McusX * g.McusY;
        }
        g.RestartInterval = options.RestartInterval;
        g.SegmentCount = g.RestartInterval > 0 ? (int)((g.TotalMcus + g.RestartInterval - 1) / g.RestartInterval) : 1;
        long perSegment = g.RestartInterval > 0 ? Math.Min(g.RestartInterval, g.TotalMcus) : g.TotalMcus;
        long slot = perSegment * g.BlocksPerMcu * MaxBytesPerBlock + 16;
        if (slot * g.SegmentCount > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(width), "Image is too large for the encoder buffers.");
        g.SlotSize = (int)slot;
        g.Header = BuildHeader(g, options);
        return g;
    }

    /// <summary>
    /// Upper bound of the encoded size in bytes for an image of this size and these options: the output view given
    /// to <see cref="EncodeAsync(ArrayView1D{byte, Stride1D.Dense}, int, int, ArrayView1D{byte, Stride1D.Dense}, JpegEncodeOptions?)"/>
    /// must hold at least this many bytes.
    /// </summary>
    public static int GetMaxEncodedSize(int width, int height, JpegEncodeOptions? options = null)
    {
        options ??= new JpegEncodeOptions();
        Validate(width, height, options);
        var g = ComputeGeometry(width, height, options);
        long total = (long)g.SegmentCount * g.SlotSize + g.Header.Length + 2L * g.SegmentCount + 2;
        if (g.SegmentCount == 1)
            total = g.Header.Length + g.SlotSize + 2;
        if (total > int.MaxValue) throw new ArgumentOutOfRangeException(nameof(width), "Image is too large.");
        return (int)total;
    }

    // ------------------------------------------------------------------
    // Header blob (tiny metadata, built on the host)
    // ------------------------------------------------------------------

    private static int[] ScaledQuant(int[] baseTable, int quality)
    {
        int scale = quality < 50 ? 5000 / quality : 200 - quality * 2;
        var t = new int[64];
        for (int i = 0; i < 64; i++)
        {
            int v = (baseTable[i] * scale + 50) / 100;
            if (v < 1) v = 1;
            if (v > 255) v = 255;
            t[i] = v;
        }
        return t;
    }

    private static void Segment(List<byte> o, int marker, ReadOnlySpan<byte> payload)
    {
        o.Add(0xFF);
        o.Add((byte)marker);
        int len = payload.Length + 2;
        o.Add((byte)(len >> 8));
        o.Add((byte)len);
        foreach (var b in payload) o.Add(b);
    }

    private static byte[] BuildHeader(Geometry g, JpegEncodeOptions options)
    {
        var o = new List<byte>(768) { 0xFF, 0xD8 };
        // JFIF 1.01, no density unit, 1:1 aspect (libjpeg defaults).
        Segment(o, 0xE0, new byte[] { (byte)'J', (byte)'F', (byte)'I', (byte)'F', 0, 1, 1, 0, 0, 1, 0, 1, 0, 0 });
        if (options.ExifOrientation != 1)
        {
            // "Exif\0\0", big-endian TIFF header, IFD0 with one SHORT entry: Orientation.
            Segment(o, 0xE1, new byte[]
            {
                (byte)'E', (byte)'x', (byte)'i', (byte)'f', 0, 0,
                (byte)'M', (byte)'M', 0, 42, 0, 0, 0, 8,
                0, 1,
                0x01, 0x12, 0, 3, 0, 0, 0, 1, 0, (byte)options.ExifOrientation, 0, 0,
                0, 0, 0, 0,
            });
        }
        var q0 = ScaledQuant(JpegTables.StdLuminanceQuant, options.Quality);
        var q1 = ScaledQuant(JpegTables.StdChrominanceQuant, options.Quality);
        var dqt = new byte[65];
        dqt[0] = 0;
        for (int i = 0; i < 64; i++) dqt[1 + i] = (byte)q0[i];
        Segment(o, 0xDB, dqt);
        if (g.NumComps > 1)
        {
            dqt[0] = 1;
            for (int i = 0; i < 64; i++) dqt[1 + i] = (byte)q1[i];
            Segment(o, 0xDB, dqt);
        }
        var sof = new List<byte> { 8, (byte)(g.Height >> 8), (byte)g.Height, (byte)(g.Width >> 8), (byte)g.Width, (byte)g.NumComps };
        for (int c = 0; c < g.NumComps; c++)
        {
            sof.Add((byte)(c + 1));
            sof.Add((byte)((g.H[c] << 4) | g.V[c]));
            sof.Add((byte)(c == 0 ? 0 : 1));
        }
        Segment(o, 0xC0, sof.ToArray());
        AddHuffman(o, 0x00, JpegTables.DcLumBits, JpegTables.DcLumVals);
        AddHuffman(o, 0x10, JpegTables.AcLumBits, JpegTables.AcLumVals);
        if (g.NumComps > 1)
        {
            AddHuffman(o, 0x01, JpegTables.DcChrBits, JpegTables.DcChrVals);
            AddHuffman(o, 0x11, JpegTables.AcChrBits, JpegTables.AcChrVals);
        }
        if (g.RestartInterval > 0)
            Segment(o, 0xDD, new byte[] { (byte)(g.RestartInterval >> 8), (byte)g.RestartInterval });
        var sos = new List<byte> { (byte)g.NumComps };
        for (int c = 0; c < g.NumComps; c++)
        {
            sos.Add((byte)(c + 1));
            sos.Add((byte)(c == 0 ? 0x00 : 0x11));
        }
        sos.Add(0);
        sos.Add(63);
        sos.Add(0);
        Segment(o, 0xDA, sos.ToArray());
        if (o.Count > HeaderReserve) throw new InvalidOperationException("JPEG header blob exceeds the reserved size.");
        return o.ToArray();
    }

    private static void AddHuffman(List<byte> o, int classAndId, int[] bits, int[] vals)
    {
        var p = new byte[1 + 16 + vals.Length];
        p[0] = (byte)classAndId;
        for (int i = 0; i < 16; i++) p[1 + i] = (byte)bits[i];
        for (int i = 0; i < vals.Length; i++) p[17 + i] = (byte)vals[i];
        Segment(o, 0xC4, p);
    }

    /// <summary>Huffman encode table: value = code | length &lt;&lt; 16 per symbol (jpeg_make_c_derived_tbl).</summary>
    private static void FillHuffmanCodes(int[] consts, int table, int[] bits, int[] vals)
    {
        int code = 0, k = 0;
        for (int len = 1; len <= 16; len++)
        {
            for (int i = 0; i < bits[len - 1]; i++)
            {
                consts[JpegEncMeta.KHuff + table * 256 + vals[k]] = code | (len << 16);
                code++;
                k++;
            }
            code <<= 1;
        }
    }

    private static int[] BuildConstants(JpegEncodeOptions options)
    {
        var c = new int[JpegEncMeta.KSize];
        var q0 = ScaledQuant(JpegTables.StdLuminanceQuant, options.Quality);
        var q1 = ScaledQuant(JpegTables.StdChrominanceQuant, options.Quality);
        for (int k = 0; k < 64; k++)
        {
            int n = JpegTables.ZigZag[k];
            c[JpegEncMeta.KQuant0 + n] = q0[k] * 8;
            c[JpegEncMeta.KQuant1 + n] = q1[k] * 8;
            c[JpegEncMeta.KUnZigzag + n] = k;
        }
        FillHuffmanCodes(c, 0, JpegTables.DcLumBits, JpegTables.DcLumVals);
        FillHuffmanCodes(c, 1, JpegTables.DcChrBits, JpegTables.DcChrVals);
        FillHuffmanCodes(c, 2, JpegTables.AcLumBits, JpegTables.AcLumVals);
        FillHuffmanCodes(c, 3, JpegTables.AcChrBits, JpegTables.AcChrVals);
        return c;
    }

    private static int[] BuildMeta(Geometry g)
    {
        var m = new int[JpegEncMeta.Size];
        m[JpegEncMeta.Width] = g.Width;
        m[JpegEncMeta.Height] = g.Height;
        m[JpegEncMeta.NumComps] = g.NumComps;
        m[JpegEncMeta.MaxH] = g.MaxH;
        m[JpegEncMeta.MaxV] = g.MaxV;
        m[JpegEncMeta.RestartInterval] = g.RestartInterval;
        m[JpegEncMeta.McusX] = g.NumComps == 1 ? g.Wib[0] : g.McusX;
        m[JpegEncMeta.McusY] = g.NumComps == 1 ? g.Hib[0] : g.McusY;
        m[JpegEncMeta.TotalMcus] = (int)g.TotalMcus;
        m[JpegEncMeta.SegCount] = g.SegmentCount;
        m[JpegEncMeta.SlotSize] = g.SlotSize;
        m[JpegEncMeta.HeaderLen] = g.Header.Length;
        m[JpegEncMeta.DstBase] = g.SegmentCount == 1 ? g.Header.Length : 0;
        m[JpegEncMeta.BlocksPerMcu] = g.BlocksPerMcu;
        for (int c = 0; c < g.NumComps; c++)
        {
            int b = JpegEncMeta.Comp + c * JpegEncMeta.CompStride;
            m[b + JpegEncMeta.CH] = g.H[c];
            m[b + JpegEncMeta.CV] = g.V[c];
            m[b + JpegEncMeta.CWidthInBlocks] = g.Wib[c];
            m[b + JpegEncMeta.CHeightInBlocks] = g.Hib[c];
            m[b + JpegEncMeta.CCoefOffset] = (int)g.CoefOffset[c];
            m[b + JpegEncMeta.CPlaneOffset] = (int)g.PlaneOffset[c];
            m[b + JpegEncMeta.CPlaneStride] = g.Wib[c] * 8;
            m[b + JpegEncMeta.CBlockPrefix] = g.BlockPrefix[c];
            m[b + JpegEncMeta.CQuant] = c == 0 ? 0 : 1;
            m[b + JpegEncMeta.CDcTbl] = c == 0 ? 0 : 1;
            m[b + JpegEncMeta.CAcTbl] = c == 0 ? 0 : 1;
            m[b + JpegEncMeta.CHf] = g.MaxH / g.H[c];
            m[b + JpegEncMeta.CVf] = g.MaxV / g.V[c];
            m[b + JpegEncMeta.CDsRows] = g.DsRows[c];
        }
        return m;
    }

    // ------------------------------------------------------------------
    // Pipeline
    // ------------------------------------------------------------------

    /// <summary>
    /// Encode RGBA pixels that are already on the accelerator into a caller-owned output view. Only the header blob
    /// and a few KB of constants go up, and one int (the length) comes back.
    /// </summary>
    /// <param name="rgba">Tightly packed RGBA8, rows top to bottom, at least <c>width * height * 4</c> bytes. Alpha is ignored.</param>
    /// <param name="width">Width in pixels (1..65535).</param>
    /// <param name="height">Height in pixels (1..65535).</param>
    /// <param name="output">Receives the JPEG file; at least <see cref="GetMaxEncodedSize"/> bytes.</param>
    /// <param name="options">Encode options, or null for the defaults (quality 90, 4:2:0).</param>
    /// <returns>The length of the JPEG file in bytes.</returns>
    public async Task<int> EncodeAsync(
        ArrayView1D<byte, Stride1D.Dense> rgba, int width, int height,
        ArrayView1D<byte, Stride1D.Dense> output, JpegEncodeOptions? options = null)
    {
        options ??= new JpegEncodeOptions();
        Validate(width, height, options);
        if (rgba.Length < (long)width * height * 4)
            throw new ArgumentException($"rgba must hold {(long)width * height * 4} bytes for {width}x{height}.", nameof(rgba));
        var g = ComputeGeometry(width, height, options);
        long needed = GetMaxEncodedSize(width, height, options);
        if (output.Length < needed)
            throw new ArgumentException($"output must hold {needed} bytes (GetMaxEncodedSize) for {width}x{height}.", nameof(output));

        _planes.Ensure(g.TotalSamples);
        _coefs.Ensure(g.TotalCoefs);
        _state.Ensure((long)g.SegmentCount * JpegEncMeta.StateStride);
        if (g.SegmentCount > 1) _scratch.Ensure((long)g.SegmentCount * g.SlotSize);

        _meta.View.CopyFromCPU(BuildMeta(g));
        _consts.View.CopyFromCPU(BuildConstants(options));
        DeviceTransfer.Upload(output, g.Header);

        _kernels.Prep(rgba, _planes.View, _meta.View, (int)g.TotalSamples);
        _kernels.FdctQuant(_planes.View, _coefs.View, _meta.View, _consts.View, (int)g.TotalBlocks);

        long perSegment = g.RestartInterval > 0 ? Math.Min(g.RestartInterval, g.TotalMcus) : g.TotalMcus;
        int budget = Math.Max(1, BlocksPerDispatch / g.BlocksPerMcu);
        int rounds = (int)((perSegment + budget - 1) / budget);
        var dst = g.SegmentCount == 1 ? output : _scratch.View;
        for (int r = 0; r < Math.Max(1, rounds); r++)
            _kernels.Entropy(_coefs.View, dst, _meta.View, _consts.View, _state.View, g.SegmentCount, budget, r);
        _kernels.Prefix(output, _meta.View, _state.View);
        if (g.SegmentCount > 1)
            _kernels.Assemble(_scratch.View, output, _meta.View, _state.View, (long)g.SegmentCount * g.SlotSize);
        await _accelerator.SynchronizeAsync();

        var status = await _meta.View.SubView(JpegEncMeta.TotalLen, 2).CopyToHostAsync();     // TotalLen, Error
        if (status[1] != 0)
            throw new InvalidOperationException("JPEG encoder output slot overflow (internal error: GetMaxEncodedSize bound violated).");
        return status[0];
    }

    // ------------------------------------------------------------------
    // Managed overload
    // ------------------------------------------------------------------

    /// <summary>
    /// Encode managed RGBA pixels. Returns the JPEG file as a managed array (a terminal CPU sink; in a browser prefer
    /// the <see cref="TypedArray"/> overload).
    /// </summary>
    public async Task<byte[]> EncodeAsync(ReadOnlyMemory<byte> rgba, int width, int height, JpegEncodeOptions? options = null)
    {
        options ??= new JpegEncodeOptions();
        Validate(width, height, options);
        long need = (long)width * height * 4;
        if (rgba.Length < need)
            throw new ArgumentException($"rgba must hold {need} bytes for {width}x{height}.", nameof(rgba));
        _stageIn.Ensure(need);
        DeviceTransfer.Upload(_stageIn.View, rgba.Span.Slice(0, (int)need));
        _stageOut.Ensure(GetMaxEncodedSize(width, height, options));
        int length = await EncodeAsync(_stageIn.View, width, height, _stageOut.View, options);
        return await DeviceTransfer.DownloadAsync(_stageOut.View, length);
    }

    // ------------------------------------------------------------------
    // TypedArray overload (browser: bulk bytes stay in JS)
    // ------------------------------------------------------------------

    /// <summary>
    /// Encode RGBA pixels held in a JS typed array (browser accelerators only). The pixels go to the GPU with
    /// <c>CopyFromJS</c> and the file comes back as a JS <see cref="Uint8Array"/>; none of it enters the .NET heap.
    /// The caller disposes the result.
    /// </summary>
    /// <exception cref="NotSupportedException">The accelerator is not a browser accelerator.</exception>
    [SupportedOSPlatform("browser")]
    public async Task<Uint8Array> EncodeAsync(TypedArray rgba, int width, int height, JpegEncodeOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(rgba);
        options ??= new JpegEncodeOptions();
        Validate(width, height, options);
        long need = (long)width * height * 4;
        if (rgba.ByteLength < need)
            throw new ArgumentException($"rgba must hold {need} bytes for {width}x{height}.", nameof(rgba));
        _stageIn.Ensure(need);
        DeviceTransfer.Upload(_stageIn.View, rgba);
        _stageOut.Ensure(GetMaxEncodedSize(width, height, options));
        int length = await EncodeAsync(_stageIn.View, width, height, _stageOut.View, options);
        return await DeviceTransfer.DownloadUint8ArrayAsync(_stageOut.View, length);
    }

    /// <summary>Release kernel resources and buffers.</summary>
    public void Dispose()
    {
        _kernels.Dispose();
        _meta.Dispose();
        _consts.Dispose();
        _planes.Dispose();
        _coefs.Dispose();
        _scratch.Dispose();
        _state.Dispose();
        _stageIn.Dispose();
        _stageOut.Dispose();
    }
}
