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

    /// <summary>
    /// Write a progressive file (libjpeg's default scan script: spectral selection and successive approximation, with
    /// per-scan optimized Huffman tables). Needs atomics on the accelerator, so it does not run on the WebGL backend.
    /// </summary>
    public bool Progressive { get; init; }

    /// <summary>
    /// Baseline only: compute optimal Huffman tables for the image (a statistics pass, like Pillow's optimize=True)
    /// instead of the standard Annex K tables. Implied by <see cref="Progressive"/>. Needs atomics (no WebGL).
    /// </summary>
    public bool OptimizeHuffman { get; init; }

    /// <summary>MCUs between restart markers (0 = no restart markers). Independent segments are encoded by parallel GPU threads.</summary>
    public int RestartInterval { get; init; }

    /// <summary>EXIF orientation 1..8 to embed as an APP1 segment (1 = write no EXIF segment).</summary>
    public int ExifOrientation { get; init; } = 1;
}

/// <summary>Baseline JPEG encoder running on an ILGPU accelerator.</summary>
/// <remarks>Not thread-safe: one in-flight encode per instance.</remarks>
public sealed class JpegEncoderGpu : IDisposable
{
    /// <summary>Blocks one entropy dispatch advances per restart segment: bounds a launch (8192 was a 37 ms launch on CUDA).</summary>
    internal const int BlocksPerDispatch = 1024;

    /// <summary>Worst-case bytes of entropy-coded data per 8x8 block (incl. 0xFF stuffing).</summary>
    internal const int MaxBytesPerBlock = 400;

    private const int HeaderReserve = 1024;

    /// <summary>Worst-case bytes per 8x8 block over ALL scans of an optimized / progressive file (incl. 0xFF stuffing).</summary>
    internal const int OptimizedBytesPerBlock = 560;

    private readonly Accelerator _accelerator;
    private readonly JpegEncoderKernels _kernels;
    private readonly JpegScanEncoderKernels _scanKernels;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _hist;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _tabs;
    private readonly StageBuffer<byte> _bitbuf;
    private readonly StageBuffer<byte> _sos;
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
        _scanKernels = new JpegScanEncoderKernels(accelerator);
        _hist = accelerator.Allocate1D<int>(4L * JpegEncMeta.HistStride);
        _tabs = accelerator.Allocate1D<int>(4L * JpegEncMeta.TabStride);
        _bitbuf = new StageBuffer<byte>(accelerator);
        _sos = new StageBuffer<byte>(accelerator);
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
        public bool Optimized, Progressive;
        public List<ScanSpec> Scans = new();
    }

    /// <summary>One scan of an optimized / progressive file.</summary>
    private sealed class ScanSpec
    {
        public int Kind;                 // 0 sequential, 1 DC first, 2 DC refine, 3 AC first, 4 AC refine
        public int Ns;
        public int[] Comps = new int[3];
        public int Ss, Se, Ah, Al;
        public long TotalMcus;
        public int McusX, SegmentCount, BlocksPerMcu, Band;
        public long PerSegmentMcus;
        public int Slot;
        public int[] Dht = new int[4];
        public int DhtCount;
        public byte[] Sos = System.Array.Empty<byte>();
        public int SosOffset;
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
        g.Progressive = options.Progressive;
        g.Optimized = options.Progressive || options.OptimizeHuffman;
        if (g.Optimized) BuildScans(g, options);
        g.Header = BuildHeader(g, options);
        return g;
    }

    // ------------------------------------------------------------------
    // Scan script (libjpeg jcparam.c jpeg_simple_progression) and per-scan headers
    // ------------------------------------------------------------------

    private static ScanSpec MakeScan(Geometry g, int[] comps, int ss, int se, int ah, int al, bool progressive)
    {
        var sc = new ScanSpec { Ns = comps.Length, Ss = ss, Se = se, Ah = ah, Al = al };
        for (int i = 0; i < comps.Length; i++) sc.Comps[i] = comps[i];
        if (!progressive) sc.Kind = 0;
        else if (ss == 0) sc.Kind = ah == 0 ? 1 : 2;
        else sc.Kind = ah == 0 ? 3 : 4;
        sc.Band = se - ss + 1;
        if (sc.Ns == 1)
        {
            sc.McusX = g.Wib[comps[0]];
            sc.TotalMcus = (long)g.Wib[comps[0]] * g.Hib[comps[0]];
            sc.BlocksPerMcu = 1;
        }
        else
        {
            sc.McusX = g.McusX;
            sc.TotalMcus = (long)g.McusX * g.McusY;
            sc.BlocksPerMcu = g.BlocksPerMcu;
        }
        sc.SegmentCount = g.RestartInterval > 0 ? (int)((sc.TotalMcus + g.RestartInterval - 1) / g.RestartInterval) : 1;
        sc.PerSegmentMcus = g.RestartInterval > 0 ? Math.Min(g.RestartInterval, sc.TotalMcus) : sc.TotalMcus;
        long slot = sc.PerSegmentMcus * sc.BlocksPerMcu * (sc.Band * 7 + 48) + 16;
        if (slot * sc.SegmentCount > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(g), "Image is too large for the encoder buffers.");
        sc.Slot = (int)slot;
        // DHT segments written before the scan, in libjpeg's order (slots: 0 DC0, 1 DC1, 2 AC0, 3 AC1).
        var dht = new List<int>();
        void AddDht(int t) { if (!dht.Contains(t)) dht.Add(t); }
        if (sc.Kind == 0)
        {
            foreach (int c in comps) { AddDht(c == 0 ? 0 : 1); AddDht(c == 0 ? 2 : 3); }
        }
        else if (sc.Kind == 1)
        {
            foreach (int c in comps) AddDht(c == 0 ? 0 : 1);
        }
        else if (sc.Kind == 3 || sc.Kind == 4)
        {
            AddDht(comps[0] == 0 ? 2 : 3);
        }
        sc.DhtCount = dht.Count;
        for (int i = 0; i < dht.Count; i++) sc.Dht[i] = dht[i];
        return sc;
    }

    private static void BuildScans(Geometry g, JpegEncodeOptions options)
    {
        bool prog = options.Progressive;
        var all = new int[g.NumComps];
        for (int c = 0; c < g.NumComps; c++) all[c] = c;
        if (!prog)
        {
            g.Scans.Add(MakeScan(g, all, 0, 63, 0, 0, false));
        }
        else if (g.NumComps == 3)
        {
            // YCbCr: luminance AC first and refined sooner than chroma.
            g.Scans.Add(MakeScan(g, all, 0, 0, 0, 1, true));
            g.Scans.Add(MakeScan(g, new[] { 0 }, 1, 5, 0, 2, true));
            g.Scans.Add(MakeScan(g, new[] { 2 }, 1, 63, 0, 1, true));
            g.Scans.Add(MakeScan(g, new[] { 1 }, 1, 63, 0, 1, true));
            g.Scans.Add(MakeScan(g, new[] { 0 }, 6, 63, 0, 2, true));
            g.Scans.Add(MakeScan(g, new[] { 0 }, 1, 63, 2, 1, true));
            g.Scans.Add(MakeScan(g, all, 0, 0, 1, 0, true));
            g.Scans.Add(MakeScan(g, new[] { 2 }, 1, 63, 1, 0, true));
            g.Scans.Add(MakeScan(g, new[] { 1 }, 1, 63, 1, 0, true));
            g.Scans.Add(MakeScan(g, new[] { 0 }, 1, 63, 1, 0, true));
        }
        else
        {
            // Gray: libjpeg's all-purpose script for one component.
            g.Scans.Add(MakeScan(g, all, 0, 0, 0, 1, true));
            g.Scans.Add(MakeScan(g, all, 1, 5, 0, 2, true));
            g.Scans.Add(MakeScan(g, all, 6, 63, 0, 2, true));
            g.Scans.Add(MakeScan(g, all, 1, 63, 2, 1, true));
            g.Scans.Add(MakeScan(g, all, 0, 0, 1, 0, true));
            g.Scans.Add(MakeScan(g, all, 1, 63, 1, 0, true));
        }
        // SOS blobs: (first scan: DRI) + SOS, with libjpeg's convention of 0 in unused table fields of progressive scans.
        int offset = 0;
        for (int s = 0; s < g.Scans.Count; s++)
        {
            var sc = g.Scans[s];
            var o = new List<byte>();
            if (s == 0 && g.RestartInterval > 0)
                Segment(o, 0xDD, new byte[] { (byte)(g.RestartInterval >> 8), (byte)g.RestartInterval });
            var p = new List<byte> { (byte)sc.Ns };
            for (int i = 0; i < sc.Ns; i++)
            {
                int c = sc.Comps[i];
                int td = c == 0 ? 0 : 1;
                int ta = c == 0 ? 0 : 1;
                if (prog)
                {
                    if (sc.Ss == 0)
                    {
                        ta = 0;
                        if (sc.Ah != 0) td = 0;
                    }
                    else
                    {
                        td = 0;
                    }
                }
                p.Add((byte)(c + 1));
                p.Add((byte)((td << 4) | ta));
            }
            p.Add((byte)sc.Ss);
            p.Add((byte)sc.Se);
            p.Add((byte)((sc.Ah << 4) | sc.Al));
            Segment(o, 0xDA, p.ToArray());
            sc.Sos = o.ToArray();
            sc.SosOffset = offset;
            offset += sc.Sos.Length;
        }
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
        if (g.Optimized)
        {
            // Headers of every scan (DHT tables up to 4 x 277 bytes + SOS) + a per-block bound that covers all scans together.
            // The MCU grid (dummy blocks included) is what the coder walks, plus one MCU of slack for its capacity guard.
            total = g.Header.Length + g.Scans.Count * 1300L + (g.TotalMcus + 1) * g.BlocksPerMcu * OptimizedBytesPerBlock + 4096;
        }
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
        Segment(o, g.Progressive ? 0xC2 : 0xC0, sof.ToArray());
        if (g.Optimized) return o.ToArray();                  // tables, DRI and SOS are written per scan on the accelerator
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

    private static int[] BuildMeta(Geometry g, long outputLength)
    {
        var m = new int[JpegEncMeta.Size];
        m[JpegEncMeta.OutPos] = g.Header.Length;
        m[JpegEncMeta.OutBound] = (int)Math.Min(outputLength, int.MaxValue);
        m[JpegEncMeta.ScanDataStart] = g.Header.Length;
        for (int s = 0; s < g.Scans.Count; s++)
        {
            var sc = g.Scans[s];
            int b = JpegEncMeta.Scans + s * JpegEncMeta.ScanStride;
            m[b + JpegEncMeta.SKind] = sc.Kind;
            m[b + JpegEncMeta.SNs] = sc.Ns;
            for (int i = 0; i < sc.Ns; i++) m[b + JpegEncMeta.SComp + i] = sc.Comps[i];
            m[b + JpegEncMeta.SSs] = sc.Ss;
            m[b + JpegEncMeta.SSe] = sc.Se;
            m[b + JpegEncMeta.SAh] = sc.Ah;
            m[b + JpegEncMeta.SAl] = sc.Al;
            m[b + JpegEncMeta.STotalMcus] = (int)sc.TotalMcus;
            m[b + JpegEncMeta.SSegCount] = sc.SegmentCount;
            m[b + JpegEncMeta.SSlot] = sc.Slot;
            m[b + JpegEncMeta.SMcusX] = sc.McusX;
            m[b + JpegEncMeta.SSosOff] = sc.SosOffset;
            m[b + JpegEncMeta.SSosLen] = sc.Sos.Length;
            m[b + JpegEncMeta.SDhtCount] = sc.DhtCount;
            for (int i = 0; i < 4; i++) m[b + JpegEncMeta.SDht + i] = sc.Dht[i];
            m[b + JpegEncMeta.SBlocksPerMcu] = sc.BlocksPerMcu;
            m[b + JpegEncMeta.SBand] = sc.Band;
        }
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
        if (g.SegmentCount > 1 && !g.Optimized) _scratch.Ensure((long)g.SegmentCount * g.SlotSize);

        _meta.View.CopyFromCPU(BuildMeta(g, output.Length));
        _consts.View.CopyFromCPU(BuildConstants(options));
        DeviceTransfer.Upload(output, g.Header);

        _kernels.Prep(rgba, _planes.View, _meta.View, (int)g.TotalSamples);
        _kernels.FdctQuant(_planes.View, _coefs.View, _meta.View, _consts.View, (int)g.TotalBlocks);

        if (g.Optimized)
            return await EncodeScansAsync(g, output);

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

    /// <summary>Optimized-table / progressive pipeline: per scan statistics pass, table generation, header, output pass.</summary>
    private async Task<int> EncodeScansAsync(Geometry g, ArrayView1D<byte, Stride1D.Dense> output)
    {
        int maxSeg = 1;
        long maxScratch = 0;
        int sosBytes = 0;
        foreach (var sc in g.Scans)
        {
            maxSeg = Math.Max(maxSeg, sc.SegmentCount);
            if (sc.SegmentCount > 1) maxScratch = Math.Max(maxScratch, (long)sc.SegmentCount * sc.Slot);
            sosBytes += sc.Sos.Length;
        }
        _state.Ensure((long)maxSeg * JpegEncMeta.StateStride);
        _bitbuf.Ensure((long)maxSeg * 1024);
        if (maxScratch > 0) _scratch.Ensure(maxScratch);
        var sosBlob = new byte[sosBytes];
        foreach (var sc in g.Scans) System.Array.Copy(sc.Sos, 0, sosBlob, sc.SosOffset, sc.Sos.Length);
        DeviceTransfer.Upload(_sos.Ensure(sosBytes), sosBlob);

        for (int s = 0; s < g.Scans.Count; s++)
        {
            var sc = g.Scans[s];
            int budget = Math.Max(1, BlocksPerDispatch / Math.Max(1, sc.BlocksPerMcu));
            int rounds = Math.Max(1, (int)((sc.PerSegmentMcus + budget - 1) / budget));
            _scanKernels.ClearHist(_hist.View);
            for (int r = 0; r < rounds; r++)
                _scanKernels.Scan(_coefs.View, output, _meta.View, _tabs.View, _hist.View, _state.View, _bitbuf.View,
                    sc.SegmentCount, s, true, budget, r);
            _scanKernels.GenTables(_hist.View, _tabs.View);
            _scanKernels.Place(output, _sos.View, _meta.View, _tabs.View, s);
            var dst = sc.SegmentCount == 1 ? output : _scratch.View;
            for (int r = 0; r < rounds; r++)
                _scanKernels.Scan(_coefs.View, dst, _meta.View, _tabs.View, _hist.View, _state.View, _bitbuf.View,
                    sc.SegmentCount, s, false, budget, r);
            _scanKernels.Finish(_meta.View, _state.View, s);
            if (sc.SegmentCount > 1)
                _kernels.Assemble(_scratch.View, output, _meta.View, _state.View, (long)sc.SegmentCount * sc.Slot);
        }
        _scanKernels.Eoi(output, _meta.View);
        await _accelerator.SynchronizeAsync();

        var status = await _meta.View.SubView(JpegEncMeta.TotalLen, 2).CopyToHostAsync();     // TotalLen, Error
        if (status[1] != 0)
            throw new InvalidOperationException($"JPEG encoder internal error {status[1]} (output slot overflow or invalid coefficient).");
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
        _scanKernels.Dispose();
        _hist.Dispose();
        _tabs.Dispose();
        _bitbuf.Dispose();
        _sos.Dispose();
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
