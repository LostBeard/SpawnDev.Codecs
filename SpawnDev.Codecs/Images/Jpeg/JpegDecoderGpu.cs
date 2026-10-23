// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// JPEG decoder, 100% accelerator-resident. The host is a pure coordinator: it uploads the file (necessary I/O),
// dispatches the JpegKernels chain, and reads back the metadata block (a few hundred ints: dimensions, sampling
// factors, one small row per scan). The entropy-coded bytes, the DCT coefficients, the sample planes and the RGBA
// pixels never touch the host unless the caller asks for managed output.
//
// API shape (same as PngDecoderGpu): three overloads per entry point, all funnelling into the ArrayView1D one.
//   - ArrayView1D : file already on the accelerator, RGBA pixels written into a caller-owned view.
//   - TypedArray  : browser. JS bytes in via CopyFromJS, JS Uint8Array pixels out; nothing enters the .NET heap.
//   - ReadOnlyMemory<byte> : managed. Desktop and small inputs; pixels come back as byte[].
//
// Supported: baseline, extended sequential and progressive Huffman JPEG (all scan types: spectral selection and
// successive approximation), 8-bit precision, restart intervals (independent restart segments decode as parallel
// GPU threads), every integral sampling ratio (4:4:4, 4:2:2, 4:2:0, 4:4:0, 4:1:1, odd and non-MCU-aligned sizes),
// grayscale, YCbCr, RGB (Adobe transform 0 / component ids R,G,B), CMYK and YCCK. 12-bit, arithmetic coding,
// lossless and hierarchical files are rejected with a specific ImageDecodeException.
//
// The output is bit-exact with libjpeg-turbo's defaults as used by Pillow: ISLOW integer IDCT, "fancy" upsampling
// (h2v1 / h1v2 / h2v2 triangle filters, replication for other ratios), libjpeg fixed-point YCbCr -> RGB. Four
// component files are read the way Pillow reads them (inverted CMYK) and converted to RGB with Pillow's formula.
//
// Known limits (documented in the tests): coefficients times quantizer values beyond the 16-bit range, which
// libjpeg-turbo's SIMD IDCT truncates, are computed in 32 bits here; no real-world encoder produces them.

using System.Runtime.Versioning;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.SpawnJS.JSObjects;

namespace SpawnDev.Codecs.Images.Jpeg;

/// <summary>Options of <see cref="JpegDecoderGpu"/>.</summary>
public sealed class JpegDecodeOptions
{
    /// <summary>
    /// Apply the EXIF orientation to the pixels (a GPU kernel). The returned <see cref="ImageInfo"/> then reports the
    /// oriented width / height and <see cref="ImageInfo.Orientation"/> 1. Default false: pixels are returned as stored
    /// and <see cref="ImageInfo.Orientation"/> reports the EXIF value.
    /// </summary>
    public bool ApplyOrientation { get; init; }
}

/// <summary>JPEG decoder running on an ILGPU accelerator.</summary>
/// <remarks>
/// Not thread-safe: one in-flight decode per instance (the metadata block, Huffman tables and staging buffers are
/// shared across calls).
/// </remarks>
public sealed class JpegDecoderGpu : IDisposable
{
    /// <summary>Blocks one decode dispatch advances per restart segment: bounds a launch (~5 us per block on one GPU thread, 8192 was a 41 ms launch on CUDA).</summary>
    internal const int BlocksPerDispatch = 1024;

    private readonly Accelerator _accelerator;
    private readonly JpegKernels _kernels;

    private readonly MemoryBuffer1D<int, Stride1D.Dense> _meta;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _huffRaw;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _huff;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _consts;

    private readonly StageBuffer<byte> _stageIn;
    private readonly StageBuffer<byte> _stageOut;
    private readonly StageBuffer<short> _coef;
    private readonly StageBuffer<byte> _planes;
    private readonly StageBuffer<byte> _rgba;
    private readonly StageBuffer<int> _segs;
    private readonly StageBuffer<int> _chunkWork;
    private readonly StageBuffer<int> _state;

    /// <summary>Compile the kernels and upload the constants.</summary>
    public JpegDecoderGpu(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernels = new JpegKernels(accelerator);

        _meta = accelerator.Allocate1D<int>(JpegMeta.Size);
        _huffRaw = accelerator.Allocate1D<int>((long)(JpegMeta.MaxHuffSets + 1) * JpegMeta.RawSetStride);
        _huff = accelerator.Allocate1D<int>(8L * JpegMeta.HuffStride);
        _consts = accelerator.Allocate1D<int>(64);
        _consts.View.CopyFromCPU(JpegTables.ZigZag);

        _stageIn = new StageBuffer<byte>(accelerator);
        _stageOut = new StageBuffer<byte>(accelerator);
        _coef = new StageBuffer<short>(accelerator);
        _planes = new StageBuffer<byte>(accelerator);
        _rgba = new StageBuffer<byte>(accelerator);
        _segs = new StageBuffer<int>(accelerator);
        _chunkWork = new StageBuffer<int>(accelerator);
        _state = new StageBuffer<int>(accelerator);
    }

    // ------------------------------------------------------------------
    // Layout derived from the scanned header facts (pure metadata arithmetic)
    // ------------------------------------------------------------------

    private sealed class CompLayout
    {
        public int Id, H, V;
        public int DownW, DownH, WidthInBlocks, HeightInBlocks, BlocksPerLine, BlockRows;
        public int UpMode, Hr, Vr;
        public long CoefOffset, PlaneOffset;
        public int BlockPrefix;
    }

    private sealed class ScanInfo
    {
        public int Ns;
        public int[] Comps = new int[4];
        public int Ss, Se, Ah, Al, RestartInterval, RstCount, EcsStart, EcsEnd;
        public long TotalMcus;
        public int SegmentCount;
        public int BlocksPerMcu;
    }

    private sealed class Layout
    {
        public int Width, Height, NumComps, SofType, MaxH, MaxV, ColorSpace, Orientation;
        public bool EoiSeen;
        public CompLayout[] Comps = System.Array.Empty<CompLayout>();
        public ScanInfo[] Scans = System.Array.Empty<ScanInfo>();
        public long TotalCoefs, TotalPlaneBytes;
        public int TotalBlocks, MaxSegments;
        public readonly int[] HostSlots = new int[JpegMeta.CompLayout + 4 * JpegMeta.CompLayoutStride - JpegMeta.HostWrittenStart];

        public ImageInfo ToInfo() => new(
            ImageFormat.Jpeg, Width, Height, false, 8, 1, Orientation, SofType == 2);
    }

    private static string DescribeError(JpegError error, int detail) => error switch
    {
        JpegError.NotJpeg => "not a JPEG file (missing SOI marker)",
        JpegError.TruncatedHeaders => "file is truncated (it ends inside the headers)",
        JpegError.BadMarker => detail != 0
            ? $"invalid marker structure (marker 0x{detail:X2} is not allowed here)"
            : "invalid marker structure (a segment does not start with 0xFF)",
        JpegError.NoFrame => "no frame header (SOF) before the first scan",
        JpegError.UnsupportedFrame => detail switch
        {
            0xC3 => "lossless JPEG is not supported",
            0xC5 or 0xC6 or 0xC7 or 0xCD or 0xCE or 0xCF => "hierarchical JPEG is not supported",
            0xC9 or 0xCA or 0xCB or 0xCC => "arithmetic-coded JPEG is not supported",
            0xDC => "JPEG with a DNL marker (height defined after the first scan) is not supported",
            _ => $"unsupported frame type (marker 0x{detail:X2})",
        },
        JpegError.UnsupportedPrecision => detail == 12
            ? "12-bit JPEG is not supported (only 8-bit sample precision)"
            : $"{detail}-bit sample precision is not supported (only 8-bit)",
        JpegError.BadFrameHeader => detail switch
        {
            1 => "invalid frame header: width is zero",
            2 => "invalid frame header: height is zero (DNL is not supported)",
            3 => "invalid frame header: more than one frame header",
            4 => "invalid frame header: bad component sampling factors or quantization table selector",
            5 => "invalid frame header: duplicate component id",
            _ when detail >= 10 => $"unsupported number of components: {detail - 10} (need 1, 3 or 4)",
            _ => "invalid frame header",
        },
        JpegError.BadQuantTable => "invalid quantization table segment (DQT)",
        JpegError.BadHuffmanTable => "invalid Huffman table (DHT) segment or impossible code lengths",
        JpegError.BadScanHeader => detail switch
        {
            1 => "invalid scan header: component selector not in the frame",
            2 => "invalid scan header: component listed twice",
            3 => "invalid scan header: Huffman table selector out of range",
            4 => "invalid scan header: more than 10 blocks in an MCU",
            5 => "invalid scan header: spectral selection / successive approximation parameters are invalid",
            _ => "invalid scan header (SOS)",
        },
        JpegError.NoImageData => "no scan (SOS) in the file",
        JpegError.EntropyTruncated => "entropy-coded data is truncated (the file ends before the image is complete)",
        JpegError.BadHuffmanCode => "corrupt entropy-coded data (a bit pattern is not a Huffman code of the table)",
        JpegError.BadCoefficientRun => "corrupt entropy-coded data (a coefficient run passes the end of the block)",
        JpegError.BadRestart => "restart markers are missing, extra or out of sequence",
        JpegError.MissingHuffmanTable => $"a scan uses Huffman table slot {detail} that was never defined",
        JpegError.MissingQuantTable => $"a component uses quantization table {detail} that was never defined",
        JpegError.FractionalSampling => "unsupported component sampling (fractional ratio between components)",
        JpegError.TooLarge => "image is too large to decode",
        JpegError.TooManyScans => $"more than {JpegMeta.MaxScans} scans or {JpegMeta.MaxHuffSets} Huffman table sets",
        JpegError.MissingEoi => "progressive file ends without an EOI marker (later scans may be missing)",
        JpegError.IncompleteScans => $"component {detail} is not completely covered by the scans",
        _ => error.ToString(),
    };

    private static ImageDecodeException Fail(JpegError error, int detail = 0) =>
        new(ImageFormat.Jpeg, DescribeError(error, detail));

    private static int CeilDiv(int a, int b) => (a + b - 1) / b;

    private static Layout ComputeLayout(int[] m, PixelFormat format, JpegDecodeOptions? options)
    {
        var err = (JpegError)m[JpegMeta.Error];
        if (err != JpegError.None) throw Fail(err, m[JpegMeta.ErrorDetail]);

        var l = new Layout
        {
            Width = m[JpegMeta.Width],
            Height = m[JpegMeta.Height],
            NumComps = m[JpegMeta.NumComps],
            SofType = m[JpegMeta.SofType],
            MaxH = m[JpegMeta.MaxH],
            MaxV = m[JpegMeta.MaxV],
            Orientation = m[JpegMeta.Orientation],
            EoiSeen = m[JpegMeta.EoiSeen] != 0,
        };
        if ((long)l.Width * l.Height * ImageInfo.BytesPerPixel(format) > int.MaxValue) throw Fail(JpegError.TooLarge);

        // Colour space (libjpeg jdapimin.c default_decompress_parms).
        int adobe = m[JpegMeta.AdobeSeen];
        int transform = m[JpegMeta.AdobeTransform];
        int id0 = m[JpegMeta.CompIn + JpegMeta.CiId];
        int id1 = m[JpegMeta.CompIn + JpegMeta.CompInStride + JpegMeta.CiId];
        int id2 = m[JpegMeta.CompIn + 2 * JpegMeta.CompInStride + JpegMeta.CiId];
        if (l.NumComps == 1) l.ColorSpace = JpegMeta.ColorGray;
        else if (l.NumComps == 3)
        {
            if (m[JpegMeta.JfifSeen] != 0) l.ColorSpace = JpegMeta.ColorYcc;
            else if (adobe != 0) l.ColorSpace = transform == 0 ? JpegMeta.ColorRgb : JpegMeta.ColorYcc;
            else if (id0 == 1 && id1 == 2 && id2 == 3) l.ColorSpace = JpegMeta.ColorYcc;
            else if (id0 == 'R' && id1 == 'G' && id2 == 'B') l.ColorSpace = JpegMeta.ColorRgb;
            else l.ColorSpace = JpegMeta.ColorYcc;
        }
        else
        {
            l.ColorSpace = adobe != 0 && transform == 0 ? JpegMeta.ColorCmyk
                         : adobe != 0 ? JpegMeta.ColorYcck : JpegMeta.ColorCmyk;
        }

        int mcusX = CeilDiv(l.Width, 8 * l.MaxH);
        int mcusY = CeilDiv(l.Height, 8 * l.MaxV);
        l.Comps = new CompLayout[l.NumComps];
        long coefOffset = 0, planeOffset = 0;
        int blockPrefix = 0;
        for (int c = 0; c < l.NumComps; c++)
        {
            int b = JpegMeta.CompIn + c * JpegMeta.CompInStride;
            var cl = new CompLayout { Id = m[b + JpegMeta.CiId], H = m[b + JpegMeta.CiH], V = m[b + JpegMeta.CiV] };
            cl.DownW = (int)(((long)l.Width * cl.H + l.MaxH - 1) / l.MaxH);
            cl.DownH = (int)(((long)l.Height * cl.V + l.MaxV - 1) / l.MaxV);
            cl.WidthInBlocks = CeilDiv(cl.DownW, 8);
            cl.HeightInBlocks = CeilDiv(cl.DownH, 8);
            cl.BlocksPerLine = mcusX * cl.H;
            cl.BlockRows = mcusY * cl.V;
            cl.CoefOffset = coefOffset;
            cl.PlaneOffset = planeOffset;
            cl.BlockPrefix = blockPrefix;

            // Upsampling method (libjpeg jdsample.c jinit_upsampler, do_fancy_upsampling = true).
            cl.Hr = l.MaxH / cl.H;
            cl.Vr = l.MaxV / cl.V;
            if (cl.H == l.MaxH && cl.V == l.MaxV) cl.UpMode = JpegMeta.UpFull;
            else if (cl.H * 2 == l.MaxH && cl.V == l.MaxV) cl.UpMode = cl.DownW > 2 ? JpegMeta.UpH2V1 : JpegMeta.UpReplicate;
            else if (cl.H == l.MaxH && cl.V * 2 == l.MaxV) cl.UpMode = JpegMeta.UpH1V2;
            else if (cl.H * 2 == l.MaxH && cl.V * 2 == l.MaxV) cl.UpMode = cl.DownW > 2 ? JpegMeta.UpH2V2 : JpegMeta.UpReplicate;
            else if (l.MaxH % cl.H == 0 && l.MaxV % cl.V == 0) cl.UpMode = JpegMeta.UpReplicate;
            else throw Fail(JpegError.FractionalSampling);

            coefOffset += (long)cl.BlocksPerLine * cl.BlockRows * 64;
            planeOffset += (long)cl.BlocksPerLine * 8 * cl.BlockRows * 8;
            blockPrefix += cl.BlocksPerLine * cl.BlockRows;
            if (coefOffset > int.MaxValue || planeOffset > int.MaxValue) throw Fail(JpegError.TooLarge);
            l.Comps[c] = cl;
        }
        l.TotalCoefs = coefOffset;
        l.TotalPlaneBytes = planeOffset;
        l.TotalBlocks = blockPrefix;

        var hs = l.HostSlots;
        int o = -JpegMeta.HostWrittenStart;
        hs[JpegMeta.OutputFormat + o] = (int)format;
        hs[JpegMeta.ColorSpace + o] = l.ColorSpace;
        hs[JpegMeta.TotalBlocks + o] = l.TotalBlocks;
        bool orient = options?.ApplyOrientation == true && l.Orientation != 1;
        hs[JpegMeta.ApplyOrientation + o] = orient ? l.Orientation : 1;
        for (int c = 0; c < l.NumComps; c++)
        {
            var cl = l.Comps[c];
            int b = JpegMeta.CompLayout + c * JpegMeta.CompLayoutStride + o;
            hs[b + JpegMeta.ClPlaneOffset] = (int)cl.PlaneOffset;
            hs[b + JpegMeta.ClPlaneStride] = cl.BlocksPerLine * 8;
            hs[b + JpegMeta.ClDownW] = cl.DownW;
            hs[b + JpegMeta.ClDownH] = cl.DownH;
            hs[b + JpegMeta.ClUpMode] = cl.UpMode;
            hs[b + JpegMeta.ClHr] = cl.Hr;
            hs[b + JpegMeta.ClVr] = cl.Vr;
            hs[b + JpegMeta.ClBlockPrefix] = cl.BlockPrefix;
            hs[b + JpegMeta.ClBlocksPerLine] = cl.BlocksPerLine;
            hs[b + JpegMeta.ClBlockRows] = cl.BlockRows;
            hs[b + JpegMeta.ClCoefOffset] = (int)cl.CoefOffset;
            hs[b + JpegMeta.ClWidthInBlocks] = cl.WidthInBlocks;
            hs[b + JpegMeta.ClHeightInBlocks] = cl.HeightInBlocks;
            hs[b + JpegMeta.ClH] = cl.H;
            hs[b + JpegMeta.ClV] = cl.V;
        }
        return l;
    }

    /// <summary>Per-scan sizes (MCU count, restart segments) from the scan table; throws on structural problems.</summary>
    private static void ComputeScans(Layout l, int[] scanRows, int scanCount)
    {
        l.Scans = new ScanInfo[scanCount];
        bool[] dcSeen = new bool[l.NumComps];
        bool[] anySeen = new bool[l.NumComps];
        for (int s = 0; s < scanCount; s++)
        {
            int b = s * JpegMeta.ScanStride;
            var si = new ScanInfo
            {
                Ns = scanRows[b + JpegMeta.ScNs],
                Ss = scanRows[b + JpegMeta.ScSs],
                Se = scanRows[b + JpegMeta.ScSe],
                Ah = scanRows[b + JpegMeta.ScAh],
                Al = scanRows[b + JpegMeta.ScAl],
                RestartInterval = scanRows[b + JpegMeta.ScRestartInterval],
                RstCount = scanRows[b + JpegMeta.ScRstCount],
                EcsStart = scanRows[b + JpegMeta.ScEcsStart],
                EcsEnd = scanRows[b + JpegMeta.ScEcsEnd],
            };
            for (int k = 0; k < si.Ns; k++)
            {
                si.Comps[k] = scanRows[b + JpegMeta.ScComp + k];
                anySeen[si.Comps[k]] = true;
                if (l.SofType == 2 && si.Ss == 0 && si.Ah == 0) dcSeen[si.Comps[k]] = true;
            }
            if (si.Ns == 1)
            {
                var cl = l.Comps[si.Comps[0]];
                si.TotalMcus = (long)cl.WidthInBlocks * cl.HeightInBlocks;
                si.BlocksPerMcu = 1;
            }
            else
            {
                si.TotalMcus = (long)CeilDiv(l.Width, 8 * l.MaxH) * CeilDiv(l.Height, 8 * l.MaxV);
                for (int k = 0; k < si.Ns; k++) si.BlocksPerMcu += l.Comps[si.Comps[k]].H * l.Comps[si.Comps[k]].V;
            }
            si.SegmentCount = 1;
            if (si.RestartInterval > 0)
                si.SegmentCount = (int)((si.TotalMcus + si.RestartInterval - 1) / si.RestartInterval);
            if (si.RstCount != si.SegmentCount - 1)
                throw new ImageDecodeException(ImageFormat.Jpeg,
                    $"{DescribeError(JpegError.BadRestart, 0)} (scan {s + 1}: {si.RstCount} restart markers found, " +
                    $"{si.SegmentCount - 1} expected from the restart interval {si.RestartInterval} and {si.TotalMcus} MCUs)");
            if (si.SegmentCount > l.MaxSegments) l.MaxSegments = si.SegmentCount;
            l.Scans[s] = si;
        }
        for (int c = 0; c < l.NumComps; c++)
        {
            if (!anySeen[c]) throw Fail(JpegError.IncompleteScans, l.Comps[c].Id);
            if (l.SofType == 2 && !dcSeen[c]) throw Fail(JpegError.IncompleteScans, l.Comps[c].Id);
        }
    }

    // ------------------------------------------------------------------
    // Pipeline
    // ------------------------------------------------------------------

    /// <summary>Marker scan; returns the header part of the metadata block.</summary>
    private async Task<int[]> ScanAsync(ArrayView1D<byte, Stride1D.Dense> file, int fileLength, bool headerOnly)
    {
        if (fileLength <= 0 || fileLength > file.Length) throw new ArgumentOutOfRangeException(nameof(fileLength));
        // Bounded launches: the per-chunk summary (parallel), then the resumable walk until it reports phase 2.
        var summary = _chunkWork.Ensure(JpegKernels.ChunkWorkInts(fileLength));
        _kernels.EcsSummary(file, fileLength, summary);
        _meta.View.SubView(JpegMeta.ScanPhase, 1).MemSetToZero();
        int launches = 1;
        while (true)
        {
            _kernels.ScanMarkers(file, fileLength, _meta.View, _huffRaw.View, _consts.View, headerOnly, summary, launches);
            await _accelerator.SynchronizeAsync();
            var phase = await _meta.View.SubView(JpegMeta.ScanPhase, 1).CopyToHostAsync();
            if (phase[0] == 2) break;
            launches = 8;
        }
        return await _meta.View.SubView(0, JpegMeta.Scans).CopyToHostAsync();
    }

    /// <summary>
    /// Read the facts about a JPEG that is on the accelerator (header scan + a metadata-sized readback; no pixels are
    /// decoded). Use the result to size the output buffer for
    /// <see cref="DecodeAsync(ArrayView1D{byte, Stride1D.Dense}, int, ArrayView1D{byte, Stride1D.Dense}, PixelFormat, JpegDecodeOptions?)"/>.
    /// </summary>
    public async Task<ImageInfo> ReadInfoAsync(ArrayView1D<byte, Stride1D.Dense> file, int fileLength)
    {
        var meta = await ScanAsync(file, fileLength, headerOnly: true);
        return ComputeLayout(meta, PixelFormat.Rgba8, null).ToInfo();
    }

    /// <summary>
    /// Decode a JPEG that is already on the accelerator into a caller-owned pixel view. Only metadata scalars cross to
    /// the host.
    /// </summary>
    /// <param name="file">The JPEG file bytes (may be longer than <paramref name="fileLength"/>).</param>
    /// <param name="fileLength">Number of valid bytes in <paramref name="file"/>.</param>
    /// <param name="output">Receives the tightly packed pixels; at least <see cref="ImageInfo.GetPixelBufferSize"/> bytes.</param>
    /// <param name="format">Pixel layout to produce.</param>
    /// <param name="options">Decode options (EXIF orientation), or null for the defaults.</param>
    /// <returns>The facts about the image (oriented dimensions when <see cref="JpegDecodeOptions.ApplyOrientation"/> is set).</returns>
    /// <exception cref="ImageDecodeException">The file is corrupt, truncated or uses an unsupported feature.</exception>
    public async Task<ImageInfo> DecodeAsync(
        ArrayView1D<byte, Stride1D.Dense> file, int fileLength,
        ArrayView1D<byte, Stride1D.Dense> output, PixelFormat format = PixelFormat.Rgba8,
        JpegDecodeOptions? options = null)
    {
        var (layout, info) = await PrepareAsync(file, fileLength, format, options);
        long outBytes = info.GetPixelBufferSize(format);
        if (output.Length < outBytes)
            throw new ArgumentException($"output must hold {outBytes} bytes for {info.Width}x{info.Height} {format}.", nameof(output));
        await RunPipelineAsync(file, layout, output, format);
        return info;
    }

    private async Task<(Layout, ImageInfo)> PrepareAsync(
        ArrayView1D<byte, Stride1D.Dense> file, int fileLength, PixelFormat format, JpegDecodeOptions? options)
    {
        var meta = await ScanAsync(file, fileLength, headerOnly: false);
        var layout = ComputeLayout(meta, format, options);
        int scanCount = meta[JpegMeta.ScanCount];
        var rows = await _meta.View.SubView(JpegMeta.Scans, (long)scanCount * JpegMeta.ScanStride).CopyToHostAsync();
        ComputeScans(layout, rows, scanCount);
        var info = layout.ToInfo();
        int orient = layout.HostSlots[JpegMeta.ApplyOrientation - JpegMeta.HostWrittenStart];
        if (orient != 1)
            info = info with { Width = orient >= 5 ? info.Height : info.Width, Height = orient >= 5 ? info.Width : info.Height, Orientation = 1 };
        return (layout, info);
    }

    private async Task RunPipelineAsync(
        ArrayView1D<byte, Stride1D.Dense> file, Layout layout, ArrayView1D<byte, Stride1D.Dense> output, PixelFormat format)
    {
        _coef.Ensure(layout.TotalCoefs);
        _planes.Ensure(layout.TotalPlaneBytes);
        _segs.Ensure(2L * layout.MaxSegments);
        _state.Ensure((long)JpegMeta.SegStateStride * layout.MaxSegments);
        int orient = layout.HostSlots[JpegMeta.ApplyOrientation - JpegMeta.HostWrittenStart];
        long pixels = (long)layout.Width * layout.Height;
        int bpp = ImageInfo.BytesPerPixel(format);
        var colorTarget = output;
        if (orient != 1) colorTarget = _rgba.Ensure(pixels * bpp);

        // Host-derived layout into the shared metadata block.
        _meta.View.SubView(JpegMeta.HostWrittenStart, layout.HostSlots.Length)
             .CopyFromCPU(ref layout.HostSlots[0], layout.HostSlots.Length);

        _kernels.ClearCoef(_coef.View, (int)layout.TotalCoefs);
        var pacer = new LaunchPacer(_accelerator);
        for (int s = 0; s < layout.Scans.Length; s++)
        {
            var si = layout.Scans[s];
            _kernels.BuildHuff(_meta.View, _huffRaw.View, _huff.View, s);
            _kernels.FindRestarts(file, _meta.View, _segs.View, _chunkWork.Ensure(JpegKernels.ChunkWorkInts(si.EcsEnd - si.EcsStart + JpegKernels.EcsChunk)),
                s, si.SegmentCount, si.EcsStart, si.EcsEnd);
            long perSegment = si.RestartInterval > 0 ? Math.Min(si.RestartInterval, si.TotalMcus) : si.TotalMcus;
            int budget = Math.Max(1, BlocksPerDispatch / Math.Max(1, si.BlocksPerMcu));
            int rounds = (int)((perSegment + budget - 1) / budget);
            for (int r = 0; r < Math.Max(1, rounds); r++)
            {
                _kernels.DecodeScan(file, _coef.View, _meta.View, _huff.View, _segs.View, _consts.View, _state.View,
                    s, si.SegmentCount, budget, r);
                await pacer.LaunchedAsync();
            }
        }
        _kernels.Idct(_coef.View, _planes.View, _meta.View, layout.TotalBlocks);
        _kernels.ColorConvert(_planes.View, _meta.View, colorTarget, layout.Width, layout.Width * layout.Height);
        if (orient != 1)
            _kernels.Orient(_rgba.View, output, layout.Width, layout.Height, orient, bpp);
        await _accelerator.SynchronizeAsync();

        var status = await _meta.View.SubView(JpegMeta.Error, 2).CopyToHostAsync();
        if (status[0] != 0) throw Fail((JpegError)status[0], status[1]);
        // Only after the data itself decoded completely: a progressive file without EOI may lack later scans.
        if (layout.SofType == 2 && !layout.EoiSeen) throw Fail(JpegError.MissingEoi);
    }

    /// <summary>
    /// Scan + decode into a staging output sized from the scan, for the managed and TypedArray overloads
    /// (one scan, not two). Returns the info; the pixels are in <see cref="_stageOut"/>.
    /// </summary>
    private async Task<ImageInfo> DecodeToStageAsync(int fileLength, PixelFormat format, JpegDecodeOptions? options)
    {
        var (layout, info) = await PrepareAsync(_stageIn.View, fileLength, format, options);
        _stageOut.Ensure(info.GetPixelBufferSize(format));
        await RunPipelineAsync(_stageIn.View, layout, _stageOut.View, format);
        return info;
    }

    // ------------------------------------------------------------------
    // Managed overloads
    // ------------------------------------------------------------------

    /// <summary>Read the facts about a managed JPEG file (no pixels are decoded).</summary>
    public async Task<ImageInfo> ReadInfoAsync(ReadOnlyMemory<byte> file)
    {
        StageManaged(file);
        return await ReadInfoAsync(_stageIn.View, file.Length);
    }

    /// <summary>
    /// Decode a managed JPEG file. Returns the pixels as a managed array (a terminal CPU sink; in a browser prefer the
    /// <see cref="TypedArray"/> overload).
    /// </summary>
    public async Task<DecodedImage> DecodeAsync(
        ReadOnlyMemory<byte> file, PixelFormat format = PixelFormat.Rgba8, JpegDecodeOptions? options = null)
    {
        StageManaged(file);
        var info = await DecodeToStageAsync(file.Length, format, options);
        var pixels = await DeviceTransfer.DownloadAsync(_stageOut.View, info.GetPixelBufferSize(format));
        return new DecodedImage(info, pixels, format);
    }

    /// <summary>Not async: it holds a span.</summary>
    private void StageManaged(ReadOnlyMemory<byte> file)
    {
        if (file.IsEmpty) throw new ArgumentException("File is empty.", nameof(file));
        _stageIn.Ensure(file.Length);
        DeviceTransfer.Upload(_stageIn.View, file.Span);
    }

    // ------------------------------------------------------------------
    // TypedArray overloads (browser: bulk bytes stay in JS)
    // ------------------------------------------------------------------

    /// <summary>Read the facts about a JPEG held in a JS typed array (browser accelerators only).</summary>
    [SupportedOSPlatform("browser")]
    public async Task<ImageInfo> ReadInfoAsync(TypedArray file)
    {
        ArgumentNullException.ThrowIfNull(file);
        long length = file.ByteLength;
        _stageIn.Ensure(length);
        DeviceTransfer.Upload(_stageIn.View, file);
        return await ReadInfoAsync(_stageIn.View, checked((int)length));
    }

    /// <summary>
    /// Decode a JPEG held in a JS typed array (browser accelerators only). The bytes go to the GPU with
    /// <c>CopyFromJS</c> and the pixels come back as a JS <see cref="Uint8Array"/>; none of it enters the .NET
    /// heap. The caller disposes the result.
    /// </summary>
    /// <exception cref="NotSupportedException">The accelerator is not a browser accelerator.</exception>
    [SupportedOSPlatform("browser")]
    public async Task<DecodedImageJS> DecodeAsync(
        TypedArray file, PixelFormat format = PixelFormat.Rgba8, JpegDecodeOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(file);
        long length = file.ByteLength;
        _stageIn.Ensure(length);
        DeviceTransfer.Upload(_stageIn.View, file);
        var info = await DecodeToStageAsync(checked((int)length), format, options);
        var pixels = await DeviceTransfer.DownloadUint8ArrayAsync(_stageOut.View, info.GetPixelBufferSize(format));
        return new DecodedImageJS(info, pixels, format);
    }

    /// <summary>Release kernel resources and buffers.</summary>
    public void Dispose()
    {
        _kernels.Dispose();
        _meta.Dispose();
        _huffRaw.Dispose();
        _huff.Dispose();
        _consts.Dispose();
        _stageIn.Dispose();
        _stageOut.Dispose();
        _coef.Dispose();
        _planes.Dispose();
        _rgba.Dispose();
        _segs.Dispose();
        _chunkWork.Dispose();
        _state.Dispose();
    }
}
