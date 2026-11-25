// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Layered PSD writer (Photoshop file format, version 1, 8 bits per channel, RGB), accelerator-resident: every pixel byte of the file is
// written by kernels (PsdKernels: pack, lay out, emit). The host is a coordinator: it checks the document, uploads managed pixels (a terminal
// upload of caller data), computes the positions of the parts from the sizes the kernels report, and builds the file's METADATA (header, the
// resolution resource, the layer records with their names, rectangles and blend keys), which is the caller's data, not codec data.
//
// File: header | color mode data (0) | image resources (0x03ED resolution, 72 dpi) | layer and mask info | merged image. Every channel is
// RLE (PackBits), bottom layer first, channels -1, 0, 1, 2 per layer; the merged image is 3 channels, or 4 with a merged alpha (then the layer
// count in the file is negative, as Photoshop writes it). Names: a Pascal name (ASCII, others '?') and the Unicode `luni` block. A layer keeps its
// full rectangle: negative coordinates and pixels past the canvas are stored as given. Hidden layers have bit 1 (value 2) set in the flags byte, every
// layer has bit 3 (value 8) set (measured on ComfyUI's export). Not supported (scatter): WebGL.
//
// Two passes over each layer's pixels: the first counts every row's packed length (rows in bounded batches), the host turns the totals into file
// positions, the second writes the rows at their offsets and the row tables. Nothing grows with the picture in one launch.

using System.Runtime.Versioning;
using System.Text;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.SpawnJS.JSObjects;

namespace SpawnDev.Codecs.Images.Psd;

/// <summary>Writes a layered PSD on an ILGPU accelerator.</summary>
/// <remarks>Not thread-safe: one in-flight call per instance.</remarks>
public sealed class PsdEncoderGpu : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly PsdKernels _kernels;
    private readonly StageBuffer<byte> _stageIn, _stageOut;
    private readonly StageBuffer<int> _rowLengths, _rowOffsets, _totals, _info;

    /// <summary>An encoder bound to <paramref name="accelerator"/>. Compiles no kernels (see <see cref="WarmUpAsync"/>).</summary>
    public PsdEncoderGpu(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernels = new PsdKernels(accelerator);
        _stageIn = new StageBuffer<byte>(accelerator);
        _stageOut = new StageBuffer<byte>(accelerator);
        _rowLengths = new StageBuffer<int>(accelerator);
        _rowOffsets = new StageBuffer<int>(accelerator);
        _totals = new StageBuffer<int>(accelerator);
        _info = new StageBuffer<int>(accelerator);
    }

    /// <summary>Compile and load every kernel the encoder uses, so the first encode does not wait for them.</summary>
    public Task WarmUpAsync() => _kernels.EnsureKernelsAsync();

    /// <summary>Threads of one pack launch: rows x width stays near this many pixels.</summary>
    internal const int PixelsPerLaunch = 1 << 21;

    // ------------------------------------------------------------------ document checks and the file layout

    /// <summary>One source of pixels: a layer (4 planes) or the merged picture (3 or 4).</summary>
    private sealed class Source
    {
        public int Width, Height, Planes, Kind;
        public ArrayView<byte> View;
        public int RowStart;          // its first row in the row tables
        public int InfoBase;          // its first int in the plane info
        public int TotalsBase;
        public int Rows => checked(Planes * Height);
    }

    private static int Checked(long v, string what) =>
        v is < 0 or > int.MaxValue ? throw new ArgumentException($"{what} does not fit a PSD file (2 GB limit).") : (int)v;

    private static void Validate(PsdDocument doc)
    {
        ArgumentNullException.ThrowIfNull(doc);
        ArgumentNullException.ThrowIfNull(doc.Composite);
        if (doc.Width is < 1 or > PsdFormat.MaxSide) throw new ArgumentOutOfRangeException(nameof(doc), $"Width {doc.Width} is not in 1..{PsdFormat.MaxSide}.");
        if (doc.Height is < 1 or > PsdFormat.MaxSide) throw new ArgumentOutOfRangeException(nameof(doc), $"Height {doc.Height} is not in 1..{PsdFormat.MaxSide}.");
        if (doc.Layers is null) throw new ArgumentException("Layers is null.", nameof(doc));
        if (doc.Layers.Count > PsdFormat.MaxLayers) throw new ArgumentException($"{doc.Layers.Count} layers; the writer takes at most {PsdFormat.MaxLayers}.", nameof(doc));
        for (int i = 0; i < doc.Layers.Count; i++)
        {
            var l = doc.Layers[i] ?? throw new ArgumentException($"Layer {i} is null.", nameof(doc));
            if (l.Width is < 1 or > PsdFormat.MaxSide || l.Height is < 1 or > PsdFormat.MaxSide)
                throw new ArgumentOutOfRangeException(nameof(doc), $"Layer {i}: {l.Width}x{l.Height} is not in 1..{PsdFormat.MaxSide}.");
            PsdFormat.BlendKey(l.Blend);
            if (l.Name is null) throw new ArgumentException($"Layer {i}: Name is null.", nameof(doc));
            CheckPixels(l.Rgba, l.RgbaView, l.Width, l.Height, $"Layer {i}");
        }
        CheckPixels(doc.Composite.Rgba, doc.Composite.RgbaView, doc.Width, doc.Height, "Composite");
    }

    private static void CheckPixels(ReadOnlyMemory<byte> rgba, ArrayView1D<byte, Stride1D.Dense>? view, int width, int height, string what)
    {
        long need = (long)width * height * 4;
        if (need > int.MaxValue - 4096) throw new ArgumentException($"{what}: {width}x{height} is too large for the PSD writer.");
        if (view is { } v)
        {
            if (v.Length < need) throw new ArgumentException($"{what}: the accelerator view holds {v.Length} bytes; {width}x{height} RGBA8 needs {need}.");
        }
        else if (rgba.Length < need)
        {
            throw new ArgumentException($"{what}: the pixels hold {rgba.Length} bytes; {width}x{height} RGBA8 needs {need}.");
        }
    }

    /// <summary>Worst case of the length of a packed row of n bytes.</summary>
    private static long MaxRow(int n) => (long)n + (n + 127) / 128;

    /// <summary>Everything the file needs besides the packed rows: the metadata bytes and where each part goes.</summary>
    private sealed class Plan
    {
        public byte[] Head = System.Array.Empty<byte>();   // the file's bytes up to the channel data (the records included), uploaded as they are
        public int PadAt, PadLength;                 // zero bytes after the channel data (rounding, and the global layer mask length)
        public int[] Info = System.Array.Empty<int>();      // per plane: compression word, row table, first row byte
        public int Total;                            // the whole file
    }

    private static Plan MakePlan(PsdDocument doc, List<Source> sources, Func<int, int, long> planeLength)
    {
        var plan = new Plan();
        var w = new BigEndian();
        int channels = doc.Composite.HasAlpha ? 4 : 3;
        w.Text("8BPS"); w.U16(1); w.Zero(6); w.U16(channels); w.U32(doc.Height); w.U32(doc.Width); w.U16(8); w.U16(3);
        w.U32(0);   // color mode data
        // image resources: the resolution (72 dpi)
        var res = new BigEndian();
        res.Text("8BIM"); res.U16(0x03ED); res.U16(0); res.U32(16);
        res.U32(72 << 16); res.U16(1); res.U16(1); res.U32(72 << 16); res.U16(1); res.U16(1);
        w.U32(res.Length); w.Bytes(res.ToArray());

        int layers = doc.Layers.Count;
        long[] planeLen = new long[sources.Sum(s => s.Planes)];
        for (int s = 0, k = 0; s < sources.Count; s++)
            for (int pl = 0; pl < sources[s].Planes; pl++, k++) planeLen[k] = planeLength(s, pl);

        // the layer section: records, then the channel data (plane order: -1, 0, 1, 2 = plane 0..3 of a layer source)
        var records = new BigEndian();
        int[] dataLength = new int[planeLen.Length];
        for (int s = 0, k = 0; s < layers; s++)
        {
            var l = doc.Layers[s];
            records.I32(l.Top); records.I32(l.Left); records.I32(checked(l.Top + l.Height)); records.I32(checked(l.Left + l.Width));
            records.U16(4);
            for (int pl = 0; pl < 4; pl++, k++)
            {
                records.U16(pl == 0 ? 0xFFFF : pl - 1);
                int length = Checked(2 + 2L * l.Height + planeLen[k], $"Layer {s}");
                dataLength[k] = length;
                records.U32(length);
            }
            records.Text("8BIM"); records.Text(PsdFormat.BlendKey(l.Blend));
            records.U8(l.Opacity); records.U8(0); records.U8(l.Visible ? 0x08 : 0x0A); records.U8(0);
            var extra = new BigEndian();
            extra.U32(0);   // layer mask data
            extra.U32(0);   // blending ranges
            // the name: one length byte, at most 255 ASCII characters, padded to a multiple of 4 (length byte included)
            string ascii = new string(l.Name.Take(255).Select(c => c is >= ' ' and <= '~' ? c : '?').ToArray());
            extra.U8(ascii.Length); extra.Text(ascii);
            extra.Pad4();
            // the Unicode name: 8BIM luni, length (string + count, padded to 4), UTF-16 code units
            var luni = new BigEndian();
            luni.U32(l.Name.Length); foreach (char c in l.Name) luni.U16(c);
            luni.Pad4();
            extra.Text("8BIM"); extra.Text("luni"); extra.U32(luni.Length); extra.Bytes(luni.ToArray());
            records.U32(extra.Length); records.Bytes(extra.ToArray());
        }

        long layerDataLength = 0;
        for (int s = 0, k = 0; s < layers; s++)
            for (int pl = 0; pl < sources[s].Planes; pl++, k++) layerDataLength += dataLength[k];
        long layerInfo = layers == 0 ? 0 : 2 + records.Length + layerDataLength;
        bool oddInfo = (layerInfo & 1) != 0;
        if (oddInfo) layerInfo++;
        if (layers == 0)
        {
            w.U32(0);   // layer and mask info: none
        }
        else
        {
            w.U32(Checked(4 + layerInfo + 4, "The layers"));
            w.U32(Checked(layerInfo, "The layers"));
            // the count is negative when the composite's first alpha is the merged transparency
            w.I16(doc.Composite.HasAlpha ? -layers : layers);
            w.Bytes(records.ToArray());
        }
        plan.Head = w.ToArray();

        // positions
        var infos = new List<int>();
        long pos = plan.Head.Length;
        for (int s = 0, k = 0; s < layers; s++)
        {
            var src = sources[s];
            src.InfoBase = infos.Count;
            for (int pl = 0; pl < src.Planes; pl++, k++)
            {
                infos.Add(Checked(pos, "The file")); infos.Add(Checked(pos + 2, "The file")); infos.Add(Checked(pos + 2 + 2L * src.Height, "The file"));
                pos += dataLength[k];
            }
        }
        plan.PadAt = Checked(pos, "The file");
        if (layers > 0)
        {
            plan.PadLength = (oddInfo ? 1 : 0) + 4;   // a zero pad byte to the even length, and the 4-byte global layer mask info length
            pos += plan.PadLength;
        }
        // merged image: one compression word, the row tables of all planes, then the rows of all planes
        var comp = sources[layers];
        comp.InfoBase = infos.Count;
        long tablesAt = pos + 2, rowsAt = tablesAt + 2L * comp.Rows;
        int kComp = 0;
        for (int s = 0; s < layers; s++) kComp += sources[s].Planes;
        for (int pl = 0; pl < comp.Planes; pl++)
        {
            infos.Add(pl == 0 ? Checked(pos, "The file") : -1);
            infos.Add(Checked(tablesAt + 2L * pl * comp.Height, "The file"));
            infos.Add(Checked(rowsAt, "The file"));
            rowsAt += planeLen[kComp + pl];
        }
        plan.Info = infos.ToArray();
        plan.Total = Checked(rowsAt, "The file");
        return plan;
    }

    private List<Source> MakeSources(PsdDocument doc)
    {
        var list = new List<Source>();
        int row = 0, totals = 0;
        void Add(int w, int h, int planes, int kind)
        {
            var s = new Source { Width = w, Height = h, Planes = planes, Kind = kind, RowStart = row, TotalsBase = totals };
            row = checked(row + s.Rows); totals += planes;
            list.Add(s);
        }
        foreach (var l in doc.Layers) Add(l.Width, l.Height, 4, PsdKernels.KindLayer);
        Add(doc.Width, doc.Height, doc.Composite.HasAlpha ? 4 : 3, doc.Composite.HasAlpha ? PsdKernels.KindCompositeAlpha : PsdKernels.KindComposite);
        return list;
    }

    /// <summary>Upper bound on the encoded size: every row at its PackBits worst case. Throws <see cref="ArgumentException"/> on an invalid document.</summary>
    public static int GetMaxEncodedSize(PsdDocument doc)
    {
        Validate(doc);
        var sources = new List<Source>();
        foreach (var l in doc.Layers) sources.Add(new Source { Width = l.Width, Height = l.Height, Planes = 4, Kind = PsdKernels.KindLayer });
        sources.Add(new Source { Width = doc.Width, Height = doc.Height, Planes = doc.Composite.HasAlpha ? 4 : 3, Kind = doc.Composite.HasAlpha ? PsdKernels.KindCompositeAlpha : PsdKernels.KindComposite });
        return MakePlan(doc, sources, (s, _) => sources[s].Height * MaxRow(sources[s].Width)).Total;
    }

    // ------------------------------------------------------------------ encode

    /// <summary>
    /// Write <paramref name="document"/> as a PSD into <paramref name="output"/> (layers and composite as accelerator views or managed bytes, see
    /// <see cref="PsdLayer"/>; a managed layer is uploaded). Returns the file length. The output holds at least
    /// <see cref="GetMaxEncodedSize"/> bytes.
    /// </summary>
    /// <exception cref="ArgumentException">The output is smaller than the file, or the document is invalid.</exception>
    public async Task<int> EncodeAsync(PsdDocument document, ArrayView1D<byte, Stride1D.Dense> output)
    {
        Validate(document);
        await _kernels.EnsureKernelsAsync();
        var sources = MakeSources(document);

        // pixels on the accelerator: the caller's views, or the managed bytes uploaded back to back
        long stagedLength = 0;
        long[] stagedAt = new long[sources.Count];
        for (int i = 0; i < sources.Count; i++)
        {
            bool managed = !IsDevice(document, i);
            stagedAt[i] = managed ? stagedLength : -1;
            if (managed) stagedLength += (long)sources[i].Width * sources[i].Height * 4;
        }
        if (stagedLength > int.MaxValue) throw new ArgumentException("The managed layers together are larger than 2 GB; pass accelerator views.");
        if (stagedLength > 0) _stageIn.Ensure(stagedLength);
        for (int i = 0; i < sources.Count; i++)
        {
            int bytes = checked(sources[i].Width * sources[i].Height * 4);
            if (stagedAt[i] >= 0)
            {
                var view = _stageIn.View.SubView(stagedAt[i], bytes);
                DeviceTransfer.Upload(view, Pixels(document, i).Span.Slice(0, bytes));
                sources[i].View = view;
            }
            else
            {
                sources[i].View = DeviceView(document, i).SubView(0, bytes);
            }
        }

        // pass 1: the packed length of every row
        int rows = 0, planes = 0;
        foreach (var s in sources) { rows = Math.Max(rows, checked(s.RowStart + s.Rows)); planes += s.Planes; }
        var rowLengths = _rowLengths.Ensure(rows);
        var rowOffsets = _rowOffsets.Ensure(rows);
        var totalsView = _totals.Ensure(planes);
        var pacer = new LaunchPacer(_accelerator);
        foreach (var s in sources)
        {
            int batch = RowsPerLaunch(s.Width);
            for (int first = 0; first < s.Rows; first += batch)
            {
                _kernels.CountRows(Math.Min(batch, s.Rows - first), s.View, rowLengths, s.Width, s.Height, s.Kind, s.RowStart, first);
                await pacer.LaunchedAsync();
            }
        }
        foreach (var s in sources)
        {
            _kernels.PlaneTotals(s.Planes, rowLengths, totalsView.SubView(s.TotalsBase, s.Planes), s.RowStart, s.Height);
            await pacer.LaunchedAsync();
        }
        await _accelerator.SynchronizeAsync();
        int[] totals = await totalsView.SubView(0, planes).CopyToHostAsync();

        // the file: where everything goes and the metadata bytes
        var planeBase = new int[sources.Count];
        for (int i = 0, k = 0; i < sources.Count; i++) { planeBase[i] = k; k += sources[i].Planes; }
        var plan = MakePlan(document, sources, (s, pl) => totals[planeBase[s] + pl]);
        if (output.Length < plan.Total)
            throw new ArgumentException($"output must hold {plan.Total} bytes (this document); it holds {output.Length}. GetMaxEncodedSize is always enough.", nameof(output));

        DeviceTransfer.Upload(output.SubView(0, plan.Head.Length), plan.Head);
        if (plan.PadLength > 0) DeviceTransfer.Upload(output.SubView(plan.PadAt, plan.PadLength), new byte[plan.PadLength]);
        _info.Ensure(plan.Info.Length);
        _info.View.SubView(0, plan.Info.Length).CopyFromCPU(plan.Info);

        // pass 2: row tables and packed rows
        foreach (var s in sources)
        {
            _kernels.Layout(s.Planes, rowLengths, rowOffsets, _info.View.SubView(s.InfoBase, s.Planes * 3), output, s.RowStart, s.Height);
            await pacer.LaunchedAsync();
        }
        foreach (var s in sources)
        {
            int batch = RowsPerLaunch(s.Width);
            for (int first = 0; first < s.Rows; first += batch)
            {
                _kernels.EmitRows(Math.Min(batch, s.Rows - first), s.View, output, rowOffsets, s.Width, s.Height, s.Kind, s.RowStart, first);
                await pacer.LaunchedAsync();
            }
        }
        await _accelerator.SynchronizeAsync();
        return plan.Total;
    }

    private static int RowsPerLaunch(int width) => Math.Max(1, PixelsPerLaunch / Math.Max(1, width));

    private static bool IsDevice(PsdDocument doc, int i) => i < doc.Layers.Count ? doc.Layers[i].RgbaView != null : doc.Composite.RgbaView != null;
    private static ReadOnlyMemory<byte> Pixels(PsdDocument doc, int i) => i < doc.Layers.Count ? doc.Layers[i].Rgba : doc.Composite.Rgba;
    private static ArrayView1D<byte, Stride1D.Dense> DeviceView(PsdDocument doc, int i) =>
        i < doc.Layers.Count ? doc.Layers[i].RgbaView!.Value : doc.Composite.RgbaView!.Value;

    /// <summary>Write <paramref name="document"/> and return the file as a managed array (a terminal CPU sink).</summary>
    public async Task<byte[]> EncodeAsync(PsdDocument document)
    {
        int max = GetMaxEncodedSize(document);
        _stageOut.Ensure(max);
        int length = await EncodeAsync(document, _stageOut.View);
        return await DeviceTransfer.DownloadAsync(_stageOut.View, length);
    }

    /// <summary>Write <paramref name="document"/> and return the file as a JS <see cref="Uint8Array"/> (browser accelerators only; the caller disposes it).</summary>
    [SupportedOSPlatform("browser")]
    public async Task<Uint8Array> EncodeToUint8ArrayAsync(PsdDocument document)
    {
        int max = GetMaxEncodedSize(document);
        _stageOut.Ensure(max);
        int length = await EncodeAsync(document, _stageOut.View);
        return await DeviceTransfer.DownloadUint8ArrayAsync(_stageOut.View, length);
    }

    /// <summary>Release kernel resources and buffers.</summary>
    public void Dispose()
    {
        _kernels.Dispose();
        _stageIn.Dispose(); _stageOut.Dispose(); _rowLengths.Dispose(); _rowOffsets.Dispose(); _totals.Dispose(); _info.Dispose();
    }

    /// <summary>A growing big-endian byte writer (file metadata).</summary>
    private sealed class BigEndian
    {
        private readonly List<byte> _b = new();
        public int Length => _b.Count;
        public byte[] ToArray() => _b.ToArray();
        public void U8(int v) => _b.Add((byte)v);
        public void U16(int v) { _b.Add((byte)(v >> 8)); _b.Add((byte)v); }
        public void I16(int v) => U16(v);
        public void U32(int v) { U16((int)((uint)v >> 16)); U16(v); }
        public void I32(int v) => U32(v);
        public void Zero(int n) { for (int i = 0; i < n; i++) _b.Add(0); }
        public void Text(string s) { foreach (byte c in Encoding.ASCII.GetBytes(s)) _b.Add(c); }
        public void Bytes(byte[] bytes) => _b.AddRange(bytes);
        public void Pad4() { while (_b.Count % 4 != 0) _b.Add(0); }
    }
}
