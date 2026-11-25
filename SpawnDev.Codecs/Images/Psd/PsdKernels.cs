// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// The PSD writer's kernels (PsdEncoderGpu). Every pixel byte of the file is written here; the host prepares the file's metadata
// (header, resources, layer records) and knows the sizes the kernels return. Two passes over the pixels: the count pass finds each packed row's
// length, the emit pass writes the rows at their file offsets. No launch grows with the picture: rows run in bounded batches
// (PsdEncoderGpu.RowsPerLaunch), and the two one-thread-per-plane kernels (PlaneTotalKernel, LayoutKernel) walk at most 30,000 rows.
//
// PackBits (Adobe / TIFF run length): a control byte n >= 0 is followed by n + 1 literal bytes, n = 257 - k is a byte repeated k times
// (k = 2..128; the writer uses repeats of 3 or more and literals up to 128). Any stream is valid PackBits; the writer's choice is greedy.
//
// Planes of a source, in file order: a layer writes its channels in the order -1 (alpha), 0 (red), 1 (green), 2 (blue) of its interleaved RGBA8
// (Sample: kind layer); the merged picture writes 0, 1, 2 (its colour as given), or 0, 1, 2 matted over white and 3 (alpha) with a merged alpha.

using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.Codecs.Images.Psd;

/// <summary>Loads and launches the PSD writer's kernels.</summary>
internal sealed class PsdKernels : IDisposable
{
    private readonly Accelerator _accelerator;
    private Action<Index1D, ArrayView<byte>, ArrayView<int>, int, int, int, int, int> _countRows = null!;
    private Action<Index1D, ArrayView<int>, ArrayView<int>, int, int> _planeTotals = null!;
    private Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<byte>, int, int> _layout = null!;
    private Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, int, int, int, int, int> _emitRows = null!;
    private Task? _load;

    /// <summary>Bind to <paramref name="accelerator"/>. Compiles nothing: await <see cref="EnsureKernelsAsync"/> first.</summary>
    public PsdKernels(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
    }

    /// <summary>Load the kernels once.</summary>
    public Task EnsureKernelsAsync()
    {
        if (_load == null || _load.IsFaulted || _load.IsCanceled) _load = KernelLoadBatch.RunAsync(_accelerator, LoadAsync);
        return _load;
    }

    private async Task LoadAsync()
    {
        var a = _accelerator;
        _countRows = await a.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<byte>, ArrayView<int>, int, int, int, int, int>(CountRowsKernel);
        _planeTotals = await a.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<int>, ArrayView<int>, int, int>(PlaneTotalsKernel);
        _layout = await a.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<byte>, int, int>(LayoutKernel);
        _emitRows = await a.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, int, int, int, int, int>(EmitRowsKernel);
    }

    /// <summary>Count the packed length of <paramref name="rows"/> (a batch of them) of a source whose planes start at row index <paramref name="rowStart"/>.</summary>
    public void CountRows(int rows, ArrayView<byte> source, ArrayView<int> rowLengths, int width, int height, int kind, int rowStart, int firstRow) =>
        _countRows(rows, source, rowLengths, width, height, kind, rowStart, firstRow);

    /// <summary>The packed length of every plane (one thread per plane), summed over its rows.</summary>
    public void PlaneTotals(int planes, ArrayView<int> rowLengths, ArrayView<int> totals, int rowStart, int height) =>
        _planeTotals(planes, rowLengths, totals, rowStart, height);

    /// <summary>
    /// Writes, per plane, the compression word and row tables into the file and computes every row's offset. <paramref name="info"/> holds, per plane,
    /// the offset of the compression word (-1: the plane has none of its own), of its row table and of its first row's bytes.
    /// </summary>
    public void Layout(int planes, ArrayView<int> rowLengths, ArrayView<int> rowOffsets, ArrayView<int> info, ArrayView<byte> output, int rowStart, int height) =>
        _layout(planes, rowLengths, rowOffsets, info, output, rowStart, height);

    /// <summary>Write the packed bytes of a batch of rows of a source at their offsets.</summary>
    public void EmitRows(int rows, ArrayView<byte> source, ArrayView<byte> output, ArrayView<int> rowOffsets, int width, int height, int kind, int rowStart, int firstRow) =>
        _emitRows(rows, source, output, rowOffsets, width, height, kind, rowStart, firstRow);

    /// <summary>
    /// The value the file stores at pixel i of a row (bytes from <paramref name="rowBase"/>) for <paramref name="plane"/>. A layer keeps straight
    /// alpha and colour in the order alpha, red, green, blue; the merged picture without alpha keeps its colour; with a merged alpha the colour is
    /// matted over white (Photoshop's merged image), the alpha plane stays straight.
    /// </summary>
    private static int Sample(ArrayView<byte> src, int rowBase, int i, int kind, int plane)
    {
        int at = rowBase + i * 4;
        if (kind == KindLayer) return src[at + (plane == 0 ? 3 : plane - 1)];
        if (kind == KindComposite || plane == 3) return src[at + plane];
        int a = src[at + 3];
        return (src[at + plane] * a + 255 * (255 - a) + 127) / 255;
    }

    /// <summary>Plane order of a layer: alpha, red, green, blue (file channel ids -1, 0, 1, 2).</summary>
    public const int KindLayer = 0;

    /// <summary>Plane order of the merged picture without alpha: red, green, blue (colour as given).</summary>
    public const int KindComposite = 1;

    /// <summary>Plane order of the merged picture with a merged alpha: red, green, blue matted over white, then alpha.</summary>
    public const int KindCompositeAlpha = 2;

    // ------------------------------------------------------------------ PackBits

    /// <summary>Length of the PackBits stream of the n samples of a row.</summary>
    private static int PackedLength(ArrayView<byte> src, int rowBase, int n, int kind, int plane)
    {
        int length = 0;
        int i = 0;
        while (i < n)
        {
            int run = RunLength(src, rowBase, n, i, kind, plane);
            if (run >= 3) { length += 2; i += run; }
            else
            {
                int lit = LiteralLength(src, rowBase, n, i, kind, plane);
                length += lit + 1;
                i += lit;
            }
        }
        return length;
    }

    /// <summary>Equal values from position i, at most 128.</summary>
    private static int RunLength(ArrayView<byte> src, int rowBase, int n, int i, int kind, int plane)
    {
        int v = Sample(src, rowBase, i, kind, plane);
        int run = 1;
        while (i + run < n && run < 128 && Sample(src, rowBase, i + run, kind, plane) == v) run++;
        return run;
    }

    /// <summary>Literal values from position i (at least 1, at most 128): up to where 3 equal values start.</summary>
    private static int LiteralLength(ArrayView<byte> src, int rowBase, int n, int i, int kind, int plane)
    {
        int lit = 0;
        while (i + lit < n && lit < 128)
        {
            if (lit > 0 && i + lit + 2 < n)
            {
                int v = Sample(src, rowBase, i + lit, kind, plane);
                if (Sample(src, rowBase, i + lit + 1, kind, plane) == v && Sample(src, rowBase, i + lit + 2, kind, plane) == v) break;
            }
            lit++;
        }
        return lit;
    }

    private static void CountRowsKernel(Index1D i, ArrayView<byte> source, ArrayView<int> rowLengths, int width, int height, int kind, int rowStart, int firstRow)
    {
        int g = firstRow + (int)i;
        int plane = g / height, row = g % height;
        rowLengths[rowStart + g] = PackedLength(source, row * width * 4, width, kind, plane);
    }

    private static void PlaneTotalsKernel(Index1D p, ArrayView<int> rowLengths, ArrayView<int> totals, int rowStart, int height)
    {
        int total = 0;
        for (int r = 0; r < height; r++) total += rowLengths[rowStart + p * height + r];
        totals[p] = total;
    }

    private static void PutBig16(ArrayView<byte> o, int at, int v)
    {
        o[at] = (byte)(v >> 8);
        o[at + 1] = (byte)v;
    }

    private static void LayoutKernel(Index1D p, ArrayView<int> rowLengths, ArrayView<int> rowOffsets, ArrayView<int> info, ArrayView<byte> output, int rowStart, int height)
    {
        int compressionAt = info[p * 3], tableAt = info[p * 3 + 1], dataAt = info[p * 3 + 2];
        if (compressionAt >= 0) PutBig16(output, compressionAt, 1);   // 1: PackBits, with the row byte counts first
        int off = dataAt;
        for (int r = 0; r < height; r++)
        {
            int length = rowLengths[rowStart + p * height + r];
            PutBig16(output, tableAt + 2 * r, length);
            rowOffsets[rowStart + p * height + r] = off;
            off += length;
        }
    }

    private static void EmitRowsKernel(Index1D i, ArrayView<byte> source, ArrayView<byte> output, ArrayView<int> rowOffsets, int width, int height, int kind, int rowStart, int firstRow)
    {
        int g = firstRow + (int)i;
        int plane = g / height, row = g % height;
        int rowBase = row * width * 4;
        int o = rowOffsets[rowStart + g];
        int at = 0;
        int x = 0;
        while (x < width)
        {
            int run = RunLength(source, rowBase, width, x, kind, plane);
            if (run >= 3)
            {
                output[o + at] = (byte)(257 - run);
                output[o + at + 1] = (byte)Sample(source, rowBase, x, kind, plane);
                at += 2;
                x += run;
            }
            else
            {
                int lit = LiteralLength(source, rowBase, width, x, kind, plane);
                output[o + at] = (byte)(lit - 1);
                for (int k = 0; k < lit; k++) output[o + at + 1 + k] = (byte)Sample(source, rowBase, x + k, kind, plane);
                at += lit + 1;
                x += lit;
            }
        }
    }

    /// <summary>Release the kernels.</summary>
    public void Dispose() { }
}
