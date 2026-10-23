// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Test/integration kernel that drives FlacCrcGpu through ILGPU.
// Computes CRC-8 + CRC-16 over a single byte range; writes the
// 8-bit + 16-bit results into outputs.
//
// No thread walks the range: one thread per FlacCrcGpu.PieceBytes (4096) piece computes both piece registers,
// then combine launches of FlacCrcGpu.CombinePieces (1024) pieces each fold them (launch count from the length,
// no readback).

using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.Codecs.Audio.Flac;

/// <summary>
/// ILGPU kernels that drive <see cref="FlacCrcGpu"/>.
/// Computes both CRC-8 and CRC-16 over the given byte range.
/// </summary>
public sealed class FlacCrcGpuKernel : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _tables;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, int> _pieceKernel;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<byte>, ArrayView<ushort>, int, int> _combineKernel;

    /// <summary>Compile.</summary>
    public FlacCrcGpuKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _tables = accelerator.Allocate1D<int>(FlacCrcGpu.TableInts);
        _tables.View.CopyFromCPU(FlacCrcGpu.BuildTables());
        _pieceKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, int>(FlacCrcPieceKernel);
        _combineKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>, ArrayView<int>, ArrayView<byte>, ArrayView<ushort>, int, int>(FlacCrcCombineKernel);
    }

    /// <summary>
    /// Compute CRC-8 (out8[0]) and CRC-16 (out16[0]) over data[0..length). Completes when both are written.
    /// </summary>
    public async Task RunAsync(ArrayView<byte> data, ArrayView<byte> out8, ArrayView<ushort> out16, int length)
    {
        if (length < 0) throw new ArgumentOutOfRangeException(nameof(length));
        int pieces = FlacCrcGpu.PiecesOf(length);
        // [0, pieces) CRC-8 registers, [pieces, 2 * pieces) CRC-16 registers.
        using var scratch = _accelerator.Allocate1D<int>(2L * pieces);
        var pacer = new LaunchPacer(_accelerator);
        _pieceKernel(pieces, data, scratch.View, _tables.View, length);
        await pacer.LaunchedAsync();
        int steps = FlacCrcGpu.CombineSteps(pieces);
        for (int step = 0; step < steps; step++)
        {
            _combineKernel(1, scratch.View, _tables.View, out8, out16, length, step);
            await pacer.LaunchedAsync();
        }
        await _accelerator.SynchronizeAsync();
    }

    /// <summary>Both registers of piece <paramref name="piece"/>.</summary>
    private static void FlacCrcPieceKernel(
        Index1D piece, ArrayView<byte> data, ArrayView<int> scratch, ArrayView<int> tables, int length)
    {
        int pieces = FlacCrcGpu.PiecesOf(length);
        FlacCrcGpu.PieceSpan(length, piece.X, out long start, out int count);
        scratch[piece.X] = FlacCrcGpu.Piece8(data, start, count, tables);
        scratch[pieces + piece.X] = FlacCrcGpu.Piece16(data, start, count, tables);
    }

    /// <summary>Combine launch <paramref name="step"/>; the last one writes both results.</summary>
    private static void FlacCrcCombineKernel(
        Index1D _, ArrayView<int> scratch, ArrayView<int> tables, ArrayView<byte> out8, ArrayView<ushort> out16,
        int length, int step)
    {
        int pieces = FlacCrcGpu.PiecesOf(length);
        int done8 = FlacCrcGpu.CombineStep8(scratch, 0, pieces, step, tables);
        int done16 = FlacCrcGpu.CombineStep16(scratch, pieces, pieces, step, tables);
        if (done8 != 0) out8[0] = (byte)scratch[0];
        if (done16 != 0) out16[0] = (ushort)scratch[pieces];
    }

    /// <summary>Release kernel resources.</summary>
    public void Dispose() => _tables.Dispose();
}
