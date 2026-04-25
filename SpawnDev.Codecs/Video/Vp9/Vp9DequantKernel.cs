// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// ILGPU kernel for VP9 dequantization. Companion to slice 134's CPU
// oracle: each coefficient is multiplied by the DC quantizer (scan
// position 0 of its block) or the AC quantizer (every subsequent
// position), then int16-saturating-cast back into the buffer.
//
// One thread per coefficient. No LocalMemory, no per-block shared
// state - the simplest kernel in the VP9 pipeline. Coverage is still
// 5/6 backends though: ArrayView<short> writes lower to atomic RMW
// on WebGL just like ArrayView<byte> writes do (sub-word stores need
// read-modify-write on the WebGL ES emulation), so the runner's
// pre-emptive `NotSupportedException` for WebGL fires here too. WGSL,
// SPIR-V, OpenCL C, CUDA PTX and the Wasm CPU emulator all handle
// 16-bit stores natively and run the kernel cleanly.
//
// Why scalar dc + ac per dispatch (not arrays):
// VP9 quantizer values are constant across all blocks belonging to
// one (plane, segment) tuple. Real decode groups blocks by that
// tuple before dispatch - the same canonical pattern slice 133's
// iHT 8x8 kernel uses for tx_type. The caller dispatches once per
// (plane, segment) group; this keeps every thread on the same
// branch direction and lets ILGPU constant-fold the quantizer
// loads.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>Per-coefficient ILGPU kernel for VP9 dequantization.</summary>
public sealed class Vp9DequantKernel : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<short>, int, int, int> _kernel;

    /// <summary>Compile the kernel onto <paramref name="accelerator"/>.</summary>
    public Vp9DequantKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<short>, int, int, int>(DequantKernel);
    }

    /// <summary>
    /// Dequantize <paramref name="blockCount"/> blocks of size
    /// <paramref name="coeffsPerBlock"/> in-place. Position 0 of each
    /// block is the DC coefficient (uses <paramref name="quant"/>.Dc);
    /// every other position uses <paramref name="quant"/>.Ac.
    /// </summary>
    public async Task RunAsync(
        Memory<short> coefficients,
        Vp9PlaneQuantizer quant,
        int blockCount,
        int coeffsPerBlock)
    {
        if (blockCount < 0) throw new ArgumentOutOfRangeException(nameof(blockCount));
        if (coeffsPerBlock <= 0) throw new ArgumentOutOfRangeException(nameof(coeffsPerBlock));
        if (blockCount == 0) return;
        long total = (long)blockCount * coeffsPerBlock;
        if (coefficients.Length < total)
            throw new ArgumentException("coefficients too small", nameof(coefficients));

        using var dCoeffs = _accelerator.Allocate1D<short>(total);
        dCoeffs.View.CopyFromCPU(coefficients.Span.ToArray());
        _kernel((Index1D)total, dCoeffs.View, coeffsPerBlock, quant.Dc, quant.Ac);
        await _accelerator.SynchronizeAsync();
        var readBack = await dCoeffs.CopyToHostAsync();
        readBack.AsSpan(0, (int)total).CopyTo(coefficients.Span);
    }

    private static void DequantKernel(
        Index1D idx,
        ArrayView<short> coeffs,
        int coeffsPerBlock,
        int dc,
        int ac)
    {
        int i = idx;
        if (i >= coeffs.Length) return;

        // Position 0 within each block is the DC slot. The integer
        // remainder is fine on every backend - VP9 block sizes
        // (16/64/256/1024) are all powers of 2 so `%` lowers to a
        // mask, but the modulo form keeps the kernel size-agnostic.
        int posInBlock = i % coeffsPerBlock;
        int q = (posInBlock == 0) ? dc : ac;

        int product = coeffs[i] * q;
        if (product > short.MaxValue) product = short.MaxValue;
        else if (product < short.MinValue) product = short.MinValue;
        coeffs[i] = (short)product;
    }

    /// <summary>Release kernel resources. Does not dispose the accelerator.</summary>
    public void Dispose() { }
}
