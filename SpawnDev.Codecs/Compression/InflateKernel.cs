// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Launchable kernel wrapper around InflateGpu.InflateZlib: one GPU thread inflates one zlib stream
// that is already on the accelerator, into an accelerator buffer. Neither the compressed nor the
// inflated bytes cross to the host; the host reads back one int (the length or a negated
// InflateError).

using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.Codecs.Compression;

/// <summary>Launches <see cref="InflateGpu.InflateZlib"/> on the accelerator.</summary>
public sealed class InflateKernel : IDisposable
{
    private readonly Action<
        Index1D,
        ArrayView<byte>, int,
        ArrayView<byte>, int,
        ArrayView<int>, ArrayView<int>, int> _kernel;

    /// <summary>Compile.</summary>
    public InflateKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<byte>, int,
            ArrayView<byte>, int,
            ArrayView<int>, ArrayView<int>, int>(InflateStreamKernel);
    }

    /// <summary>
    /// Inflate the zlib stream in <paramref name="zlib"/> (first <paramref name="zlibLength"/> bytes)
    /// into <paramref name="output"/> (first <paramref name="outputCapacity"/> bytes). The result
    /// (inflated length, or the negated <see cref="InflateError"/>) is written to
    /// <c>result[0]</c>; the caller reads that one int after synchronizing.
    /// </summary>
    /// <param name="zlib">Compressed zlib stream.</param>
    /// <param name="zlibLength">Number of valid bytes in <paramref name="zlib"/>.</param>
    /// <param name="output">Receives the inflated bytes.</param>
    /// <param name="outputCapacity">Maximum bytes the stream may inflate to; more is <see cref="InflateError.OutputOverflow"/>.</param>
    /// <param name="scratch">At least <see cref="InflateGpu.ScratchInts"/> ints of working memory.</param>
    /// <param name="result">At least one int.</param>
    /// <param name="verifyAdler">Verify the zlib Adler-32 trailer.</param>
    public void Run(
        ArrayView<byte> zlib, int zlibLength,
        ArrayView<byte> output, int outputCapacity,
        ArrayView<int> scratch, ArrayView<int> result,
        bool verifyAdler = true)
    {
        if (zlibLength < 0 || zlibLength > zlib.Length) throw new ArgumentOutOfRangeException(nameof(zlibLength));
        if (outputCapacity < 0 || outputCapacity > output.Length) throw new ArgumentOutOfRangeException(nameof(outputCapacity));
        if (scratch.Length < InflateGpu.ScratchInts)
            throw new ArgumentException($"scratch must hold {InflateGpu.ScratchInts} ints.", nameof(scratch));
        if (result.Length < 1) throw new ArgumentException("result must hold 1 int.", nameof(result));
        _kernel(1, zlib, zlibLength, output, outputCapacity, scratch, result, verifyAdler ? 1 : 0);
    }

    private static void InflateStreamKernel(
        Index1D _,
        ArrayView<byte> zlib, int zlibLength,
        ArrayView<byte> output, int outputCapacity,
        ArrayView<int> scratch, ArrayView<int> result, int verifyAdler)
    {
        result[0] = InflateGpu.InflateZlib(
            zlib, 0, zlibLength, output, 0, outputCapacity, scratch, 0, verifyAdler != 0);
    }

    /// <summary>No-op (kernels are owned by the accelerator).</summary>
    public void Dispose() { }
}
