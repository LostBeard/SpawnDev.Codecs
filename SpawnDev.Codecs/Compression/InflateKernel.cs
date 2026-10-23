// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Launchable kernel wrapper around the resumable InflateGpu decode: one GPU thread inflates one zlib
// stream that is already on the accelerator, into an accelerator buffer. Neither the compressed nor the
// inflated bytes cross to the host; the host reads back one int (the length or a negated
// InflateError).
//
// The decode runs as a set-up launch and then InflateGpu.MaxSteps(length, capacity) launches of
// InflateGpu.Step, each bounded to StepInputBytes compressed / StepOutputBytes inflated bytes, so no
// launch holds the GPU long whatever the stream size. The launch count is known on the host from the two
// sizes; launches after the end of the stream return at once.

using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.Codecs.Compression;

/// <summary>Launches the resumable zlib decode (<see cref="InflateGpu.InitZlib"/> + <see cref="InflateGpu.Step"/>) on the accelerator.</summary>
public sealed class InflateKernel : IDisposable
{
    private readonly Action<Index1D, ArrayView<byte>, int, int, ArrayView<int>, ArrayView<int>, int> _init;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>> _step;

    /// <summary>Compile.</summary>
    public InflateKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _init = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, int, int, ArrayView<int>, ArrayView<int>, int>(InflateInitKernel);
        _step = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>>(InflateStepKernel);
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
        _init(1, zlib, zlibLength, outputCapacity, scratch, result, verifyAdler ? 1 : 0);
        int steps = InflateGpu.MaxSteps(zlibLength, outputCapacity);
        for (int i = 0; i < steps; i++) _step(1, zlib, output, scratch, result);
    }

    private static void InflateInitKernel(
        Index1D _, ArrayView<byte> zlib, int zlibLength, int outputCapacity,
        ArrayView<int> scratch, ArrayView<int> result, int verifyAdler)
    {
        InflateGpu.InitZlib(zlib, 0, zlibLength, 0, outputCapacity, scratch, 0, verifyAdler != 0);
        result[0] = InflateGpu.Result(scratch, 0);
    }

    private static void InflateStepKernel(
        Index1D _, ArrayView<byte> zlib, ArrayView<byte> output, ArrayView<int> scratch, ArrayView<int> result)
    {
        InflateGpu.Step(zlib, output, scratch, 0, InflateGpu.StepInputBytes, InflateGpu.StepOutputBytes);
        result[0] = InflateGpu.Result(scratch, 0);
    }

    /// <summary>No-op (kernels are owned by the accelerator).</summary>
    public void Dispose() { }
}
