// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Test/integration kernel that drives `SilkDecodeCoreGpu.Decode` on the
// accelerator. Two-struct param shape: SilkDecodeCoreInputs body struct
// for ArrayViews, SilkDecodeCoreScalars for ints. One thread per stream.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// Drives <see cref="SilkDecodeCoreGpu.Decode"/> on the accelerator.
/// </summary>
public sealed class SilkDecodeCoreGpuTestKernel : IDisposable
{
    private readonly Action<Index1D, SilkDecodeCoreInputs, SilkDecodeCoreScalars> _kernel;

    /// <summary>Compile.</summary>
    public SilkDecodeCoreGpuTestKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, SilkDecodeCoreInputs, SilkDecodeCoreScalars>(DecodeCoreKernel);
    }

    /// <summary>Run the synthesis chain for one frame.</summary>
    public void Run(SilkDecodeCoreInputs inputs, SilkDecodeCoreScalars scalars)
    {
        _kernel(1, inputs, scalars);
    }

    private static void DecodeCoreKernel(
        Index1D _,
        SilkDecodeCoreInputs inputs,
        SilkDecodeCoreScalars scalars)
    {
        SilkDecodeCoreGpu.Decode(inputs, scalars);
    }

    /// <summary>Release.</summary>
    public void Dispose() { /* auto-grouped */ }
}
