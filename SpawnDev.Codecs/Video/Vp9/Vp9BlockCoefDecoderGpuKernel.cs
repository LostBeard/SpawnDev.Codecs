// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Test/integration kernel that exercises Vp9BlockCoefDecoderGpu by
// decoding a single transform block on the accelerator. Single-thread
// dispatch; the entire (state init, marker bit consume, decode)
// pipeline runs in one GPU thread. Mirror of
// Vp9BlockCoefEncoderTestKernel - same flag packing, same per-thread
// tokenCache convention.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Vp8;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// Drives <see cref="Vp9BlockCoefDecoderGpu.DecodeBlock"/> on the
/// accelerator for one block per dispatch. Used to verify bit-exact
/// agreement with <see cref="Vp9BlockCoefDecoder"/>.
/// </summary>
public sealed class Vp9BlockCoefDecoderTestKernel : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Action<
        Index1D,
        ArrayView<byte>, ArrayView<short>, ArrayView<ushort>, ArrayView<ushort>,
        ArrayView<byte>, ArrayView<byte>, ArrayView<byte>,
        ArrayView<int>,
        int, int, int> _kernel;

    /// <summary>Compile.</summary>
    public Vp9BlockCoefDecoderTestKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<byte>, ArrayView<short>, ArrayView<ushort>, ArrayView<ushort>,
            ArrayView<byte>, ArrayView<byte>, ArrayView<byte>,
            ArrayView<int>,
            int, int, int>(DecodeBlockKernel);
    }

    /// <summary>
    /// Decode one block. <paramref name="eobOut"/> is written with the
    /// decoded EOB position. The decoded coefficients land in
    /// <paramref name="block"/> (raster layout, pre-zeroed by the
    /// kernel).
    /// </summary>
    public void Run(
        ArrayView<byte> inBuf,
        ArrayView<short> block,
        ArrayView<ushort> scan,
        ArrayView<ushort> neighbors,
        ArrayView<byte> coefProbs,
        ArrayView<byte> consts,
        ArrayView<byte> tokenCache,
        ArrayView<int> eobOut,
        int inBufLen,
        int maxCoefs,
        int packedFlags)
    {
        if (tokenCache.Length < maxCoefs)
            throw new ArgumentException("tokenCache too short.", nameof(tokenCache));
        if (eobOut.Length < 1)
            throw new ArgumentException("eobOut must hold at least 1 entry.", nameof(eobOut));
        if (block.Length < maxCoefs)
            throw new ArgumentException("block too short.", nameof(block));
        _kernel(1, inBuf, block, scan, neighbors, coefProbs, consts, tokenCache,
                eobOut, inBufLen, maxCoefs, packedFlags);
    }

    private static void DecodeBlockKernel(
        Index1D _,
        ArrayView<byte> inBuf,
        ArrayView<short> block,
        ArrayView<ushort> scan,
        ArrayView<ushort> neighbors,
        ArrayView<byte> coefProbs,
        ArrayView<byte> consts,
        ArrayView<byte> tokenCache,
        ArrayView<int> eobOut,
        int inBufLen,
        int maxCoefs,
        int packedFlags)
    {
        int planeType = packedFlags & 1;
        int refType = (packedFlags >> 1) & 1;
        int initialCtx = (packedFlags >> 2) & 7;
        int isHighBitDepth = (packedFlags >> 5) & 1;
        // bit 6 is isTx4x4 - the decoder infers it from maxCoefs directly,
        // so we ignore that bit here. (Encoder needs it because its
        // band lookup reads at the same call site without knowing maxCoefs.)

        var state = Vp8BoolDecoderGpu.Init(inBuf, 0, inBufLen);
        // VP9 marker bit: encoder emits 0 at prob 128 right after Reset;
        // decoder consumes the same here to get the bool state in sync.
        Vp8BoolDecoderGpu.DecodeBool(ref state, inBuf, 128);

        int eob = Vp9BlockCoefDecoderGpu.DecodeBlock(
            ref state, inBuf, block, scan, neighbors, coefProbs, consts, tokenCache,
            maxCoefs, planeType, refType, initialCtx, isHighBitDepth);

        eobOut[0] = eob;
    }

    /// <summary>Release kernel resources.</summary>
    public void Dispose() { /* auto-grouped */ }
}
