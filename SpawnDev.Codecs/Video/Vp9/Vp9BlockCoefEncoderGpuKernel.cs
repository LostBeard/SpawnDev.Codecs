// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Test/integration kernel that exercises Vp9BlockCoefEncoderGpu by
// encoding a single transform block on the accelerator. Single-thread
// dispatch; the entire (state init, encode, stop) pipeline runs in
// one GPU thread so the bool-coder state stays kernel-local just like
// the real frame entropy kernel does.
//
// Per-thread tokenCache is supplied as a regular GPU buffer (sized to
// maxCoefs) rather than via LocalMemory.Allocate. This keeps the test
// surface backend-agnostic (LocalMemory has size restrictions on some
// backends and we want this kernel to exercise as many backends as
// the underlying primitives allow).
//
// The flags int packs the per-block selectors so the kernel signature
// fits in ILGPU's 15-argument Action budget:
//   bit 0       planeType (0 = Y, 1 = UV)
//   bit 1       refType   (0 = Intra, 1 = Inter)
//   bits 2..4   initialCtx (0..2)
//   bit 5       isHighBitDepth (0 = 8-bit, 1 = 12-bit)
//   bit 6       isTx4x4 (0 = 8x8/16x16/32x32, 1 = 4x4)

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Vp8;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// Drives <see cref="Vp9BlockCoefEncoderGpu.EncodeBlock"/> on the
/// accelerator for one block per dispatch. Used to verify bit-exact
/// agreement with <see cref="Vp9BlockCoefEncoder"/>.
/// </summary>
public sealed class Vp9BlockCoefEncoderTestKernel : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Action<
        Index1D,
        ArrayView<byte>, ArrayView<short>, ArrayView<ushort>, ArrayView<ushort>,
        ArrayView<byte>, ArrayView<byte>, ArrayView<byte>,
        ArrayView<long>, ArrayView<int>,
        int, int> _kernel;

    /// <summary>Compile.</summary>
    public Vp9BlockCoefEncoderTestKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<byte>, ArrayView<short>, ArrayView<ushort>, ArrayView<ushort>,
            ArrayView<byte>, ArrayView<byte>, ArrayView<byte>,
            ArrayView<long>, ArrayView<int>,
            int, int>(EncodeBlockKernel);
    }

    /// <summary>
    /// Pack the per-block selectors into a single int per the layout
    /// described in the file header.
    /// </summary>
    public static int PackFlags(int planeType, int refType, int initialCtx, int isHighBitDepth, int isTx4x4)
        => (planeType & 1)
           | ((refType & 1) << 1)
           | ((initialCtx & 7) << 2)
           | ((isHighBitDepth & 1) << 5)
           | ((isTx4x4 & 1) << 6);

    /// <summary>
    /// Encode one block. <paramref name="outLen"/> is written with the
    /// number of bytes used; <paramref name="eobOut"/> is written with
    /// the EOB position.
    /// </summary>
    public void Run(
        ArrayView<byte> outBuf,
        ArrayView<short> coefs,
        ArrayView<ushort> scan,
        ArrayView<ushort> neighbors,
        ArrayView<byte> coefProbs,
        ArrayView<byte> consts,
        ArrayView<byte> tokenCache,
        ArrayView<long> outLen,
        ArrayView<int> eobOut,
        int maxCoefs,
        int packedFlags)
    {
        if (tokenCache.Length < maxCoefs)
            throw new ArgumentException("tokenCache too short.", nameof(tokenCache));
        if (outLen.Length < 1) throw new ArgumentException("outLen must hold at least 1 entry.", nameof(outLen));
        if (eobOut.Length < 1) throw new ArgumentException("eobOut must hold at least 1 entry.", nameof(eobOut));
        _kernel(1, outBuf, coefs, scan, neighbors, coefProbs, consts, tokenCache,
                outLen, eobOut, maxCoefs, packedFlags);
    }

    private static void EncodeBlockKernel(
        Index1D _,
        ArrayView<byte> outBuf,
        ArrayView<short> coefs,
        ArrayView<ushort> scan,
        ArrayView<ushort> neighbors,
        ArrayView<byte> coefProbs,
        ArrayView<byte> consts,
        ArrayView<byte> tokenCache,
        ArrayView<long> outLen,
        ArrayView<int> eobOut,
        int maxCoefs,
        int packedFlags)
    {
        int planeType = packedFlags & 1;
        int refType = (packedFlags >> 1) & 1;
        int initialCtx = (packedFlags >> 2) & 7;
        int isHighBitDepth = (packedFlags >> 5) & 1;
        int isTx4x4 = (packedFlags >> 6) & 1;

        // VP9 emits a leading marker bit (0 at probability 128) right
        // after Init, which the decoder consumes during start. Real
        // frame integration will move this into the frame-level
        // entropy kernel; here we mirror what a per-block round-trip
        // test against the CPU side does.
        var state = Vp8BoolEncoderGpu.Init();
        Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 0, 128);

        int eob = Vp9BlockCoefEncoderGpu.EncodeBlock(
            ref state, outBuf, coefs, scan, neighbors, coefProbs, consts, tokenCache,
            maxCoefs, planeType, refType, initialCtx, isHighBitDepth, isTx4x4);

        Vp8BoolEncoderGpu.Stop(ref state, outBuf);

        outLen[0] = state.OutLen;
        eobOut[0] = eob;
    }

    /// <summary>Release kernel resources.</summary>
    public void Dispose() { /* auto-grouped */ }
}
