// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 compressed frame header emit kernel. Single-thread dispatch;
// emits the bit-exact bool-coded compressed header for a v1 keyframe.
//
// The v1 compressed header is intentionally small. For
// tx_mode=Allow32x32 with default probs and no skip-prob updates it
// reduces to:
//   1. VP9 bool-coder marker bit (0 at prob 128) - emitted by Init
//      because the decoder consumes it during start.
//   2. tx_mode = Allow32x32 = 0b11 (two literal bits at prob 128 each,
//      MSB-first).
//   3. Four coef-prob "no update" gate bits, one per tx_size 0..3,
//      each at prob 128.
//   4. Three skip-prob "no update" diff_update_prob bits at prob 252
//      (one per skip-context bucket).
//   5. Stop() - 32 trailing zero bits at prob 128.
//
// Full reference: Vp9KeyframeEncoder.EmitNoCoefProbUpdates +
// EmitNoDiffUpdate. The four coef-prob gate bits skip the entire
// (256-entry) per-tx_size update inner loop; the three skip-prob
// diff_update_prob bits skip the per-context skip-prob update inner
// loop. Both are sufficient for v1 because the decoder falls back
// to the libvpx static defaults when the gates are 0.
//
// All numeric constants come from the CPU encoder:
//   prob 128 = libvpx LITERAL_PROB
//   prob 252 = Vp9DiffUpdateProb.UpdateProb
//   tx_mode Allow32x32 = 3 -> two MSB-first bits "11"
//   biggest tx_size for Allow32x32 = Tx32x32 (value 3) -> 4 gate bits

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Vp8;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// VP9 compressed frame header emit kernel. Writes the bit-exact
/// bool-coded compressed header for a v1 keyframe to a GPU output
/// buffer.
/// </summary>
public sealed class Vp9FrameCompressedHeaderKernel : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<long>> _kernel;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<long>, int> _batchKernel;

    /// <summary>Compile.</summary>
    public Vp9FrameCompressedHeaderKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<long>>(EmitKernel);
        _batchKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<long>, int>(EmitBatchKernel);
    }

    /// <summary>
    /// Emit the v1 keyframe compressed header. <paramref name="outLen"/>
    /// is written with the number of bytes used. The caller passes the
    /// number of bytes back to the uncompressed-header kernel as
    /// firstPartitionSize.
    /// </summary>
    public void Run(ArrayView<byte> outBuf, ArrayView<long> outLen)
    {
        if (outLen.Length < 1) throw new ArgumentException("outLen must hold 1 entry.", nameof(outLen));
        // Worst case: 9 bool emits + 32 stop emits, each emitting <= 1 byte.
        // 64 bytes is more than enough; require at least that.
        if (outBuf.Length < 64)
            throw new ArgumentException("outBuf must hold at least 64 bytes.", nameof(outBuf));
        _kernel(1, outBuf, outLen);
    }

    /// <summary>Batch: extent=N, each thread emits one frame's compressed header.</summary>
    public void RunBatch(ArrayView<byte> outBuf, ArrayView<long> outLen,
        int frameCount, int outBufStride)
    {
        if (frameCount <= 0) throw new ArgumentOutOfRangeException(nameof(frameCount));
        _batchKernel(frameCount, outBuf, outLen, outBufStride);
    }

    private static void EmitBatchKernel(
        Index1D idx,
        ArrayView<byte> outBuf, ArrayView<long> outLen, int outBufStride)
    {
        int f = idx.X;
        var fOut = outBuf.SubView((long)f * outBufStride, outBufStride);
        var fOutLen = outLen.SubView(f, 1);
        EmitBody(fOut, fOutLen);
    }

    private static void EmitKernel(
        Index1D _,
        ArrayView<byte> outBuf,
        ArrayView<long> outLenOut)
    {
        EmitBody(outBuf, outLenOut);
    }

    private static void EmitBody(ArrayView<byte> outBuf, ArrayView<long> outLenOut)
    {
        var state = Vp8BoolEncoderGpu.Init();

        // 1. VP9 marker bit (0 at prob 128).
        Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 0, 128);

        // 2. tx_mode = Allow32x32 (3, two bits "11" MSB-first via WriteLiteral).
        Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 1, 128);
        Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 1, 128);

        // 3. Four coef-prob "no update" gate bits (prob 128 each).
        // For Allow32x32, biggest tx_size is Tx32x32 = 3, so the loop
        // is t = 0..3 inclusive -> 4 gate bits.
        Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 0, 128);
        Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 0, 128);
        Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 0, 128);
        Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 0, 128);

        // 4. Three skip-prob "no update" diff_update_prob bits (prob 252 each).
        Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 0, 252);
        Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 0, 252);
        Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 0, 252);

        // 5. Stop: 32 trailing zero-prob-128 bits to flush.
        Vp8BoolEncoderGpu.Stop(ref state, outBuf);

        outLenOut[0] = state.OutLen;
    }

    /// <summary>Release kernel resources.</summary>
    public void Dispose() { /* auto-grouped */ }
}
