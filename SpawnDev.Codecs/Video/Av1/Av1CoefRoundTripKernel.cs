// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Round-trip kernel for AV1 coef encoder + decoder GPU verification.
// Encodes one block via Av1CoefEncoderGpu, then decodes the same
// bitstream via Av1CoefDecoderGpu in the same dispatch. Lets the
// host verify the round-trip property (decoded coefs == input coefs
// when qDc=qAc=1, modulo the level cast to byte).
//
// Eob + CulLevel are checked separately via the encoder-only and
// decoder-only kernels - this round-trip only proves the encoder
// output is valid input to the decoder and the decoded values
// reconstruct the original.

using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>
/// Packed scalar parameters for the round-trip kernel - keeps the
/// kernel signature under ILGPU's Action&lt;...&gt; generic-arg ceiling.
/// </summary>
public struct Av1CoefRoundTripParams
{
    /// <summary>Tx size: 1 = Tx8x8, 2 = Tx16x16.</summary>
    public int TxSize;
    /// <summary>Plane: 0 = Y, 1 = U, 2 = V.</summary>
    public int Plane;
    /// <summary>Quantizer-bin index (0..3).</summary>
    public int Qctx;
    /// <summary>txb_skip CDF context.</summary>
    public int TxbSkipCtx;
    /// <summary>dc_sign CDF context.</summary>
    public int DcSignCtx;
    /// <summary>Frame base q-index.</summary>
    public int Qindex;
    /// <summary>DC quantizer scale (caller-precomputed).</summary>
    public int QDc;
    /// <summary>AC quantizer scale (caller-precomputed).</summary>
    public int QAc;
}

/// <summary>
/// Drives an end-to-end Av1CoefEncoderGpu -&gt; Av1CoefDecoderGpu
/// round-trip on the accelerator.
/// </summary>
public sealed class Av1CoefRoundTripKernel : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Action<
        Index1D,
        ArrayView<byte>, ArrayView<byte>, ArrayView<ushort>,
        ArrayView<int>, ArrayView<int>, ArrayView<byte>,
        ArrayView<long>, ArrayView<int>,
        Av1CoefRoundTripParams> _kernel;

    /// <summary>Compile.</summary>
    public Av1CoefRoundTripKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<byte>, ArrayView<byte>, ArrayView<ushort>,
            ArrayView<int>, ArrayView<int>, ArrayView<byte>,
            ArrayView<long>, ArrayView<int>,
            Av1CoefRoundTripParams>(RoundTripKernel);
    }

    /// <summary>
    /// Encode + decode one block in the same dispatch. Outputs:
    /// <c>outLen[0]</c> = encoded byte count;
    /// <c>encDecInfo[0]</c> = encEob, <c>[1]</c> = decEob.
    /// (CulLevel for both halves is verified by the encoder-only and
    /// decoder-only kernels - not duplicated here.)
    /// </summary>
    public void Run(
        ArrayView<byte> scratchBytes,
        ArrayView<byte> constsByte,
        ArrayView<ushort> constsUshort,
        ArrayView<int> coefsRaster,
        ArrayView<int> decodedCoefsRaster,
        ArrayView<byte> levelsBuf,
        ArrayView<long> outLen,
        ArrayView<int> encDecInfo,
        Av1CoefRoundTripParams parms)
    {
        _kernel(1, scratchBytes, constsByte, constsUshort,
            coefsRaster, decodedCoefsRaster, levelsBuf, outLen, encDecInfo,
            parms);
    }

    private static void RoundTripKernel(
        Index1D _,
        ArrayView<byte> scratchBytes,
        ArrayView<byte> constsByte,
        ArrayView<ushort> constsUshort,
        ArrayView<int> coefsRaster,
        ArrayView<int> decodedCoefsRaster,
        ArrayView<byte> levelsBuf,
        ArrayView<long> outLen,
        ArrayView<int> encDecInfo,
        Av1CoefRoundTripParams parms)
    {
        // Encode phase. encDecInfo[0] receives encEob (the encoder
        // also writes culLevel to encDecInfo[0] as the second write -
        // but since we pass the same view + same index, the cul
        // overwrites the eob. To capture only encEob, the simplest
        // approach is to use SubView for the cul output, sending it
        // to a throwaway slot at index 2.
        var re = Av1RangeEncoderGpu.Init();
        Av1CoefEncoderGpu.WriteCoeffsTxb(
            ref re, scratchBytes, constsByte, constsUshort,
            coefsRaster, 0, levelsBuf, 0,
            parms.TxSize, parms.Plane, parms.Qctx,
            parms.TxbSkipCtx, parms.DcSignCtx, parms.Qindex,
            encDecInfo, encDecInfo.SubView(2, 1), 0);
        // After encode: encDecInfo[0] = encEob, encDecInfo[2] = encCul (throwaway).
        Av1RangeEncoderGpu.Done(ref re, scratchBytes);
        outLen[0] = re.OutLen;

        // Decode phase. encDecInfo[1] receives decEob; the cul lands
        // at encDecInfo[3] (also throwaway).
        var rd = Av1RangeDecoderGpu.Init(scratchBytes, 0, (int)re.OutLen);
        Av1CoefDecoderGpu.ReadCoeffsTxb(
            ref rd, scratchBytes, constsByte, constsUshort,
            decodedCoefsRaster, 0, levelsBuf, 0,
            parms.TxSize, parms.Plane, parms.Qctx,
            parms.TxbSkipCtx, parms.DcSignCtx, parms.Qindex,
            parms.QDc, parms.QAc,
            encDecInfo.SubView(1, 1), encDecInfo.SubView(3, 1), 0);
    }

    /// <summary>Release kernel resources.</summary>
    public void Dispose() { /* auto-grouped */ }
}
