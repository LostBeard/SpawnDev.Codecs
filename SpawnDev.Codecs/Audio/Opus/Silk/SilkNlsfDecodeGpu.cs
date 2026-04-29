// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-callable SILK full NLSF decode pipeline. Mirror of
// SilkNlsfDecoder.Decode (libopus silk/NLSF_decode.c). Composes 4
// already-shipped GPU primitives within a single kernel thread:
//   1. SilkNlsfUnpackGpu     - per-pair ec table indices + Q8 predictors
//   2. SilkNlsfResidualDequantGpu - reverse-iterating residual dequant
//   3. SilkNlsfWeightedAddGpu - per-coef inverse-weight + first-stage add
//   4. SilkNlsfStabilizeGpu  - iterative ordering + spacing fix
//
// Single-thread per stream because all stages compose sequentially: each
// reads the output of the previous. Multiple independent streams (multi-
// channel decode) parallelize cleanly across threads.
//
// Caller provides scratch buffers for the inter-stage temporaries
// (ecIx[order] + predQ8[order] + resQ10[order]) plus the codebook
// arrays (cb1[nVec*order] + cbWght[nVec*order] + ecSel[nVec*order/2] +
// predQ8Source[2*(order-1)] + deltaMinQ15[order+1]).

using ILGPU;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// GPU-callable SILK full NLSF decode pipeline. Mirror of
/// <see cref="SilkNlsfDecoder"/>.Decode.
/// </summary>
public static class SilkNlsfDecodeGpu
{
    /// <summary>
    /// Decode one NLSF vector from its codebook path. Bit-exact vs the CPU
    /// SilkNlsfDecoder.Decode. Single-thread on the GPU; caller dispatches
    /// 1 thread per independent SILK stream.
    /// </summary>
    /// <param name="pNlsfQ15">Output Q15 NLSF vector (length order).</param>
    /// <param name="nlsfBase">Base offset.</param>
    /// <param name="nlsfIndices">Codebook path: [0]=cb1Index, [1..order]=residual indices (length order+1).</param>
    /// <param name="indicesBase">Base offset.</param>
    /// <param name="cb1NlsfQ8">Codebook first-stage NLSFs in Q8 (length nVec*order).</param>
    /// <param name="cb1Base">Base offset.</param>
    /// <param name="cb1WghtQ9">Codebook inverse weights in Q9 (length nVec*order).</param>
    /// <param name="cbWghtBase">Base offset.</param>
    /// <param name="ecSel">Codebook ec_sel bytes (length nVec*order/2).</param>
    /// <param name="ecSelBase">Base offset.</param>
    /// <param name="predQ8Source">Codebook PredQ8 array (length 2*(order-1)).</param>
    /// <param name="predQ8SrcBase">Base offset.</param>
    /// <param name="deltaMinQ15">Codebook DeltaMinQ15 array (length order+1).</param>
    /// <param name="deltaMinBase">Base offset.</param>
    /// <param name="quantStepSizeQ16">Codebook quantizer step size in Q16.</param>
    /// <param name="order">NLSF filter order (10 or 16).</param>
    /// <param name="scratch">Per-call scratch (length &gt;= 3*order shorts; reuses int slots).
    /// Must be at least 3 * MAX_LPC_ORDER = 48 shorts. Layout: ecIx[order] + predQ8 packed
    /// into shorts[order] + resQ10[order]. Contents replaced.</param>
    /// <param name="scratchBase">Base offset.</param>
    public static void DecodeAt(
        ArrayView<short> pNlsfQ15, long nlsfBase,
        ArrayView<sbyte> nlsfIndices, long indicesBase,
        ArrayView<byte> cb1NlsfQ8, long cb1Base,
        ArrayView<short> cb1WghtQ9, long cbWghtBase,
        ArrayView<byte> ecSel, long ecSelBase,
        ArrayView<byte> predQ8Source, long predQ8SrcBase,
        ArrayView<short> deltaMinQ15, long deltaMinBase,
        int quantStepSizeQ16, int order,
        ArrayView<short> scratch, long scratchBase,
        ArrayView<byte> predScratch, long predScratchBase)
    {
        int cb1Index = nlsfIndices[indicesBase + 0];
        long perEntryEcSelBase = ecSelBase + (long)cb1Index * order / 2;
        long perEntryCbBase = (long)cb1Index * order;

        // Stage 1: Unpack per-pair into ecIx (scratch[0..order]) + predQ8 (predScratch).
        // ecIx is only used by the parametric NLSF residual coding (range decode);
        // the GPU pipeline that runs after entropy decode only needs predQ8.
        // We still write ecIx for symmetry; consumer can ignore.
        for (int pair = 0; pair < order / 2; pair++)
        {
            byte entry = ecSel[perEntryEcSelBase + pair];
            int i = 2 * pair;

            // ecIx values
            int low3 = (entry >> 1) & 7;
            int high3 = (entry >> 5) & 7;
            scratch[scratchBase + i] = (short)(low3 * 9);
            scratch[scratchBase + i + 1] = (short)(high3 * 9);

            // predQ8 values
            int variant0 = entry & 1;
            int variant1 = (entry >> 4) & 1;
            predScratch[predScratchBase + i] =
                predQ8Source[predQ8SrcBase + i + variant0 * (order - 1)];
            predScratch[predScratchBase + i + 1] =
                predQ8Source[predQ8SrcBase + i + variant1 * (order - 1) + 1];
        }

        // Stage 2: ResidualDequant - sequential reverse iteration.
        // Output goes to scratch[order..2*order] as shorts.
        long resQ10Base = scratchBase + order;
        int outQ10 = 0;
        for (int i = order - 1; i >= 0; i--)
        {
            short predCoefShort = predScratch[predScratchBase + i];
            int predQ10 = (outQ10 * predCoefShort) >> 8;

            outQ10 = (int)nlsfIndices[indicesBase + 1 + i] << 10;
            if (outQ10 > 0) outQ10 -= 102;
            else if (outQ10 < 0) outQ10 += 102;

            outQ10 = predQ10 + (int)((long)outQ10 * (short)quantStepSizeQ16 >> 16);
            scratch[resQ10Base + i] = (short)outQ10;
        }

        // Stage 3: WeightedAdd - per-coefficient inverse weight + first-stage add.
        for (int i = 0; i < order; i++)
        {
            int residual = (int)scratch[resQ10Base + i] << 14;
            int weight = cb1WghtQ9[cbWghtBase + perEntryCbBase + i];
            int weightedResidual = residual / weight;
            int cb1Val = cb1NlsfQ8[cb1Base + perEntryCbBase + i];
            int nlsfQ15Tmp = weightedResidual + (cb1Val << 7);

            if (nlsfQ15Tmp < 0) nlsfQ15Tmp = 0;
            else if (nlsfQ15Tmp > 32767) nlsfQ15Tmp = 32767;
            pNlsfQ15[nlsfBase + i] = (short)nlsfQ15Tmp;
        }

        // Stage 4: Stabilize - iterative ordering + spacing fix.
        SilkNlsfStabilizeGpu.Stabilize(pNlsfQ15, nlsfBase, order, deltaMinQ15, deltaMinBase);
    }
}
