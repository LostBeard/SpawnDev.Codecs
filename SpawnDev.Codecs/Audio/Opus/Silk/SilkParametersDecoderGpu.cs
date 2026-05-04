// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-callable orchestrator port of SilkParametersDecoder.Decode (libopus
// silk/decode_parameters.c). Given decoded indices from SilkIndicesDecoderGpu,
// dequantizes every per-frame quantity needed by silk_decode_core: gains,
// NLSFs (with inter-frame interpolation), LPC coefficients per half-frame,
// pitch lags per subframe, LTP filter taps per subframe, and the LTP scale
// factor.
//
// Composes existing GPU primitives:
//   - SilkGainDecoderGpu.DequantizeAt
//   - SilkNlsfDecodeGpu.DecodeAt
//   - SilkNlsf2AGpu.ComputeAt (twice: once for second-half, once for
//     first-half if interpolation enabled)
//   - SilkPitchComputeLagsGpu.ComputeLags
//   - inline LTP gain Q14 scaling (sbyte<<7) + LtpScaleQ14 lookup
//
// Sequential per-stream because every stage shares state. Single thread
// per stream; multi-channel decode parallelizes across threads.
//
// Upstream Copyright (c) 2006-2011 Skype Limited. BSD 3-Clause.

using ILGPU;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// Output layout for SilkParametersDecoderGpu.Decode. Like
/// SilkDecodedIndicesLayout, all values share a flat buffer with
/// named offsets so callers can extract individual fields after readback.
/// Mixed-type output (ints for gains/pitchLags + shorts for nlsf/lpc/ltp).
/// We use TWO output buffers to avoid mixing types in one ArrayView.
/// </summary>
public static class SilkDecodedParametersLayout
{
    /// <summary>Output ArrayView&lt;int&gt; layout (16 ints worst case):
    /// [0..nbSubfr) = gainsQ16; [4..4+nbSubfr) = pitchL.</summary>
    public const int IntGainsQ16Offset = 0;
    /// <summary>pitchL slot start (4 ints; 0 if unvoiced).</summary>
    public const int IntPitchLOffset = 4;
    /// <summary>Total int slots needed.</summary>
    public const int IntTotalSlots = 8;

    /// <summary>Output ArrayView&lt;short&gt; layout (worst-case order=16):
    /// [0..order) = nlsfQ15; [16..16+order) = predCoefQ12 first half;
    /// [32..32+order) = predCoefQ12 second half; [48..48+nbSubfr*5) = ltpCoefQ14;
    /// [68] = ltpScaleQ14.</summary>
    public const int ShortNlsfQ15Offset = 0;
    /// <summary>Predictor coefficients Q12 first half (size order).</summary>
    public const int ShortPredCoefQ12Half1Offset = 16;
    /// <summary>Predictor coefficients Q12 second half (size order).</summary>
    public const int ShortPredCoefQ12Half2Offset = 32;
    /// <summary>LTP filter taps Q14 (size nbSubfr * 5; 0 if unvoiced).</summary>
    public const int ShortLtpCoefQ14Offset = 48;
    /// <summary>Single-short LTP scale factor Q14 (0 if unvoiced).</summary>
    public const int ShortLtpScaleQ14Offset = 68;
    /// <summary>Total short slots needed (worst case).</summary>
    public const int ShortTotalSlots = 69;
}

/// <summary>
/// GPU-callable orchestrator that dequantizes SILK frame parameters.
/// Mirror of `SilkParametersDecoder.Decode`.
/// </summary>
public static class SilkParametersDecoderGpu
{
    /// <summary>SILK signal type: voiced.</summary>
    public const int TypeVoiced = 2;
    /// <summary>SILK MAX_LPC_ORDER constant.</summary>
    public const int MaxLpcOrder = 16;

    /// <summary>
    /// Dequantize all per-frame parameters from the decoded indices buffer.
    /// </summary>
    /// <param name="indicesIn">Output of SilkIndicesDecoderGpu.Decode (flat int buffer
    /// laid out per <see cref="SilkDecodedIndicesLayout"/>).</param>
    /// <param name="indicesInBase">Offset into indicesIn.</param>
    /// <param name="cb1NlsfQ8">NLSF codebook first-stage Q8 NLSFs (length nVec*order).</param>
    /// <param name="cb1WghtQ9">NLSF codebook first-stage inverse weights Q9.</param>
    /// <param name="ecSel">NLSF codebook EcSel bytes.</param>
    /// <param name="predQ8Source">NLSF codebook PredQ8 source.</param>
    /// <param name="deltaMinQ15">NLSF codebook DeltaMinQ15 array (length order+1).</param>
    /// <param name="lsfCosTabQ12">SilkLsfCosTab.Q12 (length 129).</param>
    /// <param name="contourCb">Caller-resolved (fs_kHz, nbSubfr) pitch contour codebook.</param>
    /// <param name="contourCbSize">Codebook size for the resolved contour.</param>
    /// <param name="ltpGainTablesFlat">Flat-packed LTP gain Q7 codebooks
    /// (LtpGain0[8*5] + LtpGain1[16*5] + LtpGain2[32*5] = 280 sbytes total).</param>
    /// <param name="ltpGainOffsets">[0, 40, 120] - sbyte offsets into ltpGainTablesFlat per perIndex.</param>
    /// <param name="ltpScaleQ14Table">LtpScalesQ14 lookup [15565, 12288, 8192].</param>
    /// <param name="prevNlsfQ15InOut">In/out: previous frame's NLSFs Q15 (length order).</param>
    /// <param name="prevNlsfQ15Base">Offset.</param>
    /// <param name="lastGainIndexInOut">In/out: 1-int buffer holding last gain index.</param>
    /// <param name="lastGainIndexBase">Offset.</param>
    /// <param name="nlsfDecodeScratch">Scratch for SilkNlsfDecodeGpu (length >= 3*MaxLpcOrder shorts).</param>
    /// <param name="nlsfDecodeScratchBase">Offset.</param>
    /// <param name="nlsfDecodePredScratch">Scratch for SilkNlsfDecodeGpu predQ8 (length >= MaxLpcOrder bytes).</param>
    /// <param name="nlsfDecodePredBase">Offset.</param>
    /// <param name="nlsf2aScratch">Scratch for SilkNlsf2AGpu (length >= 65 ints).</param>
    /// <param name="nlsf2aBase">Offset.</param>
    /// <param name="nlsfIndicesScratch">Scratch holding (order+1) sbyte NLSF indices, copied
    /// from indicesIn.</param>
    /// <param name="nlsfIndicesScratchBase">Offset.</param>
    /// <param name="gainIndicesScratch">Scratch holding nbSubfr sbyte gain indices, copied
    /// from indicesIn.</param>
    /// <param name="gainIndicesScratchBase">Offset.</param>
    /// <param name="quantStepSizeQ16">codebook.QuantStepSizeQ16.</param>
    /// <param name="order">codebook.Order (10 or 16).</param>
    /// <param name="nbSubfr">Subframe count (2 or 4).</param>
    /// <param name="fsKHz">Internal SILK sample rate (8, 12, or 16).</param>
    /// <param name="conditional">0 for independent gain coding, non-zero for conditional/delta.</param>
    /// <param name="intOut">Output ArrayView&lt;int&gt; - layout per
    /// <see cref="SilkDecodedParametersLayout"/>. Must be &gt;= IntTotalSlots.</param>
    /// <param name="intOutBase">Offset.</param>
    /// <param name="shortOut">Output ArrayView&lt;short&gt; - layout per
    /// <see cref="SilkDecodedParametersLayout"/>. Must be &gt;= ShortTotalSlots.</param>
    /// <param name="shortOutBase">Offset.</param>
    public static void Decode(
        ArrayView<int> indicesIn, long indicesInBase,
        ArrayView<byte> cb1NlsfQ8,
        ArrayView<short> cb1WghtQ9,
        ArrayView<byte> ecSel,
        ArrayView<byte> predQ8Source,
        ArrayView<short> deltaMinQ15,
        ArrayView<short> lsfCosTabQ12,
        ArrayView<sbyte> contourCb, int contourCbSize,
        ArrayView<sbyte> ltpGainTablesFlat,
        ArrayView<int> ltpGainOffsets,
        ArrayView<short> ltpScaleQ14Table,
        ArrayView<short> prevNlsfQ15InOut, long prevNlsfQ15Base,
        ArrayView<int> lastGainIndexInOut, long lastGainIndexBase,
        ArrayView<short> nlsfDecodeScratch, long nlsfDecodeScratchBase,
        ArrayView<byte> nlsfDecodePredScratch, long nlsfDecodePredBase,
        ArrayView<int> nlsf2aScratch, long nlsf2aBase,
        ArrayView<sbyte> nlsfIndicesScratch, long nlsfIndicesScratchBase,
        ArrayView<sbyte> gainIndicesScratch, long gainIndicesScratchBase,
        int quantStepSizeQ16, int order, int nbSubfr, int fsKHz, int conditional,
        ArrayView<int> intOut, long intOutBase,
        ArrayView<short> shortOut, long shortOutBase)
    {
        // Read decoded indices from the SilkIndicesDecoderGpu output buffer.
        int signalType = indicesIn[indicesInBase + SilkDecodedIndicesLayout.SignalTypeOffset];
        int nlsfInterpCoefQ2 = indicesIn[indicesInBase + SilkDecodedIndicesLayout.NlsfInterpCoefQ2Offset];
        int lagIndex = indicesIn[indicesInBase + SilkDecodedIndicesLayout.LagIndexOffset];
        int contourIndex = indicesIn[indicesInBase + SilkDecodedIndicesLayout.ContourIndexOffset];
        int perIndex = indicesIn[indicesInBase + SilkDecodedIndicesLayout.PerIndexOffset];
        int ltpScaleIndex = indicesIn[indicesInBase + SilkDecodedIndicesLayout.LtpScaleIndexOffset];

        // Convert int gain + nlsf indices to sbyte (SilkGainDecoderGpu /
        // SilkNlsfDecodeGpu primitives take ArrayView<sbyte>).
        for (int k = 0; k < nbSubfr; k++)
        {
            gainIndicesScratch[gainIndicesScratchBase + k] =
                (sbyte)indicesIn[indicesInBase + SilkDecodedIndicesLayout.GainsIndicesOffset + k];
        }
        for (int i = 0; i <= order; i++)
        {
            nlsfIndicesScratch[nlsfIndicesScratchBase + i] =
                (sbyte)indicesIn[indicesInBase + SilkDecodedIndicesLayout.NlsfIndicesOffset + i];
        }

        // 1. Dequantize gains.
        // SilkGainDecoderGpu.DequantizeAt expects conditional 0 or 1.
        int condForGain = conditional == 0 ? 0 : 1;
        SilkGainDecoderGpu.DequantizeAt(
            intOut, intOutBase + SilkDecodedParametersLayout.IntGainsQ16Offset,
            gainIndicesScratch, gainIndicesScratchBase,
            lastGainIndexInOut, lastGainIndexBase,
            condForGain, nbSubfr);

        // 2. Decode NLSF vector.
        long nlsfOutBase = shortOutBase + SilkDecodedParametersLayout.ShortNlsfQ15Offset;
        SilkNlsfDecodeGpu.DecodeAt(
            shortOut, nlsfOutBase,
            nlsfIndicesScratch, nlsfIndicesScratchBase,
            cb1NlsfQ8, 0,
            cb1WghtQ9, 0,
            ecSel, 0,
            predQ8Source, 0,
            deltaMinQ15, 0,
            quantStepSizeQ16, order,
            nlsfDecodeScratch, nlsfDecodeScratchBase,
            nlsfDecodePredScratch, nlsfDecodePredBase);

        // 3. NLSF -> LPC second half.
        long lpcHalf2Base = shortOutBase + SilkDecodedParametersLayout.ShortPredCoefQ12Half2Offset;
        SilkNlsf2AGpu.ComputeAt(
            shortOut, lpcHalf2Base,
            shortOut, nlsfOutBase,
            lsfCosTabQ12, 0,
            nlsf2aScratch, nlsf2aBase,
            order);

        // 4. NLSF -> LPC first half: interpolate from prev and current NLSF if
        // interp coef < 4, otherwise copy second half.
        long lpcHalf1Base = shortOutBase + SilkDecodedParametersLayout.ShortPredCoefQ12Half1Offset;
        if (nlsfInterpCoefQ2 < 4)
        {
            // Interpolated NLSF goes into nlsfDecodeScratch's first 16 short slots
            // (we've consumed it for SilkNlsfDecodeGpu but it's free now).
            for (int i = 0; i < order; i++)
            {
                int prev = prevNlsfQ15InOut[prevNlsfQ15Base + i];
                int cur = shortOut[nlsfOutBase + i];
                int delta = cur - prev;
                int interp = prev + ((nlsfInterpCoefQ2 * delta) >> 2);
                nlsfDecodeScratch[nlsfDecodeScratchBase + i] = (short)interp;
            }
            SilkNlsf2AGpu.ComputeAt(
                shortOut, lpcHalf1Base,
                nlsfDecodeScratch, nlsfDecodeScratchBase,
                lsfCosTabQ12, 0,
                nlsf2aScratch, nlsf2aBase,
                order);
        }
        else
        {
            for (int i = 0; i < order; i++)
                shortOut[lpcHalf1Base + i] = shortOut[lpcHalf2Base + i];
        }

        // 5. Update prev NLSFs for next frame.
        for (int i = 0; i < order; i++)
            prevNlsfQ15InOut[prevNlsfQ15Base + i] = shortOut[nlsfOutBase + i];

        // 6. Voiced-only: pitch + LTP.
        if (signalType == TypeVoiced)
        {
            // Pitch lag expansion.
            SilkPitchComputeLagsGpu.ComputeLags(
                contourCb, 0, contourCbSize,
                lagIndex, contourIndex,
                fsKHz, nbSubfr,
                intOut, intOutBase + SilkDecodedParametersLayout.IntPitchLOffset);

            // LTP filter taps Q14 (sbyte << 7).
            int ltpGainCbOffset = ltpGainOffsets[perIndex];
            for (int k = 0; k < nbSubfr; k++)
            {
                int ltpIdx = indicesIn[indicesInBase + SilkDecodedIndicesLayout.LtpIndicesOffset + k];
                long src = (long)ltpGainCbOffset + (long)ltpIdx * 5;
                long dst = shortOutBase + SilkDecodedParametersLayout.ShortLtpCoefQ14Offset + (long)k * 5;
                for (int i = 0; i < 5; i++)
                {
                    sbyte tap = ltpGainTablesFlat[src + i];
                    shortOut[dst + i] = (short)((int)tap << 7);
                }
            }

            // LTP scale Q14 lookup.
            shortOut[shortOutBase + SilkDecodedParametersLayout.ShortLtpScaleQ14Offset] =
                ltpScaleQ14Table[ltpScaleIndex];
        }
        else
        {
            // Zero out voiced-only fields.
            for (int k = 0; k < nbSubfr; k++)
                intOut[intOutBase + SilkDecodedParametersLayout.IntPitchLOffset + k] = 0;
            for (int i = 0; i < nbSubfr * 5; i++)
                shortOut[shortOutBase + SilkDecodedParametersLayout.ShortLtpCoefQ14Offset + i] = 0;
            shortOut[shortOutBase + SilkDecodedParametersLayout.ShortLtpScaleQ14Offset] = 0;
        }
    }
}
