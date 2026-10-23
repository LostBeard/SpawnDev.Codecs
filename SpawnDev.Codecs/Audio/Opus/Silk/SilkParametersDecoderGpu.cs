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
    /// <param name="inputs">Tables + state buffers, packed (see <see cref="SilkParametersInputs"/>).</param>
    /// <param name="contourCbSize">Codebook size for the resolved contour.</param>
    /// <param name="state">State + scratch offsets into the buffers of <paramref name="inputs"/>.</param>
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
        SilkParametersInputs inputs, int contourCbSize,
        SilkParametersState state,
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
            inputs.SBytes[state.GainIndicesScratchAt + k] =
                (sbyte)indicesIn[indicesInBase + SilkDecodedIndicesLayout.GainsIndicesOffset + k];
        }
        for (int i = 0; i <= order; i++)
        {
            inputs.SBytes[state.NlsfIndicesScratchAt + i] =
                (sbyte)indicesIn[indicesInBase + SilkDecodedIndicesLayout.NlsfIndicesOffset + i];
        }

        // 1. Dequantize gains.
        // SilkGainDecoderGpu.DequantizeAt expects conditional 0 or 1.
        int condForGain = conditional == 0 ? 0 : 1;
        SilkGainDecoderGpu.DequantizeAt(
            intOut, intOutBase + SilkDecodedParametersLayout.IntGainsQ16Offset,
            inputs.SBytes, state.GainIndicesScratchAt,
            inputs.Ints, state.LastGainIndexAt,
            condForGain, nbSubfr);

        // 2. Decode NLSF vector.
        long nlsfOutBase = shortOutBase + SilkDecodedParametersLayout.ShortNlsfQ15Offset;
        SilkNlsfDecodeGpu.DecodeAt(
            shortOut, nlsfOutBase,
            inputs.SBytes, state.NlsfIndicesScratchAt,
            inputs.Bytes, inputs.Cb1NlsfQ8At,
            inputs.Shorts, inputs.Cb1WghtQ9At,
            inputs.Bytes, inputs.EcSelAt,
            inputs.Bytes, inputs.PredQ8SourceAt,
            inputs.Shorts, inputs.DeltaMinQ15At,
            quantStepSizeQ16, order,
            inputs.Shorts, state.NlsfDecodeScratchAt,
            inputs.Bytes, state.NlsfDecodePredScratchAt);

        // 3. NLSF -> LPC second half.
        long lpcHalf2Base = shortOutBase + SilkDecodedParametersLayout.ShortPredCoefQ12Half2Offset;
        SilkNlsf2AGpu.ComputeAt(
            shortOut, lpcHalf2Base,
            shortOut, nlsfOutBase,
            inputs.Shorts, inputs.LsfCosTabQ12At,
            inputs.Ints, state.Nlsf2aScratchAt,
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
                int prev = inputs.Shorts[state.PrevNlsfQ15At + i];
                int cur = shortOut[nlsfOutBase + i];
                int delta = cur - prev;
                int interp = prev + ((nlsfInterpCoefQ2 * delta) >> 2);
                inputs.Shorts[state.NlsfDecodeScratchAt + i] = (short)interp;
            }
            SilkNlsf2AGpu.ComputeAt(
                shortOut, lpcHalf1Base,
                inputs.Shorts, state.NlsfDecodeScratchAt,
                inputs.Shorts, inputs.LsfCosTabQ12At,
                inputs.Ints, state.Nlsf2aScratchAt,
                order);
        }
        else
        {
            for (int i = 0; i < order; i++)
                shortOut[lpcHalf1Base + i] = shortOut[lpcHalf2Base + i];
        }

        // 5. Update prev NLSFs for next frame.
        for (int i = 0; i < order; i++)
            inputs.Shorts[state.PrevNlsfQ15At + i] = shortOut[nlsfOutBase + i];

        // 6. Voiced-only: pitch + LTP.
        if (signalType == TypeVoiced)
        {
            // Pitch lag expansion.
            SilkPitchComputeLagsGpu.ComputeLags(
                inputs.SBytes, inputs.ContourCbAt, contourCbSize,
                lagIndex, contourIndex,
                fsKHz, nbSubfr,
                intOut, intOutBase + SilkDecodedParametersLayout.IntPitchLOffset);

            // LTP filter taps Q14 (sbyte << 7).
            int ltpGainCbOffset = inputs.Ints[inputs.LtpGainOffsetsAt + perIndex];
            for (int k = 0; k < nbSubfr; k++)
            {
                int ltpIdx = indicesIn[indicesInBase + SilkDecodedIndicesLayout.LtpIndicesOffset + k];
                long src = (long)ltpGainCbOffset + (long)ltpIdx * 5;
                long dst = shortOutBase + SilkDecodedParametersLayout.ShortLtpCoefQ14Offset + (long)k * 5;
                for (int i = 0; i < 5; i++)
                {
                    sbyte tap = inputs.SBytes[inputs.LtpGainTablesFlatAt + src + i];
                    shortOut[dst + i] = (short)((int)tap << 7);
                }
            }

            // LTP scale Q14 lookup.
            shortOut[shortOutBase + SilkDecodedParametersLayout.ShortLtpScaleQ14Offset] =
                inputs.Shorts[inputs.LtpScaleQ14TableAt + ltpScaleIndex];
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
