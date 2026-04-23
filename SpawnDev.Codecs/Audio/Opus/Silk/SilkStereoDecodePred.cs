// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Structural port of libopus silk/stereo_decode_pred.c + the stereo predictor
// tables from silk/tables_other.c. Decodes the mid/side predictor coefficients
// and the "mid-only" flag that gate stereo SILK synthesis.
//
// Upstream Copyright (c) 2006-2011 Skype Limited. BSD 3-Clause. See NOTICE.md.

using SpawnDev.Codecs.EntropyCoders;
using static SpawnDev.Codecs.Audio.Opus.Silk.SilkMacros;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// Stereo SILK predictor decoder. Reads the two mid/side predictor coefficients
/// (one for each half of the frame) from a range decoder, plus the optional
/// mid-only flag that suppresses the side-channel data.
/// </summary>
internal static class SilkStereoDecodePred
{
    /// <summary>Number of sub-steps inside each quantization cell. Libopus <c>STEREO_QUANT_SUB_STEPS = 5</c>.</summary>
    internal const int StereoQuantSubSteps = 5;

    /// <summary>Size of the quantization table (number of cells + 1). Libopus <c>STEREO_QUANT_TAB_SIZE = 16</c>.</summary>
    internal const int StereoQuantTabSize = 16;

    /// <summary>
    /// Q13 quantization points for the stereo predictor. Ported verbatim from
    /// libopus <c>silk_stereo_pred_quant_Q13</c>.
    /// </summary>
    internal static readonly short[] StereoPredQuantQ13 =
    {
        -13732, -10050, -8266, -7526, -6500, -5000, -2950, -820,
           820,   2950,  5000,  6500,  7526,  8266, 10050, 13732,
    };

    /// <summary>
    /// 25-symbol iCDF for the joint distribution of the two predictor "cell-index
    /// high-bits" values (used as <c>5 * ix[0][2] + ix[1][2]</c>). Ported from libopus
    /// <c>silk_stereo_pred_joint_iCDF</c>.
    /// </summary>
    internal static readonly byte[] StereoPredJointIcdf =
    {
        249, 247, 246, 245, 244,
        234, 210, 202, 201, 200,
        197, 174,  82,  59,  56,
         55,  54,  46,  22,  12,
         11,  10,   9,   7,   0,
    };

    /// <summary>2-symbol iCDF for the mid-only flag. Libopus <c>silk_stereo_only_code_mid_iCDF</c>.</summary>
    internal static readonly byte[] StereoOnlyCodeMidIcdf = { 64, 0 };

    /// <summary>
    /// Decode the two mid/side predictor coefficients into Q13. The decoded
    /// <c>predQ13[0]</c> is <c>pred0 - pred1</c> (libopus pre-subtracts the second
    /// predictor to optimize the later application stage).
    /// </summary>
    /// <param name="rangeDec">Range decoder positioned at the stereo predictor block.</param>
    /// <param name="predQ13">Output: 2 Q13 predictor values.</param>
    internal static void DecodePred(OpusRangeDecoder rangeDec, Span<int> predQ13)
    {
        if (rangeDec is null) throw new ArgumentNullException(nameof(rangeDec));
        if (predQ13.Length < 2) throw new ArgumentException("predQ13 must have 2 entries.", nameof(predQ13));

        Span<int> ix0 = stackalloc int[3];
        Span<int> ix1 = stackalloc int[3];

        // Entropy-decode joint index; split into [0][2] and [1][2].
        int n = rangeDec.DecodeIcdf(StereoPredJointIcdf, 8);
        ix0[2] = n / 5;
        ix1[2] = n - 5 * ix0[2];

        ix0[0] = rangeDec.DecodeIcdf(SilkIcdfTables.Uniform3, 8);
        ix0[1] = rangeDec.DecodeIcdf(SilkIcdfTables.Uniform5, 8);
        ix1[0] = rangeDec.DecodeIcdf(SilkIcdfTables.Uniform3, 8);
        ix1[1] = rangeDec.DecodeIcdf(SilkIcdfTables.Uniform5, 8);

        // Dequantize each predictor. SILK_FIX_CONST(0.5 / 5, 16) = (int)(0.1 * 65536 + 0.5) = 6554.
        const int halfOverSubStepsQ16 = 6554;

        ix0[0] += 3 * ix0[2];
        int low0 = StereoPredQuantQ13[ix0[0]];
        int step0 = silk_SMULWB(StereoPredQuantQ13[ix0[0] + 1] - low0, halfOverSubStepsQ16);
        predQ13[0] = silk_SMLABB(low0, step0, 2 * ix0[1] + 1);

        ix1[0] += 3 * ix1[2];
        int low1 = StereoPredQuantQ13[ix1[0]];
        int step1 = silk_SMULWB(StereoPredQuantQ13[ix1[0] + 1] - low1, halfOverSubStepsQ16);
        predQ13[1] = silk_SMLABB(low1, step1, 2 * ix1[1] + 1);

        // Libopus pre-subtracts the second predictor so that application time is cheaper.
        predQ13[0] -= predQ13[1];
    }

    /// <summary>
    /// Decode the "only mid channel is coded" flag that follows the predictors.
    /// </summary>
    /// <returns>1 if only the mid channel is coded (side channel is silent), 0 otherwise.</returns>
    internal static int DecodeMidOnly(OpusRangeDecoder rangeDec)
    {
        if (rangeDec is null) throw new ArgumentNullException(nameof(rangeDec));
        return rangeDec.DecodeIcdf(StereoOnlyCodeMidIcdf, 8);
    }
}
