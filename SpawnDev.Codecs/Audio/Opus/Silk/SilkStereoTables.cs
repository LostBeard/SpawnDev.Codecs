// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// SILK stereo predictor tables (libopus silk/tables_other.c): the GPU decoder uploads them; the CPU
// SilkStereoDecodePred in SpawnDev.Codecs.References uses the same arrays.
//
// Upstream Copyright (c) 2006-2011 Skype Limited. BSD 3-Clause. See NOTICE.md.

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>SILK stereo predictor tables (libopus <c>silk_stereo_pred_quant_Q13</c>, <c>silk_stereo_pred_joint_iCDF</c>,
/// <c>silk_stereo_only_code_mid_iCDF</c>).</summary>
public static class SilkStereoTables
{
    /// <summary>libopus <c>silk_stereo_pred_quant_Q13</c>.</summary>
    public static readonly short[] StereoPredQuantQ13 =
    {
        -13732, -10050, -8266, -7526, -6500, -5000, -2950, -820,
           820,   2950,  5000,  6500,  7526,  8266, 10050, 13732,
    };

    /// <summary>libopus <c>silk_stereo_pred_joint_iCDF</c>.</summary>
    public static readonly byte[] StereoPredJointIcdf =
    {
        249, 247, 246, 245, 244,
        234, 210, 202, 201, 200,
        197, 174,  82,  59,  56,
         55,  54,  46,  22,  12,
         11,  10,   9,   7,   0,
    };

    /// <summary>libopus <c>silk_stereo_only_code_mid_iCDF</c>.</summary>
    public static readonly byte[] StereoOnlyCodeMidIcdf = { 64, 0 };
}
