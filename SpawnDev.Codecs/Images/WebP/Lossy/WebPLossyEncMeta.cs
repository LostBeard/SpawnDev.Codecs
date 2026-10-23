// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Shared layout of the lossy WebP encoder's device buffers (WebPLossyEncoderGpu / WebPLossyEncodeKernels). Every slot
// has a name so a kernel and the host cannot disagree about it. The algorithms are libwebp 1.6.0's (see
// SpawnDev.Codecs.References Vp8LossyEncoderReference, the byte-identical CPU port these kernels are checked against).

namespace SpawnDev.Codecs.Images.WebP.Lossy;

/// <summary>Slots of the encoder's int metadata block.</summary>
internal static class WebPLossyEncMeta
{
    // ---- written by the host (config + geometry) ----
    public const int Width = 0;
    public const int Height = 1;
    public const int MbW = 2;
    public const int MbH = 3;
    public const int Method = 4;
    public const int QualityX100 = 5;        // config quality * 100 (integer use: FastMBAnalyze threshold)
    public const int SnsStrength = 6;
    public const int FilterStrength = 7;
    public const int FilterSharpnessCfg = 8;
    public const int FilterType = 9;
    public const int SegmentsCfg = 10;
    public const int ErrorDiffusion = 11;    // 1 when quality <= 98 (top_derr allocated)
    public const int MaxI4HeaderBits = 12;
    public const int MbHeaderLimitLo = 13;   // mb_header_limit (64-bit) low / high words
    public const int MbHeaderLimitHi = 14;
    public const int TokenCapacity = 15;     // ushort slots of the token buffer
    public const int Partition0Capacity = 16;
    public const int Token1Capacity = 17;    // bytes of the token partition scratch
    public const int OutCapacity = 18;       // bytes of the caller's output

    // ---- written by the kernels ----
    public const int Error = 24;             // WebPLossyEncError
    public const int Alpha = 25;             // enc->alpha (mean macroblock susceptibility)
    public const int UVAlpha = 26;           // enc->uv_alpha
    public const int NumSegments = 27;
    public const int UpdateMap = 28;
    public const int SegmentHdrSize = 29;
    public const int BaseQuant = 30;
    public const int DqY1Dc = 31, DqY2Dc = 32, DqY2Ac = 33, DqUVDc = 34, DqUVAc = 35;
    public const int FilterSimple = 36;
    public const int FilterLevel = 37;
    public const int FilterSharpness = 38;
    public const int Profile = 39;
    public const int ProbaDirty = 40;
    public const int UseSkipProba = 41;
    public const int SkipProba = 42;
    public const int NbSkip = 43;
    public const int TokenCount = 44;
    public const int MbCountDown = 45;       // VP8EncTokenLoop `cnt` (probability refresh countdown)
    public const int MaxCount = 46;
    public const int SizeP0Lo = 47, SizeP0Hi = 48;
    public const int SegProba0 = 49, SegProba1 = 50, SegProba2 = 51;
    public const int Partition0Size = 52;
    public const int Token1Size = 53;
    public const int OutLength = 54;

    // ---- buffer offsets (host): the kernels take packed buffers and SubView them by these ----
    public const int YuvUOff = 56, YuvVOff = 57;                       // in the yuv byte buffer (Y at 0)
    public const int WorkYTopOff = 58, WorkScratchOff = 59;            // in the work byte buffer (preds at 0)
    public const int StNzOff = 60, StTopDerrOff = 61, StCoeffsOff = 62, StStatsOff = 63;
    public const int StLevelCostOff = 64, StScratchOff = 65;           // in the state int buffer
    public const int PassMaxI4HeaderBits = 66;                         // max_i4_header_bits of the current pass
    public const int HasAlpha = 67;                                    // 1 when any alpha sample is not 255 (kernel 2)
    public const int AlphaPlanePresent = 68;                           // host: 0 for YUV input (no alpha plane to scan)

    // ---- per segment (4 x SegStride ints) ----
    public const int Seg = 80;
    public const int SegStride = 96;
    // fields inside a segment record
    public const int SAlpha = 0, SBeta = 1, SQuant = 2, SFStrength = 3, SMaxEdge = 4, SMinDisto = 5;
    public const int SLambdaI16 = 6, SLambdaI4 = 7, SLambdaUV = 8, SLambdaMode = 9, STLambda = 10;
    public const int SI4PenaltyLo = 11, SI4PenaltyHi = 12;
    public const int SQY1 = 16, SQY2 = 18, SQUV = 20;            // q[0], q[1] of each matrix (q[2..15] = q[1])
    public const int SIQY1 = 22, SIQY2 = 24, SIQUV = 26;         // iq[0], iq[1]
    public const int SBiasY1 = 28, SBiasY2 = 30, SBiasUV = 32;   // bias[0], bias[1]
    public const int SZtY1 = 34, SZtY2 = 36, SZtUV = 38;         // zthresh[0], zthresh[1]
    public const int SSharpY1 = 40;                              // sharpen[16] of y1 (the others are 0)

    public const int Size = Seg + 4 * SegStride;   // 464

    /// <summary>Per-macroblock record (ints): type, uv mode, skip, segment, alpha.</summary>
    public const int MbType = 0, MbUVMode = 1, MbSkip = 2, MbSegment = 3, MbAlpha = 4, MbUVAlphaRaw = 5, MbTransparent = 6;
    public const int MbStride = 8;
}

/// <summary>Errors the encoder kernels can raise.</summary>
internal enum WebPLossyEncError
{
    None = 0,
    TokenBufferTooSmall = 1,
    Partition0TooBig = 2,
    Partition0ScratchTooSmall = 3,
    TokenScratchTooSmall = 4,
    OutputTooSmall = 5,
}
