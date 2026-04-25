// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 Sequence Header OBU body parser (spec sec 5.5.1).
//
// The Sequence Header carries top-of-stream parameters that don't change
// between frames: profile, max frame dimensions, color config, scaling /
// loop filter / film grain enable flags, etc.
//
// This parser surfaces the headline fields a decoder needs:
//   - seq_profile (0 / 1 / 2)
//   - reduced_still_picture_header flag
//   - max frame width/height
//   - bit_depth (8 / 10 / 12)
//   - monochrome flag, subsampling pair
//   - color range, color config
//
// More obscure fields (operating points, decoder model, tier flags,
// timing info) are parsed-and-discarded since the decode pipeline
// doesn't yet need them.

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>AV1 sequence header surfaced to the decoder.</summary>
public sealed record Av1SequenceHeader
{
    /// <summary>seq_profile: 0 (Main), 1 (High), 2 (Professional).</summary>
    public required int SeqProfile { get; init; }

    /// <summary>True for still-picture (one-shot) bitstreams.</summary>
    public required bool StillPicture { get; init; }

    /// <summary>True when most timing / OP / model fields are absent.</summary>
    public required bool ReducedStillPictureHeader { get; init; }

    /// <summary>Maximum frame width in pixels.</summary>
    public required int MaxFrameWidth { get; init; }

    /// <summary>Maximum frame height in pixels.</summary>
    public required int MaxFrameHeight { get; init; }

    /// <summary>Sample bit depth (8, 10, or 12).</summary>
    public required int BitDepth { get; init; }

    /// <summary>True for greyscale-only streams.</summary>
    public required bool Monochrome { get; init; }

    /// <summary>Chroma subsampling X (0 = full, 1 = half).</summary>
    public required int SubsamplingX { get; init; }

    /// <summary>Chroma subsampling Y.</summary>
    public required int SubsamplingY { get; init; }

    /// <summary>True for full color range, false for studio.</summary>
    public required bool ColorRangeFull { get; init; }

    /// <summary>Frame ID numbers present.</summary>
    public required bool FrameIdNumbersPresent { get; init; }

    /// <summary>Number of bits used to signal frame_id (0 if absent).</summary>
    public required int FrameIdLengthMinus7 { get; init; }

    /// <summary>Use 128x128 superblocks (true) vs 64x64 (false).</summary>
    public required bool Use128x128Superblock { get; init; }

    /// <summary>Loop filter level enable.</summary>
    public required bool EnableFilterIntra { get; init; }

    /// <summary>Intra edge filter enable.</summary>
    public required bool EnableIntraEdgeFilter { get; init; }

    // The fields below default to false / 0 / 2 so existing record
    // construction sites stay compatible. The Av1SequenceHeaderParser
    // populates them from the bitstream when present.

    /// <summary>seq_level_idx[0] (operating point 0 level index).</summary>
    public int SeqLevelIdx0 { get; init; }

    /// <summary>enable_interintra_compound flag.</summary>
    public bool EnableInterintraCompound { get; init; }

    /// <summary>enable_masked_compound flag.</summary>
    public bool EnableMaskedCompound { get; init; }

    /// <summary>enable_warped_motion flag.</summary>
    public bool EnableWarpedMotion { get; init; }

    /// <summary>enable_dual_filter flag.</summary>
    public bool EnableDualFilter { get; init; }

    /// <summary>enable_order_hint flag.</summary>
    public bool EnableOrderHint { get; init; }

    /// <summary>enable_jnt_comp flag (only signaled when EnableOrderHint=true).</summary>
    public bool EnableJntComp { get; init; }

    /// <summary>enable_ref_frame_mvs flag (only signaled when EnableOrderHint=true).</summary>
    public bool EnableRefFrameMvs { get; init; }

    /// <summary>order_hint_bits_minus_1 (only signaled when EnableOrderHint=true).</summary>
    public int OrderHintBitsMinus1 { get; init; }

    /// <summary>seq_choose_screen_content_tools flag.</summary>
    public bool SeqChooseScreenContentTools { get; init; }

    /// <summary>seq_force_screen_content_tools value (2 = SELECT, 0/1 = explicit).</summary>
    public int SeqForceScreenContentTools { get; init; }

    /// <summary>seq_choose_integer_mv flag.</summary>
    public bool SeqChooseIntegerMv { get; init; }

    /// <summary>seq_force_integer_mv value (2 = SELECT, 0/1 = explicit).</summary>
    public int SeqForceIntegerMv { get; init; }

    /// <summary>enable_superres flag.</summary>
    public bool EnableSuperres { get; init; }

    /// <summary>enable_cdef flag.</summary>
    public bool EnableCdef { get; init; }

    /// <summary>enable_restoration flag.</summary>
    public bool EnableRestoration { get; init; }

    /// <summary>color_description_present flag.</summary>
    public bool ColorDescriptionPresent { get; init; }

    /// <summary>color_primaries (8-bit value, default 2 = UNSPECIFIED).</summary>
    public int ColorPrimaries { get; init; } = 2;

    /// <summary>transfer_characteristics (8-bit value, default 2 = UNSPECIFIED).</summary>
    public int TransferCharacteristics { get; init; } = 2;

    /// <summary>matrix_coefficients (8-bit value, default 2 = UNSPECIFIED).</summary>
    public int MatrixCoefficients { get; init; } = 2;

    /// <summary>chroma_sample_position (only signaled when subX=1 and subY=1).</summary>
    public int ChromaSamplePosition { get; init; }

    /// <summary>separate_uv_deltas flag.</summary>
    public bool SeparateUvDeltas { get; init; }

    /// <summary>film_grain_params_present flag.</summary>
    public bool FilmGrainParamsPresent { get; init; }
}

/// <summary>AV1 sequence header parser.</summary>
public static class Av1SequenceHeaderParser
{
    /// <summary>
    /// Parse the Sequence Header OBU payload (everything after the OBU
    /// header bytes, length = obu.PayloadLength).
    /// </summary>
    public static Av1SequenceHeader Parse(ReadOnlySpan<byte> payload)
    {
        var br = new Av1BitReader(payload);

        int seqProfile = (int)br.ReadBits(3);
        bool stillPicture = br.ReadFlag();
        // Per AV1 spec 5.5.1: reduced_still_picture_header is read
        // UNCONDITIONALLY (not gated on still_picture). Spec
        // constraint: reduced=1 implies still=1, but the bit is
        // always present in the bitstream.
        bool reducedStill = br.ReadFlag();

        bool timingInfoPresent = false;
        bool decoderModelInfoPresent = false;
        bool initialDisplayDelayPresent = false;

        int seqLevelIdx0 = 0;
        if (reducedStill)
        {
            // OP count = 1; operating_point_idc = 0; per spec, just one
            // f(5) read for seq_level_idx[0].
            seqLevelIdx0 = (int)br.ReadBits(5);
        }
        else
        {
            timingInfoPresent = br.ReadFlag();
            if (timingInfoPresent)
            {
                br.ReadBits(32); // num_units_in_display_tick
                br.ReadBits(32); // time_scale
                bool equalPicInterval = br.ReadFlag();
                if (equalPicInterval) ReadUvlc(ref br); // num_ticks_per_picture_minus_1
                decoderModelInfoPresent = br.ReadFlag();
                if (decoderModelInfoPresent) ReadDecoderModelInfo(ref br);
            }

            initialDisplayDelayPresent = br.ReadFlag();
            int opCntMinus1 = (int)br.ReadBits(5);
            for (int i = 0; i <= opCntMinus1; i++)
            {
                br.ReadBits(12); // operating_point_idc[i]
                int level = (int)br.ReadBits(5);
                if (i == 0) seqLevelIdx0 = level;
                if (level > 7) br.ReadBits(1); // seq_tier[i]
                if (decoderModelInfoPresent)
                {
                    bool present = br.ReadFlag();
                    if (present) br.ReadBits(20); // operating_parameters_info[i]
                }
                if (initialDisplayDelayPresent)
                {
                    bool delayPresent = br.ReadFlag();
                    if (delayPresent) br.ReadBits(4); // initial_display_delay_minus_1[i]
                }
            }
        }

        int frameWidthBitsMinus1 = (int)br.ReadBits(4);
        int frameHeightBitsMinus1 = (int)br.ReadBits(4);
        int maxFrameWidthMinus1 = (int)br.ReadBits(frameWidthBitsMinus1 + 1);
        int maxFrameHeightMinus1 = (int)br.ReadBits(frameHeightBitsMinus1 + 1);

        bool frameIdNumbersPresent = false;
        int frameIdLengthMinus7 = 0;
        int deltaFrameIdLengthMinus2 = 0;
        if (!reducedStill)
        {
            frameIdNumbersPresent = br.ReadFlag();
            if (frameIdNumbersPresent)
            {
                deltaFrameIdLengthMinus2 = (int)br.ReadBits(4);
                frameIdLengthMinus7 = (int)br.ReadBits(3);
            }
        }

        bool use128x128Superblock = br.ReadFlag();
        bool enableFilterIntra = br.ReadFlag();
        bool enableIntraEdgeFilter = br.ReadFlag();

        bool enableInterintraCompound = false;
        bool enableMaskedCompound = false;
        bool enableWarpedMotion = false;
        bool enableDualFilter = false;
        bool enableOrderHint = false;
        bool enableJntComp = false;
        bool enableRefFrameMvs = false;
        bool seqChooseScreenContentTools = false;
        int seqForceScreenContentTools = 0;
        bool seqChooseIntegerMv = false;
        int seqForceIntegerMv = 0;
        int orderHintBitsMinus1 = 0;
        if (!reducedStill)
        {
            enableInterintraCompound = br.ReadFlag();
            enableMaskedCompound = br.ReadFlag();
            enableWarpedMotion = br.ReadFlag();
            enableDualFilter = br.ReadFlag();
            enableOrderHint = br.ReadFlag();
            if (enableOrderHint)
            {
                enableJntComp = br.ReadFlag();
                enableRefFrameMvs = br.ReadFlag();
            }
            seqChooseScreenContentTools = br.ReadFlag();
            if (seqChooseScreenContentTools) seqForceScreenContentTools = 2; // SELECT_SCREEN_CONTENT_TOOLS
            else seqForceScreenContentTools = (int)br.ReadBits(1);
            if (seqForceScreenContentTools > 0)
            {
                seqChooseIntegerMv = br.ReadFlag();
                if (!seqChooseIntegerMv) seqForceIntegerMv = (int)br.ReadBits(1);
                else seqForceIntegerMv = 2;
            }
            if (enableOrderHint)
                orderHintBitsMinus1 = (int)br.ReadBits(3);
        }
        bool enableSuperres = br.ReadFlag();
        bool enableCdef = br.ReadFlag();
        bool enableRestoration = br.ReadFlag();

        // color_config:
        bool highBitDepth = br.ReadFlag();
        int bitDepth = 8;
        if (seqProfile == 2 && highBitDepth)
        {
            bool twelveBit = br.ReadFlag();
            bitDepth = twelveBit ? 12 : 10;
        }
        else if (highBitDepth)
        {
            bitDepth = 10;
        }
        bool monochrome = false;
        if (seqProfile != 1) monochrome = br.ReadFlag();
        bool colorDescPresent = br.ReadFlag();
        int colorPrimaries = 2;
        int transferChars = 2;
        int matrixCoefs = 2;
        if (colorDescPresent)
        {
            colorPrimaries = (int)br.ReadBits(8);
            transferChars = (int)br.ReadBits(8);
            matrixCoefs = (int)br.ReadBits(8);
        }
        bool colorRangeFull;
        int subX = 1, subY = 1;
        int chromaSamplePosition = 0;
        bool separateUvDeltas = false;
        if (monochrome)
        {
            colorRangeFull = br.ReadFlag();
            subX = 1;
            subY = 1;
        }
        else
        {
            colorRangeFull = br.ReadFlag();
            if (seqProfile == 0) { subX = 1; subY = 1; }
            else if (seqProfile == 1) { subX = 0; subY = 0; }
            else
            {
                if (bitDepth == 12)
                {
                    subX = (int)br.ReadBits(1);
                    if (subX != 0) subY = (int)br.ReadBits(1);
                    else subY = 0;
                }
                else
                {
                    subX = 1;
                    subY = 0;
                }
            }
            if (subX != 0 && subY != 0) chromaSamplePosition = (int)br.ReadBits(2);
            separateUvDeltas = br.ReadFlag();
        }
        bool filmGrainParamsPresent = br.BitsRemaining > 0 && br.ReadFlag();

        return new Av1SequenceHeader
        {
            SeqProfile = seqProfile,
            StillPicture = stillPicture,
            ReducedStillPictureHeader = reducedStill,
            MaxFrameWidth = maxFrameWidthMinus1 + 1,
            MaxFrameHeight = maxFrameHeightMinus1 + 1,
            BitDepth = bitDepth,
            Monochrome = monochrome,
            SubsamplingX = subX,
            SubsamplingY = subY,
            ColorRangeFull = colorRangeFull,
            FrameIdNumbersPresent = frameIdNumbersPresent,
            FrameIdLengthMinus7 = frameIdLengthMinus7,
            Use128x128Superblock = use128x128Superblock,
            EnableFilterIntra = enableFilterIntra,
            EnableIntraEdgeFilter = enableIntraEdgeFilter,
            SeqLevelIdx0 = seqLevelIdx0,
            EnableInterintraCompound = enableInterintraCompound,
            EnableMaskedCompound = enableMaskedCompound,
            EnableWarpedMotion = enableWarpedMotion,
            EnableDualFilter = enableDualFilter,
            EnableOrderHint = enableOrderHint,
            EnableJntComp = enableJntComp,
            EnableRefFrameMvs = enableRefFrameMvs,
            OrderHintBitsMinus1 = orderHintBitsMinus1,
            SeqChooseScreenContentTools = seqChooseScreenContentTools,
            SeqForceScreenContentTools = seqForceScreenContentTools,
            SeqChooseIntegerMv = seqChooseIntegerMv,
            SeqForceIntegerMv = seqForceIntegerMv,
            EnableSuperres = enableSuperres,
            EnableCdef = enableCdef,
            EnableRestoration = enableRestoration,
            ColorDescriptionPresent = colorDescPresent,
            ColorPrimaries = colorPrimaries,
            TransferCharacteristics = transferChars,
            MatrixCoefficients = matrixCoefs,
            ChromaSamplePosition = chromaSamplePosition,
            SeparateUvDeltas = separateUvDeltas,
            FilmGrainParamsPresent = filmGrainParamsPresent,
        };
    }

    private static uint ReadUvlc(ref Av1BitReader br)
    {
        int leading = 0;
        while (leading < 32 && br.ReadBits(1) == 0) leading++;
        if (leading == 32) return 0xFFFFFFFFu;
        if (leading == 0) return 0;
        uint val = (uint)br.ReadBits(leading);
        return val + (1u << leading) - 1;
    }

    private static void ReadDecoderModelInfo(ref Av1BitReader br)
    {
        br.ReadBits(5); // buffer_delay_length_minus_1
        br.ReadBits(32); // num_units_in_decoding_tick
        br.ReadBits(5);  // buffer_removal_time_length_minus_1
        br.ReadBits(5);  // frame_presentation_time_length_minus_1
    }
}

/// <summary>
/// AV1 bit reader. Same MSB-first semantics as VP9's reader but as a
/// public ref struct so internal use across the AV1 codecs is fine.
/// </summary>
public ref struct Av1BitReader
{
    private readonly ReadOnlySpan<byte> _data;
    private int _bytePos;
    private int _bitPos;

    /// <summary>Construct over a payload span.</summary>
    public Av1BitReader(ReadOnlySpan<byte> data)
    {
        _data = data;
        _bytePos = 0;
        _bitPos = 0;
    }

    /// <summary>Bits consumed so far.</summary>
    public int Position => _bytePos * 8 + _bitPos;

    /// <summary>Bits remaining to read.</summary>
    public int BitsRemaining => (_data.Length - _bytePos) * 8 - _bitPos;

    /// <summary>Read the next <paramref name="nBits"/> bits as an unsigned integer (0..32).</summary>
    public uint ReadBits(int nBits)
    {
        if (nBits < 0 || nBits > 32)
            throw new ArgumentOutOfRangeException(nameof(nBits));
        if (nBits == 0) return 0;
        if (BitsRemaining < nBits)
            throw new InvalidDataException(
                $"AV1 bit reader: only {BitsRemaining} bits left, need {nBits}.");

        uint value = 0;
        int bitsLeft = nBits;
        while (bitsLeft > 0)
        {
            int availInByte = 8 - _bitPos;
            int take = Math.Min(availInByte, bitsLeft);
            int shift = availInByte - take;
            uint chunk = ((uint)_data[_bytePos] >> shift) & ((1u << take) - 1);
            value = (value << take) | chunk;
            _bitPos += take;
            if (_bitPos == 8)
            {
                _bytePos++;
                _bitPos = 0;
            }
            bitsLeft -= take;
        }
        return value;
    }

    /// <summary>Read a single bit as a flag.</summary>
    public bool ReadFlag() => ReadBits(1) == 1;
}
