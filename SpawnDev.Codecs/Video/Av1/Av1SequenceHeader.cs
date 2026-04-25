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

        if (reducedStill)
        {
            // OP count = 1; operating_point_idc = 0; per spec, just one
            // f(5) read for seq_level_idx[0].
            br.ReadBits(5); // seq_level_idx[0]
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

        // The rest of the SH parses additional flags (interintra, masked
        // compound, warped motion, dual filter, order hint, jnt_comp,
        // ref frame mvs, screen content tools, force integer mv, OH bits,
        // superres, cdef, restoration, color config, film grain). For now
        // we fast-forward to color_config since the body up to here is
        // enough to populate the headline fields.
        if (!reducedStill)
        {
            br.ReadBits(1); // enable_interintra_compound
            br.ReadBits(1); // enable_masked_compound
            br.ReadBits(1); // enable_warped_motion
            br.ReadBits(1); // enable_dual_filter
            bool enableOrderHint = br.ReadFlag();
            if (enableOrderHint)
            {
                br.ReadBits(1); // enable_jnt_comp
                br.ReadBits(1); // enable_ref_frame_mvs
            }
            bool seqChooseScreenContentTools = br.ReadFlag();
            int seqForceScreenContentTools;
            if (seqChooseScreenContentTools) seqForceScreenContentTools = 2; // SELECT_SCREEN_CONTENT_TOOLS
            else seqForceScreenContentTools = (int)br.ReadBits(1);
            if (seqForceScreenContentTools > 0)
            {
                bool seqChooseIntegerMv = br.ReadFlag();
                if (!seqChooseIntegerMv) br.ReadBits(1); // seq_force_integer_mv
            }
            if (enableOrderHint)
                br.ReadBits(3); // order_hint_bits_minus_1
        }
        br.ReadBits(1); // enable_superres
        br.ReadBits(1); // enable_cdef
        br.ReadBits(1); // enable_restoration

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
        if (colorDescPresent)
        {
            br.ReadBits(8); // color_primaries
            br.ReadBits(8); // transfer_characteristics
            br.ReadBits(8); // matrix_coefficients
        }
        bool colorRangeFull;
        int subX = 1, subY = 1;
        if (monochrome)
        {
            colorRangeFull = br.ReadFlag();
            subX = 1;
            subY = 1;
        }
        else
        {
            // sRGB-style: profile=1 OR (profile=2 & bit=12) gates sub flags
            bool srgb = colorDescPresent && br.BitsRemaining > 0 && false; // placeholder
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
            if (subX != 0 && subY != 0) br.ReadBits(2); // chroma_sample_position
            br.ReadBits(1); // separate_uv_deltas
        }
        // film_grain_params_present, then trailing bits - not parsed.

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
