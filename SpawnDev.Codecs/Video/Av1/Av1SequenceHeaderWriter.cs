// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 Sequence Header writer - emits a minimal valid sequence_header_obu
// payload from a small set of encoder-side parameters. Intentionally
// avoids the optional / advanced fields (operating points beyond op0,
// timing info, decoder model, screen content tools, order hint, film
// grain) so the writer's first job is a tight, validated emit path
// suitable for an AV1 ENCODER's bitstream output.
//
// What the writer emits (spec sec 5.5.1, simplified):
//   seq_profile                    f(3)
//   still_picture = 0              f(1)
//   reduced_still_picture_header=0 f(1)
//   timing_info_present_flag = 0   f(1)
//   initial_display_delay_present_flag = 0    f(1)
//   operating_points_cnt_minus_1 = 0          f(5)
//   operating_point_idc[0] = 0     f(12)
//   seq_level_idx[0]               f(5)        (level <= 7, no tier bit)
//   frame_width_bits_minus_1       f(4)
//   frame_height_bits_minus_1      f(4)
//   max_frame_width_minus_1        f(frame_width_bits)
//   max_frame_height_minus_1       f(frame_height_bits)
//   frame_id_numbers_present = 0   f(1)
//   use_128x128_superblock = 0     f(1)
//   enable_filter_intra = 0        f(1)
//   enable_intra_edge_filter = 0   f(1)
//   enable_interintra = 0          f(1)
//   enable_masked_compound = 0     f(1)
//   enable_warped_motion = 0       f(1)
//   enable_dual_filter = 0         f(1)
//   enable_order_hint = 0          f(1)
//   seq_choose_screen_content_tools = 1     f(1)  -> defaults to SELECT
//   enable_superres = 0            f(1)
//   enable_cdef = 0                f(1)
//   enable_restoration = 0         f(1)
//   color_config:
//     high_bitdepth                f(1)
//     [twelve_bit if profile==2 && high]   f(1)
//     monochrome (if profile != 1) f(1)
//     color_description_present=0  f(1)
//     color_range                  f(1)
//     [subsampling_x/y if profile==2 && bit==12]    f(1)+f(1)
//     [chroma_sample_position if subX & subY]       f(2)  // SPEC says 0 (UNKNOWN)
//     separate_uv_deltas = 0       f(1)
//   film_grain_params_present = 0  f(1)
//   trailing_bits (1 + zero pad to byte)
//
// The writer pairs with Av1SequenceHeaderParser - the round-trip test
// confirms every field we emit is read back correctly.

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>Caller-facing config for emitting an AV1 SequenceHeader OBU.</summary>
public sealed record Av1SequenceHeaderConfig
{
    /// <summary>seq_profile (0 = Main 8/10-bit 4:2:0, 1 = High 4:4:4, 2 = Pro 4:2:2 / 12-bit).</summary>
    public required int SeqProfile { get; init; }

    /// <summary>seq_level_idx[0]. Common values: 0 (level 2.0) up to 7 (level 3.3).</summary>
    public int SeqLevelIdx0 { get; init; } = 0;

    /// <summary>Maximum frame width in pixels (1..2^16).</summary>
    public required int MaxFrameWidth { get; init; }

    /// <summary>Maximum frame height in pixels (1..2^16).</summary>
    public required int MaxFrameHeight { get; init; }

    /// <summary>Sample bit depth: 8, 10, or 12 (12 only valid with profile=2).</summary>
    public required int BitDepth { get; init; }

    /// <summary>True for monochrome (Y-only) streams.</summary>
    public bool Monochrome { get; init; }

    /// <summary>Chroma subsampling X. Profile 0/2 -> 1; profile 1 -> 0.</summary>
    public int SubsamplingX { get; init; } = 1;

    /// <summary>Chroma subsampling Y. Profile 0 -> 1; profile 1 -> 0; profile 2 -> see spec.</summary>
    public int SubsamplingY { get; init; } = 1;

    /// <summary>True for full-range pixel values, false for studio range.</summary>
    public bool ColorRangeFull { get; init; }

    /// <summary>use_128x128_superblock flag.</summary>
    public bool Use128x128Superblock { get; init; }

    /// <summary>enable_filter_intra flag.</summary>
    public bool EnableFilterIntra { get; init; }

    /// <summary>enable_intra_edge_filter flag.</summary>
    public bool EnableIntraEdgeFilter { get; init; }

    /// <summary>enable_interintra_compound flag.</summary>
    public bool EnableInterintraCompound { get; init; }

    /// <summary>enable_masked_compound flag.</summary>
    public bool EnableMaskedCompound { get; init; }

    /// <summary>enable_warped_motion flag.</summary>
    public bool EnableWarpedMotion { get; init; }

    /// <summary>enable_dual_filter flag.</summary>
    public bool EnableDualFilter { get; init; }

    /// <summary>enable_order_hint flag (gates jnt_comp / ref_frame_mvs / order_hint_bits).</summary>
    public bool EnableOrderHint { get; init; }

    /// <summary>enable_jnt_comp flag. Only emitted when EnableOrderHint=true.</summary>
    public bool EnableJntComp { get; init; }

    /// <summary>enable_ref_frame_mvs flag. Only emitted when EnableOrderHint=true.</summary>
    public bool EnableRefFrameMvs { get; init; }

    /// <summary>order_hint_bits_minus_1, f(3). Only emitted when EnableOrderHint=true.</summary>
    public int OrderHintBitsMinus1 { get; init; }

    /// <summary>seq_choose_screen_content_tools flag (true picks SELECT, false uses ForceScreenContentTools).</summary>
    public bool SeqChooseScreenContentTools { get; init; } = true;

    /// <summary>seq_force_screen_content_tools, f(1). Only emitted when SeqChooseScreenContentTools=false.</summary>
    public int SeqForceScreenContentTools { get; init; }

    /// <summary>seq_choose_integer_mv flag. Only emitted when seq_force_screen_content_tools > 0.</summary>
    public bool SeqChooseIntegerMv { get; init; } = true;

    /// <summary>seq_force_integer_mv, f(1). Only emitted when SeqChooseIntegerMv=false.</summary>
    public int SeqForceIntegerMv { get; init; }

    /// <summary>enable_superres flag.</summary>
    public bool EnableSuperres { get; init; }

    /// <summary>enable_cdef flag.</summary>
    public bool EnableCdef { get; init; }

    /// <summary>enable_restoration flag.</summary>
    public bool EnableRestoration { get; init; }

    /// <summary>color_description_present flag (gates color_primaries / transfer / matrix bytes).</summary>
    public bool ColorDescriptionPresent { get; init; }

    /// <summary>color_primaries, f(8). Only emitted when ColorDescriptionPresent=true.</summary>
    public int ColorPrimaries { get; init; } = 2;

    /// <summary>transfer_characteristics, f(8). Only emitted when ColorDescriptionPresent=true.</summary>
    public int TransferCharacteristics { get; init; } = 2;

    /// <summary>matrix_coefficients, f(8). Only emitted when ColorDescriptionPresent=true.</summary>
    public int MatrixCoefficients { get; init; } = 2;

    /// <summary>chroma_sample_position, f(2). Only emitted when subX=1 && subY=1.</summary>
    public int ChromaSamplePosition { get; init; } = 0;

    /// <summary>separate_uv_deltas flag.</summary>
    public bool SeparateUvDeltas { get; init; }

    /// <summary>film_grain_params_present flag.</summary>
    public bool FilmGrainParamsPresent { get; init; }

    /// <summary>
    /// Build a writer config from a parsed <see cref="Av1SequenceHeader"/>.
    /// Round-trip helper: parse(SH) -> ToConfig -> EmitPayload should
    /// produce byte-identical output to the source SH for any AV1 stream
    /// our parser handles.
    /// </summary>
    public static Av1SequenceHeaderConfig FromHeader(Av1SequenceHeader sh)
    {
        ArgumentNullException.ThrowIfNull(sh);
        return new Av1SequenceHeaderConfig
        {
            SeqProfile = sh.SeqProfile,
            SeqLevelIdx0 = sh.SeqLevelIdx0,
            MaxFrameWidth = sh.MaxFrameWidth,
            MaxFrameHeight = sh.MaxFrameHeight,
            BitDepth = sh.BitDepth,
            Monochrome = sh.Monochrome,
            SubsamplingX = sh.SubsamplingX,
            SubsamplingY = sh.SubsamplingY,
            ColorRangeFull = sh.ColorRangeFull,
            Use128x128Superblock = sh.Use128x128Superblock,
            EnableFilterIntra = sh.EnableFilterIntra,
            EnableIntraEdgeFilter = sh.EnableIntraEdgeFilter,
            EnableInterintraCompound = sh.EnableInterintraCompound,
            EnableMaskedCompound = sh.EnableMaskedCompound,
            EnableWarpedMotion = sh.EnableWarpedMotion,
            EnableDualFilter = sh.EnableDualFilter,
            EnableOrderHint = sh.EnableOrderHint,
            EnableJntComp = sh.EnableJntComp,
            EnableRefFrameMvs = sh.EnableRefFrameMvs,
            OrderHintBitsMinus1 = sh.OrderHintBitsMinus1,
            SeqChooseScreenContentTools = sh.SeqChooseScreenContentTools,
            SeqForceScreenContentTools = sh.SeqForceScreenContentTools,
            SeqChooseIntegerMv = sh.SeqChooseIntegerMv,
            SeqForceIntegerMv = sh.SeqForceIntegerMv,
            EnableSuperres = sh.EnableSuperres,
            EnableCdef = sh.EnableCdef,
            EnableRestoration = sh.EnableRestoration,
            ColorDescriptionPresent = sh.ColorDescriptionPresent,
            ColorPrimaries = sh.ColorPrimaries,
            TransferCharacteristics = sh.TransferCharacteristics,
            MatrixCoefficients = sh.MatrixCoefficients,
            ChromaSamplePosition = sh.ChromaSamplePosition,
            SeparateUvDeltas = sh.SeparateUvDeltas,
            FilmGrainParamsPresent = sh.FilmGrainParamsPresent,
        };
    }
}

/// <summary>AV1 SequenceHeader OBU payload writer.</summary>
public static class Av1SequenceHeaderWriter
{
    /// <summary>
    /// Emit the SequenceHeader OBU PAYLOAD (i.e. the bytes that go AFTER
    /// the OBU header / size prefix). Wrap with the
    /// <see cref="Av1ObuWriter.EmitObu(Av1ObuType, System.ReadOnlySpan{byte}, bool, int, int, bool?)"/>
    /// overload to produce a full OBU.
    /// </summary>
    public static byte[] EmitPayload(Av1SequenceHeaderConfig cfg)
    {
        ValidateConfig(cfg);

        var bw = new Av1BitWriter();
        bw.WriteBits(cfg.SeqProfile, 3);
        bw.WriteFlag(false); // still_picture
        bw.WriteFlag(false); // reduced_still_picture_header

        bw.WriteFlag(false); // timing_info_present_flag
        bw.WriteFlag(false); // initial_display_delay_present_flag
        bw.WriteBits(0, 5);  // operating_points_cnt_minus_1

        bw.WriteBits(0, 12); // operating_point_idc[0]
        bw.WriteBits(cfg.SeqLevelIdx0, 5);
        // No tier bit because seq_level_idx <= 7.

        int wBits = MinBitsForValue(cfg.MaxFrameWidth - 1);
        int hBits = MinBitsForValue(cfg.MaxFrameHeight - 1);
        bw.WriteBits(wBits - 1, 4); // frame_width_bits_minus_1
        bw.WriteBits(hBits - 1, 4); // frame_height_bits_minus_1
        bw.WriteBits(cfg.MaxFrameWidth - 1, wBits);
        bw.WriteBits(cfg.MaxFrameHeight - 1, hBits);

        bw.WriteFlag(false); // frame_id_numbers_present_flag
        bw.WriteFlag(cfg.Use128x128Superblock);
        bw.WriteFlag(cfg.EnableFilterIntra);
        bw.WriteFlag(cfg.EnableIntraEdgeFilter);

        bw.WriteFlag(cfg.EnableInterintraCompound);
        bw.WriteFlag(cfg.EnableMaskedCompound);
        bw.WriteFlag(cfg.EnableWarpedMotion);
        bw.WriteFlag(cfg.EnableDualFilter);
        bw.WriteFlag(cfg.EnableOrderHint);
        if (cfg.EnableOrderHint)
        {
            bw.WriteFlag(cfg.EnableJntComp);
            bw.WriteFlag(cfg.EnableRefFrameMvs);
        }
        bw.WriteFlag(cfg.SeqChooseScreenContentTools);
        int sccForce;
        if (cfg.SeqChooseScreenContentTools)
        {
            sccForce = 2; // SELECT
        }
        else
        {
            sccForce = cfg.SeqForceScreenContentTools;
            if ((uint)sccForce > 1) throw new ArgumentOutOfRangeException(nameof(cfg.SeqForceScreenContentTools));
            bw.WriteBits(sccForce, 1);
        }
        if (sccForce > 0)
        {
            bw.WriteFlag(cfg.SeqChooseIntegerMv);
            if (!cfg.SeqChooseIntegerMv)
            {
                if ((uint)cfg.SeqForceIntegerMv > 1) throw new ArgumentOutOfRangeException(nameof(cfg.SeqForceIntegerMv));
                bw.WriteBits(cfg.SeqForceIntegerMv, 1);
            }
        }
        if (cfg.EnableOrderHint)
        {
            if ((uint)cfg.OrderHintBitsMinus1 > 7) throw new ArgumentOutOfRangeException(nameof(cfg.OrderHintBitsMinus1));
            bw.WriteBits(cfg.OrderHintBitsMinus1, 3);
        }

        bw.WriteFlag(cfg.EnableSuperres);
        bw.WriteFlag(cfg.EnableCdef);
        bw.WriteFlag(cfg.EnableRestoration);

        // color_config
        bool highBitDepth = cfg.BitDepth >= 10;
        bw.WriteFlag(highBitDepth);
        if (cfg.SeqProfile == 2 && highBitDepth)
            bw.WriteFlag(cfg.BitDepth == 12);
        if (cfg.SeqProfile != 1)
            bw.WriteFlag(cfg.Monochrome);
        bw.WriteFlag(cfg.ColorDescriptionPresent);
        if (cfg.ColorDescriptionPresent)
        {
            if ((uint)cfg.ColorPrimaries > 255) throw new ArgumentOutOfRangeException(nameof(cfg.ColorPrimaries));
            if ((uint)cfg.TransferCharacteristics > 255) throw new ArgumentOutOfRangeException(nameof(cfg.TransferCharacteristics));
            if ((uint)cfg.MatrixCoefficients > 255) throw new ArgumentOutOfRangeException(nameof(cfg.MatrixCoefficients));
            bw.WriteBits(cfg.ColorPrimaries, 8);
            bw.WriteBits(cfg.TransferCharacteristics, 8);
            bw.WriteBits(cfg.MatrixCoefficients, 8);
        }
        if (cfg.Monochrome)
        {
            bw.WriteFlag(cfg.ColorRangeFull);
            // separate_uv_deltas is not emitted for monochrome.
        }
        else
        {
            bw.WriteFlag(cfg.ColorRangeFull);
            // Profile 0: subsampling fixed to (1,1) -> not written.
            // Profile 1: subsampling fixed to (0,0) -> not written.
            // Profile 2: depends on bit depth.
            if (cfg.SeqProfile == 2 && cfg.BitDepth == 12)
            {
                bw.WriteBits(cfg.SubsamplingX, 1);
                if (cfg.SubsamplingX != 0)
                    bw.WriteBits(cfg.SubsamplingY, 1);
            }
            // chroma_sample_position only emitted when subX==1 && subY==1
            // (profile 0 OR profile 2 with both bits set).
            int effSubX = cfg.SeqProfile == 0 ? 1 : (cfg.SeqProfile == 1 ? 0 : cfg.SubsamplingX);
            int effSubY = cfg.SeqProfile == 0 ? 1 : (cfg.SeqProfile == 1 ? 0 : cfg.SubsamplingY);
            if (effSubX != 0 && effSubY != 0)
            {
                if ((uint)cfg.ChromaSamplePosition > 3) throw new ArgumentOutOfRangeException(nameof(cfg.ChromaSamplePosition));
                bw.WriteBits(cfg.ChromaSamplePosition, 2);
            }
            bw.WriteFlag(cfg.SeparateUvDeltas);
        }

        bw.WriteFlag(cfg.FilmGrainParamsPresent);

        bw.WriteTrailingBits();
        return bw.ToArray();
    }

    private static void ValidateConfig(Av1SequenceHeaderConfig cfg)
    {
        if (cfg.SeqProfile < 0 || cfg.SeqProfile > 2)
            throw new ArgumentOutOfRangeException(nameof(cfg.SeqProfile));
        if (cfg.MaxFrameWidth < 1 || cfg.MaxFrameWidth > 65536)
            throw new ArgumentOutOfRangeException(nameof(cfg.MaxFrameWidth));
        if (cfg.MaxFrameHeight < 1 || cfg.MaxFrameHeight > 65536)
            throw new ArgumentOutOfRangeException(nameof(cfg.MaxFrameHeight));
        if (cfg.BitDepth != 8 && cfg.BitDepth != 10 && cfg.BitDepth != 12)
            throw new ArgumentOutOfRangeException(nameof(cfg.BitDepth));
        if (cfg.BitDepth == 12 && cfg.SeqProfile != 2)
            throw new ArgumentException("12-bit only valid with profile 2.", nameof(cfg));
        if ((uint)cfg.SeqLevelIdx0 > 7)
            throw new ArgumentOutOfRangeException(nameof(cfg.SeqLevelIdx0),
                "Writer only supports seq_level_idx <= 7 (no tier bit emitted).");
    }

    private static int MinBitsForValue(int value)
    {
        if (value < 0) throw new ArgumentOutOfRangeException(nameof(value));
        if (value == 0) return 1;
        int bits = 0;
        while (value > 0) { bits++; value >>= 1; }
        return bits;
    }
}

/// <summary>
/// AV1 bit writer - MSB-first packing matching <see cref="Av1BitReader"/>.
/// </summary>
public sealed class Av1BitWriter
{
    private readonly List<byte> _bytes = new();
    private int _curByte;
    private int _bitsInCur;

    /// <summary>Total bits written so far.</summary>
    public int BitPosition => _bytes.Count * 8 + _bitsInCur;

    /// <summary>Write an unsigned integer in <paramref name="nBits"/> bits, MSB first.</summary>
    public void WriteBits(int value, int nBits)
    {
        if (nBits < 0 || nBits > 32)
            throw new ArgumentOutOfRangeException(nameof(nBits));
        if (nBits == 0) return;
        if (nBits < 32 && ((uint)value >> nBits) != 0)
            throw new ArgumentException(
                $"Value 0x{value:X} does not fit in {nBits} bits.", nameof(value));

        for (int i = nBits - 1; i >= 0; i--)
        {
            int bit = (value >> i) & 1;
            _curByte = (_curByte << 1) | bit;
            _bitsInCur++;
            if (_bitsInCur == 8)
            {
                _bytes.Add((byte)_curByte);
                _curByte = 0;
                _bitsInCur = 0;
            }
        }
    }

    /// <summary>Write a single bit-flag.</summary>
    public void WriteFlag(bool flag) => WriteBits(flag ? 1 : 0, 1);

    /// <summary>
    /// Write the AV1 trailing-bits marker (a single 1 bit followed by zero
    /// padding to the next byte boundary). Spec sec 5.3.4.
    /// </summary>
    public void WriteTrailingBits()
    {
        WriteBits(1, 1);
        if (_bitsInCur != 0)
        {
            int pad = 8 - _bitsInCur;
            WriteBits(0, pad);
        }
    }

    /// <summary>Snapshot the produced bytes (caller flushes via WriteTrailingBits first).</summary>
    public byte[] ToArray()
    {
        if (_bitsInCur != 0)
            throw new InvalidOperationException(
                "Av1BitWriter has unaligned bits. Call WriteTrailingBits before ToArray.");
        return _bytes.ToArray();
    }
}
