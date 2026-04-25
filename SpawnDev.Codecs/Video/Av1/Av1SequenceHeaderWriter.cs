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
}

/// <summary>AV1 SequenceHeader OBU payload writer.</summary>
public static class Av1SequenceHeaderWriter
{
    /// <summary>
    /// Emit the SequenceHeader OBU PAYLOAD (i.e. the bytes that go AFTER
    /// the OBU header / size prefix). Wrap with <see cref="Av1ObuWriter.EmitObu"/>
    /// to produce a full OBU.
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
        bw.WriteFlag(false); // use_128x128_superblock
        bw.WriteFlag(false); // enable_filter_intra
        bw.WriteFlag(false); // enable_intra_edge_filter

        bw.WriteFlag(false); // enable_interintra_compound
        bw.WriteFlag(false); // enable_masked_compound
        bw.WriteFlag(false); // enable_warped_motion
        bw.WriteFlag(false); // enable_dual_filter
        bw.WriteFlag(false); // enable_order_hint
        bw.WriteFlag(true);  // seq_choose_screen_content_tools = 1 (SELECT)
        // SELECT puts seq_force_screen_content_tools at 2 (>0), so the
        // bitstream still carries seq_choose_integer_mv. Emit SELECT for
        // that too so frame-level integer-mv flags pick it up.
        bw.WriteFlag(true);  // seq_choose_integer_mv = 1
        // No order_hint_bits_minus_1 since enable_order_hint=0.

        bw.WriteFlag(false); // enable_superres
        bw.WriteFlag(false); // enable_cdef
        bw.WriteFlag(false); // enable_restoration

        // color_config
        bool highBitDepth = cfg.BitDepth >= 10;
        bw.WriteFlag(highBitDepth);
        if (cfg.SeqProfile == 2 && highBitDepth)
            bw.WriteFlag(cfg.BitDepth == 12);
        if (cfg.SeqProfile != 1)
            bw.WriteFlag(cfg.Monochrome);
        bw.WriteFlag(false); // color_description_present_flag
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
                bw.WriteBits(0, 2); // chroma_sample_position = CSP_UNKNOWN
            bw.WriteFlag(false); // separate_uv_deltas
        }

        bw.WriteFlag(false); // film_grain_params_present

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
