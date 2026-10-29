// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP8 intra prediction edge sample defaults. RFC 6386 sec 12.1 specifies
// the values to use for above/left/top-left samples that fall outside
// the frame, intentionally chosen so the predictors produce a smooth
// extrapolation rather than crashing or wrapping:
//
//   above sample out-of-frame -> 127
//   left sample out-of-frame  -> 129
//   top-left when above OOF   -> 127
//   top-left when left OOF    -> 129
//   top-left when both OOF    -> 127 (above wins)
//
// libvpx reference: vp8/common/setupintrarecon.c (vp8_setup_intra_recon
// + vp8_setup_intra_recon_top_line, plus the per-MB initializers at
// the start of decode_mb_rows).

namespace SpawnDev.Codecs.Video.Vp8;

/// <summary>VP8 out-of-frame edge sample fill values (RFC 6386 sec 12.1).</summary>
public static class Vp8IntraEdgeFill
{
    /// <summary>Default value for above samples that fall outside the frame.</summary>
    public const byte AboveDefault = 127;

    /// <summary>Default value for left samples that fall outside the frame.</summary>
    public const byte LeftDefault = 129;

    /// <summary>
    /// Resolve the top-left corner sample given which neighbors are
    /// in-frame. Mirrors the libvpx convention: when above is out-of-frame
    /// the top-left uses 127 (above's default); otherwise when left is
    /// out-of-frame the top-left uses 129; otherwise the actual sample
    /// at the top-left position is used (caller passes via <paramref name="actual"/>).
    /// </summary>
    public static byte ResolveTopLeft(bool haveAbove, bool haveLeft, byte actual)
    {
        if (!haveAbove) return AboveDefault;
        if (!haveLeft) return LeftDefault;
        return actual;
    }

    /// <summary>
    /// Fill a 16-byte above-row buffer with the libvpx default (127) for
    /// out-of-frame slots. Caller may overwrite slots that are in-frame
    /// from the recon buffer.
    /// </summary>
    public static void FillAboveRow16(Span<byte> above)
    {
        if (above.Length < 16) throw new ArgumentException("above must hold >= 16 bytes", nameof(above));
        above.Slice(0, 16).Fill(AboveDefault);
    }

    /// <summary>Fill a 16-byte left-column buffer with the libvpx default (129).</summary>
    public static void FillLeftColumn16(Span<byte> left)
    {
        if (left.Length < 16) throw new ArgumentException("left must hold >= 16 bytes", nameof(left));
        left.Slice(0, 16).Fill(LeftDefault);
    }

    /// <summary>Fill an 8-byte chroma above-row buffer with the libvpx default (127).</summary>
    public static void FillAboveRow8(Span<byte> above)
    {
        if (above.Length < 8) throw new ArgumentException("above must hold >= 8 bytes", nameof(above));
        above.Slice(0, 8).Fill(AboveDefault);
    }

    /// <summary>Fill an 8-byte chroma left-column buffer with the libvpx default (129).</summary>
    public static void FillLeftColumn8(Span<byte> left)
    {
        if (left.Length < 8) throw new ArgumentException("left must hold >= 8 bytes", nameof(left));
        left.Slice(0, 8).Fill(LeftDefault);
    }

    /// <summary>Fill a 4-byte luma 4x4 above row + extra above-right (8 bytes total).</summary>
    public static void FillAboveRow4Plus4(Span<byte> aboveExtended)
    {
        if (aboveExtended.Length < 8) throw new ArgumentException("aboveExtended must hold >= 8 bytes", nameof(aboveExtended));
        aboveExtended.Slice(0, 8).Fill(AboveDefault);
    }

    /// <summary>Fill a 4-byte luma 4x4 left column.</summary>
    public static void FillLeftColumn4(Span<byte> left)
    {
        if (left.Length < 4) throw new ArgumentException("left must hold >= 4 bytes", nameof(left));
        left.Slice(0, 4).Fill(LeftDefault);
    }
}
