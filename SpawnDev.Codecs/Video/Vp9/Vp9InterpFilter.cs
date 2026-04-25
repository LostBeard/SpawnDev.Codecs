// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 interpolation filter enum + parser. The header field selects
// the per-frame motion-compensation filter, or the SWITCHABLE
// signal (per-block selection in the compressed bitstream).
//
// Bitstream layout (libvpx read_interp_filter):
//   bit 0: 1 -> SWITCHABLE
//   bit 0 = 0: next 2 bits = filter index 0..3
//
// libvpx INTERP_FILTER values:
//   EIGHTTAP        = 0
//   EIGHTTAP_SMOOTH = 1
//   EIGHTTAP_SHARP  = 2
//   BILINEAR        = 3
//   SWITCHABLE      = 4

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 interpolation filter (libvpx INTERP_FILTER).</summary>
public enum Vp9InterpFilter : byte
{
    /// <summary>Standard 8-tap interpolation filter.</summary>
    EightTap = 0,
    /// <summary>Smoother 8-tap variant.</summary>
    EightTapSmooth = 1,
    /// <summary>Sharper 8-tap variant.</summary>
    EightTapSharp = 2,
    /// <summary>Cheap bilinear filter.</summary>
    Bilinear = 3,
    /// <summary>Per-block selection (signalled in the compressed bitstream).</summary>
    Switchable = 4,
}

/// <summary>Parser for the interpolation filter field.</summary>
public static class Vp9InterpFilterParser
{
    /// <summary>
    /// Read a 1-or-3 bit interpolation filter selector. Mirror of libvpx
    /// <c>read_interp_filter</c>.
    /// </summary>
    internal static Vp9InterpFilter Parse(ref Vp9BitReader reader)
    {
        if (reader.ReadFlag())
            return Vp9InterpFilter.Switchable;
        return (Vp9InterpFilter)reader.ReadBits(2);
    }

    /// <summary>Convenience overload for unit tests.</summary>
    public static Vp9InterpFilter Parse(ReadOnlySpan<byte> data)
    {
        var r = new Vp9BitReader(data);
        return Parse(ref r);
    }
}
