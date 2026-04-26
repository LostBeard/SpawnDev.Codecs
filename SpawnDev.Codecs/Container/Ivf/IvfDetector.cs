// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// IVF format detection utility. Quickly tests whether a byte buffer
// is a well-formed IVF stream and what codec it carries.

namespace SpawnDev.Codecs.Container.Ivf;

/// <summary>Codec carried by an IVF stream, identified by FourCC.</summary>
public enum IvfCodec
{
    /// <summary>Not a recognized IVF stream.</summary>
    Unknown = 0,
    /// <summary>VP8 (FourCC = "VP80").</summary>
    Vp8,
    /// <summary>VP9 (FourCC = "VP90").</summary>
    Vp9,
    /// <summary>AV1 (FourCC = "AV01").</summary>
    Av1,
    /// <summary>Some other codec (parser identified IVF but FourCC is unrecognized).</summary>
    Other,
}

/// <summary>IVF format detector.</summary>
public static class IvfDetector
{
    /// <summary>
    /// Quick test: does <paramref name="data"/> start with a valid IVF
    /// signature ('DKIF') and well-formed file header?
    /// </summary>
    public static bool IsIvf(ReadOnlySpan<byte> data)
    {
        if (data.Length < 32) return false;
        if (data[0] != (byte)'D' || data[1] != (byte)'K'
            || data[2] != (byte)'I' || data[3] != (byte)'F') return false;
        return true;
    }

    /// <summary>
    /// Identify the codec carried by an IVF stream by inspecting its
    /// FourCC field. Returns <see cref="IvfCodec.Unknown"/> when the
    /// data is not a valid IVF stream.
    /// </summary>
    public static IvfCodec DetectCodec(ReadOnlySpan<byte> data)
    {
        if (!IsIvf(data)) return IvfCodec.Unknown;
        try
        {
            var header = IvfReader.ParseHeader(data);
            return header.FourCc switch
            {
                "VP80" => IvfCodec.Vp8,
                "VP90" => IvfCodec.Vp9,
                "AV01" => IvfCodec.Av1,
                _ => IvfCodec.Other,
            };
        }
        catch
        {
            return IvfCodec.Unknown;
        }
    }
}
