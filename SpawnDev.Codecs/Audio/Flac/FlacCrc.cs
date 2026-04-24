// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// CRC-8 and CRC-16 implementations used by the FLAC bitstream.
// - Frame header is protected by CRC-8 (polynomial x^8 + x^2 + x + 1 = 0x07).
// - Whole-frame footer is protected by CRC-16 (polynomial x^16 + x^15 + x^2 + 1 = 0x8005).
// Both start at 0x00, no reflection, no xor-out. Matches libFLAC crc.c / crc.h.

namespace SpawnDev.Codecs.Audio.Flac;

/// <summary>
/// Byte-oriented CRC-8 (poly 0x07) and CRC-16 (poly 0x8005) as used by FLAC.
/// </summary>
internal static class FlacCrc
{
    private static readonly byte[] Crc8Table = BuildCrc8Table();
    private static readonly ushort[] Crc16Table = BuildCrc16Table();

    private static byte[] BuildCrc8Table()
    {
        var t = new byte[256];
        for (int i = 0; i < 256; i++)
        {
            int c = i;
            for (int j = 0; j < 8; j++)
                c = ((c & 0x80) != 0) ? ((c << 1) ^ 0x07) : (c << 1);
            t[i] = (byte)c;
        }
        return t;
    }

    private static ushort[] BuildCrc16Table()
    {
        var t = new ushort[256];
        for (int i = 0; i < 256; i++)
        {
            int c = i << 8;
            for (int j = 0; j < 8; j++)
                c = ((c & 0x8000) != 0) ? ((c << 1) ^ 0x8005) : (c << 1);
            t[i] = (ushort)c;
        }
        return t;
    }

    /// <summary>
    /// Compute CRC-8 over a span of bytes. Initial value 0x00; polynomial 0x07.
    /// </summary>
    internal static byte Compute8(ReadOnlySpan<byte> data)
    {
        byte crc = 0;
        for (int i = 0; i < data.Length; i++)
            crc = Crc8Table[crc ^ data[i]];
        return crc;
    }

    /// <summary>
    /// Compute CRC-16 over a span of bytes. Initial value 0x0000; polynomial 0x8005.
    /// </summary>
    internal static ushort Compute16(ReadOnlySpan<byte> data)
    {
        ushort crc = 0;
        for (int i = 0; i < data.Length; i++)
            crc = (ushort)(Crc16Table[(crc >> 8) ^ data[i]] ^ (crc << 8));
        return crc;
    }
}
