// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// FLAC CRC-8 + CRC-16, GPU-callable form. Bit-exact mirror of FlacCrc.
//
// Used by FLAC frame encoder + decoder to compute / verify the
// frame-header CRC-8 (poly 0x07) and frame-footer CRC-16 (poly 0x8005).
// Both start at 0x00, no reflection, no xor-out.
//
// In-kernel implementation uses bit-shift loops instead of byte tables.
// This avoids the need to upload a 256-entry CRC table for each kernel
// invocation; the bit-shift loop unrolls nicely on every backend.

using ILGPU;

namespace SpawnDev.Codecs.Audio.Flac;

/// <summary>GPU-callable FLAC CRC-8 + CRC-16 helpers.</summary>
public static class FlacCrcGpu
{
    /// <summary>
    /// Compute CRC-8 over <paramref name="data"/> bytes [<paramref name="start"/>,
    /// <paramref name="start"/>+<paramref name="length"/>). Polynomial 0x07.
    /// Initial value 0x00.
    /// </summary>
    public static byte Compute8(ArrayView<byte> data, long start, int length)
    {
        int crc = 0;
        for (int i = 0; i < length; i++)
        {
            crc ^= data[start + i];
            for (int j = 0; j < 8; j++)
            {
                if ((crc & 0x80) != 0) crc = (crc << 1) ^ 0x07;
                else crc <<= 1;
            }
            crc &= 0xFF;
        }
        return (byte)crc;
    }

    /// <summary>
    /// Compute CRC-16 over <paramref name="data"/> bytes [<paramref name="start"/>,
    /// <paramref name="start"/>+<paramref name="length"/>). Polynomial 0x8005.
    /// Initial value 0x0000.
    /// </summary>
    public static ushort Compute16(ArrayView<byte> data, long start, int length)
    {
        int crc = 0;
        for (int i = 0; i < length; i++)
        {
            crc ^= ((int)data[start + i]) << 8;
            for (int j = 0; j < 8; j++)
            {
                if ((crc & 0x8000) != 0) crc = (crc << 1) ^ 0x8005;
                else crc <<= 1;
            }
            crc &= 0xFFFF;
        }
        return (ushort)crc;
    }
}
