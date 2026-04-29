// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Vorbis LSB-first bit writer, GPU-callable form. Bit-exact mirror
// of VorbisBitWriter for in-kernel use by the upcoming Vorbis
// encoder pipeline.

using ILGPU;

namespace SpawnDev.Codecs.Audio.Vorbis;

/// <summary>In-kernel state for the Vorbis bit writer.</summary>
public struct VorbisBitWriterGpuState
{
    /// <summary>Accumulator for the in-progress byte.</summary>
    public int CurrentByte;
    /// <summary>Next bit position within CurrentByte (0..7).</summary>
    public int BitPos;
    /// <summary>Number of complete bytes written.</summary>
    public long OutLen;
}

/// <summary>
/// GPU-callable LSB-first bit writer matching Vorbis I packing.
/// </summary>
public static class VorbisBitWriterGpu
{
    /// <summary>Initialize a fresh writer state.</summary>
    public static VorbisBitWriterGpuState Init() => new VorbisBitWriterGpuState
    {
        CurrentByte = 0,
        BitPos = 0,
        OutLen = 0,
    };

    /// <summary>
    /// Append <paramref name="nBits"/> low-order bits of <paramref name="value"/>
    /// to the stream (LSB-first).
    /// </summary>
    public static void WriteBits(
        ref VorbisBitWriterGpuState state, ArrayView<byte> outBuf,
        uint value, int nBits)
    {
        if (nBits == 0) return;
        if (nBits < 32) value &= (1u << nBits) - 1u;

        int remaining = nBits;
        int srcBit = 0;
        while (remaining > 0)
        {
            int spaceInByte = 8 - state.BitPos;
            int take = spaceInByte < remaining ? spaceInByte : remaining;
            uint chunk = (value >> srcBit) & ((1u << take) - 1u);
            state.CurrentByte |= (int)(chunk << state.BitPos);
            state.BitPos += take;
            srcBit += take;
            remaining -= take;
            if (state.BitPos == 8)
            {
                outBuf[state.OutLen] = (byte)state.CurrentByte;
                state.OutLen++;
                state.CurrentByte = 0;
                state.BitPos = 0;
            }
        }
    }

    /// <summary>Flush the in-progress byte (zero-pads high bits).</summary>
    public static void Finish(
        ref VorbisBitWriterGpuState state, ArrayView<byte> outBuf)
    {
        if (state.BitPos != 0)
        {
            outBuf[state.OutLen] = (byte)state.CurrentByte;
            state.OutLen++;
            state.CurrentByte = 0;
            state.BitPos = 0;
        }
    }
}
