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

    /// <summary>
    /// Write a canonical Huffman codebook entry. The codeword's bits are
    /// emitted MSB-first within the codeword (which is the canonical
    /// Vorbis convention; the LSB-first stream packing is handled by
    /// the underlying WriteBits). Mirrors
    /// VorbisAudioEncoder.WriteCodebookEntry.
    /// </summary>
    /// <param name="state">Bit writer state (mutated).</param>
    /// <param name="outBuf">Output byte buffer.</param>
    /// <param name="codes">Flat codeword array; codes[entry] is the canonical bit pattern.</param>
    /// <param name="codesBase">Base offset.</param>
    /// <param name="lengths">Flat codeword length array (bits per entry).</param>
    /// <param name="lengthsBase">Base offset.</param>
    /// <param name="entry">Codebook entry index.</param>
    public static void WriteCodebookEntry(
        ref VorbisBitWriterGpuState state, ArrayView<byte> outBuf,
        ArrayView<uint> codes, long codesBase,
        ArrayView<int> lengths, long lengthsBase,
        int entry)
    {
        uint code = codes[codesBase + entry];
        int length = lengths[lengthsBase + entry];
        for (int b = length - 1; b >= 0; b--)
        {
            uint bit = (code >> b) & 1u;
            WriteBits(ref state, outBuf, bit, 1);
        }
    }
}
