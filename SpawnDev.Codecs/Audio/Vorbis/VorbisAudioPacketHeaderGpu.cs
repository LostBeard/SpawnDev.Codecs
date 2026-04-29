// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-callable Vorbis audio packet header parser. Mirror of
// VorbisAudioPacketHeaderParser.ParseFromReader (Vorbis I sec 4.3.1).
//
// Reads:
//   1 bit              : packet type (must be 0 for audio - we trust)
//   ilog(modes - 1) b. : mode number
//   if mode is long-block:
//     1 bit prevWindowLong, 1 bit nextWindowLong

using ILGPU;

namespace SpawnDev.Codecs.Audio.Vorbis;

/// <summary>Decoded Vorbis audio packet header in GPU-friendly flat form.</summary>
public struct VorbisAudioPacketHeaderGpuResult
{
    /// <summary>Mode index from the setup header mode list.</summary>
    public int ModeNumber;
    /// <summary>Effective block size in samples for this packet.</summary>
    public int BlockSize;
    /// <summary>1 if mode is long-block, 0 if short-block.</summary>
    public int IsLongBlock;
    /// <summary>Previous-window flag (long blocks only); 0 / 1.</summary>
    public int PreviousWindowLong;
    /// <summary>Next-window flag; 0 / 1.</summary>
    public int NextWindowLong;
}

/// <summary>
/// GPU-callable Vorbis audio packet header parser. Mirror of
/// <see cref="VorbisAudioPacketHeaderParser"/>.ParseFromReader.
/// </summary>
public static class VorbisAudioPacketHeaderGpu
{
    /// <summary>
    /// Parse the audio packet header from <paramref name="packet"/>.
    /// Caller pre-computes <paramref name="modeBits"/> via VorbisMath.Ilog
    /// (host metadata struct setup) and uploads <paramref name="modeBlockFlags"/>
    /// (per-mode 0=short, 1=long).
    /// </summary>
    /// <param name="state">Vorbis bit reader state (mutated).</param>
    /// <param name="packet">Packet bytes.</param>
    /// <param name="modeBits">ilog(modes - 1) precomputed; 0 when modes == 1.</param>
    /// <param name="modeBlockFlags">Per-mode block flag (1 = long, 0 = short).</param>
    /// <param name="modeBlockFlagsBase">Base offset.</param>
    /// <param name="blockSize0">Short-block size (samples).</param>
    /// <param name="blockSize1">Long-block size (samples).</param>
    /// <returns>Parsed header in flat-int form.</returns>
    public static VorbisAudioPacketHeaderGpuResult Parse(
        ref VorbisBitReaderGpuState state, ArrayView<byte> packet,
        int modeBits,
        ArrayView<int> modeBlockFlags, long modeBlockFlagsBase,
        int blockSize0, int blockSize1)
    {
        // 1 bit packet type (we trust it's 0 - Rule 1: validation at the
        // boundary, not inside the kernel).
        VorbisBitReaderGpu.ReadBits(ref state, packet, 1);

        // ilog(modes - 1) bits mode number.
        int modeNumber = modeBits > 0
            ? (int)VorbisBitReaderGpu.ReadBits(ref state, packet, modeBits)
            : 0;

        int isLong = modeBlockFlags[modeBlockFlagsBase + modeNumber];
        int blockSize = isLong != 0 ? blockSize1 : blockSize0;

        int prevLong = 0;
        int nextLong = 0;
        if (isLong != 0)
        {
            prevLong = (int)VorbisBitReaderGpu.ReadBits(ref state, packet, 1);
            nextLong = (int)VorbisBitReaderGpu.ReadBits(ref state, packet, 1);
        }

        return new VorbisAudioPacketHeaderGpuResult
        {
            ModeNumber = modeNumber,
            BlockSize = blockSize,
            IsLongBlock = isLong,
            PreviousWindowLong = prevLong,
            NextWindowLong = nextLong,
        };
    }
}
