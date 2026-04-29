// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// FLAC v1 GPU frame reader. Symmetric to FlacFrameWriterGpu - parses
// one VERBATIM-only frame from the input bytes, recovers the
// per-channel samples, verifies the CRC-8 (header) and CRC-16 (full
// frame).
//
// Frame layout this helper handles (matches FlacFrameWriterGpu's
// output exactly):
//   [0..N-1]  Frame header bits (variable length depending on UTF-8
//             frame number length)
//   [N]       CRC-8 of header bytes
//   [N+1..]   Per-channel VERBATIM subframes (8-bit header + samples)
//   [end-1, end-2]  CRC-16 over all preceding bytes (big-endian)
//
// V1 only handles the encoder's exact output shape (4096-block,
// 44.1 kHz, 16-bit, 1..8 channels independent, all VERBATIM,
// fixed blocking). Other configurations would parse incorrectly.

using ILGPU;

namespace SpawnDev.Codecs.Audio.Flac;

/// <summary>
/// GPU-callable FLAC v1 frame reader. Single-frame helper - parses
/// header, decodes VERBATIM subframes, verifies CRC-8 + CRC-16.
/// </summary>
public static class FlacFrameReaderGpu
{
    /// <summary>
    /// Parse one FLAC frame starting at <paramref name="frameBase"/>
    /// in <paramref name="data"/>. Returns the byte length of the
    /// frame (or 0 if header sync mismatch / CRC failure).
    /// Decoded samples written to <paramref name="samples"/> in
    /// channel-major layout (channel 0 first, then channel 1, etc.)
    /// starting at <paramref name="samplesBase"/>.
    /// </summary>
    /// <param name="frameLength">Total frame length in bytes (caller-known
    /// from the encoder's output, or computed via successive parsing).
    /// Used for the CRC-16 range.</param>
    public static long DecodeFrame(
        ArrayView<byte> data, long frameBase, int frameLength,
        int blockSize, int channels, int bps,
        ArrayView<int> samples, long samplesBase,
        ArrayView<int> statusOut)
    {
        // statusOut[0]: 0 = success, 1 = sync mismatch, 2 = CRC8 fail, 3 = CRC16 fail.
        statusOut[0] = 0;

        // ---- Parse + verify header bits ----
        var r = FlacBitReaderGpu.Init(frameLength);
        // Skip to frameBase by re-init with adjusted base. The reader
        // doesn't take a base offset, so we read from frameBase via
        // explicit byte indexing in the helpers. Simpler: SubView the
        // input data so reader sees a buffer starting at frameBase.
        // Since SubView isn't kernel-callable cleanly, we use the
        // reader on the raw view and set BytePos = frameBase as int.
        r.BytePos = (int)frameBase;

        // Sync code: 14 bits = 0x3FFE.
        uint sync = FlacBitReaderGpu.ReadBits(ref r, data, 14);
        if (sync != FlacConstants.FrameSyncCode)
        {
            statusOut[0] = 1;
            return 0;
        }
        // Skip reserved + blocking-strategy + bsize-code + srate-code +
        // channel-assignment + sample-size-code + reserved (1+1+4+4+4+3+1 = 18 bits).
        FlacBitReaderGpu.ReadBits(ref r, data, 18);
        // Skip UTF-8 frame number (variable - 1 to 7 bytes; trust the
        // encoder's output and read until we see a non-continuation
        // byte boundary).
        SkipUtf8Number(ref r, data);
        FlacBitReaderGpu.AlignToByte(ref r);
        // Header bytes consumed = current BytePos - frameBase.
        int headerBytes = r.BytePos - (int)frameBase;
        // Verify CRC-8 over header bytes (excluding the CRC-8 byte itself).
        byte expectedCrc8 = FlacCrcGpu.Compute8(data, frameBase, headerBytes);
        byte actualCrc8 = data[r.BytePos];
        if (expectedCrc8 != actualCrc8)
        {
            statusOut[0] = 2;
            return 0;
        }
        // Skip CRC-8 byte.
        r.BytePos++;

        // ---- Decode per-channel VERBATIM subframes ----
        for (int ch = 0; ch < channels; ch++)
        {
            int status = FlacVerbatimSubframeGpu.Decode(
                ref r, data,
                samples, samplesBase + (long)ch * blockSize,
                blockSize, bps);
            if (status != 1)
            {
                // Subframe header mismatch (not VERBATIM-only).
                statusOut[0] = 4;
                return 0;
            }
        }
        FlacBitReaderGpu.AlignToByte(ref r);

        // ---- Verify CRC-16 over full frame (excluding the CRC-16 bytes) ----
        int preFooterLen = r.BytePos - (int)frameBase;
        ushort expectedCrc16 = FlacCrcGpu.Compute16(data, frameBase, preFooterLen);
        ushort actualCrc16 = (ushort)((data[frameBase + preFooterLen] << 8)
            | data[frameBase + preFooterLen + 1]);
        if (expectedCrc16 != actualCrc16)
        {
            statusOut[0] = 3;
            return 0;
        }

        return preFooterLen + 2;
    }

    /// <summary>
    /// Skip a UTF-8-encoded number in the bit stream. The first byte
    /// determines the length: 0xxxxxxx = 1 byte, 110xxxxx = 2 bytes,
    /// 1110xxxx = 3 bytes, etc., up to 7 bytes max for 36-bit values.
    /// Each continuation byte is 10xxxxxx.
    /// </summary>
    private static void SkipUtf8Number(ref FlacBitReaderGpuState r, ArrayView<byte> data)
    {
        // Read first byte to determine length.
        uint b = FlacBitReaderGpu.ReadBits(ref r, data, 8);
        int trailing;
        if ((b & 0x80) == 0) trailing = 0;       // 0xxxxxxx -> 1 byte total
        else if ((b & 0xE0) == 0xC0) trailing = 1; // 110xxxxx -> 2 bytes total
        else if ((b & 0xF0) == 0xE0) trailing = 2; // 1110xxxx
        else if ((b & 0xF8) == 0xF0) trailing = 3; // 11110xxx
        else if ((b & 0xFC) == 0xF8) trailing = 4; // 111110xx
        else if ((b & 0xFE) == 0xFC) trailing = 5; // 1111110x
        else trailing = 6;                         // 11111110
        for (int i = 0; i < trailing; i++)
        {
            FlacBitReaderGpu.ReadBits(ref r, data, 8);
        }
    }
}
