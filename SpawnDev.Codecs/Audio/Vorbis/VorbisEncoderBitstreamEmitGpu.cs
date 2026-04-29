// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-callable Vorbis encoder bitstream emit composite. Mirror of the
// per-packet bit-pack logic in VorbisAudioEncoder.EncodeAudioPacket
// (lines 273-310) for the v1 mono encoder shape:
//   - 1 bit packet type = 0
//   - 0 bits mode (single mode -> ilog(0) = 0 bits)
//   - 1 bit floor nonzero = 1
//   - endpointBits x 2 endpoint posteriors
//   - 1 codebook entry (classbook entry 0)
//   - count residue codebook entries from residueQ[]
//
// Single-thread dispatch (bitstream emit is sequential).

using ILGPU;

namespace SpawnDev.Codecs.Audio.Vorbis;

/// <summary>
/// GPU-callable Vorbis encoder full-packet bitstream emitter.
/// </summary>
public static class VorbisEncoderBitstreamEmitGpu
{
    /// <summary>
    /// Emit one complete v1 Vorbis audio packet (header + posteriors +
    /// classbook entry + residue entries) into <paramref name="outBuf"/>.
    /// Returns the byte length written via <paramref name="outLen"/>[0].
    /// </summary>
    /// <param name="outBuf">Output byte buffer.</param>
    /// <param name="outLen">Output length (1 element).</param>
    /// <param name="residueQ">Per-bin residue codebook entry indices.</param>
    /// <param name="residueQBase">Base offset.</param>
    /// <param name="count">Residue entry count (= halfBlock).</param>
    /// <param name="posteriorY0">Floor endpoint Y[0] (low band).</param>
    /// <param name="posteriorY1">Floor endpoint Y[1] (high band).</param>
    /// <param name="endpointBits">Bits per endpoint (8/7/7/6 for multiplier 1/2/3/4).</param>
    /// <param name="modeBits">ilog(modes - 1); 0 for single-mode.</param>
    /// <param name="classbookCodes">Classbook code table.</param>
    /// <param name="classbookCodesBase">Base offset.</param>
    /// <param name="classbookLengths">Classbook length table.</param>
    /// <param name="classbookLengthsBase">Base offset.</param>
    /// <param name="residueBookCodes">Residue book code table.</param>
    /// <param name="residueBookCodesBase">Base offset.</param>
    /// <param name="residueBookLengths">Residue book length table.</param>
    /// <param name="residueBookLengthsBase">Base offset.</param>
    public static void EmitPacket(
        ArrayView<byte> outBuf, ArrayView<long> outLen,
        ArrayView<int> residueQ, long residueQBase, int count,
        int posteriorY0, int posteriorY1,
        int endpointBits, int modeBits,
        ArrayView<uint> classbookCodes, long classbookCodesBase,
        ArrayView<int> classbookLengths, long classbookLengthsBase,
        ArrayView<uint> residueBookCodes, long residueBookCodesBase,
        ArrayView<int> residueBookLengths, long residueBookLengthsBase)
    {
        var state = VorbisBitWriterGpu.Init();

        // Packet type = 0 (audio).
        VorbisBitWriterGpu.WriteBits(ref state, outBuf, 0u, 1);

        // Mode index. modeBits == 0 for single-mode (no bits emitted).
        if (modeBits > 0)
        {
            // v1 encoder uses mode 0 always.
            VorbisBitWriterGpu.WriteBits(ref state, outBuf, 0u, modeBits);
        }

        // Floor nonzero bit.
        VorbisBitWriterGpu.WriteBits(ref state, outBuf, 1u, 1);

        // Endpoint posteriors.
        VorbisBitWriterGpu.WriteBits(ref state, outBuf, (uint)posteriorY0, endpointBits);
        VorbisBitWriterGpu.WriteBits(ref state, outBuf, (uint)posteriorY1, endpointBits);

        // Classbook entry (always entry 0 in v1).
        VorbisBitWriterGpu.WriteCodebookEntry(
            ref state, outBuf,
            classbookCodes, classbookCodesBase,
            classbookLengths, classbookLengthsBase,
            0);

        // Per-bin residue codebook entries.
        VorbisEncoderResidueEmitGpu.EmitAll(
            ref state, outBuf,
            residueQ, residueQBase, count,
            residueBookCodes, residueBookCodesBase,
            residueBookLengths, residueBookLengthsBase);

        VorbisBitWriterGpu.Finish(ref state, outBuf);
        outLen[0] = state.OutLen;
    }
}
