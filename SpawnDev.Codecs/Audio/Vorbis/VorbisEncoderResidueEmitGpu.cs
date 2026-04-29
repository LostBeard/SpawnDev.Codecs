// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-callable composite primitive: write a sequence of codebook
// entries from a per-bin index array to the bitstream. Mirrors the
// inner residue-emission loop in VorbisAudioEncoder.EncodeAudioPacket
// (lines 303-307):
//
//   for (int i = 0; i < half; i++) WriteCodebookEntry(writer, residueBookCodes, residueQ[i]);
//
// Single-thread dispatch (the bit writer is sequential).

using ILGPU;

namespace SpawnDev.Codecs.Audio.Vorbis;

/// <summary>
/// GPU-callable Vorbis encoder residue emit composite primitive.
/// </summary>
public static class VorbisEncoderResidueEmitGpu
{
    /// <summary>
    /// Emit <paramref name="count"/> codebook entries from
    /// <paramref name="residueQ"/> via WriteCodebookEntry, in order,
    /// using a single shared codebook (codes + lengths). Single-thread
    /// dispatch.
    /// </summary>
    public static void EmitAll(
        ref VorbisBitWriterGpuState state, ArrayView<byte> outBuf,
        ArrayView<int> residueQ, long residueBase, int count,
        ArrayView<uint> codebookCodes, long codesBase,
        ArrayView<int> codebookLengths, long lengthsBase)
    {
        for (int i = 0; i < count; i++)
        {
            int entry = residueQ[residueBase + i];
            VorbisBitWriterGpu.WriteCodebookEntry(
                ref state, outBuf,
                codebookCodes, codesBase,
                codebookLengths, lengthsBase,
                entry);
        }
    }
}
