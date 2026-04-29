// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-callable FLAC subframe header parser. Mirror of
// FlacSubframeHeaderParser.Parse (libFLAC stream_decoder.c
// ::read_subframe_, RFC 9639 Section 10.1). Reads the 1-2 byte
// subframe header into kind / order / wastedBits scalars.
//
// Sequential per-stream because the bit reader state evolves over a
// few sequential reads. One-thread-per-stream on the GPU. Multiple
// FLAC channels parallelize across threads.
//
// Composes FlacBitReaderGpu for the bit reads. The output kind is
// encoded as the FlacSubframeKind enum's underlying int value
// (Constant=0, Verbatim=1, Fixed=2, Lpc=3).

using ILGPU;

namespace SpawnDev.Codecs.Audio.Flac;

/// <summary>
/// GPU-callable FLAC subframe header parser. Mirror of
/// <see cref="FlacSubframeHeaderParser"/>.Parse.
/// </summary>
public static class FlacSubframeHeaderGpu
{
    /// <summary>FlacSubframeKind.Constant.</summary>
    public const int KIND_CONSTANT = 0;
    /// <summary>FlacSubframeKind.Verbatim.</summary>
    public const int KIND_VERBATIM = 1;
    /// <summary>FlacSubframeKind.Fixed.</summary>
    public const int KIND_FIXED = 2;
    /// <summary>FlacSubframeKind.Lpc.</summary>
    public const int KIND_LPC = 3;

    /// <summary>
    /// Parse one subframe header. Writes 3 ints to the output buffer:
    /// [outBase + 0] = kind (KIND_*), [outBase + 1] = order, [outBase + 2] = wastedBits.
    /// Bit-exact vs the CPU FlacSubframeHeaderParser.Parse modulo input
    /// validation (caller verifies the reserved bit + reserved-code paths).
    /// </summary>
    /// <param name="state">Bit reader state.</param>
    /// <param name="data">Underlying byte buffer.</param>
    /// <param name="output">Output: 3-int header tuple (length &gt;= 3).</param>
    /// <param name="outBase">Base offset.</param>
    public static void ParseAt(
        ref FlacBitReaderGpuState state,
        ArrayView<byte> data,
        ArrayView<int> output, long outBase)
    {
        // Reserved bit (skip).
        FlacBitReaderGpu.ReadBits(ref state, data, 1);

        // 6-bit type code.
        int code = (int)FlacBitReaderGpu.ReadBits(ref state, data, 6);
        int kind;
        int order;
        if (code == 0)
        {
            kind = KIND_CONSTANT;
            order = 0;
        }
        else if (code == 1)
        {
            kind = KIND_VERBATIM;
            order = 0;
        }
        else if ((code & 0b111000) == 0b001000)
        {
            kind = KIND_FIXED;
            order = code & 0b000111;
        }
        else if ((code & 0b100000) == 0b100000)
        {
            kind = KIND_LPC;
            order = (code & 0b011111) + 1;
        }
        else
        {
            // Reserved encoding - return -1 sentinel for kind to let caller detect.
            kind = -1;
            order = 0;
        }

        // 1-bit wasted-bits flag, then unary count if flag is set.
        int wastedFlag = (int)FlacBitReaderGpu.ReadBits(ref state, data, 1);
        int wastedBits = 0;
        if (wastedFlag != 0)
        {
            wastedBits = FlacBitReaderGpu.ReadUnary(ref state, data) + 1;
        }

        output[outBase + 0] = kind;
        output[outBase + 1] = order;
        output[outBase + 2] = wastedBits;
    }
}
