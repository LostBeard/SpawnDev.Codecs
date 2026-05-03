// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Opus entropy coder (range coder), GPU-callable form. Bit-exact
// mirror of the CPU OpusRangeDecoder (libopus celt/ec_dec.c +
// celt/entcode.c) for the decoder side.
//
// Important note about the "shared with AV1" design:
// libopus and libaom both descend from the Daala range coder family,
// but the on-the-wire bit-stream formats differ. libopus uses an
// inverse-CDF (icdf) representation with default ftb=8, byte-by-byte
// front + back buffer reads, and a per-codepoint normalization loop;
// libaom (od_ec) uses q15 CDF with a 32-bit dif window and OdEcWindowSize
// refill semantics. The state structs and bit-loading semantics are
// NOT byte-compatible — running an Opus packet through Av1RangeDecoderGpu
// will not decode correctly. This file ships the libopus-shape decoder
// so SILK + CELT integration kernels can decode real Opus packets.
//
// The encoder side (`OpusRangeEncoderGpu`) currently delegates to AV1's
// Daala encoder. That delegation is incorrect for any encoder use that
// emits libopus-format bytes; it remains here as scaffolding from an
// earlier design pass and must be rewritten before the OpusEncoderGpu
// pair lands. Decoder side IS correct and is what the SILK decode
// integration kernel uses.

using ILGPU;
using SpawnDev.Codecs.Video.Av1;

namespace SpawnDev.Codecs.Audio.Opus;

/// <summary>
/// In-kernel state for the Opus range encoder. SCAFFOLDING ONLY -
/// currently mirrors the AV1 Daala encoder shape; libopus encoder
/// state shape differs and this struct will be rewritten when
/// `OpusRangeEncoderGpu` actually emits libopus bytes. Do not rely on
/// the field layout here.
/// </summary>
public struct OpusRangeEncoderGpuState
{
    /// <summary>Encoder low value (high bits of current code).</summary>
    public ulong Low;
    /// <summary>Current range, normalized to [32768, 65535].</summary>
    public uint Rng;
    /// <summary>Bit counter; starts at -9.</summary>
    public int Cnt;
    /// <summary>Number of bytes written.</summary>
    public long OutLen;
}

/// <summary>
/// In-kernel state for the Opus range decoder. Mirrors the mutable
/// fields of <see cref="SpawnDev.Codecs.EntropyCoders.OpusRangeDecoder"/>
/// (libopus <c>ec_ctx</c>). Immutable buffer info is passed to each
/// helper as a separate `ArrayView&lt;byte&gt;` + start offset + storage
/// length so the state struct stays small.
/// </summary>
public struct OpusRangeDecoderGpuState
{
    /// <summary>Forward-read offset into the buffer (bytes consumed from front).</summary>
    public uint Offs;
    /// <summary>Backward-read offset into the buffer (bytes consumed from back).</summary>
    public uint EndOffs;
    /// <summary>Window of raw bits read from the back, accumulated bottom-up.</summary>
    public uint EndWindow;
    /// <summary>Bits currently held in <see cref="EndWindow"/>.</summary>
    public int NEndBits;
    /// <summary>
    /// Whole bits "used" so far (used by <c>Tell</c>). Starts at
    /// <c>EC_CODE_BITS + 1 - ((EC_CODE_BITS - EC_CODE_EXTRA) / EC_SYM_BITS) * EC_SYM_BITS</c>
    /// = 9 per the libopus init.
    /// </summary>
    public int NBitsTotal;
    /// <summary>Current range. Always &gt; 0 after Init.</summary>
    public uint Rng;
    /// <summary>Current value (low part of code).</summary>
    public uint Val;
    /// <summary>Scratch range/extension used by Decode/Update.</summary>
    public uint Ext;
    /// <summary>Last byte read by ReadByte; carried across Normalize calls.</summary>
    public int Rem;
    /// <summary>Non-zero if the decoder has detected a malformed stream (e.g. bad uint).</summary>
    public int Error;
}

/// <summary>
/// GPU-callable Opus range encoder. SCAFFOLDING - delegates to AV1's
/// Daala encoder. NOT bit-correct for libopus output; rewrite required
/// before encoder integration. Decoder side (<see cref="OpusRangeDecoderGpu"/>)
/// is the working primitive.
/// </summary>
public static class OpusRangeEncoderGpu
{
    /// <summary>Initialize a fresh encoder state. Scaffolding (delegates to AV1).</summary>
    public static OpusRangeEncoderGpuState Init()
    {
        var av1 = Av1RangeEncoderGpu.Init();
        return new OpusRangeEncoderGpuState
        {
            Low = av1.Low,
            Rng = av1.Rng,
            Cnt = av1.Cnt,
            OutLen = av1.OutLen,
        };
    }

    /// <summary>q15 CDF top value (32768).</summary>
    public const int CdfProbTop = Av1RangeEncoderGpu.CdfProbTop;
}

/// <summary>
/// GPU-callable Opus range decoder. Bit-exact mirror of the CPU
/// `OpusRangeDecoder`; primitives operate on a libopus-shape state
/// struct + an `ArrayView&lt;byte&gt;` buffer. Used by SILK + CELT
/// integration kernels.
/// </summary>
public static class OpusRangeDecoderGpu
{
    // Constants from libopus celt/mfrngcod.h + celt/entcode.h
    private const int EC_SYM_BITS = 8;
    private const int EC_CODE_BITS = 32;
    private const uint EC_SYM_MAX = (1u << EC_SYM_BITS) - 1u;            // 0xFF
    private const uint EC_CODE_TOP = 1u << (EC_CODE_BITS - 1);           // 0x80000000
    private const uint EC_CODE_BOT = EC_CODE_TOP >> EC_SYM_BITS;         // 0x00800000
    private const int EC_CODE_EXTRA = (EC_CODE_BITS - 2) % EC_SYM_BITS + 1; // 7
    private const int EC_WINDOW_SIZE = 32;

    /// <summary>q15 CDF top value (32768). Provided for cross-codec consumers.</summary>
    public const int CdfProbTop = 1 << 15;

    /// <summary>
    /// Initialize the decoder state to read <paramref name="storage"/>
    /// bytes from <paramref name="buf"/> starting at <paramref name="bufStart"/>.
    /// Mirrors libopus <c>ec_dec_init</c>.
    /// </summary>
    public static OpusRangeDecoderGpuState Init(
        ArrayView<byte> buf, int bufStart, uint storage)
    {
        var state = new OpusRangeDecoderGpuState
        {
            Offs = 0,
            EndOffs = 0,
            EndWindow = 0,
            NEndBits = 0,
            NBitsTotal = EC_CODE_BITS + 1 -
                ((EC_CODE_BITS - EC_CODE_EXTRA) / EC_SYM_BITS) * EC_SYM_BITS,
            Rng = 1u << EC_CODE_EXTRA,
            Ext = 0,
            Rem = 0,
            Error = 0,
        };
        state.Rem = ReadByte(ref state, buf, bufStart, storage);
        state.Val = state.Rng - 1u - (uint)(state.Rem >> (EC_SYM_BITS - EC_CODE_EXTRA));
        Normalize(ref state, buf, bufStart, storage);
        return state;
    }

    /// <summary>
    /// Decode a symbol given an "inverse" CDF table. The table must be
    /// monotonically non-increasing with a final entry of 0.
    /// <paramref name="ftb"/> is the number of bits of precision (libopus
    /// SILK uses 8 by default; CELT mostly 8). Mirrors libopus
    /// <c>ec_dec_icdf</c>.
    /// </summary>
    public static int DecodeIcdf(
        ref OpusRangeDecoderGpuState state,
        ArrayView<byte> buf, int bufStart, uint storage,
        ArrayView<byte> icdf, long icdfBase, int ftb)
    {
        uint s = state.Rng;
        uint d = state.Val;
        uint r = s >> ftb;
        int ret = -1;
        uint t;
        do
        {
            t = s;
            ret++;
            s = r * icdf[icdfBase + ret];
        }
        while (d < s);
        state.Val = d - s;
        state.Rng = t - s;
        Normalize(ref state, buf, bufStart, storage);
        return ret;
    }

    /// <summary>
    /// 16-bit-entry variant of <see cref="DecodeIcdf"/>. Some libopus
    /// CELT tables exceed 255 in width and use ushort entries.
    /// </summary>
    public static int DecodeIcdf16(
        ref OpusRangeDecoderGpuState state,
        ArrayView<byte> buf, int bufStart, uint storage,
        ArrayView<ushort> icdf, long icdfBase, int ftb)
    {
        uint s = state.Rng;
        uint d = state.Val;
        uint r = s >> ftb;
        int ret = -1;
        uint t;
        do
        {
            t = s;
            ret++;
            s = r * icdf[icdfBase + ret];
        }
        while (d < s);
        state.Val = d - s;
        state.Rng = t - s;
        Normalize(ref state, buf, bufStart, storage);
        return ret;
    }

    /// <summary>
    /// Decode a bit with probability <c>1 / (1 &lt;&lt; logp)</c> of being 1.
    /// Mirrors libopus <c>ec_dec_bit_logp</c>.
    /// </summary>
    public static int DecodeBitLogP(
        ref OpusRangeDecoderGpuState state,
        ArrayView<byte> buf, int bufStart, uint storage,
        int logp)
    {
        uint r = state.Rng;
        uint d = state.Val;
        uint s = r >> logp;
        int ret = d < s ? 1 : 0;
        if (ret == 0) state.Val = d - s;
        state.Rng = ret != 0 ? s : r - s;
        Normalize(ref state, buf, bufStart, storage);
        return ret;
    }

    /// <summary>
    /// Read <paramref name="bits"/> raw bits from the END of the
    /// stream (back-window). Used for SILK pulses / shell-coder
    /// excitation reads. Mirrors libopus <c>ec_dec_bits</c>.
    /// </summary>
    public static uint DecodeBits(
        ref OpusRangeDecoderGpuState state,
        ArrayView<byte> buf, int bufStart, uint storage,
        int bits)
    {
        uint window = state.EndWindow;
        int available = state.NEndBits;
        if (available < bits)
        {
            do
            {
                window |= (uint)ReadByteFromEnd(ref state, buf, bufStart, storage) << available;
                available += EC_SYM_BITS;
            }
            while (available <= EC_WINDOW_SIZE - EC_SYM_BITS);
        }
        uint ret = window & ((1u << bits) - 1u);
        window >>= bits;
        available -= bits;
        state.EndWindow = window;
        state.NEndBits = available;
        state.NBitsTotal += bits;
        return ret;
    }

    /// <summary>
    /// Normalize the (rng, val) coder state, pulling additional bytes
    /// from the front of the buffer as needed. Mirrors libopus
    /// <c>ec_dec_normalize</c>.
    /// </summary>
    public static void Normalize(
        ref OpusRangeDecoderGpuState state,
        ArrayView<byte> buf, int bufStart, uint storage)
    {
        while (state.Rng <= EC_CODE_BOT)
        {
            state.NBitsTotal += EC_SYM_BITS;
            state.Rng <<= EC_SYM_BITS;
            int sym = state.Rem;
            state.Rem = ReadByte(ref state, buf, bufStart, storage);
            sym = (sym << EC_SYM_BITS | state.Rem) >> (EC_SYM_BITS - EC_CODE_EXTRA);
            state.Val = ((state.Val << EC_SYM_BITS) + (EC_SYM_MAX & (uint)~sym)) & (EC_CODE_TOP - 1u);
        }
    }

    private static int ReadByte(
        ref OpusRangeDecoderGpuState state,
        ArrayView<byte> buf, int bufStart, uint storage)
    {
        if (state.Offs < storage)
        {
            int b = buf[bufStart + (int)state.Offs];
            state.Offs++;
            return b;
        }
        return 0;
    }

    private static int ReadByteFromEnd(
        ref OpusRangeDecoderGpuState state,
        ArrayView<byte> buf, int bufStart, uint storage)
    {
        if (state.EndOffs < storage)
        {
            state.EndOffs++;
            return buf[bufStart + (int)(storage - state.EndOffs)];
        }
        return 0;
    }
}
