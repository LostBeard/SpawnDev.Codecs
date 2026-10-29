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
// The encoder side (`OpusRangeEncoderGpu`) is a statement-for-statement
// port of libopus celt/entenc.c (via the CPU OpusRangeEncoder in
// SpawnDev.Codecs.References): range-coded symbols grow from the front
// of the buffer, raw bits (ec_enc_bits) from the back, and Done() joins
// them. Byte-identical with Concentus' EntropyCoder.

using ILGPU;

namespace SpawnDev.Codecs.Audio.Opus;

/// <summary>
/// In-kernel state for the Opus range encoder (libopus <c>ec_ctx</c>, encoder fields). The output buffer is passed to each
/// helper as an <c>ArrayView&lt;byte&gt;</c> + start offset + storage length, like the decoder.
/// </summary>
public struct OpusRangeEncoderGpuState
{
    /// <summary>Range-coded bytes written at the front of the buffer.</summary>
    public uint Offs;
    /// <summary>Raw-bit bytes written at the back of the buffer.</summary>
    public uint EndOffs;
    /// <summary>Raw bits not yet flushed to the back of the buffer.</summary>
    public uint EndWindow;
    /// <summary>Bits held in <see cref="EndWindow"/>.</summary>
    public int NEndBits;
    /// <summary>Whole bits used so far plus <c>EC_CODE_BITS + 1</c> (libopus <c>nbits_total</c>).</summary>
    public int NBitsTotal;
    /// <summary>Current range.</summary>
    public uint Rng;
    /// <summary>Low end of the current range.</summary>
    public uint Val;
    /// <summary>Count of buffered 0xFF symbols waiting for a carry (libopus <c>ext</c>).</summary>
    public uint Ext;
    /// <summary>Buffered output byte awaiting a possible carry, or -1 (libopus <c>rem</c>).</summary>
    public int Rem;
    /// <summary>Non-zero after a buffer overflow or an impossible patch (libopus <c>error</c>).</summary>
    public int Error;
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
/// GPU-callable Opus range encoder: libopus <c>celt/entenc.c</c>, statement for statement (the CPU
/// <c>OpusRangeEncoder</c> in SpawnDev.Codecs.References is the same code on managed arrays). Every helper takes the
/// output buffer as <c>buf</c> + <c>bufStart</c> + <c>storage</c> (the packet's byte budget).
/// </summary>
public static class OpusRangeEncoderGpu
{
    private const int EC_SYM_BITS = 8;
    private const int EC_CODE_BITS = 32;
    private const uint EC_SYM_MAX = (1u << EC_SYM_BITS) - 1u;            // 0xFF
    private const int EC_CODE_SHIFT = EC_CODE_BITS - EC_SYM_BITS - 1;    // 23
    private const uint EC_CODE_TOP = 1u << (EC_CODE_BITS - 1);           // 0x80000000
    private const uint EC_CODE_BOT = EC_CODE_TOP >> EC_SYM_BITS;         // 0x00800000
    private const int EC_UINT_BITS = 8;
    private const int EC_WINDOW_SIZE = 32;
    private const int BITRES = 3;

    /// <summary>q15 CDF top value (32768).</summary>
    public const int CdfProbTop = 1 << 15;

    /// <summary>A fresh encoder (libopus <c>ec_enc_init</c>; the buffer is passed to each call).</summary>
    public static OpusRangeEncoderGpuState Init()
    {
        return new OpusRangeEncoderGpuState
        {
            Offs = 0,
            EndOffs = 0,
            EndWindow = 0,
            NEndBits = 0,
            NBitsTotal = EC_CODE_BITS + 1,
            Rng = EC_CODE_TOP,
            Val = 0,
            Ext = 0,
            Rem = -1,
            Error = 0,
        };
    }

    /// <summary>Encode the symbol with cumulative frequencies <c>[fl, fh)</c> out of <paramref name="ft"/> (libopus <c>ec_encode</c>).</summary>
    public static void Encode(ref OpusRangeEncoderGpuState state, ArrayView<byte> buf, int bufStart, uint storage,
        uint fl, uint fh, uint ft)
    {
        uint r = state.Rng / ft;
        if (fl > 0)
        {
            state.Val += state.Rng - r * (ft - fl);
            state.Rng = r * (fh - fl);
        }
        else
        {
            state.Rng -= r * (ft - fh);
        }
        Normalize(ref state, buf, bufStart, storage);
    }

    /// <summary><see cref="Encode"/> with <c>ft = 1 &lt;&lt; bits</c>, without the division (libopus <c>ec_encode_bin</c>).</summary>
    public static void EncodeBin(ref OpusRangeEncoderGpuState state, ArrayView<byte> buf, int bufStart, uint storage,
        uint fl, uint fh, int bits)
    {
        uint r = state.Rng >> bits;
        uint ft = 1u << bits;
        if (fl > 0)
        {
            state.Val += state.Rng - r * (ft - fl);
            state.Rng = r * (fh - fl);
        }
        else
        {
            state.Rng -= r * (ft - fh);
        }
        Normalize(ref state, buf, bufStart, storage);
    }

    /// <summary>Encode a bit that is 1 with probability <c>1 / (1 &lt;&lt; logp)</c> (libopus <c>ec_enc_bit_logp</c>).</summary>
    public static void EncodeBitLogP(ref OpusRangeEncoderGpuState state, ArrayView<byte> buf, int bufStart, uint storage,
        int val, int logp)
    {
        uint r = state.Rng;
        uint l = state.Val;
        uint s = r >> logp;
        r -= s;
        if (val != 0) state.Val = l + r;
        state.Rng = val != 0 ? s : r;
        Normalize(ref state, buf, bufStart, storage);
    }

    /// <summary>Encode symbol <paramref name="s"/> of an inverse-CDF table at <paramref name="icdfBase"/> (libopus <c>ec_enc_icdf</c>).</summary>
    public static void EncodeIcdf(ref OpusRangeEncoderGpuState state, ArrayView<byte> buf, int bufStart, uint storage,
        int s, ArrayView<byte> icdf, long icdfBase, int ftb)
    {
        uint r = state.Rng >> ftb;
        if (s > 0)
        {
            state.Val += state.Rng - r * icdf[icdfBase + s - 1];
            state.Rng = r * (uint)(icdf[icdfBase + s - 1] - icdf[icdfBase + s]);
        }
        else
        {
            state.Rng -= r * icdf[icdfBase + s];
        }
        Normalize(ref state, buf, bufStart, storage);
    }

    /// <summary>16-bit-entry variant of <see cref="EncodeIcdf"/>.</summary>
    public static void EncodeIcdf16(ref OpusRangeEncoderGpuState state, ArrayView<byte> buf, int bufStart, uint storage,
        int s, ArrayView<ushort> icdf, long icdfBase, int ftb)
    {
        uint r = state.Rng >> ftb;
        if (s > 0)
        {
            state.Val += state.Rng - r * icdf[icdfBase + s - 1];
            state.Rng = r * (uint)(icdf[icdfBase + s - 1] - icdf[icdfBase + s]);
        }
        else
        {
            state.Rng -= r * icdf[icdfBase + s];
        }
        Normalize(ref state, buf, bufStart, storage);
    }

    /// <summary>
    /// Encode <paramref name="fl"/> in <c>[0, ft)</c>, <paramref name="ft"/> &gt;= 2 (libopus <c>ec_enc_uint</c>): the top
    /// 8 bits range-coded, the rest as raw bits. A smaller <paramref name="ft"/> sets <see cref="OpusRangeEncoderGpuState.Error"/>.
    /// </summary>
    public static void EncodeUint(ref OpusRangeEncoderGpuState state, ArrayView<byte> buf, int bufStart, uint storage,
        uint fl, uint ft)
    {
        if (ft <= 1u) { state.Error = -1; return; }
        uint decoded = ft - 1u;
        int ftb = EcIlog(decoded);
        if (ftb > EC_UINT_BITS)
        {
            ftb -= EC_UINT_BITS;
            uint scaledFt = (decoded >> ftb) + 1u;
            uint scaledFl = fl >> ftb;
            Encode(ref state, buf, bufStart, storage, scaledFl, scaledFl + 1u, scaledFt);
            EncodeBits(ref state, buf, bufStart, storage, fl & ((1u << ftb) - 1u), ftb);
        }
        else
        {
            Encode(ref state, buf, bufStart, storage, fl, fl + 1u, decoded + 1u);
        }
    }

    /// <summary>Write <paramref name="bits"/> (1..25) raw bits at the back of the buffer (libopus <c>ec_enc_bits</c>).</summary>
    public static void EncodeBits(ref OpusRangeEncoderGpuState state, ArrayView<byte> buf, int bufStart, uint storage,
        uint fl, int bits)
    {
        uint window = state.EndWindow;
        int used = state.NEndBits;
        if (used + bits > EC_WINDOW_SIZE)
        {
            do
            {
                state.Error |= WriteByteAtEnd(ref state, buf, bufStart, storage, window & EC_SYM_MAX);
                window >>= EC_SYM_BITS;
                used -= EC_SYM_BITS;
            }
            while (used >= EC_SYM_BITS);
        }
        window |= fl << used;
        used += bits;
        state.EndWindow = window;
        state.NEndBits = used;
        state.NBitsTotal += bits;
    }

    /// <summary>Overwrite the first <paramref name="nbits"/> (at most 8) bits of the stream (libopus <c>ec_enc_patch_initial_bits</c>).</summary>
    public static void PatchInitialBits(ref OpusRangeEncoderGpuState state, ArrayView<byte> buf, int bufStart,
        uint val, int nbits)
    {
        int shift = EC_SYM_BITS - nbits;
        uint mask = (uint)(((1 << nbits) - 1) << shift);
        if (state.Offs > 0)
        {
            buf[bufStart] = (byte)((buf[bufStart] & ~mask) | (val << shift));
        }
        else if (state.Rem >= 0)
        {
            state.Rem = (int)(((uint)state.Rem & ~mask) | (val << shift));
        }
        else if (state.Rng <= (EC_CODE_TOP >> nbits))
        {
            state.Val = (state.Val & ~(mask << EC_CODE_SHIFT)) | (val << (EC_CODE_SHIFT + shift));
        }
        else
        {
            state.Error = -1;
        }
    }

    /// <summary>
    /// Shrink the packet to <paramref name="newStorage"/> bytes, moving the raw-bit bytes at the back up (libopus
    /// <c>ec_enc_shrink</c>). Returns the new storage; the caller passes it to every later call.
    /// </summary>
    public static uint Shrink(ref OpusRangeEncoderGpuState state, ArrayView<byte> buf, int bufStart, uint storage,
        uint newStorage)
    {
        // memmove(buf + size - end_offs, buf + storage - end_offs, end_offs): the destination is at or before the
        // source, so a forward copy is safe.
        for (uint k = 0; k < state.EndOffs; k++)
            buf[bufStart + (int)(newStorage - state.EndOffs + k)] = buf[bufStart + (int)(storage - state.EndOffs + k)];
        return newStorage;
    }

    /// <summary>Whole bits used so far (libopus <c>ec_tell</c>).</summary>
    public static int Tell(ref OpusRangeEncoderGpuState state) => state.NBitsTotal - EcIlog(state.Rng);

    /// <summary>Bits used so far in 1/8 bits (libopus <c>ec_tell_frac</c>).</summary>
    public static uint TellFrac(ref OpusRangeEncoderGpuState state)
    {
        uint nbits = (uint)state.NBitsTotal << BITRES;
        int l = EcIlog(state.Rng);
        uint r = state.Rng >> (l - 16);
        uint b = (r >> 12) - 8;
        if (r > TellFracCorrection(b)) b++;
        l = (l << 3) + (int)b;
        return nbits - (uint)l;
    }

    // libopus correction[8] = {35733, 38967, 42495, 46340, 50535, 55109, 60097, 65535}, as a switch (no table upload).
    private static uint TellFracCorrection(uint b) => b switch
    {
        0 => 35733u, 1 => 38967u, 2 => 42495u, 3 => 46340u,
        4 => 50535u, 5 => 55109u, 6 => 60097u, _ => 65535u,
    };

    /// <summary>
    /// Finish the packet (libopus <c>ec_enc_done</c>): flush the range coder with the fewest bits that decode
    /// correctly, then the raw bits, and zero the gap between them. The packet is <c>storage</c> bytes.
    /// </summary>
    public static void Done(ref OpusRangeEncoderGpuState state, ArrayView<byte> buf, int bufStart, uint storage)
    {
        int l = EC_CODE_BITS - EcIlog(state.Rng);
        uint msk = (EC_CODE_TOP - 1u) >> l;
        uint end = (state.Val + msk) & ~msk;
        if ((end | msk) >= state.Val + state.Rng)
        {
            l++;
            msk >>= 1;
            end = (state.Val + msk) & ~msk;
        }
        while (l > 0)
        {
            CarryOut(ref state, buf, bufStart, storage, (int)(end >> EC_CODE_SHIFT));
            end = (end << EC_SYM_BITS) & (EC_CODE_TOP - 1u);
            l -= EC_SYM_BITS;
        }
        if (state.Rem >= 0 || state.Ext > 0) CarryOut(ref state, buf, bufStart, storage, 0);

        uint window = state.EndWindow;
        int used = state.NEndBits;
        while (used >= EC_SYM_BITS)
        {
            state.Error |= WriteByteAtEnd(ref state, buf, bufStart, storage, window & EC_SYM_MAX);
            window >>= EC_SYM_BITS;
            used -= EC_SYM_BITS;
        }

        if (state.Error == 0)
        {
            uint gapEnd = storage - state.EndOffs;
            for (uint k = state.Offs; k < gapEnd; k++) buf[bufStart + (int)k] = 0;
            if (used > 0)
            {
                if (state.EndOffs >= storage)
                {
                    state.Error = -1;
                }
                else
                {
                    l = -l;
                    if (state.Offs + state.EndOffs >= storage && l < used)
                    {
                        window &= (uint)((1 << l) - 1);
                        state.Error = -1;
                    }
                    int pos = bufStart + (int)(storage - state.EndOffs - 1u);
                    buf[pos] = (byte)(buf[pos] | window);
                }
            }
        }
    }

    private static int WriteByte(ref OpusRangeEncoderGpuState state, ArrayView<byte> buf, int bufStart, uint storage, uint value)
    {
        if (state.Offs + state.EndOffs >= storage) return -1;
        buf[bufStart + (int)state.Offs] = (byte)value;
        state.Offs++;
        return 0;
    }

    private static int WriteByteAtEnd(ref OpusRangeEncoderGpuState state, ArrayView<byte> buf, int bufStart, uint storage, uint value)
    {
        if (state.Offs + state.EndOffs >= storage) return -1;
        state.EndOffs++;
        buf[bufStart + (int)(storage - state.EndOffs)] = (byte)value;
        return 0;
    }

    // Output a symbol with carry propagation: 0xFF symbols wait in Ext until a later symbol settles the carry.
    private static void CarryOut(ref OpusRangeEncoderGpuState state, ArrayView<byte> buf, int bufStart, uint storage, int c)
    {
        if (c != (int)EC_SYM_MAX)
        {
            int carry = c >> EC_SYM_BITS;
            if (state.Rem >= 0) state.Error |= WriteByte(ref state, buf, bufStart, storage, (uint)(state.Rem + carry));
            if (state.Ext > 0)
            {
                uint sym = (EC_SYM_MAX + (uint)carry) & EC_SYM_MAX;
                do
                {
                    state.Error |= WriteByte(ref state, buf, bufStart, storage, sym);
                    state.Ext--;
                }
                while (state.Ext > 0);
            }
            state.Rem = (int)((uint)c & EC_SYM_MAX);
        }
        else
        {
            state.Ext++;
        }
    }

    private static void Normalize(ref OpusRangeEncoderGpuState state, ArrayView<byte> buf, int bufStart, uint storage)
    {
        while (state.Rng <= EC_CODE_BOT)
        {
            CarryOut(ref state, buf, bufStart, storage, (int)(state.Val >> EC_CODE_SHIFT));
            state.Val = (state.Val << EC_SYM_BITS) & (EC_CODE_TOP - 1u);
            state.Rng <<= EC_SYM_BITS;
            state.NBitsTotal += EC_SYM_BITS;
        }
    }

    private static int EcIlog(uint v)
    {
        int n = 0;
        while (v != 0u) { v >>= 1; n++; }
        return n;
    }
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
    /// Decode a uniformly-distributed integer in <c>[0, ft)</c> with
    /// non-power-of-2 range. Composes <c>ec_decode</c> + <c>ec_dec_update</c>
    /// + (for large ranges) <c>ec_dec_bits</c> per libopus <c>ec_dec_uint</c>.
    /// Used by CELT post-filter octave (range=6), anti-collapse seed
    /// extraction, and bit-allocator residual reads.
    /// </summary>
    /// <param name="state">Range decoder state (advanced in place).</param>
    /// <param name="buf">Encoded packet buffer.</param>
    /// <param name="bufStart">Offset of the packet in <paramref name="buf"/>.</param>
    /// <param name="storage">Length of the packet in bytes.</param>
    /// <param name="ft">Total range; result is in <c>[0, ft)</c>. Caller
    /// must ensure <c>ft &gt;= 2</c> (libopus contract).</param>
    public static uint DecodeUint(
        ref OpusRangeDecoderGpuState state,
        ArrayView<byte> buf, int bufStart, uint storage,
        uint ft)
    {
        const int EC_UINT_BITS = 8;
        if (ft <= 1u) { state.Error = 1; return 0; }
        uint decoded = ft - 1u;
        int ftb = EcIlog(decoded);
        if (ftb > EC_UINT_BITS)
        {
            ftb -= EC_UINT_BITS;
            uint scaledFt = (decoded >> ftb) + 1u;
            // ec_decode(scaledFt): divisive uniform decode.
            state.Ext = state.Rng / scaledFt;
            uint sd = state.Val / state.Ext;
            uint s = scaledFt - (sd + 1u < scaledFt ? sd + 1u : scaledFt);
            // ec_dec_update(s, s+1, scaledFt):
            uint upd = state.Ext * (scaledFt - (s + 1u));
            state.Val -= upd;
            state.Rng = s > 0u ? state.Ext * ((s + 1u) - s) : state.Rng - upd;
            Normalize(ref state, buf, bufStart, storage);
            // Then read raw bits for the bottom-half.
            uint t = (s << ftb) | DecodeBits(ref state, buf, bufStart, storage, ftb);
            if (t <= decoded) return t;
            state.Error = 1;
            return decoded;
        }
        else
        {
            state.Ext = state.Rng / ft;
            uint sd = state.Val / state.Ext;
            uint s = ft - (sd + 1u < ft ? sd + 1u : ft);
            uint upd = state.Ext * (ft - (s + 1u));
            state.Val -= upd;
            state.Rng = s > 0u ? state.Ext * ((s + 1u) - s) : state.Rng - upd;
            Normalize(ref state, buf, bufStart, storage);
            return s;
        }
    }

    /// <summary>Bit position of the highest set bit + 1 (libopus
    /// <c>EC_ILOG</c>). Returns 0 for input 0.</summary>
    private static int EcIlog(uint v)
    {
        int n = 0;
        while (v != 0u) { v >>= 1; n++; }
        return n;
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
