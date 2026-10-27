// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// EmitStep for a dynamic block by the lanes of a group, with the same bytes. A dynamic block's bits after its header are
// the tokens' codes in order, then the end-of-block code, then byte alignment (final block) or a sync flush (3 zero bits,
// alignment, 00 00 FF FF). Every token's bit count is known from the planned codes, so: each lane sums the bits of its
// slice of the tokens, a prefix sum over the lanes gives the bit each slice starts at, and every lane ORs its codes into a
// zeroed word buffer; then the lanes copy the words to the block's output after the header bytes PlanBlock wrote (its
// pending bits, fewer than 8, start the words). Stored blocks (rare: incompressible data) stay with EmitStep, which finds
// the dynamic ones already done.

using ILGPU;

namespace SpawnDev.Codecs.Compression;

public static partial class DeflateGpu
{
    /// <summary>uint words of scratch <see cref="EmitBlockLanes"/> needs per block of <paramref name="blockSize"/> bytes.</summary>
    public static int EmitWordsPerBlock(int blockSize) => MaxBlockOutput(blockSize) / 4 + 4;

    /// <summary>Shared ints <see cref="EmitBlockLanes"/> needs for <paramref name="lanes"/> lanes.</summary>
    public static int EmitSharedInts(int lanes) => lanes;

    /// <summary>
    /// Writes a planned DYNAMIC block (PlanBlock done, phase EmitTokens) by the <paramref name="lanes"/> threads of a group
    /// (<paramref name="lane"/> = Group.IdxX), exactly the bytes EmitStep writes, with <paramref name="words"/> =
    /// <see cref="EmitWordsPerBlock"/> uint words of scratch at <paramref name="wordsAt"/> and <paramref name="part"/> =
    /// <see cref="EmitSharedInts"/> ints of shared memory. Any other block is left as it is (EmitStep handles it).
    /// </summary>
    public static void EmitBlockLanes(bool isFinal, ArrayView<byte> o, int outStart, ArrayView<int> s, int sb, int blockSize,
        ArrayView<uint> words, long wordsAt, ArrayView<int> part, int lane, int lanes)
    {
        int tokAt = sb + HashSize + blockSize;
        int tail = tokAt + blockSize;
        int st = tail + StateOff;
        bool active = Group.BarrierAnd(s[st + StPhase] == PhaseEmitTokens);
        if (active)
        {
            int nWords = EmitWordsPerBlock(blockSize);
            for (int i = lane; i < nWords; i += lanes) words[wordsAt + i] = 0;
            int tokCount = s[st + StTokens];
            int per = (tokCount + lanes - 1) / lanes;
            int from = lane * per;
            int to = from + per;
            if (to > tokCount) to = tokCount;
            int bits = 0;
            for (int t = from; t < to; t++) bits += TokenBits(s, tail, s[tokAt + t]);
            part[lane] = bits;
            Group.Barrier();
            int before = 0, all = 0;
            for (int j = 0; j < lanes; j++)
            {
                int x = part[j];
                all += x;
                if (j < lane) before += x;
            }
            // Bit 0 of the words is bit 0 of the byte at the writer's position; PlanBlock left Count (< 8) bits pending.
            int pending = s[st + StCount];
            int bit = pending + before;
            for (int t = from; t < to; t++) bit = OrToken(words, wordsAt, bit, s, tail, s[tokAt + t]);
            int pos = s[st + StWPos];
            int end = s[st + StWEnd];
            int eobLen = s[tail + LitLenOff + 256];
            int endBit = pending + all + eobLen;
            int alignBits = isFinal ? endBit : endBit + 3;
            int nBytes = (alignBits + 7) >> 3;
            int total = nBytes + (isFinal ? 0 : 4);
            bool overflow = (s[st + StOverflow] != 0) | (pos + total > end);
            if (lane == 0)
            {
                OrWordBits(words, wordsAt, 0, (uint)s[st + StAcc] & ((1u << pending) - 1u), pending);
                OrWordBits(words, wordsAt, pending + all, (uint)s[tail + LitCodeOff + 256], eobLen);
            }
            Group.Barrier();
            if (!overflow)
            {
                for (int k = lane; k < nBytes; k += lanes) o[pos + k] = (byte)(words[wordsAt + (k >> 2)] >> (8 * (k & 3)));
                if ((lane == 0) & !isFinal)
                {
                    o[pos + nBytes] = 0x00;
                    o[pos + nBytes + 1] = 0x00;
                    o[pos + nBytes + 2] = 0xFF;
                    o[pos + nBytes + 3] = 0xFF;
                }
            }
            if (lane == 0)
            {
                s[st + StOutLen] = overflow ? -1 : pos + total - outStart;
                s[st + StPhase] = PhaseDone;
            }
        }
    }

    /// <summary>Bits EmitStep writes for one token (literal, or length code + extra + distance code + extra).</summary>
    private static int TokenBits(ArrayView<int> s, int tail, int tok)
    {
        if (tok < 256) return s[tail + LitLenOff + tok];
        int ml = tok >> 16;
        int md = tok & 0xFFFF;
        return s[tail + LitLenOff + LengthSymbol(ml)] + LengthExtraBits(ml) + s[tail + DistLenOff + DistSymbol(md)] + DistExtraBits(md);
    }

    /// <summary>ORs one token's bits at <paramref name="bit"/> in EmitStep's order; returns the bit after them.</summary>
    private static int OrToken(ArrayView<uint> words, long wordsAt, int bit, ArrayView<int> s, int tail, int tok)
    {
        if (tok < 256)
        {
            int n = s[tail + LitLenOff + tok];
            OrWordBits(words, wordsAt, bit, (uint)s[tail + LitCodeOff + tok], n);
            return bit + n;
        }
        int ml = tok >> 16;
        int md = tok & 0xFFFF;
        int ls = LengthSymbol(ml);
        int n1 = s[tail + LitLenOff + ls];
        OrWordBits(words, wordsAt, bit, (uint)s[tail + LitCodeOff + ls], n1);
        bit += n1;
        int leb = LengthExtraBits(ml);
        if (leb > 0)
        {
            OrWordBits(words, wordsAt, bit, (uint)LengthExtraValue(ml), leb);
            bit += leb;
        }
        int ds = DistSymbol(md);
        int n2 = s[tail + DistLenOff + ds];
        OrWordBits(words, wordsAt, bit, (uint)s[tail + DistCodeOff + ds], n2);
        bit += n2;
        int deb = DistExtraBits(md);
        if (deb > 0)
        {
            OrWordBits(words, wordsAt, bit, (uint)DistExtraValue(md), deb);
            bit += deb;
        }
        return bit;
    }

    /// <summary>ORs the <paramref name="n"/> (at most 31) low bits of <paramref name="v"/> in at <paramref name="bit"/>, LSB first.</summary>
    private static void OrWordBits(ArrayView<uint> words, long wordsAt, int bit, uint v, int n)
    {
        if (n > 0)
        {
            int w = bit >> 5;
            int sh = bit & 31;
            Atomic.Or(ref words[wordsAt + w], v << sh);
            if (sh + n > 32) Atomic.Or(ref words[wordsAt + w + 1], v >> (32 - sh));
        }
    }
}
