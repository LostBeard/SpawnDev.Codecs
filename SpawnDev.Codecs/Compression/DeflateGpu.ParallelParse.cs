// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// The LZ77 parse of a block in three parallel-friendly steps, with exactly the tokens ParseStep writes. ParseStep
// inserts every position into the hash chains in order and searches at a position before inserting it, so the chain a
// search at p starts from is "the latest earlier position with the same 3-byte hash", which does not depend on the
// parse at all:
//   1. BuildChainsLanes: 32 lanes per block, 32 positions per step: a position's predecessor is the latest earlier
//      lane of its step with its hash, else the hash head; then the step's last position of each hash becomes the head.
//   2. FindMatchThread: one thread per position, the same chain walk as FindMatch; the match goes to the token array.
//   3. WalkParseStep: one thread per block, ParseStep's greedy or lazy choices from those matches (resumable). It writes
//      the tokens over the matches: token n is written after the match at position p &gt;= n was read, and later reads
//      are at positions after p.
// The literal/length and distance frequencies are not counted here (the GDeflate plan counts the merged tokens).

using ILGPU;

namespace SpawnDev.Codecs.Compression;

public static partial class DeflateGpu
{
    /// <summary>Threads per block of <see cref="BuildChainsLanes"/> (and positions per step).</summary>
    public const int ChainLanes = 32;

    /// <summary>
    /// Step 1, by the <see cref="ChainLanes"/> lanes of a group (<paramref name="lane"/> = Group.IdxX, with
    /// <paramref name="hs"/> = <see cref="ChainLanes"/> ints of shared memory): the hash chains of the whole block, as
    /// ParseStep's insertions leave them. The heads must be cleared (<see cref="InitBlockThread"/>).
    /// </summary>
    public static void BuildChainsLanes(ArrayView<byte> data, int start, int len, ArrayView<int> s, int sb, int blockSize,
        ArrayView<int> hs, int lane)
    {
        int headAt = sb;
        int prevAt = sb + HashSize;
        int st = sb + HashSize + 2 * blockSize + StateOff;
        bool active = Group.BarrierAnd(s[st + StPhase] == PhaseParse);
        if (active)
        {
            int c = 0;
            bool going = Group.BarrierOr(c < len);
            while (going)
            {
                int p = c + lane;
                bool has = p + 3 <= len;
                int h = -1;
                if (has) h = Hash3(data, start + p);
                hs[lane] = h;
                Group.Barrier();
                int pred = -1;
                bool last = has;
                if (has)
                {
                    for (int j = 0; j < ChainLanes; j++)
                    {
                        if (hs[j] == h)
                        {
                            if (j < lane) pred = c + j;
                            else if (j > lane) last = false;
                        }
                    }
                    if (pred < 0) pred = s[headAt + h];
                    s[prevAt + p] = pred;
                }
                // Every lane has read the heads (and the hashes) of this step before any head moves.
                Group.Barrier();
                if (last) s[headAt + h] = p;
                Group.Barrier();
                c += ChainLanes;
                going = Group.BarrierOr(c < len);
            }
        }
    }

    /// <summary>Step 2, thread <paramref name="k"/> (0 .. blockSize - 1): the best match at position k into the token array.</summary>
    public static void FindMatchThread(ArrayView<byte> data, int start, int len, int k, int level, ArrayView<int> s, int sb, int blockSize)
    {
        int prevAt = sb + HashSize;
        int tokAt = prevAt + blockSize;
        int st = tokAt + blockSize + StateOff;
        if ((k < len) & (s[st + StPhase] == PhaseParse))
        {
            int maxChain = level == 1 ? 8 : (level == 2 ? 32 : 64);
            int niceLen = level == 1 ? 32 : (level == 2 ? 128 : 258);
            int cand = -1;
            if (k + 3 <= len) cand = s[prevAt + k];
            int compares = 0;
            s[tokAt + k] = FindMatchFrom(data, start, k, len, s, prevAt, cand, maxChain, niceLen, ref compares);
        }
    }

    /// <summary>
    /// Step 3, one thread per block: continues ParseStep's choices (greedy, or lazy at level 3) from the matches of step
    /// 2 for at most <paramref name="positionBudget"/> positions (plus the rest of the match it is in), then saves its
    /// state. Writes the same tokens as ParseStep; a call after the parse ended does nothing.
    /// </summary>
    public static void WalkParseStep(ArrayView<byte> data, int start, int len, int level, ArrayView<int> s, int sb, int blockSize,
        int positionBudget)
    {
        int tokAt = sb + HashSize + blockSize;
        int st = tokAt + blockSize + StateOff;
        if (s[st + StPhase] != PhaseParse) return;
        bool lazy = level == 3;
        int pos = s[st + StPos];
        int n = s[st + StTokens];
        int pendingLen = s[st + StPendLen];
        int pendingDist = s[st + StPendDist];
        int posLimit = len;
        if (positionBudget < len - pos) posLimit = pos + positionBudget;
        bool going = pos < posLimit;
        while (going)
        {
            int found = s[tokAt + pos];
            int bestLen = found >> 16;
            int bestDist = found & 0xFFFF;
            if (!lazy)
            {
                if (bestLen >= 3)
                {
                    s[tokAt + n] = found;
                    pos += bestLen;
                }
                else
                {
                    s[tokAt + n] = data[start + pos];
                    pos++;
                }
                n++;
            }
            else if (pendingLen > 0)
            {
                if (bestLen > pendingLen)
                {
                    // A longer match starts here: the previous byte becomes a literal.
                    s[tokAt + n] = data[start + pos - 1];
                    n++;
                    pendingLen = bestLen;
                    pendingDist = bestDist;
                    pos++;
                }
                else
                {
                    // Take the pending match (it started at pos - 1).
                    s[tokAt + n] = (pendingLen << 16) | pendingDist;
                    n++;
                    pos = pos - 1 + pendingLen;
                    pendingLen = 0;
                }
            }
            else if (bestLen >= 3)
            {
                pendingLen = bestLen;
                pendingDist = bestDist;
                pos++;
            }
            else
            {
                s[tokAt + n] = data[start + pos];
                n++;
                pos++;
            }
            going = pos < posLimit;
        }
        if (pos >= len)
        {
            if (pendingLen > 0)
            {
                s[tokAt + n] = (pendingLen << 16) | pendingDist;
                n++;
            }
            pendingLen = 0;
            s[st + StPhase] = PhaseParsed;
        }
        s[st + StPos] = pos;
        s[st + StTokens] = n;
        s[st + StPendLen] = pendingLen;
        s[st + StPendDist] = pendingDist;
    }
}
