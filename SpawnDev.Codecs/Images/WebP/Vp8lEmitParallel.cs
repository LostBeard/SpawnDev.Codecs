// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// The main image's token stream of a VP8L file, written in parallel: the same bits Vp8lEncodeGpu.WriteTokens writes one
// token after another (EmitTokensKernel, one thread), after the parallel tokeniser (Vp8lTokenizeParallel) left in its
// scratch which position starts a token (Flag) and the first token of every emit block (BlockCount, after its scan).
//
//   BitsKernel    thread per emit block: the bits of its tokens (code lengths + extra bits; the prefix-code group of a token
//                 is the meta tile of its first pixel, as WriteTokens / the decoder choose it)
//   ScanKernel    one thread: each block's first bit (64-bit sum, after the bits the writer still holds), the writer's
//                 pending bits into word 0, the whole bytes and the bits left over
//   WriteKernel   thread per emit block: its tokens into a 32-bit word scratch at its first bit; neighbouring blocks share
//                 a word at their boundary, so every word is OR-ed in (Atomic.Or; the scratch starts zeroed)
//   CopyKernel    thread per output byte: the whole bytes from the word scratch to the output at the writer's position
//   FinishKernel  one thread: the writer (position, the bits of the partial last byte, overflow) for the tail
//
// Bits are least significant first in both the byte stream and the words (bit k of the stream is bit k & 31 of word
// k >> 5), so the bytes are the words' little-endian bytes. A token is at most 60 bits and covers at least one pixel:
// the scratch holds 2 words per pixel + 2.

using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.Codecs.Images.WebP;

/// <summary>The VP8L token stream written in parallel (see the file header).</summary>
public static class Vp8lEmitParallel
{
    /// <summary>Words of the scratch for <paramref name="pixels"/> pixels.</summary>
    public static long Words(long pixels) => 2 * pixels + 2;

    /// <summary>The bits the writer of a block holds before they fill a word.</summary>
    public struct WordWriter
    {
        /// <summary>The word the next bits go to.</summary>
        public int Word;
        /// <summary>The bits not yet OR-ed into <see cref="Word"/> (low bits first).</summary>
        public uint Acc;
        /// <summary>How many bits <see cref="Acc"/> holds, including the block's start offset in its first word.</summary>
        public int Nbits;
    }

    /// <summary>Appends the low <paramref name="n"/> (0..24) bits of <paramref name="value"/>.</summary>
    private static void Put(ref WordWriter ww, ArrayView<int> words, uint value, int n)
    {
        if (n > 0)
        {
            ww.Acc |= value << ww.Nbits;
            int nb = ww.Nbits + n;
            if (nb >= 32)
            {
                Atomic.Or(ref words[ww.Word], (int)ww.Acc);
                ww.Word++;
                // nb >= 32 with n <= 24 means Nbits >= 8: the shift is 8..24
                ww.Acc = value >> (32 - ww.Nbits);
                ww.Nbits = nb - 32;
            }
            else
            {
                ww.Nbits = nb;
            }
        }
    }

    private static void PutSymbol(ref WordWriter ww, ArrayView<int> words, ArrayView<int> w, int cb, int sym)
    {
        int packed = w[cb + sym];
        Put(ref ww, words, (uint)(packed & 0xFFFF), packed >> 16);
    }

    private static int SymbolBits(ArrayView<int> w, int cb, int sym) => w[cb + sym] >> 16;

    /// <summary>The first int of the packed codes of the prefix-code group of pixel <paramref name="p"/> (WriteTokens' rule).</summary>
    private static int CodesFor(ArrayView<int> w, ArrayView<int> meta, int p, int xsize, int setAlpha)
    {
        int set = meta[WebPEncMeta.RegionMain];
        if (meta[WebPEncMeta.Groups] > 1)
        {
            int metaBits = meta[WebPEncMeta.MetaBits];
            int tilesPerRow = (xsize + (1 << metaBits) - 1) >> metaBits;
            int y = p / xsize;
            int x = p - y * xsize;
            set = meta[WebPEncMeta.RegionGroupSets]
                + w[meta[WebPEncMeta.RegionAssign] + (y >> metaBits) * tilesPerRow + (x >> metaBits)] * Vp8lEncLayout.SetInts;
        }
        return set + 2 * setAlpha;
    }

    /// <summary>Thread b: the bits of emit block b's tokens.</summary>
    public static void BitsKernel(Index1D b, ArrayView<int> s, ArrayView<int> w, ArrayView<int> tok, ArrayView<int> meta,
        ArrayView<int> tables, Vp8lTokenizeParallel.Args a)
    {
        int start = b * Vp8lTokenizeParallel.EmitBlock;
        int end = start + Vp8lTokenizeParallel.EmitBlock;
        if (end > a.N) end = a.N;
        int t = s[a.BlockCount + b];
        int cacheSize = a.CacheBits > 0 ? 1 << a.CacheBits : 0;
        int setAlpha = Vp8lEncodeGpu.SetAlpha(cacheSize);
        int bits = 0;
        for (int p = start; p < end; p++)
        {
            if (s[a.Flag + p] != 0)
            {
                int cb = CodesFor(w, meta, p, a.XSize, setAlpha);
                int t0 = tok[t * 2];
                int type = t0 & 3;
                if (type == 0)
                {
                    uint px = (uint)tok[t * 2 + 1];
                    bits += SymbolBits(w, cb + Vp8lEncodeGpu.AlphaOff(0, cacheSize), (int)((px >> 8) & 0xFFu))
                        + SymbolBits(w, cb + Vp8lEncodeGpu.AlphaOff(1, cacheSize), (int)((px >> 16) & 0xFFu))
                        + SymbolBits(w, cb + Vp8lEncodeGpu.AlphaOff(2, cacheSize), (int)(px & 0xFFu))
                        + SymbolBits(w, cb + Vp8lEncodeGpu.AlphaOff(3, cacheSize), (int)(px >> 24));
                }
                else if (type == 1)
                {
                    bits += SymbolBits(w, cb + Vp8lEncodeGpu.AlphaOff(0, cacheSize), 280 + (t0 >> 2));
                }
                else
                {
                    int lp = Vp8lEncodeGpu.PrefixEncode(t0 >> 2);
                    int dp = Vp8lEncodeGpu.PrefixEncode(Vp8lEncodeGpu.DistanceToPlaneCode(a.XSize, tok[t * 2 + 1], tables));
                    bits += SymbolBits(w, cb + Vp8lEncodeGpu.AlphaOff(0, cacheSize), 256 + (lp & 0xFF)) + ((lp >> 8) & 0x1F)
                        + SymbolBits(w, cb + Vp8lEncodeGpu.AlphaOff(4, cacheSize), dp & 0xFF) + ((dp >> 8) & 0x1F);
                }
                t++;
            }
        }
        s[a.EBits + b] = bits;
    }

    /// <summary>One thread: each block's first word and bit, the writer's pending bits, the bytes and the bits left over.</summary>
    public static void ScanKernel(Index1D _, ArrayView<int> s, ArrayView<int> w, ArrayView<int> words, Vp8lTokenizeParallel.Args a)
    {
        int st = Vp8lEncLayout.StepState;
        int nb0 = w[st + Vp8lEncLayout.SEmitNbits];
        long run = nb0;
        for (int b = 0; b < a.EmitBlocks; b++)
        {
            s[a.EWord + b] = (int)(run >> 5);
            s[a.EShift + b] = (int)(run & 31);
            run += s[a.EBits + b];
        }
        if (nb0 > 0) Atomic.Or(ref words[0], (int)((uint)w[st + Vp8lEncLayout.SEmitAcc] & ((1u << nb0) - 1u)));
        s[a.EState + EBytes] = (int)(run >> 3);
        s[a.EState + ELeft] = (int)(run & 7);
    }

    /// <summary>Slots of the emit state (Vp8lTokenizeParallel.Layout.EState).</summary>
    public const int EBytes = 0, ELeft = 1, EStateInts = 2;

    /// <summary>Thread b: emit block b's tokens at its first bit (WriteTokens' bits, OR-ed into the word scratch).</summary>
    public static void WriteKernel(Index1D b, ArrayView<int> s, ArrayView<int> w, ArrayView<int> tok, ArrayView<int> meta,
        ArrayView<int> tables, ArrayView<int> words, Vp8lTokenizeParallel.Args a)
    {
        int start = b * Vp8lTokenizeParallel.EmitBlock;
        int end = start + Vp8lTokenizeParallel.EmitBlock;
        if (end > a.N) end = a.N;
        int t = s[a.BlockCount + b];
        int cacheSize = a.CacheBits > 0 ? 1 << a.CacheBits : 0;
        int setAlpha = Vp8lEncodeGpu.SetAlpha(cacheSize);
        var ww = new WordWriter { Word = s[a.EWord + b], Acc = 0u, Nbits = s[a.EShift + b] };
        for (int p = start; p < end; p++)
        {
            if (s[a.Flag + p] != 0)
            {
                int cb = CodesFor(w, meta, p, a.XSize, setAlpha);
                int t0 = tok[t * 2];
                int type = t0 & 3;
                if (type == 0)
                {
                    uint px = (uint)tok[t * 2 + 1];
                    PutSymbol(ref ww, words, w, cb + Vp8lEncodeGpu.AlphaOff(0, cacheSize), (int)((px >> 8) & 0xFFu));
                    PutSymbol(ref ww, words, w, cb + Vp8lEncodeGpu.AlphaOff(1, cacheSize), (int)((px >> 16) & 0xFFu));
                    PutSymbol(ref ww, words, w, cb + Vp8lEncodeGpu.AlphaOff(2, cacheSize), (int)(px & 0xFFu));
                    PutSymbol(ref ww, words, w, cb + Vp8lEncodeGpu.AlphaOff(3, cacheSize), (int)(px >> 24));
                }
                else if (type == 1)
                {
                    PutSymbol(ref ww, words, w, cb + Vp8lEncodeGpu.AlphaOff(0, cacheSize), 280 + (t0 >> 2));
                }
                else
                {
                    int lp = Vp8lEncodeGpu.PrefixEncode(t0 >> 2);
                    int dp = Vp8lEncodeGpu.PrefixEncode(Vp8lEncodeGpu.DistanceToPlaneCode(a.XSize, tok[t * 2 + 1], tables));
                    PutSymbol(ref ww, words, w, cb + Vp8lEncodeGpu.AlphaOff(0, cacheSize), 256 + (lp & 0xFF));
                    Put(ref ww, words, (uint)(lp >> 13), (lp >> 8) & 0x1F);
                    PutSymbol(ref ww, words, w, cb + Vp8lEncodeGpu.AlphaOff(4, cacheSize), dp & 0xFF);
                    Put(ref ww, words, (uint)(dp >> 13), (dp >> 8) & 0x1F);
                }
                t++;
            }
        }
        if (ww.Nbits > 0) Atomic.Or(ref words[ww.Word], (int)ww.Acc);
    }

    /// <summary>
    /// Thread b: the per-tile histograms (Vp8lEncodeGroups.TileHistogramsStep's counts) of emit block b's tokens, each in
    /// the meta tile of its first pixel; tiles are shared between blocks, so the counts are atomic adds (sums: the same
    /// histograms in any order). Run after TileHistogramsInit cleared them.
    /// </summary>
    public static void TileHistKernel(Index1D b, ArrayView<int> s, ArrayView<int> w, ArrayView<int> tok, ArrayView<int> meta,
        ArrayView<int> tables, Vp8lTokenizeParallel.Args a)
    {
        int start = b * Vp8lTokenizeParallel.EmitBlock;
        int end = start + Vp8lTokenizeParallel.EmitBlock;
        if (end > a.N) end = a.N;
        int t = s[a.BlockCount + b];
        int cacheSize = a.CacheBits > 0 ? 1 << a.CacheBits : 0;
        int setAlpha = Vp8lEncodeGpu.SetAlpha(cacheSize);
        int bits = meta[WebPEncMeta.MetaBits];
        int tilesPerRow = (a.XSize + (1 << bits) - 1) >> bits;
        int th = meta[WebPEncMeta.RegionTileHist];
        for (int p = start; p < end; p++)
        {
            if (s[a.Flag + p] != 0)
            {
                int y = p / a.XSize;
                int x = p - y * a.XSize;
                int hb = th + ((y >> bits) * tilesPerRow + (x >> bits)) * setAlpha;
                int t0 = tok[t * 2];
                int type = t0 & 3;
                if (type == 0)
                {
                    uint px = (uint)tok[t * 2 + 1];
                    Atomic.Add(ref w[hb + Vp8lEncodeGpu.AlphaOff(0, cacheSize) + (int)((px >> 8) & 0xFFu)], 1);
                    Atomic.Add(ref w[hb + Vp8lEncodeGpu.AlphaOff(1, cacheSize) + (int)((px >> 16) & 0xFFu)], 1);
                    Atomic.Add(ref w[hb + Vp8lEncodeGpu.AlphaOff(2, cacheSize) + (int)(px & 0xFFu)], 1);
                    Atomic.Add(ref w[hb + Vp8lEncodeGpu.AlphaOff(3, cacheSize) + (int)(px >> 24)], 1);
                }
                else if (type == 1)
                {
                    Atomic.Add(ref w[hb + Vp8lEncodeGpu.AlphaOff(0, cacheSize) + 280 + (t0 >> 2)], 1);
                }
                else
                {
                    int lp = Vp8lEncodeGpu.PrefixEncode(t0 >> 2);
                    int dp = Vp8lEncodeGpu.PrefixEncode(Vp8lEncodeGpu.DistanceToPlaneCode(a.XSize, tok[t * 2 + 1], tables));
                    Atomic.Add(ref w[hb + Vp8lEncodeGpu.AlphaOff(0, cacheSize) + 256 + (lp & 0xFF)], 1);
                    Atomic.Add(ref w[hb + Vp8lEncodeGpu.AlphaOff(4, cacheSize) + (dp & 0xFF)], 1);
                }
                t++;
            }
        }
    }

    /// <summary>Thread k: whole byte k of the token stream to the output at the writer's position.</summary>
    public static void CopyKernel(Index1D k, ArrayView<byte> output, ArrayView<int> s, ArrayView<int> w, ArrayView<int> words,
        ArrayView<int> meta, Vp8lTokenizeParallel.Args a)
    {
        if (k < s[a.EState + EBytes])
        {
            int pos = w[Vp8lEncLayout.StepState + Vp8lEncLayout.SEmitPos] + k;
            if (pos < meta[WebPEncMeta.OutputCapacity])
                output[pos] = (byte)(((uint)words[k >> 2] >> ((k & 3) * 8)) & 0xFFu);
        }
    }

    /// <summary>One thread: the writer after the tokens (position, the partial last byte's bits, overflow).</summary>
    public static void FinishKernel(Index1D _, ArrayView<int> s, ArrayView<int> w, ArrayView<int> words, ArrayView<int> meta,
        Vp8lTokenizeParallel.Args a)
    {
        int st = Vp8lEncLayout.StepState;
        int bytes = s[a.EState + EBytes];
        int left = s[a.EState + ELeft];
        int pos = w[st + Vp8lEncLayout.SEmitPos] + bytes;
        uint last = ((uint)words[bytes >> 2] >> ((bytes & 3) * 8)) & 0xFFu;
        w[st + Vp8lEncLayout.SEmitAcc] = left > 0 ? (int)(last & ((1u << left) - 1u)) : 0;
        w[st + Vp8lEncLayout.SEmitNbits] = left;
        w[st + Vp8lEncLayout.SEmitPos] = pos;
        if (pos > meta[WebPEncMeta.OutputCapacity]) w[st + Vp8lEncLayout.SEmitOverflow] = 1;
    }
}
