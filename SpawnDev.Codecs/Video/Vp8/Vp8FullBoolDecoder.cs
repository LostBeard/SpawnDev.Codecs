// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP8 boolean decoder for the full-syntax keyframe decoder. 32-bit only (no 64-bit value register, so it runs on
// every backend including WebGPU), and it detects a truncated partition instead of inventing zero bits.
//
// It is the RFC 6386 section 7.3 decoder: a 16-bit window `Value` over the stream, `Range` in 128..255, and one byte
// shifted in every eight shifted-out bits. The decision for a bit only looks at the top 8 bits of the window, so a
// decision at `Shifts` already-consumed bit positions needs stream bits [Shifts, Shifts + 8), i.e. 1 + ceil(Shifts / 8)
// bytes. If the partition is shorter than that, `Err` is set (this is exactly libwebp's "needed a byte past the end"
// rule; the RFC decoder's eager second byte is NOT counted, so a partition libwebp accepts is accepted here).
// After an error the decoder keeps running on zero bytes so a kernel needs no early exit; the host raises the error.

using ILGPU;

namespace SpawnDev.Codecs.Video.Vp8;

/// <summary>State of one bool decoder (a partition).</summary>
internal struct Vp8Bd
{
    public int Pos;       // next byte to shift in (absolute index into the encoded buffer)
    public int End;       // end of the partition (exclusive, absolute)
    public int Value;
    public int Range;
    public int BitCount;  // 0..7: bits shifted since the last byte was loaded
    public int Shifts;    // total bits shifted out
    public int Len;       // partition length in bytes
    public int Err;       // 1 once a decision needed a byte past the end
}

/// <summary>GPU-callable bool decoder operations.</summary>
internal static class Vp8BdOps
{
    /// <summary>Start a decoder over [start, start + len) of <paramref name="buf"/>.</summary>
    public static Vp8Bd Init(ArrayView<byte> buf, int start, int len)
    {
        var s = new Vp8Bd { Pos = start, End = start + len, Range = 255, BitCount = 0, Shifts = 0, Len = len, Err = 0 };
        int b0 = 0, b1 = 0;
        if (len > 0) b0 = buf[start];
        if (len > 1) b1 = buf[start + 1];
        s.Value = (b0 << 8) | b1;
        s.Pos = start + 2;
        return s;
    }

    /// <summary>Load a decoder from the state buffer.</summary>
    public static Vp8Bd Load(ArrayView<int> st, int slot)
    {
        int b = slot * Vp8FullMeta.BdStateInts;
        var s = new Vp8Bd();
        s.Pos = st[b + 0]; s.End = st[b + 1]; s.Value = st[b + 2]; s.Range = st[b + 3];
        s.BitCount = st[b + 4]; s.Shifts = st[b + 5]; s.Len = st[b + 6]; s.Err = st[b + 7];
        return s;
    }

    /// <summary>Store a decoder to the state buffer.</summary>
    public static void Save(ArrayView<int> st, int slot, Vp8Bd s)
    {
        int b = slot * Vp8FullMeta.BdStateInts;
        st[b + 0] = s.Pos; st[b + 1] = s.End; st[b + 2] = s.Value; st[b + 3] = s.Range;
        st[b + 4] = s.BitCount; st[b + 5] = s.Shifts; st[b + 6] = s.Len; st[b + 7] = s.Err;
    }

    /// <summary>Decode one bit; <paramref name="prob"/> is the probability of a 0, 0..255.</summary>
    public static int Bool(ref Vp8Bd s, ArrayView<byte> buf, int prob)
    {
        int split = 1 + (((s.Range - 1) * prob) >> 8);
        int bigSplit = split << 8;
        int bit = 0;
        int range = split;
        if (s.Value >= bigSplit)
        {
            bit = 1;
            range = s.Range - split;
            s.Value -= bigSplit;
        }
        // The decision window [Shifts, Shifts + 8) must lie inside the partition.
        if (1 + ((s.Shifts + 7) >> 3) > s.Len) s.Err = 1;

        // Renormalise: shift until Range >= 128 (range is 1..255 here).
        int shift = 0;
        if (range < 16) { shift = 4; range <<= 4; }
        if (range < 64) { shift += 2; range <<= 2; }
        if (range < 128) { shift += 1; range <<= 1; }
        s.Range = range;
        if (shift > 0)
        {
            s.Value <<= shift;
            s.Shifts += shift;
            s.BitCount += shift;
            if (s.BitCount >= 8)
            {
                s.BitCount -= 8;
                int next = 0;
                if (s.Pos < s.End) next = buf[s.Pos];
                s.Pos++;
                s.Value |= next << s.BitCount;
            }
        }
        return bit;
    }

    /// <summary>Decode <paramref name="bits"/> raw bits (probability 128), most significant first.</summary>
    public static int Literal(ref Vp8Bd s, ArrayView<byte> buf, int bits)
    {
        int v = 0;
        for (int i = 0; i < bits; i++) v = (v << 1) | Bool(ref s, buf, 128);
        return v;
    }

    /// <summary>A flag bit, then a <paramref name="bits"/>-bit magnitude and a sign bit when the flag is set; else 0.</summary>
    public static int OptionalSigned(ref Vp8Bd s, ArrayView<byte> buf, int bits)
    {
        int present = Bool(ref s, buf, 128);
        int v = 0;
        if (present != 0)
        {
            v = Literal(ref s, buf, bits);
            int neg = Bool(ref s, buf, 128);
            if (neg != 0) v = -v;
        }
        return v;
    }
}
