// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 arithmetic-coded probability update primitive. Mirror of
// libvpx vp9/decoder/vp9_dsubexp.c (vp9_diff_update_prob and the
// helpers it needs: inv_recenter_nonneg, decode_uniform,
// inv_remap_prob, decode_term_subexp).
//
// Used everywhere VP9 entropy probabilities can update mid-frame:
//   read_coef_probs (transform coefficient probs per tx_size)
//   read_tx_mode_probs
//   read_inter_mode_probs / partition_probs / y_mode_prob / etc.
//   read_mv_probs (with a different update-probability constant)
//
// The wire encoding for one update:
//   bit 0: arithmetic-coded with DIFF_UPDATE_PROB=252; 1 means
//          "this prob has an update on this frame".
//   if updated:
//     decode_term_subexp produces a 0..254 delta encoding;
//     inv_remap_prob converts it (using the inv_map_table LUT and
//     a recenter-around-current-prob step) back into a new
//     [1, 255] probability value.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// VP9 arithmetic-coded probability update primitive. Bit-exact
/// against libvpx <c>vp9_diff_update_prob</c>.
/// </summary>
public static class Vp9DiffUpdateProb
{
    /// <summary>libvpx <c>DIFF_UPDATE_PROB</c>.</summary>
    public const int UpdateProb = 252;

    /// <summary>libvpx <c>MAX_PROB</c>.</summary>
    public const int MaxProb = 255;

    /// <summary>
    /// libvpx inv_map_table[255]. The first 19 entries spread early
    /// codes across the probability range; the rest fill in a
    /// monotone permutation. Last entry duplicates 253 to cover
    /// the v=254 input.
    /// </summary>
    public static readonly byte[] InvMapTable = new byte[MaxProb]
    {
        7, 20, 33, 46, 59, 72, 85, 98, 111, 124, 137, 150, 163, 176, 189,
        202, 215, 228, 241, 254, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11,
        12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27,
        28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
        44, 45, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 60,
        61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 73, 74, 75, 76,
        77, 78, 79, 80, 81, 82, 83, 84, 86, 87, 88, 89, 90, 91, 92,
        93, 94, 95, 96, 97, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108,
        109, 110, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 125,
        126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 138, 139, 140, 141,
        142, 143, 144, 145, 146, 147, 148, 149, 151, 152, 153, 154, 155, 156, 157,
        158, 159, 160, 161, 162, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173,
        174, 175, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 190,
        191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 203, 204, 205, 206,
        207, 208, 209, 210, 211, 212, 213, 214, 216, 217, 218, 219, 220, 221, 222,
        223, 224, 225, 226, 227, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238,
        239, 240, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 253
    };

    /// <summary>
    /// Read one arithmetic-coded probability update from <paramref name="reader"/>
    /// for <paramref name="currentProb"/>. Returns the new probability
    /// (which may equal <paramref name="currentProb"/> when no update
    /// was signalled).
    /// </summary>
    public static byte Read(Vp9BoolDecoder reader, byte currentProb)
    {
        ArgumentNullException.ThrowIfNull(reader);
        if (reader.Read(UpdateProb) == 0)
            return currentProb;

        int delp = DecodeTermSubexp(reader);
        return (byte)InvRemapProb(delp, currentProb);
    }

    /// <summary>
    /// libvpx <c>inv_recenter_nonneg(v, m)</c>: maps a non-negative
    /// signed-magnitude delta back around the recenter point.
    /// </summary>
    public static int InvRecenterNonneg(int v, int m)
    {
        if (v > 2 * m) return v;
        return (v & 1) != 0 ? m - ((v + 1) >> 1) : m + (v >> 1);
    }

    /// <summary>
    /// libvpx <c>decode_uniform(r)</c>: 8-bit uniform decoder used as
    /// the tail of the term sub-exp encoding.
    /// </summary>
    public static int DecodeUniform(Vp9BoolDecoder reader)
    {
        ArgumentNullException.ThrowIfNull(reader);
        const int l = 8;
        const int m = (1 << l) - 191;  // 65
        int v = (int)reader.ReadLiteral(l - 1);  // 7 bits
        return v < m ? v : (v << 1) - m + reader.ReadBit();
    }

    /// <summary>
    /// libvpx <c>decode_term_subexp(r)</c>: cascading 4 / 4 / 5 /
    /// uniform-8 bit reads selected by 1-bit gates.
    /// </summary>
    public static int DecodeTermSubexp(Vp9BoolDecoder reader)
    {
        ArgumentNullException.ThrowIfNull(reader);
        if (reader.ReadBit() == 0) return (int)reader.ReadLiteral(4);
        if (reader.ReadBit() == 0) return (int)reader.ReadLiteral(4) + 16;
        if (reader.ReadBit() == 0) return (int)reader.ReadLiteral(5) + 32;
        return DecodeUniform(reader) + 64;
    }

    /// <summary>
    /// libvpx <c>inv_remap_prob(v, m)</c>: convert the term-subexp
    /// delta v back into an absolute probability around the current
    /// probability m. Returns a value in [1, MaxProb].
    /// </summary>
    public static int InvRemapProb(int v, int m)
    {
        if ((uint)v >= MaxProb)
            throw new ArgumentOutOfRangeException(nameof(v),
                "v must be in [0, MaxProb)");
        v = InvMapTable[v];
        m--;
        if ((m << 1) <= MaxProb)
            return 1 + InvRecenterNonneg(v, m);
        return MaxProb - InvRecenterNonneg(v, MaxProb - 1 - m);
    }
}
