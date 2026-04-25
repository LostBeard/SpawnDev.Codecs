// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 motion vector probability storage + parser. Mirror of libvpx
// nmv_context (mv joint probs + per-component sub-tables) and
// vp9/decoder/vp9_decodeframe.c read_mv_probs / update_mv_probs.
//
// Update primitive (different from diff_update_prob):
//   if (read(MV_UPDATE_PROB)) p[i] = (read_literal(7) << 1) | 1;
// New prob is always odd (LSB forced to 1).
//
// Layout per component (libvpx nmv_component):
//   sign        1 entry
//   classes    10 entries (MV_CLASSES - 1)
//   class0      1 entry  (CLASS0_SIZE - 1)
//   bits       10 entries (MV_OFFSET_BITS)
//   class0_fp[CLASS0_SIZE=2]: 3 entries each = 6 entries
//   fp          3 entries (MV_FP_SIZE - 1)
//   class0_hp   1 entry (only when allow_hp)
//   hp          1 entry (only when allow_hp)
// Per-component total: 31 entries (or 33 with HP).
//
// Plus 3 joint probs (MV_JOINTS - 1).
// Total: 3 + 2 * 31 = 65 entries (or 69 with HP).

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// Per-component VP9 motion vector probabilities (libvpx nmv_component).
/// </summary>
public sealed class Vp9MvComponentProbs
{
    /// <summary>libvpx <c>MV_CLASSES</c>.</summary>
    public const int MvClasses = 11;

    /// <summary>libvpx <c>CLASS0_SIZE</c>.</summary>
    public const int Class0Size = 2;

    /// <summary>libvpx <c>MV_OFFSET_BITS</c>.</summary>
    public const int MvOffsetBits = 10;

    /// <summary>libvpx <c>MV_FP_SIZE</c>.</summary>
    public const int MvFpSize = 4;

    /// <summary>1 sign-bit prob.</summary>
    public byte Sign { get; set; }

    /// <summary>10 MV class probs (MV_CLASSES - 1).</summary>
    public byte[] Classes { get; } = new byte[MvClasses - 1];

    /// <summary>1 class-0 prob (CLASS0_SIZE - 1).</summary>
    public byte Class0 { get; set; }

    /// <summary>10 offset-bit probs.</summary>
    public byte[] Bits { get; } = new byte[MvOffsetBits];

    /// <summary>2 sub-arrays of 3 fractional-position class-0 probs each.</summary>
    public byte[,] Class0Fp { get; } = new byte[Class0Size, MvFpSize - 1];

    /// <summary>3 fractional-position probs.</summary>
    public byte[] Fp { get; } = new byte[MvFpSize - 1];

    /// <summary>1 high-precision class-0 prob (used only when allow_hp).</summary>
    public byte Class0Hp { get; set; }

    /// <summary>1 high-precision prob (used only when allow_hp).</summary>
    public byte Hp { get; set; }
}

/// <summary>VP9 motion vector probabilities (libvpx nmv_context).</summary>
public sealed class Vp9MvProbs
{
    /// <summary>libvpx <c>MV_JOINTS</c>.</summary>
    public const int MvJoints = 4;

    /// <summary>libvpx <c>MV_UPDATE_PROB</c>.</summary>
    public const int MvUpdateProb = 252;

    /// <summary>3 joint probs (MV_JOINTS - 1).</summary>
    public byte[] Joints { get; } = new byte[MvJoints - 1];

    /// <summary>Per-component (vertical, horizontal) sub-tables.</summary>
    public Vp9MvComponentProbs[] Components { get; } = new[]
    {
        new Vp9MvComponentProbs(),
        new Vp9MvComponentProbs(),
    };
}

/// <summary>Parser for the read_mv_probs section of the compressed header.</summary>
public static class Vp9MvProbsParser
{
    /// <summary>
    /// libvpx <c>update_mv_probs(p, n, r)</c>: walk n entries, each
    /// with an update flag; if set, replace with a 7-bit literal
    /// shifted to make an odd value. Single-element overload for
    /// the scalar entries (sign, class0, class0_hp, hp).
    /// </summary>
    public static byte UpdateMvProb(Vp9BoolDecoder reader, byte current)
    {
        ArgumentNullException.ThrowIfNull(reader);
        if (reader.Read(Vp9MvProbs.MvUpdateProb) == 0)
            return current;
        return (byte)((reader.ReadLiteral(7) << 1) | 1);
    }

    /// <summary>Update a span of probs in place.</summary>
    public static void UpdateMvProbs(Vp9BoolDecoder reader, Span<byte> probs)
    {
        ArgumentNullException.ThrowIfNull(reader);
        for (int i = 0; i < probs.Length; i++)
            probs[i] = UpdateMvProb(reader, probs[i]);
    }

    /// <summary>
    /// Walk the entire MV probability tree per libvpx
    /// <c>read_mv_probs</c>.
    /// </summary>
    /// <param name="probs">MV probabilities. Modified in place.</param>
    /// <param name="allowHighPrecision">
    /// True when the frame allows 1/8-pel MVs (allow_high_precision_mv);
    /// gates the class0_hp and hp sub-tables.
    /// </param>
    /// <param name="reader">Compressed-header arithmetic reader.</param>
    public static void Read(Vp9MvProbs probs, bool allowHighPrecision, Vp9BoolDecoder reader)
    {
        ArgumentNullException.ThrowIfNull(probs);
        ArgumentNullException.ThrowIfNull(reader);

        // Joints.
        UpdateMvProbs(reader, probs.Joints);

        // Pass 1 over both components: sign, classes, class0, bits.
        for (int i = 0; i < 2; i++)
        {
            var c = probs.Components[i];
            c.Sign = UpdateMvProb(reader, c.Sign);
            UpdateMvProbs(reader, c.Classes);
            c.Class0 = UpdateMvProb(reader, c.Class0);
            UpdateMvProbs(reader, c.Bits);
        }

        // Pass 2 over both components: class0_fp, fp.
        for (int i = 0; i < 2; i++)
        {
            var c = probs.Components[i];
            for (int j = 0; j < Vp9MvComponentProbs.Class0Size; j++)
            {
                Span<byte> row = stackalloc byte[Vp9MvComponentProbs.MvFpSize - 1];
                row[0] = c.Class0Fp[j, 0];
                row[1] = c.Class0Fp[j, 1];
                row[2] = c.Class0Fp[j, 2];
                UpdateMvProbs(reader, row);
                c.Class0Fp[j, 0] = row[0];
                c.Class0Fp[j, 1] = row[1];
                c.Class0Fp[j, 2] = row[2];
            }
            UpdateMvProbs(reader, c.Fp);
        }

        // Optional HP pass.
        if (allowHighPrecision)
        {
            for (int i = 0; i < 2; i++)
            {
                var c = probs.Components[i];
                c.Class0Hp = UpdateMvProb(reader, c.Class0Hp);
                c.Hp = UpdateMvProb(reader, c.Hp);
            }
        }
    }
}
