// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 motion vector. Mirror of libvpx MV { int16_t row, col }.
//
// VP9 stores MVs in 1/8-pel resolution (or 1/4-pel when
// allow_high_precision_mv is false). Components are 14-bit signed
// (libvpx MV_IN_USE_BITS = 14) with the convention:
//
//   row > 0 = downward motion
//   col > 0 = rightward motion
//
// To convert to Q4 (1/16-pel) for the convolve walker, left-shift
// each component by 1 (see <see cref="Vp9MvSubPel.OneEighthPelToQ4"/>).

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 motion vector pair (row + column).</summary>
public readonly record struct Vp9Mv(int Row, int Col)
{
    /// <summary>libvpx <c>MV_IN_USE_BITS</c>: components stored in 14 signed bits.</summary>
    public const int InUseBits = 14;

    /// <summary>libvpx <c>MV_LOW</c>: component lower bound = -16384.</summary>
    public const int Low = -(1 << InUseBits);

    /// <summary>libvpx <c>MV_UPP</c>: component upper bound (exclusive) = 16384.</summary>
    public const int Upp = 1 << InUseBits;

    /// <summary>The zero motion vector.</summary>
    public static readonly Vp9Mv Zero = new Vp9Mv(0, 0);

    /// <summary>Component-wise addition.</summary>
    public static Vp9Mv operator +(Vp9Mv a, Vp9Mv b) =>
        new Vp9Mv(a.Row + b.Row, a.Col + b.Col);

    /// <summary>Component-wise subtraction.</summary>
    public static Vp9Mv operator -(Vp9Mv a, Vp9Mv b) =>
        new Vp9Mv(a.Row - b.Row, a.Col - b.Col);

    /// <summary>Component-wise negation.</summary>
    public static Vp9Mv operator -(Vp9Mv a) =>
        new Vp9Mv(-a.Row, -a.Col);

    /// <summary>True when both components are exactly zero.</summary>
    public bool IsZero => Row == 0 && Col == 0;

    /// <summary>
    /// Clamp this MV's components to the valid VP9 range
    /// [<see cref="Low"/>, <see cref="Upp"/> - 1] = [-16384, 16383].
    /// </summary>
    public Vp9Mv Clamp() => new Vp9Mv(
        Math.Clamp(Row, Low, Upp - 1),
        Math.Clamp(Col, Low, Upp - 1));

    /// <summary>
    /// Clamp this MV's components to a caller-supplied bounding box
    /// in the same units as the MV (1/8-pel for VP9 stored MVs).
    /// Used by libvpx <c>clamp_mv</c> when the reference window for
    /// an MV must stay within the frame plus border.
    /// </summary>
    /// <param name="minRow">Lower bound for the row component.</param>
    /// <param name="maxRow">Upper bound for the row component.</param>
    /// <param name="minCol">Lower bound for the col component.</param>
    /// <param name="maxCol">Upper bound for the col component.</param>
    public Vp9Mv Clamp(int minRow, int maxRow, int minCol, int maxCol)
    {
        if (minRow > maxRow)
            throw new ArgumentException("minRow must be <= maxRow.", nameof(minRow));
        if (minCol > maxCol)
            throw new ArgumentException("minCol must be <= maxCol.", nameof(minCol));
        return new Vp9Mv(
            Math.Clamp(Row, minRow, maxRow),
            Math.Clamp(Col, minCol, maxCol));
    }
}
