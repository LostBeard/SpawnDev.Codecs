// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 motion vector pair reader. Decodes a (vertical, horizontal)
// diff pair from the compressed bitstream by reading the joint
// type and then conditionally reading each component.
//
// libvpx reference: vp9/decoder/vp9_decodemv.c read_mv (the
// component-reading portion - this slice does NOT add the
// reference MV; that's the caller's job once block context is
// available).
//
// Component ordering follows libvpx convention:
//   probs.Components[0] = vertical
//   probs.Components[1] = horizontal
//
// Read order: VERTICAL FIRST, then HORIZONTAL. Order matters
// because the bool decoder consumes bits sequentially.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 motion vector pair reader.</summary>
public static class Vp9MvPairReader
{
    /// <summary>
    /// libvpx <c>mv_joint_vertical</c>: true when the joint indicates
    /// the vertical component is non-zero (Hzvnz or Hnzvnz).
    /// </summary>
    public static bool JointHasVertical(Vp9MvJointType joint) =>
        joint == Vp9MvJointType.Hzvnz || joint == Vp9MvJointType.Hnzvnz;

    /// <summary>
    /// libvpx <c>mv_joint_horizontal</c>: true when the joint
    /// indicates the horizontal component is non-zero (Hnzvz or
    /// Hnzvnz).
    /// </summary>
    public static bool JointHasHorizontal(Vp9MvJointType joint) =>
        joint == Vp9MvJointType.Hnzvz || joint == Vp9MvJointType.Hnzvnz;

    /// <summary>
    /// Read a (vertical, horizontal) MV diff pair. Returns (0, 0)
    /// when the joint type is <see cref="Vp9MvJointType.Zero"/>.
    /// </summary>
    public static (int Vertical, int Horizontal) ReadDiff(
        Vp9BoolDecoder reader, Vp9MvProbs probs, bool useHp)
    {
        ArgumentNullException.ThrowIfNull(reader);
        return ReadDiff(p => reader.Read(p), probs, useHp);
    }

    /// <summary>
    /// Pure-function variant taking a bit-reader delegate. See
    /// <see cref="Vp9MvComponentReader"/> for the per-component logic.
    /// </summary>
    public static (int Vertical, int Horizontal) ReadDiff(
        Func<byte, int> readBit, Vp9MvProbs probs, bool useHp)
    {
        ArgumentNullException.ThrowIfNull(readBit);
        ArgumentNullException.ThrowIfNull(probs);

        var joint = Vp9MvJointTree.Decode(readBit, probs.Joints);
        int v = JointHasVertical(joint)
            ? Vp9MvComponentReader.ReadComponent(readBit, probs.Components[0], useHp)
            : 0;
        int h = JointHasHorizontal(joint)
            ? Vp9MvComponentReader.ReadComponent(readBit, probs.Components[1], useHp)
            : 0;
        return (v, h);
    }
}
