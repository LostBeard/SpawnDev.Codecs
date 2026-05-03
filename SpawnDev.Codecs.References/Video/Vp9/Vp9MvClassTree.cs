// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 motion vector class tree. The magnitude of a non-zero MV
// component is encoded as (class, offset) where class is the
// power-of-two bucket and offset selects within the bucket. The
// 11 classes cover integer-pel MV ranges:
//
//   Class0  : (0,    2]    integer pel
//   Class1  : (2,    4]    integer pel
//   Class2  : (4,    8]    integer pel
//   Class3  : (8,    16]   integer pel
//   Class4  : (16,   32]   integer pel
//   Class5  : (32,   64]   integer pel
//   Class6  : (64,   128]  integer pel
//   Class7  : (128,  256]  integer pel
//   Class8  : (256,  512]  integer pel
//   Class9  : (512,  1024] integer pel
//   Class10 : (1024, 2048] integer pel
//
// libvpx reference: vp9/common/vp9_entropymv.c vp9_mv_class_tree
// (20 entries = 10 internal nodes x 2 branches).
//
// Tree shape (libvpx layout):
//   i=0   : -Class0, 2
//   i=2   : -Class1, 4
//   i=4   :       6, 8
//   i=6   : -Class2, -Class3
//   i=8   :      10, 12
//   i=10  : -Class4, -Class5
//   i=12  : -Class6, 14
//   i=14  :      16, 18
//   i=16  : -Class7, -Class8
//   i=18  : -Class9, -Class10

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 motion vector magnitude class (libvpx MV_CLASS_TYPE).</summary>
public enum Vp9MvClassType : byte
{
    /// <summary>(0, 2] integer pel.</summary>
    Class0 = 0,
    /// <summary>(2, 4] integer pel.</summary>
    Class1 = 1,
    /// <summary>(4, 8] integer pel.</summary>
    Class2 = 2,
    /// <summary>(8, 16] integer pel.</summary>
    Class3 = 3,
    /// <summary>(16, 32] integer pel.</summary>
    Class4 = 4,
    /// <summary>(32, 64] integer pel.</summary>
    Class5 = 5,
    /// <summary>(64, 128] integer pel.</summary>
    Class6 = 6,
    /// <summary>(128, 256] integer pel.</summary>
    Class7 = 7,
    /// <summary>(256, 512] integer pel.</summary>
    Class8 = 8,
    /// <summary>(512, 1024] integer pel.</summary>
    Class9 = 9,
    /// <summary>(1024, 2048] integer pel.</summary>
    Class10 = 10,
}

/// <summary>VP9 motion vector class tree topology and decoder.</summary>
public static class Vp9MvClassTree
{
    /// <summary>libvpx <c>MV_CLASSES</c>.</summary>
    public const int Classes = 11;

    /// <summary>
    /// libvpx <c>vp9_mv_class_tree</c>, 20 entries (10 internal nodes
    /// x 2 branches). Negative values are leaf class types; non-negative
    /// values are byte indices of the next node within this same array.
    /// </summary>
    public static readonly sbyte[] Tree = new sbyte[]
    {
        -(sbyte)Vp9MvClassType.Class0,  2,                                  // 0
        -(sbyte)Vp9MvClassType.Class1,  4,                                  // 2
        6,                              8,                                  // 4
        -(sbyte)Vp9MvClassType.Class2, -(sbyte)Vp9MvClassType.Class3,       // 6
        10,                             12,                                 // 8
        -(sbyte)Vp9MvClassType.Class4, -(sbyte)Vp9MvClassType.Class5,       // 10
        -(sbyte)Vp9MvClassType.Class6,  14,                                 // 12
        16,                             18,                                 // 14
        -(sbyte)Vp9MvClassType.Class7, -(sbyte)Vp9MvClassType.Class8,       // 16
        -(sbyte)Vp9MvClassType.Class9, -(sbyte)Vp9MvClassType.Class10,      // 18
    };

    /// <summary>
    /// Walk the MV class tree given a 10-entry probability vector
    /// (libvpx <c>nmv_component.classes</c>:
    /// <see cref="Vp9MvComponentProbs.Classes"/>).
    /// </summary>
    public static Vp9MvClassType Decode(Func<byte, int> readBit, ReadOnlySpan<byte> probs)
    {
        ArgumentNullException.ThrowIfNull(readBit);
        if (probs.Length < Classes - 1)
            throw new ArgumentException(
                $"probs must hold {Classes - 1} entries for the MV class tree", nameof(probs));

        int i = 0;
        while (true)
        {
            int bit = readBit(probs[i >> 1]);
            sbyte next = Tree[i + bit];
            if (next <= 0)
                return (Vp9MvClassType)(-next);
            i = next;
        }
    }
}
