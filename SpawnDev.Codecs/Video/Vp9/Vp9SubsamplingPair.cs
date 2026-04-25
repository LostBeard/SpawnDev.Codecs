// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 chroma subsampling pair. VP9 stores subsampling as two bits:
//   ss_x = 1 -> chroma horizontally half-res
//   ss_y = 1 -> chroma vertically half-res
//
// Combinations with named YUV format equivalents:
//   ss_x=0, ss_y=0 -> 4:4:4 (full chroma resolution)
//   ss_x=1, ss_y=0 -> 4:2:2 (chroma half horiz, full vert)
//   ss_x=0, ss_y=1 -> 4:4:0 (chroma full horiz, half vert; uncommon)
//   ss_x=1, ss_y=1 -> 4:2:0 (chroma half both axes; the default
//                            for VP9 Profile 0)
//
// libvpx reference: vp9/common/vp9_blockd.h struct
// macroblockd_plane.subsampling_x / subsampling_y.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 chroma subsampling pair.</summary>
public readonly record struct Vp9SubsamplingPair(int SubsamplingX, int SubsamplingY)
{
    /// <summary>4:4:4 - full chroma resolution.</summary>
    public static readonly Vp9SubsamplingPair Yuv444 = new Vp9SubsamplingPair(0, 0);

    /// <summary>4:2:2 - chroma half horizontal, full vertical.</summary>
    public static readonly Vp9SubsamplingPair Yuv422 = new Vp9SubsamplingPair(1, 0);

    /// <summary>4:4:0 - chroma full horizontal, half vertical.</summary>
    public static readonly Vp9SubsamplingPair Yuv440 = new Vp9SubsamplingPair(0, 1);

    /// <summary>4:2:0 - chroma half both axes (VP9 Profile 0 default).</summary>
    public static readonly Vp9SubsamplingPair Yuv420 = new Vp9SubsamplingPair(1, 1);

    /// <summary>True when both axes are half-resolution.</summary>
    public bool Is420 => SubsamplingX == 1 && SubsamplingY == 1;

    /// <summary>True when neither axis is subsampled.</summary>
    public bool Is444 => SubsamplingX == 0 && SubsamplingY == 0;

    /// <summary>Chroma plane width in pixels for a given luma width.</summary>
    public int ChromaWidth(int lumaWidth) => lumaWidth >> SubsamplingX;

    /// <summary>Chroma plane height in pixels for a given luma height.</summary>
    public int ChromaHeight(int lumaHeight) => lumaHeight >> SubsamplingY;

    /// <summary>Chroma plane size in pixels for a given luma size.</summary>
    public int ChromaPixelCount(int lumaWidth, int lumaHeight) =>
        ChromaWidth(lumaWidth) * ChromaHeight(lumaHeight);
}
