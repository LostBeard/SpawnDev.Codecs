// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 block-coef decode plane + reference enums extracted to main
// library so the GPU integration kernels don't depend on the CPU
// reference Vp9BlockCoefDecoder (which lives in SpawnDev.Codecs.References).

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 block-coef decode shared enums (libvpx PLANE_TYPES + REF_TYPES).</summary>
public static class Vp9BlockCoefEnums
{
    /// <summary>Plane type for coefficient probability lookup (libvpx PLANE_TYPES).</summary>
    public enum PlaneType
    {
        /// <summary>Luma (Y) plane. PLANE_TYPES index 0.</summary>
        Y = 0,
        /// <summary>Chroma (U/V) plane. PLANE_TYPES index 1.</summary>
        Uv = 1,
    }

    /// <summary>Reference type for coefficient probability lookup (libvpx REF_TYPES).</summary>
    public enum RefType
    {
        /// <summary>Intra-predicted block. REF_TYPES index 0.</summary>
        Intra = 0,
        /// <summary>Inter-predicted block. REF_TYPES index 1.</summary>
        Inter = 1,
    }
}
