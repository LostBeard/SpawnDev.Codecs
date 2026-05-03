// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 intra prediction edge buffer fill helper. Mirror of the
// boundary-fill convention used by libvpx
// vp9/common/vp9_reconintra.c build_intra_predictors() before
// calling the per-mode predictors.
//
// libvpx fills edges that are out-of-frame with neutral mid-range
// constants so the predictor produces a defined (if low-quality)
// output:
//
//   above row missing      -> memset(above, 127, bs)   (and 2*bs if need_aboveright)
//   above-left corner pix. -> 127 if !up_available
//                          -> 129 if up_available && !left_available
//                          -> caller's reference value otherwise
//   left column missing    -> memset(left, 129, bs)
//
// Plus the right-edge replication: when the above row is available
// but the block is at the right frame boundary, libvpx copies the
// last in-block above sample (above[N-1]) into above[N..2N-1] so
// D45 / D63 (which read 2N samples) have something defined.
//
// This helper does the FILL only - the caller is responsible for
// writing the real edge samples (from already-reconstructed neighbor
// blocks) into the buffers when the edges ARE available. The helper
// runs after the caller's copy and only touches the slots flagged
// as missing.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// VP9 intra prediction edge buffer fill. Bit-exact against libvpx
/// vp9_reconintra.c build_intra_predictors() boundary-fill behavior.
/// </summary>
public static class Vp9IntraEdgeFill
{
    /// <summary>libvpx fill value for a missing above row.</summary>
    public const byte AboveFill = 127;

    /// <summary>libvpx fill value for a missing left column.</summary>
    public const byte LeftFill = 129;

    /// <summary>libvpx fill value for the corner when above is missing.</summary>
    public const byte CornerFillNoAbove = 127;

    /// <summary>
    /// libvpx fill value for the corner when above is available but
    /// left is missing.
    /// </summary>
    public const byte CornerFillAboveOnly = 129;

    /// <summary>
    /// Fill the above buffer with the libvpx default (127) when the
    /// above row is unavailable. When <paramref name="needAboveRight"/>
    /// is true (D45 / D63 modes), 2N samples are filled; otherwise N.
    /// </summary>
    public static void FillMissingAbove(Span<byte> above, int n, bool needAboveRight = false)
    {
        ValidateSize(n);
        int len = needAboveRight ? 2 * n : n;
        if (above.Length < len)
            throw new ArgumentException($"above must hold {len} samples", nameof(above));
        for (int i = 0; i < len; i++) above[i] = AboveFill;
    }

    /// <summary>
    /// Fill the left buffer with the libvpx default (129) when the
    /// left column is unavailable.
    /// </summary>
    public static void FillMissingLeft(Span<byte> left, int n)
    {
        ValidateSize(n);
        if (left.Length < n)
            throw new ArgumentException($"left must hold {n} samples", nameof(left));
        for (int i = 0; i < n; i++) left[i] = LeftFill;
    }

    /// <summary>
    /// Resolve the top-left corner sample per libvpx: 127 when above
    /// is missing, 129 when above is present but left is missing, and
    /// the caller's reference value otherwise.
    /// </summary>
    public static byte ResolveCorner(bool hasAbove, bool hasLeft, byte refValue)
        => hasAbove
            ? (hasLeft ? refValue : CornerFillAboveOnly)
            : CornerFillNoAbove;

    /// <summary>
    /// Replicate <c>above[N-1]</c> into <c>above[N..2N-1]</c> when the
    /// block sits at the right frame boundary so the above-right
    /// extension samples used by D45 / D63 are defined. Caller invokes
    /// this only when the above row IS available but no above-right
    /// neighbor block exists.
    /// </summary>
    public static void ReplicateAboveRight(Span<byte> above, int n)
    {
        ValidateSize(n);
        if (above.Length < 2 * n)
            throw new ArgumentException($"above must hold {2 * n} samples for right-edge replication", nameof(above));
        byte fill = above[n - 1];
        for (int i = n; i < 2 * n; i++) above[i] = fill;
    }

    private static void ValidateSize(int n)
    {
        if (n != 4 && n != 8 && n != 16 && n != 32)
            throw new ArgumentOutOfRangeException(nameof(n), "n must be 4, 8, 16, or 32");
    }
}
