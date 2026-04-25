// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 intra prediction unified dispatcher. Routes each of the 10
// intra modes (DC_PRED..TM_PRED, see Vp9IntraMode) to its CPU
// predictor and selects the correct DC variant from the edge-
// availability flags. Mirror of the dispatch shape used by libvpx
// vp9/common/vp9_reconintra.c build_intra_predictors().
//
// Edge buffer convention (caller's responsibility, same as libvpx):
//   - above: at least N samples (2N for D45 / D63 - they read the
//            extension samples at above[N..2N-1]).
//   - left:  at least N samples.
//   - topLeft: the corner sample diagonally above-left of the block
//             (libvpx above[-1]). Ignored by modes that do not use
//             it.
//   - When an edge is out-of-frame, libvpx fills with 129 (above) /
//     127 (left) / 128 (topLeft) before calling the predictor; this
//     dispatcher does NOT pre-fill - it expects the caller to have
//     already populated the buffers consistently with the boundary
//     state. The haveAbove / haveLeft flags here are used only to
//     pick the DC variant.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// Unified VP9 intra prediction entry point. Dispatches the 10 intra
/// modes to their CPU predictors and picks the correct DC variant
/// based on edge availability. Bit-exact against libvpx
/// vp9_reconintra.c build_intra_predictors() per-mode dispatch.
/// </summary>
public static class Vp9IntraPredictor
{
    /// <summary>
    /// Predict an n*n intra block into <paramref name="dst"/>.
    /// </summary>
    /// <param name="mode">VP9 intra mode (0..9).</param>
    /// <param name="topLeft">
    /// Corner sample diagonally above-left of the block. Used by D135 /
    /// D117 / D153 and TM. Ignored by other modes.
    /// </param>
    /// <param name="above">
    /// Above-row samples. Caller must supply at least N entries, or
    /// 2N for D45 / D63. Ignored when <paramref name="mode"/> is
    /// H_PRED or D207_PRED.
    /// </param>
    /// <param name="left">
    /// Left-column samples (N entries). Ignored when
    /// <paramref name="mode"/> is V_PRED, D45_PRED, or D63_PRED.
    /// </param>
    /// <param name="dst">Destination block (n*stride bytes).</param>
    /// <param name="n">Block size (4, 8, 16, or 32).</param>
    /// <param name="stride">Stride in bytes for <paramref name="dst"/>.</param>
    /// <param name="haveAbove">
    /// True when the above row is in-frame. Selects DC variant only.
    /// </param>
    /// <param name="haveLeft">
    /// True when the left column is in-frame. Selects DC variant only.
    /// </param>
    public static void Predict(
        Vp9IntraMode mode,
        byte topLeft,
        ReadOnlySpan<byte> above,
        ReadOnlySpan<byte> left,
        Span<byte> dst, int n, int stride,
        bool haveAbove = true, bool haveLeft = true)
    {
        switch (mode)
        {
            case Vp9IntraMode.DcPred:
                if (haveAbove && haveLeft)
                    Vp9DcPredictor.DcPredict(above, left, dst, n, stride);
                else if (haveAbove)
                    Vp9DcPredictor.DcPredictTop(above, dst, n, stride);
                else if (haveLeft)
                    Vp9DcPredictor.DcPredictLeft(left, dst, n, stride);
                else
                    Vp9DcPredictor.DcPredict128(dst, n, stride);
                break;

            case Vp9IntraMode.VPred:
                Vp9VHPredictor.VPredict(above, dst, n, stride);
                break;

            case Vp9IntraMode.HPred:
                Vp9VHPredictor.HPredict(left, dst, n, stride);
                break;

            case Vp9IntraMode.TmPred:
                Vp9TmPredictor.TmPredict(topLeft, above, left, dst, n, stride);
                break;

            case Vp9IntraMode.D45Pred:
                Vp9DirectionalPredictor.D45Predict(above, dst, n, stride);
                break;

            case Vp9IntraMode.D63Pred:
                Vp9DirectionalPredictor.D63Predict(above, dst, n, stride);
                break;

            case Vp9IntraMode.D135Pred:
                Vp9DirectionalPredictor.D135Predict(topLeft, above, left, dst, n, stride);
                break;

            case Vp9IntraMode.D117Pred:
                Vp9DirectionalPredictor.D117Predict(topLeft, above, left, dst, n, stride);
                break;

            case Vp9IntraMode.D153Pred:
                Vp9DirectionalPredictor.D153Predict(topLeft, above, left, dst, n, stride);
                break;

            case Vp9IntraMode.D207Pred:
                Vp9DirectionalPredictor.D207Predict(left, dst, n, stride);
                break;

            default:
                throw new ArgumentOutOfRangeException(
                    nameof(mode), mode, "Unknown VP9 intra mode");
        }
    }
}
