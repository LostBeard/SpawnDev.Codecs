// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 intra prediction dispatcher. Maps a (mode, edge buffer) pair to the
// appropriate <see cref="Av1IntraPredictor"/> routine. For modes that are
// not yet implemented (the directional D45/D67/D113/D135/D157/D203 family),
// falls back to DC prediction so the pipeline can produce *some* output
// rather than throwing partway through a frame.
//
// Spec reference: AV1 Bitstream and Decoding Process Specification
//   sec 7.11.2 Intra prediction process
//   sec 7.11.2.4 Directional intra prediction process

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>AV1 intra prediction dispatch helpers.</summary>
public static class Av1IntraPredictDispatch
{
    /// <summary>libaom <c>ANGLE_STEP</c>: angle delta units are multiplied by 3.</summary>
    private const int AngleStep = 3;

    /// <summary>
    /// Predict a (bw, bh) block from the supplied edge buffers using the requested mode
    /// and (for directional modes) the supplied angle delta in [-3, +3].
    /// Writes the predictor into <paramref name="dst"/> at row stride <paramref name="stride"/>.
    /// </summary>
    public static void Predict(
        Av1IntraMode mode,
        Av1IntraEdge edge,
        Span<byte> dst, int stride, int bw, int bh,
        int angleDelta = 0)
    {
        switch (mode)
        {
            case Av1IntraMode.Dc:
                if (edge.HaveAbove && edge.HaveLeft)
                    Av1IntraPredictor.Dc(dst, stride, bw, bh, edge.Above, edge.Left);
                else if (edge.HaveAbove)
                    Av1IntraPredictor.DcTop(dst, stride, bw, bh, edge.Above, edge.Left);
                else if (edge.HaveLeft)
                    Av1IntraPredictor.DcLeft(dst, stride, bw, bh, edge.Above, edge.Left);
                else
                    Av1IntraPredictor.Dc128(dst, stride, bw, bh, edge.Above, edge.Left);
                break;
            case Av1IntraMode.Vertical:
                if (angleDelta == 0)
                {
                    Av1IntraPredictor.Vertical(dst, stride, bw, bh, edge.Above, edge.Left);
                }
                else
                {
                    int p = 90 + angleDelta * AngleStep;
                    Av1DirectionalPredictor.Predict(p, dst, stride, bw, bh,
                        edge.Above, edge.Left, edge.AboveLeft);
                }
                break;
            case Av1IntraMode.Horizontal:
                if (angleDelta == 0)
                {
                    Av1IntraPredictor.Horizontal(dst, stride, bw, bh, edge.Above, edge.Left);
                }
                else
                {
                    int p = 180 + angleDelta * AngleStep;
                    Av1DirectionalPredictor.Predict(p, dst, stride, bw, bh,
                        edge.Above, edge.Left, edge.AboveLeft);
                }
                break;
            case Av1IntraMode.Smooth:
                Av1IntraPredictor.Smooth(dst, stride, bw, bh, edge.Above, edge.Left);
                break;
            case Av1IntraMode.SmoothV:
                Av1IntraPredictor.SmoothV(dst, stride, bw, bh, edge.Above, edge.Left);
                break;
            case Av1IntraMode.SmoothH:
                Av1IntraPredictor.SmoothH(dst, stride, bw, bh, edge.Above, edge.Left);
                break;
            case Av1IntraMode.Paeth:
            {
                Span<byte> corner = stackalloc byte[1] { edge.AboveLeft };
                Av1IntraPredictor.Paeth(dst, stride, bw, bh, edge.Above, corner, edge.Left);
                break;
            }
            // Directional modes: dispatch through Av1DirectionalPredictor.
            case Av1IntraMode.D45:
            case Av1IntraMode.D67:
            case Av1IntraMode.D113:
            case Av1IntraMode.D135:
            case Av1IntraMode.D157:
            case Av1IntraMode.D203:
            {
                int baseAngle = Av1DirectionalPredictor.ModeToAngleMap[(int)mode];
                int pAngle = baseAngle + angleDelta * AngleStep;
                Av1DirectionalPredictor.Predict(pAngle, dst, stride, bw, bh,
                    edge.Above, edge.Left, edge.AboveLeft);
                break;
            }
            default:
                throw new ArgumentException($"Unknown intra mode {mode}", nameof(mode));
        }
    }
}
