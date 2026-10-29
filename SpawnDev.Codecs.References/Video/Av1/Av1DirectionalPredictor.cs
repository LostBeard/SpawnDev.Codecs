// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 directional intra prediction. Bit-exact port of libaom
// av1/common/reconintra.c <c>av1_dr_prediction_z1_c</c> /
// <c>av1_dr_prediction_z2_c</c> / <c>av1_dr_prediction_z3_c</c> plus the
// per-mode angle map and the dr_intra_derivative dx/dy lookup.
//
// Upstream Copyright (c) 2016, Alliance for Open Media. All rights reserved.
// Upstream license: BSD 2-Clause + AV1 Patent License 1.0. See NOTICE.md.
//
// AV1 directional modes:
//   D45_PRED  (mode index 3)  -> base angle 45  -> z1 (NE)
//   D135_PRED (mode index 4)  -> base angle 135 -> z2 (NW)
//   D113_PRED (mode index 5)  -> base angle 113 -> z2
//   D157_PRED (mode index 6)  -> base angle 157 -> z2
//   D203_PRED (mode index 7)  -> base angle 203 -> z3 (SW)
//   D67_PRED  (mode index 8)  -> base angle 67  -> z1
//
// p_angle = base_angle + angle_delta * 3.
//
// Spec reference: AV1 Bitstream and Decoding Process Specification
//   sec 7.11.2.4 Directional intra prediction process

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>AV1 directional intra prediction (D45/D67/D113/D135/D157/D203).</summary>
public static class Av1DirectionalPredictor
{
    /// <summary>libaom <c>mode_to_angle_map[INTRA_MODES]</c> from blockd.h.</summary>
    public static readonly int[] ModeToAngleMap = new int[]
    {
        0, 90, 180, 45, 135, 113, 157, 203, 67, 0, 0, 0, 0,
    };

    /// <summary>libaom <c>dr_intra_derivative[90]</c> from reconintra.h.</summary>
    public static readonly int[] DrIntraDerivative = new int[]
    {
        0,    0,    0,
        1023, 0,    0,
        547,  0,    0,
        372,  0,    0, 0, 0,
        273,  0,    0,
        215,  0,    0,
        178,  0,    0,
        151,  0,    0,
        132,  0,    0,
        116,  0,    0,
        102,  0,    0, 0,
        90,   0,    0,
        80,   0,    0,
        71,   0,    0,
        64,   0,    0,
        57,   0,    0,
        51,   0,    0,
        45,   0,    0, 0,
        40,   0,    0,
        35,   0,    0,
        31,   0,    0,
        27,   0,    0,
        23,   0,    0,
        19,   0,    0,
        15,   0,    0, 0, 0,
        11,   0,    0,
        7,    0,    0,
        3,    0,    0,
    };

    /// <summary>libaom <c>av1_get_dx</c>.</summary>
    public static int GetDx(int angle)
    {
        if (angle > 0 && angle < 90) return DrIntraDerivative[angle];
        if (angle > 90 && angle < 180) return DrIntraDerivative[180 - angle];
        return 1;
    }

    /// <summary>libaom <c>av1_get_dy</c>.</summary>
    public static int GetDy(int angle)
    {
        if (angle > 90 && angle < 180) return DrIntraDerivative[angle - 90];
        if (angle > 180 && angle < 270) return DrIntraDerivative[270 - angle];
        return 1;
    }

    /// <summary>
    /// Predict a (bw, bh) block using a directional intra mode. <paramref name="pAngle"/>
    /// is the absolute prediction angle (mode_to_angle_map[mode] + angle_delta * 3).
    /// </summary>
    /// <remarks>
    /// <paramref name="above"/> must contain bw + bh + extra reconstructed pixels
    /// covering at least <c>bw + bh</c> samples; <paramref name="left"/> similarly.
    /// <paramref name="aboveLeft"/> is the top-left corner pixel, used by z2 above[-1].
    /// upsample_above and upsample_left are 0 (we don't run the intra-edge upsampler).
    /// </remarks>
    public static void Predict(int pAngle, Span<byte> dst, int stride, int bw, int bh,
        ReadOnlySpan<byte> above, ReadOnlySpan<byte> left, byte aboveLeft)
    {
        int dx = GetDx(pAngle);
        int dy = GetDy(pAngle);
        if (pAngle > 0 && pAngle < 90)
        {
            DrPredZ1(dst, stride, bw, bh, above, dx);
        }
        else if (pAngle > 90 && pAngle < 180)
        {
            DrPredZ2(dst, stride, bw, bh, above, left, aboveLeft, dx, dy);
        }
        else if (pAngle > 180 && pAngle < 270)
        {
            DrPredZ3(dst, stride, bw, bh, left, dy);
        }
        else
        {
            // pAngle = 90 -> V_PRED, 180 -> H_PRED, 0 / 270 -> degenerate.
            // These should be dispatched as V_PRED / H_PRED by the caller.
            // Fall back to copying the appropriate edge.
            if (pAngle == 90 || pAngle == 0)
            {
                for (int r = 0; r < bh; r++)
                    above.Slice(0, bw).CopyTo(dst.Slice(r * stride, bw));
            }
            else
            {
                for (int r = 0; r < bh; r++)
                    dst.Slice(r * stride, bw).Fill(left[r]);
            }
        }
    }

    /// <summary>
    /// libaom <c>av1_dr_prediction_z1_c</c>: 0 &lt; angle &lt; 90, predict from above.
    /// </summary>
    private static void DrPredZ1(Span<byte> dst, int stride, int bw, int bh,
        ReadOnlySpan<byte> above, int dx)
    {
        // upsample_above = 0 in our pipeline.
        const int upsampleAbove = 0;
        int maxBaseX = ((bw + bh) - 1) << upsampleAbove;
        int fracBits = 6 - upsampleAbove;
        int baseInc = 1 << upsampleAbove;
        int x = dx;
        for (int r = 0; r < bh; r++, x += dx)
        {
            int baseIdx = x >> fracBits;
            int shift = ((x << upsampleAbove) & 0x3F) >> 1;
            if (baseIdx >= maxBaseX)
            {
                // Past the end: rest of block is filled with above[maxBaseX].
                byte fill = above[maxBaseX];
                for (int rr = r; rr < bh; rr++)
                    dst.Slice(rr * stride, bw).Fill(fill);
                return;
            }
            for (int c = 0; c < bw; c++, baseIdx += baseInc)
            {
                if (baseIdx < maxBaseX)
                {
                    int val = above[baseIdx] * (32 - shift) + above[baseIdx + 1] * shift;
                    dst[r * stride + c] = (byte)((val + 16) >> 5);
                }
                else
                {
                    dst[r * stride + c] = above[maxBaseX];
                }
            }
        }
    }

    /// <summary>
    /// libaom <c>av1_dr_prediction_z2_c</c>: 90 &lt; angle &lt; 180, predict from above and left.
    /// </summary>
    private static void DrPredZ2(Span<byte> dst, int stride, int bw, int bh,
        ReadOnlySpan<byte> above, ReadOnlySpan<byte> left, byte aboveLeft, int dx, int dy)
    {
        const int upsampleAbove = 0;
        const int upsampleLeft = 0;
        int minBaseX = -(1 << upsampleAbove);
        int fracBitsX = 6 - upsampleAbove;
        int fracBitsY = 6 - upsampleLeft;
        // libaom's above[base_x] when base_x = -1 returns above[-1] = aboveLeft.
        // We synthesise this by adding 1 to base_x and reading from a virtual
        // array where index 0 = aboveLeft, indices 1.. = above[0..].
        for (int r = 0; r < bh; r++)
        {
            for (int c = 0; c < bw; c++)
            {
                int val;
                int y = r + 1;
                int x = (c << 6) - y * dx;
                int baseX = x >> fracBitsX;
                if (baseX >= minBaseX)
                {
                    int shift = ((x * (1 << upsampleAbove)) & 0x3F) >> 1;
                    byte ax0 = baseX < 0 ? aboveLeft : above[baseX];
                    byte ax1 = (baseX + 1) < 0 ? aboveLeft : above[baseX + 1];
                    val = ax0 * (32 - shift) + ax1 * shift;
                    val = (val + 16) >> 5;
                }
                else
                {
                    int x2 = c + 1;
                    int y2 = (r << 6) - x2 * dy;
                    int baseY = y2 >> fracBitsY;
                    int shift = ((y2 * (1 << upsampleLeft)) & 0x3F) >> 1;
                    byte ly0 = baseY < 0 ? aboveLeft : left[baseY];
                    byte ly1 = (baseY + 1) < 0 ? aboveLeft : left[baseY + 1];
                    val = ly0 * (32 - shift) + ly1 * shift;
                    val = (val + 16) >> 5;
                }
                dst[r * stride + c] = (byte)val;
            }
        }
    }

    /// <summary>
    /// libaom <c>av1_dr_prediction_z3_c</c>: 180 &lt; angle &lt; 270, predict from left.
    /// </summary>
    private static void DrPredZ3(Span<byte> dst, int stride, int bw, int bh,
        ReadOnlySpan<byte> left, int dy)
    {
        const int upsampleLeft = 0;
        int maxBaseY = (bw + bh - 1) << upsampleLeft;
        int fracBits = 6 - upsampleLeft;
        int baseInc = 1 << upsampleLeft;
        int y = dy;
        for (int c = 0; c < bw; c++, y += dy)
        {
            int baseIdx = y >> fracBits;
            int shift = ((y << upsampleLeft) & 0x3F) >> 1;
            for (int r = 0; r < bh; r++, baseIdx += baseInc)
            {
                if (baseIdx < maxBaseY)
                {
                    int val = left[baseIdx] * (32 - shift) + left[baseIdx + 1] * shift;
                    dst[r * stride + c] = (byte)((val + 16) >> 5);
                }
                else
                {
                    // Past the end: fill remaining with left[maxBaseY].
                    byte fill = left[maxBaseY];
                    for (int rr = r; rr < bh; rr++)
                        dst[rr * stride + c] = fill;
                    break;
                }
            }
        }
    }
}
