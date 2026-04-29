// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-callable Vorbis Floor 1 line/point renderer. Mirror of
// VorbisFloor1Curve.RenderPoint + RenderLine for in-kernel use.
//
// These are the inner-most building blocks of Floor 1 curve rendering
// (Vorbis I sec 7.2.4). The outer orchestration (walk control points
// + invoke RenderLine per segment) lives in the future Vorbis GPU
// decoder integration.
//
// RenderLine writes the curve into outBuf[x0..min(x1, halfBlock))
// using the 256-entry inverse-dB lookup. Caller pre-uploads the
// lookup table (VorbisFloor1InverseDbGpu.BuildInverseDbTable).

using ILGPU;

namespace SpawnDev.Codecs.Audio.Vorbis;

/// <summary>
/// GPU-callable Vorbis Floor 1 inner renderers (RenderPoint + RenderLine).
/// </summary>
public static class VorbisFloor1RenderGpu
{
    /// <summary>
    /// Integer-exact linear interpolation of Y between two endpoints at
    /// position X. Mirrors VorbisFloor1Curve.RenderPoint.
    /// </summary>
    public static int RenderPoint(int x0, int y0, int x1, int y1, int x)
    {
        int dy = y1 - y0;
        int adx = x1 - x0;
        int ady = dy < 0 ? -dy : dy;
        int err = ady * (x - x0);
        int off = err / adx;
        return dy < 0 ? y0 - off : y0 + off;
    }

    /// <summary>
    /// Rasterize the line segment (x0,y0)-(x1,y1) into <paramref name="outBuf"/>
    /// starting at <paramref name="outBase"/> using the inverse-dB lookup at
    /// <paramref name="inverseDbTable"/>. Stops at <paramref name="halfBlock"/>.
    /// Mirrors VorbisFloor1Curve.RenderLine.
    /// </summary>
    public static void RenderLine(
        int x0, int y0, int x1, int y1,
        ArrayView<float> outBuf, long outBase,
        ArrayView<float> inverseDbTable, long inverseDbBase,
        int halfBlock)
    {
        int dy = y1 - y0;
        int adx = x1 - x0;
        int ady = dy < 0 ? -dy : dy;
        int baseStep = dy / adx;
        int sign = dy < 0 ? -1 : 1;
        int err = 0;
        int y = y0;
        int absBaseStep = baseStep < 0 ? -baseStep : baseStep;
        int xEnd = x1 < halfBlock ? x1 : halfBlock;
        for (int x = x0; x < xEnd; x++)
        {
            err += ady - adx * absBaseStep;
            if (err >= adx) { err -= adx; y += sign; }
            y += baseStep;
            int yClamp = y < 0 ? 0 : (y > 255 ? 255 : y);
            outBuf[outBase + x] = inverseDbTable[inverseDbBase + yClamp];
        }
    }
}
