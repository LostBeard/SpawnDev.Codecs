// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-callable Vorbis Floor 1 curve renderer (orchestrator).
// Mirror of VorbisFloor1Curve.Render. Walks control points + invokes
// the line/point renderers + applies the inverse-dB table to produce
// a complete spectral envelope curve.
//
// Pipeline (Vorbis I sec 7.2.4):
//   - Phase 1: synthesize finalY from decodedY using LowNeighbour /
//     HighNeighbour predictions (RenderPoint per i >= 2).
//   - Phase 2: sort X-list ascending; emit step2-flagged segments via
//     RenderLine; emit trailing constant tail.
//
// Caller flattens VorbisFloor1Config to flat ArrayViews (XList +
// Multiplier scalar) - see VorbisFloor1Config (host-side metadata
// struct setup is allowed under the CARDINAL rule).
//
// Scratch requirements (per call):
//   - scratchInt: 2 * values (finalY[values] + order[values])
//   - scratchByte: values bytes (step2Flag[values])

using ILGPU;

namespace SpawnDev.Codecs.Audio.Vorbis;

/// <summary>
/// GPU-callable Vorbis Floor 1 curve orchestrator. Mirror of
/// <see cref="VorbisFloor1Curve"/>.Render.
/// </summary>
public static class VorbisFloor1RenderCurveGpu
{
    /// <summary>
    /// Render the floor curve into <paramref name="curveOut"/>.
    /// </summary>
    /// <param name="xList">Flat XList array (control point X coords).</param>
    /// <param name="xListBase">Base offset into <paramref name="xList"/>.</param>
    /// <param name="values">Number of control points (XList length).</param>
    /// <param name="decodedY">Flat decoded posterior Y array.</param>
    /// <param name="decodedYBase">Base offset into <paramref name="decodedY"/>.</param>
    /// <param name="multiplier">Floor multiplier (1, 2, 3, or 4 per spec).</param>
    /// <param name="halfBlock">Half block size (output curve length).</param>
    /// <param name="curveOut">Output curve buffer.</param>
    /// <param name="curveOutBase">Base offset into <paramref name="curveOut"/>.</param>
    /// <param name="inverseDbTable">256-entry inverse-dB lookup table.</param>
    /// <param name="inverseDbBase">Base offset into <paramref name="inverseDbTable"/>.</param>
    /// <param name="scratchInt">Per-call int scratch (size 2*values).</param>
    /// <param name="scratchIntBase">Base offset (finalY at +0, order at +values).</param>
    /// <param name="scratchByte">Per-call byte scratch (size values, step2Flag).</param>
    /// <param name="scratchByteBase">Base offset.</param>
    public static void Render(
        ArrayView<int> xList, long xListBase, int values,
        ArrayView<int> decodedY, long decodedYBase,
        int multiplier, int halfBlock,
        ArrayView<float> curveOut, long curveOutBase,
        ArrayView<float> inverseDbTable, long inverseDbBase,
        ArrayView<int> scratchInt, long scratchIntBase,
        ArrayView<byte> scratchByte, long scratchByteBase)
    {
        long finalYBase = scratchIntBase;
        long orderBase = scratchIntBase + values;

        int range = multiplier == 1 ? 256
                  : multiplier == 2 ? 128
                  : multiplier == 3 ? 86
                  : 64;

        // Phase 1: synthesize finalY + step2Flag.
        scratchInt[finalYBase + 0] = decodedY[decodedYBase + 0];
        scratchInt[finalYBase + 1] = decodedY[decodedYBase + 1];
        scratchByte[scratchByteBase + 0] = 1;
        scratchByte[scratchByteBase + 1] = 1;
        for (int i = 2; i < values; i++) scratchByte[scratchByteBase + i] = 0;

        for (int i = 2; i < values; i++)
        {
            int xi = xList[xListBase + i];
            // LowNeighbour (largest x < xi, considering j < i).
            int bestLowX = -1;
            int low = 0;
            // HighNeighbour (smallest x > xi, considering j < i).
            int bestHighX = int.MaxValue;
            int high = 0;
            for (int j = 0; j < i; j++)
            {
                int xj = xList[xListBase + j];
                if (xj < xi && xj > bestLowX) { bestLowX = xj; low = j; }
                if (xj > xi && xj < bestHighX) { bestHighX = xj; high = j; }
            }

            int predicted = VorbisFloor1RenderGpu.RenderPoint(
                xList[xListBase + low], scratchInt[finalYBase + low],
                xList[xListBase + high], scratchInt[finalYBase + high],
                xi);
            int val = decodedY[decodedYBase + i];
            int highRoom = range - predicted;
            int lowRoom = predicted;
            int room = 2 * (lowRoom < highRoom ? lowRoom : highRoom);

            if (val != 0)
            {
                scratchByte[scratchByteBase + low] = 1;
                scratchByte[scratchByteBase + high] = 1;
                scratchByte[scratchByteBase + i] = 1;

                int finalY;
                if (val >= room)
                {
                    finalY = highRoom > lowRoom
                        ? val - lowRoom + predicted
                        : predicted - val + highRoom - 1;
                }
                else
                {
                    finalY = (val & 1) != 0
                        ? predicted - ((val + 1) >> 1)
                        : predicted + (val >> 1);
                }
                scratchInt[finalYBase + i] = finalY;
            }
            else
            {
                scratchInt[finalYBase + i] = predicted;
            }
        }

        // Phase 2: sort X list ascending (insertion sort - simple, stable, OK for small N).
        for (int i = 0; i < values; i++) scratchInt[orderBase + i] = i;
        for (int i = 1; i < values; i++)
        {
            int curIdx = scratchInt[orderBase + i];
            int curX = xList[xListBase + curIdx];
            int j = i - 1;
            while (j >= 0)
            {
                int prevIdx = scratchInt[orderBase + j];
                int prevX = xList[xListBase + prevIdx];
                if (prevX <= curX) break;
                scratchInt[orderBase + j + 1] = prevIdx;
                j--;
            }
            scratchInt[orderBase + j + 1] = curIdx;
        }

        // Phase 3: clear curveOut, then render piecewise-linear segments + tail.
        for (int x = 0; x < halfBlock; x++) curveOut[curveOutBase + x] = 0f;

        int firstIdx = scratchInt[orderBase + 0];
        int hx = 0;
        int lx = 0;
        int ly = scratchInt[finalYBase + firstIdx] * multiplier;

        for (int seg = 1; seg < values; seg++)
        {
            int currentIdx = scratchInt[orderBase + seg];
            if (scratchByte[scratchByteBase + currentIdx] == 0) continue;
            int hy = scratchInt[finalYBase + currentIdx] * multiplier;
            hx = xList[xListBase + currentIdx];
            VorbisFloor1RenderGpu.RenderLine(lx, ly, hx, hy,
                curveOut, curveOutBase, inverseDbTable, inverseDbBase, halfBlock);
            lx = hx;
            ly = hy;
        }

        // Constant tail (Vorbis I "render trailing segment").
        if (hx < halfBlock)
        {
            int yClamp = ly < 0 ? 0 : (ly > 255 ? 255 : ly);
            float tailVal = inverseDbTable[inverseDbBase + yClamp];
            for (int x = hx; x < halfBlock; x++)
                curveOut[curveOutBase + x] = tailVal;
        }
    }
}
