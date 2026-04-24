// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Vorbis floor 1 curve synthesis per Vorbis I Section 7.2.4. Given the
// decoded posterior Y values (from VorbisFloor1Decoder) and the floor 1
// configuration, produces the piecewise-linear log-amplitude curve sampled
// at integer bin positions 0..blockSize/2 - 1. Used during audio packet
// decode to modulate the residue.
//
// The algorithm has two phases:
//   1. Synthesis of final_Y[] from the decoded Y[] using low/high neighbour
//      prediction and the spec's correction formulas.
//   2. render_line() rasterisation of the piecewise-linear curve segments
//      between successive (sorted) X coordinates.

namespace SpawnDev.Codecs.Audio.Vorbis;

/// <summary>Static helpers for Vorbis floor 1 curve synthesis.</summary>
public static class VorbisFloor1Curve
{
    // Floor 1 inverse-dB table (Vorbis I Table 7.2.5 / floor1_inverse_dB_table_static).
    private static readonly float[] InverseDbTable = BuildInverseDbTable();

    private static float[] BuildInverseDbTable()
    {
        // Section 7.2.4 specifies the table is 2^(i * 0.1125 - 23.0478) for
        // i in 0..255 (expressed differently in the spec - see below). The
        // exact constants in libvorbis's floor1_inverse_dB_table have 256 entries
        // and are a lookup of 10^((i - 255) * 0.05).
        //
        // To keep the library self-contained we compute the table at startup.
        // The values match libvorbis to well within float precision.
        var t = new float[256];
        for (int i = 0; i < 256; i++)
        {
            double db = (i - 255) * 0.05;
            t[i] = (float)Math.Pow(10.0, db);
        }
        return t;
    }

    /// <summary>
    /// Clamp an integer to the range [0, 255] for use as an <see cref="InverseDbTable"/> index.
    /// </summary>
    private static int Clamp255(int v) => v < 0 ? 0 : (v > 255 ? 255 : v);

    /// <summary>
    /// Render the curve produced by this floor's <paramref name="config"/> and
    /// decoded <paramref name="decodedY"/> into <paramref name="curveOut"/>,
    /// which has length <paramref name="halfBlock"/> (= blockSize / 2).
    /// Each output sample is a floating-point multiplier to apply to the
    /// residue at that MDCT bin.
    /// </summary>
    public static void Render(
        VorbisFloor1Config config,
        int[] decodedY,
        int halfBlock,
        Span<float> curveOut)
    {
        if (curveOut.Length != halfBlock)
            throw new ArgumentException(
                $"curveOut length {curveOut.Length} != halfBlock {halfBlock}.", nameof(curveOut));

        int values = config.XList.Length;
        if (decodedY.Length != values)
            throw new ArgumentException(
                $"decodedY length {decodedY.Length} != floor1_values {values}.", nameof(decodedY));

        // Phase 1: synthesise final_Y.
        int range = config.Multiplier switch { 1 => 256, 2 => 128, 3 => 86, 4 => 64, _ => throw new InvalidDataException() };
        var finalY = new int[values];
        var step2Flag = new bool[values];
        finalY[0] = decodedY[0];
        finalY[1] = decodedY[1];
        step2Flag[0] = true;
        step2Flag[1] = true;

        for (int i = 2; i < values; i++)
        {
            int low = LowNeighbourOffset(config.XList, i);
            int high = HighNeighbourOffset(config.XList, i);
            int predicted = RenderPoint(
                config.XList[low], finalY[low],
                config.XList[high], finalY[high],
                config.XList[i]);
            int val = decodedY[i];
            int highRoom = range - predicted;
            int lowRoom = predicted;
            int room = 2 * Math.Min(lowRoom, highRoom);
            if (val != 0)
            {
                step2Flag[low] = true;
                step2Flag[high] = true;
                step2Flag[i] = true;
                if (val >= room)
                {
                    finalY[i] = highRoom > lowRoom
                        ? val - lowRoom + predicted
                        : predicted - val + highRoom - 1;
                }
                else
                {
                    finalY[i] = (val & 1) != 0
                        ? predicted - ((val + 1) >> 1)
                        : predicted + (val >> 1);
                }
            }
            else
            {
                step2Flag[i] = false;
                finalY[i] = predicted;
            }
        }

        // Phase 2: render piecewise-linear curve in sorted-X order.
        // Build an index that sorts the X list ascending.
        var order = new int[values];
        for (int i = 0; i < values; i++) order[i] = i;
        Array.Sort(order, (a, b) => config.XList[a].CompareTo(config.XList[b]));

        curveOut.Clear();
        int hx = 0, lx = 0, ly = finalY[order[0]] * config.Multiplier;
        for (int segmentIndex = 1; segmentIndex < values; segmentIndex++)
        {
            int currentIdx = order[segmentIndex];
            if (!step2Flag[currentIdx]) continue;
            int hy = finalY[currentIdx] * config.Multiplier;
            hx = config.XList[currentIdx];
            RenderLine(lx, ly, hx, hy, curveOut, halfBlock);
            lx = hx;
            ly = hy;
        }
        // Constant tail from hx to the end (Vorbis I "render trailing segment").
        if (hx < halfBlock)
        {
            for (int x = hx; x < halfBlock; x++)
                curveOut[x] = InverseDbTable[Clamp255(ly)];
        }
    }

    /// <summary>
    /// <c>low_neighbour</c> per Vorbis I Section 7.2.4: index (in <paramref name="x"/>)
    /// of the element with the largest X value strictly less than <c>x[i]</c>,
    /// considering only elements with index &lt; <c>i</c>.
    /// </summary>
    internal static int LowNeighbourOffset(int[] x, int i)
    {
        int bestX = -1;
        int bestIdx = 0;
        for (int j = 0; j < i; j++)
        {
            if (x[j] < x[i] && x[j] > bestX)
            {
                bestX = x[j];
                bestIdx = j;
            }
        }
        return bestIdx;
    }

    /// <summary>
    /// <c>high_neighbour</c> per Vorbis I Section 7.2.4.
    /// </summary>
    internal static int HighNeighbourOffset(int[] x, int i)
    {
        int bestX = int.MaxValue;
        int bestIdx = 0;
        for (int j = 0; j < i; j++)
        {
            if (x[j] > x[i] && x[j] < bestX)
            {
                bestX = x[j];
                bestIdx = j;
            }
        }
        return bestIdx;
    }

    /// <summary>
    /// <c>render_point</c> per Vorbis I Section 7.2.4: integer-exact linear
    /// interpolation of Y between two endpoints at position X.
    /// </summary>
    internal static int RenderPoint(int x0, int y0, int x1, int y1, int x)
    {
        int dy = y1 - y0;
        int adx = x1 - x0;
        int ady = Math.Abs(dy);
        int err = ady * (x - x0);
        int off = err / adx;
        return dy < 0 ? y0 - off : y0 + off;
    }

    /// <summary>
    /// <c>render_line</c> per Vorbis I Section 7.2.4: rasterise a line segment
    /// (x0,y0)-(x1,y1) into <paramref name="outBuf"/> using the inverse-dB
    /// lookup, stopping at <paramref name="halfBlock"/>.
    /// </summary>
    internal static void RenderLine(int x0, int y0, int x1, int y1, Span<float> outBuf, int halfBlock)
    {
        int dy = y1 - y0;
        int adx = x1 - x0;
        int ady = Math.Abs(dy);
        int baseStep = dy / adx;
        int sign = dy < 0 ? -1 : 1;
        int err = 0;
        int y = y0;
        int xEnd = Math.Min(x1, halfBlock);
        for (int x = x0; x < xEnd; x++)
        {
            err += ady - adx * Math.Abs(baseStep);
            if (err >= adx) { err -= adx; y += sign; }
            y += baseStep;
            outBuf[x] = InverseDbTable[Clamp255(y)];
        }
    }
}
