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

    // Floor 1 inverse-dB table (Vorbis I Section 10.1, normative).
    // This is the verbatim 256-entry static table from the Vorbis I spec
    // (matches libvorbis lib/floor1.c FLOOR1_fromdB_LOOKUP exactly). The
    // table runs from 1.0649863e-7 (i=0) up to 1.0F (i=255). It is NOT a
    // closed-form 10^((i-255)*0.05) curve - the values are normative and
    // must match the spec literally for bit-exact decode.
    private static readonly float[] InverseDbTable =
    {
        1.0649863e-07F, 1.1341951e-07F, 1.2079015e-07F, 1.2863978e-07F,
        1.3699951e-07F, 1.4590251e-07F, 1.5538408e-07F, 1.6548181e-07F,
        1.7623575e-07F, 1.8768855e-07F, 1.9988561e-07F, 2.128753e-07F,
        2.2670913e-07F, 2.4144197e-07F, 2.5713223e-07F, 2.7384213e-07F,
        2.9163793e-07F, 3.1059021e-07F, 3.3077411e-07F, 3.5226968e-07F,
        3.7516214e-07F, 3.9954229e-07F, 4.2550680e-07F, 4.5315863e-07F,
        4.8260743e-07F, 5.1396998e-07F, 5.4737065e-07F, 5.8294187e-07F,
        6.2082472e-07F, 6.6116941e-07F, 7.0413592e-07F, 7.4989464e-07F,
        7.9862701e-07F, 8.5052630e-07F, 9.0579828e-07F, 9.6466216e-07F,
        1.0273513e-06F, 1.0941144e-06F, 1.1652161e-06F, 1.2409384e-06F,
        1.3215816e-06F, 1.4074654e-06F, 1.4989305e-06F, 1.5963394e-06F,
        1.7000785e-06F, 1.8105592e-06F, 1.9282195e-06F, 2.0535261e-06F,
        2.1869758e-06F, 2.3290978e-06F, 2.4804557e-06F, 2.6416497e-06F,
        2.8133190e-06F, 2.9961443e-06F, 3.1908506e-06F, 3.3982101e-06F,
        3.6190449e-06F, 3.8542308e-06F, 4.1047004e-06F, 4.3714470e-06F,
        4.6555282e-06F, 4.9580707e-06F, 5.2802740e-06F, 5.6234160e-06F,
        5.9888572e-06F, 6.3780469e-06F, 6.7925283e-06F, 7.2339451e-06F,
        7.7040476e-06F, 8.2047000e-06F, 8.7378876e-06F, 9.3057248e-06F,
        9.9104632e-06F, 1.0554501e-05F, 1.1240392e-05F, 1.1970856e-05F,
        1.2748789e-05F, 1.3577278e-05F, 1.4459606e-05F, 1.5399272e-05F,
        1.6400004e-05F, 1.7465768e-05F, 1.8600792e-05F, 1.9809576e-05F,
        2.1096914e-05F, 2.2467911e-05F, 2.3928002e-05F, 2.5482978e-05F,
        2.7139006e-05F, 2.8902651e-05F, 3.0780908e-05F, 3.2781225e-05F,
        3.4911534e-05F, 3.7180282e-05F, 3.9596466e-05F, 4.2169667e-05F,
        4.4910090e-05F, 4.7828601e-05F, 5.0936773e-05F, 5.4246931e-05F,
        5.7772202e-05F, 6.1526565e-05F, 6.5524908e-05F, 6.9783085e-05F,
        7.4317983e-05F, 7.9147585e-05F, 8.4291040e-05F, 8.9768747e-05F,
        9.5602426e-05F, 0.00010181521F, 0.00010843174F, 0.00011547824F,
        0.00012298267F, 0.00013097477F, 0.00013948625F, 0.00014855085F,
        0.00015820453F, 0.00016848555F, 0.00017943469F, 0.00019109536F,
        0.00020351382F, 0.00021673929F, 0.00023082423F, 0.00024582449F,
        0.00026179955F, 0.00027881276F, 0.00029693158F, 0.00031622787F,
        0.00033677814F, 0.00035866388F, 0.00038197188F, 0.00040679456F,
        0.00043323036F, 0.00046138411F, 0.00049136745F, 0.00052329927F,
        0.00055730621F, 0.00059352311F, 0.00063209358F, 0.00067317058F,
        0.00071691700F, 0.00076350630F, 0.00081312324F, 0.00086596457F,
        0.00092223983F, 0.00098217216F, 0.0010459992F, 0.0011139742F,
        0.0011863665F, 0.0012634633F, 0.0013455702F, 0.0014330129F,
        0.0015261382F, 0.0016253153F, 0.0017309374F, 0.0018434235F,
        0.0019632195F, 0.0020908006F, 0.0022266726F, 0.0023713743F,
        0.0025254795F, 0.0026895994F, 0.0028643847F, 0.0030505286F,
        0.0032487691F, 0.0034598925F, 0.0036847358F, 0.0039241906F,
        0.0041792066F, 0.0044507950F, 0.0047400328F, 0.0050480668F,
        0.0053761186F, 0.0057254891F, 0.0060975636F, 0.0064938176F,
        0.0069158225F, 0.0073652516F, 0.0078438871F, 0.0083536271F,
        0.0088964928F, 0.009474637F, 0.010090352F, 0.010746080F,
        0.011444421F, 0.012188144F, 0.012980198F, 0.013823725F,
        0.014722068F, 0.015678791F, 0.016697687F, 0.017782797F,
        0.018938423F, 0.020169149F, 0.021479854F, 0.022875735F,
        0.024362330F, 0.025945531F, 0.027631618F, 0.029427276F,
        0.031339626F, 0.033376252F, 0.035545228F, 0.037855157F,
        0.040315199F, 0.042935108F, 0.045725273F, 0.048696758F,
        0.051861348F, 0.055231591F, 0.058820850F, 0.062643361F,
        0.066714279F, 0.071049749F, 0.075666962F, 0.080584227F,
        0.085821044F, 0.091398179F, 0.097337747F, 0.10366330F,
        0.11039993F, 0.11757434F, 0.12521498F, 0.13335215F,
        0.14201813F, 0.15124727F, 0.16107617F, 0.17154380F,
        0.18269168F, 0.19456402F, 0.20720788F, 0.22067342F,
        0.23501402F, 0.25028656F, 0.26655159F, 0.28387361F,
        0.30232132F, 0.32196786F, 0.34289114F, 0.36517414F,
        0.38890521F, 0.41417847F, 0.44109412F, 0.46975890F,
        0.50028648F, 0.53279791F, 0.56742212F, 0.60429640F,
        0.64356699F, 0.68538959F, 0.72993007F, 0.77736504F,
        0.82788260F, 0.88168307F, 0.9389798F, 1.0F,
    };

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
