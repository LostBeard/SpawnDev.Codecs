// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// libwebp's constant tables (Vp8EncTables.g.cs, extracted from the C source) packed into ONE int buffer for the encoder
// kernels (no static tables inside kernels: every backend gets the same uploaded constants). Offsets are named.

namespace SpawnDev.Codecs.Images.WebP.Lossy;

internal static class WebPLossyEncTables
{
    public const int EntropyCost = 0;                          // 256
    public const int LevelFixedCosts = EntropyCost + 256;       // 2048
    public const int EncBands = LevelFixedCosts + 2048;         // 17
    public const int LevelCodes = EncBands + 17;                // 67 x 2
    public const int FixedCostsUV = LevelCodes + 134;           // 4
    public const int FixedCostsI16 = FixedCostsUV + 4;          // 4
    public const int FixedCostsI4 = FixedCostsI16 + 4;          // 10 x 10 x 10
    public const int CoeffsProba0 = FixedCostsI4 + 1000;        // 4 x 8 x 3 x 11
    public const int CoeffsUpdateProba = CoeffsProba0 + 1056;   // 4 x 8 x 3 x 11
    public const int BModesProba = CoeffsUpdateProba + 1056;    // 10 x 10 x 9
    public const int Zigzag = BModesProba + 900;                // 16
    public const int DcTable = Zigzag + 16;                     // 128
    public const int AcTable = DcTable + 128;                   // 128
    public const int AcTable2 = AcTable + 128;                  // 128
    public const int BiasMatrices = AcTable2 + 128;             // 3 x 2
    public const int FreqSharpening = BiasMatrices + 6;         // 16
    public const int WeightY = FreqSharpening + 16;             // 16
    public const int LevelsFromDelta = WeightY + 16;            // 8 x 64
    public const int Cat3 = LevelsFromDelta + 512;              // 3
    public const int Cat4 = Cat3 + 3;                           // 4
    public const int Cat5 = Cat4 + 4;                           // 5
    public const int Cat6 = Cat5 + 5;                           // 11
    public const int Norm = Cat6 + 11;                          // 128
    public const int NewRange = Norm + 128;                     // 128
    public const int Size = NewRange + 128;

    public static int[] Build()
    {
        var t = new int[Size];
        static void Put<T>(int[] dst, int at, T[] src) where T : struct
        {
            for (int i = 0; i < src.Length; i++) dst[at + i] = Convert.ToInt32(src[i]);
        }
        Put(t, EntropyCost, Vp8EncTables.VP8EntropyCost);
        Put(t, LevelFixedCosts, Vp8EncTables.VP8LevelFixedCosts);
        Put(t, EncBands, Vp8EncTables.VP8EncBands);
        Put(t, LevelCodes, Vp8EncTables.VP8LevelCodes);
        Put(t, FixedCostsUV, Vp8EncTables.VP8FixedCostsUV);
        Put(t, FixedCostsI16, Vp8EncTables.VP8FixedCostsI16);
        Put(t, FixedCostsI4, Vp8EncTables.VP8FixedCostsI4);
        Put(t, CoeffsProba0, Vp8EncTables.VP8CoeffsProba0);
        Put(t, CoeffsUpdateProba, Vp8EncTables.VP8CoeffsUpdateProba);
        Put(t, BModesProba, Vp8EncTables.kBModesProba);
        Put(t, Zigzag, Vp8EncTables.kZigzag);
        Put(t, DcTable, Vp8EncTables.kDcTable);
        Put(t, AcTable, Vp8EncTables.kAcTable);
        Put(t, AcTable2, Vp8EncTables.kAcTable2);
        Put(t, BiasMatrices, Vp8EncTables.kBiasMatrices);
        Put(t, FreqSharpening, Vp8EncTables.kFreqSharpening);
        Put(t, WeightY, Vp8EncTables.kWeightY);
        Put(t, LevelsFromDelta, Vp8EncTables.kLevelsFromDelta);
        Put(t, Cat3, Vp8EncTables.VP8Cat3);
        Put(t, Cat4, Vp8EncTables.VP8Cat4);
        Put(t, Cat5, Vp8EncTables.VP8Cat5);
        Put(t, Cat6, Vp8EncTables.VP8Cat6);
        Put(t, Norm, Vp8EncTables.kNorm);
        Put(t, NewRange, Vp8EncTables.kNewRange);
        return t;
    }

    /// <summary>
    /// libwebp VP8SetSegmentParams' power law, q(alpha) = clip((int)(127 * (1 - c_base ^ (1 - amp * alpha))), 0, 127), for
    /// every segment susceptibility alpha in [-127, 127] (index alpha + 127). Built on the host from the CONFIG only (quality,
    /// sns): it needs double-precision pow rounded like libwebp's, which WGSL does not have.
    /// </summary>
    public static int[] BuildQuantTable(float quality, int snsStrength)
    {
        var q = new int[255];
        double amp = 0.9 * snsStrength / 100.0 / 128.0;
        double Q = quality / 100.0;
        double linearC = Q < 0.75 ? Q * (2.0 / 3.0) : 2.0 * Q - 1.0;
        double cBase = Math.Pow(linearC, 1 / 3.0);
        for (int a = -127; a <= 127; a++)
        {
            double expn = 1.0 - amp * a;
            double c = Math.Pow(cBase, expn);
            int v = (int)(127.0 * (1.0 - c));
            q[a + 127] = v < 0 ? 0 : v > 127 ? 127 : v;
        }
        return q;
    }
}
