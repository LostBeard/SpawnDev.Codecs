// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 default transform-type and transform-size CDF tables.
//
// Upstream Copyright (c) 2016, Alliance for Open Media. All rights reserved.
// Upstream license: BSD 2-Clause + AV1 Patent License 1.0. See NOTICE.md.
// Upstream commit: 136511836e54093f24d23f02cf93943ff5fc97a2 (libaom main).
// Upstream source: aomedia.googlesource.com/aom av1/common/entropymode.c
//   default_intra_ext_tx_cdf  (lines 178-366)
//   default_inter_ext_tx_cdf  (lines 368-402)
//   default_tx_size_cdf       (lines 872-886)
//
// Constants (av1/common/enums.h + entropymode.h):
//   EXT_TX_SETS_INTRA      = 3
//   EXT_TX_SETS_INTER      = 4
//   EXT_TX_SIZES           = 4   (sizes that use extended transforms)
//   INTRA_MODES            = 13  (DC_PRED .. PAETH_PRED)
//   TX_TYPES               = 16
//   MAX_TX_CATS            = 4   (TX_SIZES - TX_SIZE_CTX_MIN)
//   TX_SIZE_CONTEXTS       = 3
//   MAX_TX_DEPTH           = 2
//
// The intra/inter ext_tx tables are jagged: each transform set selects a
// different active TX_TYPE subset, so per-row symbol counts vary by set.
// Set 0 in both is unused by the bitstream (no choice -> identity decode);
// preserved here as length-1 zero arrays to match the libaom layout.
//
// Stored as inverse CDF (ICDF: CDF_PROB_TOP - cumprob), padded to libaom's
// CDF_SIZE() row width per AOM_CDFn macro semantics. Compatible directly
// with Av1RangeDecoder.DecodeCdfQ15.

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>
/// AV1 default CDFs for transform-type (intra/inter ext_tx) and transform-size
/// (tx_size) entropy decode.
/// </summary>
internal static class Av1DefaultTxfmCdfs
{
    /// <summary>
    /// <c>default_intra_ext_tx_cdf[EXT_TX_SETS_INTRA][EXT_TX_SIZES][INTRA_MODES][CDF_SIZE(TX_TYPES)]</c>.
    /// Per-set, per-tx-size, per-intra-mode CDF over the active TX_TYPE subset for the set.
    /// Set 0 is the no-extended-tx case (single-symbol identity, present as zeros).
    /// </summary>
    public static readonly ushort[][][][] DefaultIntraExtTxCdf =
        new ushort[][][][]
    {
        new ushort[][][]
        {
            new ushort[][]
            {
                new ushort[] { 0 },
                new ushort[] { 0 },
                new ushort[] { 0 },
                new ushort[] { 0 },
                new ushort[] { 0 },
                new ushort[] { 0 },
                new ushort[] { 0 },
                new ushort[] { 0 },
                new ushort[] { 0 },
                new ushort[] { 0 },
                new ushort[] { 0 },
                new ushort[] { 0 },
                new ushort[] { 0 },
            },
            new ushort[][]
            {
                new ushort[] { 0 },
                new ushort[] { 0 },
                new ushort[] { 0 },
                new ushort[] { 0 },
                new ushort[] { 0 },
                new ushort[] { 0 },
                new ushort[] { 0 },
                new ushort[] { 0 },
                new ushort[] { 0 },
                new ushort[] { 0 },
                new ushort[] { 0 },
                new ushort[] { 0 },
                new ushort[] { 0 },
            },
            new ushort[][]
            {
                new ushort[] { 0 },
                new ushort[] { 0 },
                new ushort[] { 0 },
                new ushort[] { 0 },
                new ushort[] { 0 },
                new ushort[] { 0 },
                new ushort[] { 0 },
                new ushort[] { 0 },
                new ushort[] { 0 },
                new ushort[] { 0 },
                new ushort[] { 0 },
                new ushort[] { 0 },
                new ushort[] { 0 },
            },
            new ushort[][]
            {
                new ushort[] { 0 },
                new ushort[] { 0 },
                new ushort[] { 0 },
                new ushort[] { 0 },
                new ushort[] { 0 },
                new ushort[] { 0 },
                new ushort[] { 0 },
                new ushort[] { 0 },
                new ushort[] { 0 },
                new ushort[] { 0 },
                new ushort[] { 0 },
                new ushort[] { 0 },
                new ushort[] { 0 },
            },
        },
        new ushort[][][]
        {
            new ushort[][]
            {
                new ushort[] { 31233, 24733, 23307, 20017, 9301, 4943, 0, 0 },
                new ushort[] { 32204, 29433, 23059, 21898, 14625, 4674, 0, 0 },
                new ushort[] { 32096, 29521, 29092, 20786, 13353, 9641, 0, 0 },
                new ushort[] { 27489, 18883, 17281, 14724, 9241, 2516, 0, 0 },
                new ushort[] { 28345, 26694, 24783, 22352, 7075, 3470, 0, 0 },
                new ushort[] { 31282, 28527, 23308, 22106, 16312, 5074, 0, 0 },
                new ushort[] { 32329, 29930, 29246, 26031, 14710, 9014, 0, 0 },
                new ushort[] { 31578, 28535, 27913, 21098, 12487, 8391, 0, 0 },
                new ushort[] { 31723, 28456, 24121, 22609, 14124, 3433, 0, 0 },
                new ushort[] { 32566, 29034, 28021, 25470, 15641, 8752, 0, 0 },
                new ushort[] { 32321, 28456, 25949, 23884, 16758, 8910, 0, 0 },
                new ushort[] { 32491, 28399, 27513, 23863, 16303, 10497, 0, 0 },
                new ushort[] { 29359, 27332, 22169, 17169, 13081, 8728, 0, 0 },
            },
            new ushort[][]
            {
                new ushort[] { 30898, 19026, 18238, 16270, 8998, 5070, 0, 0 },
                new ushort[] { 32442, 23972, 18136, 17689, 13496, 5282, 0, 0 },
                new ushort[] { 32284, 25192, 25056, 18325, 13609, 10177, 0, 0 },
                new ushort[] { 31642, 17428, 16873, 15745, 11872, 2489, 0, 0 },
                new ushort[] { 32113, 27914, 27519, 26855, 10669, 5630, 0, 0 },
                new ushort[] { 31469, 26310, 23883, 23478, 17917, 7271, 0, 0 },
                new ushort[] { 32457, 27473, 27216, 25883, 16661, 10096, 0, 0 },
                new ushort[] { 31885, 24709, 24498, 21510, 15479, 11219, 0, 0 },
                new ushort[] { 32027, 25188, 23450, 22423, 16080, 3722, 0, 0 },
                new ushort[] { 32658, 25362, 24853, 23573, 16727, 9439, 0, 0 },
                new ushort[] { 32405, 24794, 23411, 22095, 17139, 8294, 0, 0 },
                new ushort[] { 32615, 25121, 24656, 22832, 17461, 12772, 0, 0 },
                new ushort[] { 29257, 26436, 21603, 17433, 13445, 9174, 0, 0 },
            },
            new ushort[][]
            {
                new ushort[] { 28087, 23406, 18725, 14043, 9362, 4681, 0, 0 },
                new ushort[] { 28087, 23406, 18725, 14043, 9362, 4681, 0, 0 },
                new ushort[] { 28087, 23406, 18725, 14043, 9362, 4681, 0, 0 },
                new ushort[] { 28087, 23406, 18725, 14043, 9362, 4681, 0, 0 },
                new ushort[] { 28087, 23406, 18725, 14043, 9362, 4681, 0, 0 },
                new ushort[] { 28087, 23406, 18725, 14043, 9362, 4681, 0, 0 },
                new ushort[] { 28087, 23406, 18725, 14043, 9362, 4681, 0, 0 },
                new ushort[] { 28087, 23406, 18725, 14043, 9362, 4681, 0, 0 },
                new ushort[] { 28087, 23406, 18725, 14043, 9362, 4681, 0, 0 },
                new ushort[] { 28087, 23406, 18725, 14043, 9362, 4681, 0, 0 },
                new ushort[] { 28087, 23406, 18725, 14043, 9362, 4681, 0, 0 },
                new ushort[] { 28087, 23406, 18725, 14043, 9362, 4681, 0, 0 },
                new ushort[] { 28087, 23406, 18725, 14043, 9362, 4681, 0, 0 },
            },
            new ushort[][]
            {
                new ushort[] { 28087, 23406, 18725, 14043, 9362, 4681, 0, 0 },
                new ushort[] { 28087, 23406, 18725, 14043, 9362, 4681, 0, 0 },
                new ushort[] { 28087, 23406, 18725, 14043, 9362, 4681, 0, 0 },
                new ushort[] { 28087, 23406, 18725, 14043, 9362, 4681, 0, 0 },
                new ushort[] { 28087, 23406, 18725, 14043, 9362, 4681, 0, 0 },
                new ushort[] { 28087, 23406, 18725, 14043, 9362, 4681, 0, 0 },
                new ushort[] { 28087, 23406, 18725, 14043, 9362, 4681, 0, 0 },
                new ushort[] { 28087, 23406, 18725, 14043, 9362, 4681, 0, 0 },
                new ushort[] { 28087, 23406, 18725, 14043, 9362, 4681, 0, 0 },
                new ushort[] { 28087, 23406, 18725, 14043, 9362, 4681, 0, 0 },
                new ushort[] { 28087, 23406, 18725, 14043, 9362, 4681, 0, 0 },
                new ushort[] { 28087, 23406, 18725, 14043, 9362, 4681, 0, 0 },
                new ushort[] { 28087, 23406, 18725, 14043, 9362, 4681, 0, 0 },
            },
        },
        new ushort[][][]
        {
            new ushort[][]
            {
                new ushort[] { 26214, 19661, 13107, 6554, 0, 0 },
                new ushort[] { 26214, 19661, 13107, 6554, 0, 0 },
                new ushort[] { 26214, 19661, 13107, 6554, 0, 0 },
                new ushort[] { 26214, 19661, 13107, 6554, 0, 0 },
                new ushort[] { 26214, 19661, 13107, 6554, 0, 0 },
                new ushort[] { 26214, 19661, 13107, 6554, 0, 0 },
                new ushort[] { 26214, 19661, 13107, 6554, 0, 0 },
                new ushort[] { 26214, 19661, 13107, 6554, 0, 0 },
                new ushort[] { 26214, 19661, 13107, 6554, 0, 0 },
                new ushort[] { 26214, 19661, 13107, 6554, 0, 0 },
                new ushort[] { 26214, 19661, 13107, 6554, 0, 0 },
                new ushort[] { 26214, 19661, 13107, 6554, 0, 0 },
                new ushort[] { 26214, 19661, 13107, 6554, 0, 0 },
            },
            new ushort[][]
            {
                new ushort[] { 26214, 19661, 13107, 6554, 0, 0 },
                new ushort[] { 26214, 19661, 13107, 6554, 0, 0 },
                new ushort[] { 26214, 19661, 13107, 6554, 0, 0 },
                new ushort[] { 26214, 19661, 13107, 6554, 0, 0 },
                new ushort[] { 26214, 19661, 13107, 6554, 0, 0 },
                new ushort[] { 26214, 19661, 13107, 6554, 0, 0 },
                new ushort[] { 26214, 19661, 13107, 6554, 0, 0 },
                new ushort[] { 26214, 19661, 13107, 6554, 0, 0 },
                new ushort[] { 26214, 19661, 13107, 6554, 0, 0 },
                new ushort[] { 26214, 19661, 13107, 6554, 0, 0 },
                new ushort[] { 26214, 19661, 13107, 6554, 0, 0 },
                new ushort[] { 26214, 19661, 13107, 6554, 0, 0 },
                new ushort[] { 26214, 19661, 13107, 6554, 0, 0 },
            },
            new ushort[][]
            {
                new ushort[] { 31641, 19954, 9996, 5285, 0, 0 },
                new ushort[] { 32623, 26007, 20788, 6101, 0, 0 },
                new ushort[] { 32406, 26881, 21090, 16043, 0, 0 },
                new ushort[] { 32383, 17555, 14181, 2075, 0, 0 },
                new ushort[] { 32743, 29854, 9634, 4865, 0, 0 },
                new ushort[] { 32708, 28298, 21019, 8777, 0, 0 },
                new ushort[] { 32731, 29436, 18257, 11320, 0, 0 },
                new ushort[] { 32611, 26448, 19732, 15329, 0, 0 },
                new ushort[] { 32649, 26049, 19862, 3372, 0, 0 },
                new ushort[] { 32721, 27231, 20192, 11269, 0, 0 },
                new ushort[] { 32499, 26692, 21510, 9653, 0, 0 },
                new ushort[] { 32685, 27153, 20767, 15540, 0, 0 },
                new ushort[] { 30800, 27212, 20745, 14221, 0, 0 },
            },
            new ushort[][]
            {
                new ushort[] { 26214, 19661, 13107, 6554, 0, 0 },
                new ushort[] { 26214, 19661, 13107, 6554, 0, 0 },
                new ushort[] { 26214, 19661, 13107, 6554, 0, 0 },
                new ushort[] { 26214, 19661, 13107, 6554, 0, 0 },
                new ushort[] { 26214, 19661, 13107, 6554, 0, 0 },
                new ushort[] { 26214, 19661, 13107, 6554, 0, 0 },
                new ushort[] { 26214, 19661, 13107, 6554, 0, 0 },
                new ushort[] { 26214, 19661, 13107, 6554, 0, 0 },
                new ushort[] { 26214, 19661, 13107, 6554, 0, 0 },
                new ushort[] { 26214, 19661, 13107, 6554, 0, 0 },
                new ushort[] { 26214, 19661, 13107, 6554, 0, 0 },
                new ushort[] { 26214, 19661, 13107, 6554, 0, 0 },
                new ushort[] { 26214, 19661, 13107, 6554, 0, 0 },
            },
        },
    };

    /// <summary>
    /// <c>default_inter_ext_tx_cdf[EXT_TX_SETS_INTER][EXT_TX_SIZES][CDF_SIZE(TX_TYPES)]</c>.
    /// Per-set, per-tx-size CDF over the active TX_TYPE subset for the set.
    /// Set 0 is the no-extended-tx case (single-symbol identity, present as zeros).
    /// </summary>
    public static readonly ushort[][][] DefaultInterExtTxCdf =
        new ushort[][][]
    {
        new ushort[][]
        {
            new ushort[] { 0 },
            new ushort[] { 0 },
            new ushort[] { 0 },
            new ushort[] { 0 },
        },
        new ushort[][]
        {
            new ushort[] { 28310, 27208, 25073, 23059, 19438, 17979, 15231, 12502, 11264, 9920, 8834, 7294, 5041, 3853, 2137, 0, 0 },
            new ushort[] { 31123, 30195, 27990, 27057, 24961, 24146, 22246, 17411, 15094, 12360, 10251, 7758, 5652, 3912, 2019, 0, 0 },
            new ushort[] { 30720, 28672, 26624, 24576, 22528, 20480, 18432, 16384, 14336, 12288, 10240, 8192, 6144, 4096, 2048, 0, 0 },
            new ushort[] { 30720, 28672, 26624, 24576, 22528, 20480, 18432, 16384, 14336, 12288, 10240, 8192, 6144, 4096, 2048, 0, 0 },
        },
        new ushort[][]
        {
            new ushort[] { 30037, 27307, 24576, 21845, 19115, 16384, 13653, 10923, 8192, 5461, 2731, 0, 0 },
            new ushort[] { 30037, 27307, 24576, 21845, 19115, 16384, 13653, 10923, 8192, 5461, 2731, 0, 0 },
            new ushort[] { 31998, 30347, 27543, 19861, 16949, 13841, 11207, 8679, 6173, 4242, 2239, 0, 0 },
            new ushort[] { 30037, 27307, 24576, 21845, 19115, 16384, 13653, 10923, 8192, 5461, 2731, 0, 0 },
        },
        new ushort[][]
        {
            new ushort[] { 16384, 0, 0 },
            new ushort[] { 28601, 0, 0 },
            new ushort[] { 30770, 0, 0 },
            new ushort[] { 32020, 0, 0 },
        },
    };

    /// <summary>
    /// <c>default_tx_size_cdf[MAX_TX_CATS][TX_SIZE_CONTEXTS][CDF_SIZE(MAX_TX_DEPTH+1)]</c>.
    /// Per-block transform-size depth selection. Indexed by the tx_size category
    /// (block-size dependent) and the 3-context above+left predictor.
    /// </summary>
    public static readonly ushort[][][] DefaultTxSizeCdf =
        new ushort[][][]
    {
        new ushort[][]
        {
            new ushort[] { 12800, 0, 0 },
            new ushort[] { 12800, 0, 0 },
            new ushort[] { 8448, 0, 0 },
        },
        new ushort[][]
        {
            new ushort[] { 20496, 2596, 0, 0 },
            new ushort[] { 20496, 2596, 0, 0 },
            new ushort[] { 14091, 1920, 0, 0 },
        },
        new ushort[][]
        {
            new ushort[] { 19782, 17588, 0, 0 },
            new ushort[] { 19782, 17588, 0, 0 },
            new ushort[] { 8466, 7166, 0, 0 },
        },
        new ushort[][]
        {
            new ushort[] { 26986, 21293, 0, 0 },
            new ushort[] { 26986, 21293, 0, 0 },
            new ushort[] { 15965, 10009, 0, 0 },
        },
    };
}
