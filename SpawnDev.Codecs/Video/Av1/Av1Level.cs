// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 levels (AV1 Bitstream & Decoding Process Specification, Annex A.3): which seq_level_idx / seq_tier a stream
// may signal in its sequence header.
//
// A level is a promise to the decoder: a decoder built for level X.Y must decode every stream that conforms to it,
// and players ask "can this device play level X.Y" before they start. So the level a stream signals has to be one
// the stream really conforms to, and the lowest such level reaches the most decoders (the note under seq_level_idx
// in 6.4.1). A stream that fits no level signals 31, "maximum parameters": no promise at all.
//
// What a level limits, and where the number comes from for this encoder's keyframe streams (one shown frame per
// temporal unit, no scaling, no super-resolution, one operating point):
//
//   picture     UpscaledWidth * FrameHeight <= MaxPicSize, width <= MaxHSize, height <= MaxVSize
//   rates       width * height * frames per second <= MaxDisplayRate and <= MaxDecodeRate (every frame is decoded
//               once and shown once), frame headers per second = frames per second <= MaxHeaderRate
//   tiles       NumTiles <= MaxTiles, TileCols <= MaxTileCols, tiles per second <= MaxTiles * 120
//   every level frame at least 16 x 16, every tile's cropped width and height at least 8, tiles at most 4096 wide
//               and 4096 * 2304 in area, largest tile * frame headers per second <= 588,251,136
//   frame size  CompressedRatio >= Max(0.8, MinCompBasis * SpeedAdj): a limit on the bytes of one Frame OBU
//   bit rate    the decoder model of Annex E feeds the smoothing buffer at MaxBitrate (MainMbps, or HighMbps in the
//               High tier, which exists from level 4.0). Used here as a PEAK limit: no temporal unit is larger than
//               MaxBitrate / frames per second, so every temporal unit arrives within its own frame period. That is
//               sufficient for the model (a frame is complete one period before the next is due), and for keyframe
//               streams, whose frames are close in size, it is close to the average the model needs.
//
// The first three groups are known before a frame is coded (FirstViolation with no sizes, Signals); the last two
// depend on the coded bytes (MaxFrameObuBytes: the largest Frame OBU a level / tier allows). The encoder gives the
// assembly kernel the list of candidates with their byte limits, and the kernel writes the sequence header of the
// first candidate the coded frame fits (Av1KeyframeFraming.BuildFramingBlob, Av1ObuAssembleKernel).
//
// Order of preference: levels ascending, at each level the Main tier before the High tier, then 31.

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>The level a sequence header signals: <c>seq_level_idx</c> and <c>seq_tier</c>.</summary>
/// <param name="SeqLevelIdx">seq_level_idx: level X.Y with X = 2 + (idx &gt;&gt; 2), Y = idx &amp; 3; 31 = maximum parameters.</param>
/// <param name="SeqTier">seq_tier: 0 = Main, 1 = High (level 4.0 and above only).</param>
public readonly record struct Av1LevelSignal(int SeqLevelIdx, int SeqTier)
{
    /// <summary>True for seq_level_idx 31: the stream promises no level.</summary>
    public bool IsMaximumParameters => SeqLevelIdx == Av1Level.MaximumParameters;

    /// <summary>"4.1 Main", "6.0 High", "maximum parameters".</summary>
    public override string ToString() =>
        IsMaximumParameters ? "maximum parameters" : $"{2 + (SeqLevelIdx >> 2)}.{SeqLevelIdx & 3} {(SeqTier == 0 ? "Main" : "High")}";
}

/// <summary>What the level limits are measured against: one coded video sequence of keyframes.</summary>
public readonly struct Av1LevelStream
{
    /// <summary>seq_profile (0 = Main).</summary>
    public int SeqProfile { get; init; }
    /// <summary>Frame width in pixels.</summary>
    public int Width { get; init; }
    /// <summary>Frame height in pixels.</summary>
    public int Height { get; init; }
    /// <summary>The tile grid of every frame.</summary>
    public Av1TileLayout Tiles { get; init; }
    /// <summary>Frames per second, numerator.</summary>
    public int FrameRateNum { get; init; }
    /// <summary>Frames per second, denominator.</summary>
    public int FrameRateDen { get; init; }
    /// <summary>Bytes of the largest Frame OBU (header, size field and payload). 0 = not known: the frame size limit is not checked.</summary>
    public long FrameObuBytes { get; init; }
    /// <summary>Bytes of the largest temporal unit (every OBU of it). 0 = not known: the bit rate limit is not checked.</summary>
    public long TemporalUnitBytes { get; init; }
}

/// <summary>AV1 levels: the limits of Annex A.3 and the level a stream conforms to.</summary>
public static class Av1Level
{
    /// <summary>seq_level_idx of level 2.0.</summary>
    public const int Level2_0 = 0;
    /// <summary>seq_level_idx of level 2.1.</summary>
    public const int Level2_1 = 1;
    /// <summary>seq_level_idx of level 3.0.</summary>
    public const int Level3_0 = 4;
    /// <summary>seq_level_idx of level 3.1.</summary>
    public const int Level3_1 = 5;
    /// <summary>seq_level_idx of level 4.0.</summary>
    public const int Level4_0 = 8;
    /// <summary>seq_level_idx of level 4.1.</summary>
    public const int Level4_1 = 9;
    /// <summary>seq_level_idx of level 5.0.</summary>
    public const int Level5_0 = 12;
    /// <summary>seq_level_idx of level 5.1.</summary>
    public const int Level5_1 = 13;
    /// <summary>seq_level_idx of level 5.2.</summary>
    public const int Level5_2 = 14;
    /// <summary>seq_level_idx of level 5.3.</summary>
    public const int Level5_3 = 15;
    /// <summary>seq_level_idx of level 6.0.</summary>
    public const int Level6_0 = 16;
    /// <summary>seq_level_idx of level 6.1.</summary>
    public const int Level6_1 = 17;
    /// <summary>seq_level_idx of level 6.2.</summary>
    public const int Level6_2 = 18;
    /// <summary>seq_level_idx of level 6.3.</summary>
    public const int Level6_3 = 19;
    /// <summary>seq_level_idx 31, "maximum parameters": no level-based constraint, for streams that conform to no level.</summary>
    public const int MaximumParameters = 31;

    /// <summary>The largest tile area times frame headers per second any level allows (4096 x 2176 at 60 fps, times 1.1).</summary>
    public const long MaxTileSamplesPerSecond = 588_251_136;

    /// <summary>One row of the level tables of Annex A.3.</summary>
    /// <param name="SeqLevelIdx">seq_level_idx of the level.</param>
    /// <param name="MaxPicSize">Luma samples of a frame.</param>
    /// <param name="MaxHSize">Frame width.</param>
    /// <param name="MaxVSize">Frame height.</param>
    /// <param name="MaxDisplayRate">Luma samples shown per second.</param>
    /// <param name="MaxDecodeRate">Luma samples decoded per second.</param>
    /// <param name="MaxHeaderRate">Frame headers per second.</param>
    /// <param name="MainBitrate">MainMbps in bits per second.</param>
    /// <param name="HighBitrate">HighMbps in bits per second; 0 = the level has no High tier.</param>
    /// <param name="MainCR">Compression ratio basis of the Main tier.</param>
    /// <param name="HighCR">Compression ratio basis of the High tier; 0 = the level has no High tier.</param>
    /// <param name="MaxTiles">Tiles of a frame.</param>
    /// <param name="MaxTileCols">Tile columns of a frame.</param>
    public readonly record struct Limits(
        int SeqLevelIdx, long MaxPicSize, int MaxHSize, int MaxVSize, long MaxDisplayRate, long MaxDecodeRate,
        int MaxHeaderRate, long MainBitrate, long HighBitrate, int MainCR, int HighCR, int MaxTiles, int MaxTileCols)
    {
        /// <summary>True from level 4.0: the level has a High tier.</summary>
        public bool HasHighTier => HighBitrate > 0;
    }

    // Annex A.3, both tables, in seq_level_idx order. Levels 2.2, 2.3, 3.2, 3.3, 4.2, 4.3 and 7.x are not defined.
    private static readonly Limits[] s_table =
    {
        new(Level2_0,   147456,  2048, 1152,     4_423_680,     5_529_600, 150,   1_500_000,           0, 2, 0,   8,  4),
        new(Level2_1,   278784,  2816, 1584,     8_363_520,    10_454_400, 150,   3_000_000,           0, 2, 0,   8,  4),
        new(Level3_0,   665856,  4352, 2448,    19_975_680,    24_969_600, 150,   6_000_000,           0, 2, 0,  16,  6),
        new(Level3_1,  1065024,  5504, 3096,    31_950_720,    39_938_400, 150,  10_000_000,           0, 2, 0,  16,  6),
        new(Level4_0,  2359296,  6144, 3456,    70_778_880,    77_856_768, 300,  12_000_000,  30_000_000, 4, 4,  32,  8),
        new(Level4_1,  2359296,  6144, 3456,   141_557_760,   155_713_536, 300,  20_000_000,  50_000_000, 4, 4,  32,  8),
        new(Level5_0,  8912896,  8192, 4352,   267_386_880,   273_715_200, 300,  30_000_000, 100_000_000, 6, 4,  64,  8),
        new(Level5_1,  8912896,  8192, 4352,   534_773_760,   547_430_400, 300,  40_000_000, 160_000_000, 8, 4,  64,  8),
        new(Level5_2,  8912896,  8192, 4352, 1_069_547_520, 1_094_860_800, 300,  60_000_000, 240_000_000, 8, 4,  64,  8),
        new(Level5_3,  8912896,  8192, 4352, 1_069_547_520, 1_176_502_272, 300,  60_000_000, 240_000_000, 8, 4,  64,  8),
        new(Level6_0, 35651584, 16384, 8704, 1_069_547_520, 1_176_502_272, 300,  60_000_000, 240_000_000, 8, 4, 128, 16),
        new(Level6_1, 35651584, 16384, 8704, 2_139_095_040, 2_189_721_600, 300, 100_000_000, 480_000_000, 8, 4, 128, 16),
        new(Level6_2, 35651584, 16384, 8704, 4_278_190_080, 4_379_443_200, 300, 160_000_000, 800_000_000, 8, 4, 128, 16),
        new(Level6_3, 35651584, 16384, 8704, 4_278_190_080, 4_706_009_088, 300, 160_000_000, 800_000_000, 8, 4, 128, 16),
    };

    /// <summary>The defined levels, lowest first.</summary>
    public static ReadOnlySpan<Limits> Table => s_table;

    /// <summary>The limits of the level <paramref name="seqLevelIdx"/>; false when it is not a defined level (31 included).</summary>
    public static bool TryGetLimits(int seqLevelIdx, out Limits limits)
    {
        foreach (var l in s_table)
        {
            if (l.SeqLevelIdx != seqLevelIdx) continue;
            limits = l;
            return true;
        }
        limits = default;
        return false;
    }

    private static void RequireStream(in Av1LevelStream s)
    {
        if (s.Width <= 0 || s.Height <= 0) throw new ArgumentOutOfRangeException(nameof(s), "Width and Height must be positive.");
        if (s.FrameRateNum <= 0 || s.FrameRateDen <= 0) throw new ArgumentOutOfRangeException(nameof(s), "The frame rate must be positive.");
        if (s.SeqProfile is < 0 or > 2) throw new ArgumentOutOfRangeException(nameof(s), "SeqProfile must be 0, 1 or 2.");
        if (s.Tiles.TileCols < 1 || s.Tiles.TileRows < 1) throw new ArgumentException("Tiles is not a tile grid (use Av1TileLayout.Compute).", nameof(s));
    }

    /// <summary>TileWidth of the widest tile and TileHeight of the highest, in luma samples (the mi grid: the frame rounded up to 8).</summary>
    private static (long Width, long Height) LargestTile(in Av1LevelStream s)
    {
        long miCols = ((s.Width + 7) >> 3) << 1, miRows = ((s.Height + 7) >> 3) << 1;
        return (Math.Min(s.Tiles.TileWidthSb * 16L, miCols) * 4, Math.Min(s.Tiles.TileHeightSb * 16L, miRows) * 4);
    }

    /// <summary>
    /// The first constraint of Annex A.3 every defined level puts on a stream that <paramref name="s"/> breaks, or
    /// null: such a stream conforms to no level and can only signal <see cref="MaximumParameters"/>.
    /// </summary>
    public static string? FirstViolationOfEveryLevel(in Av1LevelStream s)
    {
        RequireStream(s);
        if (s.Width < 16) return $"FrameWidth {s.Width} is less than 16";
        if (s.Height < 16) return $"FrameHeight {s.Height} is less than 16";
        var (tileW, tileH) = LargestTile(s);
        if (tileW > 4096) return $"TileWidth {tileW} is more than 4096";
        if (tileW * tileH > 4096L * 2304) return $"TileWidth * TileHeight {tileW * tileH} is more than 4096 * 2304";
        long croppedW = s.Width - (s.Tiles.TileCols - 1) * s.Tiles.TileWidthSb * 64L;
        long croppedH = s.Height - (s.Tiles.TileRows - 1) * s.Tiles.TileHeightSb * 64L;
        if (croppedW < 8) return $"CroppedTileWidth {croppedW} of the last tile column is less than 8";
        if (croppedH < 8) return $"CroppedTileHeight {croppedH} of the last tile row is less than 8";
        if (tileW * tileH * s.FrameRateNum > MaxTileSamplesPerSecond * s.FrameRateDen)
            return $"MaxTileSizeInLumaSamples {tileW * tileH} * frame headers per second is more than {MaxTileSamplesPerSecond}";
        return null;
    }

    /// <summary>The picture size and rate limits of <paramref name="l"/> (not its tile limits): the first one broken, or null.</summary>
    private static string? FirstPictureViolation(in Limits l, in Av1LevelStream s)
    {
        long pic = (long)s.Width * s.Height;
        if (pic > l.MaxPicSize) return $"UpscaledWidth * FrameHeight {pic} is more than MaxPicSize {l.MaxPicSize}";
        if (s.Width > l.MaxHSize) return $"UpscaledWidth {s.Width} is more than MaxHSize {l.MaxHSize}";
        if (s.Height > l.MaxVSize) return $"FrameHeight {s.Height} is more than MaxVSize {l.MaxVSize}";
        if ((Int128)pic * s.FrameRateNum > (Int128)l.MaxDisplayRate * s.FrameRateDen)
            return $"TotalDisplayLumaSampleRate {pic} * {s.FrameRateNum}/{s.FrameRateDen} is more than MaxDisplayRate {l.MaxDisplayRate}";
        if ((Int128)pic * s.FrameRateNum > (Int128)l.MaxDecodeRate * s.FrameRateDen)
            return $"TotalDecodedLumaSampleRate {pic} * {s.FrameRateNum}/{s.FrameRateDen} is more than MaxDecodeRate {l.MaxDecodeRate}";
        if (s.FrameRateNum > (long)l.MaxHeaderRate * s.FrameRateDen)
            return $"NumFrameHeadersSec {s.FrameRateNum}/{s.FrameRateDen} is more than MaxHeaderRate {l.MaxHeaderRate}";
        return null;
    }

    private static int PicSizeProfileFactor(int seqProfile) => seqProfile == 0 ? 15 : seqProfile == 1 ? 30 : 36;

    private static int BitrateProfileFactor(int seqProfile) => seqProfile + 1;

    /// <summary>
    /// The first constraint of Annex A.3 that the stream <paramref name="s"/> breaks when it signals
    /// <paramref name="signal"/>, or null when it conforms. The frame size and bit rate limits are checked only when
    /// <paramref name="s"/> carries the sizes. <see cref="MaximumParameters"/> has no constraint.
    /// </summary>
    public static string? FirstViolation(Av1LevelSignal signal, in Av1LevelStream s)
    {
        RequireStream(s);
        if (signal.SeqTier is not (0 or 1)) return $"seq_tier {signal.SeqTier} is not 0 or 1";
        if (signal.IsMaximumParameters) return null;
        if (!TryGetLimits(signal.SeqLevelIdx, out var l)) return $"seq_level_idx {signal.SeqLevelIdx} is not a defined level";
        if (signal.SeqTier == 1 && !l.HasHighTier) return $"seq_tier 1 is signalled for a level below 4.0 (seq_level_idx {signal.SeqLevelIdx})";

        string? v = FirstPictureViolation(l, s);
        if (v != null) return v;
        long tiles = s.Tiles.TileCount;
        if (tiles * s.FrameRateNum > l.MaxTiles * 120L * s.FrameRateDen)
            return $"tiles per second {tiles} * {s.FrameRateNum}/{s.FrameRateDen} is more than MaxTiles * 120 = {l.MaxTiles * 120}";
        if (tiles > l.MaxTiles) return $"NumTiles {tiles} is more than MaxTiles {l.MaxTiles}";
        if (s.Tiles.TileCols > l.MaxTileCols) return $"TileCols {s.Tiles.TileCols} is more than MaxTileCols {l.MaxTileCols}";
        v = FirstViolationOfEveryLevel(s);
        if (v != null) return v;

        long pic = (long)s.Width * s.Height;
        if (s.FrameObuBytes > 0)
        {
            // CompressedRatio = UnCompressedSize / CompressedSize >= Max(0.8, MinCompBasis * SpeedAdj),
            // SpeedAdj = TotalDecodedLumaSampleRate / MaxDisplayRate. In integers, both bounds separately.
            long uncompressed = (pic * PicSizeProfileFactor(s.SeqProfile)) >> 3;
            long compressed = s.FrameObuBytes - 128;
            int basis = signal.SeqTier == 0 ? l.MainCR : l.HighCR;
            if (compressed > 0)
            {
                if ((Int128)compressed * 4 > (Int128)uncompressed * 5)
                    return $"CompressedRatio {uncompressed}/{compressed} is less than 0.8";
                if ((Int128)compressed * basis * pic * s.FrameRateNum > (Int128)uncompressed * l.MaxDisplayRate * s.FrameRateDen)
                    return $"CompressedRatio {uncompressed}/{compressed} is less than MinCompBasis {basis} * SpeedAdj";
            }
        }
        if (s.TemporalUnitBytes > 0)
        {
            long maxBitrate = (signal.SeqTier == 0 ? l.MainBitrate : l.HighBitrate) * BitrateProfileFactor(s.SeqProfile);
            if ((Int128)s.TemporalUnitBytes * 8 * s.FrameRateNum > (Int128)maxBitrate * s.FrameRateDen)
                return $"a temporal unit of {s.TemporalUnitBytes} bytes at {s.FrameRateNum}/{s.FrameRateDen} per second is more than MaxBitrate {maxBitrate}";
        }
        return null;
    }

    /// <summary>
    /// The largest Frame OBU (header byte, size field and payload, in bytes) a stream like <paramref name="s"/> may
    /// carry when it signals <paramref name="signal"/> and its temporal units hold <paramref name="otherObuBytes"/>
    /// more bytes (temporal delimiter, sequence header): the frame size limit and the bit rate limit of the level and
    /// tier, whichever is lower. <see cref="long.MaxValue"/> for <see cref="MaximumParameters"/>. The sizes in
    /// <paramref name="s"/> are not read; a signal the stream cannot carry at any size throws.
    /// </summary>
    public static long MaxFrameObuBytes(Av1LevelSignal signal, in Av1LevelStream s, int otherObuBytes)
    {
        if (otherObuBytes < 0) throw new ArgumentOutOfRangeException(nameof(otherObuBytes));
        var unsized = WithoutSizes(s);
        string? v = FirstViolation(signal, unsized);
        if (v != null) throw new ArgumentException($"The stream does not conform to {signal} at any frame size: {v}.", nameof(signal));
        if (signal.IsMaximumParameters) return long.MaxValue;
        TryGetLimits(signal.SeqLevelIdx, out var l);

        long pic = (long)s.Width * s.Height;
        long uncompressed = (pic * PicSizeProfileFactor(s.SeqProfile)) >> 3;
        int basis = signal.SeqTier == 0 ? l.MainCR : l.HighCR;
        // compressed * 4 <= uncompressed * 5  and  compressed * basis * pic * fps <= uncompressed * MaxDisplayRate
        Int128 byRatioFloor = (Int128)uncompressed * 5 / 4;
        Int128 bySpeed = (Int128)uncompressed * l.MaxDisplayRate * s.FrameRateDen / ((Int128)basis * pic * s.FrameRateNum);
        Int128 frame = Int128.Min(byRatioFloor, bySpeed) + 128;
        // temporal unit * 8 * fps <= MaxBitrate
        long maxBitrate = (signal.SeqTier == 0 ? l.MainBitrate : l.HighBitrate) * BitrateProfileFactor(s.SeqProfile);
        Int128 unit = (Int128)maxBitrate * s.FrameRateDen / ((Int128)8 * s.FrameRateNum);
        frame = Int128.Min(frame, unit - otherObuBytes);
        return frame < 0 ? 0 : (long)frame;
    }

    private static Av1LevelStream WithoutSizes(in Av1LevelStream s) => new()
    {
        SeqProfile = s.SeqProfile, Width = s.Width, Height = s.Height, Tiles = s.Tiles,
        FrameRateNum = s.FrameRateNum, FrameRateDen = s.FrameRateDen,
    };

    /// <summary>
    /// Every level and tier the stream <paramref name="s"/> can signal as far as its picture size, frame rate and
    /// tile grid go (the sizes in <paramref name="s"/> are not read), in order of preference: levels ascending, Main
    /// tier before High tier, and last <see cref="MaximumParameters"/>, which any stream can signal.
    /// </summary>
    public static List<Av1LevelSignal> Signals(in Av1LevelStream s)
    {
        var unsized = WithoutSizes(s);
        var list = new List<Av1LevelSignal>();
        foreach (var l in s_table)
        {
            if (FirstViolation(new Av1LevelSignal(l.SeqLevelIdx, 0), unsized) != null) continue;
            list.Add(new Av1LevelSignal(l.SeqLevelIdx, 0));
            if (l.HasHighTier) list.Add(new Av1LevelSignal(l.SeqLevelIdx, 1));
        }
        list.Add(new Av1LevelSignal(MaximumParameters, 0));
        return list;
    }

    /// <summary>
    /// The level and tier the stream <paramref name="s"/> signals: the first of <see cref="Signals"/> it conforms
    /// to, with the frame size and bit rate limits applied when <paramref name="s"/> carries the sizes.
    /// </summary>
    public static Av1LevelSignal Select(in Av1LevelStream s)
    {
        foreach (var signal in Signals(s))
            if (FirstViolation(signal, s) == null) return signal;
        return new Av1LevelSignal(MaximumParameters, 0);   // not reached: 31 has no constraint
    }

    /// <summary>
    /// The tile grid of a <paramref name="width"/> x <paramref name="height"/> picture shown at
    /// <paramref name="frameRateNum"/> / <paramref name="frameRateDen"/> frames per second that stays inside a level:
    /// <see cref="Av1TileLayout.Compute"/> with the tile limits (MaxTiles, MaxTileCols, tiles per second, cropped
    /// tiles of at least 8 pixels) of the level <paramref name="levelLimit"/>, or of the lowest level the picture
    /// size and frame rate fit when that is higher. So the tiles never make the stream need a higher level than
    /// that. <see cref="MaximumParameters"/> = no level limit (and a picture that fits no level gets none):
    /// <see cref="Av1TileLayout.Compute"/> as is.
    /// </summary>
    public static Av1TileLayout TileLayout(
        int width, int height, int tileSizeSuperblocks, int maxTiles, int frameRateNum, int frameRateDen, int levelLimit)
    {
        if (frameRateNum <= 0) throw new ArgumentOutOfRangeException(nameof(frameRateNum));
        if (frameRateDen <= 0) throw new ArgumentOutOfRangeException(nameof(frameRateDen));
        if (levelLimit == MaximumParameters) return Av1TileLayout.Compute(width, height, tileSizeSuperblocks, maxTiles);
        if (!TryGetLimits(levelLimit, out _))
            throw new ArgumentOutOfRangeException(nameof(levelLimit), $"seq_level_idx {levelLimit} is not a defined AV1 level (or 31).");
        var s = new Av1LevelStream
        {
            Width = width, Height = height, Tiles = Av1TileLayout.SingleTile(width, height),
            FrameRateNum = frameRateNum, FrameRateDen = frameRateDen,
        };
        foreach (var l in s_table)
        {
            if (l.SeqLevelIdx < levelLimit || FirstPictureViolation(l, s) != null) continue;
            long perSecond = l.MaxTiles * 120L * frameRateDen / frameRateNum;
            int most = (int)Math.Max(1, Math.Min(Math.Min(maxTiles, l.MaxTiles), perSecond));
            return Av1TileLayout.Compute(width, height, tileSizeSuperblocks, most, l.MaxTileCols, minEdgeTilePixels: 8);
        }
        return Av1TileLayout.Compute(width, height, tileSizeSuperblocks, maxTiles);
    }
}
