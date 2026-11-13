// AV1 level signalling: the sequence header's seq_level_idx / seq_tier (AV1 spec Annex A.3).
//
// A level is a promise to the decoder, so the encoder signals the lowest level and tier the stream really conforms
// to, and 31 ("maximum parameters") when it conforms to none. Picture size, frame rate and tile grid are known up
// front; the frame size and bit rate limits depend on the coded bytes, so the level is chosen after the tiles are
// coded, on the accelerator (Av1ObuAssembleKernel, from the candidates Av1KeyframeFraming puts in the framing blob).
//
// Oracles, and what each can fail on:
//  - Av1Level_Select_MatchesTheSpecificationTables (no accelerator): levels derived by hand from the two tables of
//    Annex A.3, starting with the table's own "Example" column (426x240@30 = 2.0 ... 7680x4320@120 = 6.2). Fails on a
//    wrong table value or a wrong constraint.
//  - Av1Level_FrameSizeLimits (no accelerator): the largest Frame OBU of a level / tier, by hand for the bit rate
//    and compression ratio rules, and MaxFrameObuBytes against FirstViolation at every boundary (limit conforms,
//    limit + 1 does not): two formulations of the same rule, one solved for the bytes, one checking them.
//  - Av1Level_TileLayout_StaysInsideTheLevel (no accelerator): the grid under a level limit, by hand, and over a
//    sweep: the tiles never make the stream need more than the limit (or the level the picture itself needs).
//  - Av1ObuAssembleKernel_Level_PicksTheCandidateAtTheExactLimit (every backend): the kernel's choice at a Frame OBU
//    of exactly a candidate's limit and of one byte more, with synthetic tile bytes. Fails on < instead of <=, on a
//    table read at the wrong offset, on a prefix copied from the wrong candidate.
//  - Av1KeyframeEncoderGpu_Level_SignalsTheLowestLevelTheStreamConformsTo (every backend): real encodes. The unit
//    equals the CPU reference's, the level it signals has no violation for the unit's measured sizes
//    (Av1Level.FirstViolation: the specification's inequalities, not the limits the kernel compared against), and
//    every level before it in the order of preference has one.
//  - Av1KeyframeEncoderGpu_Level_FfmpegReadsTheSignalledLevel (desktop): ffprobe, a black box, reads the same
//    seq_level_idx, and libdav1d decodes the stream to the encoder's reconstruction: the tier bit moves every later
//    bit of the sequence header, and a level limit gives tile shapes no other test uses.
//  - Av1Isobmff_UnifySequenceHeaders and VideoFileEncoderGpu_Av1_OneLevelForTheFile: a file has one sequence header;
//    it is the one of the largest frame, and the file as a whole conforms to it and to nothing lower.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Container.Mp4;
using SpawnDev.Codecs.Video;
using SpawnDev.Codecs.Video.Av1;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private static Av1LevelStream Av1LevelStreamOf(int w, int h, Av1TileLayout tiles, int num, int den, long frameObu = 0, long unit = 0) => new()
    {
        Width = w, Height = h, Tiles = tiles, FrameRateNum = num, FrameRateDen = den, FrameObuBytes = frameObu, TemporalUnitBytes = unit,
    };

    /// <summary>The OBUs of a temporal unit or sample: (obu_type, start, whole length, payload offset, payload length).</summary>
    private static List<(int Type, int Start, int Length, int PayloadOffset, int PayloadLength)> Av1TestObus(byte[] data)
    {
        var list = new List<(int, int, int, int, int)>();
        int pos = 0;
        while (pos < data.Length)
        {
            int start = pos;
            byte header = data[pos++];
            if ((header & 4) != 0) pos++;
            if ((header & 2) == 0) throw new Exception("test OBU walker: an OBU without a size field");
            long size = 0;
            for (int i = 0; ; i++)
            {
                byte b = data[pos++];
                size |= (long)(b & 0x7F) << (7 * i);
                if ((b & 0x80) == 0) break;
            }
            list.Add(((header >> 3) & 15, start, pos + (int)size - start, pos, (int)size));
            pos += (int)size;
        }
        if (pos != data.Length) throw new Exception("test OBU walker: the last OBU runs past the end");
        return list;
    }

    /// <summary>What a temporal unit (or sample) signals and weighs: the level, its sequence header OBU, the Frame OBU's bytes.</summary>
    private static (Av1LevelSignal Signal, byte[] SequenceHeaderObu, long FrameObuBytes) Av1LevelRead(byte[] unit)
    {
        Av1LevelSignal? signal = null;
        byte[]? shObu = null;
        long frame = 0;
        foreach (var o in Av1TestObus(unit))
        {
            if (o.Type == 1)
            {
                var sh = Av1SequenceHeaderParser.Parse(unit.AsSpan(o.PayloadOffset, o.PayloadLength));
                signal = new Av1LevelSignal(sh.SeqLevelIdx0, sh.SeqTier0);
                shObu = unit.AsSpan(o.Start, o.Length).ToArray();
            }
            else if (o.Type == 6) frame += o.Length;
        }
        if (signal == null || frame == 0) throw new Exception("test: the unit has no sequence header or no Frame OBU");
        return (signal.Value, shObu!, frame);
    }

    /// <summary>
    /// Null when <paramref name="signal"/> is the lowest level the stream conforms to: no violation for it, and one
    /// for every level before it in the order of preference. Otherwise what is wrong.
    /// </summary>
    private static string? Av1LevelIsTheLowestConforming(Av1LevelSignal signal, in Av1LevelStream stream)
    {
        string? v = Av1Level.FirstViolation(signal, stream);
        if (v != null) return $"signals {signal} but does not conform to it: {v}";
        var order = Av1Level.Signals(stream);
        int at = order.IndexOf(signal);
        if (at < 0) return $"signals {signal}, which is not among the levels its picture, rate and tiles allow ({string.Join(", ", order)})";
        for (int k = 0; k < at; k++)
            if (Av1Level.FirstViolation(order[k], stream) == null) return $"signals {signal} but also conforms to the lower {order[k]}";
        return null;
    }

    /// <summary>
    /// Levels derived by hand from the tables of Annex A.3 for a stream with no size information: (width, height,
    /// tile size, frame rate) -> seq_level_idx (Main tier). The first twelve are the specification's own examples.
    /// </summary>
    [TestMethod(Category = "CpuOnly")]
    public Task Av1Level_Select_MatchesTheSpecificationTables()
    {
        CpuOnlyTest();
        const int max = int.MaxValue;
        var cases = new (int w, int h, int tileSize, int num, int den, int expected, string why)[]
        {
            // the "Example" column of the second table (fewest tiles, except where one tile is too large for the rate)
            (426, 240, max, 30, 1, Av1Level.Level2_0, "example 2.0"),
            (640, 360, max, 30, 1, Av1Level.Level2_1, "example 2.1"),
            (854, 480, max, 30, 1, Av1Level.Level3_0, "example 3.0"),
            (1280, 720, max, 30, 1, Av1Level.Level3_1, "example 3.1"),
            (1920, 1080, max, 30, 1, Av1Level.Level4_0, "example 4.0"),
            (1920, 1080, max, 60, 1, Av1Level.Level4_1, "example 4.1"),
            (3840, 2160, max, 30, 1, Av1Level.Level5_0, "example 5.0"),
            (3840, 2160, max, 60, 1, Av1Level.Level5_1, "example 5.1"),
            (3840, 2160, 30, 120, 1, Av1Level.Level5_2, "example 5.2: 2 x 2 tiles of 1920x1088"),
            (7680, 4320, max, 30, 1, Av1Level.Level6_0, "example 6.0: 2 x 2 tiles of 3840x2176"),
            (7680, 4320, max, 60, 1, Av1Level.Level6_1, "example 6.1"),
            (7680, 4320, 30, 120, 1, Av1Level.Level6_2, "example 6.2: 4 x 4 tiles of 1920x1088"),
            // one 3840x2160 tile at 120 fps: 8,294,400 * 120 > 588,251,136 luma samples per second in one tile
            (3840, 2160, max, 120, 1, Av1Level.MaximumParameters, "largest tile times frame rate"),
            // rates
            (1920, 1080, max, 30000, 1001, Av1Level.Level4_0, "29.97 fps"),
            (1920, 1080, max, 35, 1, Av1Level.Level4_1, "2,073,600 * 35 > MaxDisplayRate 70,778,880 of 4.0"),
            (426, 240, max, 150, 1, Av1Level.Level3_0, "102,240 * 150 > MaxDisplayRate of 2.1; 150 headers per second allowed"),
            (426, 240, max, 151, 1, Av1Level.Level4_0, "151 frame headers per second > MaxHeaderRate 150 of 2.x and 3.x"),
            (64, 64, max, 300, 1, Av1Level.Level4_0, "300 frame headers per second"),
            (64, 64, max, 301, 1, Av1Level.MaximumParameters, "301 frame headers per second > 300 of every level"),
            // tiles
            (832, 480, 1, 30, 1, Av1Level.Level6_0, "13 x 8 = 104 tiles in 13 columns: MaxTiles 128, MaxTileCols 16"),
            (832, 480, 2, 30, 1, Av1Level.Level4_0, "7 x 4 = 28 tiles in 7 columns: MaxTiles 32, MaxTileCols 8"),
            (832, 480, 2, 150, 1, Av1Level.Level5_0, "28 tiles * 150 per second > 32 * 120 of 4.x"),
            (832, 64, 1, 30, 1, Av1Level.Level6_0, "13 tiles in 13 columns: MaxTileCols 16 only from 6.0"),
            (320, 200, 3, 30, 1, Av1Level.Level2_0, "2 x 2 tiles"),
            (1920, 1080, 1, 30, 1, Av1Level.MaximumParameters, "15 x 17 = 255 tiles > MaxTiles 128 of 6.x"),
            (1088, 1024, 1, 30, 1, Av1Level.MaximumParameters, "9 x 16 = 144 tiles"),
            // limits of every level
            (130, 66, 1, 30, 1, Av1Level.MaximumParameters, "the last tile column and row keep 2 pixels"),
            (130, 64, 1, 30, 1, Av1Level.MaximumParameters, "the last tile column keeps 2 pixels (CroppedTileWidth < 8), one tile row"),
            (128, 66, 1, 30, 1, Av1Level.MaximumParameters, "the last tile row keeps 2 pixels (CroppedTileHeight < 8), whole tile columns"),
            (136, 72, 1, 30, 1, Av1Level.Level2_0, "8 pixels in the last tile column and row"),
            (130, 66, max, 30, 1, Av1Level.Level2_0, "one tile"),
            (200, 9, max, 30, 1, Av1Level.MaximumParameters, "FrameHeight 9 < 16"),
            (15, 64, max, 30, 1, Av1Level.MaximumParameters, "FrameWidth 15 < 16"),
            (16, 16, max, 30, 1, Av1Level.Level2_0, "the smallest frame a level allows"),
        };
        var failures = new List<string>();
        foreach (var (w, h, tileSize, num, den, expected, why) in cases)
        {
            var stream = Av1LevelStreamOf(w, h, Av1TileLayout.Compute(w, h, tileSize), num, den);
            var got = Av1Level.Select(stream);
            if (got != new Av1LevelSignal(expected, 0))
                failures.Add($"{w}x{h} tile size {(tileSize == max ? "max" : tileSize.ToString())} at {num}/{den}: {got} (seq_level_idx {got.SeqLevelIdx}), expected seq_level_idx {expected} ({why})");
        }

        // The order of preference: levels ascending, Main before High (High from 4.0), last maximum parameters.
        var order = Av1Level.Signals(Av1LevelStreamOf(1920, 1080, Av1TileLayout.SingleTile(1920, 1080), 30, 1));
        string orderText = string.Join(" ", order.Select(s => $"{s.SeqLevelIdx}/{s.SeqTier}"));
        const string expectedOrder = "8/0 8/1 9/0 9/1 12/0 12/1 13/0 13/1 14/0 14/1 15/0 15/1 16/0 16/1 17/0 17/1 18/0 18/1 19/0 19/1 31/0";
        if (orderText != expectedOrder) failures.Add($"1920x1080 at 30: order {orderText}, expected {expectedOrder}");
        order = Av1Level.Signals(Av1LevelStreamOf(426, 240, Av1TileLayout.SingleTile(426, 240), 30, 1));
        if (!order.Take(5).SequenceEqual(new[] { new Av1LevelSignal(0, 0), new Av1LevelSignal(1, 0), new Av1LevelSignal(4, 0), new Av1LevelSignal(5, 0), new Av1LevelSignal(8, 0) }))
            failures.Add("426x240 at 30: the levels below 4.0 have no High tier: " + string.Join(" ", order.Take(5)));

        Equal(14, Av1Level.Table.Length, "defined levels");
        True(!Av1Level.TryGetLimits(2, out _) && !Av1Level.TryGetLimits(20, out _) && !Av1Level.TryGetLimits(31, out _), "2.2, 7.0 and 31 are not defined levels");
        Equal("4.1 High", new Av1LevelSignal(9, 1).ToString());
        Equal("2.0 Main", new Av1LevelSignal(0, 0).ToString());
        Equal("maximum parameters", new Av1LevelSignal(31, 0).ToString());
        // a tier the level does not have, an undefined level
        var one = Av1LevelStreamOf(64, 64, Av1TileLayout.SingleTile(64, 64), 30, 1);
        True(Av1Level.FirstViolation(new Av1LevelSignal(Av1Level.Level3_1, 1), one) != null, "3.1 has no High tier");
        True(Av1Level.FirstViolation(new Av1LevelSignal(6, 0), one) != null, "seq_level_idx 6 (3.2) is not defined");
        True(Av1Level.FirstViolation(new Av1LevelSignal(31, 0), Av1LevelStreamOf(15, 9, Av1TileLayout.SingleTile(15, 9), 500, 1, 1L << 40, 1L << 40)) == null,
            "maximum parameters has no constraint");
        True(failures.Count == 0, string.Join("\n", failures));
        return Task.CompletedTask;
    }

    /// <summary>
    /// The limits that depend on the coded bytes. By hand, for 1920x1080 at 30 fps (2,073,600 samples, UnCompressedSize
    /// 3,888,000): bit rate, level 4.0 Main 12 Mbit/s = 50,000 bytes per frame period, High 30 Mbit/s = 125,000, 4.1
    /// Main 20 Mbit/s = 83,333; compression ratio at 4.0 Main, MinCompBasis 4 * SpeedAdj 62,208,000 / 70,778,880 =
    /// 3.515625, so CompressedSize up to 3,888,000 / 3.515625 = 1,105,920 and a Frame OBU up to 1,106,048; the floor
    /// of 0.8 gives CompressedSize up to 4,860,000. And for every level of several streams: the limit conforms, one
    /// byte more does not.
    /// </summary>
    [TestMethod(Category = "CpuOnly")]
    public Task Av1Level_FrameSizeLimits_MatchTheSpecificationRules()
    {
        CpuOnlyTest();
        var failures = new List<string>();
        var tile = Av1TileLayout.SingleTile(1920, 1080);
        var hd30 = Av1LevelStreamOf(1920, 1080, tile, 30, 1);
        void Limit(Av1LevelSignal signal, in Av1LevelStream s, int other, long expected, string why)
        {
            long got = Av1Level.MaxFrameObuBytes(signal, s, other);
            if (got != expected) failures.Add($"MaxFrameObuBytes({signal}, {s.Width}x{s.Height} at {s.FrameRateNum}/{s.FrameRateDen}, {other}) = {got}, expected {expected} ({why})");
        }
        Limit(new(Av1Level.Level4_0, 0), hd30, 16, 49_984, "12 Mbit/s / 30 = 50,000 bytes, less 16");
        Limit(new(Av1Level.Level4_0, 1), hd30, 16, 124_984, "30 Mbit/s / 30 = 125,000 bytes, less 16");
        Limit(new(Av1Level.Level4_1, 0), hd30, 16, 83_317, "20 Mbit/s / 30 = 83,333 bytes, less 16");
        Limit(new(Av1Level.Level4_0, 0), hd30, 0, 50_000, "no other OBU");
        Limit(new(Av1Level.MaximumParameters, 0), hd30, 16, long.MaxValue, "maximum parameters");
        // 20 fps, 6.3 High: 800 Mbit/s / 20 = 5,000,000 bytes is above the ratio floor: 3,888,000 / 0.8 + 128
        Limit(new(Av1Level.Level6_3, 1), Av1LevelStreamOf(1920, 1080, tile, 20, 1), 16, 4_860_128, "CompressedRatio 0.8");
        // 30000/1001 fps: 12,000,000 * 1001 / (8 * 30000) = 50,050 bytes
        Limit(new(Av1Level.Level4_0, 0), Av1LevelStreamOf(1920, 1080, tile, 30000, 1001), 0, 50_050, "29.97 fps");

        // The compression ratio rules, without the bit rate (no temporal unit size given).
        void Ratio(Av1LevelSignal signal, long frameObu, bool conforms, string why)
        {
            string? v = Av1Level.FirstViolation(signal, Av1LevelStreamOf(1920, 1080, tile, 30, 1, frameObu));
            if ((v == null) != conforms) failures.Add($"1920x1080 at 30, {signal}, Frame OBU {frameObu}: {(v ?? "conforms")}, expected {(conforms ? "to conform" : "a violation")} ({why})");
        }
        Ratio(new(Av1Level.Level4_0, 0), 1_106_048, true, "CompressedSize 1,105,920 = UnCompressedSize / (4 * SpeedAdj)");
        Ratio(new(Av1Level.Level4_0, 0), 1_106_049, false, "one byte over MinCompBasis * SpeedAdj");
        Ratio(new(Av1Level.Level6_3, 1), 4_860_128, true, "CompressedSize 4,860,000 = UnCompressedSize / 0.8");
        Ratio(new(Av1Level.Level6_3, 1), 4_860_129, false, "one byte over the floor of 0.8");
        Ratio(new(Av1Level.Level4_0, 0), 100, true, "a Frame OBU under 128 bytes has no compressed size");

        // Select with sizes: the first level and tier the bytes fit.
        void Selects(long frameObu, Av1LevelSignal expected)
        {
            var got = Av1Level.Select(Av1LevelStreamOf(1920, 1080, tile, 30, 1, frameObu, frameObu + 16));
            if (got != expected) failures.Add($"1920x1080 at 30, Frame OBU {frameObu}: Select {got}, expected {expected}");
        }
        Selects(49_984, new(Av1Level.Level4_0, 0));
        Selects(49_985, new(Av1Level.Level4_0, 1));
        Selects(124_984, new(Av1Level.Level4_0, 1));
        Selects(124_985, new(Av1Level.Level4_1, 1));   // 4.1 Main allows 83,317: skipped
        Selects(3_333_317, new(Av1Level.Level6_2, 1)); // 800 Mbit/s / 30 = 3,333,333, less 16
        Selects(3_333_318, new(Av1Level.MaximumParameters, 0));

        // Every boundary: a Frame OBU of exactly the limit conforms, one byte more does not.
        const int other = 17;
        var streams = new[]
        {
            hd30,
            Av1LevelStreamOf(1920, 1080, tile, 1, 1),
            Av1LevelStreamOf(1920, 1080, tile, 60000, 1001),
            Av1LevelStreamOf(832, 480, Av1TileLayout.Compute(832, 480, 1), 30, 1),
            Av1LevelStreamOf(320, 200, Av1TileLayout.Compute(320, 200, 3), 25, 1),
            Av1LevelStreamOf(3840, 2160, Av1TileLayout.SingleTile(3840, 2160), 60, 1),
            Av1LevelStreamOf(64, 64, Av1TileLayout.SingleTile(64, 64), 120, 1),
            Av1LevelStreamOf(16, 16, Av1TileLayout.SingleTile(16, 16), 300, 1),
        };
        int boundaries = 0;
        foreach (var s in streams)
            foreach (var signal in Av1Level.Signals(s))
            {
                if (signal.IsMaximumParameters) continue;
                long limit = Av1Level.MaxFrameObuBytes(signal, s, other);
                string tag = $"{s.Width}x{s.Height} at {s.FrameRateNum}/{s.FrameRateDen}, {signal}, limit {limit}";
                if (limit <= 0) { failures.Add($"{tag}: no frame fits"); continue; }
                string? at = Av1Level.FirstViolation(signal, Av1LevelStreamOf(s.Width, s.Height, s.Tiles, s.FrameRateNum, s.FrameRateDen, limit, limit + other));
                string? over = Av1Level.FirstViolation(signal, Av1LevelStreamOf(s.Width, s.Height, s.Tiles, s.FrameRateNum, s.FrameRateDen, limit + 1, limit + 1 + other));
                if (at != null) failures.Add($"{tag}: a Frame OBU of the limit does not conform: {at}");
                if (over == null) failures.Add($"{tag}: a Frame OBU one byte over the limit conforms");
                boundaries++;
            }
        True(boundaries >= 80, $"only {boundaries} boundaries were checked");

        // A signal the stream cannot carry at any size has no limit to ask for.
        bool threw = false;
        try { Av1Level.MaxFrameObuBytes(new Av1LevelSignal(Av1Level.Level3_0, 0), hd30, 16); } catch (ArgumentException) { threw = true; }
        True(threw, "MaxFrameObuBytes for a level the picture does not fit must throw ArgumentException");
        True(failures.Count == 0, string.Join("\n", failures));
        return Task.CompletedTask;
    }

    /// <summary>
    /// The tile grid under a level limit (Av1Level.TileLayout), by hand from the tile limits of the level tables:
    /// (width, height, tile size, maxTiles, frame rate, limit) -> tile columns x rows and the level then signalled.
    /// </summary>
    [TestMethod(Category = "CpuOnly")]
    public Task Av1Level_TileLayout_StaysInsideTheLevel()
    {
        CpuOnlyTest();
        var cases = new (int w, int h, int tileSize, int maxTiles, int fps, int limit, int cols, int rows, int level, string why)[]
        {
            (832, 480, 1, 256, 30, Av1Level.MaximumParameters, 13, 8, Av1Level.Level6_0, "no limit: one superblock per tile"),
            // 13 x 8 superblocks in at most 16 tiles and 6 columns: 2 x 8 tiles of 7 x 1 superblocks are the smallest
            (832, 480, 1, 256, 30, Av1Level.Level3_0, 2, 8, Av1Level.Level3_0, "3.0: MaxTiles 16, MaxTileCols 6"),
            (832, 480, 1, 256, 30, Av1Level.Level2_0, 2, 8, Av1Level.Level3_0, "the picture needs 3.0: its tile limits apply"),
            (832, 480, 1, 256, 30, Av1Level.Level5_1, 7, 8, Av1Level.Level5_0, "5.1: 64 tiles in 8 columns; 56 tiles need 5.0"),
            // at most 4 tiles: 1 x 4 tiles of 13 x 2 superblocks (26) are smaller than 2 x 2 of 7 x 4 (28)
            (832, 480, 1, 4, 30, Av1Level.Level5_1, 1, 4, Av1Level.Level3_0, "maxTiles 4 is below the level's"),
            // 30 x 17 superblocks in at most 32 tiles and 8 columns: 8 x 4 tiles of 4 x 5 superblocks
            (1920, 1080, 1, 256, 30, Av1Level.Level4_0, 8, 4, Av1Level.Level4_0, "4.0: MaxTiles 32, MaxTileCols 8"),
            // 240 fps needs 5.1 (497,664,000 samples per second); 64 * 120 tiles per second / 240 = 32 tiles per frame
            (1920, 1080, 1, 256, 240, Av1Level.Level4_0, 8, 4, Av1Level.Level5_1, "tiles per second <= MaxTiles * 120"),
            (1920, 1080, 1, 256, 30, Av1Level.Level6_0, 15, 6, Av1Level.Level6_0, "6.0: 128 tiles in 16 columns: 15 x 6 tiles of 2 x 3"),
            // 3 x 2 superblocks of a 130x66 picture: a second tile column or row would keep 2 pixels
            (130, 66, 1, 256, 30, Av1Level.Level2_0, 1, 1, Av1Level.Level2_0, "CroppedTileWidth / Height >= 8"),
            (136, 72, 1, 256, 30, Av1Level.Level2_0, 3, 2, Av1Level.Level2_0, "8 pixels left to the last column and row"),
            (135, 72, 1, 256, 30, Av1Level.Level2_0, 1, 2, Av1Level.Level2_0, "7 pixels left to the last column"),
        };
        var failures = new List<string>();
        foreach (var (w, h, tileSize, maxTiles, fps, limit, cols, rows, level, why) in cases)
        {
            var t = Av1Level.TileLayout(w, h, tileSize, maxTiles, fps, 1, limit);
            var got = Av1Level.Select(Av1LevelStreamOf(w, h, t, fps, 1));
            if (t.TileCols != cols || t.TileRows != rows || got != new Av1LevelSignal(level, 0))
                failures.Add($"{w}x{h} tile size {tileSize} max {maxTiles} at {fps} limit {limit}: {t.TileCols} x {t.TileRows} tiles, {got}; " +
                    $"expected {cols} x {rows}, seq_level_idx {level} ({why})");
        }
        // Without a limit the grid is Av1TileLayout.Compute's.
        True(Av1Level.TileLayout(1088, 1024, 1, 256, 30, 1, Av1Level.MaximumParameters).Equals(Av1TileLayout.Compute(1088, 1024, 1, 256)), "no limit = Compute");

        // Sweep: the tiles never make a stream need more than the limit, or than its picture and rate need alone.
        int checkedCount = 0;
        for (int w = 16; w <= 2000; w += 67)
            for (int h = 16; h <= 1100; h += 59)
                foreach (int fps in new[] { 24, 60 })
                    foreach (int limit in new[] { Av1Level.Level2_0, Av1Level.Level3_0, Av1Level.Level4_0, Av1Level.Level5_1, Av1Level.Level6_0 })
                        foreach (int tileSize in new[] { 1, 3 })
                        {
                            int alone = Av1Level.Select(Av1LevelStreamOf(w, h, Av1TileLayout.SingleTile(w, h), fps, 1)).SeqLevelIdx;
                            var t = Av1Level.TileLayout(w, h, tileSize, 256, fps, 1, limit);
                            int got = Av1Level.Select(Av1LevelStreamOf(w, h, t, fps, 1)).SeqLevelIdx;
                            if (got > Math.Max(alone, limit))
                                failures.Add($"{w}x{h} at {fps} tile size {tileSize} limit {limit}: {t.TileCols} x {t.TileRows} tiles need seq_level_idx {got}, the picture alone {alone}");
                            checkedCount++;
                        }
        True(checkedCount > 5000, $"sweep of {checkedCount}");

        bool threw = false;
        try { Av1Level.TileLayout(64, 64, 1, 256, 30, 1, 2); } catch (ArgumentOutOfRangeException) { threw = true; }
        True(threw, "a limit that is not a defined level must throw ArgumentOutOfRangeException");
        True(failures.Count == 0, string.Join("\n", failures.Take(20)));
        return Task.CompletedTask;
    }

    /// <summary>
    /// The sequence header writer codes seq_tier exactly when seq_level_idx is above 7, and the parser reads both
    /// back; the bit after it (frame_width_bits_minus_1 ...) lands where the parser expects it, or the picture size
    /// would come back wrong.
    /// </summary>
    [TestMethod(Category = "CpuOnly")]
    public Task Av1SequenceHeaderWriter_LevelAndTier_RoundTrip()
    {
        CpuOnlyTest();
        var failures = new List<string>();
        Av1SequenceHeaderConfig Config(int level, int tier) => new()
        {
            SeqProfile = 0, SeqLevelIdx0 = level, SeqTier0 = tier, MaxFrameWidth = 832, MaxFrameHeight = 480, BitDepth = 8,
            SeqChooseScreenContentTools = false, SeqForceScreenContentTools = 0,
        };
        byte[] low = Av1SequenceHeaderWriter.EmitPayload(Config(Av1Level.Level3_0, 0));
        foreach (var (level, tier) in new[] { (0, 0), (1, 0), (4, 0), (7, 0), (8, 0), (8, 1), (13, 1), (16, 0), (19, 1), (31, 0) })
        {
            byte[] payload = Av1SequenceHeaderWriter.EmitPayload(Config(level, tier));
            var sh = Av1SequenceHeaderParser.Parse(payload);
            if (sh.SeqLevelIdx0 != level || sh.SeqTier0 != tier || sh.MaxFrameWidth != 832 || sh.MaxFrameHeight != 480 || sh.BitDepth != 8)
                failures.Add($"level {level} tier {tier}: read back level {sh.SeqLevelIdx0} tier {sh.SeqTier0} {sh.MaxFrameWidth}x{sh.MaxFrameHeight} {sh.BitDepth}-bit");
            var again = Av1SequenceHeaderWriter.EmitPayload(Av1SequenceHeaderConfig.FromHeader(sh));
            if (!again.AsSpan().SequenceEqual(payload)) failures.Add($"level {level} tier {tier}: parse -> FromHeader -> emit changed the payload");
        }
        // the bits before seq_level_idx: 3 + 1 + 1 + 1 + 1 + 5 + 12 = 24 bits; then the level's 5 bits and the tier
        byte[] high = Av1SequenceHeaderWriter.EmitPayload(Config(Av1Level.Level6_0, 1));
        Equal(Av1Level.Level3_0 << 3, low[3] & 0xF8, "seq_level_idx 4 in the top 5 bits of byte 3");
        Equal((Av1Level.Level6_0 << 3) | 4, high[3] & 0xFC, "seq_level_idx 16, then seq_tier 1");
        Throws<ArgumentException>(() => Av1SequenceHeaderWriter.EmitPayload(Config(Av1Level.Level3_1, 1)), "seq_tier 1 below level 4.0");
        Throws<ArgumentOutOfRangeException>(() => Av1SequenceHeaderWriter.EmitPayload(Config(32, 0)), "seq_level_idx 32");
        Throws<ArgumentOutOfRangeException>(() => Av1SequenceHeaderWriter.EmitPayload(Config(8, 2)), "seq_tier 2");
        True(failures.Count == 0, string.Join("\n", failures));
        return Task.CompletedTask;
    }

    /// <summary>
    /// One sequence header for a file. Three temporal units of the CPU reference encoder (a hard picture at a fine
    /// and at a coarse quantizer, shown at 120 fps so the sizes land in different levels): each signals its own
    /// level; after UnifySequenceHeaders all carry the sequence header of the highest, and every other byte is as it
    /// was.
    /// </summary>
    [TestMethod(Category = "CpuOnly")]
    public Task Av1Isobmff_UnifySequenceHeaders_TheHighestLevelForEveryUnit()
    {
        CpuOnlyTest();
        const int w = 64, h = 64;
        var (y, u, v) = Av1EdgeTestPicture(w, h, 91);
        byte[] Unit(int q) => Av1KeyframeEncoder.EncodeKeyFrame(y, w, u, (w + 1) / 2, v, w, h, q, frameRateNum: 120);
        var originals = new List<byte[]> { Unit(230), Unit(12), Unit(90) };
        var before = originals.Select(Av1LevelRead).ToList();
        var levels = before.Select(b => b.Signal).ToList();
        True(levels.Distinct().Count() >= 2, "the units must signal different levels for this test to mean anything: " + string.Join(", ", levels));
        var highest = levels.OrderBy(s => s.SeqLevelIdx).ThenBy(s => s.SeqTier).Last();
        byte[] highestObu = before[levels.IndexOf(highest)].SequenceHeaderObu;

        foreach (bool samples in new[] { false, true })
        {
            var units = originals.Select(o => samples ? Av1Isobmff.ToSample(o) : (byte[])o.Clone()).ToList();
            var result = Av1Isobmff.UnifySequenceHeaders(units);
            Equal(highest.ToString(), result.ToString(), "the returned level");
            for (int k = 0; k < units.Count; k++)
            {
                var after = Av1LevelRead(units[k]);
                Equal(Convert.ToHexString(highestObu), Convert.ToHexString(after.SequenceHeaderObu), $"unit {k}: sequence header OBU");
                // everything but the sequence header OBU is unchanged
                var was = Av1TestObus(samples ? Av1Isobmff.ToSample(originals[k]) : originals[k]);
                var now = Av1TestObus(units[k]);
                Equal(was.Count, now.Count, $"unit {k}: OBU count");
                for (int o = 0; o < was.Count; o++)
                {
                    Equal(was[o].Type, now[o].Type, $"unit {k} OBU {o}: type");
                    if (was[o].Type == 1) continue;
                    byte[] source = samples ? Av1Isobmff.ToSample(originals[k]) : originals[k];
                    True(source.AsSpan(was[o].Start, was[o].Length).SequenceEqual(units[k].AsSpan(now[o].Start, now[o].Length)), $"unit {k} OBU {o} (type {was[o].Type}) changed");
                }
            }
            // the sequence as a whole conforms to the level it now signals, and to none below it
            var tiles = Av1TileLayout.Compute(w, h, 1);
            long maxFrame = units.Max(x => Av1LevelRead(x).FrameObuBytes);
            long maxUnit = originals.Max(o => (long)o.Length - before[originals.IndexOf(o)].SequenceHeaderObu.Length) + highestObu.Length;
            string? wrong = Av1LevelIsTheLowestConforming(result, Av1LevelStreamOf(w, h, tiles, 120, 1, maxFrame, maxUnit));
            True(wrong == null, $"the unified sequence {wrong}");
        }

        // not one sequence: a different picture size
        var (y2, u2, v2) = Av1EdgeTestPicture(96, 64, 5);
        var mixed = new List<byte[]> { Unit(60), Av1KeyframeEncoder.EncodeKeyFrame(y2, 96, u2, 48, v2, 96, 64, 60) };
        Throws<InvalidDataException>(() => Av1Isobmff.UnifySequenceHeaders(mixed), "sequence headers that differ in the picture size");
        Throws<InvalidDataException>(() => Av1Isobmff.UnifySequenceHeaders(new List<byte[]>()), "no unit");
        return Task.CompletedTask;
    }

    /// <summary>
    /// The assembly kernel's choice at the exact limits. One 320x200 tile at 30 fps; for each of the first candidates
    /// a synthetic tile whose Frame OBU is exactly the candidate's limit, and one byte more. The temporal unit must
    /// start with the prefix (temporal delimiter + sequence header) of the first candidate that allows the length:
    /// the limit itself still gets the candidate, one byte more gets a later one (4.0 High + 1 byte skips 4.1 Main,
    /// whose limit is lower, and lands on 4.1 High).
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "NotOnWebGL")]
    public async Task Av1ObuAssembleKernel_Level_PicksTheCandidateAtTheExactLimit()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            const int w = 320, h = 200, q = 32;
            var tiles = Av1TileLayout.SingleTile(w, h);
            var candidates = Av1KeyframeFraming.BuildLevelCandidates(w, h, tiles, 30, 1);
            byte[] blob = Av1KeyframeFraming.BuildFramingBlob(w, h, q, tiles, 30, 1, out int bOffset, out int bLen);
            True(candidates.Count >= 9, $"{candidates.Count} candidates");
            Equal("2.0 Main", candidates[0].Signal.ToString());
            Equal("4.0 High", candidates[5].Signal.ToString());
            Equal("4.1 Main", candidates[6].Signal.ToString());
            True(candidates[6].MaxFrameObuBytes < candidates[5].MaxFrameObuBytes, "4.1 Main allows less than 4.0 High");

            int cap = (int)candidates[7].MaxFrameObuBytes + 64;
            using var obu = new Av1ObuAssembleKernel(acc);
            using var dBlob = acc.Allocate1D<byte>(blob.Length);
            using var dTile = acc.Allocate1D<byte>(cap);
            using var dTileLen = acc.Allocate1D<long>(1);
            using var dOffsets = acc.Allocate1D<int>(Av1ObuAssembleKernel.OffsetsLength(1));
            using var dOut = acc.Allocate1D<byte>(cap);
            using var dOutLen = acc.Allocate1D<int>(1);
            dBlob.View.CopyFromCPU(blob);
            var tileBytes = new byte[cap];
            for (int i = 0; i < cap; i++) tileBytes[i] = (byte)(i * 7 + 3);
            dTile.View.CopyFromCPU(tileBytes);

            var failures = new List<string>();
            var report = new List<string>();
            for (int c = 0; c <= 6; c++)
                foreach (int over in new[] { 0, 1 })
                {
                    long target = candidates[c].MaxFrameObuBytes + over;   // Frame OBU bytes: 1 + leb128(payload) + payload
                    int tileLen = -1;
                    for (int leb = 1; leb <= 4 && tileLen < 0; leb++)
                    {
                        long payload = target - 1 - leb;
                        int need = payload < 1 << 7 ? 1 : payload < 1 << 14 ? 2 : payload < 1 << 21 ? 3 : 4;
                        if (need == leb && payload - bLen >= 1) tileLen = (int)(payload - bLen);
                    }
                    if (tileLen < 0) { failures.Add($"candidate {c} + {over}: no tile length gives a Frame OBU of {target} bytes"); continue; }
                    // the first candidate that allows the length (the expectation, from the limits as plain numbers)
                    int expected = 0;
                    while (candidates[expected].MaxFrameObuBytes < target) expected++;
                    if (over == 0 && expected != c && candidates[c].MaxFrameObuBytes > candidates[Math.Max(c - 1, 0)].MaxFrameObuBytes)
                        failures.Add($"candidate {c}: the test expects candidate {expected} at its own limit");

                    dTileLen.View.CopyFromCPU(new long[] { tileLen });
                    obu.Run(dBlob.View, dTile.View, dTileLen.View, dOffsets.View, dOut.View, dOutLen.View, bOffset, bLen, 1, cap, cap);
                    await acc.SynchronizeAsync();
                    int n = (await dOutLen.CopyToHostAsync())[0];
                    byte[] unit = await dOut.View.SubView(0, n).CopyToHostAsync();
                    byte[] prefix = candidates[expected].Prefix;
                    string tag = $"Frame OBU {target} bytes (limit of {candidates[c].Signal}{(over == 1 ? " + 1" : "")})";
                    if (n != prefix.Length + target) { failures.Add($"{tag}: unit of {n} bytes, expected {prefix.Length + target}"); continue; }
                    var read = Av1LevelRead(unit);
                    if (read.Signal != candidates[expected].Signal) failures.Add($"{tag}: signals {read.Signal}, expected {candidates[expected].Signal}");
                    if (!unit.AsSpan(0, prefix.Length).SequenceEqual(prefix)) failures.Add($"{tag}: the unit does not start with the prefix of {candidates[expected].Signal}");
                    if (read.FrameObuBytes != target) failures.Add($"{tag}: the Frame OBU is {read.FrameObuBytes} bytes");
                    // the tile bytes follow the uncompressed header untouched
                    if (!unit.AsSpan(n - tileLen, tileLen).SequenceEqual(tileBytes.AsSpan(0, tileLen))) failures.Add($"{tag}: the tile bytes changed");
                    if (!unit.AsSpan(n - tileLen - bLen, bLen).SequenceEqual(blob.AsSpan(bOffset, bLen))) failures.Add($"{tag}: the uncompressed header changed");
                    report.Add($"{target} -> {read.Signal}");
                }
            Console.WriteLine($"{acc.AcceleratorType}: AV1 level chosen by the assembly kernel, Frame OBU bytes -> level: " + string.Join(" | ", report));
            True(failures.Count == 0, string.Join("\n", failures));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    /// <summary>(width, height, tile size, q, frame rate, level limit, picture seed).</summary>
    private static readonly (int w, int h, int tileSize, int q, int num, int den, int limit, int seed)[] Av1LevelCases =
    {
        (130, 66, int.MaxValue, 48, 30, 1, Av1Level.MaximumParameters, 1),      // one tile, a small frame
        (130, 66, 1, 48, 30, 1, Av1Level.MaximumParameters, 1),                 // a 2-pixel tile column: no level
        (130, 66, 1, 48, 30, 1, Av1Level.Level2_0, 1),                          // the limit merges the tiles
        (200, 9, int.MaxValue, 48, 30, 1, Av1Level.MaximumParameters, 2),       // 9 rows: no level
        (320, 200, 3, 16, 30, 1, Av1Level.MaximumParameters, 3),                // a fine quantizer: many bytes
        (320, 200, 3, 230, 30, 1, Av1Level.MaximumParameters, 3),               // the same picture, coarse: few bytes
        (320, 200, 3, 230, 240, 1, Av1Level.MaximumParameters, 3),              // the same bytes at 8 times the rate
        (320, 200, 3, 60, 30000, 1001, Av1Level.MaximumParameters, 3),
        (832, 480, 1, 40, 30, 1, Av1Level.MaximumParameters, 4),                // 104 tiles in 13 columns
        (832, 480, 1, 40, 30, 1, Av1Level.Level3_0, 4),                         // 2 x 8 tiles of 7 x 1 superblocks
        (832, 480, 1, 200, 30, 1, Av1Level.Level5_1, 4),                        // 7 x 8 tiles
    };

    /// <summary>
    /// Real encodes on the accelerator: the temporal unit equals the CPU reference's (which picks the level on the
    /// host), and the level it signals is the lowest the unit conforms to, checked with the specification's
    /// inequalities against the unit's measured Frame OBU and total bytes. The cases that fit no level signal 31; a
    /// finer quantizer and a higher frame rate must each raise the level of the same picture.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "NotOnWebGL")]
    public async Task Av1KeyframeEncoderGpu_Level_SignalsTheLowestLevelTheStreamConformsTo()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var enc = new Av1KeyframeEncoderGpu(acc);
            Equal(30, enc.FrameRateNumerator, "default frame rate");
            Equal(1, enc.FrameRateDenominator, "default frame rate");
            Equal(Av1Level.MaximumParameters, enc.LevelLimit, "default: no level limit");
            bool threwRate = false, threwLimit = false;
            try { enc.SetFrameRate(0); } catch (ArgumentOutOfRangeException) { threwRate = true; }
            try { enc.LevelLimit = 3; } catch (ArgumentOutOfRangeException) { threwLimit = true; }
            True(threwRate, "a frame rate of 0 must throw ArgumentOutOfRangeException");
            True(threwLimit, "a level limit that is not a defined level must throw ArgumentOutOfRangeException");
            Equal(30, enc.FrameRateNumerator, "a rejected frame rate must leave the setting unchanged");
            Equal(Av1Level.MaximumParameters, enc.LevelLimit, "a rejected limit must leave the setting unchanged");

            var failures = new List<string>();
            var report = new List<string>();
            var signals = new List<Av1LevelSignal>();
            foreach (var (w, h, tileSize, q, num, den, limit, seed) in Av1LevelCases)
            {
                string tag = $"{Av1TileTag(w, h, tileSize)} q{q} at {num}/{den}{(limit == Av1Level.MaximumParameters ? "" : $" limit {limit}")}";
                var (y, u, v) = Av1EdgeTestPicture(w, h, 700 + seed);
                enc.TileSizeSuperblocks = tileSize;
                enc.SetFrameRate(num, den);
                enc.LevelLimit = limit;
                var layout = enc.GetTileLayout(w, h);
                byte[] cpu = Av1KeyframeEncoder.EncodeKeyFrame(y, w, u, (w + 1) / 2, v, w, h, q, tileSize,
                    frameRateNum: num, frameRateDen: den, levelLimit: limit);
                byte[] gpu = await enc.EncodeKeyFrameAsync(y, u, v, w, h, q);
                if (!gpu.AsSpan().SequenceEqual(cpu))
                {
                    int first = 0, lim = Math.Min(gpu.Length, cpu.Length);
                    while (first < lim && gpu[first] == cpu[first]) first++;
                    failures.Add($"{tag}: {gpu.Length} bytes vs CPU reference {cpu.Length}, first difference at byte {first}");
                }
                var read = Av1LevelRead(gpu);
                signals.Add(read.Signal);
                string? wrong = Av1LevelIsTheLowestConforming(read.Signal, Av1LevelStreamOf(w, h, layout, num, den, read.FrameObuBytes, gpu.Length));
                if (wrong != null) failures.Add($"{tag}: the {gpu.Length}-byte unit {wrong}");
                report.Add($"{tag}: {layout.TileCols} x {layout.TileRows} tiles, {gpu.Length} B, {read.Signal}");
            }
            // What the cases are there to show (indices into Av1LevelCases).
            void Expect(int k, bool condition, string what)
            {
                if (!condition) failures.Add($"case {k} ({signals[k]}): {what}");
            }
            int Rank(Av1LevelSignal s) => s.SeqLevelIdx * 2 + s.SeqTier;
            Expect(1, signals[1].IsMaximumParameters, "a 2-pixel tile column fits no level");
            Expect(2, !signals[2].IsMaximumParameters, "the level limit must give a grid that fits a level");
            Expect(3, signals[3].IsMaximumParameters, "a 9-row frame fits no level");
            Expect(4, Rank(signals[4]) > Rank(signals[5]), $"q16 must signal a higher level than q230 ({signals[5]})");
            Expect(6, Rank(signals[6]) > Rank(signals[5]), $"240 fps must signal a higher level than 30 fps ({signals[5]})");
            Expect(8, signals[8].SeqLevelIdx >= Av1Level.Level6_0, "104 tiles in 13 columns need 6.0 at least");
            Expect(9, Rank(signals[9]) < Rank(signals[8]), $"the level 3.0 grid must signal a lower level than the default grid ({signals[8]})");
            Expect(10, signals[10].SeqLevelIdx <= Av1Level.Level5_1, "a limit of 5.1 and a coarse quantizer: 5.1 at most");
            Console.WriteLine($"{acc.AcceleratorType}: AV1 level signalled: " + string.Join(" | ", report));
            True(failures.Count == 0, string.Join("\n", failures));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    /// <summary>
    /// FFmpeg as a black box: ffprobe reads the level the stream signals (its own sequence header parser), and
    /// libdav1d decodes the stream to the encoder's reconstruction, byte for byte: with the tier bit present (levels
    /// from 4.0), with 31, and with the tile shapes a level limit gives.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "NotOnWebGL,DesktopOnly")]
    public async Task Av1KeyframeEncoderGpu_Level_FfmpegReadsTheSignalledLevel()
    {
        FfmpegOracle.RequireAvailable();
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var enc = new Av1KeyframeEncoderGpu(acc);
            var failures = new List<string>();
            var report = new List<string>();
            bool sawTier = false, sawLow = false, sawMax = false;
            foreach (var (w, h, tileSize, q, num, den, limit, seed) in Av1LevelCases)
            {
                string tag = $"{Av1TileTag(w, h, tileSize)} q{q} at {num}/{den}{(limit == Av1Level.MaximumParameters ? "" : $" limit {limit}")}";
                var (y, u, v) = Av1EdgeTestPicture(w, h, 700 + seed);
                enc.TileSizeSuperblocks = tileSize;
                enc.SetFrameRate(num, den);
                enc.LevelLimit = limit;
                byte[] unit = await enc.EncodeKeyFrameAsync(y, u, v, w, h, q);
                byte[] recon = await enc.ReadReconAsync();
                var signal = Av1LevelRead(unit).Signal;
                sawTier |= signal.SeqLevelIdx > 7 && !signal.IsMaximumParameters;
                sawLow |= signal.SeqLevelIdx <= 7;
                sawMax |= signal.IsMaximumParameters;
                string probed;
                try { probed = FfmpegOracle.Probe(unit, ".obu", "-select_streams", "v:0", "-show_entries", "stream=level", "-of", "csv=p=0").Trim(); }
                catch (Exception e) { failures.Add($"{tag}: ffprobe failed: {e.Message}"); continue; }
                if (probed != signal.SeqLevelIdx.ToString()) failures.Add($"{tag}: ffprobe reads level '{probed}', the stream signals seq_level_idx {signal.SeqLevelIdx}");
                List<byte[]> decoded;
                try { decoded = FfmpegOracle.DecodeVideoI420(unit, ".obu", w, h); }
                catch (Exception e) { failures.Add($"{tag}: ffmpeg could not decode the {unit.Length}-byte temporal unit: {e.Message}"); continue; }
                if (decoded.Count != 1 || !decoded[0].AsSpan().SequenceEqual(recon))
                    failures.Add($"{tag}: ffmpeg decoded {decoded.Count} frames; the picture is not the encoder's reconstruction");
                report.Add($"{tag}: {signal} = ffprobe {probed}");
            }
            True(sawTier && sawLow && sawMax, $"the cases must cover a level with the tier bit ({sawTier}), one without ({sawLow}) and 31 ({sawMax})");
            Console.WriteLine($"{acc.AcceleratorType}: AV1 level read by ffprobe: " + string.Join(" | ", report));
            True(failures.Count == 0, string.Join("\n", failures));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    /// <summary>
    /// A file has one level. Frames of very different sizes (flat, noise, flat) at 120 fps would each signal their
    /// own; in the MP4 every sample carries the same sequence header, the av1C record states its level and tier, and
    /// the file as a whole (its largest frame, its frame rate, its tiles) conforms to that level and to none below.
    /// Then the level limit option: a 320x64 picture is 5 tile columns by default (level 3.0 at least) and 3 with
    /// a limit of 2.0.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "NotOnWebGL")]
    public async Task VideoFileEncoderGpu_Av1_OneLevelForTheFile()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            const int w = 64, h = 64, fps = 120;
            var frames = new List<float[]>();
            uint state = 12345;
            for (int k = 0; k < 3; k++)
            {
                var f = new float[w * h * 3];
                for (int i = 0; i < f.Length; i++)
                {
                    state = state * 1664525 + 1013904223;
                    f[i] = k == 1 ? (state >> 8) / 16777216f : 0.5f;
                }
                frames.Add(f);
            }
            using var enc = new VideoFileEncoderGpu(acc);
            var options = new VideoFileEncodeOptions { Codec = VideoCodec.Av1, QIndex = 20 };
            var mp4 = await enc.EncodeAsync(frames, w, h, fps, 1, null, options);
            var samples = Mp4Samples(mp4, out var av1c);
            Equal(3, samples.Count);
            True(av1c != null, "no av1C");
            var reads = samples.Select(Av1LevelRead).ToList();
            var tiles = Av1TileLayout.Compute(w, h, 1);
            // each frame alone: the flat ones fit a lower level than the noise (or the test shows nothing)
            const int td = 2;   // the temporal delimiter OBU the sample form drops
            var alone = samples.Select((s, k) => Av1Level.Select(Av1LevelStreamOf(w, h, tiles, fps, 1, reads[k].FrameObuBytes, s.Length + td))).ToList();
            True(alone.Distinct().Count() >= 2, "the frames alone must fit different levels for this test to mean anything: " + string.Join(", ", alone));
            // one sequence header
            for (int k = 1; k < samples.Count; k++)
                Equal(Convert.ToHexString(reads[0].SequenceHeaderObu), Convert.ToHexString(reads[k].SequenceHeaderObu), $"sample {k}: sequence header OBU");
            var signal = reads[0].Signal;
            Equal(signal.SeqLevelIdx, av1c![1] & 31, "av1C seq_level_idx_0");
            Equal(signal.SeqTier, av1c[2] >> 7, "av1C seq_tier_0");
            Equal(Convert.ToHexString(Av1Isobmff.BuildAv1C(samples[0])), Convert.ToHexString(av1c));
            // the file conforms to it and to nothing lower
            string? wrong = Av1LevelIsTheLowestConforming(signal,
                Av1LevelStreamOf(w, h, tiles, fps, 1, reads.Max(r => r.FrameObuBytes), samples.Max(s => s.Length) + td));
            True(wrong == null, $"the file {wrong}");
            var highestAlone = alone.OrderBy(s => s.SeqLevelIdx).ThenBy(s => s.SeqTier).Last();
            Equal(highestAlone.ToString(), signal.ToString(), "the file's level is its largest frame's");

            // The level limit option reaches the encoder: 5 tile columns need 3.0 (MaxTileCols 6), 3 columns fit 2.0.
            var smooth = TestFrames(320, 64, 2, smooth: true);
            var coarse = new VideoFileEncodeOptions { Codec = VideoCodec.Av1, QIndex = 200 };
            var free = Av1LevelRead(Mp4Samples(await enc.EncodeAsync(smooth, 320, 64, 16, 1, null, coarse), out _)[0]);
            var limited = Av1LevelRead(Mp4Samples(await enc.EncodeAsync(smooth, 320, 64, 16, 1, null, coarse with { Av1LevelLimit = Av1Level.Level2_0 }), out _)[0]);
            True(free.Signal.SeqLevelIdx >= Av1Level.Level3_0, $"320x64 without a limit signals {free.Signal}; 5 tile columns need 3.0 at least");
            True(limited.Signal.SeqLevelIdx < Av1Level.Level3_0, $"320x64 with a limit of 2.0 signals {limited.Signal}");
            // and back: the limit of one file does not stay for the next
            var freeAgain = Av1LevelRead(Mp4Samples(await enc.EncodeAsync(smooth, 320, 64, 16, 1, null, coarse), out _)[0]);
            Equal(free.Signal.ToString(), freeAgain.Signal.ToString(), "the level limit of a file must not carry over to the next");
            Console.WriteLine($"{acc.AcceleratorType}: AV1 file levels: frames alone {string.Join(", ", alone)} -> file {signal}; 320x64 {free.Signal}, limited to 2.0 {limited.Signal}");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }
}
