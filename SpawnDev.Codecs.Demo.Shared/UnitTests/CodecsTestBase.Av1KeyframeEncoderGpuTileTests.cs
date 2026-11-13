// Av1KeyframeEncoderGpu with tiles: one GPU thread per tile of every frame (TileSizeSuperblocks).
//
// What a tile changes: the frame header's tile_info (uniform spacing, TileColsLog2 / TileRowsLog2), the tile group
// (a flag byte, a 4-byte tile_size_minus_1 before every tile but the last), and inside a tile: a fresh range coder,
// zeroed contexts, and no above / left neighbour at the tile's first row / column.
//
// Oracles, and what each can fail on:
//  - Av1TileLayout_Compute (no accelerator): the grid against values derived by hand from AV1 spec 5.9.15.
//  - Tiles_MatchCpuReference (every backend that runs serial stream kernels): the temporal unit equals the CPU
//    Av1KeyframeEncoder's for the same tile size, byte for byte. Catches a kernel that differs from the reference
//    (codegen, a thread reading another tile's slot), not a reading of the specification that both share.
//  - Tiles_FfmpegDecodesWhatTheEncoderReconstructs (desktop): FFmpeg (libdav1d), a black-box executable, decodes the
//    temporal unit; its picture equals the encoder's reconstruction, byte for byte. Fails on a wrong tile_info, a
//    wrong size prefix, a missing flag byte, or a neighbour read across a tile edge.
//  - Tiles_BatchEqualsSingleFrameEncodes: working thread = frame * tileCount + tile; every frame of a batch equals
//    its single-frame encode.
//  - SerialThreadPlacement (no accelerator): where the tile threads sit in a launch; every working thread exactly
//    once. On CUDA / OpenCL the encode tests above run through a spread placement, on the others through the packed one.
//  - Tiles_DefaultLayoutDecodesOnHardwareDecoders (desktop): a frame gets at most MaxTiles tiles (256, the measured
//    limit of NVDEC / D3D11VA / D3D12VA); the default stream of a picture with 272 superblocks decodes on libaom
//    and on every hardware decoder FFmpeg has on the machine.
//  - Tiles_AreFasterThanOneTile: the point of it all, asserted on the GPU backends (4x; measured 69x to 95x), and
//    the times go to the run log ("[report]" lines) on every lane.
// Sizes: 130x66 (3 x 2 superblocks, both edges partial), 200x9 (4 x 1, one row of 8x8 leaves), 320x200 with tile
// size 3 (5 x 4 superblocks in tiles of 3 x 2: a narrower last tile column, several superblock rows per tile),
// 832x480 (13 x 8 superblocks: 104 tiles at tile size 1), 1088x1024 (17 x 16 = 272 superblocks: over the limit,
// coded as 9 x 16 tiles of 2 x 1).

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Av1;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    /// <summary>(width, height, tileSizeSuperblocks, expected tile columns, expected tile rows).</summary>
    private static readonly (int w, int h, int tileSize, int cols, int rows)[] Av1TileCases =
    {
        (130, 66, 1, 3, 2), (130, 66, 2, 2, 1), (130, 66, int.MaxValue, 1, 1),
        (200, 9, 1, 4, 1), (200, 9, 2, 2, 1), (200, 9, int.MaxValue, 1, 1),
        (320, 200, 3, 2, 2),
        (832, 480, 1, 13, 8), (832, 480, 2, 7, 4), (832, 480, int.MaxValue, 1, 1),
        // 17 x 16 superblocks = 272 one-superblock tiles, more than the 256 a frame gets by default: 9 x 16 tiles
        (1088, 1024, 1, 9, 16),
    };

    private static string Av1TileTag(int w, int h, int tileSize) =>
        $"{w}x{h} tile size {(tileSize == int.MaxValue ? "max" : tileSize.ToString())}";

    /// <summary>
    /// The tile grid against values derived by hand from the specification's tile_info() (uniform spacing):
    /// (width, height, tile size) -> superblocks, the coded log2 values and their codable range, tile size, tile count.
    /// </summary>
    [TestMethod(Category = "CpuOnly")]
    public Task Av1TileLayout_Compute_MatchesTheSpecificationDerivation()
    {
        CpuOnlyTest();
        // w, h, tileSize, maxTiles -> sbCols, sbRows, colsLog2, rowsLog2, minCols, maxCols, minRows, maxRows, tileW, tileH, cols, rows
        const int any = int.MaxValue, hw = Av1TileLayout.HardwareDecoderMaxTiles;
        var cases = new (int w, int h, int ts, int maxTiles, int[] expected)[]
        {
            (64, 64, 1, hw,             new[] { 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1 }),
            (1, 1, 1, hw,               new[] { 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1 }),
            (65, 64, 1, hw,             new[] { 2, 1, 1, 0, 0, 1, 0, 0, 1, 1, 2, 1 }),
            (130, 66, 1, hw,            new[] { 3, 2, 2, 1, 0, 2, 0, 1, 1, 1, 3, 2 }),
            (130, 66, 2, hw,            new[] { 3, 2, 1, 0, 0, 2, 0, 1, 2, 2, 2, 1 }),
            (130, 66, any, hw,          new[] { 3, 2, 0, 0, 0, 2, 0, 1, 3, 2, 1, 1 }),
            (200, 9, 1, hw,             new[] { 4, 1, 2, 0, 0, 2, 0, 0, 1, 1, 4, 1 }),
            (320, 200, 3, hw,           new[] { 5, 4, 1, 1, 0, 3, 0, 2, 3, 2, 2, 2 }),
            (832, 480, 1, hw,           new[] { 13, 8, 4, 3, 0, 4, 0, 3, 1, 1, 13, 8 }),
            (832, 480, 2, hw,           new[] { 13, 8, 3, 2, 0, 4, 0, 3, 2, 2, 7, 4 }),
            // a request of 3 gets tiles of 2: uniform spacing only halves, (13 + 3) >> 2 = 4 is still above 3
            (832, 480, 3, hw,           new[] { 13, 8, 3, 2, 0, 4, 0, 3, 2, 2, 7, 4 }),
            (832, 480, 4, hw,           new[] { 13, 8, 2, 1, 0, 4, 0, 3, 4, 4, 4, 2 }),
            (832, 480, any, hw,         new[] { 13, 8, 0, 0, 0, 4, 0, 3, 13, 8, 1, 1 }),
            (1024, 1024, 1, hw,         new[] { 16, 16, 4, 4, 0, 4, 0, 4, 1, 1, 16, 16 }),     // 256 tiles: exactly the limit
            (1920, 1080, 1, any,        new[] { 30, 17, 5, 5, 0, 5, 0, 5, 1, 1, 30, 17 }),     // 510 tiles without the limit
            // The limit of 256 tiles. 1920x1080: 30 x 17 = 510 is too many; halving the columns gives 15 x 17 = 255
            // tiles of 2 x 1 superblocks (halving the rows instead: 30 x 9 = 270, still too many).
            (1920, 1080, 1, hw,         new[] { 30, 17, 4, 5, 0, 5, 0, 5, 2, 1, 15, 17 }),
            // 1088x1024: 17 x 16 = 272. 9 x 16 = 144 and 17 x 8 = 136 both have tiles of 2 superblocks: the one with more tiles.
            (1088, 1024, 1, hw,         new[] { 17, 16, 4, 4, 0, 5, 0, 4, 2, 1, 9, 16 }),
            // 1280x1024: 20 x 16 = 320. 20 x 8 and 10 x 16 are both 160 tiles of 2 superblocks: the one with more columns.
            (1280, 1024, 1, hw,         new[] { 20, 16, 5, 3, 0, 5, 0, 4, 1, 2, 20, 8 }),
            // 3840x2160: 60 x 34 = 2040. 15 x 17 = 255 tiles of 4 x 2 superblocks (8); 60 x 4 = 240 tiles of 1 x 9 is the next best (9).
            (3840, 2160, 1, hw,         new[] { 60, 34, 4, 5, 0, 6, 0, 6, 4, 2, 15, 17 }),
            // a limit of 16: 2 x 8 tiles of 7 x 1 superblocks (7) beat 4 x 4 tiles of 4 x 2 (8)
            (832, 480, 1, 16,           new[] { 13, 8, 1, 3, 0, 4, 0, 3, 7, 1, 2, 8 }),
            (832, 480, 1, 1,            new[] { 13, 8, 0, 0, 0, 4, 0, 3, 13, 8, 1, 1 }),
            // 128 superblock columns: a tile is at most 64 superblocks (4096 pixels) wide, and there are at most 64
            // tile columns, so the smallest tile is 2 superblocks wide; 128 x 68 superblocks exceed one tile's area
            (8192, 4320, 1, any,        new[] { 128, 68, 6, 6, 1, 6, 0, 6, 2, 2, 64, 34 }),
            (8192, 4320, any, hw,       new[] { 128, 68, 1, 1, 1, 6, 1, 6, 64, 34, 2, 2 }),
            // a limit below what the picture size needs (4 tiles at least): the fewest tiles
            (8192, 4320, 1, 1,          new[] { 128, 68, 1, 1, 1, 6, 1, 6, 64, 34, 2, 2 }),
            // 64 x 40 superblocks = 2560 > 2304: one tile column is allowed, one tile is not
            (4096, 2560, any, hw,       new[] { 64, 40, 0, 1, 0, 6, 1, 6, 64, 20, 1, 2 }),
        };
        var failures = new List<string>();
        foreach (var (w, h, ts, maxTiles, e) in cases)
        {
            var t = Av1TileLayout.Compute(w, h, ts, maxTiles);
            if (maxTiles == hw && !Av1TileLayout.Compute(w, h, ts).Equals(t)) failures.Add($"{Av1TileTag(w, h, ts)}: the default limit is not {hw}");
            if (t.TileCount > maxTiles && (w, h, maxTiles) != (8192, 4320, 1)) failures.Add($"{Av1TileTag(w, h, ts)}: {t.TileCount} tiles exceed the limit of {maxTiles}");
            int[] got =
            {
                t.SbCols, t.SbRows, t.TileColsLog2, t.TileRowsLog2, t.MinLog2TileCols, t.MaxLog2TileCols,
                t.MinLog2TileRows, t.MaxLog2TileRows, t.TileWidthSb, t.TileHeightSb, t.TileCols, t.TileRows,
            };
            if (!got.AsSpan().SequenceEqual(e))
                failures.Add($"{Av1TileTag(w, h, ts)}: got [{string.Join(", ", got)}], expected [{string.Join(", ", e)}]");
            if (t.TileCount != e[10] * e[11]) failures.Add($"{Av1TileTag(w, h, ts)}: TileCount {t.TileCount}");
        }
        var single = Av1TileLayout.SingleTile(832, 480);
        if (single.TileCount != 1) failures.Add($"SingleTile(832, 480) has {single.TileCount} tiles");
        True(failures.Count == 0, string.Join("\n", failures));

        bool threwZero = false, threwNegative = false, threwReference = false, threwMaxTiles = false;
        try { Av1TileLayout.Compute(64, 64, 1, 0); } catch (ArgumentOutOfRangeException) { threwMaxTiles = true; }
        True(threwMaxTiles, "a tile limit of 0 must throw ArgumentOutOfRangeException");
        try { Av1TileLayout.Compute(64, 64, 0); } catch (ArgumentOutOfRangeException) { threwZero = true; }
        try { Av1TileLayout.Compute(64, 64, -1); } catch (ArgumentOutOfRangeException) { threwNegative = true; }
        var plane = new byte[64 * 64];
        try { Av1KeyframeEncoder.EncodeKeyFrame(plane, 64, plane, 32, plane, 64, 64, 32, 0); }
        catch (ArgumentOutOfRangeException) { threwReference = true; }
        True(threwZero, "a tile size of 0 must throw ArgumentOutOfRangeException");
        True(threwNegative, "a negative tile size must throw ArgumentOutOfRangeException");
        True(threwReference, "the CPU reference must reject a tile size of 0");
        return Task.CompletedTask;
    }

    /// <summary>
    /// SerialThreadPlacement, host arithmetic only: for every combination the working threads of the launch are
    /// exactly 0 .. ThreadCount - 1, each once (a thread missing = a tile never encoded, a thread twice = two launch
    /// threads writing one tile's state), and the layouts measured on the RTX 4070 are the ones chosen.
    /// </summary>
    [TestMethod(Category = "CpuOnly")]
    public Task SerialThreadPlacement_EveryWorkingThreadOnce_AndTheMeasuredLayouts()
    {
        CpuOnlyTest();
        var failures = new List<string>();
        // threads, groupSize, warpSize, multiprocessors -> expected (groupSize, warpSize, warps, lanes)
        var layouts = new (int n, int g, int ws, int sm, int[] expected)[]
        {
            (104, 640, 32, 46, new[] { 640, 32, 3, 1 }),      // CUDA, one 832x480 frame: 3 threads per group, a warp each
            (104, 256, 32, 46, new[] { 256, 32, 3, 1 }),      // OpenCL
            (1664, 640, 32, 46, new[] { 640, 32, 19, 2 }),    // 16 frames: 37 per group -> 2 lanes in 19 warps
            (6656, 640, 32, 46, new[] { 640, 32, 19, 8 }),    // 64 frames: 145 per group -> 8 lanes in 19 warps
            (104, 64, 32, 64, new[] { 64, 32, 2, 1 }),        // WebGPU (group 64, two halves, 64 groups assumed): 2 per group
            (832, 64, 32, 64, new[] { 64, 32, 2, 7 }),        // WebGPU, 8 frames: 13 per group -> 7 lanes in both halves
            (1, 640, 32, 46, new[] { 640, 32, 1, 1 }),
            (46, 640, 32, 46, new[] { 640, 32, 1, 1 }),
            (47, 640, 32, 46, new[] { 640, 32, 2, 1 }),
            (46 * 640, 640, 32, 46, new[] { 1, 1, 1, 1 }),    // every group full: packed
            (46 * 640 - 1, 640, 32, 46, new[] { 1, 1, 1, 1 }),
            (46 * 639, 640, 32, 46, new[] { 640, 32, 20, 32 }),
            (104, 8, 8, 12, new[] { 1, 1, 1, 1 }),            // 9 threads per group exceed a group of 8: packed
            (60, 8, 8, 12, new[] { 8, 8, 1, 5 }),             // one warp per group: 5 lanes of it
            (104, 100, 32, 46, new[] { 1, 1, 1, 1 }),         // group not a whole number of warps: packed
            (104, 16, 32, 46, new[] { 1, 1, 1, 1 }),          // group smaller than a warp: packed
            (104, 640, 0, 46, new[] { 1, 1, 1, 1 }),          // no warp size reported: packed
            (104, 640, 32, 0, new[] { 1, 1, 1, 1 }),          // no multiprocessor count reported: packed
        };
        foreach (var (n, g, ws, sm, expected) in layouts)
        {
            var p = SerialThreadPlacement.Create(n, g, ws, sm);
            string tag = $"threads {n} group {g} warp {ws} multiprocessors {sm}";
            int[] got = { p.GroupSize, p.WarpSize, p.WarpsPerGroup, p.LanesPerWarp };
            if (!got.AsSpan().SequenceEqual(expected))
                failures.Add($"{tag}: layout [{string.Join(", ", got)}], expected [{string.Join(", ", expected)}]");
            if (p.ThreadCount != n) failures.Add($"{tag}: ThreadCount {p.ThreadCount}");
        }
        // Every working thread exactly once, over a sweep of device shapes and thread counts.
        foreach (int g in new[] { 1, 32, 64, 256, 640, 1024 })
            foreach (int ws in new[] { 1, 8, 32, 64 })
                foreach (int sm in new[] { 1, 12, 46 })
                    foreach (int n in new[] { 1, 2, 3, 31, 32, 33, 104, 105, 1000, 1664, 4099 })
                    {
                        var p = SerialThreadPlacement.Create(n, g, ws, sm);
                        string tag = $"threads {n} group {g} warp {ws} multiprocessors {sm}";
                        long extent = p.Extent;
                        if (extent % p.GroupSize != 0) { failures.Add($"{tag}: extent {extent} is not whole groups of {p.GroupSize}"); continue; }
                        if (p.LanesPerWarp > p.WarpSize || p.WarpsPerGroup * p.WarpSize > p.GroupSize)
                        { failures.Add($"{tag}: {p.WarpsPerGroup} warps x {p.LanesPerWarp} lanes do not fit a group of {p.GroupSize} (warp {p.WarpSize})"); continue; }
                        var seen = new int[n];
                        bool bad = false;
                        for (long i = 0; i < extent && !bad; i++)
                        {
                            int t = SerialThreadPlacement.WorkingThread(p, (int)i);
                            if (t == -1) continue;
                            if (t < 0 || t >= n) { failures.Add($"{tag}: launch thread {i} -> working thread {t}"); bad = true; }
                            else seen[t]++;
                        }
                        if (bad) continue;
                        for (int t = 0; t < n; t++)
                            if (seen[t] != 1) { failures.Add($"{tag}: working thread {t} is run by {seen[t]} launch threads"); break; }
                    }
        var packed = SerialThreadPlacement.Packed(7);
        if (packed.Extent != 7 || SerialThreadPlacement.WorkingThread(packed, 6) != 6) failures.Add("Packed(7) is not the identity");
        bool threw = false;
        try { SerialThreadPlacement.Create(0, 640, 32, 46); } catch (ArgumentOutOfRangeException) { threw = true; }
        if (!threw) failures.Add("a thread count of 0 must throw ArgumentOutOfRangeException");
        True(failures.Count == 0, string.Join("\n", failures));
        return Task.CompletedTask;
    }

    [TestMethod(Timeout = 1800000, Category = "NotOnWebGL")]
    public async Task Av1KeyframeEncoderGpu_Tiles_MatchCpuReference()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var enc = new Av1KeyframeEncoderGpu(acc);
            Equal(1, enc.TileSizeSuperblocks, "default tile size: one superblock per tile");
            bool threwZero = false;
            try { enc.TileSizeSuperblocks = 0; } catch (ArgumentOutOfRangeException) { threwZero = true; }
            True(threwZero, "TileSizeSuperblocks = 0 must throw ArgumentOutOfRangeException");
            Equal(1, enc.TileSizeSuperblocks, "a rejected tile size must leave the setting unchanged");

            const int baseQIndex = 48;
            var failures = new List<string>();
            var report = new List<string>();
            byte[]? previous = null;
            (int w, int h) previousSize = (0, 0);
            int previousTiles = 0;
            foreach (var (w, h, tileSize, cols, rows) in Av1TileCases)
            {
                string tag = Av1TileTag(w, h, tileSize);
                // the same picture for every tile size of a picture size, so the sizes can be compared
                var (y, u, v) = Av1EdgeTestPicture(w, h, 500 + w);
                enc.TileSizeSuperblocks = tileSize;
                var layout = enc.GetTileLayout(w, h);
                if (layout.TileCols != cols || layout.TileRows != rows)
                    failures.Add($"{tag}: grid {layout.TileCols} x {layout.TileRows}, expected {cols} x {rows}");
                byte[] cpu = Av1KeyframeEncoder.EncodeKeyFrame(y, w, u, (w + 1) / 2, v, w, h, baseQIndex, tileSize);
                byte[] gpu = await enc.EncodeKeyFrameAsync(y, u, v, w, h, baseQIndex);
                report.Add($"{tag}: {layout.TileCount} tiles, {gpu.Length} B");
                if (!gpu.AsSpan().SequenceEqual(cpu))
                {
                    int first = 0, lim = Math.Min(gpu.Length, cpu.Length);
                    while (first < lim && gpu[first] == cpu[first]) first++;
                    failures.Add($"{tag}: {gpu.Length} bytes vs CPU reference {cpu.Length}, first difference at byte {first}");
                }
                // The tile size must change the stream: a setting that is ignored would pass everything above.
                if (previous != null && previousSize == (w, h) && previousTiles != layout.TileCount && gpu.AsSpan().SequenceEqual(previous))
                    failures.Add($"{tag}: the temporal unit equals the one coded with {previousTiles} tiles");
                previous = gpu; previousSize = (w, h); previousTiles = layout.TileCount;
            }
            Console.WriteLine($"{acc.AcceleratorType}: AV1 keyframes with tiles at q{baseQIndex}: " + string.Join(" | ", report));
            True(failures.Count == 0, string.Join("\n", failures));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod(Timeout = 1800000, Category = "NotOnWebGL,DesktopOnly")]
    public async Task Av1KeyframeEncoderGpu_Tiles_FfmpegDecodesWhatTheEncoderReconstructs()
    {
        FfmpegOracle.RequireAvailable();
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var enc = new Av1KeyframeEncoderGpu(acc);
            var failures = new List<string>();
            var report = new List<string>();
            foreach (var (w, h, tileSize, _, _) in Av1TileCases)
                foreach (int q in new[] { 24, 120 })
                {
                    if (w * h > 100000 && q != 24) continue;   // the large picture once per tile size
                    string tag = $"{Av1TileTag(w, h, tileSize)} q{q}";
                    var (y, u, v) = Av1EdgeTestPicture(w, h, 600 + w + q);
                    enc.TileSizeSuperblocks = tileSize;
                    byte[] unit = await enc.EncodeKeyFrameAsync(y, u, v, w, h, q);
                    byte[] recon = await enc.ReadReconAsync();
                    List<byte[]> decoded;
                    try { decoded = FfmpegOracle.DecodeVideoI420(unit, ".obu", w, h); }
                    catch (Exception e) { failures.Add($"{tag}: ffmpeg could not decode the {unit.Length}-byte temporal unit: {e.Message}"); continue; }
                    if (decoded.Count != 1) { failures.Add($"{tag}: ffmpeg decoded {decoded.Count} frames, expected 1"); continue; }
                    int firstDiff = -1, count = 0;
                    for (int i = 0; i < Math.Min(recon.Length, decoded[0].Length); i++)
                        if (recon[i] != decoded[0][i]) { if (firstDiff < 0) firstDiff = i; count++; }
                    if (firstDiff >= 0 || recon.Length != decoded[0].Length)
                        failures.Add($"{tag}: ffmpeg != encoder reconstruction, first differing byte {firstDiff} of {recon.Length}, " +
                            $"{count} bytes differ (lengths {decoded[0].Length} / {recon.Length})");
                    long sad = 0;
                    for (int i = 0; i < w * h; i++) sad += Math.Abs(recon[i] - y[i]);
                    double mae = (double)sad / (w * h);
                    if (mae > (q == 24 ? 6.0 : 24.0)) failures.Add($"{tag}: reconstruction is {mae:F2} grey levels from the source on average");
                    report.Add($"{tag}: {unit.Length} B, MAE {mae:F2}");
                }
            Console.WriteLine($"{acc.AcceleratorType}: AV1 tiled keyframes decoded by ffmpeg: " + string.Join(" | ", report));
            True(failures.Count == 0, string.Join("\n", failures));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    /// <summary>
    /// The tile limit. A frame gets at most <c>MaxTiles</c> tiles, 256 by default, because the hardware AV1 decoders
    /// measured (NVDEC, D3D11VA, D3D12VA on an RTX 4070) fail on more. At 1088x1024 (272 superblocks): the default
    /// layout is 9 x 16 = 144 tiles, the limit raised gives 17 x 16 = 272; libdav1d decodes both to the encoder's
    /// reconstruction. Every hardware decoder FFmpeg has here (one that decodes the single-tile stream of the same
    /// picture correctly) must decode the DEFAULT stream to the same bytes; what it does with 272 tiles is reported.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "NotOnWebGL,DesktopOnly")]
    public async Task Av1KeyframeEncoderGpu_Tiles_DefaultLayoutDecodesOnHardwareDecoders()
    {
        FfmpegOracle.RequireAvailable();
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            const int w = 1088, h = 1024, q = 40;
            using var enc = new Av1KeyframeEncoderGpu(acc);
            Equal(Av1TileLayout.HardwareDecoderMaxTiles, enc.MaxTiles, "default MaxTiles");
            bool threw = false;
            try { enc.MaxTiles = 0; } catch (ArgumentOutOfRangeException) { threw = true; }
            True(threw, "MaxTiles = 0 must throw ArgumentOutOfRangeException");
            var (y, u, v) = Av1EdgeTestPicture(w, h, 950);

            async Task<(byte[] unit, byte[] recon, int tiles)> EncodeAsync(int tileSize, int maxTiles)
            {
                enc.TileSizeSuperblocks = tileSize;
                enc.MaxTiles = maxTiles;
                byte[] unit = await enc.EncodeKeyFrameAsync(y, u, v, w, h, q);
                return (unit, await enc.ReadReconAsync(), enc.GetTileLayout(w, h).TileCount);
            }
            var limited = await EncodeAsync(1, Av1TileLayout.HardwareDecoderMaxTiles);
            var unlimited = await EncodeAsync(1, int.MaxValue);
            var oneTile = await EncodeAsync(int.MaxValue, Av1TileLayout.HardwareDecoderMaxTiles);
            Equal(144, limited.tiles, "tiles of the default layout at 1088x1024");
            Equal(272, unlimited.tiles, "tiles with the limit raised");
            Equal(1, oneTile.tiles, "tiles of the single-tile layout");

            var failures = new List<string>();
            void Check(string what, string[] options, (byte[] unit, byte[] recon, int tiles) s, bool report)
            {
                try
                {
                    var frames = options.Length == 0
                        ? FfmpegOracle.DecodeVideoI420(s.unit, ".obu", w, h)
                        : FfmpegOracle.DecodeVideoI420(s.unit, ".obu", w, h, options, report ? 15_000 : 60_000);
                    if (frames.Count != 1 || !frames[0].AsSpan().SequenceEqual(s.recon))
                        throw new Exception($"{frames.Count} frame(s), not the encoder's reconstruction");
                    if (report) Console.WriteLine($"[report] {what}: {s.tiles} tiles decode to the encoder's reconstruction");
                }
                catch (Exception e)
                {
                    string text = e.Message.Length > 160 ? e.Message.Substring(0, 160) : e.Message;
                    if (report) Console.WriteLine($"[report] {what}: {s.tiles} tiles do NOT decode: {text.Replace('\n', ' ').Replace('\r', ' ')}");
                    else failures.Add($"{what}: {s.tiles} tiles: {text}");
                }
            }
            Check("libdav1d", System.Array.Empty<string>(), limited, false);
            Check("libdav1d", System.Array.Empty<string>(), unlimited, false);
            Check("libaom", new[] { "-c:v", "libaom-av1" }, limited, false);

            var hardware = new (string name, string[] options)[]
            {
                ("NVDEC (av1_cuvid)", new[] { "-c:v", "av1_cuvid" }),
                ("D3D11VA", new[] { "-hwaccel", "d3d11va", "-c:v", "av1" }),
                ("D3D12VA", new[] { "-hwaccel", "d3d12va", "-c:v", "av1" }),
            };
            int present = 0;
            foreach (var (name, options) in hardware)
            {
                // The decoder exists here when it decodes the single-tile stream of this picture correctly.
                bool available;
                try
                {
                    var frames = FfmpegOracle.DecodeVideoI420(oneTile.unit, ".obu", w, h, options, 60_000);
                    available = frames.Count == 1 && frames[0].AsSpan().SequenceEqual(oneTile.recon);
                }
                catch { available = false; }
                if (!available) { Console.WriteLine($"[report] {name}: not available here"); continue; }
                present++;
                Check(name, options, limited, false);
                Check(name, options, unlimited, true);
            }
            Console.WriteLine($"[report] {acc.AcceleratorType}: {present} of {hardware.Length} hardware AV1 decoders present; default layout {limited.tiles} tiles " +
                $"({limited.unit.Length} B), unlimited {unlimited.tiles} tiles ({unlimited.unit.Length} B), one tile {oneTile.unit.Length} B");
            True(failures.Count == 0, string.Join("\n", failures));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    /// <summary>Device-resident 832x480 test frames (the same picture <paramref name="frames"/> times, frame-major).</summary>
    private static (MemoryBuffer1D<byte, Stride1D.Dense> y, MemoryBuffer1D<byte, Stride1D.Dense> u, MemoryBuffer1D<byte, Stride1D.Dense> v)
        Av1SpeedTestFrames(Accelerator acc, int w, int h, int frames)
    {
        var (y, u, v) = Av1EdgeTestPicture(w, h, 900);
        var dY = acc.Allocate1D<byte>((long)frames * y.Length);
        var dU = acc.Allocate1D<byte>((long)frames * u.Length);
        var dV = acc.Allocate1D<byte>((long)frames * v.Length);
        for (int f = 0; f < frames; f++)
        {
            DeviceTransfer.Upload(dY.View.SubView((long)f * y.Length, y.Length), y);
            DeviceTransfer.Upload(dU.View.SubView((long)f * u.Length, u.Length), u);
            DeviceTransfer.Upload(dV.View.SubView((long)f * v.Length, v.Length), v);
        }
        return (dY, dU, dV);
    }

    /// <summary>
    /// The reason tiles exist: on a GPU a keyframe coded as one-superblock tiles must take a fraction of the time of
    /// the same keyframe coded as one tile (one thread walking the whole frame was 1.3 frames per second at 832x480).
    /// Planes are on the device; an encode ends with the length readback. Asserted on the GPU backends (measured
    /// 2026-11-13, RTX 4070: CUDA 91x, OpenCL 88x; the bound is 4x); the CPU accelerators only report. The
    /// "[report]" line carries the times to the run log on every lane.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "NotOnWebGL")]
    public async Task Av1KeyframeEncoderGpu_Tiles_AreFasterThanOneTile()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            const int w = 832, h = 480, q = 40, batch = 8;
            bool gpu = acc.AcceleratorType is AcceleratorType.Cuda or AcceleratorType.OpenCL or AcceleratorType.WebGPU;
            int runs = gpu ? 3 : 1;
            using var enc = new Av1KeyframeEncoderGpu(acc);
            var (dY, dU, dV) = Av1SpeedTestFrames(acc, w, h, batch);
            int cap = Av1KeyframeEncoderGpu.GetMaxEncodedSize(w, h);
            using var dOut = acc.Allocate1D<byte>((long)batch * cap);
            using var dLens = acc.Allocate1D<int>(batch);
            try
            {
                // ms per keyframe (the fastest of the runs, after one untimed encode) and the keyframe's length
                async Task<(double ms, int len)> TimeAsync(int tileSize, int frames)
                {
                    enc.TileSizeSuperblocks = tileSize;
                    double best = double.MaxValue;
                    int len = 0;
                    for (int r = 0; r <= runs; r++)
                    {
                        var sw = System.Diagnostics.Stopwatch.StartNew();
                        if (frames == 1)
                            len = await enc.EncodeKeyFrameAsync(dY.View, dU.View, dV.View, w, h, dOut.View, q);
                        else
                        {
                            await enc.EncodeKeyFramesBatchAsync(dY.View, dU.View, dV.View, frames, w, h, dOut.View, dLens.View, q);
                            len = (await dLens.View.CopyToHostAsync())[frames - 1];
                        }
                        sw.Stop();
                        if (r > 0) best = Math.Min(best, sw.Elapsed.TotalMilliseconds / frames);
                    }
                    return (best, len);
                }
                var oneTile = await TimeAsync(int.MaxValue, 1);
                var tiles = await TimeAsync(1, 1);
                var batched = await TimeAsync(1, batch);
                string F(double ms) => ms.ToString("F2", System.Globalization.CultureInfo.InvariantCulture);
                Console.WriteLine($"[report] {acc.AcceleratorType} AV1 keyframe {w}x{h} q{q}, ms per keyframe: one tile {F(oneTile.ms)} ({oneTile.len} B), " +
                    $"104 tiles {F(tiles.ms)} ({tiles.len} B), 104 tiles in a batch of {batch}: {F(batched.ms)}; {F(oneTile.ms / tiles.ms)}x");

                Equal(tiles.len, batched.len, "a frame of the batch has the length of the single-frame encode");
                True(tiles.len > oneTile.len, $"104 tiles ({tiles.len} B) must cost more bytes than one tile ({oneTile.len} B): are the tiles coded?");
                True(tiles.len <= oneTile.len * 1.02, $"104 tiles cost {tiles.len} B against {oneTile.len} B for one tile: more than 2%");
                if (gpu)
                    True(tiles.ms * 4 < oneTile.ms,
                        $"{acc.AcceleratorType}: 104 tiles took {F(tiles.ms)} ms per keyframe, one tile {F(oneTile.ms)} ms: less than 4x faster");
            }
            finally { dY.Dispose(); dU.Dispose(); dV.Dispose(); }
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task Av1KeyframeEncoderGpu_Tiles_BatchEqualsSingleFrameEncodes()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var enc = new Av1KeyframeEncoderGpu(acc);
            const int frameCount = 3, width = 130, height = 66, baseQIndex = 40;
            foreach (int tileSize in new[] { 1, 2 })
            {
                enc.TileSizeSuperblocks = tileSize;
                var ys = new byte[frameCount][]; var us = new byte[frameCount][]; var vs = new byte[frameCount][];
                var single = new byte[frameCount][];
                for (int f = 0; f < frameCount; f++)
                {
                    (ys[f], us[f], vs[f]) = Av1EdgeTestPicture(width, height, 700 + 10 * tileSize + f);
                    single[f] = await enc.EncodeKeyFrameAsync(ys[f], us[f], vs[f], width, height, baseQIndex);
                    byte[] cpu = Av1KeyframeEncoder.EncodeKeyFrame(ys[f], width, us[f], (width + 1) / 2, vs[f], width, height, baseQIndex, tileSize);
                    EqualBytes(cpu, single[f], $"tile size {tileSize}, frame {f}: single-frame GPU encode vs CPU reference");
                }
                True(!single[0].AsSpan().SequenceEqual(single[1]), "the test frames must differ, or a frame mix-up would pass");
                var batch = await enc.EncodeKeyFramesBatchAsync(
                    System.Array.ConvertAll(ys, p => (ReadOnlyMemory<byte>)p),
                    System.Array.ConvertAll(us, p => (ReadOnlyMemory<byte>)p),
                    System.Array.ConvertAll(vs, p => (ReadOnlyMemory<byte>)p),
                    width, height, baseQIndex);
                Equal(frameCount, batch.Length, "batch frame count");
                for (int f = 0; f < frameCount; f++)
                    EqualBytes(single[f], batch[f], $"tile size {tileSize}, frame {f}: batch vs single-frame encode at {width}x{height}");
            }
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }
}
