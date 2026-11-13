// Av1KeyframeEncoderGpu's transform pass: parallel inside a tile.
//
// The tile walk is two passes (Av1FrameSequentialEncodeKernel): a transform pass in which the 16x16 positions of a
// tile are taken in anti-diagonal wavefront order (a position predicts from the reconstruction above it and to its
// left), and an entropy pass, one thread per tile, that codes the stored coefficients. On the GPUs the transform
// pass is an explicitly grouped launch with a barrier between diagonals, and its shape (working lanes per tile, group
// threads per lane, tiles per group) is chosen per launch; the CPU accelerators walk the same diagonals with one
// thread per tile.
//
// Every other encoder test runs through the shape the accelerator picks for its picture. These two cover the rest:
//  - Wavefront_EveryLaunchShapeGivesTheReferenceBytes: the same pictures under forced shapes (one lane, fewer lanes
//    than the widest diagonal = several rounds per diagonal, a lane count that is not a power of two, lanes spread
//    apart, several tiles per group with a last group that is not full), each equal to the CPU reference byte for
//    byte. A wrong lane-to-position mapping skips or repeats positions; a missing barrier lets a position read a
//    neighbour that is not reconstructed yet.
//  - Wavefront_ManyTilesBatchEqualsSingleFrameEncodes: a batch with more tiles than 32 per multiprocessor, where
//    the NVIDIA GPUs switch to one lane per tile in groups of 40 tiles; the choice itself is asserted there.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Av1;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    /// <summary>True on the accelerators whose transform pass is the grouped wavefront launch.</summary>
    private static bool Av1HasWavefrontLaunch(Accelerator acc) =>
        acc.AcceleratorType is AcceleratorType.Cuda or AcceleratorType.OpenCL or AcceleratorType.WebGPU;

    [TestMethod(Timeout = 1800000, Category = "NotOnWebGL")]
    public async Task Av1KeyframeEncoderGpu_Wavefront_EveryLaunchShapeGivesTheReferenceBytes()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            // (lanes per tile, group threads per lane, tiles per group); 0 = the accelerator's own choice
            var shapes = new (int lanes, int spread, int tiles)[]
            {
                (0, 0, 0), (1, 1, 1), (2, 1, 1), (3, 1, 1), (4, 1, 1), (8, 1, 1), (1, 2, 3), (2, 4, 2), (4, 1, 8), (1, 16, 40),
                // the lanes of a tile a warp (32 threads) or two apart: lanes in one warp advance together, so only
                // these show a missing barrier on a GPU (a lane with no position on a diagonal runs ahead and reads
                // a neighbour that is still being reconstructed)
                (2, 32, 1), (4, 32, 1), (4, 64, 1),
            };
            // 130x66: 3 x 2 one-superblock tiles (edge leaves on both sides), and one tile of 9 x 5 positions (13
            // diagonals: two launches; 5 positions on the widest); 320x200 tile size 3: 2 x 2 tiles, the last column
            // narrower; 200x9: one row of 8x8 leaves
            var cases = new (int w, int h, int tileSize)[] { (130, 66, 1), (130, 66, int.MaxValue), (320, 200, 3), (200, 9, 2) };
            const int q = 44;
            using var enc = new Av1KeyframeEncoderGpu(acc);
            var failures = new List<string>();
            int runs = 0;
            foreach (var (w, h, tileSize) in cases)
            {
                var (y, u, v) = Av1EdgeTestPicture(w, h, 900 + w + (tileSize == int.MaxValue ? 7 : tileSize));
                // Another picture of the same size, encoded before every checked encode: the encoder keeps its
                // reconstruction buffer, so a position that read a neighbour too early would otherwise find the
                // right pixels left over from the previous encode of the same picture.
                var (y2, u2, v2) = Av1EdgeTestPicture(w, h, 400 + w + (tileSize == int.MaxValue ? 7 : tileSize));
                byte[] cpu = Av1KeyframeEncoder.EncodeKeyFrame(y, w, u, (w + 1) / 2, v, w, h, q, tileSize);
                byte[] cpu2 = Av1KeyframeEncoder.EncodeKeyFrame(y2, w, u2, (w + 1) / 2, v2, w, h, q, tileSize);
                True(!cpu.AsSpan().SequenceEqual(cpu2), "the two pictures of a size must differ");
                enc.TileSizeSuperblocks = tileSize;
                foreach (var (lanes, spread, tiles) in shapes)
                {
                    if (lanes > 0 && (!Av1HasWavefrontLaunch(acc) || (long)lanes * spread * tiles > acc.MaxNumThreadsPerGroup)) continue;
                    Av1FrameSequentialEncodeKernel.TransformGroupOverride = default;
                    await enc.EncodeKeyFrameAsync(y2, u2, v2, w, h, q);
                    Av1FrameSequentialEncodeKernel.TransformGroupOverride = (lanes, spread, tiles);
                    byte[] gpu = await enc.EncodeKeyFrameAsync(y, u, v, w, h, q);
                    runs++;
                    if (!gpu.AsSpan().SequenceEqual(cpu))
                    {
                        int first = 0, lim = Math.Min(gpu.Length, cpu.Length);
                        while (first < lim && gpu[first] == cpu[first]) first++;
                        failures.Add($"{Av1TileTag(w, h, tileSize)}, lanes {lanes} spread {spread} tiles per group {tiles}: " +
                            $"{gpu.Length} bytes vs CPU reference {cpu.Length}, first difference at byte {first}");
                    }
                }
            }
            // a batch under a shape with several tiles per group: 3 frames x 6 tiles = 18 tiles in groups of 4
            if (Av1HasWavefrontLaunch(acc))
            {
                const int w = 130, h = 66, frames = 3;
                var ys = new byte[frames][]; var us = new byte[frames][]; var vs = new byte[frames][];
                var reference = new byte[frames][];
                for (int f = 0; f < frames; f++)
                {
                    (ys[f], us[f], vs[f]) = Av1EdgeTestPicture(w, h, 950 + f);
                    reference[f] = Av1KeyframeEncoder.EncodeKeyFrame(ys[f], w, us[f], (w + 1) / 2, vs[f], w, h, q, 1);
                }
                enc.TileSizeSuperblocks = 1;
                Av1FrameSequentialEncodeKernel.TransformGroupOverride = (2, 2, 4);
                var batch = await enc.EncodeKeyFramesBatchAsync(
                    System.Array.ConvertAll(ys, p => (ReadOnlyMemory<byte>)p),
                    System.Array.ConvertAll(us, p => (ReadOnlyMemory<byte>)p),
                    System.Array.ConvertAll(vs, p => (ReadOnlyMemory<byte>)p),
                    w, h, q);
                runs++;
                for (int f = 0; f < frames; f++)
                    if (!batch[f].AsSpan().SequenceEqual(reference[f]))
                        failures.Add($"batch of {frames} at {w}x{h}, lanes 2 spread 2 tiles per group 4: frame {f} differs from the CPU reference");
            }
            Console.WriteLine($"{acc.AcceleratorType}: AV1 transform launch shapes: {runs} encodes equal to the CPU reference" +
                (Av1HasWavefrontLaunch(acc) ? "" : " (one thread per tile on this accelerator: the shapes do not apply)"));
            True(runs >= cases.Length, $"only {runs} encodes ran");
            True(failures.Count == 0, string.Join("\n", failures));
        }
        finally
        {
            Av1FrameSequentialEncodeKernel.TransformGroupOverride = default;
            ReleaseAccelerator(ctx, acc);
        }
    }

    /// <summary>
    /// The shape the kernel picks, and a batch large enough for the many-tiles shape: 1500 frames of 64x64 (one
    /// tile each; an RTX 4070 has 46 multiprocessors, the switch is above 32 tiles per multiprocessor). Three
    /// pictures in rotation; every frame of the batch equals the single-frame encode of its picture.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "NotOnWebGL,DesktopOnly")]
    public async Task Av1KeyframeEncoderGpu_Wavefront_ManyTilesBatchEqualsSingleFrameEncodes()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            bool nvidia = acc.AcceleratorType == AcceleratorType.Cuda
                || (acc.AcceleratorType == AcceleratorType.OpenCL
                    && acc.Device is global::ILGPU.Runtime.OpenCL.CLDevice cl && cl.Vendor == global::ILGPU.Runtime.OpenCL.CLDeviceVendor.Nvidia);
            const int w = 64, h = 64, q = 60, pictures = 3;
            int frames = nvidia ? 32 * acc.NumMultiprocessors + 28 : 12;
            using (var kernel = new Av1FrameSequentialEncodeKernel(acc))
            {
                // few tiles: a group per tile, a lane per position of the widest diagonal (next power of two)
                Equal("(4, 1, 1)", kernel.GetTransformGroup(4, 104).ToString(), "104 tiles of 4 x 4 positions");
                Equal("(8, 1, 1)", kernel.GetTransformGroup(5, 1).ToString(), "one tile, 5 positions on the widest diagonal");
                Equal("(1, 1, 1)", kernel.GetTransformGroup(1, 4).ToString(), "tiles one position wide");
                if (nvidia)
                {
                    int warp = acc.WarpSize;
                    Equal($"(1, {warp / 2}, 40)", kernel.GetTransformGroup(4, frames).ToString(), $"{frames} tiles on {acc.NumMultiprocessors} multiprocessors");
                    Equal("(4, 1, 1)", kernel.GetTransformGroup(4, 32L * acc.NumMultiprocessors).ToString(), "exactly 32 tiles per multiprocessor: still the wavefront lanes");
                }
            }

            using var enc = new Av1KeyframeEncoderGpu(acc);
            var ys = new byte[pictures][]; var us = new byte[pictures][]; var vs = new byte[pictures][];
            var single = new byte[pictures][];
            for (int k = 0; k < pictures; k++)
            {
                (ys[k], us[k], vs[k]) = Av1EdgeTestPicture(w, h, 970 + k);
                single[k] = await enc.EncodeKeyFrameAsync(ys[k], us[k], vs[k], w, h, q);
                byte[] cpu = Av1KeyframeEncoder.EncodeKeyFrame(ys[k], w, us[k], (w + 1) / 2, vs[k], w, h, q);
                EqualBytes(cpu, single[k], $"picture {k}: single-frame GPU encode vs CPU reference");
            }
            True(!single[0].AsSpan().SequenceEqual(single[1]) && !single[1].AsSpan().SequenceEqual(single[2]), "the pictures must differ, or a frame mix-up would pass");
            var by = new ReadOnlyMemory<byte>[frames]; var bu = new ReadOnlyMemory<byte>[frames]; var bv = new ReadOnlyMemory<byte>[frames];
            for (int f = 0; f < frames; f++) { by[f] = ys[f % pictures]; bu[f] = us[f % pictures]; bv[f] = vs[f % pictures]; }
            var batch = await enc.EncodeKeyFramesBatchAsync(by, bu, bv, w, h, q);
            Equal(frames, batch.Length, "batch frame count");
            int wrong = 0, firstWrong = -1;
            for (int f = 0; f < frames; f++)
                if (!batch[f].AsSpan().SequenceEqual(single[f % pictures])) { wrong++; if (firstWrong < 0) firstWrong = f; }
            Console.WriteLine($"{acc.AcceleratorType}: AV1 batch of {frames} one-tile frames ({(nvidia ? "many-tiles shape" : "the accelerator's usual shape")}): {frames - wrong} equal to their single-frame encode");
            True(wrong == 0, $"{wrong} of {frames} frames differ from their single-frame encode, the first is frame {firstWrong}");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    /// <summary>
    /// The symbol pass and the range coder pass alternate over waves of 16x16 slots (WaveSlots per tile): the range
    /// coder of a tile stops after a wave and resumes in the next launch. Forced wave sizes (1, 3, 5 slots: a wave
    /// that ends inside a superblock; 16; 64; and a memory budget so small that a wave is one slot) on tiles of 16
    /// and of 96 slots, single frames and a batch: every temporal unit equals the CPU reference.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "NotOnWebGL")]
    public async Task Av1KeyframeEncoderGpu_SymbolWaves_EveryWaveSizeGivesTheReferenceBytes()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            // (most slots per wave, memory budget in bytes); 0 = the constant of the kernel class
            var waves = new (int maxWave, long budget)[] { (0, 0), (1, 0), (3, 0), (5, 0), (16, 0), (64, 0), (0, 1) };
            // 130x66 as 6 one-superblock tiles (16 slots each) and as one tile of 3 x 2 superblocks (96 slots);
            // 320x200 tile size 3: tiles of 3 x 2 and 2 x 2 superblocks (96 and 64 slots: tiles of different lengths)
            var cases = new (int w, int h, int tileSize)[] { (130, 66, 1), (130, 66, int.MaxValue), (320, 200, 3) };
            const int q = 36;
            using var enc = new Av1KeyframeEncoderGpu(acc);
            var failures = new List<string>();
            int runs = 0;
            foreach (var (w, h, tileSize) in cases)
            {
                var (y, u, v) = Av1EdgeTestPicture(w, h, 1100 + w + (tileSize == int.MaxValue ? 7 : tileSize));
                byte[] cpu = Av1KeyframeEncoder.EncodeKeyFrame(y, w, u, (w + 1) / 2, v, w, h, q, tileSize);
                enc.TileSizeSuperblocks = tileSize;
                foreach (var (maxWave, budget) in waves)
                {
                    Av1FrameSequentialEncodeKernel.WaveOverride = (budget, maxWave);
                    byte[] gpu = await enc.EncodeKeyFrameAsync(y, u, v, w, h, q);
                    runs++;
                    if (!gpu.AsSpan().SequenceEqual(cpu))
                    {
                        int first = 0, lim = Math.Min(gpu.Length, cpu.Length);
                        while (first < lim && gpu[first] == cpu[first]) first++;
                        failures.Add($"{Av1TileTag(w, h, tileSize)}, most slots per wave {maxWave}, budget {budget}: " +
                            $"{gpu.Length} bytes vs CPU reference {cpu.Length}, first difference at byte {first}");
                    }
                }
            }
            {
                const int w = 130, h = 66, frames = 3;
                var ys = new byte[frames][]; var us = new byte[frames][]; var vs = new byte[frames][];
                var reference = new byte[frames][];
                for (int f = 0; f < frames; f++)
                {
                    (ys[f], us[f], vs[f]) = Av1EdgeTestPicture(w, h, 1150 + f);
                    reference[f] = Av1KeyframeEncoder.EncodeKeyFrame(ys[f], w, us[f], (w + 1) / 2, vs[f], w, h, q, int.MaxValue);
                }
                enc.TileSizeSuperblocks = int.MaxValue;
                Av1FrameSequentialEncodeKernel.WaveOverride = (0, 5);
                var batch = await enc.EncodeKeyFramesBatchAsync(
                    System.Array.ConvertAll(ys, p => (ReadOnlyMemory<byte>)p),
                    System.Array.ConvertAll(us, p => (ReadOnlyMemory<byte>)p),
                    System.Array.ConvertAll(vs, p => (ReadOnlyMemory<byte>)p),
                    w, h, q);
                runs++;
                for (int f = 0; f < frames; f++)
                    if (!batch[f].AsSpan().SequenceEqual(reference[f]))
                        failures.Add($"batch of {frames} one-tile frames at {w}x{h}, 5 slots per wave: frame {f} differs from the CPU reference");
            }
            // The wave size itself (it changes memory and launches, never a byte, so the encodes above cannot see it).
            // A slot is 3200 ints + 1384 bytes = 14184 bytes; the budget is 96 MB.
            Av1FrameSequentialEncodeKernel.WaveOverride = default;
            Equal(16, Av1FrameSequentialEncodeKernel.GetWaveSlots(104, 16), "104 tiles of 16 slots: a tile's slots in one wave");
            Equal(64, Av1FrameSequentialEncodeKernel.GetWaveSlots(1, 1664), "one tile of 1664 slots: MaxWaveSlots");
            Equal(64, Av1FrameSequentialEncodeKernel.GetWaveSlots(32, 320), "32 tiles of 320 slots: MaxWaveSlots");
            Equal(4, Av1FrameSequentialEncodeKernel.GetWaveSlots(1664, 16), "1664 tiles: 96 MB / (1664 * 14184 bytes) = 4 slots");
            Equal(1, Av1FrameSequentialEncodeKernel.GetWaveSlots(100000, 16), "more tiles than the budget has slots for: one slot each");
            Av1FrameSequentialEncodeKernel.WaveOverride = (1, 0);
            Equal(1, Av1FrameSequentialEncodeKernel.GetWaveSlots(104, 16), "a budget of one byte: one slot");
            Av1FrameSequentialEncodeKernel.WaveOverride = (0, 5);
            Equal(5, Av1FrameSequentialEncodeKernel.GetWaveSlots(104, 16), "at most 5 slots per wave");
            Console.WriteLine($"{acc.AcceleratorType}: AV1 symbol waves: {runs} encodes equal to the CPU reference");
            True(failures.Count == 0, string.Join("\n", failures));
        }
        finally
        {
            Av1FrameSequentialEncodeKernel.WaveOverride = default;
            ReleaseAccelerator(ctx, acc);
        }
    }

    /// <summary>
    /// Noise at the finest quantizer (q index 1): nearly every coefficient is large, so nearly every coefficient
    /// costs a base level, four range symbols, a sign and a Golomb code: a symbol slot close to the worst case it is
    /// sized for (<c>Av1SymbolStreamGpu.SlotInts</c>), and Golomb codes of a dozen bits. A slot that overflowed
    /// would run into the entries of the next. 16x16 leaves (64x64), 8x8 edge leaves (70x50), several tiles (130x66).
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "NotOnWebGL")]
    public async Task Av1KeyframeEncoderGpu_FinestQuantizerNoise_MatchesCpuReference()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var enc = new Av1KeyframeEncoderGpu(acc);
            var failures = new List<string>();
            var report = new List<string>();
            var rng = new Random(20261113);
            foreach (var (w, h, tileSize) in new[] { (64, 64, 1), (70, 50, 1), (130, 66, 1), (130, 66, int.MaxValue) })
                foreach (int q in new[] { 1, 2 })
                {
                    var (y, u, v) = MakeRandomYuv420(w, h, rng);
                    byte[] cpu = Av1KeyframeEncoder.EncodeKeyFrame(y, w, u, (w + 1) / 2, v, w, h, q, tileSize);
                    enc.TileSizeSuperblocks = tileSize;
                    byte[] gpu = await enc.EncodeKeyFrameAsync(y, u, v, w, h, q);
                    int positions = ((w + 15) / 16) * ((h + 15) / 16);
                    report.Add($"{Av1TileTag(w, h, tileSize)} q{q}: {gpu.Length} B, {gpu.Length / positions} B per 16x16 position");
                    if (!gpu.AsSpan().SequenceEqual(cpu))
                    {
                        int first = 0, lim = Math.Min(gpu.Length, cpu.Length);
                        while (first < lim && gpu[first] == cpu[first]) first++;
                        failures.Add($"{Av1TileTag(w, h, tileSize)} q{q}: {gpu.Length} bytes vs CPU reference {cpu.Length}, first difference at byte {first}");
                    }
                    // the picture is noise and the quantizer the finest: it must cost more than a byte per pixel, or
                    // the test is not near the worst case it is there for
                    if (gpu.Length < w * h) failures.Add($"{Av1TileTag(w, h, tileSize)} q{q}: only {gpu.Length} bytes for {w * h} pixels of noise");
                }
            Console.WriteLine($"{acc.AcceleratorType}: AV1 noise at the finest quantizers: " + string.Join(" | ", report));
            True(failures.Count == 0, string.Join("\n", failures));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    /// <summary>
    /// The symbol hand-over itself (Av1SymbolStreamGpu), without the encoder: a list of operations (CDF symbols of
    /// alphabets of 2 to 16 symbols, runs of 1 to 32 raw bits) coded twice on the accelerator, once directly
    /// (Av1RangeEncoderGpu.EncodeCdfQ15 / EncodeBits) and once as symbol entries that are then fed to the range
    /// coder. The two byte streams must be equal. Runs longer than 15 bits and longer than one entry (24 bits) do
    /// not occur in 8-bit pictures, so only this test reaches them.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "NotOnWebGL")]
    public async Task Av1SymbolStreamGpu_Entries_CodeTheBytesOfTheDirectCalls()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            // inverse CDF rows of 17 values: row r has an alphabet of AlphabetSizes[r] symbols, values falling to 0
            int[] alphabetSizes = { 2, 3, 4, 5, 7, 9, 13, 14, 16 };
            var rng = new Random(77123);
            var icdf = new ushort[alphabetSizes.Length * 17];
            for (int r = 0; r < alphabetSizes.Length; r++)
            {
                int nsyms = alphabetSizes[r];
                var cuts = new int[nsyms - 1];
                for (int k = 0; k < cuts.Length; k++) cuts[k] = 64 + rng.Next(32768 - 128);
                System.Array.Sort(cuts);
                System.Array.Reverse(cuts);
                for (int k = 0; k < nsyms - 1; k++) icdf[r * 17 + k] = (ushort)cuts[k];
                icdf[r * 17 + nsyms - 1] = 0;
            }
            // operations of 4 ints: kind (0 = CDF symbol, 1 = raw bits), a, b, c
            //   CDF: a = symbol, b = row, c = alphabet size;  bits: a = value, b = count
            const int opCount = 900;
            var ops = new int[opCount * 4];
            int runsOver15 = 0, runsOver24 = 0, firstSymbols = 0, lastSymbols = 0;
            for (int i = 0; i < opCount; i++)
            {
                if (i % 3 == 2)
                {
                    int count = i < 96 ? 1 + (i / 3) : 1 + rng.Next(32);   // every count 1 .. 32 at least once
                    ops[i * 4] = 1;
                    ops[i * 4 + 1] = rng.Next() ^ (rng.Next() << 16);
                    ops[i * 4 + 2] = count;
                    if (count > 15) runsOver15++;
                    if (count > 24) runsOver24++;
                }
                else
                {
                    int r = rng.Next(alphabetSizes.Length);
                    int sym = i % 7 == 0 ? 0 : i % 7 == 1 ? alphabetSizes[r] - 1 : rng.Next(alphabetSizes[r]);
                    ops[i * 4] = 0;
                    ops[i * 4 + 1] = sym;
                    ops[i * 4 + 2] = r;
                    ops[i * 4 + 3] = alphabetSizes[r];
                    if (sym == 0) firstSymbols++;
                    if (sym == alphabetSizes[r] - 1) lastSymbols++;
                }
            }
            True(runsOver15 > 20 && runsOver24 > 8 && firstSymbols > 50 && lastSymbols > 50,
                $"the operations must cover long runs and both ends of the alphabets ({runsOver15}, {runsOver24}, {firstSymbols}, {lastSymbols})");

            const int cap = 16384;
            using var dOps = acc.Allocate1D<int>(ops.Length);
            using var dIcdf = acc.Allocate1D<ushort>(icdf.Length);
            using var dSym = acc.Allocate1D<int>(opCount * 2 + 8);
            using var dDirect = acc.Allocate1D<byte>(cap);
            using var dVia = acc.Allocate1D<byte>(cap);
            using var dLens = acc.Allocate1D<int>(4);
            dOps.View.CopyFromCPU(ops);
            dIcdf.View.CopyFromCPU(icdf);
            dDirect.View.MemSetToZero();
            dVia.View.MemSetToZero();
            dLens.View.MemSetToZero();
            var kernel = acc.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<int>, ArrayView<ushort>, ArrayView<int>, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, int>(
                Av1SymbolStreamEqualsDirectKernel);
            kernel(2, dOps.View, dIcdf.View, dSym.View, dDirect.View, dVia.View, dLens.View, opCount);
            await acc.SynchronizeAsync();
            int[] lens = await dLens.CopyToHostAsync();
            byte[] direct = await dDirect.CopyToHostAsync();
            byte[] via = await dVia.CopyToHostAsync();
            Console.WriteLine($"{acc.AcceleratorType}: AV1 symbol entries: {opCount} operations, {lens[2]} entries, {lens[0]} bytes direct, {lens[1]} bytes through the entries");
            True(lens[0] > 600 && lens[0] < cap, $"the direct stream is {lens[0]} bytes");
            Equal(lens[0], lens[1], "bytes through the entries vs direct");
            int first = 0;
            while (first < lens[0] && direct[first] == via[first]) first++;
            True(first == lens[0], $"the stream through the entries differs from the direct one at byte {first} of {lens[0]}");
            // 600 CDF symbols are one entry each; a run of up to 24 bits is one entry, a longer one two
            Equal(600 + 300 + runsOver24, lens[2], "entries written");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    /// <summary>Thread 0 codes the operations directly, thread 1 through symbol entries; lens = bytes direct, bytes through entries, entries.</summary>
    private static void Av1SymbolStreamEqualsDirectKernel(
        Index1D idx, ArrayView<int> ops, ArrayView<ushort> icdf, ArrayView<int> sym,
        ArrayView<byte> direct, ArrayView<byte> via, ArrayView<int> lens, int opCount)
    {
        var re = Av1RangeEncoderGpu.Init();
        if (idx.X == 0)
        {
            for (int i = 0; i < opCount; i++)
            {
                int a = ops[i * 4 + 1], b = ops[i * 4 + 2], c = ops[i * 4 + 3];
                if (ops[i * 4] == 0) Av1RangeEncoderGpu.EncodeCdfQ15(ref re, direct, a, icdf, b * 17, c);
                else Av1RangeEncoderGpu.EncodeBits(ref re, direct, (uint)a, b);
            }
            Av1RangeEncoderGpu.Done(ref re, direct);
            lens[0] = (int)re.OutLen;
        }
        else
        {
            int n = 0;
            for (int i = 0; i < opCount; i++)
            {
                int a = ops[i * 4 + 1], b = ops[i * 4 + 2], c = ops[i * 4 + 3];
                if (ops[i * 4] == 0) Av1SymbolStreamGpu.Cdf(sym, 0, ref n, a, icdf, b * 17, c);
                else Av1SymbolStreamGpu.Bits(sym, 0, ref n, (uint)a, b);
            }
            sym[0] = n;
            for (int i = 1; i <= n; i++) Av1SymbolStreamGpu.Encode(ref re, via, sym[i]);
            Av1RangeEncoderGpu.Done(ref re, via);
            lens[1] = (int)re.OutLen;
            lens[2] = n;
        }
    }
}
