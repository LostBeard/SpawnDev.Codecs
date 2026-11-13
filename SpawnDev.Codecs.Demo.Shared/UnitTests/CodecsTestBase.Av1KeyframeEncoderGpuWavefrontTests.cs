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
}
