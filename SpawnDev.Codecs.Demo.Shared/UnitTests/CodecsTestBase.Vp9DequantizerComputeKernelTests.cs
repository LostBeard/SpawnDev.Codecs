// Cross-backend tests for Vp9DequantizerComputeKernel. Validates byte-for-byte
// parity against Vp9Dequantizer.PlaneQuantizer for both Y and UV planes
// across the full baseQIndex range and a sweep of typical delta combinations.
//
// This kernel is the foundational primitive for the future
// Vp9KeyframeEncoderGpu / Vp9KeyframeDecoderGpu integration classes - it
// computes the 4 plane dequantizers (Y_DC, Y_AC, UV_DC, UV_AC) from
// baseQIndex + per-plane delta values + the 256-entry DC/AC lookup tables.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public async Task Vp9DequantizerComputeKernel_ZeroDeltas_MatchesCpuReference()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Vp9DequantizerComputeKernel(acc);
            using var dDcLookup = acc.Allocate1D<short>(256);
            using var dAcLookup = acc.Allocate1D<short>(256);
            using var dOut = acc.Allocate1D<int>(4);
            dDcLookup.View.CopyFromCPU(Vp9DequantizerComputeKernel.BuildDcQLookup());
            dAcLookup.View.CopyFromCPU(Vp9DequantizerComputeKernel.BuildAcQLookup());

            // Sweep across the full quantizer index range with no deltas.
            // This is the v1 / typical encoder configuration: per-frame
            // baseQIndex with no per-plane offsets.
            int[] baseQIndices = { 0, 1, 16, 32, 64, 96, 128, 160, 192, 224, 254, 255 };
            foreach (var qi in baseQIndices)
            {
                kernel.Run(dDcLookup.View, dAcLookup.View, dOut.View, qi, 0, 0, 0, 0);
                await acc.SynchronizeAsync();
                var gpu = await dOut.CopyToHostAsync();

                var yPlane = Vp9Dequantizer.PlaneQuantizer(qi, 0, 0);
                var uvPlane = Vp9Dequantizer.PlaneQuantizer(qi, 0, 0);
                Equal((int)yPlane.Dc, gpu[0]);
                Equal((int)yPlane.Ac, gpu[1]);
                Equal((int)uvPlane.Dc, gpu[2]);
                Equal((int)uvPlane.Ac, gpu[3]);
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9DequantizerComputeKernel_AsymmetricYUVDeltas_MatchesCpuReference()
    {
        // VP9 allows independent y_dc_delta_q, uv_dc_delta_q, uv_ac_delta_q
        // (and an implicit y_ac_delta_q of 0 - VP9 ties the frame baseQ to
        // y_ac directly per spec sec 6.2.4). Test a sweep of asymmetric
        // configurations that match real-world encoder choices.
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Vp9DequantizerComputeKernel(acc);
            using var dDcLookup = acc.Allocate1D<short>(256);
            using var dAcLookup = acc.Allocate1D<short>(256);
            using var dOut = acc.Allocate1D<int>(4);
            dDcLookup.View.CopyFromCPU(Vp9DequantizerComputeKernel.BuildDcQLookup());
            dAcLookup.View.CopyFromCPU(Vp9DequantizerComputeKernel.BuildAcQLookup());

            // Each row: { baseQ, yDc, yAc, uvDc, uvAc }
            // Range is [-15, 15] per VP9 spec sec 6.2.4 delta encoding (4-bit
            // signed magnitude); we use values inside that interval.
            int[][] cases =
            {
                new[] { 64, 0, 0, 0, 0 },
                new[] { 64, 4, 0, -4, 0 },
                new[] { 64, -4, 0, 4, 0 },
                new[] { 100, 8, -3, 2, 5 },
                new[] { 100, -8, 3, -2, -5 },
                new[] { 32, 15, 0, 15, 0 },     // max positive deltas
                new[] { 32, -15, 0, -15, 0 },   // max negative deltas
                new[] { 200, 0, 0, 8, 4 },
                new[] { 16, -3, 0, -7, -2 },
            };

            foreach (var c in cases)
            {
                int qi = c[0]; int yDc = c[1]; int yAc = c[2]; int uvDc = c[3]; int uvAc = c[4];

                kernel.Run(dDcLookup.View, dAcLookup.View, dOut.View, qi, yDc, yAc, uvDc, uvAc);
                await acc.SynchronizeAsync();
                var gpu = await dOut.CopyToHostAsync();

                var yPlane = Vp9Dequantizer.PlaneQuantizer(qi, yDc, yAc);
                var uvPlane = Vp9Dequantizer.PlaneQuantizer(qi, uvDc, uvAc);
                Equal((int)yPlane.Dc, gpu[0]);
                Equal((int)yPlane.Ac, gpu[1]);
                Equal((int)uvPlane.Dc, gpu[2]);
                Equal((int)uvPlane.Ac, gpu[3]);
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9DequantizerComputeKernel_ClampsAtTableBoundaries()
    {
        // Indices outside [0, 255] must clamp identically to libvpx
        // vp9_dc_quant / vp9_ac_quant. Push past both boundaries to verify
        // the GPU clamp matches the CPU oracle.
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Vp9DequantizerComputeKernel(acc);
            using var dDcLookup = acc.Allocate1D<short>(256);
            using var dAcLookup = acc.Allocate1D<short>(256);
            using var dOut = acc.Allocate1D<int>(4);
            dDcLookup.View.CopyFromCPU(Vp9DequantizerComputeKernel.BuildDcQLookup());
            dAcLookup.View.CopyFromCPU(Vp9DequantizerComputeKernel.BuildAcQLookup());

            // baseQ=0 with very negative deltas -> clamp at 0
            kernel.Run(dDcLookup.View, dAcLookup.View, dOut.View, 0, -100, -100, -100, -100);
            await acc.SynchronizeAsync();
            var gpu = await dOut.CopyToHostAsync();
            Equal((int)Vp9Dequantizer.DcQLookup8[0], gpu[0]);
            Equal((int)Vp9Dequantizer.AcQLookup8[0], gpu[1]);
            Equal((int)Vp9Dequantizer.DcQLookup8[0], gpu[2]);
            Equal((int)Vp9Dequantizer.AcQLookup8[0], gpu[3]);

            // baseQ=255 with very positive deltas -> clamp at 255
            kernel.Run(dDcLookup.View, dAcLookup.View, dOut.View, 255, 100, 100, 100, 100);
            await acc.SynchronizeAsync();
            gpu = await dOut.CopyToHostAsync();
            Equal((int)Vp9Dequantizer.DcQLookup8[255], gpu[0]);
            Equal((int)Vp9Dequantizer.AcQLookup8[255], gpu[1]);
            Equal((int)Vp9Dequantizer.DcQLookup8[255], gpu[2]);
            Equal((int)Vp9Dequantizer.AcQLookup8[255], gpu[3]);

            // Exactly on the boundary - 255+0=255 should NOT clamp (it's
            // already the last in-range index).
            kernel.Run(dDcLookup.View, dAcLookup.View, dOut.View, 255, 0, 0, 0, 0);
            await acc.SynchronizeAsync();
            gpu = await dOut.CopyToHostAsync();
            Equal((int)Vp9Dequantizer.DcQLookup8[255], gpu[0]);
            Equal((int)Vp9Dequantizer.AcQLookup8[255], gpu[1]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9DequantizerComputeKernel_RandomSweep_MatchesCpuReference()
    {
        // Stress sweep: 256 random (baseQ, deltas) tuples across the legal
        // VP9 parameter space. Catches any off-by-one / sign / clamp drift
        // that hand-picked cases miss.
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Vp9DequantizerComputeKernel(acc);
            using var dDcLookup = acc.Allocate1D<short>(256);
            using var dAcLookup = acc.Allocate1D<short>(256);
            using var dOut = acc.Allocate1D<int>(4);
            dDcLookup.View.CopyFromCPU(Vp9DequantizerComputeKernel.BuildDcQLookup());
            dAcLookup.View.CopyFromCPU(Vp9DequantizerComputeKernel.BuildAcQLookup());

            var rng = new Random(unchecked((int)0xDE9117EAu));
            for (int trial = 0; trial < 256; trial++)
            {
                int qi = rng.Next(0, 256);
                int yDc = rng.Next(-15, 16);
                int yAc = rng.Next(-15, 16);
                int uvDc = rng.Next(-15, 16);
                int uvAc = rng.Next(-15, 16);

                kernel.Run(dDcLookup.View, dAcLookup.View, dOut.View, qi, yDc, yAc, uvDc, uvAc);
                await acc.SynchronizeAsync();
                var gpu = await dOut.CopyToHostAsync();

                var yPlane = Vp9Dequantizer.PlaneQuantizer(qi, yDc, yAc);
                var uvPlane = Vp9Dequantizer.PlaneQuantizer(qi, uvDc, uvAc);
                Equal((int)yPlane.Dc, gpu[0]);
                Equal((int)yPlane.Ac, gpu[1]);
                Equal((int)uvPlane.Dc, gpu[2]);
                Equal((int)uvPlane.Ac, gpu[3]);
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
