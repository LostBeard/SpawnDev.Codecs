// Cross-backend tests for Vp9FrameAssembleKernel. Verifies the
// concatenation produces the same byte sequence the CPU encoder
// emits via Buffer.BlockCopy.

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
    public async Task Vp9FrameAssembleKernel_ThreeStreams_MatchesCpuConcat()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            var rng = new Random(unchecked((int)0xA55E11B7u));
            var u = new byte[20];
            var c = new byte[14];
            var t = new byte[97];
            rng.NextBytes(u);
            rng.NextBytes(c);
            rng.NextBytes(t);

            // CPU oracle: simple byte-array concat.
            var expected = new byte[u.Length + c.Length + t.Length];
            Buffer.BlockCopy(u, 0, expected, 0, u.Length);
            Buffer.BlockCopy(c, 0, expected, u.Length, c.Length);
            Buffer.BlockCopy(t, 0, expected, u.Length + c.Length, t.Length);

            using var kernel = new Vp9FrameAssembleKernel(acc);
            using var dU = acc.Allocate1D<byte>(u.Length);
            using var dC = acc.Allocate1D<byte>(c.Length);
            using var dT = acc.Allocate1D<byte>(t.Length);
            using var dOut = acc.Allocate1D<byte>(expected.Length);
            using var dOutLen = acc.Allocate1D<long>(1);
            dU.View.CopyFromCPU(u);
            dC.View.CopyFromCPU(c);
            dT.View.CopyFromCPU(t);
            dOut.View.CopyFromCPU(new byte[expected.Length]);

            kernel.Run(dU.View, dC.View, dT.View, dOut.View, dOutLen.View,
                       u.Length, c.Length, t.Length);
            await acc.SynchronizeAsync();

            long outLen = (await dOutLen.CopyToHostAsync())[0];
            var actual = await dOut.CopyToHostAsync();

            Equal((long)expected.Length, outLen);
            for (int i = 0; i < expected.Length; i++)
            {
                if (expected[i] != actual[i])
                    throw new Exception(
                        $"byte mismatch at offset {i}: expected=0x{expected[i]:X2} actual=0x{actual[i]:X2}");
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9FrameAssembleKernel_EmptyTile_StillProducesHeaderRun()
    {
        // Edge case: tile bytes of length 0. Encoder shouldn't produce
        // this in practice (tile data carries at least the EOB token
        // emit chain), but the assembler must handle the boundary
        // condition cleanly.
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            var u = new byte[] { 0xDE, 0xAD, 0xBE, 0xEF };
            var c = new byte[] { 0xCA, 0xFE };
            var t = Array.Empty<byte>();
            var expected = new byte[] { 0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE };

            using var kernel = new Vp9FrameAssembleKernel(acc);
            using var dU = acc.Allocate1D<byte>(u.Length);
            using var dC = acc.Allocate1D<byte>(c.Length);
            // ILGPU rejects zero-length allocations, so allocate 1 byte
            // when the tile is empty (the kernel reads 0 bytes from it).
            using var dT = acc.Allocate1D<byte>(Math.Max(1, t.Length));
            using var dOut = acc.Allocate1D<byte>(expected.Length);
            using var dOutLen = acc.Allocate1D<long>(1);
            dU.View.CopyFromCPU(u);
            dC.View.CopyFromCPU(c);
            dT.View.CopyFromCPU(new byte[1]);
            dOut.View.CopyFromCPU(new byte[expected.Length]);

            kernel.Run(dU.View, dC.View, dT.View, dOut.View, dOutLen.View,
                       u.Length, c.Length, t.Length);
            await acc.SynchronizeAsync();

            long outLen = (await dOutLen.CopyToHostAsync())[0];
            var actual = await dOut.CopyToHostAsync();
            Equal((long)expected.Length, outLen);
            for (int i = 0; i < expected.Length; i++) Equal(expected[i], actual[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
