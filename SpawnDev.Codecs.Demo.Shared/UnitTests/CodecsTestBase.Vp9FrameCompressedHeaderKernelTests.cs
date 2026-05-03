// Cross-backend tests for Vp9FrameCompressedHeaderKernel. The
// GPU-emitted compressed header must be byte-for-byte identical to
// what the CPU Vp9KeyframeEncoder builds for the v1 keyframe path.
// The compressed header content is fixed (no parameters vary in v1),
// so a single byte-comparison is the entire test surface.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private static byte[] BuildVp9CompressedHeaderCpu()
    {
        // Replicate Vp9KeyframeEncoder.EncodeKeyFrame's compressed header
        // path: tx_mode = Allow32x32 + 4 coef-prob gate bits (no update)
        // + 3 skip-prob no-update diff_update_prob bits.
        var enc = new Vp9BoolEncoder();
        enc.WriteLiteral((int)Vp9TxMode.Allow32x32, 2);
        // Four coef-prob gate bits at prob 128, one per tx_size 0..3.
        for (int t = 0; t <= (int)Vp9TxSize.Tx32x32; t++)
            enc.Write(0, 128);
        // Three skip-prob "no update" bits at diff_update_prob = 252.
        for (int k = 0; k < Vp9SkipProbs.SkipContexts; k++)
            enc.Write(0, Vp9DiffUpdateProb.UpdateProb);
        return enc.Stop();
    }

    private static async Task<byte[]> BuildVp9CompressedHeaderGpuAsync(Accelerator acc)
    {
        using var kernel = new Vp9FrameCompressedHeaderKernel(acc);
        using var dOutBuf = acc.Allocate1D<byte>(64);
        using var dOutLen = acc.Allocate1D<long>(1);
        dOutBuf.View.CopyFromCPU(new byte[64]);
        kernel.Run(dOutBuf.View, dOutLen.View);
        await acc.SynchronizeAsync();

        long outLen = (await dOutLen.CopyToHostAsync())[0];
        var bytes = await dOutBuf.CopyToHostAsync();
        var result = new byte[outLen];
        Array.Copy(bytes, result, outLen);
        return result;
    }

    [TestMethod]
    public async Task Vp9FrameCompressedHeaderKernel_Allow32x32_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            var cpu = BuildVp9CompressedHeaderCpu();
            var gpu = await BuildVp9CompressedHeaderGpuAsync(acc);

            Equal(cpu.Length, gpu.Length);
            for (int i = 0; i < cpu.Length; i++)
            {
                if (cpu[i] != gpu[i])
                    throw new Exception(
                        $"compressed header byte mismatch at offset {i}: " +
                        $"cpu=0x{cpu[i]:X2} gpu=0x{gpu[i]:X2}");
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
