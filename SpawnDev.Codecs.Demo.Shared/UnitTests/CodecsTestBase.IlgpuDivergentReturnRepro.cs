// Minimal ILGPU repro: a per-element kernel whose threads split between an early `return` branch and a fall-through
// branch inside one warp.
//
// Found 2026-11-08 in Vp9OutputKernels.PackPlanesKernel (VP9 decoder readback): on CUDA, for a 66x66 picture the
// output bytes 4356..4383 (the U-plane threads of the warp 4352..4383, whose first 4 threads are Y threads taking the
// early return) came back 0. CPU and OpenCL are right; sizes whose plane boundaries fall on warp boundaries pass.
// The variants below keep the kernel shape and change one thing each. The expected bytes come from running the same
// method on the host (.NET semantics).
//
// Root cause (Data, ILGPU): the O2 uniformity analysis started every kernel parameter as warp-uniform, including an
// implicitly grouped kernel's thread index, so the branch on `at < ySize` (directly on the index) was emitted as
// `bra.uni` and ptxas ran the warp straddling the plane boundary down one path. The 64-bit/32-bit split between the
// variants was incidental to how each branch got emitted, not the trigger.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public async Task IlgpuDivergentReturnRepro_PackPlanes_MatchesHost()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // 66x66 4:2:0 picture in a 128x128 / 64x64 padded layout (the VP9 frame buffer geometry).
            int w = 66, h = 66, uvW = 33, uvH = 33, yStride = 128, uvStride = 64;
            long yOff = 0, uOff = 128L * 128, vOff = uOff + 64L * 64;
            var src = new byte[vOff + 64L * 64];
            for (int k = 0; k < src.Length; k++) src[k] = (byte)(1 + k % 251);   // never 0
            long total = (long)w * h + 2L * uvW * uvH;

            using var dSrc = acc.Allocate1D(src);
            using var dDst = acc.Allocate1D<byte>(total);
            var fails = new List<string>();
            await RunPackVariant(acc, "early return, long index (VP9 PackPlanesKernel)", PackEarlyReturnLong, src, dSrc, dDst, total, w, h, uvW, uvH, yStride, uvStride, yOff, uOff, vOff, fails);
            await RunPackVariant(acc, "early return, int index", PackEarlyReturnInt, src, dSrc, dDst, total, w, h, uvW, uvH, yStride, uvStride, yOff, uOff, vOff, fails);
            await RunPackVariant(acc, "if / else, long index", PackIfElseLong, src, dSrc, dDst, total, w, h, uvW, uvH, yStride, uvStride, yOff, uOff, vOff, fails);
            if (fails.Count > 0) throw new Exception($"{acc.AcceleratorType}:\n" + string.Join("\n", fails));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    private static async Task RunPackVariant(Accelerator acc, string name,
        Action<Index1D, ArrayView<byte>, ArrayView<byte>, int, int, int, int, int, int, long, long, long> kernelMethod,
        byte[] src, MemoryBuffer1D<byte, Stride1D.Dense> dSrc, MemoryBuffer1D<byte, Stride1D.Dense> dDst, long total,
        int w, int h, int uvW, int uvH, int yStride, int uvStride, long yOff, long uOff, long vOff, List<string> fails)
    {
        dDst.MemSetToZero();
        var kernel = acc.LoadAutoGroupedStreamKernel(kernelMethod);
        kernel((int)total, dSrc.View, dDst.View, w, h, uvW, uvH, yStride, uvStride, yOff, uOff, vOff);
        var gpu = await dDst.CopyToHostAsync();
        // host reference: the same method, element by element
        var want = new byte[total];
        for (long k = 0; k < total; k++) want[k] = PackReference(k, src, w, h, uvW, uvH, yStride, uvStride, yOff, uOff, vOff);
        var bad = new List<long>();
        for (long k = 0; k < total; k++) if (gpu[k] != want[k]) bad.Add(k);
        if (bad.Count > 0)
            fails.Add($"{name}: {bad.Count}/{total} bytes wrong, first {bad[0]} last {bad[^1]} (got {gpu[bad[0]]}, want {want[bad[0]]})");
    }

    private static byte PackReference(long i, byte[] src, int w, int h, int uvW, int uvH, int yStride, int uvStride, long yOffset, long uOffset, long vOffset)
    {
        long ySize = (long)w * h, uvSize = (long)uvW * uvH;
        long at = i;
        if (at < ySize) return src[yOffset + (at / w) * yStride + at % w];
        at -= ySize;
        long plane = at < uvSize ? uOffset : vOffset;
        if (at >= uvSize) at -= uvSize;
        return src[plane + (at / uvW) * uvStride + at % uvW];
    }

    // Vp9OutputKernels.PackPlanesKernel verbatim.
    private static void PackEarlyReturnLong(Index1D i, ArrayView<byte> src, ArrayView<byte> dst, int w, int h, int uvW, int uvH,
        int yStride, int uvStride, long yOffset, long uOffset, long vOffset)
    {
        long ySize = (long)w * h, uvSize = (long)uvW * uvH;
        long at = i;
        if (at < ySize)
        {
            dst[at] = src[yOffset + (at / w) * yStride + at % w];
            return;
        }
        at -= ySize;
        long plane = at < uvSize ? uOffset : vOffset;
        if (at >= uvSize) at -= uvSize;
        dst[i] = src[plane + (at / uvW) * uvStride + at % uvW];
    }

    // Same shape with 32-bit index math.
    private static void PackEarlyReturnInt(Index1D i, ArrayView<byte> src, ArrayView<byte> dst, int w, int h, int uvW, int uvH,
        int yStride, int uvStride, long yOffset, long uOffset, long vOffset)
    {
        int ySize = w * h, uvSize = uvW * uvH;
        int at = i;
        if (at < ySize)
        {
            dst[at] = src[yOffset + (at / w) * yStride + at % w];
            return;
        }
        at -= ySize;
        long plane = at < uvSize ? uOffset : vOffset;
        if (at >= uvSize) at -= uvSize;
        dst[i] = src[plane + (at / uvW) * uvStride + at % uvW];
    }

    // Same math, the Y branch as if / else instead of an early return.
    private static void PackIfElseLong(Index1D i, ArrayView<byte> src, ArrayView<byte> dst, int w, int h, int uvW, int uvH,
        int yStride, int uvStride, long yOffset, long uOffset, long vOffset)
    {
        long ySize = (long)w * h, uvSize = (long)uvW * uvH;
        long at = i;
        if (at < ySize)
        {
            dst[at] = src[yOffset + (at / w) * yStride + at % w];
        }
        else
        {
            at -= ySize;
            long plane = at < uvSize ? uOffset : vOffset;
            if (at >= uvSize) at -= uvSize;
            dst[i] = src[plane + (at / uvW) * uvStride + at % uvW];
        }
    }
}
