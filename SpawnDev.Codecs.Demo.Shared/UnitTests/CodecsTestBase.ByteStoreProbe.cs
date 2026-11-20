// A measurement, not a check: what one thread's serial BYTE stores cost on this accelerator, next to the same bytes
// stored as 32-bit words. The serial inflate writes every output byte one at a time (a PNG decode of 512x512 RGB: 786 KB);
// WebGPU has no byte stores (SpawnDev.ILGPU emulates a sub-word store with atomic read-modify-write on the word), and the
// PNG decode's inflate took 436 ms there against 44 ms on Wasm (RTX 4070, 2026-11-20). It passes whenever the bytes are
// right; the times go to the report line. MEASURED 2026-11-20 (RTX 4070), 786,432 bytes in 24 launches: CUDA bytes
// 12 ms / words 4; WebGPU 45 / 53; Wasm 7 / 75. Byte stores are NOT the inflate's WebGPU cost (436 ms); the serial
// Huffman decode itself is (CUDA 148 ms too).

using System.Diagnostics;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private static void SerialByteStoreKernel(Index1D i, ArrayView<byte> dst, int start, int count)
    {
        if (i != 0) return;
        for (int k = start; k < start + count; k++) dst[k] = (byte)(k * 7 + 3);
    }

    private static void SerialWordStoreKernel(Index1D i, ArrayView<uint> dst, int start, int count)
    {
        if (i != 0) return;
        for (int k = start; k < start + count; k++)
        {
            int b = 4 * k;
            dst[k] = (uint)(byte)(b * 7 + 3) | ((uint)(byte)((b + 1) * 7 + 3) << 8)
                   | ((uint)(byte)((b + 2) * 7 + 3) << 16) | ((uint)(byte)((b + 3) * 7 + 3) << 24);
        }
    }

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task Probe_SerialByteStores_VersusWordStores()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            const int n = 786432, perLaunch = 32768;   // a 512x512 RGB PNG's filtered bytes; bounded launches like the inflate
            var bytesK = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, int, int>(SerialByteStoreKernel);
            var wordsK = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<uint>, int, int>(SerialWordStoreKernel);
            using var buf = acc.Allocate1D<byte>(n);
            using var wbuf = acc.Allocate1D<uint>(n / 4);
            bytesK(1, buf.View, 0, 4); wordsK(1, wbuf.View, 0, 1);
            await acc.SynchronizeAsync();
            var sw = Stopwatch.StartNew();
            for (int s = 0; s < n; s += perLaunch) bytesK(1, buf.View, s, perLaunch);
            await acc.SynchronizeAsync();
            double byteMs = sw.Elapsed.TotalMilliseconds;
            sw.Restart();
            for (int s = 0; s < n / 4; s += perLaunch / 4) wordsK(1, wbuf.View, s, perLaunch / 4);
            await acc.SynchronizeAsync();
            double wordMs = sw.Elapsed.TotalMilliseconds;
            var a = await buf.View.CopyToHostAsync();
            var w = await wbuf.View.CopyToHostAsync();
            for (int k = 0; k < n; k++)
            {
                byte want = (byte)(k * 7 + 3);
                if (a[k] != want || (byte)(w[k >> 2] >> (8 * (k & 3))) != want) throw new Exception($"byte {k} wrong");
            }
            Console.WriteLine($"[report] {acc.AcceleratorType}: one thread, {n} bytes in {n / perLaunch} launches: byte stores {byteMs:F0} ms, " +
                $"the same bytes as 32-bit word stores {wordMs:F0} ms");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }
}
