// Tests for DeviceVerify, the accelerator-side compare every image test relies on: it must find a single planted
// difference anywhere (first / last byte, chunk and fold boundaries), count several, honour the tolerance and report the
// FIRST mismatch.

using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public async Task DeviceVerify_FindsPlantedMismatches_CountsThem_AndReportsTheFirst()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // 300000 bytes = 586 chunks of 512 -> two fold passes
            int n = 300000;
            var data = new byte[n];
            for (int i = 0; i < n; i++) data[i] = (byte)(i * 7 + (i >> 9));
            using var a = DeviceVerify.Upload(acc, data);
            using var b = DeviceVerify.Upload(acc, data);
            Equal(((long)0, (long)-1), await DeviceVerify.DiffAsync(acc, a.View, b.View, n), "identical buffers");

            foreach (int at in new[] { 0, 1, 511, 512, 262143, 262144, n - 1 })
            {
                using var c = DeviceVerify.Upload(acc, data);
                DeviceTransfer.Upload(c.View.SubView(at, 1), new[] { (byte)(data[at] ^ 0x40) });
                Equal(((long)1, (long)at), await DeviceVerify.DiffAsync(acc, a.View, c.View, n), $"one mismatch at {at}");
            }

            var many = (byte[])data.Clone();
            int[] planted = { 1000, 70000, 299999 };
            foreach (int at in planted) many[at] ^= 1;
            using var m = DeviceVerify.Upload(acc, many);
            Equal(((long)3, (long)1000), await DeviceVerify.DiffAsync(acc, a.View, m.View, n), "three mismatches");
            Equal(((long)0, (long)-1), await DeviceVerify.DiffAsync(acc, a.View, m.View, n, tolerance: 1), "within tolerance 1");
            Equal(((long)0, (long)-1), await DeviceVerify.DiffAsync(acc, a.View, m.View, 1000), "prefix before the first mismatch");

            string? msg = await DeviceVerify.CompareAsync(acc, m.View, data, bytesPerPixel: 4, width: 100);
            True(msg != null && msg.StartsWith("3 of 300000 bytes differ") && msg.Contains("pixel x=50 y=2"), $"message: {msg}");
            using var p = DeviceVerify.Poisoned(acc, 777);
            Equal(null, await DeviceVerify.AllEqualAsync(acc, p.View, 777, 0xAB), "poisoned buffer");
            DeviceTransfer.Upload(p.View.SubView(776, 1), new byte[] { 0 });
            True(await DeviceVerify.AllEqualAsync(acc, p.View, 777, 0xAB) != null, "a changed last byte is found");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
