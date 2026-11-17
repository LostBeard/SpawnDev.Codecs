// FrameHash on the accelerator must equal FrameHash on the host for every length shape (empty, inside one chunk, word
// tails of 1-3 bytes, exact chunk multiples, a 1080p 4:2:0 frame), and a one-byte change must change the hash. The host
// function is libvpx_ref.c's fh_* (the VP9 every-frame tests compare against hashes that tool wrote, which pins the C
// and C# versions together on every frame of every vector).

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod(Category = "NotOnWebGL")]
    public async Task FrameHash_Device_EqualsHost_AndSeesOneByte()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            const int chunk = FrameHash.Chunk;
            int[] lengths = { 0, 1, 2, 3, 4, 5, 7, chunk - 1, chunk, chunk + 1, chunk + 3, 3 * chunk, 3 * chunk + 2, 1920 * 1080 * 3 / 2 };
            var rng = new Random(20261117);
            var failures = new List<string>();
            foreach (int n in lengths)
            {
                var bytes = new byte[n];
                rng.NextBytes(bytes);
                using var d = acc.Allocate1D<byte>(Math.Max(1, n));
                if (n > 0) d.View.SubView(0, n).CopyFromCPU(bytes);
                string host = FrameHash.Compute(bytes), dev = await FrameHash.ComputeAsync(acc, d.View, n);
                if (host != dev) failures.Add($"{n} bytes: device {dev}, host {host}");
                if (n == 0) continue;
                // one byte changed (the last one: it sits in the last chunk's tail when there is one)
                bytes[n - 1] ^= 0x01;
                d.View.SubView(n - 1, 1).CopyFromCPU(new[] { bytes[n - 1] });
                string host2 = FrameHash.Compute(bytes), dev2 = await FrameHash.ComputeAsync(acc, d.View, n);
                if (host2 != dev2) failures.Add($"{n} bytes, last byte changed: device {dev2}, host {host2}");
                if (host2 == host) failures.Add($"{n} bytes: changing the last byte did not change the hash");
            }
            True(failures.Count == 0, string.Join("\n", failures));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }
}
