// Vp9FrameAssembleKernel.RunTiles: the frame layout of VP9 spec section 6.4 / libvpx encode_tiles: uncompressed header,
// compressed header, then every tile, each but the last prefixed by its size as 4 big-endian bytes. Checked against
// the layout built on the host for 1 to 16 tiles, empty tiles, tile sizes that need every prefix byte (> 2^24), and a
// tile that outgrew its slot (the frame length is -1 and nothing is assembled).

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod(Category = "NotOnWebGL")]
    public async Task Vp9FrameAssembleKernel_Tiles_MatchesSpecLayout()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var asm = new Vp9FrameAssembleKernel(acc);
            var rng = new Random(77);
            var failures = new List<string>();
            // (tiles, tile stride, one tile longer than 2^24 bytes, one tile over its slot)
            var cases = new (int Tiles, int Stride, bool Huge, bool Overflow)[]
            {
                (1, 300, false, false), (2, 300, false, false), (4, 257, false, false), (16, 64, false, false),
                (3, 100, false, true), (2, (1 << 24) + 300, true, false),
            };
            foreach (var (tiles, stride, huge, overflowCase) in cases)
            {
                // the > 2^24-byte tile is built on the host: desktop lanes only (the browser host is single-threaded Wasm)
                if (huge && OperatingSystem.IsBrowser()) continue;
                int u = rng.Next(5, 30), c = rng.Next(0, 40);
                var uh = new byte[64]; var ch = new byte[64];
                rng.NextBytes(uh); rng.NextBytes(ch);
                var slots = new byte[(long)tiles * stride];
                var lens = new long[tiles];
                for (int t = 0; t < tiles; t++)
                {
                    lens[t] = t % 3 == 1 ? 0 : rng.Next(1, Math.Min(stride, 4096) + 1);   // empty tiles too
                    if (huge && t == 0) lens[t] = (1 << 24) + 257;
                    for (int k = 0; k < Math.Min(lens[t], 4096); k++) slots[(long)t * stride + k] = (byte)rng.Next(256);
                    if (huge && t == 0) for (long k = 4096; k < lens[t]; k++) slots[k] = (byte)(k * 7);
                }
                if (overflowCase) lens[1] = stride + 1;
                // expected
                var want = new List<byte>();
                want.AddRange(uh.AsSpan(0, u).ToArray());
                want.AddRange(ch.AsSpan(0, c).ToArray());
                for (int t = 0; t < tiles; t++)
                {
                    if (t + 1 < tiles) { want.Add((byte)(lens[t] >> 24)); want.Add((byte)(lens[t] >> 16)); want.Add((byte)(lens[t] >> 8)); want.Add((byte)lens[t]); }
                    for (long k = 0; k < Math.Min(lens[t], stride); k++) want.Add(slots[(long)t * stride + k]);
                }
                long cap = want.Count + 16;
                using var dUh = acc.Allocate1D(uh);
                using var dCh = acc.Allocate1D(ch);
                using var dSlots = acc.Allocate1D(slots);
                using var dLens = acc.Allocate1D(lens);
                using var dUl = acc.Allocate1D(new long[] { u });
                using var dCl = acc.Allocate1D(new long[] { c });
                using var dOut = acc.Allocate1D<byte>(cap);
                using var dOutLen = acc.Allocate1D<long>(1);
                dOut.View.MemSet(0xAB);
                asm.RunTiles(dUh.View, dCh.View, dSlots.View, stride, dLens.View, tiles, dOut.View, dOutLen.View, dUl.View, dCl.View);
                await acc.SynchronizeAsync();
                long got = (await dOutLen.View.CopyToHostAsync())[0];
                string name = $"{tiles} tiles, stride {stride}{(huge ? ", a tile over 2^24 bytes" : "")}{(overflowCase ? ", a tile over its slot" : "")}";
                if (overflowCase)
                {
                    if (got != -1) failures.Add($"{name}: length {got}, expected -1");
                    else if (await DeviceVerify.AllEqualAsync(acc, dOut.View, cap, 0xAB) is string w) failures.Add($"{name}: assembled bytes anyway: {w}");
                    continue;
                }
                if (got != want.Count) { failures.Add($"{name}: length {got}, expected {want.Count}"); continue; }
                string? diff = await DeviceVerify.CompareHostResultAsync(acc, want.ToArray(), dOut.View, want.Count) is string d0 ? d0 : null;
                if (diff != null) failures.Add($"{name}: {diff}");
            }
            if (failures.Count > 0) throw new Exception(string.Join("\n", failures));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }
}
