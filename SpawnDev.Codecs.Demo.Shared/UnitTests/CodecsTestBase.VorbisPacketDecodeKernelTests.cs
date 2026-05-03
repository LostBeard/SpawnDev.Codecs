// Cross-backend smoke test for VorbisPacketDecodeKernel - the
// in-progress v2 GPU integration kernel that wires header parse +
// per-channel floor decode + per-submap residue decode in one
// dispatch (Plans/PLAN-Vorbis-Decoder-V2-GPU-BitStream-Decode.md
// Step 2). Verifies that ILGPU's LoadAutoGroupedStreamKernel accepts
// the kernel signature - in particular the
// VorbisPacketDecodeStaticInputs struct param (38 ArrayView fields).
//
// If a backend rejects the kernel (e.g. WebGPU binding-count limit
// or struct-of-ArrayView marshaling gap), this test surfaces the
// failure as a clean error message before the DecoderGpu
// integration risks breaking the working v1 decoder tests.
//
// This test does NOT verify correctness of the decoded output -
// that's covered by the integration tests once Step 3 (DecoderGpu
// dispatch the kernel) lands.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Audio.Vorbis;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public async Task VorbisPacketDecodeKernel_LoadsOnAccelerator()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // Just try to load the kernel. If ILGPU compiles it for the
            // backend, the signature is acceptable. If a backend rejects
            // the kernel parameter list (e.g. struct-of-ArrayView marshaling
            // gap), the load will throw - and that failure tells us where
            // the v2 DecoderGpu integration would fail BEFORE it can break
            // the existing v1 decoder tests.
            var kernel = acc.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView<byte>,
                int, int, int, int, int, int,
                VorbisPacketDecodeStaticInputs,
                ArrayView<int>,
                ArrayView<float>,
                ArrayView<int>,
                ArrayView<float>>(VorbisPacketDecodeKernel.Run);

            True(kernel is not null,
                "VorbisPacketDecodeKernel.Run must load via LoadAutoGroupedStreamKernel; "
                + "if this fails, ILGPU's kernel-parameter handling for the "
                + "VorbisPacketDecodeStaticInputs struct (38 ArrayView fields) needs review.");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
