// Tests for Vp8KeyframeDecoderGpu - end-to-end GPU keyframe decoder.
// Round-trip: encode via Vp8KeyframeEncoder (CPU reference), decode
// via Vp8KeyframeWalker (CPU) to get reference recon, decode the
// same bytes via Vp8KeyframeDecoderGpu, verify GPU recon matches CPU
// recon byte-for-byte. Both decoders read the SAME bitstream, so
// they MUST produce identical output.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Vp8;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public async Task Vp8KeyframeDecoderGpu_SingleMbFrame_MatchesCpuDecoder()
    {
        await RunDecoderRoundTripTest(width: 16, height: 16, seed: unchecked((int)0xDEC0DE10));
    }

    [TestMethod]
    public async Task Vp8KeyframeDecoderGpu_MultiMb_2x2_MatchesCpuDecoder()
    {
        await RunDecoderRoundTripTest(width: 32, height: 32, seed: unchecked((int)0xDEC0DE20));
    }

    [TestMethod]
    public async Task Vp8KeyframeDecoderGpu_MultiMb_4x4_MatchesCpuDecoder()
    {
        await RunDecoderRoundTripTest(width: 64, height: 64, seed: unchecked((int)0xDEC0DE40));
    }

    private async Task RunDecoderRoundTripTest(int width, int height, int seed)
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            using var dec = new Vp8KeyframeDecoderGpu(acc);
            const int baseQIndex = 30;

            // 1. Generate random YUV.
            var rng = new Random(seed);
            var ySrc = new byte[width * height];
            var uSrc = new byte[(width / 2) * (height / 2)];
            var vSrc = new byte[(width / 2) * (height / 2)];
            for (int i = 0; i < ySrc.Length; i++) ySrc[i] = (byte)rng.Next(0, 256);
            for (int i = 0; i < uSrc.Length; i++) uSrc[i] = (byte)rng.Next(0, 256);
            for (int i = 0; i < vSrc.Length; i++) vSrc[i] = (byte)rng.Next(0, 256);

            // 2. Encode via CPU reference.
            byte[] encoded = Vp8KeyframeEncoder.EncodeKeyFrame(
                ySrc, ySrcStride: width,
                uSrc, uvSrcStride: width / 2,
                vSrc,
                width, height, baseQIndex);

            // 3. CPU decode via Vp8KeyframeWalker.
            var cpuTag = Vp8FrameTagParser.Parse(encoded.AsSpan());
            const int firstPartOffset = 10;
            int firstPartLen = cpuTag.FirstPartitionSize;
            var firstPart = encoded.AsSpan(firstPartOffset, firstPartLen).ToArray();
            var cpuModeReader = new Vp8BoolDecoder(firstPart);
            var cpuHdr = Vp8FrameHeaderParser.ParseKeyFrameHeader(cpuModeReader);
            int tokenOffset = firstPartOffset + firstPartLen;
            int tokenLen = encoded.Length - tokenOffset;
            var tokenPart = encoded.AsSpan(tokenOffset, tokenLen).ToArray();
            var cpuFb = new Vp8FrameBuffer(width, height);
            var cpuEc = new Vp8EntropyContexts(cpuFb.MbCols);
            Vp8KeyframeWalker.Decode(cpuTag, cpuHdr, cpuModeReader, tokenPart, cpuFb, cpuEc);

            // Pack the CPU walker's output (which uses MB-aligned strides) to
            // tight width/height for comparison.
            var cpuY = new byte[width * height];
            var cpuU = new byte[(width / 2) * (height / 2)];
            var cpuV = new byte[(width / 2) * (height / 2)];
            for (int r = 0; r < height; r++)
                Buffer.BlockCopy(cpuFb.YPlane, r * cpuFb.YStride, cpuY, r * width, width);
            for (int r = 0; r < height / 2; r++)
            {
                Buffer.BlockCopy(cpuFb.UPlane, r * cpuFb.UvStride, cpuU, r * (width / 2), width / 2);
                Buffer.BlockCopy(cpuFb.VPlane, r * cpuFb.UvStride, cpuV, r * (width / 2), width / 2);
            }

            // 4. GPU decode via Vp8KeyframeDecoderGpu.
            var gpuFrame = dec.DecodeKeyFrame(encoded, baseQIndex);

            // 5. Compare.
            Equal(width, gpuFrame.Width, "width");
            Equal(height, gpuFrame.Height, "height");
            int yMismatches = 0;
            int firstBadY = -1;
            for (int i = 0; i < cpuY.Length; i++)
            {
                if (cpuY[i] != gpuFrame.YPlane[i])
                {
                    if (firstBadY < 0) firstBadY = i;
                    yMismatches++;
                }
            }
            int uMismatches = 0;
            for (int i = 0; i < cpuU.Length; i++)
                if (cpuU[i] != gpuFrame.UPlane[i]) uMismatches++;
            int vMismatches = 0;
            for (int i = 0; i < cpuV.Length; i++)
                if (cpuV[i] != gpuFrame.VPlane[i]) vMismatches++;
            Equal(0, yMismatches, $"{width}x{height} Y plane (first bad i={firstBadY})");
            Equal(0, uMismatches, $"{width}x{height} U plane");
            Equal(0, vMismatches, $"{width}x{height} V plane");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
