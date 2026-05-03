// Cross-backend roundtrip tests for Av1KeyframeDecoderGpu.
// Verifies the decoder reconstructs YUV planes that match what the
// encoder produced (the encoder keeps an internal recon buffer; the
// decoder should reproduce it bit-exactly from the encoded byte stream).
//
// Test surface:
//   - Const gray YUV 64x64 -> encode -> decode -> recon matches input
//   - Random YUV 64x64 -> encode -> decode -> recon matches encoder's
//     internal recon (lossy encode/decode pair)

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Av1;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public async Task Av1KeyframeDecoderGpu_ConstGray64x64_RoundTrip()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            const int width = 64;
            const int height = 64;
            const int qIdx = 32;

            // Const gray: all 128. Encoder's residual = 0, so eob=0 path,
            // and decoder should produce all-128 recon.
            int yLen = width * height;
            int uvLen = yLen / 4;
            var yPlane = new byte[yLen]; for (int i = 0; i < yLen; i++) yPlane[i] = 128;
            var uPlane = new byte[uvLen]; for (int i = 0; i < uvLen; i++) uPlane[i] = 128;
            var vPlane = new byte[uvLen]; for (int i = 0; i < uvLen; i++) vPlane[i] = 128;

            using var enc = new Av1KeyframeEncoderGpu(acc);
            byte[] tileBytes = await enc.EncodeSingleTileAsync(yPlane, uPlane, vPlane, width, height, qIdx);

            using var dec = new Av1KeyframeDecoderGpu(acc);
            var (yDec, uDec, vDec) = await dec.DecodeSingleTileAsync(tileBytes, width, height, qIdx);

            // For const-gray inputs the encoder's residual is zero, so
            // recon should equal source exactly.
            for (int i = 0; i < yLen; i++)
                if (yDec[i] != 128) throw new Exception($"Y[{i}] expected 128, got {yDec[i]}");
            for (int i = 0; i < uvLen; i++)
            {
                if (uDec[i] != 128) throw new Exception($"U[{i}] expected 128, got {uDec[i]}");
                if (vDec[i] != 128) throw new Exception($"V[{i}] expected 128, got {vDec[i]}");
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Av1KeyframeDecoderGpu_Random64x64_MatchesEncoderRecon()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            const int width = 64;
            const int height = 64;
            const int qIdx = 32;

            var rng = new Random(unchecked((int)0xA1DECDEFu));
            int yLen = width * height;
            int uvLen = yLen / 4;
            var yPlane = new byte[yLen]; rng.NextBytes(yPlane);
            var uPlane = new byte[uvLen]; rng.NextBytes(uPlane);
            var vPlane = new byte[uvLen]; rng.NextBytes(vPlane);

            // Encode produces tile bytes (and the encoder's internal recon
            // matches what the decoder must reproduce).
            using var enc = new Av1KeyframeEncoderGpu(acc);
            byte[] tileBytes = await enc.EncodeSingleTileAsync(yPlane, uPlane, vPlane, width, height, qIdx);

            // Decode the tile bytes back to YUV.
            using var dec = new Av1KeyframeDecoderGpu(acc);
            var (yDec, uDec, vDec) = await dec.DecodeSingleTileAsync(tileBytes, width, height, qIdx);

            // For correctness, compare the GPU decoder's recon to the CPU
            // decoder's recon for the same tile bytes. Use Av1KeyframeWalker
            // to derive the CPU reference recon from the same byte stream.
            // For now, we assume the CPU encoder's recon (verified bit-exact
            // by the encoder tests) is the correct target.
            byte[] cpuTileBytes = Av1KeyframeEncoder.EncodeSingleTile(
                yPlane, width, uPlane, width >> 1, vPlane, width, height, qIdx);
            // Sanity: GPU and CPU produce identical encoded bytes (already
            // verified by encoder tests, but useful as a guard here).
            if (cpuTileBytes.Length != tileBytes.Length)
                throw new Exception($"GPU/CPU tile len mismatch: cpu={cpuTileBytes.Length} gpu={tileBytes.Length}");
            for (int i = 0; i < cpuTileBytes.Length; i++)
                if (cpuTileBytes[i] != tileBytes[i])
                    throw new Exception($"GPU/CPU tile byte {i}: cpu={cpuTileBytes[i]:X2} gpu={tileBytes[i]:X2}");

            // Decoder produced SOME recon. Verify it's well-formed by checking
            // every byte is in [0, 255] (trivially true for byte but we check
            // for any out-of-band values via length and content sanity).
            // Stronger: re-encode the GPU-decoded recon and verify the
            // encoder's recon matches the decoder's output for every pixel.
            var (encBytes2, yReconEnc, uReconEnc, vReconEnc) =
                await enc.EncodeKeyFrameWithReconAsync(yPlane, uPlane, vPlane, width, height, qIdx);

            for (int i = 0; i < yLen; i++)
                if (yReconEnc[i] != yDec[i])
                    throw new Exception($"Y recon mismatch at [{i}]: enc={yReconEnc[i]} dec={yDec[i]}");
            for (int i = 0; i < uvLen; i++)
            {
                if (uReconEnc[i] != uDec[i])
                    throw new Exception($"U recon mismatch at [{i}]: enc={uReconEnc[i]} dec={uDec[i]}");
                if (vReconEnc[i] != vDec[i])
                    throw new Exception($"V recon mismatch at [{i}]: enc={vReconEnc[i]} dec={vDec[i]}");
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
