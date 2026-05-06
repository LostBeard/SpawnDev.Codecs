// Cross-backend smoke tests for Vp9FrameEntropyKernel.RunMultiTile - the
// per-tile dispatch path that lands as part of the VP9 multi-tile entropy
// architecture (vp9-multi-tile-entropy branch increments 1-5).
//
// The minimal smoke: RunMultiTile with TileCols=TileRows=1 must produce
// byte-identical output to the legacy single-tile Run path. If the
// multi-tile dispatch can't agree with single-tile at the trivial config,
// it can't be relied on at any larger config.

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
    public async Task Vp9FrameEntropyKernel_RunMultiTile_OneTile_MatchesSingleTileRun()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // Smallest SB-aligned VP9 frame: 64x64 = 1 SB = 16 MBs.
            int mbCols = 4;
            int mbRows = 4;
            int frameMiCols = mbCols * 2;
            int frameMiRows = mbRows * 2;
            int mbCount = mbCols * mbRows;

            // Set up real-shaped coef buffers (all-zero is a valid input -
            // the walker still emits the skip-flag CDF + intra-mode CDF +
            // coef tokens; tile bytes won't be zero, just deterministic).
            using var dY = acc.Allocate1D<short>(mbCount * 256);
            using var dU = acc.Allocate1D<short>(mbCount * 64);
            using var dV = acc.Allocate1D<short>(mbCount * 64);
            using var dByteConsts = acc.Allocate1D<byte>(Vp9KeyframeConstantsGpu.BuildByteConstsBuffer().Length);
            using var dUshortConsts = acc.Allocate1D<ushort>(Vp9KeyframeConstantsGpu.BuildUshortConstsBuffer().Length);
            dY.View.MemSetToZero();
            dU.View.MemSetToZero();
            dV.View.MemSetToZero();
            dByteConsts.View.CopyFromCPU(Vp9KeyframeConstantsGpu.BuildByteConstsBuffer());
            dUshortConsts.View.CopyFromCPU(Vp9KeyframeConstantsGpu.BuildUshortConstsBuffer());

            using var entropy = new Vp9FrameEntropyKernel(acc);

            // Worst-case tile bytes for this frame.
            long worstCaseTile = mbCount * 1024L + 256L;

            // Single-tile baseline: existing Run() path.
            using var dOutSingle = acc.Allocate1D<byte>(worstCaseTile);
            using var dLenSingle = acc.Allocate1D<long>(1);
            dOutSingle.View.MemSetToZero();
            dLenSingle.View.MemSetToZero();
            entropy.Run(
                dY.View, dU.View, dV.View,
                dOutSingle.View, dLenSingle.View,
                dByteConsts.View, dUshortConsts.View,
                mbCols, mbRows, frameMiCols, frameMiRows);
            await acc.SynchronizeAsync();
            var singleLen = (await dLenSingle.CopyToHostAsync())[0];
            var singleBytes = (await dOutSingle.CopyToHostAsync()).AsSpan(0, (int)singleLen).ToArray();

            // Multi-tile path with TileCols=TileRows=1 (single tile, full
            // frame range). Output buffer is per-tile-stride * numTiles =
            // worstCaseTile * 1 = worstCaseTile.
            using var dOutMulti = acc.Allocate1D<byte>(worstCaseTile);
            using var dLenMulti = acc.Allocate1D<long>(1);
            dOutMulti.View.MemSetToZero();
            dLenMulti.View.MemSetToZero();
            var tileStrides = new Vp9FrameEntropyTileStrides
            {
                MbCols = mbCols,
                MbRows = mbRows,
                FrameMiCols = frameMiCols,
                FrameMiRows = frameMiRows,
                TileCols = 1,
                TileRows = 1,
                Log2TileCols = 0,
                Log2TileRows = 0,
                OutBufStride = (int)worstCaseTile,
            };
            entropy.RunMultiTile(
                dY.View, dU.View, dV.View,
                dOutMulti.View, dLenMulti.View,
                dByteConsts.View, dUshortConsts.View,
                tileStrides);
            await acc.SynchronizeAsync();
            var multiLen = (await dLenMulti.CopyToHostAsync())[0];
            var multiBytes = (await dOutMulti.CopyToHostAsync()).AsSpan(0, (int)multiLen).ToArray();

            // Single-tile path (EncodeFrameBody) and multi-tile-with-N=1 path
            // (EncodeTileBody, full-frame range) must produce identical bytes.
            // Any divergence means the parallel walker bodies disagree on
            // bool encoder init, above/left reset semantics, or the SB walk
            // shape - they're maintained as duplicates deliberately to keep
            // the single-tile kernel Wasm-safe (per-function local count).
            Equal(singleLen, multiLen, "tile byte length");
            for (int i = 0; i < singleBytes.Length; i++)
            {
                if (singleBytes[i] != multiBytes[i])
                    throw new Exception(
                        $"byte mismatch at offset {i}: single=0x{singleBytes[i]:X2} multi=0x{multiBytes[i]:X2}");
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
