// Demo: read the first partition decision from BBB.webm's first frame.
// Composes Vp9BoolDecoder + Vp9PartitionProbs + Vp9PartitionTree on
// real VP9 tile bytes to show the entropy decode path producing a
// meaningful symbol.

#:project SpawnDev.Codecs/SpawnDev.Codecs.csproj
using System;
using System.IO;
using System.Linq;
using SpawnDev.Codecs.Container.Matroska;
using SpawnDev.Codecs.Video;
using SpawnDev.Codecs.Video.Vp9;

string webmPath = "SpawnDev.Codecs.Demo.Shared/TestData/Big_Buck_Bunny_180_10s.webm";
using var stream = File.OpenRead(webmPath);
var container = new MatroskaContainer(stream);
var video = container.Tracks.First(t => t.IsVideo);
var first = container.Frames.First(f => f.TrackNumber == video.TrackNumber);

var decoder = new Vp9Decoder();
var sink = new IgnoreSink();
await decoder.DecodeFrameAsync(first.Data, sink);

Console.WriteLine();
Console.WriteLine("============================================================");
Console.WriteLine("  VP9 first-partition decode demo");
Console.WriteLine("============================================================");
Console.WriteLine();
Console.WriteLine($"Source: {webmPath}");
Console.WriteLine($"Frame: KeyFrame {decoder.LastFrameHeader?.FrameWidth}x{decoder.LastFrameHeader?.FrameHeight}, "
    + $"{decoder.LastTileGroup?.Tiles.Count} tile(s)");
Console.WriteLine();
Console.WriteLine($"Compressed header parsed:");
Console.WriteLine($"  tx_mode  = {decoder.LastCompressedResult?.TxMode}");
Console.WriteLine($"  ref_mode = {decoder.LastCompressedResult?.ReferenceMode}");
var seg = decoder.LastCompleteHeader?.Segmentation;
Console.WriteLine($"  segmentation_enabled = {seg?.Enabled}");
Console.WriteLine($"  segmentation_update_map = {seg?.UpdateMap}");
Console.WriteLine($"  segmentation_temporal_update = {seg?.TemporalUpdate}");
Console.WriteLine();

// Initialize bool decoder over tile 0 bytes.
var tile0 = decoder.LastTileGroup!.Tiles[0];
var data = first.Data.ToArray();
var tileBytes = new byte[tile0.Length];
Buffer.BlockCopy(data, tile0.Offset, tileBytes, 0, tile0.Length);

Console.WriteLine($"Tile 0: offset={tile0.Offset}, length={tile0.Length}");
Console.Write("Tile 0 first 16 bytes:");
for (int i = 0; i < Math.Min(16, tileBytes.Length); i++)
    Console.Write($" {tileBytes[i]:X2}");
Console.WriteLine();
var skipProbsDump = decoder.LastCompressedState!.SkipProbs.Probs;
Console.WriteLine($"Skip probs (post-compressed-header): [{skipProbsDump[0]}, {skipProbsDump[1]}, {skipProbsDump[2]}]");
var p32x32Dump = decoder.LastCompressedState.TxModeProbs.P32x32;
Console.WriteLine($"P32x32[0,*]: [{p32x32Dump[0,0]}, {p32x32Dump[0,1]}, {p32x32Dump[0,2]}]");
Console.WriteLine($"P32x32[1,*]: [{p32x32Dump[1,0]}, {p32x32Dump[1,1]}, {p32x32Dump[1,2]}]");
var p16x16Dump = decoder.LastCompressedState.TxModeProbs.P16x16;
Console.WriteLine($"P16x16[0,*]: [{p16x16Dump[0,0]}, {p16x16Dump[0,1]}]");
Console.WriteLine($"P16x16[1,*]: [{p16x16Dump[1,0]}, {p16x16Dump[1,1]}]");

// First superblock at top-left: 64x64 (sizeIdx=3), both above + left
// out of frame so splitState=0.
var br = new Vp9BoolDecoder(tileBytes, 0, tileBytes.Length);
var probs = Vp9PartitionProbs.KeyframeProbs(sizeIdx: 3, splitState: 0);
Console.WriteLine($"Partition probs (64x64, both unsplit context): "
    + $"None_vs_else={probs[0]}, Horz_vs_else={probs[1]}, Vert_vs_Split={probs[2]}");

var partition = Vp9PartitionTree.Decode(p => br.Read(p), probs);

Console.WriteLine();
Console.WriteLine("============================================================");
Console.WriteLine($"  FIRST PARTITION DECISION (top-left 64x64): {partition}");
Console.WriteLine("============================================================");
Console.WriteLine();
Console.WriteLine("Interpretation:");
switch (partition)
{
    case Vp9PartitionType.None:
        Console.WriteLine("  -> Block is decoded as one 64x64 transform block.");
        break;
    case Vp9PartitionType.Horz:
        Console.WriteLine("  -> Block split horizontally: 64x32 top + 64x32 bottom.");
        break;
    case Vp9PartitionType.Vert:
        Console.WriteLine("  -> Block split vertically: 32x64 left + 32x64 right.");
        break;
    case Vp9PartitionType.Split:
        Console.WriteLine("  -> Block split into 4 quarter-sized 32x32 sub-blocks; recurse.");
        break;
}

// If split, decode the first 32x32 sub-block's partition decision.
if (partition == Vp9PartitionType.Split)
{
    Console.WriteLine();
    Console.WriteLine("Recursing into top-left 32x32 sub-block...");
    // 32x32 SB at top-left corner of the 64x64: above + left still out of frame.
    var probs32 = Vp9PartitionProbs.KeyframeProbs(sizeIdx: 2, splitState: 0);
    Console.WriteLine($"Partition probs (32x32, both unsplit context): "
        + $"None={probs32[0]}, Horz={probs32[1]}, Vert/Split={probs32[2]}");
    var partition32 = Vp9PartitionTree.Decode(p => br.Read(p), probs32);
    Console.WriteLine($"  Top-left 32x32 partition: {partition32}");

    if (partition32 == Vp9PartitionType.None)
    {
        // Leaf at 32x32: read skip flag for this block. For an intra
        // keyframe block, intra_inter is implicit (always 1=intra), so
        // the next bit is the skip flag.
        // skip_probs[ctx]: ctx=0 here (above + left both unavailable).
        byte skipProb = decoder.LastCompressedState!.SkipProbs.Probs[0];
        Console.WriteLine($"  Skip prob (context 0): {skipProb}");
        int skipFlag = br.Read(skipProb);
        Console.WriteLine($"  Skip flag for top-left 32x32: {skipFlag}");
        Console.WriteLine($"    -> {(skipFlag != 0 ? "all-zero residual (skip)" : "has coefficients")}");

        // tx_size: libvpx read_intra_frame_mode_info reads tx_size between
        // skip and y_mode. Vp9TxSizeDecoder.ReadTxSize is a no-op when
        // tx_mode != TxModeSelect; for TxModeSelect (common in libvpx
        // output) it consumes 1-3 bits via the per-context tx_size_probs
        // tree. Top-left no-neighbor tx_size_context = 1 per libvpx
        // get_tx_size_context: (above_ctx + left_ctx) > max_tx_size with
        // both defaulting to max_tx_size when neighbors are missing.
        var txMode = decoder.LastCompressedResult!.TxMode;
        var maxTxSize32 = Vp9MaxTxSize.ForBlockSize(Vp9BlockSize.Block32x32);
        Span<byte> txProbs32 = stackalloc byte[3]
        {
            decoder.LastCompressedState!.TxModeProbs.P32x32[1, 0],
            decoder.LastCompressedState!.TxModeProbs.P32x32[1, 1],
            decoder.LastCompressedState!.TxModeProbs.P32x32[1, 2],
        };
        var txSize32 = Vp9TxSizeDecoder.ReadTxSize(txMode, maxTxSize32, br, txProbs32);
        Console.WriteLine($"  tx_mode = {txMode}, tx_size for 32x32: {txSize32}");

        // Intra Y mode: for top-left block, above + left are out of frame
        // so libvpx treats them as DcPred. Mode is read regardless of
        // skip flag (mode determines prediction even with no residual).
        var yModeProbs = Vp9IntraModeProbs.KeyframeYProbs(Vp9IntraMode.DcPred, Vp9IntraMode.DcPred);
        var yMode = Vp9IntraModeTree.Decode(p => br.Read(p), yModeProbs);
        Console.WriteLine($"  Intra Y mode (above=Dc, left=Dc): {yMode}");

        // ACTUAL PIXEL DECODE for top-left 32x32 Y block.
        // Since skip=1, residual is zero - the prediction IS the output.
        // Above row + left col are out of frame, libvpx fills with 127/129.
        if (skipFlag == 1)
        {
            const int N = 32;
            var aboveBuf = new byte[N * 2]; // 2N for D45/D63 paths
            var leftBuf = new byte[N];
            // Out-of-frame fills per libvpx convention.
            Array.Fill(aboveBuf, (byte)127);
            Array.Fill(leftBuf, (byte)129);
            byte topLeft = 127;
            var dst = new byte[N * N];
            Vp9IntraPredictor.Predict(yMode, topLeft, aboveBuf, leftBuf, dst, N, stride: N);

            // Compute simple statistics of the decoded block.
            int min = 255, max = 0, sum = 0;
            for (int i = 0; i < dst.Length; i++)
            {
                if (dst[i] < min) min = dst[i];
                if (dst[i] > max) max = dst[i];
                sum += dst[i];
            }
            int mean = sum / dst.Length;
            Console.WriteLine();
            Console.WriteLine($"  ACTUAL DECODED 32x32 Y BLOCK PIXELS:");
            Console.WriteLine($"    min={min}, max={max}, mean={mean}");
            Console.WriteLine($"    Top-left 4x4 sample:");
            for (int row = 0; row < 4; row++)
            {
                Console.Write("      ");
                for (int col = 0; col < 4; col++)
                    Console.Write($"{dst[row * N + col],4}");
                Console.WriteLine();
            }

            Console.WriteLine();
            Console.WriteLine("  Note: skip=1 means the residual is zero, so prediction IS the output.");
            Console.WriteLine("  For top-left DcPred with no neighbors, prediction is 128 by spec.");
            Console.WriteLine("  Compare to ffmpeg ground truth for the actual block to confirm.");
        }
    }
    else if (partition32 == Vp9PartitionType.Split)
    {
        var probs16 = Vp9PartitionProbs.KeyframeProbs(sizeIdx: 1, splitState: 0);
        var partition16 = Vp9PartitionTree.Decode(p => br.Read(p), probs16);
        Console.WriteLine($"  Top-left 16x16 partition: {partition16}");

        if (partition16 == Vp9PartitionType.None)
        {
            // 16x16 leaf - read skip + tx_size + Y mode (libvpx order).
            byte sp = decoder.LastCompressedState!.SkipProbs.Probs[0];
            int sk = br.Read(sp);
            Console.WriteLine($"  Skip flag for 16x16: {sk}");

            // tx_size between skip and y_mode (no-op unless TxModeSelect).
            // Top-left no-neighbor tx_size_context = 1.
            var txMode16 = decoder.LastCompressedResult!.TxMode;
            var maxTxSize16 = Vp9MaxTxSize.ForBlockSize(Vp9BlockSize.Block16x16);
            Span<byte> txProbs16 = stackalloc byte[2]
            {
                decoder.LastCompressedState!.TxModeProbs.P16x16[1, 0],
                decoder.LastCompressedState!.TxModeProbs.P16x16[1, 1],
            };
            var txSize16 = Vp9TxSizeDecoder.ReadTxSize(txMode16, maxTxSize16, br, txProbs16);
            Console.WriteLine($"  tx_mode = {txMode16}, tx_size for 16x16: {txSize16}");

            var ym = Vp9IntraModeTree.Decode(p => br.Read(p),
                Vp9IntraModeProbs.KeyframeYProbs(Vp9IntraMode.DcPred, Vp9IntraMode.DcPred));
            Console.WriteLine($"  Intra Y mode for 16x16: {ym}");
            if (sk == 0)
            {
                // Decode 16x16 Y coefficients. For DcPred + Tx16x16 the
                // tx_type is DCT_DCT, which uses the Default scan order.
                var coefBlock = new short[256];
                int eob = Vp9BlockCoefDecoder.DecodeBlockCoefficients(
                    readBit: p => br.Read(p),
                    txSize: Vp9TxSize.Tx16x16,
                    scanType: Vp9ScanType.Default,
                    planeType: Vp9BlockCoefDecoder.PlaneType.Y,
                    refType: Vp9BlockCoefDecoder.RefType.Intra,
                    block: coefBlock,
                    isHighBitDepth: false,
                    coefProbs: decoder.LastCompressedState!.CoefProbs[(int)Vp9TxSize.Tx16x16]);
                Console.WriteLine($"  Coefficients decoded, EOB at scan position {eob}");
                // Print all non-zero coefficients in raster order.
                Console.Write("    Non-zero raster positions:");
                int nonZeroCount = 0;
                for (int i = 0; i < 256; i++)
                {
                    if (coefBlock[i] != 0)
                    {
                        Console.Write($" [{i}]={coefBlock[i]}");
                        if (++nonZeroCount >= 10) break;
                    }
                }
                Console.WriteLine();

                // Dequantize using Y plane quantizer at frame qindex.
                int qindex = decoder.LastCompleteHeader!.Quantization.BaseQIndex;
                var planeQuant = Vp9Dequantizer.PlaneQuantizer(qindex, dcDelta: 0, acDelta: 0);
                Console.WriteLine($"  qindex={qindex}, Y plane Dc={planeQuant.Dc}, Ac={planeQuant.Ac}");
                Vp9Dequantizer.DequantizeInPlace(coefBlock, planeQuant);
                Console.Write($"  After dequant: first 4 (scan order): ");
                for (int i = 0; i < 4; i++) Console.Write($"{coefBlock[i]} ");
                Console.WriteLine();

                // Reorder from scan-order to raster-order. The coef
                // decoder writes to raster positions via scan[c]; verify
                // by reading scan position 0 = raster position 0 (the DC
                // is always at the top-left corner regardless of scan).
                // Actually the decoder already writes to raster positions
                // (block[scan[c]] = value), so coefBlock IS raster order.

                // Predict (DcPred no neighbors -> 128 everywhere).
                const int N = 16;
                var ab = new byte[N * 2];
                var lf = new byte[N];
                Array.Fill(ab, (byte)127);
                Array.Fill(lf, (byte)129);
                var dst16 = new byte[N * N];
                Vp9IntraPredictor.Predict(ym, 127, ab, lf, dst16, N, stride: N);

                // iDCT (DCT_DCT for Tx16x16 + DcPred) + add to prediction.
                Vp9InverseTransform.Apply(
                    Vp9TxType.DctDct, Vp9TxSize.Tx16x16,
                    coefBlock, dst16, stride: N);

                int mn = 255, mx = 0, sm = 0;
                for (int i = 0; i < dst16.Length; i++)
                {
                    if (dst16[i] < mn) mn = dst16[i];
                    if (dst16[i] > mx) mx = dst16[i];
                    sm += dst16[i];
                }
                Console.WriteLine($"  Reconstructed 16x16 Y: min={mn}, max={mx}, mean={sm/dst16.Length}");
                Console.WriteLine($"  Top-left 4x4 sample:");
                for (int row = 0; row < 4; row++)
                {
                    Console.Write("    ");
                    for (int col = 0; col < 4; col++)
                        Console.Write($"{dst16[row * N + col],4}");
                    Console.WriteLine();
                }
                Console.WriteLine($"  ffmpeg ground truth for top-left 16x16 Y: range 67-75.");
            }
            if (sk == 1)
            {
                const int N = 16;
                var ab = new byte[N * 2];
                var lf = new byte[N];
                Array.Fill(ab, (byte)127);
                Array.Fill(lf, (byte)129);
                var dst16 = new byte[N * N];
                Vp9IntraPredictor.Predict(ym, 127, ab, lf, dst16, N, stride: N);
                int mn = 255, mx = 0, sm = 0;
                for (int i = 0; i < dst16.Length; i++)
                {
                    if (dst16[i] < mn) mn = dst16[i];
                    if (dst16[i] > mx) mx = dst16[i];
                    sm += dst16[i];
                }
                Console.WriteLine($"  Predicted 16x16 Y: min={mn}, max={mx}, mean={sm/dst16.Length}");
            }
        }
        else if (partition16 == Vp9PartitionType.Split)
        {
            var probs8 = Vp9PartitionProbs.KeyframeProbs(sizeIdx: 0, splitState: 0);
            var partition8 = Vp9PartitionTree.Decode(p => br.Read(p), probs8);
            Console.WriteLine($"  Top-left 8x8 partition:  {partition8}");
        }
    }
}

internal sealed class IgnoreSink : IVideoFrameSink
{
    public ValueTask OnFrameAsync(
        ReadOnlyMemory<byte> y, int ys,
        ReadOnlyMemory<byte> u, int us,
        ReadOnlyMemory<byte> v, int vs,
        long pts) => ValueTask.CompletedTask;
}
