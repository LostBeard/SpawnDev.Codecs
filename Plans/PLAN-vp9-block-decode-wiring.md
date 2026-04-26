# PLAN: VP9 block-level decode wiring

**Status:** chain composes, pixel values don't match ffmpeg ground truth yet.
**Created:** 2026-04-25 (Tuvok)

## What's already shipped

The primitives are individually correct and unit-tested:
- `Vp9BoolDecoder` - libvpx vpx_reader port, byte-aligned init + sentinel bit verified
- `Vp9PartitionTree.Decode(reader, probs)` - 4-leaf tree walk, matches libvpx vp9_partition_tree
- `Vp9PartitionProbs.KeyframeProbs / DefaultProbs` - kf_partition_probs + default_partition_probs tables
- `Vp9SkipProbs` - parsed from compressed header, exposed via `Vp9Decoder.LastCompressedState.SkipProbs`
- `Vp9IntraModeTree.Decode(reader, probs)` - 9-internal-node mode tree
- `Vp9IntraModeProbs.KeyframeYProbs(above, left)` - kf_y_mode_prob table indexed by neighbor modes
- `Vp9IntraPredictor.Predict(mode, topLeft, above, left, dst, n, stride)` - all 10 modes (DcPred, VPred, HPred, D45/63/117/135/153/207, TmPred)
- `Vp9IntraEdgeFill` - libvpx 127/129 out-of-frame conventions
- `Vp9InverseTransform.Apply` - iDCT / iADST / iHT family
- `Vp9IntraBlockDecode` - composes predict + iHT into a single block-level call

Every primitive has unit tests + cross-backend coverage (CPU/CUDA/OpenCL/WebGPU/WebGL/Wasm).

## What's NOT yet wired

The composition step that drives a real frame's tile data through the partition + mode info + coefficient decode chain. Tonight's `vp9_first_partition.cs` demo wired the chain end-to-end on BBB.webm's first frame and got:

  64x64: Split, 32x32: Split, 16x16: None (leaf)
  Skip flag: 1 (all-zero residual)
  Y mode: DcPred
  Predicted 16x16: all 128

But ffmpeg's actual decode shows top-left 16x16 has values 67-75, not 128. So the chain has a bit-position drift somewhere.

## Likely cause - bit-position drift

The chain runs on real BBB tile bytes deterministically. Some upstream symbol read is consuming wrong bits. Candidates:

1. **Partition probability context** - my code uses `KeyframeProbs(sizeIdx=3, splitState=0)` for the top-left 64x64 SB. libvpx uses `partition_plane_context(...)` which gives ctx=12 for the (0,0) 64x64 case (= sizeIdx=3 * 4 + above_unsplit*2 + left_unsplit = 12). The two should match. But... need to verify.

2. **Missing pre-partition read** - VP9 might read OTHER bits before the partition. Investigated: segmentation seg_id_predicted (only when `seg.UpdateMap && seg.TemporalUpdate`). BBB has segmentation_enabled=false so this branch shouldn't fire. Verified by dumping `decoder.LastCompleteHeader.Segmentation`. So not seg_id.

3. **Bool decoder init** - Vp9BoolDecoder constructor reads a sentinel bit during init (per libvpx). Verified value is 0 (no throw). Looks right.

4. **Mode info ordering** - For a leaf intra block at keyframe: I read skip first, then Y mode. libvpx may read Y mode FIRST, then skip. Need to verify against vp9_decode_mb_mode_mv or equivalent.

5. **Y mode neighbor context** - Top-left block has no neighbors. libvpx uses `default_intra_mode = DcPred` for missing neighbors. My code uses `DcPred` for both above + left. Should match.

6. **kf_y_mode_prob layout** - libvpx's `vp9_kf_y_mode_probs[10][10][9]` is indexed by `[above][left][prob_index]`. My `Vp9IntraModeProbs.KeyframeYProbs(above, left)` should return the 9-element prob span. Need to verify the index math.

## Path to fix

**Option A (cheap)** - add debug output to my demo: print every bit the bool decoder reads, with the prob used. Then run libvpx with similar debug instrumentation, compare per-bit output to find where drift starts.

**Option B (expensive)** - read libvpx's vp9_decode_partition + vp9_decode_mode_info code very carefully and trace by hand against the demo. Match every read.

**Option C (pragmatic)** - implement a complete frame walker, drive all 60 IVF frames through, compare each YUV pixel to ffmpeg. Whatever doesn't match identifies the broken layer.

Option A is fastest if libvpx debug log can be obtained. Option C is the eventual destination anyway.

## Estimated effort

Once the partition + mode info chain is bit-exact, the rest of the block decoder (coefficient decode + dequantization + iHT + reconstruction) reuses primitives we already have. Conservative estimate:

- 1-2 days to find + fix the partition / mode info drift (Option A)
- 1-2 days to wire the coefficient decoder (Vp9BlockCoefDecoder is shipped, just needs to be called per block in raster order)
- 1 day to wire dequantization (libvpx tables + per-frame qindex math)
- 1 day to wire the full block walker that recurses through partition tree

So: ~1 week of focused work to get first-frame BIT-EXACT YUV decode.

After that, inter-frame decode (motion compensation, ref frame pool) is another ~2-3 weeks.

## Why ship the framing first

The encoder framing layer is COMPLETE today (BIT-EXACT validated through ffmpeg+dav1d for AV1, ffmpeg for VP9). Consumer-facing analyzers + validators are shipped. So:

- A library consumer can READ AV1+VP9 metadata + remux containers today
- A library consumer who wants to write a custom AV1 encoder has the framing layer + writers ready - they just need an entropy coder + transforms + mode decision
- A library consumer who wants to DECODE pixels still needs to wait for the block walker

The decoder pixel work is genuinely valuable but it's the harder, longer-tail part. The framing work unblocks more consumers immediately.

## References

- libvpx: vp9/decoder/vp9_decodeframe.c (decode_partition, decode_mb_mode_info)
- VP9 spec: sec 6.4 (Tile data syntax)
- ffmpeg: libvpx-decoded YUV is the ground truth for cross-validation
