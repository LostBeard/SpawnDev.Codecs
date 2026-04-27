# PLAN: VP9 block-level decode wiring

**Status:** chain composes, pixel values don't match ffmpeg ground truth yet. **2026-04-26 update: missing `tx_size` read identified as primary suspect.**
**Created:** 2026-04-25 (Tuvok)
**Updated:** 2026-04-26 (Tuvok) - cleared 5 of 6 original suspects, narrowed to one root cause.

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
- **`Vp9TxSizeDecoder.ReadTxSize`** - mirror of libvpx read_tx_size + read_selected_tx_size (THIS PRIMITIVE EXISTS BUT IS NOT BEING CALLED FROM THE DEMO CHAIN - SEE BELOW)

Every primitive has unit tests + cross-backend coverage (CPU/CUDA/OpenCL/WebGPU/WebGL/Wasm).

## Original symptom

`vp9_first_partition.cs` demo wires the chain end-to-end on BBB.webm's first frame and gets:

  64x64: Split, 32x32: Split, 16x16: None (leaf)
  Skip flag: 1 (all-zero residual)
  Y mode: DcPred
  Predicted 16x16: all 128

But ffmpeg's actual decode shows top-left 16x16 has values 67-75, not 128.

## 2026-04-26 root cause: missing tx_size read between skip and y_mode

**libvpx `read_intra_frame_mode_info` reads symbols in this exact order:**

```c
mi->segment_id = read_intra_segment_id(...);   // skipped for BBB (segmentation off)
mi->skip       = read_skip(...);
mi->tx_size    = read_tx_size(cm, xd, 1, r);   // ← THE MISSING STEP
mi->mode       = read_intra_mode(r, get_y_mode_probs(...));
```

Source: `vp9/decoder/vp9_decodemv.c` `read_intra_frame_mode_info`, verified via raw GitHub fetch 2026-04-26.

The `1` argument to `read_tx_size` is `allow_select`. The function reads bits ONLY when `tx_mode == TX_MODE_SELECT`; otherwise it's a no-op returning a fixed size. **For TxModeSelect-encoded content (which is the common case for libvpx output, including BBB), this read consumes 1-3 bits between the skip flag and the Y mode read.**

`vp9_first_partition.cs` lines 99-110 do not call this read. So the bool decoder position drifts by 1-3 bits at the y_mode read for any TxModeSelect-mode frame, which propagates to every downstream symbol.

### Fix recipe (apply next session, verify via PMT)

For the 16x16 leaf path in `vp9_first_partition.cs` after the skip read (line 102), before the y_mode read (line 109), insert:

```csharp
// libvpx read_intra_frame_mode_info reads tx_size BEFORE y_mode when
// tx_mode allows selection. Without this read, the bool decoder
// position drifts and every downstream symbol is wrong.
var txMode = decoder.LastCompressedResult!.TxMode;
var maxTxSize = Vp9MaxTxSize.ForBlockSize(Vp9BlockSize.Block16x16);  // = Tx16x16
// tx_size_probs row for ctx=0 (no above + left tx_size context).
// P16x16 = compressed-header-parsed tx_size_probs for 16x16 max blocks.
var txSizeProbs = decoder.LastCompressedState!.TxModeProbs.P16x16.AsSpan(0 /*ctx*/, 2);
var txSize = Vp9TxSizeDecoder.ReadTxSize(txMode, maxTxSize, br, txSizeProbs);
Console.WriteLine($"  tx_size for 16x16: {txSize}");
```

Mirror the same pattern for the 32x32 leaf path (line 117) using `maxTxSize = Tx32x32` and `P32x32` with a 3-entry slice.

The actual `tx_size_probs` field name on `LastCompressedState` may differ; verify by reading `Vp9CompressedHeaderState.cs` and `Vp9TxModeProbs.cs`. If `LastCompressedResult.TxMode` is NOT TxModeSelect, ReadTxSize will return without consuming bits, and the chain runs as before - but for BBB's keyframe, expect TxModeSelect.

### How to verify the fix worked

Run the demo. After the fix:
- The bool decoder position at y_mode read should now be correct.
- Y mode should likely change from DcPred to something else (since the prior value was decoded from the wrong bits).
- The predicted block pixels should fall in ffmpeg's ground-truth range (top-left 16x16: 67-75, top-left 32x32: 57-112).

If pixels match ffmpeg: bug closed. If not: drift is still upstream OR the prior decisions (partition tree at 64x64, 32x32) are also being affected by some prior missing read. That would be an unlikely-but-possible second bug.

## Suspects from 2026-04-25 - status update

| # | Original Suspect | Status |
|---|------------------|--------|
| 1 | Partition probability context for top-left 64x64 | **CLEARED 2026-04-26.** libvpx `dec_partition_plane_context` formula = `(left*2 + above) + bsl * PARTITION_PLOFFSET` with both=0 for missing neighbors → ctx=12 → flat offset 36-38 → bytes `{174, 35, 49}`. Matches our `KeyframeProbs(3, 0)` exactly. |
| 2 | Missing pre-partition seg_id read | Eliminated 2026-04-25. BBB has segmentation_enabled=false. |
| 3 | Bool decoder init sentinel bit | Eliminated 2026-04-25. Verified value is 0 (no throw). |
| 4 | Mode info read ordering (skip vs y_mode) | **REFINED 2026-04-26.** The order skip-then-y_mode IS correct. The actual missing piece is the `tx_size` read between them when `tx_mode == TxModeSelect`. See root cause above. |
| 5 | Y mode neighbor context default = DcPred for missing | Plan-stated, not directly cited from libvpx in 2026-04-25 note. NOT verified from libvpx source this session, but `Vp9KfYModeProbsTests.Vp9KfYModeProbs_DriveProbsIntoIntraModeTree_DecodesDcPredOnFirstZeroBit` end-to-end-tests the (DcPred, DcPred) path, and the tables are correct. If the tx_size fix doesn't close the gap, this becomes the next target. |
| 6 | kf_y_mode_prob layout `[above][left][prob]` row-major | **CLEARED 2026-04-26.** Already covered by 8 existing tests in `CodecsTestBase.Vp9KfYModeProbsTests.cs`: pinned values for (DcPred,DcPred), (TmPred,TmPred), (D45Pred,HPred), full-domain helper coverage, libvpx vp9_kf_y_mode_prob[0][0] = `{137, 30, 42, 148, 151, 207, 70, 52, 91}` matches our table exactly. Cross-validated against libvpx vp9_entropymode.c via raw GitHub fetch. |

## Verification approach for what's left

After the tx_size fix lands and pixels match ffmpeg for the top-left block:

1. Wire the coefficient decoder (`Vp9BlockCoefDecoder` is shipped, just needs to be called per block in raster order, gated on `skip == 0`). For BBB top-left where skip=1, the residual is zero, but for non-skip blocks coefficient decode is required.
2. Wire dequantization (libvpx tables + per-frame qindex math).
3. Wire the full block walker that recurses through partition tree.

Estimated effort to first-frame BIT-EXACT YUV after this session's findings: ~3-5 days, down from the original ~1 week, because the bisection search just collapsed.

After that, inter-frame decode (motion compensation, ref frame pool) is another ~2-3 weeks.

## Why ship the framing first

The encoder framing layer is COMPLETE today (BIT-EXACT validated through ffmpeg+dav1d for AV1, ffmpeg for VP9). Consumer-facing analyzers + validators are shipped. So:

- A library consumer can READ AV1+VP9 metadata + remux containers today
- A library consumer who wants to write a custom AV1 encoder has the framing layer + writers ready - they just need an entropy coder + transforms + mode decision
- A library consumer who wants to DECODE pixels still needs to wait for the block walker, but tx_size fix lights the path

The decoder pixel work is genuinely valuable but it's the harder, longer-tail part. The framing work unblocks more consumers immediately.

## References

- libvpx: `vp9/decoder/vp9_decodemv.c` `read_intra_frame_mode_info` (skip then tx_size then y_mode order)
- libvpx: `vp9/decoder/vp9_decodemv.c` `read_tx_size` + `read_selected_tx_size` (mirrored in `Vp9TxSizeDecoder`)
- libvpx: `vp9/decoder/vp9_decodeframe.c` `dec_partition_plane_context` (ctx formula verified 2026-04-26)
- libvpx: `vp9/common/vp9_entropymode.c` `vp9_kf_partition_probs` + `vp9_kf_y_mode_prob` (table values verified 2026-04-26)
- VP9 spec: sec 6.4 (Tile data syntax), sec 6.4.13 (read_tx_size)
- ffmpeg: libvpx-decoded YUV is the ground truth for cross-validation
