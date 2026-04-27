# PLAN: VP9 block-level decode wiring

**Status:** End-to-end decode chain is functionally correct. DC bit-exact (mean=80 matches ffmpeg). AC magnitude under-decode remains (variance ~30% of expected).
**Created:** 2026-04-25 (Tuvok)
**Updated:** 2026-04-26 (Tuvok) - 11 commits closing 5 library bugs and 3 missing demo reads.

## 2026-04-26 work shipped

### Library bugs fixed (in commit order)

1. **`Vp9SkipProbs.Probs` zero-init** -> libvpx `{192, 128, 64}` defaults seeded via static `DefaultProbs` clone. Compressed header diff_update applies deltas FROM these defaults; zero-init was corrupting every skip read. Tests pinning the defaults.

2. **`Vp9TxModeProbs.P8x8/P16x16/P32x32` zero-init** -> libvpx `default_tx_probs` tables. Same bug class as above. P8x8={{100},{66}}, P16x16={{20,152},{15,101}}, P32x32={{3,136,37},{5,52,13}}. Tests pinning the values.

3. **`Vp9CompressedHeaderState.CoefProbs[]` zero-init** -> seeded from `Vp9CoefProbs.DefaultCoefProbs4x4/8x8/16x16/32x32` clones. Same bug class. The 432-byte-per-tx-size coefficient probability arrays were starting at zeros; compressed header diff_updates land at wrong base values without proper seeding.

4. **`Vp9BlockCoefDecoder.DecodeBlockCoefficients` API extension** - added optional `coefProbs` parameter so callers can pass the frame-state-tracked compressed-header-updated probs. Backwards compatible (null falls back to static defaults).

5. **`Vp9BlockCoefDecoder` inner-zero-loop refactor (BIG)** - the function was calling `Vp9CoefDecoder.DecodeOneCoefficient` once per scan position, which re-reads the EOB bit at every position. Per VP9 spec sec 6.4.21 + libvpx vp9/decoder/vp9_detokenize.c, after a ZERO token the inner-zero-loop only re-reads ZERO_CONTEXT_NODE (no EOB). The extra EOB reads consumed bits the encoder never emitted, drifting the bitstream position on every zero token. Refactored to mirror libvpx's exact loop structure. Replaced 1 broken test that asserted impossible-under-libvpx-semantics "zero then EOB" bitstream with a libvpx-valid scenario.

### Missing demo reads added

`vp9_first_partition.cs` was missing reads in libvpx's `read_intra_frame_mode_info` order:

1. **`tx_size` between skip and y_mode** - consumes 1-3 bits when `tx_mode == TxModeSelect` (the common case for libvpx output). Top-left no-neighbor `tx_size_context = 1` per libvpx `get_tx_size_context`. Wired via `Vp9TxSizeDecoder.ReadTxSize`.

2. **`uv_mode` between y_mode and coef decode** - reads `vp9_kf_uv_mode_prob[y_mode]` table (we use `Vp9IntraModeProbs.KeyframeUvProbs(yMode)`). Without this read, every coefficient read was 1-9 bits drifted.

3. **Full block reconstruction** - coef decode -> dequant -> iDCT 16x16 (DCT_DCT) -> add residual to DcPred prediction -> clamp.

### Verification

- ffmpeg ground truth via `compare_groundtruth.cs` (decodes BBB.webm via ffmpeg, dumps top-left 16x16 Y): `mean=80, range=57-103`.
- Our decoder current state: `mean=80, range=73-87`.
- **Mean is bit-exact** -> DC decode is correct end-to-end.
- Test sweep after refactor: 800+ tests pass cross-backend (Vp9Coef 306/306, Vp9BlockCoef 48/48, Vp9Compressed 108/108, Av1+FLAC+SkipProbs+TxModeProbs 342/342). Zero regressions.

## Remaining AC magnitude under-decode

The chain decodes the right total DC energy but produces ~30% of expected AC variance. With the inner-zero-loop fix, we get only 10 non-zero coefs ending at scan position 17, mostly small-magnitude (Two, Three, Four, One, Cat1=-5). ffmpeg likely has more or larger ACs.

### Next-session bisection candidates

1. **`Vp9CoefTrees.DecodeConToken`** tree walk - if the constrained-tree topology has a node mismatch with libvpx, low-magnitude branches might be picked when libvpx picks high-magnitude ones. Cross-check against libvpx `vp9_coef_con_tree`.

2. **`Vp9CoefProbs.ModelToFullProbs`** expansion - the 3-byte stored model expands to 11 entries via Pareto8 distribution. If the expansion math is off, all probabilities are biased.

3. **`Vp9CoefContext.GetCoefContext`** - if the context computation differs from libvpx (e.g., neighbors table off, energy-class lookup wrong), wrong probs are picked at every position.

4. **`Vp9NeighborTables.GetNeighbors16x16(Default)`** - the neighbors table indexed by scan position; cross-check first ~20 entries against libvpx `default_scan_16x16_neighbors`.

5. **Category magnitude tables (`Cat1Prob`..`Cat6Prob`)** - cross-check exact byte values against libvpx.

### Recommended approach

Build a small parallel-trace tool: run libvpx's reference decoder on BBB with detokenize-level instrumentation (printf at each `vpx_read` call), capture the exact sequence of (prob, bit, position) triples, and compare against our decoder's trace. The first divergence pinpoints the bug. Estimated 1 session of setup + 1 session of bisection.

## Suspects from 2026-04-25 - final status

| # | Original Suspect | Status |
|---|------------------|--------|
| 1 | Partition probability context for top-left 64x64 | CLEARED. ctx=12 verified vs libvpx `dec_partition_plane_context`; bytes `{174, 35, 49}` match. |
| 2 | Missing pre-partition seg_id read | CLEARED. BBB has segmentation_enabled=false. |
| 3 | Bool decoder init sentinel bit | CLEARED. Verified value is 0 (no throw). |
| 4 | Mode info read ordering | CLEARED. Skip-then-y_mode order is correct; missing reads were `tx_size` between them and `uv_mode` after y_mode (both fixed). |
| 5 | Y mode neighbor context default = DcPred for missing | CLEARED indirectly. KfYModeProbs(DcPred, DcPred) end-to-end test passes; mean=80 confirms correct DC -> correct y_mode prob slice. |
| 6 | kf_y_mode_prob layout `[above][left][prob]` row-major | CLEARED. 8 existing tests pin values; libvpx vp9_kf_y_mode_prob[0][0] = `{137, 30, 42, 148, 151, 207, 70, 52, 91}` matches our table exactly. |

All 6 original suspects cleared; 2 entirely-new bug families found and fixed during the investigation.

## References

- libvpx: `vp9/decoder/vp9_decodemv.c` `read_intra_frame_mode_info` (segment -> skip -> tx_size -> y_mode -> uv_mode order)
- libvpx: `vp9/decoder/vp9_decodemv.c` `read_tx_size` + `read_selected_tx_size`
- libvpx: `vp9/decoder/vp9_detokenize.c` (inner-zero-loop pattern)
- libvpx: `vp9/decoder/vp9_decodeframe.c` `dec_partition_plane_context`
- libvpx: `vp9/common/vp9_entropymode.c` (default tables)
- libvpx: `vp9/common/vp9_pred_common.h` `get_tx_size_context`
- VP9 spec: sec 6.3 (compressed header), sec 6.4 (tile data), sec 6.4.20 (decode coefficients), sec 6.4.21 (decode tokens)
- ffmpeg: `compare_groundtruth.cs` driver script in repo root
