# SpawnDev.Codecs CHANGELOG

## Unreleased

Initial development. Project still in pre-release. See README.md for the
working feature matrix.

### 2026-04-27/28 - all encoders + decoders working through public APIs

**Bug fixes (4 critical, 2 follow-on):**

- VP8 encoder: fix Y2 PLANE_TYPE (was using 3 = Y_WITH_DC, should be 1 = Y2) + add reconstruction write-back so multi-MB frames decode pixel-exact through ffmpeg (commit `beae150`).
- Vorbis encoder: amplitude bug closed - MDCT 4/N normalization moved from decoder to encoder (libvorbis convention) + residue codebook anchored at exactly 0 to eliminate ±half-step quantization noise. Single-tone test now matches source amplitude (peak 0.34 vs 0.30, RMS 0.124 vs 0.122) (commit `14ebe2e`).
- AV1 encoder: dav1d MSAC compatibility fix - 4 cumulative bugs closed (EOB token off-by-one, wrong txsize_log2_minus4 table, freelance txb_skip/dc_sign context formulas, missing qctx threading). 16x16 flat Y=128 now reconstructs exact through libdav1d (commit `5ce6c38`).
- VP9 walker + encoder: per-plane ENTROPY_CONTEXT propagation - was always passing ctx=0 for the first scan position, causing block (0,1) onward to mis-decode; libvpx passes the combined neighbor context. After fix, 32x32 multi-block decodes byte-exact vs ffmpeg (commit `3267c69`).
- AV1 decoder pipeline: 6 walker gaps closed (zigzag scan tables direction, qctx already done by encoder fix, CFL alpha magnitudes from libaom CDFs, directional intra modes via z1/z2/z3 from reconintra.c, tx_type via intra_ext_tx CDF, 32x32 + 64x64 inverse DCT 1D primitives). Walker no longer hits NotImplementedException; pixel-mean drift vs ffmpeg remains a separate slice (commit absorbed into `2a7a63b`).
- VP9 tile_info: min/max log2 formulas were transposed vs spec sec 6.2.14 - the encoder skipped the increment bit while the spec-compliant decoder still expected it; ffmpeg rejected every keyframe wider than 320px. After fix, 16x16 through 1920x1088 all decode through ffmpeg native VP9 (commit `be10e55`).

**Public API surface wirings:**

- `Vp8Decoder.DecodeFrameAsync` now routes keyframes through `Vp8KeyframeWalker` (was a `NotImplementedException` stub) (commit `087c99e`).
- `Vp9Decoder.DecodeFrameAsync` and `Av1Decoder.DecodeFrameAsync` now call their walkers and emit real YUV planes; placeholder mid-gray fallback only when the walker can't handle a frame (inter, etc.) (commit `2814322`).

**Verification + benchmark coverage:**

- `verify_all_codecs.cs` extended with VP9 multi-block, AV1 libdav1d, Vp8/Vp9/Av1Decoder API sections - 11/11 sections pass.
- `all_codecs_working_demo.cs` 14/14 entries pass.
- `benchmark_all_codecs.cs`, `benchmark_vs_ffmpeg.cs`, `benchmark_video_psnr.cs`, `benchmark_audio_quality.cs`, `benchmark_bbb_transcode.cs` - new unified benchmark suite.
- `bbb_transcode_artifacts.cs` produces VLC-playable mp4/ogg/flac/opus from TJ's BBB FullHD source.

**Blazor WASM:**

- `/transcode` page added: encode + decode all 3 video codecs in browser, render YUV->RGBA on HTMLCanvas (commit `d107dd2`).

**More fixes (post-overnight):**

- VP8 encoder: validate `baseQIndex` 0..127 at the API boundary - 7-bit field per RFC 6386 sec 9.6, larger values silently wrapped causing decoder/encoder quantizer mismatch (Q=150 PSNR collapsed to 8.76 dB) (commit `179d56a`).
- Vorbis encoder: fit floor 1 curve to per-block spectrum envelope. Encoder was forcing floor to a constant ~0.94 and asking the residue book to carry the entire dynamic range alone; real audio bins peaked at ~0.005 and rounded to zero. Per-block adaptive floor + 1024-entry residue book. BBB SNR 0.35 dB -> 35.71 dB (now beating ffmpeg's libvorbis at 20.88 dB) (commit `c67d8ec`).
- AV1 encoder + walker: 4 multi-block bugs closed - mi_cols formula was `>>3` should be `>>2` (mi units are 4 px not 8); min_log2_tiles formula used `64` should use `max_tile_area_sb = 2304`; `disable_cdf_update` was 0 should be 1; `GatherVertAlike` / `GatherHorzAlike` were 50/50 stubs replaced with libaom prob-summing formulas; `DecodePartition` now handles all four `has_rows / has_cols` cases per `read_partition`. libdav1d acceptance: 16x16 only -> ALL sizes 16x16..256x256 + 1920x1072 flat input (commit `8a43b8f`).

**Known remaining gaps:**

- AV1 multi-block with content variation: a separate bug fires when scan order has non-skip → skip block transitions. All-skip multi-block frames work; constant-content frames work to 1920x1072. BBB content (varied) still has 56/60 frames rejected by libdav1d. Distinct from the bugs closed in `8a43b8f`.
- VP8/VP9 inter frames + loop filter still NotImplementedException.



### Video codec foundations

**AV1 ENCODER FRAMING (BIT-EXACT validated against libaom-av1 + ffmpeg + libdav1d)**

- `Av1ObuWriter.EmitObu(...)` - emits AV1 OBU header bytes + extension + LEB128 size
- `Av1SequenceHeaderWriter.EmitPayload(Av1SequenceHeaderConfig)` - 28 fields, BIT-EXACT vs libaom-av1's BBB SH (14/14 bytes)
- `Av1FrameHeaderWriter.EmitPayload(Av1FrameHeaderConfig, Av1SequenceHeader)` - 16 fields, prefix through allow_intrabc
- `Av1BitWriter` - MSB-first bit packer
- `Av1IvfRemuxer.RemuxToBytes(...)` - high-level IVF round-trip remux (BBB 77,725/77,725 bytes BIT-EXACT)
- `Av1IvfRemuxer.RemuxToBytesWithShSubstitution(...)` - SH-substitution remux (BBB byte-identical with config-driven SH)
- Closed-loop bridges: `Av1SequenceHeaderConfig.FromHeader(sh)` + `Av1FrameHeaderConfig.FromHeader(fh)`

**AV1 DECODER pipeline (parser side)**

- `Av1ObuParser`, `Av1SequenceHeaderParser`, `Av1FrameHeaderParser` (16 fields)
- `Av1Decoder` exposes `LastSequenceHeader`, `LastFrameHeader`, `LastFrameObuCounts`, `CumulativeObuCounts`, `CumulativeFrameTypeCounts`, `ShowExistingFrameCount`, `TotalTemporalUnits`
- Placeholder pixels until block decode wires up

**VP9 ENCODER FRAMING (BIT-EXACT validated against ffmpeg)**

- `Vp9SuperframeWriter.Emit(IReadOnlyList<byte[]>)` - VP9 spec Annex B.1 packer (300 BBB packets BIT-EXACT round-trip)
- `Vp9IvfRemuxer.RemuxToBytes(...)` - IVF round-trip remux (byte-identical to source)
- `Vp9FrameHeaderWriter` - uncompressed header prefix (full header writer pending)

**VP9 DECODER pipeline**

- Superframe parser, complete uncompressed header parser, compressed header parser
- `Vp9Decoder` exposes `LastFrameHeader`, `LastCompleteHeader`, `LastCompressedResult`, `LastCompressedState`, `LastTileGroup`, `CumulativeFrameTypeCounts`, `ShowExistingFrameCount`, `TotalCodedFrames`, `TotalVisibleFrames`
- Placeholder pixels until block decode wires up
- ~58 slices of block decode foundation (intra prediction kernels, iDCT family, MV decode chain)

**Consumer APIs**

- `Av1StreamAnalyzer.Analyze(ivfBytes)` + `Vp9StreamAnalyzer.Analyze(packets)` - high-level introspection
- `Av1StreamSummary.ToReport()` + `Vp9StreamSummary.ToReport()` - human-readable reports
- `Av1StreamValidator.Validate(...)` + `Vp9StreamValidator.Validate(...)` - bitstream QA
- `IvfDetector.IsIvf(bytes)` + `IvfDetector.DetectCodec(bytes)` - format detection

**Containers**

- IVF reader + writer (32-byte DKIF header + per-frame size/pts)
- Matroska / WebM reader (via SpawnDev.EBML 3.0.0)
- Ogg reader (RFC 3533)
- RIFF/WAVE + AIFF reader/writer

**Audio codecs**

- FLAC: encoder + decoder, ffmpeg cross-validated bit-exact, competitive with libFLAC (20× better on constants, 24% better on linear ramps)
- Opus SILK decode + Opus-in-Ogg packager
- Opus CELT decode WORKING via Concentus 2.2.2 (BSD-3) backbone, bit-exact across 6 ILGPU backends
- Opus encoder WORKING via Concentus 2.2.2 backbone: mono + stereo, 8/16/24/48 kHz, 2.5 / 5 / 10 / 20 ms frames, VoIP / Audio / RestrictedLowDelay applications, encode + decode round-trip across all 6 ILGPU backends (126 tests)
- Vorbis: structural decoder + minimum-viable encoder, AMPLITUDE-CORRECT as of 2026-04-27. MDCT normalisation follows libvorbis convention (4/N on encoder forward, unscaled inverse). Residue codebook anchors entry N/2 at exactly 0 so noise-gated bins decode silently. ffmpeg-decoded our-ogg lands within 1% RMS / 12% peak of source amplitude (pre-fix was "deafening"); our-decoded libvorbis-ogg lands within 5% RMS of source (pre-fix was 600x too quiet).

**Cross-backend coverage**

- 510+ tests pass on every ILGPU backend: CPU, CUDA, OpenCL, WebGPU, WebGL, Wasm
- All AV1 + VP9 + IVF + FLAC writer/parser/analyzer/validator surfaces cross-backend green

### Known issues

- VP9 decoder block-level pixel chain composes end-to-end on real BBB tile bytes but predicted values don't match ffmpeg ground truth. Bit-position drift somewhere upstream in partition / mode info chain. See `Plans/PLAN-vp9-block-decode-wiring.md` for the diagnostic plan.
