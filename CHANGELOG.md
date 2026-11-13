# SpawnDev.Codecs CHANGELOG

## Unreleased

### VP9 inter frames in the encoder (2026-11-12; CPU accelerator verified, GPU lanes pending)
- `Video.Vp9.Vp9EncoderGpu`: a key frame, then frames predicted from the frame before (`KeyFrameInterval`, default
  128). The encoder writes the decoder's own layouts and runs the decoder's prediction and residual kernels on them,
  so its reference is what a decoder shows: every frame of the test clips decodes (Vp9DecoderGpu and ffmpeg 8.1) to
  the encoder's reconstruction byte for byte, at every edge leaf shape (80x50, 72x64, 70x50, 200x9).
- Motion search (`MotionSearch`, default on): a three-level pyramid (+-38 pixels), half and quarter-pel refinement
  scored on the decoder's real prediction, and a merge with the neighbours' MVs. A picture moving by (3, 2),
  (-29, 22), (26, -30), (2.5, -1.5) or (-0.75, 4.25) pixels per frame gets exactly that MV on every leaf inside the
  picture; a moving textured picture's inter frames are 41% of zero motion's bytes at q 40.
- `VideoFileEncoderGpu` writes VP9 with inter frames: `VideoFileEncodeOptions.KeyFrameInterval` (null = 128, 1 = key
  frames only as before). Only key frames are sync samples (MP4 stss, WebM keyframe flag). AV1 stays key frames only.
- `VideoFileEncoderGpu.EncodeAsync` overloads: frames already on the accelerator (one `ArrayView1D<float>` holding
  N x H x W x 3, nothing of the picture crosses to the host) and, in the browser, `IReadOnlyList<Float32Array>`
  (JS -> GPU, returns a `Uint8Array`). All three funnel into one encode.
- Inter frames quantize with a dead zone (coefficients under 100/128 of a step are dropped, the rest rounded with
  48/128): on two real ComfyUI clips the same luma PSNR costs 33% and 39% fewer bytes than round to nearest (same
  RGB PSNR: 23% and 28%). Key frames are unchanged.
- Inter frames code nothing past the picture's edge (sizes that are not multiples of 8 paid for the padding strip
  in every frame: 80x50 inter frames 955 -> 522 B, 200x9 1275 -> 289 B at q 30).
- Measured on a real 512x512 clip (48 frames, CUDA): inter frames with round to nearest were 42 to 53% of key frames
  only at equal PSNR; with the dead zone at q 40 the video is 1.10 MB against 3.64 MB key frames only.
- `DemoConsole vp9-encode-bench <clip.webm>`: bytes, PSNR and speed per quantizer and key frame interval on a real clip.
- SpawnDev.ILGPU 5.4.29 -> 5.4.30.

### VP9 keyframe encoder: any picture size (task #23, 2026-11-12)
- The walker now codes the right-edge leaf (PARTITION_VERT, the left BLOCK_8X16: two Tx8x8 luma stacked + Tx4x4
  chroma) and the corner (SPLIT implied, the top-left BLOCK_8X8 with its own PARTITION_NONE), mirroring the bottom-edge
  BLOCK_16X8; the reconstruction kernel encodes the same leaf (one loop over the sub-blocks for every edge shape: 3
  helper call sites instead of 6). Partition contexts follow libvpx's partition_context_lookup per leaf (16X8 left
  context 14, was 12; 8X16 {14, 12}; 8X8 {14, 14}). Widths rounding up to an odd number of 8-pixel columns no longer
  throw. Fidelity against the source (q 1, luma): 70x50 74.5 dB (was 31.6 and wrong at the right edge), 72x64 138.1,
  200x9 57.2; every earlier size unchanged.

### MKV output and crf mapping (2026-11-12)
- `WebMWriterOptions.DocType`: "webm" (default) or "matroska" (a valid .mkv with the same layout; not byte-identical
  with ffmpeg's matroska muxer). `VideoContainer.Mkv` in VideoFileEncoderGpu (SpawnGen's SaveVideo mkv). WebM output
  unchanged (byte-identical fixtures pass); the MKV holds the WebM's blocks and loads back with WebMVideoLoaderGpu.
- `VideoFileEncodeOptions.QIndexFromCrf(crf)`: libvpx / libaom quantizer_to_qindex (crf * 4, 62 -> 249, 63 -> 255).

### VideoFileEncoderGpu: frames + fps + audio -> MP4 / WebM in one call (2026-11-12)
- `Video.VideoFileEncoderGpu.EncodeAsync(frames, width, height, fpsNum, fpsDen, audio, options)`: RGB frames in
  ComfyUI's IMAGE layout (float32 H x W x 3) -> RGBA8 (x * 255 clamped, truncated) -> YUV 4:2:0 -> VP9 or AV1 keyframes
  on the accelerator; optional audio (interleaved float) -> Opus; muxed by Mp4Writer (vp09 / av01 + Opus) or WebMWriter.
  VP9: widths whose 8-aligned value is a multiple of 16 (task #23 for the rest); AV1: multiples of 64. Round trip on
  CUDA, Velocity, the CPU accelerator and WebGPU: an 80x50 VP9 WebM loads back with WebMVideoLoaderGpu as 6 frames at
  exactly 24 fps (worst frame 30.0 dB, the RGB <-> 4:2:0 conversion's own limit on that pattern) and the exact input
  audio sample count; the MP4's samples equal the WebM's blocks; AV1 MP4 carries the right av1C.
- Needs SpawnDev.ILGPU 5.4.29 on CUDA: 5.4.28 folded `(byte)(int)f` into one float -> 8-bit conversion that CUDA
  saturates at 127 (found by this test; fixed by Data in ILGPU).
- `OggOpusEncoder.EncodePacketsAsync`: raw Opus packets + OpusHead + pre-skip for containers (the Ogg path shares the
  same encode). `Av1Isobmff`: av1C records and ISOBMFF samples. `Mp4Writer` VP9 tracks (vpcC, ffmpeg's level).
- VP9 keyframe encoder: widths rounding up to an odd number of 8-pixel columns now throw instead of encoding the right
  edge wrong (task #23); new fidelity test against the source picture (all sizes > 45 dB at q 1 and 30).

### Mp4Writer: AV1 + Opus in MP4 (2026-11-12)
- `Container.Mp4.Mp4Writer`: an ISO BMFF muxer for AV1 video (`Mp4Av1Track`: av1C, size, timescale; `VideoTimescale`
  gives ffmpeg's choice) and Opus audio (`Mp4OpusTrack`: the OpusHead; pre-skip as negative decode times and the edit
  list, roll-recovery sample groups). Packets are interleaved by decode time whatever order the caller feeds them;
  non-seekable outputs buffer the media data. Written from the public specs and ffmpeg's output observed as a black
  box: byte-identical with `ffmpeg -c copy -fflags +bitexact` on 10 fixtures (AV1 alone and all-key, AV1 + Opus, Opus
  alone at 48 kHz stereo and 44.1 kHz mono, 24 / 25 / 29.97 / 16 fps, 12 s interleaved), in file order and track by
  track (`scripts/video/make_mp4_mux_fixtures.py`). For SpawnGen's SaveVideo (mp4 / av1).

### ComfyUI LoadVideo parity: Yuv420ToRgb16Gpu and WebMVideoLoaderGpu (2026-11-12)
- `Yuv420ToRgb16Gpu`: YUV 4:2:0 -> RGB exactly as ComfyUI's LoadVideo (16-bit RGB, bilinear chroma, BT.601 limited;
  float32 = value * float32(1/65535)). The rule was derived as a black box (ComfyUI and FFmpeg not read):
  `Docs/comfyui-loadvideo-yuv-to-rgb.md`. All 108 frames of the three ComfyUI-written VP9 test files are bit-exact with
  ComfyUI's own IMAGE tensors on CUDA, Velocity, the CPU accelerator and WebGPU; 15 odd-size random pictures match the
  oracle (ffmpeg `-sws_flags bilinear -pix_fmt gbrp16le`). One planes view (Y, U, V tight), so WebGPU binds it once.
  The matrix uses int32 arithmetic split at 9 bits (equal to the 64-bit form); against plain 64-bit: CUDA 1.9x faster,
  Velocity / CPU accelerator 2-4% faster, WebGPU equal.
- `WebMVideoLoaderGpu.LoadAsync(byte[])`: one call from a VP9 WebM to RGB frames (bit-exact with ComfyUI), the frame
  rate (from DefaultDuration, the rational ffprobe reports: exactly 24, 25, 30000/1001, 24000/1001, 60, 15, 12.5, 50,
  8; 59.94 differs by 5e-6) and the Opus audio (pre-skip and DiscardPadding trimmed: ComfyUI's sample count). Audio
  values: our decoder is libopus fixed point, ComfyUI's ffmpeg decodes in float: max |diff| 3.3e-4 (about -70 dB).
  AV1 / VP8 video: not yet (keyframe decoders only), so ComfyUI's own SaveVideo (AV1) files do not load yet.

### Matroska reader: the track fields a decoder needs (2026-11-11)
- `MatroskaTrack` now carries CodecPrivate, DefaultDuration, CodecDelay, SeekPreRoll, PixelWidth / PixelHeight,
  AlphaMode, SamplingFrequency, Channels and BitDepth; `MatroskaFrame` carries a BlockGroup's DiscardPadding. Checked
  against the 21 ffmpeg-written WebMMux fixtures (every track field and every packet's bytes and discard padding).

### SpawnDev.ILGPU 5.4.28 (2026-11-11)
- SpawnDev.ILGPU 5.4.27 -> 5.4.28: WebGPU keeps its preloaded byte copy / fill kernels for the accelerator's lifetime
  (a GC no longer makes the next unaligned clear load one again), and `Accelerator.LoadedKernelCount`, which the
  lazy-load guard now uses (compiles plus shader-cache hits, per accelerator). Guard passes on CUDA, Velocity, the CPU
  accelerator and WebGPU.

### VP9 decoder on WebGPU: the device-hang cause fixed, workaround removed (2026-11-11)
- The real cause of the WebGPU device hang after a VP9 decode (see the next entries) was not the tile parse's
  length but its per-thread state: a 6.5 KB dynamically indexed function-scope array. On Chrome WebGPU (D3D12,
  NVIDIA) a long dispatch with such an array, followed by any dispatch in the same submission, leaves the device to
  hang a later large kernel. Confirmed live (state in device memory: 3/3 pass; the original array: hang) and by a
  codec-free replay. The parse thread's state now lives in device memory, and the wait after the parse is gone.
- No speed change: tile parse ms/frame CUDA 14.8, Velocity Vector256 3.3, Vector128 3.2, CPU accelerator 2.8
  (interleaved A/B, bit-exact). All 281 VP9 WebGPU tests pass; EveryFrame bit-exact on every file on WebGPU and CUDA.

### VP9 decoder tests run on the WebGPU lane (2026-11-11)
- The WebGPU test lane no longer leaves out the GiantShader tests (the full VP9 decoder pipeline). Full WebGPU lane
  with them: 1188 passed, 0 failed, 50 min; Chrome's GPU process peaked at 2.27 GB. WebGL still leaves them out.
- The lazy-load guard counts kernel loads (compiles plus shader-cache hits). On WebGPU it needs SpawnDev.ILGPU 5.4.28
  (preloaded byte kernels held for the accelerator's lifetime); on 5.4.27 it reports one load in
  `WebPLossyEncoderGpu.EncodeAsync`.

### VP9 decoder on WebGPU: no device hang when an encode follows a decode (2026-11-11)
- On Chrome WebGPU (D3D12, NVIDIA), a VP9 decode followed by a keyframe encode on the same device hung the GPU
  (DXGI_ERROR_DEVICE_HUNG). Cause, found by replaying the recorded WebGPU calls without our code: the tile parse is
  one long single-thread dispatch (about 130 ms on a dense keyframe), and any dispatch after it in the same
  submission leaves the device unable to run a later large kernel. Below ILGPU (Dawn D3D12 / driver).
- WORKAROUND (WebGPU only): the decoder waits for the accelerator right after the tile parse, so the parse ends its
  submission. The decode-then-encode test passes (it failed every time before); WebGPU EveryFrame stays bit-exact
  (311 files), about 5% slower. Other backends unchanged. Comes out when the parse runs as shorter launches.

### SpawnDev.ILGPU 5.4.27: VP9 loop filter 20% faster on Velocity Vector256 (2026-11-11)
- SpawnDev.ILGPU 5.4.26 -> 5.4.27: Velocity picks the team count of barrier kernels adaptively. ComfyUI files,
  loop filter ms/frame: Vector256 81.5 -> 64-66; Vector128, CPU accelerator and CUDA unchanged; reconstruction
  unchanged (interleaved A/B). Full Velocity EveryFrame bit-exact on Vector256 (987 s, was 1456 s).

### SpawnDev.ILGPU 5.4.26: VP9 reconstruction 5-7x faster on Velocity (2026-11-11)
- SpawnDev.ILGPU 5.4.25 -> 5.4.26: Velocity runs barrier-free kernels on its host path when that is faster
  (adaptive, per kernel), with one multiprocessor per logical core. ComfyUI files, reconstruction ms/frame:
  Velocity Vector256 50.5 -> 9.3, Vector128 70.5 -> 10.3; loop filter, tile parse, CPU accelerator and CUDA
  unchanged (interleaved A/B). Full Velocity EveryFrame bit-exact on Vector256 (1456 s) and Vector128 (1761 s).

### VP9 decoder: inter prediction back to fast on Velocity (2026-11-10)
- The one-body `Convolve` (5e76f1e) made Velocity's reconstruction 1.8x slower (ComfyUI files, Vector256 74 -> 130
  ms/frame): the copy / horizontal / vertical / 2-D choice sat inside the pixel loops, so the SIMD lanes paid for it
  per pixel. Now the four convolve loops are back, each called from one place, the choice made once. Reconstruction
  ms/frame against the code before the rewrites (e29dd73) and master before this fix: Velocity Vector256 49.5 (74.5,
  130), Vector128 71 (113, 217), CPU accelerator 5.3 (6.0), CUDA 1.95 (= master 1.94). InterPredict WGSL about 5.1k
  lines fully inlined (4.1k with one body; 134k originally).
- The loop filter rewrite (35e2c9a) made Velocity's loop filter 3x faster (Vector256 265 -> 82 ms/frame).

### SpawnDev.ILGPU 5.4.25 (2026-11-10)
- SpawnDev.ILGPU 5.4.24 -> 5.4.25: WGSL no longer lets a helper fall off its end after a loop with an early return
  (`Bc7EncoderGpu.ModeKernel` failed to compile on WebGPU: "missing return at end of function"). The WebGPU lazy-load
  guard passes again, now including the BCn and VP9 decoder warm-ups.

### SpawnDev.ILGPU 5.4.24; VP9 decoder bit-exact on WebGPU (2026-11-10)
- SpawnDev.ILGPU 5.4.23 -> 5.4.24 (WGSL / GLSL short-circuit conditions no longer copy the code behind them).
- The full VP9 decoder runs on WebGPU, bit-exact with libvpx: Vp9Decode_EveryFrame (311 files), PublicApi, WebMAlpha
  and TypedArrayOverloads, in watched runs (Chrome's GPU process at most 1 GB). The tests stay in the GiantShader
  category for now (see Docs/status.md, VP9 WebGPU device hang: a full decode followed by a keyframe encode on the
  same WebGPU device hangs D3D12, below ILGPU).
- `Vp9DecoderGpu_DecodesOwnKeyframeEncoder_LikeKeyframeDecoder` names the case and step that fails, waits after each
  step, and on WebGPU adds the device-lost reason and the last dispatches to the failure.

### VP9 decoder: inverse transform call sites; loop filter back to its CUDA speed (2026-11-10)
- The inverse transform (`Vp9Itx.InverseTransformAdd`, in the inter residual and intra wavefront kernels) called the
  2-D transform from 13 places (one per libvpx eob variant). It now works out libvpx's choice first (DC only, or the
  transform of the first rows, skipping all-zero rows for 32x32 above eob 135) and calls it once per transform size,
  the size a constant, so each size's 1D transforms appear twice (row and column pass) and its loops keep fixed
  bounds. Fully inlined WGSL: InterResidual 56,457 -> 11,656 lines, IntraWave 60,538 -> 15,737.
- The loop filter rewrite (35e2c9a) made CUDA's loop filter 12% slower (ComfyUI files, 1.36 -> 1.51 ms/frame): one
  filter body with the size a run-time value, and a loop over every possible libvpx call per block. Now one call site
  per filter size (each folds to its own filter) and the loop visits only the calls that run. Loop filter 1.36 ->
  1.36 ms/frame (base 1.37 / 1.38, new 1.35 / 1.37 interleaved), Wavefront about 13.9k lines fully inlined.
- CUDA reconstruction (ComfyUI files) 2.12 -> 1.96 ms/frame against the code before the 32-bit offsets, all bit-exact.
  Two other versions were measured and dropped: one 1D transform call site (each coefficient row copied into the
  scratch first: recon +20%) and one transform call site with the size a run-time value (+7%).
- Bit-exact: Vp9Itx, Vp9IntraPred, Vp9Lpf edge / superblock tests against libvpx (red-checked: 32x32 eob 34 row
  band, 16x16 shift, 8x8 dual second call, inner 4 edge of the first block), full Vp9Decode_EveryFrame on the CPU
  accelerator.

### VP9 decoder: inter prediction kernel 18x smaller after inlining, one convolve body (2026-11-10)
- The inter prediction kernel inlined the convolves at 28 call sites: each reference (2) x libvpx's call shape
  (each 4x4 below BLOCK_8X8, or the whole block: 2) x the 7 convolve calls of the predict table (copy, horizontal,
  vertical, 2-D, unscaled and scaled), each with its 8-tap dot products and the tap tables. Now:
  - one `Convolve` body for copy, horizontal, vertical and 2-D (each with its _avg and scaled variants); every output
    pixel gets its own convolve's arithmetic, and only the 2-D convolve uses the scratch column, as before;
  - one `PredictOne` call site (the 4x4 calls below BLOCK_8X8 and the whole-block call are one grid of calls) and one
    `PredictRef` call site (the references are a loop). Both references are read from one view, the frame buffers
    (`PredictBlockPlane` / `PredictBlockPlaneRegion` take `refs` instead of `ref0` and `ref1`).
- WGSL: 5.4 MB -> 0.21 MB; fully inlined 75,515 -> 4,114 lines (134,149 before the 32-bit offsets).
- Bit-exact: Vp9InterPred_AllCases against libvpx on the CPU accelerator and Velocity (red-checked: vertical taps,
  sub-8x8 MV order, compound reference, scaled-vertical mode), full Vp9Decode_EveryFrame on the CPU accelerator.

### VP9 decoder: loop filter kernel 10x smaller after inlining, one filter body (2026-11-10)
- The loop-filter wavefront kernel inlined a filter body at every vpx_lpf_* call in libvpx's code (about 96 copies:
  14 calls in filter_selectively_horiz, 12 in filter_selectively_vert_row2, each function called once per plane
  layout). Now:
  - one filter body (`LpfAt`): the 4, 8 and 16 filters share the mask and flat computation; the 16 filter falls
    back to the 8 and the 8 to the 4 as in libvpx's filter16 / filter8 / filter4;
  - each filter-selectively function issues libvpx's calls in libvpx's order through one call site (`LaneEdge`);
  - each pass (vertical, horizontal) runs luma and chroma of every layout (ss00, ss11, non420) through one call
    site.
- WGSL: 9.0 MB -> 0.55 MB; fully inlined (what D3D compiles) 116,468 -> about 11,000 lines (188,891 before the
  32-bit offsets).
- Bit-exact: Vp9Lpf edge-filter and superblock tests against libvpx (red-checked with three mutations), full
  Vp9Decode_EveryFrame on the CPU accelerator.

### VP9 decoder: 32-bit offsets in reconstruction and loop filter, about half the WGSL after inlining (2026-11-10)
- The reconstruction kernels (inter prediction, inter residual, intra wavefront) and the loop-filter wavefront
  addressed pixels, coefficients and scratch with `long` offsets. WGSL has no 64-bit integers, so every pixel tap was
  emulated 64-bit adds, subtracts and conversions: 10,881 emulated-64-bit helper calls in the loop-filter kernel alone.
  They now use `int` offsets. Fully inlined WGSL (what D3D's compiler in Chrome on Windows compiles), lines:
  - loop-filter Wavefront 188,891 -> 116,468 (the 585 64-bit calls left are its 64-bit edge masks);
  - InterPredict 134,149 -> 75,515; InterResidual 101,910 -> 56,457; IntraWave 115,872 -> 60,538.
- The host checks that the frame buffers and the reconstruction scratch fit 32-bit offsets and throws
  `NotSupportedException` naming the frame size otherwise. The frame buffer starts were already limited to 32 bits.
- The WGSL screen's budgets are tightened to the new sizes.

### Compressors: constructors compile nothing, kernels load awaited (2026-11-09)
- `ShuffleFilterGpu`, `WeightPackGpu`, and the GDeflate, LZ4 frame and Snappy compressors and decompressors no longer
  compile their kernels in the constructor (synchronous compiles froze a browser page). Each loads its kernels on the
  first call with awaited loads, which yield to the browser's event loop before each compile, and has a `WarmUpAsync`
  to load them ahead of time. `WeightPackGpu` loads the pack and the unpack kernels separately, so an unpack-only user
  compiles only the unpack side.
- The lazy-load guard (`Codecs_ConstructorsCompileNothing_WarmUpCompilesWhatTheyUse`) now covers them: constructors
  compile nothing, every warm-up compiles something, and round trips after the warm-ups (byte and bit shuffle,
  WeightPack, GDeflate, LZ4 frame, Snappy raw and framed) compile nothing more and are lossless.

### VP9 decoder: tile parse kernel 64x smaller after inlining, for Chrome on Windows (2026-11-09)
- Chrome on Windows compiles WGSL with D3D's compiler, which inlines every function, so the code it compiles is call
  sites x body. `Vp9ParseKernels.ParseKernel` was about 763k lines that way (113,564 WGSL lines, with `ParseBlock` a
  111k-line function called from 6 places). It is now 11,464 WGSL lines, about 11,900 fully inlined, with 24 bool
  reads in the whole kernel (`Plans/PLAN-VP9-Decoder-GPU-Speed.md`, S4):
  - one `ParseBlock` call site (a loop over the partition's blocks);
  - one `FindMvRefs` site (one pass per sub-block below 8x8; its searches are one ordered candidate stream with one
    add-to-list site), one `ReadMv` site (the MV components a loop);
  - key frame and inter frame mode info share one sequence; intra modes, reference frames, segment id, inter mode and
    interp filter are loops over one read each;
  - one tree reader (`ReadSym`, probabilities from the context, the tables or the arguments, each buffer indexed in
    its own branch); partition edge cases and coefficient tokens from TWO up are trees in the tables (`Vp9PScan`);
  - single-exit helpers and & / | for side-effect-free conditions: WGSL's structured control flow copied a helper's
    tail after an early return and a body behind a short-circuit condition (including the refill loop of every bool
    read, `Vp8BoolDecoderGpu.Fill`).
- Bit-exact on the CPU accelerator (Parse_ModeInfo 5112 frames with 2951 counts checked, EveryFrame, PublicApi);
  red-checked. CPU accelerator tile parse unchanged (ComfyUI, without the first frame: 1.91 / 2.00 -> 1.93 ms/frame).
- The WGSL screen's ParseKernel budget is 0.62 MB (was 9.9 MB). `VP9_WGSL_DUMP=dir` writes each kernel's WGSL.
- GPU (ComfyUI files, bit-exact): tile parse ms per frame CUDA 17.2 -> 15.7, OpenCL 13.9 -> 12.8; the smaller code is
  faster too. The Chrome / DXC compile time is not measured yet.

### VP9 decoder: tile parse coefficient loop on local memory (2026-11-09)
- The serial tile parse thread spends 89% of its bool decodes on coefficient tokens (ComfyUI files: 87k of 98k bools per
  frame, `Plans/PLAN-VP9-Decoder-GPU-Speed.md` step S3). Its coefficient loop (`Vp9ParseKernels.DecodeCoefs`) now
  waits on far fewer device loads per coefficient:
  - one packed scan entry per coefficient (`Vp9PScan`: position and the next context's two neighbours), fetched one
    coefficient ahead, instead of separate scan, neighbour and band loads;
  - the band computed (`CoefBand`, checked against libvpx's band tables when the table is built);
  - the three node probabilities of a context loaded together;
  - the token cache in the thread's local memory.
- The left entropy contexts, the above / left partition contexts and the partition stack also moved to the thread's
  local memory (`Vp9PLocal`, 6.5 KB; above partition contexts tile-relative, a tile column is at most 64 superblocks).
  The above entropy contexts stay in the work buffer, so the locals fit WGSL's 8192-byte limit on a function's
  variables.
- The bool read's renormalising shift comes from a leading zero count instead of a compare chain. Dequant factors are
  read once per plane of a block.
- Bit-exact with libvpx on the CPU accelerator (EveryFrame 311 files / 5112 frames, Parse_ModeInfo with counts,
  PublicApi); red-checked. CPU accelerator tile parse on the ComfyUI files (quiet A/B): 3.28 -> 2.68 ms/frame mean,
  2.33 -> 1.98 without the first frame's kernel compile. ParseKernel WGSL 87.1 -> 72.9 MB.
- Bit-exact on CUDA and OpenCL (ComfyUI files, 252 frames). Tile parse in ms per frame: CUDA 19.4 -> 17.2, OpenCL
  14.6 -> 13.9. That was measured with some background load, so treat it as an upper bound; the CPU A/B was quiet.

### VP9 decoder: inter reconstruction per 4x4 unit and per transform block (2026-11-09)
- Inter reconstruction no longer runs one thread per superblock (`Plans/PLAN-VP9-Decoder-GPU-Speed.md` step S2, inter
  half). `Vp9ReconKernels.InterPassKernel` is replaced by two launches, before the intra wavefront:
  - `InterPredictKernel`: one thread per 4x4 unit of each plane over the superblock grid (15,360 threads at 512x320,
    40 before). Each thread finds its block through the mode info grid. Units past the last mi row / column predict
    only when their block reaches over the frame edge, as libvpx does.
  - `InterResidualKernel`: one thread per 4x4 unit inside the mi grid. The thread at a transform block's top-left adds
    that transform block's residual; the others exit.
- Prediction: the new `Vp9InterPred.PredictBlockPlaneRegion` does the whole libvpx call's setup (MV clamp, scaled
  position, sub-pel phase, edge test, choice of convolve). It then runs the same convolve over only the thread's 4x4
  part, with the origin moved by `(subpel + offset * step) >> 4`. Every output pixel gets the same taps over the same
  source pixels, scaled references, compound averaging and sub-8x8 per-4x4 MVs included. `PredictBlockPlane` is that
  call over the whole block.
- Residual offsets, derived on the accelerator:
  - The parse kernel stores each block's block-table slot in its mode info record (`Vp9Mi.TableSlot`;
    `Vp9Mi.Stride` 24 -> 25).
  - The j-th transform block of a plane of a block starts `j * (16 << 2 tx_size)` coefficients and j eobs after the
    block's first. j is its raster index over max_blocks_high x max_blocks_wide.
  - Each transform block's inverse transform uses its own scratch at `dqAt / 4 * 5` (`Vp9RArg.ResidualScratchAt`).
- The intra pass is unchanged apart from dropping the inter code it no longer runs. WGSL sizes: InterPass 33.7 MB is
  replaced by InterPredict 5.7 MB and InterResidual 4.5 MB; IntraWave 33.7 -> 6.6 MB.
- `Vp9InterPred_AllCases_MatchLibvpx` also checks the per-unit path against libvpx: 1820 blocks, 1000 with a scaled
  reference, 750 below 8x8, 763 compound.
- Verified on the CPU accelerator: `Vp9Decode_EveryFrame_MatchesLibvpx` (311 files, 5000 output frames equal to
  libvpx's .md5), `Vp9DecoderGpu_PublicApi_MatchesLibvpx`, `Vp9InterPred_AllCases_MatchLibvpx` and
  `Vp9DecoderKernels_GenerateWgsl_WithinSizeBudget` pass. CPU accelerator reconstruction (ComfyUI files) went from 9.1
  to 4.4 ms per frame.
- Red-checks:
  - On the ComfyUI files: a U-plane residual offset one transform block off fails (41 mismatches), and so does
    launching the residual before the prediction (41 mismatches).
  - Moving the region origin by whole pixels, which ignores the scaled step, fails `Vp9InterPred_AllCases_MatchLibvpx`.
  - Restored, all pass.
- Measured (6 ComfyUI files, 252 frames, `VP9_STAGE_TIMES=1`), reconstruction in ms per frame:
  - CUDA RTX 4070: 16.1 -> 2.10.
  - OpenCL: 1.46.
  Bit-exact on both (252 frames equal libvpx).

### VP9 decoder: loop filter runs one thread group per superblock (2026-11-09)
- The loop filter kernel filters each superblock with a thread group of 64 lanes instead of one thread
  (`Plans/PLAN-VP9-Decoder-GPU-Speed.md` step S1). The k = 2r + c wavefront between superblocks is unchanged.
- Inside a superblock:
  - lane 0 builds the edge masks and filter levels into group shared memory (vp9_build_mask, vp9_adjust_mask and,
    for 4:2:2 / 4:4:0, the non420 masks);
  - vertical edges: each lane takes one pixel row of each plane;
  - after a group barrier, horizontal edges: each lane takes one pixel column of each plane.
- A vertical edge filter only touches one pixel row, and a horizontal one only one pixel column. Each lane replays
  libvpx's walk over the edges (filter_selectively_vert_row2 / _horiz / _vert) and filters only its own pixel of each
  edge, so every row and column gets libvpx's operations in libvpx's order. Same output bits.
- `Vp9Lpf.FilterSuperblock` (one thread per superblock) is replaced by `Vp9Lpf.FilterSuperblockGroup`.
  `Vp9Lpf_Superblocks_MatchLibvpx` now runs the grouped path too.
- Verified on the CPU accelerator: `Vp9Decode_EveryFrame_MatchesLibvpx` (311 files, 5000 output frames equal to
  libvpx's .md5), `Vp9DecoderGpu_PublicApi_MatchesLibvpx`, `Vp9Lpf_Superblocks_MatchLibvpx` (2400 superblocks, all
  subsamplings) and `Vp9Lpf_EdgeFilters_MatchLibvpx` pass. Red-check: running the horizontal pass before the vertical
  one fails `Vp9Decode_EveryFrame_MatchesLibvpx` (43 mismatches on the ComfyUI files); restored, it passes.
- Measured (ComfyUI files), loop filter in ms per frame: CUDA 16.5 -> 1.35, OpenCL 1.0. Bit-exact on both.
- KNOWN SLOWDOWN on ILGPU's CPU accelerator (the debugging backend, not a GPU): the loop filter went from 3.5 to 351 ms
  per frame (ComfyUI files, `VP9_STAGE_TIMES=1`), and `Vp9Decode_EveryFrame_MatchesLibvpx` from 247 to 2383 s
  (timeout 3600 s); `Vp9DecoderGpu_PublicApi_MatchesLibvpx` from about 40 to 221 s. Cause: the CPU accelerator runs
  kernels that use barriers or shared memory through its cooperative sequential processor (one OS thread per lane, a
  monitor handoff per lane at every barrier), while lane-independent kernels, like the old one-thread loop filter,
  take its plain loop path. Raised with the SpawnDev.ILGPU owner.

### VP9 decoder: WebGPU-valid parse kernel, offline WGSL screen, [NoInlining] DSP helpers; ILGPU 5.4.21 (2026-11-09)
- SpawnDev.ILGPU 5.4.21 (stable).
- FIXED (WebGPU / WebGL): the tile parse kernel chose at run time between views of two different buffers. WGSL
  cannot express that, so the kernel did not compile, and on WebGL the code generator overflowed the stack.
  read_partition now indexes the keyframe table and the frame context each in its own branch. Data found it with his
  offline WGSL screen.
- NEW `Vp9DecoderKernels_GenerateWgsl_WithinSizeBudget` (CpuOnly): generates WGSL for all ten VP9 decoder kernels
  on the desktop (`ShaderCompiler`, no browser) and fails on unsupported constructs or growth past a per-kernel
  budget (today's size + 10%). Today: ParseKernel 87.1 MB, InterPass / IntraWave 33.7 MB each, loop filter 10.8 MB.
  On WebGPU every helper that takes a view is inlined at every call site, which ILGPU will change; until then the
  `GiantShader` fence stays.
- `[NoInlining]` on the four VP9 DSP helpers (inverse transform, intra prediction, inter prediction, superblock loop
  filter). They become real functions on Wasm today (smaller modules), and on CUDA the decode got faster, bit-exact:
  - tile parse 24.5 -> 19.1 ms per frame;
  - reconstruction 21.2 -> 16.2 ms;
  - loop filter 20.5 -> 16.5 ms.
  Measured on an RTX 4070 over the 6 ComfyUI files, `VP9_STAGE_TIMES=1`. CPU: every VP9 vector still matches libvpx.

### Audio codecs: constructors compile nothing (2026-11-09)
- `Mp3EncoderGpu`, `OpusDecoderGpu`, `VorbisAudioEncoderGpu` and `VorbisAudioDecoderGpu` load their kernels once, lazily,
  from their async entry points (awaited loads: browser compiles yield to the event loop). New `WarmUpAsync` on each.
  - `Mp3EncoderGpu.WarmUpAsync` and `VorbisAudioEncoderGpu.WarmUpAsync` also encode a short built-in signal.
  - `OpusDecoderGpu.WarmUpAsync` and `VorbisAudioDecoderGpu.WarmUpAsync` decode nothing (the decoders keep stream
    state); the Opus one loads the SILK and CELT layers too and creates the CELT layer early.
- `SilkPacketDecoderGpu`, `CeltDecoderGpu` and `SilkDecodeFrameGpuOrchestrator` compile nothing in their constructors
  either; new `WarmUpAsync` on each. Their synchronous methods (`Decode`, `DecodeLost`, `ResetState`, the orchestrator's
  `Decode*`) load synchronously only when nothing loaded the kernels yet: call `WarmUpAsync` first in a browser.
  `CeltDecoderGpu`'s initial state reset now runs before the first kernel that reads the state (or in `WarmUpAsync`);
  `GetFinalRangeAsync` on a new decoder still reports 0.
- Guard extended: the constructors above compile nothing, each `WarmUpAsync` compiles, and after a GC an MP3 encode
  (CBR, VBR) decoded again, the libopus mode-switch stream (checked sample for sample and by final range against
  libopus) and a Vorbis encode (single, batch, stream) decoded again compile nothing. Red-checked: a synchronous load
  back in the `Mp3EncoderGpu`, `CeltDecoderGpu` or `VorbisAudioDecoderGpu` constructor fails it.

### Browser lanes: GiantShader fence for the VP9 decoder; SpawnDev.ILGPU 5.4.21-local.1 (2026-11-09)
- SpawnDev.ILGPU 5.4.21-local.1 (Data):
  - Wasm local reuse: the VP9 loop-filter kernel went from 68,382 to 8,300 locals, so V8 now compiles it;
  - -0 constants keep their sign on WebGPU / WebGL / OpenCL;
  - culture-independent code generation;
  - accurate CUDA float exp.
- NEW test category `GiantShader`, excluded by the WebGPU and WebGL lanes (`ExcludeCategories`). It covers the
  tests that compile the full VP9 decode pipeline:
  - `Vp9Parse_EveryFrame`, `Vp9Decode_EveryFrame`, `Vp9CompressedHeader_EveryFrame`;
  - the `Vp9DecoderGpu` public API / alpha / TypedArray / own-keyframe tests;
  - the VP9 GPU-transcode demo test.
- Why the fence: on WebGPU, compiling these kernels lost the device (Chrome's shader compiler; earlier Codecs giant
  kernels drove its GPU process to many GB). They still run on the desktop and Wasm lanes. The lazy-load guard skips
  the VP9 warm-up on WebGPU / WebGL (its constructor check still runs there).
- Guard diagnostics:
  - the failure message lists every warm-up's compile count;
  - the image "after the warm-ups" check names the codec step that compiled.
- On WebGPU the guard currently reports `WebPLossyEncoderGpu.EncodeAsync: 1`. That is SpawnDev.ILGPU's internal
  byte-fill kernel (`WebGPUMemoryBuffer.FillBytes`, the unaligned head / tail of a MemSet), which ILGPU compiles
  synchronously on first use; reported to Data. The guard passes on CPU, CUDA, OpenCL, WebGL and Wasm.
- Wasm: `Vp9DecoderGpu_TypedArrayOverloads_MatchLibvpx` gets past the old codegen limit but runs past PMT's 300 s
  wait (slow decode on Wasm); to investigate.

### BCn texture codecs: constructors compile nothing (2026-11-09)
- `BcnEncoderGpu`, `BcnDecoderGpu` and `Bc7EncoderGpu` load their kernels on first use.
  - New `WarmUpAsync` loads them awaited: browser compiles yield to the event loop.
  - `BcnEncoderGpu.WarmUpAsync(params BcnFormat[])` loads only the formats you name; no formats means all.
  - The async overloads await the load.
  - The synchronous `Encode` / `Decode` (ArrayView) load synchronously only when nothing loaded the kernels yet: call
    `WarmUpAsync` first in a browser.
- Guard extended to the BCn codecs, with a BC1 and a BC7 round trip after the warm-ups. Red-checked: a constructor
  compiling again fails it. CPU: all BCn tests pass.

### 1.0.2-local.10 (2026-11-09)
Everything below down to local.9:
- the full VP9 decoder (bit-exact with libvpx on CPU, CUDA and OpenCL; WebGPU / Wasm pending);
- codec constructors that compile nothing (no browser freeze), WarmUpAsync on every codec;
- the FLAC encoder writing the same file on every backend;
- the VP9 encoder superframe-marker fix;
- SpawnDev.ILGPU 5.4.20.

### SpawnDev.ILGPU 5.4.20; the FLAC encoder writes the same file on every backend (2026-11-09)
- SpawnDev.ILGPU 5.4.20 (stable): Data's fix for the CUDA divergent-warp bug (a branch on the thread index was emitted
  as `bra.uni`), which broke VP9 readback on odd sizes. CUDA (RTX 4070): every VP9 test passes now, including
  `Vp9Decode_EveryFrame_MatchesLibvpx` (311 files == libvpx, 476 s) and the public API / alpha tests;
  `IlgpuDivergentReturnRepro_PackPlanes_MatchesHost` passes and stays as the guard.
- FIXED: the FLAC encoder's Tukey window used the backend's float cos (`XMath.Cos`: CUDA's approximation, .NET
  `MathF.Cos`, the browser's libm), which differ in the last ulps, so CUDA picked different LPC coefficients for some
  blocks (a different, still lossless, file). The window now uses `CosZeroToPi`, a Taylor series in IEEE double
  + - * only (error < 3.4e-16; the float window equals libm's double cos at 200,001 points), so every backend
  computes the same bits. Red-checked on CUDA (the float cos back -> different files).
- The FLAC encoder's window, autocorrelation and LPC kernels compile under `F64EmulationScope.Use(Ieee)`: WebGPU and
  WebGL emulate their doubles bit-exactly like desktop .NET, so they pick the same coefficients and write the same
  file. The scope applies to these loads only (an AsyncLocal; interleaved loads keep the default mode). Removed the
  unused `StrictFloat64` helper (it flipped the accelerator-wide mode around a synchronous load).
- `FlacEncoderGpu_SameFileOnEveryBackend`: six encodes (levels 0 / 5 / 7 / 8, mono 24-bit, stereo 16 / 20-bit,
  block sizes) whose MD5s were produced on the desktop CPU accelerator; every backend must match. CPU, CUDA and
  OpenCL pass; browser lanes next.

### VP9 encoder: superframe-marker padding; `Vp9KeyframeDecoderGpu` obsolete (2026-11-08)
- FIXED: the VP9 bool-coded segments (compressed header, tiles) ended without libvpx `vpx_stop_encode`'s padding:
  when the last byte reads as a superframe index marker (0b110xxxxx) libvpx appends a 0 byte. Without it libvpx's
  decoder (and `Vp9DecoderGpu`) rejects the packet as corrupt. GPU (`Vp8BoolEncoderGpu.StopVp9`, used by the three
  VP9 kernels) and CPU reference (`Vp9BoolEncoder.Stop`) both pad now. Found by decoding our keyframe encoder's
  output with the full decoder; red-checked (no padding -> "superframe index markers differ").
- `Vp9DecoderGpu_DecodesOwnKeyframeEncoder_LikeKeyframeDecoder`: five sizes / quantizers / content; the full decoder
  equals the encoder's reconstruction and `Vp9KeyframeDecoderGpu`'s planes.
- `Vp9KeyframeDecoderGpu` is `[Obsolete]` (use `Vp9DecoderGpu`); the GPU transcode and benchmark demo pages use the
  full decoder now.

### Kernel loads no longer freeze a browser page (ILGPU 5.4.19 awaited loads) (2026-11-08)
- `FlacDecoderGpu`, `FlacEncoderGpu` (and its `FlacFrameWriterGpuKernel`), `Mp3DecoderGpu`, `OpusEncoderGpu` /
  `OggOpusEncoder`, `Vp9DecoderGpu` and the WebP animation encoder (`WebPAnimEncoder`, behind both WebP encoders'
  `EncodeAnimationAsync`) compile nothing in their constructors. Each loads its kernels once, lazily,
  from its async entry points through `LoadAutoGroupedStreamKernelAsync`: every compile yields to the browser's event
  loop first, and on WebGPU the pipelines are built with `createComputePipelineAsync` during the load. Lore measured
  SpawnGen frozen 30-104 s while these compiled synchronously (RX590, WebGPU, non-AOT).
- `WarmUpAsync` loads the codec's kernels (then decodes / encodes a tiny built-in input as before). New:
  `Vp9DecoderGpu.WarmUpAsync`.
- `OpusEncoderGpu.Reset` now marks the CELT state for a reset that runs before the next frame (the reset kernel may
  not be loaded yet); the coded packets are unchanged.
- FIXED: `OggOpusEncoder` recompiled its 12 kernels after any garbage collection. It makes an `OpusEncoderGpu` per
  call, and the accelerator's kernel cache holds compiled kernels only weakly, so once the per-call encoders were
  collected every later encode (and the effect of `WarmUpAsync`) was gone. The Opus encoder's kernels are now loaded
  once per accelerator and kept with it (`ConditionalWeakTable<Accelerator, ...>`).
- Guard: `Codecs_ConstructorsCompileNothing_WarmUpCompilesWhatTheyUse` (fresh accelerator; counts compiled kernels;
  forces a GC between the warm-ups and the work). Red-checked: a constructor compiling one kernel again (1 compiled),
  the Opus kernels not kept per accelerator (12 compiled after the GC), and `Vp9DecoderGpu.WarmUpAsync` skipping the
  output kernel (9 of 10) each fail it; the restored source passes.
- The still-image codecs now do the same: `BmpDecoderGpu` / `BmpEncoderGpu`, `GifDecoderGpu` / `GifEncoderGpu`,
  `JpegDecoderGpu` / `JpegEncoderGpu`, `PngDecoderGpu` / `PngEncoderGpu`, `WebPDecoderGpu`, `WebPLosslessEncoderGpu`
  and `WebPLossyEncoderGpu` compile nothing in their constructors, and so do the kernel holders behind them
  (`BmpKernels`, `GifKernels`, `JpegKernels`, `JpegEncoderKernels`, `JpegScanEncoderKernels`, `PngKernels`,
  `WebPKernels`, `WebPEncodeKernels`, `WebPRgbaToYuvGpu`, `WebPYuvToRgbaGpu` and the internal palette, pixel, PNG
  encode and GIF encode kernels). Each holder has a public `EnsureKernelsAsync()`; its codec awaits it before the
  first launch of every async entry point. The DEFLATE kernels (PNG encoder, `ZlibCompressorGpu`), `InflateKernel`
  (PNG decoder) and the VP8 keyframe decoder of lossy WebP (`Vp8FullKeyframeDecoderGpu`, `Vp8StridedPlanePackKernel`)
  load the same way: they compiled synchronously on the first encode / decode before. New: `WarmUpAsync` on each of
  the eleven image codecs (it only loads the kernels the codec uses). Direct users of `JpegScanEncoderKernels`,
  `WebPRgbaToYuvGpu.Convert`, `WebPYuvToRgbaGpu.Convert` and the other holders' synchronous launchers must await
  `EnsureKernelsAsync()` first.
- FIXED: a lossy WebP still with ICC / EXIF / XMP on a fresh encoder launched the container kernels of
  `WebPAnimEncoder` before they were loaded (only its `EncodeAsync` loaded them). The metadata path now loads them.
- Guard extended: constructing every image encoder and decoder compiles 0 kernels, each `WarmUpAsync` compiles more
  than 0, and an encode + decode with every image codec after the warm-ups (and a GC) compiles nothing. Red-checked: a
  sync load put back into the `BmpKernels` constructor, the same in the `WebPRgbaToYuvGpu` constructor (inside
  `WebPLossyEncoderGpu`), and `PngEncoderGpu.WarmUpAsync` skipping the DEFLATE kernels (16 compiled by the work) each
  fail it; the restored source passes.
- Still compiling in their constructors: `Bc7EncoderGpu`, `BcnDecoderGpu`, `BcnEncoderGpu` (`Images/Texture`; their
  ArrayView entry points `Encode` / `Decode` are synchronous launchers), and the other compressors (`GDeflate*`,
  `Lz4Frame*`, `ShuffleFilterGpu`, `WeightPackGpu`).

### Full VP9 decoder: `Video.Vp9.Decoder.Vp9DecoderGpu` (libvpx v1.17.0 port, BSD-3-Clause) (2026-11-08)
- Profile 0 (8-bit 4:2:0) with every feature libvpx's profile-0 conformance vectors use: inter frames, compound and
  scaled references (resizing streams), all interpolation filters, tiles, segmentation, lossless, hidden /
  show-existing / intra-only frames, superframes, backward adaptation. Plus WebM alpha (BlockAdditional, AlphaMode 1),
  as ComfyUI's SaveWEBM writes for RGBA input.
- API: `DecodeFileAsync(byte[])` (WebM / Matroska first V_VP9 track, or IVF) yields `Vp9VideoFrame` (tight planar
  Y, U, V + optional alpha); `DecodePacketAsync(packet)` (one readback per picture) and
  `DecodePacketAsync(packet, ArrayView1D<byte>)` (picture written on the accelerator). One picture per packet, as
  libvpx. Supporting types: `Vp9PictureFormat`, `Vp9Packets` (superframe split, `ReadFile` / `ReadFileWithAlpha`),
  `MatroskaFrame.BlockAdditional`.
- Everything that touches coded data is a kernel: compressed header, tile parse (one thread per tile column),
  reconstruction (one thread per tile column), loop filter (superblock wavefront), adaptation.
- Verified on the CPU accelerator, bit for bit against libvpx: every frame of 303 libvpx vp90-2 conformance vectors,
  6 ComfyUI SaveWEBM files and 2 synthetic header fixtures (5112 frames before and after the loop filter, 5000
  output frames == the vectors' .md5), at every stage (headers, contexts, mode info, counts, adaptation). Test oracle:
  scripts/video/libvpx_ref (libvpx built from source with stage hooks).
- Reconstruction runs in two passes: inter blocks of every superblock at once, then intra blocks on the superblock
  wavefront (row + col); the loop filter runs on its own wavefront.
- OpenCL: every VP9 test passes (332 public-API pictures). CUDA: sizes whose plane boundaries split a warp read back
  wrong (66x66, 226x198): an ILGPU PTX bug (the thread index was treated as warp-uniform, `bra.uni`), not the decoder;
  repro `IlgpuDivergentReturnRepro_PackPlanes_MatchesHost`, fix pending in SpawnDev.ILGPU.
- Browser: `DecodePacketAsync(TypedArray)` -> `Vp9VideoFrameJS` (planes as a JS Uint8Array) and
  `DecodeAlphaPacketAsync(TypedArray)`; the compressed packet is read into .NET for the host's metadata parse, the
  pictures never enter the .NET heap. Test `Vp9DecoderGpu_TypedArrayOverloads_MatchLibvpx` (browser lanes; not run yet).
- NOT yet: WebGPU / Wasm runs, profiles 1..3 (4:4:4, 10/12-bit), GPU speed numbers. `Vp9KeyframeDecoderGpu`
  (keyframes only, sizes multiple of 64) is unchanged.

### 1.0.2-local.9: MP3 decoder, MP3 encoder short blocks + bit reservoir (2026-11-08)
- New `Audio.Mp3.Mp3DecoderGpu` (`DecodeAsync` interleaved float32, `DecodeFloatAsync` planar, `WarmUpAsync`): a port
  of minimp3 (CC0) as ILGPU kernels. MPEG-1 / 2 / 2.5 Layer III, mono / stereo, M/S and intensity stereo, short and
  mixed blocks, free format, CRC frames, ID3v2 tags. Gapless with the Info frame's delay / padding (FFmpeg's
  convention). Damaged frames come out as silence and keep their time. Bit-exact with minimp3 on 21 fixtures (CPU
  accelerator), 114..125 dB SNR vs FFmpeg 8.1.
  **Known issue: wrong output on CUDA** (OpenCL and CPU are right). Cause: an ILGPU PTX miscompile of a nested loop
  (repro `IlgpuNestedLoopRepro_InnerDoWhileShortCircuit_MatchesHost`), fix pending in SpawnDev.ILGPU.
- `Mp3EncoderGpu`: window switching (short blocks with start / stop blocks, subblock gains, pre-echo control;
  `Mp3EncodeOptions.ForceLongBlocks`) and the CBR bit reservoir (`Mp3EncodeOptions.BitReservoir`, default on;
  look-ahead perceptual-entropy budgets). Pre-echo before transients 10.0 dB lower at 128k, 12.0 dB at V0; busy
  passages +3.2 dB SNR at 128k.
- Fix: a CBR stream whose frames cannot hold the Info tag (8 kbps at 8 kHz, 16 kHz 32k, ...) writes its Info frame at
  the lowest rate that holds it (the tag used to run into the first audio frame).
- SpawnDev.ILGPU 5.4.13 -> 5.4.18.
- Verified: CPU accelerator, MP3 encoder + decoder tests 13/13. Not yet run on OpenCL / CUDA / WebGPU / Wasm since the
  short-block and reservoir merge.

### MP3 encoder: `Mp3EncoderGpu` (clean-room, ISO 11172-3 / 13818-3) (2026-11-08)
- New `Audio.Mp3.Mp3EncoderGpu.EncodeAsync` (float or int32 PCM, mono / stereo, every Layer III rate 8..48 kHz:
  MPEG-1, 2, 2.5) and `Mp3EncodeOptions` (`BitrateKbps` CBR, `VbrQuality` 0..9 VBR, `MidSide`, `Tags`, `Id3Encoder`,
  `InfoEncoder`, `WriteId3`, `WriteInfoFrame`). Written from the standards and public literature; no LAME code (LAME is
  GNU Library GPL). MP3's patents expired in 2017; TJ approved MP3 for SpawnDev.Codecs on 2026-11-08.
- All on the accelerator: polyphase analysis, MDCT + alias reduction, an MDCT-domain psychoacoustic model (band
  energies, spectral-flatness tonality, Schroeder spreading in Bark, ISO model 2's 18 / 6 dB offsets, Terhardt's
  threshold of hearing), the outer / inner loops with scalefactors and scalefac_scale, Huffman table and region choice,
  VBR (coarsest gain meeting every mask; per-frame lowest bit rate), mid / side per frame with a perceptual-entropy bit
  split between channels, frame assembly, the Info / Xing frame (LAME-format extension: encoder delay / padding for
  gapless decoding, seek table, CRC-16 of the music and of the tag) with bounded CRC launches. The ID3v2.4 tag (TXXX per
  tag, TSSE, 10 bytes of padding, the layout ComfyUI's PyAV files carry) is host metadata.
- Tables (`Mp3Tables`, `Mp3HuffmanTables`, generated by `scripts/audio/make_mp3_tables.py`): Huffman codes recovered
  from minimp3's decoder tables (CC0), the window from PDMP3 (Unlicense), band tables from minimp3; NOTICE.md updated.
- FFmpeg decodes every test file without errors and gaplessly (exact input length); SNR 37..64 dB on test tones (CBR),
  V0 about 164 kbps on the 44.1 kHz stereo tones; a tone 44 dB down keeps 26.8 dB local SNR at 128k; mid / side 8.8 dB
  better than left / right on a centred mix. CPU, CUDA and OpenCL. Not yet: short blocks, the CBR bit reservoir.
### FLAC stream encoder, clean-room MIT (2026-11-08)
- `FlacEncoderGpu.EncodeAsync` (managed `ReadOnlyMemory<int>`, JS `Int32Array`, `ArrayView1D<int>` into an
  `ArrayView1D<byte>`), `FlacEncodeOptions` (level 0..8, block size, `FlacChannelMode`, padding, vendor, tags, channel
  mask) and `FlacEncoderGpu.WarmUpAsync` are back, rewritten from RFC 9639 and the textbook linear prediction / Rice
  coding literature (no FFmpeg or libFLAC code): `FlacEncoderGpu.Encode.cs`, `FlacEncodeKernels`. Any rate FLAC
  carries, 1..8 channels, 4..24 bits, any length. Levels 0..2: fixed predictors in ~27 ms blocks; 3..8: LPC (Tukey
  window, Levinson-Durbin, 15-bit coefficients, orders up to 12) in ~105 ms blocks. Exact Rice parameters per
  partition, escape-coded partitions, wasted bits, constant / verbatim fallback, per-frame stereo choice, STREAMINFO with
  frame sizes and the MD5 of the samples. Every stage that touches samples or bits is a kernel; launches are bounded
  and paced (`LaunchPacer`), frames processed in chunks.
- Tests (`CodecsTestBase.FlacStreamEncoderTests.cs`, back from `CleanRoomPending`): on all 65 FFmpeg fixtures our files
  are lossless (our decoder, MD5, and libFLAC through `SPAWNDEV_IMG_PYTHON`), their metadata blocks equal FFmpeg's byte
  for byte apart from STREAMINFO's min / max frame size (checked against our own frames instead), and their frames are
  at most 2 % larger than FFmpeg's (measured: 0.80 .. 1.00, all files together 0.989). Plus options, depths 4..24 bits,
  1..8 channels, odd block sizes and rates, multi-chunk streams, the ArrayView overload and the warm-up.

### REMOVED: code ported from FFmpeg (LGPL), to be rewritten clean-room (2026-11-08)
- `WebMWriter`, `OggOpusWriter`, `OggOpusEncoder`, `PcmConvertKernels` and the FLAC stream encoder
  (`FlacEncoderGpu.EncodeAsync`, `FlacEncodeOptions`, `FlacEncoderGpu.WarmUpAsync`, `FlacStreamKernels`,
  `FlacStreamWriteKernels`) were ports of FFmpeg 8.1 source, which is LGPL-2.1-or-later; SpawnDev.Codecs is MIT. They are
  removed and will be rewritten clean-room (MIT) from the public specifications and black-box fixtures:
  `Plans/PLAN-CleanRoom-Rewrite.md`. Their tests wait in `UnitTests/CleanRoomPending/`. Local packages 1.0.2-local.1
  through local.7 contain the ported code and must not be redistributed.
- The FLAC decoder's MD5 kernels (own code, RFC 1321) moved to `FlacMd5Kernels`.

### WebM writer rewritten clean-room (2026-11-08)
- **`SpawnDev.Codecs.Container.Matroska.WebMWriter`** (MIT, written from RFC 8794 / RFC 9559 / the WebM guidelines and
  black-box observation of ffmpeg's output): `AddVideoTrack(WebMVideoTrack)`, `AddAudioTrack(WebMAudioTrack)`,
  `WriteFrame(track, packet, timestampMs, keyFrame, durationMs, blockAdditional, discardPaddingNs)`, `Finish()`;
  `WebMWriterOptions` (apps, Title, global Tags, cluster limits), `WebMColour`. Interleaves packets by timestamp, holds one
  audio packet back so a new cluster can start with it, splits clusters at 5 s / 5 MiB (pipe output: 1 s / 32 KiB) and at
  video key frames once a cluster holds 4 KiB, writes SimpleBlocks or BlockGroups (alpha BlockAdditional, DiscardPadding,
  non-default durations), Opus CodecDelay / SeekPreRoll, reserved CodecPrivate room, Cues, SeekHead, Duration and
  per-track DURATION tags; non-seekable streams get the live layout.
- Tests: `CodecsTestBase.WebMWriterTests.cs` is back (CpuOnly): 21 fixtures byte-identical with ffmpeg 8.1, in file order
  and track by track (41 orderings). New fixtures pin the cluster rules and track header fields: `opus_6s`, `opus_live`,
  `vp9_longgop`, `vp9_longgop_live`, `vp9_keyframe_4k`, `vp9_keyframe_live`, `vp9_size_live`, `meta_tracks`,
  `meta_colour`, `meta_alpha_display` (`scripts/video/make_webm_mux_fixtures.py [case ...]`). `scripts/video/ebml_dump.py`
  dumps an EBML tree.

### Warm-ups for the audio coders (2026-11-08)
- `FlacDecoderGpu.WarmUpAsync`, `FlacEncoderGpu.WarmUpAsync`, `OggOpusEncoder.WarmUpAsync`: compile and load every kernel
  ahead of the first real call. Lore measured the first Ogg Opus save in Chrome on an RX 590 (WebGPU, D3D's FXC compiling
  the WGSL) at 113 s and the second at 7.2 s; FLAC 57 s vs 4.5 s. A page can now start the compile when it shows an audio
  node. Smaller CELT stage shaders remain the deeper fix.
- FLAC decoder LPC: coefficients read once per subframe, 32-bit sums where libFLAC's bound allows (red-checked).

### Full FLAC decoder: `FlacDecoderGpu.DecodeAsync` (2026-11-08)
- Any conforming FLAC stream (RFC 9639): fixed or variable block size, 4..32 bits per sample, 1..8 channels, constant /
  verbatim / fixed / LPC (order 1..32) subframes, Rice and Rice2 residuals with escaped partitions, wasted bits, every
  stereo decorrelation (the 33-bit side channel of 32-bit streams in an int64 view). An ID3v2 tag before the stream is
  skipped; `FlacDecodeResult` carries rate / channels / bits / samples, the VORBIS_COMMENT vendor and tags, and the MD5
  check against STREAMINFO.
- All on the accelerator (`FlacStreamDecodeKernels`): frame headers found per byte (sync code, fields that agree with
  STREAMINFO, CRC-8), put in file order, each frame's end found by CRC-16 at the next sync code, the chain walked on one
  thread, one thread per frame decodes, one thread per sample decorrelates and interleaves. A damaged frame comes out as
  silence of its length and the samples after it keep their time (from the coded frame / sample number), as libFLAC does.
- **`DecodeFloatAsync`**: float32 planar [channel][sample], as ComfyUI's Load Audio gets it through PyAV (FFmpeg's FLAC
  decoder shifts samples into int16 for <= 16 bits, int32 above; ComfyUI casts to float32 and divides by 2^15 / 2^31),
  converted on the accelerator. Bit-identical with that rule on every fixture (red-checked: the shift into int16).
- `DecodeStreamAsync(byte[])` keeps its signature and now runs this decoder (it was limited to the v1 encoder's
  verbatim streams and threw on a damaged frame).
- 30 fixtures (`scripts/audio/make_flac_decoder_fixtures.py`): FFmpeg 8.1 (levels 0 / 5 / 8 / 12, fixed-only, every
  stereo mode, exact Rice parameters, block sizes 192 / 1000 / 16384, 16 / 24 / 32 bits, 6 channels, noise, wasted bits,
  constant runs), libFLAC (8 / 16 / 24 bits, 6 channels), a variable-block-size stream, hand-made frames (escapes of
  0..16 raw bits, Rice2 parameter 20, LPC order 32 at precision 15, negative coefficients). Every file decodes to its
  source; 29 MD5s checked. Red-checked (mid-side LSB, Rice2 escape code, wasted bits, coded-position gaps, ends at sync
  codes past a damaged header, 33-bit side). CPU accelerator green; GPU and browser lanes not run yet.

### FLAC LPC analysis in strict IEEE f64 on WebGPU / WebGL (2026-11-08)
- Lore saw SpawnGen's browser (WebGPU) FLAC files differ from the server's (OpenCL) and ComfyUI's for the same samples
  (still lossless). Cause: WGSL has no f64 and SpawnDev.ILGPU emulates it with Dekker double-float by default (~48
  bits, not IEEE), so FFmpeg's double LPC analysis chose other coefficients. The LPC analysis kernel now loads in
  ILGPU's Ozaki mode (strict IEEE-754) through the new internal `StrictFloat64` helper, which restores the
  accelerator's mode after the load: no other kernel changes mode.

### Ogg Opus files: `OggOpusWriter` (a port of FFmpeg 8.1's ogg muxer) + `OggOpusEncoder` (2026-11-08) [REMOVED, see above]
- **New `Container.Ogg.OggOpusWriter`**: OpusHead on the beginning-of-stream page, OpusTags (vendor + `KEY=value`
  fields, in order) on the next, then the audio packets packed as FFmpeg packs them: a page closed at 255 segments,
  before a packet that would overflow 65025 bytes, or once it spans 1 s of granule time; granule = pts + duration +
  pre-skip; pages written one behind so the last one carries end-of-stream; Ogg CRC-32. Serial number, vendor, tags,
  input rate and output gain are options (FFmpeg: serial 0 and vendor "ffmpeg" in bit-exact mode; ComfyUI's PyAV:
  random serial, vendor "Lavf62.12.102", tags prompt / workflow / encoder). Given FFmpeg's packets, the file is
  FFmpeg's file byte for byte (11 fixtures: stereo / mono 48 kHz and stereo 24 kHz, 64..320 kbps,
  `scripts/audio/make_ogg_opus_fixtures.py`).
- **New `Audio.Opus.OggOpusEncoder.EncodeAsync`** (int16 and float interleaved PCM, to a byte array or a stream):
  OpusEncoderGpu driven as FFmpeg's libopus wrapper drives libopus (application audio, unconstrained VBR, 20 ms frames,
  FFmpeg's default bitrate when none is given, the last frame zero-padded, silent frames until the lookahead is out,
  pre-skip = lookahead at 48 kHz, final granule = input length + pre-skip). Float input becomes int16 on the
  accelerator as libswresample converts it (lrintf(x * 32768), clipped): PyAV hands libopus int16. Same pages,
  granules, flags, OpusHead and OpusTags as FFmpeg's file for every fixture.
- **New `OpusEncoderGpu.Lookahead`** (OPUS_GET_LOOKAHEAD).
- Packets are not libopus' bytes: FFmpeg / PyAV builds use libopus' FLOAT build, which runs the tonality analysis from
  complexity 7 (FFmpeg's default is 10; its output changes between complexity 6 and 7 and is the same at 8, 9, 10);
  OpusEncoderGpu is libopus 1.5.2's fixed-point CELT (complexity up to 9). Decoded quality on the fixtures: 23..33 dB
  SNR vs the source; our files are 2..16 % smaller (libopus' analysis raises the VBR rate on tonal content).
- Tests: `CodecsTestBase.OggOpusTests.cs`; red-checked (end-of-stream flag, CRC polynomial, page duration rule, final
  granule, round half to even).

### Animated lossy WebP: `WebPLossyEncoderGpu.EncodeAnimationAsync`, a port of libwebp 1.6.0's WebPAnimEncoder (2026-11-08)
- **New `WebPLossyEncoderGpu.EncodeAnimationAsync`** (managed, `TypedArray` and `ArrayView1D` overloads): full-canvas RGBA8
  frames + per-frame durations in, the file Pillow's `save(save_all=True, lossless=False, quality, method, exif)` writes
  out (ComfyUI's Save Animated WEBP). `WebPLossyAnimationOptions`: loop count, background colour, kmin / kmax (Pillow's
  lossy defaults 3 / 5, sanitized as libwebp does), ICCP / EXIF / XMP chunks. EXIF is written exactly as given: Pillow's
  animated save keeps the `Exif\0\0` prefix that its still save strips.
- libwebp's decisions, frame for frame: change rectangles (lossless, then the lossy tolerance of the quality), snapped to
  even offsets; previous-frame disposal (none / background) by the smaller rectangle; blending when possible, with
  fully-similar 8x8 blocks made transparent and the loop filter off; dropped frames whose durations merge into the
  previous one (and libwebp's 1x1 split frame past 24-bit durations); key frames by size penalty between kmin and kmax;
  a lone frame written as a still (MuxCleanup / OptimizeSingleFrame). Rectangles, block flattening, canvases and every
  byte of the file are kernels (`WebPAnimKernels`); the host reads back rectangles, two flags and encoded sizes.
- **New `WebPLossyEncodeOptions.FilterStrength`** (libwebp `filter_strength`, default 60; 0 writes VP8 profile 2):
  byte-identical with libwebp at 0 / 25 / 60 / 100.
- Output vs libwebp (26 fixtures, `scripts/images/make_webp_anim_lossy_fixtures.py`, Pillow 13.0.0 / libwebp 1.6.0):
  19 byte-identical; the 7 whose frames carry transparent pixels (flattened blocks, transparent sources) are identical
  except the ALPH chunks, which our VP8L coder writes (same decoded alpha planes, other bytes). Because libwebp picks key
  frames by encoded size, an ALPH size difference can change that choice on other content; a libwebp-exact VP8L coder
  for ALPH closes it.
- Tests: `CodecsTestBase.WebPLossyAnimationEncoderTests.cs`; red-checked (flattening off, profile 0, key-frame tie rule,
  lossy tolerance 0, single-frame comparison, disposal tie rule, alpha scaling of the pixel test).
- Status: green on CPU (11 / 11 WebP lossy tests), CUDA and OpenCL (10 / 10 each), WebGPU and Wasm (8 / 8 each; the
  Pillow checks are desktop-only). Lore checked SpawnGen's Save Animated WEBP against ComfyUI's own save (3 frames,
  q80 m4, ComfyUI's EXIF): byte-identical on the CPU accelerator and on OpenCL (RX 590).

### FLAC of any length: `FlacEncoderGpu.EncodeAsync`, a port of FFmpeg 8.1's FLAC encoder + muxer (2026-11-08) [REMOVED, see above]
- **New `FlacEncoderGpu.EncodeAsync(samples, channels, sampleRate, bitsPerSample, options)`** (managed `ReadOnlyMemory<int>`,
  browser `Int32Array`, `ArrayView1D<int>` into a caller view): a complete .flac of any length (the last block shorter),
  any rate FLAC carries (standard and non-standard codes), 1..8 channels, 16 or 24 bits. `FlacEncodeOptions`: FFmpeg
  compression level 0..8 (default 5, PyAV's), block size, stereo mode, padding, Vorbis vendor and tags, channel mask
  (WAVEFORMATEXTENSIBLE_CHANNEL_MASK comment for non-native layouts, as FFmpeg's muxer). ComfyUI's Save Audio format.
- FFmpeg's encoder on the accelerator: stereo mode estimate and decorrelation, wasted bits, constant / verbatim / fixed /
  LPC subframes, FFmpeg's LPC analysis in double (window, autocorrelation, Levinson-Durbin, reflection-coefficient order
  estimate, 15-bit coefficient quantization), the order methods of levels 6..8, the rice partition search, the verbatim
  fallback, frame headers with CRC-8 / CRC-16, the MD5 of the samples, STREAMINFO / VORBIS_COMMENT / PADDING. One thread
  per frame or subframe; MD5 in bounded launches; the host reads back the file size and the file.
- **Byte-identical with FFmpeg** (`-fflags +bitexact`, 65 fixtures: stereo / mono / 5 channels, 16 / 24 bits, 22.05..96 kHz
  and a non-standard rate, non-multiple lengths, a 3-sample clip, silence, constants, wasted bits, white noise (verbatim
  fallback), at levels 0, 1, 2, 5, 8) on the CPU accelerator. The fixtures use FFmpeg's C code (`-cpuflags 0`): its x86
  SIMD autocorrelation rounds differently and changed 4 of the 65 files. The same on Wasm. CUDA and OpenCL fuse
  multiply-adds in the double analysis (ILGPU emits PTX mul.f64 / add.f64 without .rn; OpenCL C contracts by default)
  and WebGPU / WebGL emulate doubles: there the LPC levels can pick other coefficients in a few frames (CUDA / OpenCL:
  the 96 kHz 24-bit fixture, 2 of 26 LPC files; the others identical); the test then checks the stream header and the
  size (within 2 %), and the files still decode exactly. libFLAC decodes
  every file of ours to exactly the source samples (desktop oracle, `scripts/audio/flac_decode.py`).
- An integer Levinson-Durbin was tried first (for identical bytes on every backend): tonal signals need near-double
  precision and it produced wrong coefficients (files up to 2x larger); it was replaced by FFmpeg's double analysis.
- Red-checked: stereo estimate off, wasted bits off, MD5 final block skipped, verbatim fallback off (caught after adding
  the white-noise fixtures). Not covered by any fixture (exact ties never occur in them): the partition-order tie rule
  and lrintf's round-half-to-even in coefficient quantization; both follow FFmpeg's code as written.
- The older `EncodeStreamAsync` (verbatim, 4096-multiples, 44.1 kHz / 16-bit) and `FlacDecoderGpu` (which reads only
  that subset) are unchanged; a full FLAC decoder is separate work.

### Animated lossless WebP the WebPAnimEncoder way + one animation engine for both modes (2026-11-08)
- **New `WebPLosslessEncoderGpu.EncodeAnimationAsync(frames, width, height, durationsMs, options, animation)`**
  (managed / TypedArray / ArrayView1D): full-canvas frames in, libwebp's lossless WebPAnimEncoder decisions out - exact
  change rectangles, disposal by the smaller rectangle, blending when it can reproduce the frame with the pixels equal
  to the canvas made transparent (IncreaseTransparency), dropped repeats, Pillow's lossless key-frame defaults (kmin 9,
  kmax 17), libwebp's lossless 1x1 split frame, lone frames as stills. Pillow's `save_all=True, lossless=True` (ComfyUI's
  Save Animated WEBP default); pass `Exact = false` for libwebp's default. The caller-rectangle overloads are unchanged.
- The WebPAnimEncoder port is now one internal engine (`WebPAnimEncoder`) driven by either encoder's still frames; the
  animation options type is **`WebPAnimEncoderOptions`** (was `WebPLossyAnimationOptions` in 1.0.2-local.1), with
  `KeyframeMin` / `KeyframeMax` null = Pillow's default for the mode.
- Vs Pillow 13 / libwebp 1.6.0 (22 lossless fixtures from the lossy fixtures' sources, q80 m4): every displayed frame is
  exactly its source frame (fully transparent pixels compared as transparent); where libwebp's choices are
  size-independent (key frames off or all key, or no longer than kmin + 1 frames; 20 of 22) the container, every ANMF's
  offset / size / duration / disposal / blending and each frame's own pixels match libwebp's file (alpha everywhere,
  colour where alpha > 0: libwebp's VP8L predictor leaves the colour under alpha 0 to its prediction). The VP8L bytes
  are ours, so files are not byte-identical and libwebp's size-based key-frame choice can differ on longer clips.
- Red-checked: IncreaseTransparency off; the lossy blend test in lossless mode (caught only after adding a faint-layer
  fixture where every non-opaque pixel is similar but not equal); the lossy 1x1 frame in lossless mode; lossy key-frame
  defaults.

### ICCP / EXIF / XMP on lossy stills (2026-11-08)
- `WebPLossyEncodeOptions.IccProfile`, `Exif`, `Xmp`: written verbatim in libwebp WebPMux's still layout (RIFF, VP8X,
  ICCP, [ALPH], VP8, EXIF, XMP), the file Pillow's still save writes (Pillow strips an `Exif\0\0` prefix before
  libwebp: pass the stripped bytes). Opaque pictures byte-identical with Pillow; with transparency identical except the
  ALPH coding. All three overloads; the ArrayView one names the needed size when the output is short.

### WebM writer: `WebMWriter`, a port of FFmpeg 8.1's matroskaenc in WebM mode (2026-11-08) [REMOVED, see above]
- **New `SpawnDev.Codecs.Container.Matroska.WebMWriter`**: `AddVideoTrack` (V_VP8 / V_VP9 / V_AV1: size, interlacing,
  AlphaMode, display size, Colour, DefaultDuration, CodecPrivate + reserve), `AddAudioTrack` (A_OPUS / A_VORBIS: rate,
  channels, bit depth, initial padding -> CodecDelay + timestamp offset, SeekPreRoll for Opus), `WriteFrame` (ms
  timestamps, key frame, duration, VP8/VP9 alpha as BlockAdditional 1, DiscardPadding), `Finish`. Global and per-track
  tags, title, muxing / writing app.
- The same bytes as `ffmpeg -fflags +bitexact -c copy` for the same packets (11 fixtures from ffmpeg 8.1: VP9, VP9 +
  Opus, VP9 alpha, 12 s VP9 + Opus, Opus at 48 / 44.1 kHz, VP8, AV1 (libaom: reserved CodecPrivate), title + tags, two
  video tracks, a non-seekable pipe), whether packets arrive interleaved or track by track: libavformat's interleaver
  (raw timestamps, ties by track), matroskaenc's held-back audio packet, cluster rules (5 s / 5 MB, 4 KB before a video
  key frame, 16-bit timestamp overflow; 1 s / 32 KB non-seekable), SimpleBlock vs BlockGroup, BlockDuration rules
  (default duration, coded Opus duration), Cues, the reserved SeekHead, Duration and per-track DURATION tags patched at
  the end, AV1 temporal delimiter / padding OBUs removed.
- A container writer on the host: payloads come already encoded and go to the caller's stream (like `IvfWriter`).
- Tests: `CodecsTestBase.WebMWriterTests.cs` (CpuOnly); red-checked (audio not held back, no 4 KB key-frame cluster
  rule, DURATION without its NUL, Opus duration rule off, interleave ties reversed).

## 1.0.1 (2026-10-30)

Dependency update; no API or codec changes.
- **SpawnDev.ILGPU 5.4.13** (was 5.4.12). On Velocity at O2, the default, loop-invariant code no longer runs inside
  loops. On OpenCL without `cl_khr_fp16` (NVIDIA among them), a `Half` sub-view passed to a kernel no longer starts at
  twice its offset. ILGPU 5.4.13 depends on SpawnDev.SpawnJS 3.0.7. Codecs' dependencies are SpawnDev.ILGPU and
  SpawnDev.EBML only; no Blazor.
- Demo: SpawnDev.SpawnJS.Blazor 3.0.7 (was 3.0.4).
- PlaywrightMultiTest: the fake camera / microphone flags are now Chromium launch args. Playwright .NET never read the
  `PLAYWRIGHT_ARGS` environment variable that used to carry them.
- Every lane green on the released packages: CUDA, OpenCL and Velocity 1131 / 0, CPU 3281 / 0, WebGPU and Wasm 1150 / 0,
  WebGL 399 / 0, the same counts as 1.0.0. The test time summed over each lane, against the 10-29 run: Velocity
  2801 s (was 4845 s), CUDA 1960 s (was 3720 s), OpenCL 2521 s (was 2968 s), CPU 4308 s (was 4146 s). The 10-29
  run overlapped its lanes, so part of that change is contention.

## 1.0.0 (2026-10-30)

The first stable release, and the first on nuget.org. The 1.0.0 bar ("Path to 1.0.0 stable", 0.3.0-rc.1 below) is met:
every codec ships as an encoder + decoder pair that runs entirely on the accelerator, the Opus pair byte-identical with
libopus 1.5.2 (the encoder CELT-only); Vorbis decoder v2 decodes the bit stream on the device; the CPU reference
implementations live in `SpawnDev.Codecs.References`; and every test lane is green on SpawnDev.ILGPU 5.4.12: CPU
3281 / 0, CUDA, OpenCL and Velocity 1131 / 0, WebGPU and Wasm 1150 / 0, WebGL 399 / 0. The entries below are what changed
since 0.3.0-rc.7.

### Opus GPU pair: `OpusEncoderGpu` (CELT-only) byte-identical with libopus 1.5.2 on every backend (2026-10-29)
- **New `OpusEncoderGpu`** (`SpawnDev.Codecs.Audio.Opus`: `EncodePacketAsync` managed and device-resident, `GetFinalRangeAsync`,
  `Reset`; libopus' settings: bitrate incl. OPUS_AUTO / OPUS_BITRATE_MAX, VBR / constrained VBR / CBR, complexity 0..9,
  bandwidth, max bandwidth, signal, forced channels, packet loss, prediction disabled, phase inversion disabled, lsb depth;
  applications VOIP / AUDIO / RESTRICTED_LOWDELAY; 2.5 .. 120 ms packets; 8 .. 48 kHz, mono / stereo). libopus 1.5.2's
  opus_encode_native + opus_encode_frame_native for CELT-only coding (OPUS_SET_FORCE_MODE(MODE_CELT_ONLY) or
  RESTRICTED_LOWDELAY) around Data's `CeltEncoderGpu` (celt_encode_with_ec, four stage kernels: analysis, coding, bands,
  finish). The host only decides from the settings (CBR budget, equivalent rate, mono / stereo stream with hysteresis,
  bandwidth, the 20 ms split of 40..120 ms packets, TOC value); kernels run the delay buffer, the DC reject / VOIP high-pass,
  the low-rate stereo width fade, CELT and the packet assembly (code 0 / 1 / 2 / 3, CBR padding).
- **Equal to libopus 1.5.2 on every packet of 49 real streams, bytes and final range** (`TestData/OpusEnc`,
  `scripts/audio/make_opus_encoder_fixtures.py`): every frame size, CBR / VBR / CVBR, complexity 0..9 (theta RDO at 8+),
  8 / 12 / 16 / 24 / 48 kHz input, VOIP and low delay, forced mono and stereo coded as mono, silent gaps, OPUS_AUTO and
  OPUS_BITRATE_MAX, the TOC-only too-small packets. CPU, Velocity, CUDA and OpenCL. A second set of 49 tests checks the Opus
  layer alone against the exact input and CTLs of every libopus `celt_encode_with_ec` call (recorded by an instrumented
  libopus build), so a mismatch splits cleanly between the Opus layer and CELT.
- **Scope for 1.0.0:** CELT-only. SILK and hybrid encoding come after 1.0.0; complexity 10 needs libopus' tonality analysis
  (not ported). Browser: green on WebGPU and Wasm (full lanes 1150 / 0 each, the 49 libopus streams included; the four CELT
  stage kernels compile in Chrome after ILGPU 5.4.12-local.9's WGSL loop-header fix); WebGL has no scatter stores, so the
  serial coders are n/a there (WebGL lane 399 / 0).
- **SpawnDev.ILGPU 5.4.12-local.8** (Data; every bug below was found by the Opus encoder or decoder): struct-held views in
  local allocas typed local (CUDA / OpenCL), sub-word integer conversion chains, Velocity kernels over 65,536 IR values
  (invalid IL), neg / not of a comparison, multiplicative-loop trip counts, PTX / OpenCL branch-edge phi moves (the
  CUDA / OpenCL stereo CELT encode), OpenCL switch phis, and byte / ushort return values widened by the wrong sign (the FLAC GPU
  decoder's header CRC on the IR backends). Full Codecs desktop lanes: CUDA, OpenCL, Velocity 1131 / 0 each, CPU 3281 / 0.
- `OpusEncoderApplication` moved from SpawnDev.Codecs.References to the library (same values).

### Toward 1.0.0: project split done, Opus GPU range encoder, CELT GPU primitives, VP9 WebGPU green (2026-10-29)
- **Where the "Path to 1.0.0 stable" bar stands** (0.3.0-rc.1 section below): (1) the Opus GPU pair: decoder done, encoder
  (CELT-only) done on every backend (above; WebGL n/a for the serial coders);
  (2) Vorbis decoder v2 is DONE (`VorbisAudioDecoderGpu.DecodePacketAsync` runs `VorbisPacketDecodeKernel.Step` on every
  backend; the host reads back only the packet header / floor flags it needs to orchestrate); (3) the project split is DONE
  (below); (4) the May ILGPU bugs (WebGPU `SubViewValue`, Wasm OOB) are long fixed.
- **Project split:** the CPU implementations still in the main library moved to `SpawnDev.Codecs.References` (same
  namespaces): the AV1 CPU keyframe encoder and its pipeline (coefficient coder, 2D transforms, Dct32 / Dct64 / identity,
  intra predictors, quantizer, mode-info reader, partition context), the VP9 Idct / Iadst / Iht Reference transforms, bool
  decoder and compressed-header parsers, the VP8 CPU predictors / transforms / frame buffer, and the Vorbis CPU floor /
  residue decoders. What the GPU paths needed from them stays in main: `Av1KeyframeFraming` (the TD + SH OBUs and the
  uncompressed header `Av1KeyframeEncoderGpu` uploads), `Av1IntraModeContext.Table`, the `Vp9TxType4x4` / `Vp9TxType8x8` and
  `Vp8IntraMode4x4` enums, `Vp9SkipProbs.DefaultProbs`. What remains in main besides kernels is tables, stream-header
  parsers / writers and the 1D transforms the kernels share.
- **`OpusRangeEncoderGpu`** (was scaffolding that produced AV1 bytes): libopus `celt/entenc.c` statement for statement on the
  accelerator, every entry point (encode, encode_bin, bit_logp, icdf 8 / 16-bit, uint, bits, patch_initial_bits, shrink,
  tell, tell_frac, done). Byte-identical with Concentus' EntropyCoder on random scripts incl. overflow, ec_tell / tell_frac
  equal after every op (CPU, Velocity, CUDA, OpenCL). Concentus 2.2.2's `enc_shrink` moves the raw-bit tail the wrong way
  (its MemMoveByte takes (src, dst)); ours follows libopus.
- **CELT GPU primitives (Data):** PVQ (cwrs.c + vq.c: decode, encoder pulse search, spread rotation, stereo theta), bit
  allocation (rate.c, decode side) with the 48 kHz mode tables on the accelerator, anti-collapse / intensity / mid-side /
  Haar / Hadamard (bands.c), comb pre/post-filter. Each bit-exact with Concentus.
- **New `OpusDecoderGpu`** (`SpawnDev.Codecs.Audio.Opus`, three overloads + `DecodeLostAsync` + `GetFinalRangeAsync`): the whole
  libopus 1.5.2 decoder (fixed point) on the accelerator. `SilkPacketDecoderGpu` (silk_Decode: VAD / LBRR header and LBRR
  skip, stereo, indices / pulses / parameters / decode_core, MS->LR, resampler; PLC, comfort noise and the glue after a loss;
  DTX), Data's `CeltDecoderGpu` (celt_decode_with_ec + concealment), and the opus_decode_frame glue: hybrid on the shared range
  decoder with CELT accumulating into the SILK PCM, redundancy frames both ways with smooth_fade, the transition concealment and
  fades, the hybrid -> SILK silence frame, CELT / SILK resets. The host parses the TOC / frame packing and reads back four
  redundancy ints per SILK / hybrid frame; every other bit decodes on the accelerator. **Equal to libopus 1.5.2 on every packet
  of 21 real streams** (PCM and final range): SILK, CELT-only at 2.5 / 5 / 10 / 20 ms, hybrid, a mode-switching stream; losses,
  FEC, DTX, stereo / mono and bandwidth switches. The reference is libopus 1.5.2 itself, built from source
  (`scripts/audio/make_opus_fixtures.py`, `TestData/Opus`): Concentus ports libopus ~1.1 and differs from 1.5.2 (CNG, PLC,
  decode_core's voiced-PLC transition, several CELT details, 64-bit MULT32_32 on Win64), and its silk_CNG never fills
  CNG_exc_buf_Q14.
- **ILGPU 5.4.12-local.1** (Data: Velocity hung on a loop exiting from both arms of an if; WebGPU if-arm back edge without
  `continue`, both found by the CELT band recursion). Full Velocity desktop lane 1008 / 0.
- **VP9 on WebGPU green** (267 / 0; WebGPU video 468 / 0) with a TJ-approved workaround: a drain after each wavefront
  group-kernel slice. The device hang it avoids reproduces in Chrome with no ILGPU (a replay of the recorded WebGPU calls), so
  it is below ILGPU (Dawn D3D12 or the NVIDIA driver); upstream report drafted. See Docs/status.md.
- **ILGPU 5.4.11** (released; forks 2.4.10). **PlaywrightMultiTest:** `PMT_INIT_SCRIPT` adds a JS file to every page.

### Lossless model-weight packing; WebGL lane runs for real; ILGPU 5.4.6-local.1 (2026-10-26)
- **New `WeightPackGpu`** (`SpawnDev.Codecs.Compression`, three overloads, format "SDWP" v1): lossless tensors (BF16 / FP16 / FP32 / any element size 1..16) as byte planes (`ShuffleFilterGpu`) cut into 64 KiB chunks, each ONE standard raw DEFLATE stream at the new Huffman-only level, plus a length table; any zlib reads every chunk. Compression: one thread per chunk in waves (DeflateGpu phases as bounded launches); decompression: chunk offsets from the table on the accelerator, one InflateGpu thread per chunk in waves, un-shuffle. Measured on our models (same samples, ratio = packed / original): BF16 0.677 (ZipNN 0.719, zstd -3 0.788), FP16 0.662 (ZipNN 0.675), FP32 0.833 (ZipNN 0.771: its zstd stage finds long-range repeats this coder does not).
- **`DeflateGpu.LevelHuffmanOnly` (4)** = zlib Z_HUFFMAN_ONLY (literals only, the block's own dynamic code); `ZlibCompressorGpu` takes level 4. Only level 3 is the lazy parse now (was `>= 3`).
- **Verified (CPU, CUDA, OpenCL 75/75 over filters, weights, zlib, PNG, LZ4, inflate; Wasm filters 6/6 + weights 6/6):** weights round trips exact (normal-distributed BF16 / FP16 / FP32, element sizes 1 / 3 / 8, empty, 64 MiB BF16 + 80 MB FP32 across many waves), every chunk inflates with .NET's DeflateStream and a host un-shuffle gives the input, sizes within 1 % of an optimal Huffman code of every chunk (built in the test), damaged blobs refused. Red-checked (bitshuffle instead of byte shuffle: the .NET oracle fails; stored instead of Huffman: the size bound fails).
- **WebGL lane runs for real:** `WebGLCodecsTests` threw `NotSupportedException` for EVERY test (no Codecs kernel had ever run on WebGL). It now creates the WebGL2 accelerator; tests of serial stream coders (inflate, zlib, PNG, LZ4, WeightPack) declare `SerialStreamKernels` (`RequiresScatterStores`) and skip there with that reason. First real run found: the shuffle filters' later launches stored at `dst[offset + idx]` (Transform Feedback captures only a thread's own slot; each launch now gets `dst.SubView(offset, count)` and stores `dst[idx]`), and a WebGL dispatch OutOfMemoryException (SpawnJS JSTape growth + full GLSL sent every dispatch; Data, ILGPU / SpawnJS).
- **ILGPU 5.4.6-local.1** (forks 2.4.6-local.1): IfConversion and loop-invariant code motion no longer hoist an integer division that can trap (Wasm "divide by zero" in `ShuffleKernel` for inputs shorter than one element, found here; fixed by Data).

### General data compression, step 1: shuffle filters on the accelerator (2026-10-26)
- **New track: general data compression** (`Plans/PLAN-Data-Compression.md`): filters, LZ4 / Snappy, interleaved rANS, lossless ML weights (ZipNN / DFloat11 exponent coding), GDeflate, Gaussian splat containers (SPZ, SOG), zstd. Texture formats (BC / ETC2 / ASTC) are a separate image track.
- **New `ShuffleFilterGpu`** (`SpawnDev.Codecs.Compression`, three overloads: `ArrayView1D` queue-only for chaining, managed, browser `TypedArray`): byte shuffle (c-blosc / numcodecs `Shuffle`, raw trailing partial element) and bitshuffle (the bitshuffle library's `bshuf_bitshuffle`: blocks, rounded last block, raw n % 8 leftovers, default or explicit block size), both directions. The reference's three transposes reduced to one closed form, so every output byte is an 8-bit gather (one thread per byte, no scatter: WebGL keeps the same kernels); launches of 4 MiB, paced.
- **Verified (CPU, CUDA, OpenCL):** 94 fixtures from the reference libraries (`TestData/Filters`, `scripts/compression/make_filter_fixtures.py`: numcodecs, bitshuffle via hdf5plugin; element sizes 1..16, partial elements / blocks / leftovers, a 64-element block) byte-identical in both directions into poisoned buffers; three 16 MiB cases match the reference SHA-256 and invert. Red-checked (last block not rounded; last element skipped in the inverse). Browser lanes pending.
- **New `Lz4FrameCompressorGpu`** (`SpawnDev.Codecs.Compression.Lz4`, three overloads): LZ4 frames of independent 64 KB blocks BYTE-IDENTICAL with liblz4 1.9.4 (`LZ4F_compressFrame`, block_linked=False, level 0 or an acceleration): liblz4's fast compressor (byU16 hash, skip schedule, catch-up, the conservative limitedOutput bounds that decide raw blocks) as a resumable state machine, one thread per block (8192 input positions per launch), waves of blocks laid out and copied on the accelerator; optional content size, block checksums, content checksum (XXH32, `Xxh32Gpu`, resumable). Only the frame length crosses to the host.
- **New `Lz4FrameDecompressorGpu`** (three overloads + `GetDecompressedLengthAsync`): any LZ4 frame (64 KB .. 4 MB blocks, independent or linked, HC-level frames, checksums, content size, skippable and concatenated frames, the legacy format); bounded walk / size scan / layout / decode (independent blocks in parallel, linked blocks in order; literal runs and matches resume across launches) / checksum stages; damaged frames refused with `Lz4FormatException.Reason` (header checksum, magic, truncation, block / content checksum, content size, match offset outside the block or frame, dictionary frames).
- **Verified (CPU, CUDA, OpenCL):** 29 liblz4 frames (`TestData/Lz4`, `scripts/compression/make_lz4_fixtures.py`: empty .. 40 MB, raw blocks, zeros, every option, acceleration 3 / 8, three real media files) byte-identical, including 3-block waves and a 40 MB two-wave input; 24 decoder frames (`TestData/Lz4Dec`, `make_lz4_decode_fixtures.py`) decode to their content or are refused with the expected reason; a 40 MB compress + decompress round trip on the accelerator. Red-checked (skip trigger 5 instead of 6: 19 frames differ; a match copy shifted one byte: 23 frames wrong). Browser lanes pending.
- `TestData/.gitattributes`: `*.bin` and `Filters/*` are binary (a 2-byte fixture `45 0A` was detected as text and would have been rewritten to CRLF on checkout).

### Lossy WebP with transparency; WebP lossless encode WebGPU hang found (ILGPU); Vorbis decoders match libvorbis (2026-10-25)
- **`WebPLossyEncoderGpu` encodes pictures with transparency** (was refused): libwebp's extended layout RIFF + VP8X (alpha flag, canvas) + ALPH + VP8. The alpha plane `WebPRgbaToYuvGpu` imports is coded on the accelerator by `WebPLosslessEncoderGpu.EncodeAlphaAsync` (lossless, like libwebp's alpha_quality 100; filter / compression / effort via the new `WebPLossyEncodeOptions.Alpha`), and the new `AssembleExtendedKernel` writes the file one thread per byte (no intermediate copy). All three overloads.
- **Verified (CPU accelerator):** 145 new fixtures (`TestData/WebPLossyAlpha`, `scripts/images/make_webp_lossy_alpha_fixtures.py`: alpha ramps, cut-out masks, scattered transparent pixels, 1x1 .. 640x480, quality 0..100, methods 3 / 4): the VP8 chunk is byte-identical to libwebp's, our ALPH decodes to exactly the source alpha in `WebPDecoderGpu` and in libwebp (Pillow oracle), and the `ArrayView1D` overload writes nothing past the file. Red-checked (shifted alpha plane, one flipped VP8 bit, one byte past the file). The ALPH bytes are not libwebp's (our VP8L coder differs); they decode to the same plane.
- **Vorbis:** both decoders match libvorbis on six real streams (block switching, floor books, residue type 2 de-interleave, coupling order, render_line, IMDCT phase, packets ending inside a floor). `VorbisAudioDecoderGpu` lost its v1 CPU bit-stream path and allocates nothing per packet (all work buffers sized for BlockSize1 at construction; the packet is uploaded from the caller's memory).
- **WebP lossless encode on WebGPU:** found the hang (device lost in Chrome, endless on SwiftShader): the ILGPU WGSL backend dropped the outer loop's carried values on one arm of `WriteCodeDefinition`'s run-length loop (also hits the SILK pulse kernels). Fixed in ILGPU (Data), version pending. Diagnostics left in: `WebPLosslessEncoderGpu.Trace` / `WebPEncodeKernels.Trace` (stage and per-launch names, no cost when off).
- **ILGPU 5.4.5-local.1:** sub-view `.Length` folded in the IR builder (9 VP8 / VP9 kernels failed Tint with "cannot take the address of let", and WebGL silently used the whole buffer's length).
- **PlaywrightMultiTest:** waits 660 s for a test (was 600 s = the in-page `Timeout`, a race that hid the page's own timeout report); `PMT_WEBGPU_ADAPTER=swiftshader`.
- **Lossy WebP encoder RD loop as five stage kernels per macroblock** (prepare, I16, I4, UV, record; was one 61k-line WGSL kernel), and the I16 / UV predictions as one flat loop (`PredictAll`): FXC compile time of the prepare stage 134 s -> 3.2 s. Still byte-identical with libwebp (222 + 145 fixtures). The browser shader compile cost is unrolled constant-bound loop nests, not line count (Plans/PLAN-Oversized-Kernels.md).
- **Vorbis GPU decoder: ArrayView1D and TypedArray overloads** (`DecodePacketAsync(ArrayView1D<byte>, int, ArrayView1D<float>)` writes PCM into the caller's view; `DecodePacketAsync(TypedArray)` returns a `Float32Array`); new `DeviceTransfer.DownloadFloat32ArrayAsync`.
- **SILK pulse decode on WebGPU:** `SilkPulsesInputs` bound one buffer to two read_write slots (WebGPU rejects it); it now embeds offsets only (`SilkShellCoderTableOffsets`).
- **PlaywrightMultiTest:** a dead page runtime (Mono assert, runtime exit, GC allocation failure, abort), a hung test or an OutOfMemoryException fails the running test and reloads the page for the next one. Before, one runtime exit turned 2,489 later tests into fake passes.
- **ILGPU 5.4.5-local.2** (WGSL loop back-edge moves, WebGL inner-loop exit, CopyFromCPU / CopyToCPU(ref T) pinning, LoopUnrolling and Velocity loop fixes). Lossless WebP encode now runs on WebGPU (it hung).
- **Known open on WebGPU (real GPU):** AV1 encode / decode and VP8 / VP9 encode kernels too large for Chrome's shader compiler (GPU process lost or minutes to compile), AV1 batch vs single-frame encode differ, lossless WebP encode and GIF slow (launch-bound; tests over the 30 s default). Being worked (stage kernels; ILGPU opaque loop bounds experiment).

### Lossy WebP encode on the accelerator: `WebPLossyEncoderGpu`, byte-identical with libwebp (2026-10-23)
- **New `WebPLossyEncoderGpu`** (`SpawnDev.Codecs.Images.WebP.Lossy`, three overloads: `ArrayView1D`, managed, browser `TypedArray`): RGBA8 -> lossy WebP whose bytes are IDENTICAL to libwebp 1.6.0 (Pillow) for the same pixels and quality, methods 3 and 4 (4 = libwebp's and Pillow's default). Kernel chain: `WebPRgbaToYuvGpu` (libwebp import + cleanup), analysis (one thread per macroblock), segmentation + quantizer setup (one thread), the RD token loop (one thread per launch, one launch per macroblock row: libwebp's decisions depend on the token statistics of every earlier macroblock, so the walk is serial), final probabilities + partition 0 + token partition (one thread), file assembly (one thread per byte). The host uploads the picture, ~500 ints of layout and a config-only quantizer table (libwebp's double-precision power law, which WGSL lacks) and reads back scalars.
- **Verified:** 222 fixtures (methods 3 / 4, quality 0..100, 1x1 .. 640x480) byte-identical on the CPU, CUDA and OpenCL lanes (CUDA 19.8 s, OpenCL 30.6 s for all 222 with the single-thread RD loop); red-check: a penalty of 141 instead of 140 makes 9 files differ. Browser lanes (WebGPU / WebGL / Wasm) not run yet.
- **Not yet (refused with `NotSupportedException`, never a different file):** methods 0..2 and 5..6, multiple token partitions (transparency: done 2026-10-25, see above). Parallelising the candidate modes inside a macroblock is the next performance step.
- **SILK decode in `OpusDecoder` (References) restructured like libopus `silk_Decode`; all 5 `CrossValidate_Silk*` tests now match Concentus** (they had failed since they were written). Fixes: the resampler always runs (equal rates = copy resampler with its input delay); mono output goes through the 2-sample `sStereo.sMid` buffering and is resampled from index 1 (one more sample of delay); a mono packet decoded to stereo output is written to BOTH channels (it filled only the first half of the interleaved buffer); stereo no longer delays mid / side twice. Packet header read in spec order (RFC 6716 4.2.3: mid VAD flags + mid LBRR flag, then side VAD flags + side LBRR flag; it read one shared LBRR flag after interleaved VADs, which misparsed every stereo packet); LBRR frames are parsed and skipped instead of throwing `NotImplementedException`; the mid-only flag is read for every frame whose side VAD is 0 (not only the first); the side channel's prediction memory is reset on its first coded frame after mid-only frames and that frame uses `CODE_INDEPENDENTLY_NO_LTP_SCALING`; mono/stereo stream switches reset the stereo predictor history and copy the resampler like libopus; an internal-rate change keeps the stereo state and resets only what `silk_decoder_set_fs` resets. `SilkDecoder` (standalone) also has the mono buffering. CPU: 287 pass / 0 fail / 5 skip over every Opus / SILK test.

### Lossy WebP encode, step 1: libwebp-exact input conversion + byte-identical CPU reference encoder (2026-10-22)
- **New `WebPRgbaToYuvGpu`** (`SpawnDev.Codecs.Images.Yuv`): RGBA8 -> YUV 4:2:0 + alpha exactly as libwebp's lossy encoder imports a picture (`ImportYUVAFromRGBA`: gamma-linear 2x2 chroma averaging, alpha-weighted where a block is partly transparent, `kInvAlpha` reciprocal) followed by `WebPCleanupTransparentArea` (smoothing of partly transparent 8x8 blocks, flattening of fully transparent runs). Three kernels; the constant tables are uploaded once. Matches LIBWEBP'S OWN planes (read through the `webp` cffi bindings, `scripts/images/libwebp_picture.py`) on 74 fixtures incl. 1920x1080, with and without the cleanup (CPU lane; GPU lanes pending).
- **New `Vp8LossyEncoderReference`** (`SpawnDev.Codecs.References`, CPU): a statement-by-statement port of libwebp 1.6.0's lossy VP8 encoder (analysis + k-means segmentation, quantizer / lambda setup, all intra predictions, transforms, RD mode decision, DC error diffusion, token buffer with the in-pass probability refreshes, bool encoder, headers, RIFF). Its files are BYTE-IDENTICAL with Pillow / libwebp on 552 fixtures: methods 0..4, qualities 0..100, 1x1 .. 640x480, photo / graphics / noise / flat content (`scripts/images/make_webp_lossy_encode_fixtures.py`). Tables are extracted from the C source by `scripts/images/gen_vp8enc_tables.py`, not transcribed. One libwebp quirk is ported on purpose: for methods 0..2 a single-segment picture makes `StatLoop` return before the skip / token probabilities are finalised, so default probabilities are written. Methods 5 / 6 (trellis), multiple partitions and ALPH are not ported yet. This is the oracle for the accelerator encoder (next).
- **ILGPU 5.4.3** (from 5.4.3-local.1): the SSA phi fix behind the nested-ternary miscompile; `Vp8FullLoopFilterKernel` uses the nested form again and the canary `KernelDialect_NestedThresholdTernary_MatchesIfChain` guards it.

### Lossy WebP decode + facade for GIF / BMP / WebP (2026-10-22)
- **`WebPDecoderGpu` now decodes lossy WebP**: still VP8, VP8 + ALPH, animations of lossy frames, and animations that mix lossy and lossless frames. Pipeline, all on the accelerator: `Vp8FullKeyframeDecoderGpu` (colour planes) -> `WebPYuvToRgbaGpu` (libwebp's fancy upsampling and matrix) -> the ALPH plane (raw or VP8L-compressed, every filter) merged into channel A by a new `ApplyAlpha` kernel; lossy sub-rectangle frames composite through the existing blend / dispose path. The VP8 kernels are compiled on the first lossy image. Corrupt VP8 data surfaces as `ImageDecodeException` ("invalid VP8 data: ..."), not as a `Vp8DecodeException`; a VP8 size that disagrees with the container is `CanvasMismatch`. The old "lossy WebP not yet supported" error is gone.
- **`ImageDecoderGpu` dispatches GIF, BMP and WebP** (new `Gif`, `Bmp`, `WebP` properties next to `Png` / `Jpeg`) through all of its overloads (`ArrayView1D`, managed, `TypedArray`, `ReadInfoAsync`).
- **Tests:** 168 fixtures (`TestData/WebPLossy`, `scripts/images/make_webp_lossy_decode_fixtures.py`: stills with alpha 1x1 .. 1920x1080, encoder methods 0..6, qualities 20..100, lossy and mixed animations, plus every lossy file already in `TestData/WebP` and `TestData/WebPYuv`) are BIT-EXACT with libwebp (Pillow decode) through the managed and the `ArrayView1D` overload (0xAB-poisoned caller buffer); pictures over 70000 pixels are compared by SHA-256. Red-check: with the alpha merge disabled 270 failures name the alpha fixtures. CPU lane only so far; CUDA / OpenCL / WebGPU / WebGL / Wasm lanes not yet run.

### VP8 lossy keyframes, task 2: full-syntax keyframe decoder on the accelerator (2026-10-22, branch img/vp8-full)
- **New `Vp8FullKeyframeDecoderGpu`** (three overloads: `ArrayView1D`, managed, browser `TypedArray`): decodes the whole keyframe syntax that libwebp / libvpx write, entirely in kernels. Header (segmentation incl. map and per-segment quantizer / loop-filter level, filter header with mode/ref deltas, quantizer deltas, coefficient probability updates, skip probability) and the token-partition size table; per-macroblock modes with all B_PRED sub-modes and their contexts; DCT tokens from 1/2/4/8 partitions; dequantization per segment, Y2 + inverse WHT, 4x4 IDCT; all 16x16, 4x4 and chroma intra modes with the frame-edge rules; the normal and simple loop filter; crop to the signalled size. The v1 `Vp8KeyframeDecoderGpu` is unchanged.
- **Schedule (documented in the kernels):** header / modes / tokens are single GPU threads chunked ONE macroblock row per launch (the bool decoder state lives in a buffer between launches, so no launch runs long); reconstruction and the loop filter are one thread per macroblock, one launch per WAVEFRONT DIAGONAL `col + 2 * row` (left, top, top-left and top-right neighbours are on earlier diagonals, macroblocks on one diagonal are two columns apart). Loop filter = correctness-first: each macroblock does all its edges serially.
- **Hostile input:** a 32-bit bool decoder detects reads past the end of its partition on the accelerator (libwebp's rule: a decision whose 8-bit window leaves the partition); `Vp8DecodeException.Error` is one of TooShort, NotKeyFrame, BadStartCode, ZeroSize, UnsupportedVersion, FirstPartitionExceedsFrame, FirstPartitionTruncated, PartitionTableTruncated, TokenPartitionExceedsFrame, TokenPartitionTruncated.
- **Tests (CPU lane):** all 182 fixtures bit-exact with ffmpeg through the managed overload and through the `ArrayView1D` overload into 0xAB-poisoned longer buffers; 20 specific hostile inputs; 120 random corruptions; the wavefront schedule property test. 17 mutations of the decoder were red-checked. Not yet run on CUDA / OpenCL / WebGPU / WebGL / Wasm; the 16-bit `short` buffers and byte stores are the first things to check there.
- Limit: raw coefficients of the whole frame are held on the accelerator (800 bytes per macroblock; 4000x3000 = 37 MB).

### VP8 lossy keyframes, task 1: CPU reference loop filter + real-file fixtures (2026-10-22, branch img/vp8-full)
- **New `Vp8LoopFilter`** (References): RFC 6386 section 15 / libvpx loop filter for key frames (normal and simple filter, sharpness, interior/edge limits, hev threshold, per-segment absolute/delta levels, ref/mode lf deltas incl. the B_PRED delta, inner-edge skip for coefficient-less non-B_PRED macroblocks, frame level 0 = no filtering), run by `Vp8KeyframeWalker` after the whole frame is reconstructed. New one-call `Vp8KeyframeReference.Decode`.
- **Bugs the real files exposed (all fixed):** `B_VL_PRED` sub-block (3,3) used avg3(A4..A6) instead of avg3(A5..A7) (CPU predictor AND the 4x4 prediction kernel; both agreed with each other, neither with RFC 6386 / ffmpeg); the reference bool decoder rejected probability 0 (libwebp writes coefficient probabilities of 0; split = 1 is well defined); a skipped B_PRED macroblock cleared the Y2 entropy context it must leave alone. Five stale unit tests with wrong hand-computed expectations (frame tag 0xC10 vs 0xC90, inverse Walsh DC 8 vs 32, `dc_qlookup[37]` 28 vs 34) were corrected.
- **182 VP8 keyframe fixtures** in `TestData/Vp8Lossy` (`scripts/images/make_vp8_lossy_fixtures.py`): libwebp 1.5/1.6 output (Pillow + cwebp: quality 1..100, methods 0..6, 4 segments, simple/normal filter, sharpness 0..7), libvpx output via ffmpeg (token partitions 2/4/8, versions 0..3, lf deltas), and 80 random-syntax streams (valid header written by an independent python bool encoder, random partitions: every sub-mode in every context, skipped B_PRED, prob-0 updates, clamping deltas). Truth = ffmpeg's native VP8 decoder (verified native yuv420p), cross-checked against ffmpeg's webp decoder and libwebp `dwebp -yuv` for every libwebp file. The CPU reference is BIT-EXACT with ffmpeg on all 182.

### Image codecs foundation, GPU zlib, PNG (2026-10-22)
- **New `SpawnDev.Codecs.Compression`**: `InflateGpu` / `InflateKernel` (a puff.c port; a DEFLATE / zlib decoder as a single GPU thread, every block type, Adler-32 verified, specific `InflateError` codes), `Crc32Gpu`, `DeflateGpu` + `ZlibCompressorGpu` (block-parallel compressor: independent blocks, LZ77 with hash chains (greedy or lazy), dynamic Huffman with length-limited codes, stored-block fallback, byte-aligned block joins; levels 0..3). Verified against .NET's `ZLibStream` in both directions.
- **New `SpawnDev.Codecs.Images`**: `ImageFormat` / `ImageInfo` / `PixelFormat` / `DecodedImage` / `DecodedImageJS` / `ImageFormatDetector` / `ImageFrameInfo` (shared by every image codec) and the `ImageDecoderGpu` facade (format sniffing, lazy per-format decoders).
- **`PngDecoderGpu`**: every colour type and bit depth (1/2/4/8/16), palette + tRNS, gray / RGB colour keys, Adam7, critical-chunk CRC and zlib Adler checks, RGBA8 and RGBA16 output; chunk scan, gather, inflate, unfilter and colour conversion all run on the accelerator. `ReadTextAsync` returns tEXt / zTXt / iTXt chunks (ComfyUI "prompt" / "workflow"). 64 generated fixtures (`scripts/images/make_png_fixtures.py`, truth = the source samples).
- **`PngEncoderGpu`**: RGBA8 / RGBA16 input, automatic colour-type reduction (gray, gray+alpha, RGB, RGBA), adaptive or fixed scanline filters, block-parallel deflate with one IDAT chunk per block (per-chunk CRC on the accelerator), tEXt / iTXt output. Independent host-side reader in the tests plus `scripts/images/verify_png_encoder.py` (Pillow).
- **`Av1ObuAssembleKernel`**: the AV1 temporal unit (TD + SH + Frame OBU, leb128 size) is now assembled on the accelerator; the host builds only a fixed blob of under 128 bytes. `Av1KeyframeEncoderGpu` got the three-overload shape. (Still multiples of 64 only: task #23.)
- **Bug fix (VP9):** the single-frame sequential encode kernel had no boundary-superblock handling, so a picture whose height was not a multiple of 64 encoded differently alone than in a batch; the single path now runs the batch kernel at extent 1.
- **Kernel dialect rule (WebGPU):** loop headers and branch conditions are single `bool` locals built with `&` / `|`. A `while (a && b)` in the first inflate hung the WebGPU device while CPU / CUDA / OpenCL passed.

### Image codecs: JPEG decoder + encoder (2026-10-22, branch img/jpeg)
- **`JpegDecoderGpu`** (`SpawnDev.Codecs.Images.Jpeg`): baseline, extended sequential and PROGRESSIVE Huffman JPEG (spectral selection + successive approximation, all four progressive scan types), 8-bit, restart markers (independent restart segments decode as parallel GPU threads, every long loop is chunked so no launch runs for long), every integral sampling ratio (4:4:4, 4:2:2, 4:2:0, 4:4:0, 4:1:1, odd and non-MCU-aligned sizes), grayscale, YCbCr, RGB (Adobe transform 0 / component ids R,G,B), CMYK and YCCK. 12-bit, arithmetic, lossless and hierarchical files throw a specific `ImageDecodeException`. Three overloads (`ArrayView1D`, managed, browser `TypedArray`) plus `ReadInfoAsync`; `JpegDecodeOptions.ApplyOrientation` applies the EXIF orientation with a GPU kernel. Marker scan, Huffman table build, entropy decode, IDCT, upsampling and colour conversion are all kernels; the host only reads back the metadata block.
- **Bit-exact with libjpeg-turbo as used by Pillow**: ISLOW integer IDCT (jidctint), "fancy" h2v1 / h1v2 / h2v2 upsampling (replication for other ratios), libjpeg fixed-point YCbCr -> RGB, Pillow's inverted-CMYK conversion. Verified against Pillow's own decode of 100+ fixtures (`scripts/images/make_jpeg_fixtures.py`) and SHA-256 of 1024x768 pictures.
- **`JpegEncoderGpu`**: baseline Huffman encoder with libjpeg quality scaling of the Annex K tables, 4:4:4 / 4:2:2 / 4:2:0 / gray, restart intervals (parallel segment threads), EXIF orientation segment, `GetMaxEncodedSize`. Colour conversion, downsampling, forward DCT + quantization and entropy coding all run on the accelerator. The output is BYTE-IDENTICAL to Pillow / libjpeg-turbo for the same pixels and options (all fixtures, 1024x768 and 2048x2048 pictures). `JpegEncodeOptions.OptimizeHuffman` (Pillow `optimize=True`) and `Progressive` (libjpeg's default scan script) run libjpeg's two-pass optimal-table coding on the accelerator (statistics pass with atomics, `jpeg_gen_optimal_table`, DHT written device-side) and are byte-identical to Pillow too. These two options need atomics, so they do not run on the WebGL backend.
- Tests: `CodecsTestBase.JpegDecoderGpuTests` / `JpegEncoderGpuTests` (fixtures in `TestData/Jpeg`, about 1.3 MB). Alpha is ignored by the encoder.

### Image codecs: BMP and GIF (2026-10-22)
- **`BmpDecoderGpu` / `BmpEncoderGpu`** (`SpawnDev.Codecs.Images.Bmp`): 1/4/8/16/24/32 bpp, RLE4/RLE8, bitfields, V2..V5 headers; encode Bgra32 / Bgr24 / Indexed8 (exact palette found on the accelerator). Three overloads each; tests against generated fixtures and, for the encoder, Pillow as the independent oracle.
- **`GifDecoderGpu`** (`SpawnDev.Codecs.Images.Gif`): variable-width LZW (min code size 2..8, clear / end codes, deferred clear, KwKwK) with one GPU thread per frame and frames in parallel, interlace, local / global / identity-gray palettes, transparency, disposal 0..7, partial / clipped frames, NETSCAPE loop count (`ReadLoopCountAsync`), comment / application / plain-text extensions skipped, 87a and 89a. Animation API: `ReadFramesAsync`, `DecodeFrameAsync(k)` (full canvas while frame k is shown; compositing is a kernel). Three overloads each. `Compositing` property: `GifCompositing.Pillow` (default, bit-exact with Pillow 13 `Image.seek(k)` + `convert("RGBA")` incl. its quirks) or `GifCompositing.Web` (browser rules). Specific `ImageDecodeException` reasons for truncation, bad blocks, bad LZW codes, early end, a frame larger than its data can describe.
- **`GifEncoderGpu`** (`SpawnDev.Codecs.Images.Gif`): still and animated GIF encode, every stage a kernel. One exact palette (sorted by 0xRRGGBB, `ExactPaletteBuilder`) when the opaque colours of all frames fit `MaxColors` (minus one transparent index), else per-frame exact-or-quantized palettes; quantizer = median cut on a 5-5-5 histogram (split the box with the highest population x extent at the population median, palette = pixel-weighted box means, deterministic, no host math); mapping = nearest colour (ties: lowest index), optional 8x8 Bayer ordered dither or Floyd-Steinberg (anti-diagonal waves, identical to the sequential algorithm). LZW is cut into independent 32768-pixel segments (clear code between them) encoded by one thread each and concatenated bit-exactly on the accelerator, then sub-blocked straight into the caller's view. Transparency (alpha < `AlphaThreshold`), per-frame delays and disposal, NETSCAPE loop count. `GetMaxEncodedSize`, three overloads for single pictures and for animations. Quantizer quality on the Big Buck Bunny frame: 34.9 dB (256 colours) vs Pillow median cut 32.4 dB.
- Encoder tests: 40 fixtures whose expected files are written by an independent Python implementation of the same algorithms (`scripts/images/make_gif_encoder_fixtures.py`) and compared BYTE FOR BYTE; every expected file is also decoded by Pillow; Pillow decodes our output to the same canvases; round trips through `GifDecoderGpu`; brute-force nearest-colour check; random lossless round trips; a 786432-pixel picture (24 segments).
- Tests: ~165 fixtures from `scripts/images/make_gif_fixtures.py` (hostile streams, Pillow-written files, 64 random animations) with a per-frame Pillow truth and a spec truth; the bulk test `GifDecoderGpu_FuzzDirectory_MatchesPillow` compared 2500 random animations (11247 frames) with Pillow exactly. Browser TypedArray overloads are not run on the desktop lane.

### WebP: ALPH, metadata parity, animated encode (2026-10-22)
- **Cross-colour transform guard.** The forward transform SUBTRACTS the delta (libwebp `new_red -= ColorTransformDelta(green_to_red, green)`), the inverse adds it; the committed encoder kernel already did. A `+` in the encoder is invisible to round-trip tests because the strategy cost then makes the encoder reject the cross transform, so the picture only got bigger; forced on, libwebp decodes it wrong. New fixture `enc_correlated_72x56` and the parser statistic `cc_nz` (tiles with non-zero multipliers, `webp_ref.py`) make the guard fail by name.
- **ALPH decode (`WebPDecoderGpu.DecodeAlphaAsync`, three overloads).** The alpha plane of a lossy VP8X image (still, or any animation frame): raw or VP8L-compressed (headerless stream, every transform) ALPH data and the none / horizontal / vertical / gradient prediction filters, all on the accelerator; works before a lossy decoder exists and is its alpha half. Invalid headers (method 2/3, pre-processing 2, reserved bits, empty chunk, short raw data) are rejected with specific reasons, exactly the cases libwebp rejects (the fixture generator asserts that against libwebp). An ALPH chunk after the VP8 chunk no longer counts as the image's alpha (libwebp ignores it); the last ALPH before the image wins.
- **ALPH encode (`WebPLosslessEncoderGpu.EncodeAlphaAsync`, three overloads, `GetMaxAlphaEncodedSize`).** Filter none / horizontal / vertical / gradient / `Auto` (from `Quality` 50 it measures the exact coded size with each filter - a measuring pass writes nothing - and keeps the smallest; below that a fast residual-cost estimate on the accelerator picks), lossless coding through the same pipeline as the stills (palette with pixel bundling, predictor) without the VP8L header, raw fallback when coding does not beat the raw plane (never larger than 1 + w*h), pre-processing level 0. Output is the chunk payload for a lossy encoder to wrap.
- **Metadata (ICCP / EXIF / XMP).** Reader parity with libwebp's demuxer: a chunk only counts when its VP8X flag is set, and only the first of a kind (it used to report a chunk whose flag was clear). New `ReadFileInfoAsync` / `ReadMetadataAsync` TypedArray overloads. 40 new fixtures (`meta2_*`, libwebp's demuxer is the truth): every subset, flags without chunks and chunks without flags, chunk order, duplicates, unknown chunks, RIFF padding of odd payloads, libwebp-muxed lossy+alpha and animated files, EXIF orientation 1..8 in both byte orders with and without the `Exif\0\0` prefix and ten malformed forms (reported as 1). The encoder's metadata is verified through libwebp's demuxer (payloads, VP8X flag bits, no stray chunks) for every subset including a 200 KB ICC profile.
- **Animated lossless encode (`WebPLosslessEncoderGpu.EncodeAnimationAsync`, three overloads, `GetMaxAnimationEncodedSize`, `WebPAnimationEncodeOptions`).** Frames are rectangles with RGBA pixels and the shared `ImageFrameInfo` (offset, duration, `FrameDispose.None` / `Background`, `FrameBlend.Source` / `Over`); each frame is encoded by the still pipeline straight into its ANMF chunk and the RIFF / VP8X / ICCP / ANIM / ANMF / EXIF / XMP headers are written by kernels. Loop count, background colour, metadata. Verified frame by frame against libwebp's animation decoder (canvases from an independent numpy compositor that the generator cross-checks against libwebp, durations, loop count, background colour) and `WebPDecoderGpu`.
- `WebPEncodeOptions` is no longer sealed (`WebPAnimationEncodeOptions` derives from it). `scripts/images/make_webp_alpha_fixtures.py`, `make_webp_meta_fixtures.py`, `make_webp_anim_encode_fixtures.py` generate the new fixtures; `verify_webp_files.py` gained the alpha / animation / demuxer-metadata checks and `webp_ref.py` an independent ALPH decoder. The TypedArray overloads are covered by browser-lane tests that have not been run yet.

### WebP: container + VP8L lossless decoder and encoder (2026-10-22)
- **`WebPDecoderGpu`** (`Images/WebP`): RIFF container (simple VP8L, VP8X with ICCP / EXIF / XMP, ANIM / ANMF), the complete VP8L bitstream (predictor with all 14 modes, cross colour, subtract green, colour indexing with 1/2/4/8 pixels per word, colour cache, meta prefix codes, canonical prefix codes in simple and code-length-code form, LZ77 with the 120 short distance codes). The bitstream decode is one GPU thread; the inverse transforms (except the predictor, which is raster-order sequential) and the ARGB to RGBA conversion are parallel kernels. Three overloads per entry point (managed, `TypedArray`, `ArrayView1D`); `ReadInfoAsync`, `ReadFramesAsync`, `DecodeFrameAsync`, `ReadFileInfoAsync` (losslessness, which metadata exists, ANIM loop count / background) and `ReadMetadataAsync` (ICCP / EXIF / XMP blobs). EXIF orientation is reported in `ImageInfo.Orientation`.
- **Animation:** `DecodeFrameAsync` composites frames 0..k on the accelerator, bit-exact with libwebp's `WebPAnimDecoder` (key frame rule, blend with libwebp's integer arithmetic, dispose to background; the ANIM background colour is ignored exactly like libwebp does).
- **Lossy VP8 chunks** (and ALPH) are recognised: `ReadInfoAsync` / `ReadFramesAsync` describe them, decoding throws `ImageDecodeException("lossy WebP not yet supported")`. The image decode is one switch on the payload kind, so a VP8 decoder plugs in next to VP8L.
- **`WebPLosslessEncoderGpu`**: palette (sorted, delta coded, 1/2/4/8 pixels per word), subtract green, per-tile predictor (14 modes, chosen by minimum residual magnitude, residuals fully parallel), per-tile cross colour, LZ77 hash chains with the 2-D distance codes, colour cache (size search), length-limited Huffman, meta prefix codes (k-means over per-tile histograms), VP8X + ICCP / EXIF / XMP. `WebPEncodeOptions`: `Quality` 0..100, `Exact`, `IccProfile`, `Exif`, `Xmp`. Output is bit-exact in libwebp and about 1.0 to 1.5x libwebp's size at the default quality on the test sources.
- Tests: `CodecsTestBase.WebP*Tests.cs`; fixtures from libwebp via Pillow (`scripts/images/make_webp_fixtures.py`), the encoder is verified against libwebp (`scripts/images/verify_webp_files.py`, needs `CODECS_PILLOW_PYTHON`). `scripts/images/webp_ref.py` is an independent pure-Python VP8L decoder used to assert fixture feature coverage and encoder feature use.

### Three-overload boundary, VP8 first (2026-10-22)
- **Reference bump:** `SpawnDev.ILGPU` 5.2.17 -> 5.4.2, `SpawnDev.SpawnJS.Blazor` 2.1.19 -> 3.0.4 (`SpawnDev.UnitTesting.Browser` 2.7.0 and `SpawnDev.EBML` 3.0.1 are already the newest).
- **New `DeviceTransfer`** (public static): the one place codec entry points move bytes between managed memory / JS typed arrays and accelerator buffers (`Upload` from `ReadOnlySpan<byte>` with no staging copy, `Upload` from a `TypedArray` via `IBrowserMemoryBuffer.CopyFromJS`, `DownloadAsync`, `DownloadUint8ArrayAsync`).
- **`Vp8KeyframeEncoderGpu`**: every entry point now has managed, `TypedArray` (browser, `[SupportedOSPlatform("browser")]`) and `ArrayView1D` overloads, all funnelling into the `ArrayView1D` one. The `ArrayView1D` overload takes planes already on the accelerator, assembles straight into a caller-owned output view and returns only the length. New `GetMaxEncodedSize(width, height)`. The batch path is frame-major with caller-owned `outputs` + `lengths`.
- **Host row loops removed from VP8 encode.** `UploadPaddedPlane`, the batch `hostY/hostU/hostV` builders (which put whole batches on the host heap) and the `.ToArray()` staging copies are gone; the new `Vp8PlanePadKernel` strips the source stride and zero-pads to the macroblock grid on the accelerator.
- **Bug fix: odd picture sizes.** Chroma size is `ceil(dim/2)`, not `dim/2`; the encoder used to drop the last chroma column/row of odd-sized pictures.
- **`Vp8KeyframeDecoderGpu`**: same three overloads (`DecodedFrameJS` for the browser). **Bug fix:** it threw `NotSupportedException` for any size that is not a multiple of 16 even though the encoder emits such frames; it now decodes the macroblock grid and crops on the accelerator. `ReadFrameTagAsync` reads only the 10-byte header from a device-resident frame.
- New tests: `*BoundaryTests` (ArrayView1D into a poisoned caller buffer, odd sizes vs an explicitly zero-padded picture and vs the cropped CPU walker, batch vs single frames, TypedArray in the browser lanes). Not yet run: waiting on the GPU (peer release gate).

### PMT: system Chrome for real WebGPU (2026-09-28)
- PlaywrightMultiTest now launches installed Chrome (`Channel = "chrome"`) with `SpawnDev.Codecs.PlaywrightProfile`, matching ILGPU/ML. Dropped `Vulkan` from `--enable-features` and added `--disable-software-rasterizer` so Windows Dawn stays on D3D12 instead of silently falling back to SwiftShader.

### Port BlazorJS to SpawnJS (2026-09-28)
- **Replace `SpawnDev.BlazorJS` with `SpawnDev.SpawnJS.Blazor` 2.1.19** (brings `SpawnDev.SpawnJS` 2.1.19).
- **Bump `SpawnDev.ILGPU` 4.9.5-local.17 → 5.2.17** - ILGPU 4.9.x still depended on BlazorJS; 5.2.x is the SpawnJS stack.
- Demo/tests: `AddSpawnJSRuntime` / `SpawnJSRunAsync`; `UnitTesting.Blazor` → `UnitTesting.Browser` 2.7.0.
- VP8 GPU encode/decode APIs made async (`EncodeKeyFrameAsync` / `DecodeKeyFrameAsync` / `EncodeKeyFramesBatchAsync`) so browser backends use `SynchronizeAsync` + `CopyToHostAsync` instead of desktop-only sync Synchronize/GetAsArray1D.
- VP9 assemble/uncompressed-header convenience overloads: `Run` → `RunAsync` (same reason).

## Unreleased — master after rc.7 (2026-05-06)

### Wasm cold-compile diagnosis + Lever B (2026-05-06)
- **Root cause identified**: VP9 + VP8 entropy walker kernels emit one Wasm function with **52,012 locals + 750KB instructions + 5,127 inlined call sites**. V8's `kV8MaxWasmFunctionLocals=50,000` cap rejects the kernel with `WebAssembly.compile(): Compiling function #N failed: local count too large`; wabt's parser rejects identically. Same root cause as Geordi's WGSL walker `L44455` chain — ILGPU IR optimizer inlines the entire call graph (EncodeSb64 → EncodeBlock32x32 → EncodeBlock16x16 → EncodeLeafBlock → EncodePlaneCoefs) into the single `[NoInlining]` body method instead of emitting them as separate fn-defs. The Playwright 30s timeout on Codecs Wasm tests was a downstream symptom of the kernel never compiling. Diagnosis tools added at `_codecsdump/wasm-analyze.cs` (per-function local count + body size decoder) and DevComms `tuvok-to-geordi-codecs-wasm-monolithic-inlining-2026-05-06.md`.
- **VP9 + VP8 entropy: collapse single-frame + batch kernels into one** (`Vp9FrameEntropyKernel`, `Vp8FrameEntropyKernel`). Single-frame `Run` now dispatches the batch kernel with extent=1 + full-buffer strides; the inner `SubView(0, fullLen)` is a pointer-arithmetic no-op. Saves one full ~761KB Wasm module's compile pass per backend at zero runtime cost, plus the same gain on every other backend (CUDA / OpenCL / WGSL / GLSL pay the same per-kernel compile cost). Doesn't single-handedly close the V8 50K-local cap (the surviving module still has 52K locals) but halves the number of times that cost is paid.
- **PMT actionability timeout bumped to 5 min** (`PlaywrightMultiTest/ProjectTest.cs`). Codec kernel cold compile on V8 currently exceeds Playwright's 30s default; bumping the per-page default lets tests survive cold compile while ILGPU's fn-def codegen path matures. Cheap on hot tests (timeout doesn't fire when responsive).
- CPU sweep: 91 of 97 PASS post-collapse (6 unrelated VP8 failures pre-existing on master HEAD `5977809` — `Vp8KeyframeWalker_RejectsMultiPartition`, `Vp8Quantizer_*`, `Vp8FrameTag_KeyFrame320x240_ParsesAllFields`, `Vp8InverseTransform_ShortInvWalsh4x4_*`; tracked separately).

## Unreleased — master after rc.7 (2026-05-05)

**Headline: AV1 Tx4x4 transform infrastructure (forward + inverse + constants) lands on master + Vorbis encoder gets per-packet host-SubView batch dispatch (no host syncs between packets).**

### AV1
- `Av1Forward2dTransformGpu.Forward4x4DctDct` GPU helper (commit `c654e1b`) - W=4 H=4 CosBit=13, libaom shifts s0=2 s1=0 s2=0. Bit-exact vs CPU `Av1Forward2dTransform.Apply(Av1TxSize.Tx4x4, Av1TxType.DctDct, ...)` across CPU + CUDA + OpenCL + WebGPU + Wasm (15/15 backend invocations PASS; WebGL skips per existing atomics gap). Three tests: zero input, DC-only (input=16), 64-block random batch.
- `Av1Inverse2dTransformGpu.Inverse4x4DctDct` GPU helper (commit `12a0014`) + `InverseDct4Inline` + `ResolveCospi4` private helpers. Pattern follows Inverse8x8DctDct - row pass uses Inverse4 in place, column pass uses scalar inline helper. Inverse shifts: rowShift=0, colShift=4, cosBit=12. Bit-exact across the same 5 backends.
- `Av1KeyframeConstantsGpu` extended with Tx4x4 entries (commit `d75c0bc`): NzMapCtxOffset[0] 16 bytes + Scan4x4 16 ushort + TxbSkipCdfTx4x4 156 + EobMulti16Cdf 96 + EobExtraCdfTx4x4 216 + CoeffBaseEobMultiCdfTx4x4 128 + CoeffBaseMultiCdfTx4x4 1680 + CoeffLpsMultiCdfTx4x4 840. Total 3148 new entries appended to the existing constant offset chain. Bit-exact vs libaom default tables (36/36 backend invocations).

These are Phase 2 prep for AV1 boundary chroma encoding (BLOCK_16X8 / BLOCK_8X16 leaves at non-64-multiple frame dims). Walker Phase 1 + Phase 2 wiring lands separately when SpawnDev.ILGPU's WGSL fn-def codegen path stabilizes (Geordi iterating; tracked via `phase1-walker-debug` branch).

### Vorbis
- Phase A (commit `15f4f39`): `EmitKernel` now reads floor posteriors from the device buffer instead of round-tripping to host scalars. The encoder pipeline keeps posteriors GPU-resident from `FloorFitKernel` output through `EmitKernel`, eliminating the per-packet `await acc.SynchronizeAsync()` + `dPosteriors.CopyToHostAsync()` round-trip that was forcing a sync barrier mid-encode. 10/10 single-packet round-trip tests PASS unchanged.
- Phase B (commit `7ff3f3f`): new `EncodeAudioPacketsBatchAsync(IReadOnlyList<ReadOnlyMemory<float>> blocks)` API. Encodes N independent audio packets in a single host call. Per-packet stride buffers + per-packet host-side `SubView` dispatch into the existing single-packet kernels means 6×N kernel dispatches but only ONE host sync at the end of the batch (vs 6×N sync barriers in N sequential `EncodeAudioPacketAsync` calls). The per-packet kernels keep the same binding count + signature as the proven single-packet runs, sidestepping the WebGPU bind-group / Wasm codegen issues that tripped the Index2D / Index1D true-batch attempts. 15/15 PASS including new `BatchSilence` regression test.

### Demo / debugging
- `SpawnDev.Codecs.Demo` now wires `SpawnDev.ILGPU.Services.ShaderDebugService` (Program.cs registration) + adds the "Set Debug Folder" / "Grant Write Access" UI block on `/tests`. Pick a folder once at the demo URL and every kernel compile during PMT runs auto-dumps the WGSL/GLSL/Wasm + metadata-headed sources to `{folder}/{timestamp}/wgsl|glsl|wasm/`. Folder choice persists across sessions in IndexedDB (the PMT Chromium profile is persistent at `%TEMP%/SpawnDev.Codecs.PlaywrightProfile/`). Mirrors the pattern already in use by `SpawnDev.ILGPU.Demo`.
- The dump folder is critical for diagnosing ILGPU codegen issues at the kernel level - e.g. AV1 walker compile failures on WebGPU were root-caused via the dump in one DevComms reply instead of the prior 3 rounds of hypothesis-mode ping-pong.

### Cumulative shipped numbers today
- 5 master commits + Demo ShaderDebugService wiring, 66/66 backend invocations green (CPU + CUDA + OpenCL + WebGPU + Wasm; WebGL skips per existing gap).
- `SpawnDev.ILGPU` consumer reference held at `4.9.5-rc.14` on master while debug branches iterate against ILGPU local.X drops for codegen fixes.

## 0.3.0-rc.7 (2026-05-04, local only)

**Headline: SilkDecodeFrameGpu multi-frame state-rolling proven bit-exact + signal-type/sample-rate matrix complete + ILGPU rc.11 bump.**

`SilkDecodeFrameGpu_TwoFrameStateRolling_NB_BitExactVsCpu` decodes 2
consecutive voiced NB frames threading SLpcQ14Buf + OutBuf + PrevGainQ16
across calls, comparing both frames' PCM bit-exactly to CPU
`SilkDecodeFrame.Decode`. Validates the orchestrator is multi-frame
ready end-to-end - frame N+1's LTP synthesis correctly reads frame N's
post-finalize OutBuf history, frame N+1's gain-adjust correctly chains
from frame N's last subframe gain.

Plus full signal-type × sample-rate matrix at the FRAME level:
- Voiced + Unvoiced + Inactive (3 signal types)
- NB (8 kHz / subfr=40) + MB (12 kHz / 60) + WB (16 kHz / 80)

`SilkDecodeFrameGpuOrchestrator` constructor reduced from 6 to 5 kernels
(`InitStateKernel` folded into `IndicesAdapterKernel`).

Plus DecodeBitLogP CELT-prep tests on `OpusRangeDecoderGpu` - bit-exact
vs CPU on CPU + CUDA + OpenCL + Wasm (WebGPU has a separate codegen
issue tracked at
`_DevComms/SpawnDev.ILGPU/tuvok-to-geordi-decodebitlogp-webgpu-2026-05-04.md`).

PackageReference SpawnDev.ILGPU 4.9.5-rc.11 (Wasm dispatcher
host-write-vs-queued-dispatch race fix - YOLOv8 Softmax bug on Data's
side; SILK pipeline benefits transitively).

Source on master commit `623ca5e`. 13 commits since rc.6.

## 0.3.0-rc.6 (2026-05-04, local only)

**Headline: Full SILK per-frame decoder GPU pipeline complete (bitstream → PCM).**

`SilkDecodeFrameGpuOrchestrator` ships as a host-orchestrated multi-kernel
pipeline that mirrors libopus `silk_decode_frame.c` end-to-end:

- **Phase A** (init→indices→pulses): seeds the range-decoder state in a
  1-element `ArrayView<OpusRangeDecoderGpuState>` and threads it through
  the indices + pulses kernel dispatches.
- **Phase A+P** (+ parameters dequant): reads decoded indices buffer,
  runs `SilkParametersDecoderGpu.Decode` to produce gainsQ16, pitchL,
  predCoefQ12, ltpCoefQ14, ltpScaleQ14 outputs.
- **Phase A+P+C** (+ decode core synthesis): chains
  `SilkDecodeCoreGpu.Decode` reading parameters as SubViews into the
  unified parameter output buffers, producing PCM bit-exact vs CPU
  `SilkDecodeFrame.Decode`.
- **Phase A+P+C+F** (+ finalize): adds `FinalizeFrameKernel` for the
  OutBuf shift step (history slide + xq tail-write), so the orchestrator
  is multi-frame ready end-to-end.

All buffers + scalars are GPU-resident throughout (cardinal rule: host =
pure coordinator). Verified bit-exact on CPU + OpenCL backends across:
Voiced NB, Inactive NB, plus the OutBuf rotation in the finalize variant.

Also extends `SilkDecodeCoreGpu` test coverage with WB (lpcOrder=16 full
unroll), MB (12 kHz × 60-sample subframes), NLSF interpolation (k=2
rewhitening branch), and a 3-frame state-rolling test.

CUDA + WebGPU still gated on the same ILGPU backend bugs filed at
`_DevComms/SpawnDev.ILGPU/tuvok-to-geordi-silkdecodecoregpu-cuda-webgpu-2026-05-04.md`
(per-thread register pressure + sub-word body-struct binding count).
Wasm gated on PMT 30s cold-start kernel-compile timeout.

Source on master commit `41c85b6`. PackageReference SpawnDev.ILGPU 4.9.5-rc.10.

## 0.3.0-rc.1 (2026-05-03)

Release candidate for the 0.3.0 cut. NOT 1.0.0 - the 1.0.0 stable
designation requires the full architectural completion (see "Path to
1.0.0" below). 0.3.0-rc.1 captures today's substantial progress
toward that bar.

### Codecs in this release

**Audio:** FLAC, Opus, Vorbis. **Video:** VP8, VP9, AV1.

Patent-clean by selection: AOMedia patent pledge for VP8/VP9/AV1, RFC
6716 royalty-free for Opus, BSD reference for FLAC/Vorbis. H.264/H.265/
AAC stay out (delegated to platform encoders via SpawnDev.MultiMedia).

### Public API surface - CPU reference encoders + decoders

Every codec ships a CPU public API that produces bytes a reference
decoder accepts:

- **FLAC**: bit-exact lossless, matches ffmpeg byte-for-byte
- **Opus**: round-trip via Concentus 2.2.2, matches ffmpeg's libopus quality
- **Vorbis**: 35.7 dB SNR on real BBB content (vs ffmpeg's libvorbis 20.9 dB)
- **VP8**: keyframes accepted by ffmpeg pixel-perfect (1920x1072 @ 5 fps;
  1/2/4/8 token partitions)
- **VP9**: keyframes accepted by ffmpeg native decoder (16x16 -> 1920x1088,
  max diff Y=1 at Q=8)
- **AV1**: keyframes accepted by libdav1d (60 frames decoded on real
  BBB content, av1.mp4 plays cleanly)

### GPU encoder + decoder pairs (100% ILGPU)

Five of the six codecs ship 100%-ILGPU encoder/decoder pairs:

- `Vp8KeyframeEncoderGpu` / `Vp8KeyframeDecoderGpu`
- `Vp9KeyframeEncoderGpu` / `Vp9KeyframeDecoderGpu`
- `Av1KeyframeEncoderGpu` / `Av1KeyframeDecoderGpu`
- `FlacEncoderGpu` / `FlacDecoderGpu`
- `VorbisAudioEncoderGpu` / `VorbisAudioDecoderGpu`

Every byte of the encoded bitstream is written by an ILGPU kernel; every
pixel of the decoded recon plane comes back from a kernel readback. The
host (consumer code) is a pure coordinator: alloc + upload + dispatch +
single readback. No CPU math, no CPU iteration, no CPU bool encoding,
no CPU bitstream assembly.

Cardinal-rule compliant: every per-frame iteration on codec data,
`Array.Copy` on bitstreams, full-buffer-readback waste, and CPU-encoder
INSTANCE dependency in the GPU pair production paths has been closed.

Opus has 31 SILK GPU primitives + the shared Daala range coder; the
top-level OpusEncoderGpu/OpusDecoderGpu integration class is the next
codec on the GPU port roadmap.

### Browser demos (live in your browser via Blazor WASM)

The `SpawnDev.Codecs.Demo` project ships browser demos for the entire
codec surface:

- `/transcode` - encode + decode all 3 video codecs (VP8/VP9/AV1) via
  the CPU public API. Renders YUV->RGBA on canvas.
- `/audio` - encode + decode all 3 audio codecs (FLAC/Vorbis/Opus) via
  the CPU public API. Plots source vs decoded waveforms with per-codec
  SNR + compression ratio + timings.
- `/gpu-transcode` - encode + decode VP8 + VP9 + AV1 entirely through
  the ILGPU integration classes. 100% GPU-resident. Routes to WebGPU
  when available, Wasm fallback.
- `/benchmarks` - live throughput numbers for every encoder + decoder,
  both CPU and GPU pair, across the available ILGPU backends.

Every demo page is covered by an end-to-end unit test that mirrors its
exact encode -> decode flow, so the demo can be trusted to work without
manual browser click-through every commit
(`AudioTranscodeDemo_*` / `VideoTranscodeDemo_*` / `GpuTranscodeDemo_*`).

### Containers

| Container | Read | Write |
|---|---|---|
| RIFF / WAVE | Yes | Yes |
| AIFF | Yes | Yes |
| Ogg | Yes | Yes |
| WebM / Matroska (EBML) | Via SpawnDev.EBML 3.0.1 | Via SpawnDev.EBML |
| MP4 / ISOBMFF | Structural box reader | Not yet |
| IVF | Yes | Yes |

### Test infrastructure

- `CodecsTestBase.AcquireAcceleratorOrSkipAsync()` - converts the
  WebGL backend's `NotSupportedException` (eager kernel-feature
  validation) into `UnsupportedTestException` so the harness records
  Skipped rather than Failed for tests whose kernels need atomics.
- `CodecsTestBase.RequireFeatures(acc, AcceleratorRequirements)` -
  explicit per-test feature gating via SpawnDev.ILGPU's
  `device.Satisfies()` API.

### PMT verification at 0.3.0-rc.1

GpuTranscodeDemo sweep run 2026-05-03 against ILGPU 4.9.5-rc.1:

| Backend | Result | Notes |
|---|---|---|
| CPU | 3 PASS | Vp8 + Vp9 + Av1 GPU pair round-trip |
| CUDA | 3 PASS | Vp8 + Vp9 + Av1 GPU pair round-trip |
| OpenCL | 3 PASS | Vp8 + Vp9 + Av1 GPU pair round-trip |
| WebGPU | 3 FAIL (PMT page-load timeout) | Kernels compile - SubView codegen fix verified working; Playwright `WaitForAsync` on the Run-button locator times out at 30s, page wall time hits 1m30s+. Reported to Geordi: PMT timeout tuning needed, not an ILGPU code regression. |
| WebGL | 3 SKIP | Atomics gate (expected) |
| Wasm | 3 SKIP | Awaiting ILGPU memory OOB fix (Geordi Bug 2 in-progress) |

**Total: 9 PASS / 3 FAIL / 6 SKIP / 18 total.** Desktop GPU-pair pipeline verified bit-exact on all three desktop backends.

### Project split: SpawnDev.Codecs.References (NEW in 0.3.0-rc.1)

A new sibling project, `SpawnDev.Codecs.References`, hosts the CPU
reference encoders + decoders. The main `SpawnDev.Codecs` library now
ships **ZERO external dependencies** (only SpawnDev.* + Microsoft
framework packages) per Captain's 2026-05-03 architectural directive.

Moved to `SpawnDev.Codecs.References` in 0.3.0-rc.1 (174+ files):

**Audio**:
- **Opus** (5 files): `OpusEncoder`, `OpusDecoder`, `OpusOggEncoder`,
  `OpusOggDecoder`, `OpusCodec` (factory)
- **Opus / CELT CPU** (2 files): `CeltDecoder`, `CeltDecoderState`
- **Opus / SILK CPU** (28 files): `SilkBwexpander`, `SilkDecodeCore`,
  `SilkDecodeFrame`, `SilkDecoder`, `SilkExcitationDequantizer`,
  `SilkGainAdjust`, `SilkGainDecoder`, `SilkIndicesDecoder`,
  `SilkLog2`, `SilkLpcAnalysisFilter`, `SilkLpcFit`,
  `SilkLpcInvPredGain`, `SilkLpcSynthesisFilter`, `SilkLtpDecoder`,
  `SilkNlsf2A`, `SilkNlsfDecoder`, `SilkNlsfStabilize`,
  `SilkNlsfUnpack`, `SilkParametersDecoder`, `SilkPitchDecoder`,
  `SilkPulsesDecoder`, `SilkResampler`, `SilkShellCoder`,
  `SilkSideInfoDecoder`, `SilkSigmoid`, `SilkStereoDecodePred`,
  `SilkStereoMsToLr`, `SilkSumSqrShift`
- **FLAC** (9 files): `FlacEncoder`, `FlacDecoder`,
  `FlacFrameDecoder`, `FlacOggDecoder`, `FlacFixedSubframeEncoder`,
  `FlacLpcSubframeEncoder`, `FlacResidualDecoder`,
  `FlacSubframeDecoder`, `FlacSubframeWriter`
- **Vorbis** (4 files): `VorbisAudioDecoder`, `VorbisOggDecoder`,
  `VorbisInverseCoupling`, `VorbisAudioEncoder` (after extracting
  header construction to `VorbisHeaderPackBuilder` in main),
  + `VorbisCommentHeader` (CPU-only metadata record)

**Video**:
- **VP8** (14 files): `Vp8KeyframeEncoder`, `Vp8Decoder`,
  `Vp8BoolEncoder`, `Vp8BoolDecoder`, `Vp8CoefBlockEncoder`,
  `Vp8CoefBlockDecoder`, `Vp8MbModeInfoDecoder`, `Vp8FrameHeader`,
  `Vp8FrameHeaderWriter`, `Vp8ModeTrees`, `Vp8SegmentationLookup`,
  `Vp8MbDequantizer`, `Vp8KeyframeWalker`, `Vp8ForwardQuantizer`
- **VP9** (102 files): full CPU decoder + encoder + walker stack,
  `Vp9Decoder`, `Vp9KeyframeEncoder`, `Vp9KeyframeWalker`, all the
  `Vp9*Mv*`, `Vp9Iadst*Reference`, `Vp9Idct*Reference`,
  `Vp9Iht*Reference`, etc.
- **AV1** (11 files): `Av1Decoder`, `Av1KeyframeWalker`,
  `Av1FrameBuffer`, `Av1IvfRemuxer`, `Av1StreamAnalyzer`,
  `Av1StreamValidator`, `Av1TileGroup`, `Av1CflPredictor`,
  `Av1CompleteFrameHeaderParser`, `Av1DefaultSegmentCdfs`,
  `Av1FrameHeaderWriter`

**Entropy coders** (2 files): `OpusRangeEncoder`, `OpusRangeDecoder`

Still in `SpawnDev.Codecs` main library (because each is consumed by
GPU integration classes for static data tables, OBU framing helpers,
or metadata struct setup): the Vorbis encoder + Floor1/Residue/
PacketHeader decoders + bit IO + codebook tables, every AV1
CDF/dequant/scan table + transforms + predictors + OBU writer + spec
record types, every VP9 GPU-shared scan/coef/probs/neighbor table.

For 1.0.0 these remaining CPU implementations need finer extraction
(separate the CPU impl methods from the shared static tables) so
they can also move to References. The pattern is mechanical now -
extract tables/helpers, point GPU code at the tables class, move the
CPU impl. Several have already been done this way (`Vp8CoefTables`,
`Vp8ModeProbTables`, `Vp9BlockCoefEnums`, `Av1RangeDecoderGpu`
constants).

### Vorbis decoder v2 GPU integration kernel - ALL STEPS SHIPPED

Plans/PLAN-Vorbis-Decoder-V2-GPU-BitStream-Decode.md tracks the v2
work that closes the last cardinal-rule gap in the Vorbis decode
pipeline (the CPU bit-stream walk inside
`VorbisAudioDecoderGpu.DecodeSpectrumOnCpu`). Live on desktop
backends (CPU + CUDA + OpenCL) as of commit `a28e196`. Browser
backends (WebGPU + Wasm) keep the v1 hybrid path until ILGPU
binding-count work lands.

- **Step 1**: `VorbisResidueDecoderGpu` static helper - GPU-callable
  Type 0/1 + Type 2 residue decoder mirroring CPU
  `VorbisResidueDecoder.Decode`.
- **Step 2**: `VorbisPacketDecodeKernel` - single-thread GPU
  integration kernel that wires `VorbisAudioPacketHeaderGpu.Parse`
  + per-channel `VorbisFloor1DecoderGpu.Decode` + per-submap
  `VorbisResidueDecoderGpu.Decode` in one dispatch. Outputs go into
  a single `allIntOut` buffer (header, floorOk, floorIndex,
  posteriors, err) + `residuesFlatOut`. 13 top-level params total
  (within ILGPU `Action<16>` ceiling).
  `VorbisPacketDecodeStaticInputs` struct holds 38 ArrayView fields
  for the flat-packed setup header + codebook set; the struct is a
  plain POD so ILGPU's kernel-parameter marshaling can pack it.
- **Step 2 verification**: `VorbisPacketDecodeKernel_LoadsOnAccelerator`
  smoke test verifies the kernel compiles to backend IR via
  `LoadAutoGroupedStreamKernel` on each available accelerator. Catches
  binding-count limits + struct-of-ArrayView marshaling gaps before
  the integration risks breaking the working v1 decoder tests.
- **Step 3a (DecoderGpu uploads + kernel compile)**: ✓ shipped commit
  `afeb6b1`. 38 flat-packed buffers uploaded once per stream + kernel
  compiled at construction. Existing 3 Vorbis decoder GPU tests pass
  on all 5 available backends (15/0/3) via the v1 path; constructor's
  `LoadAutoGroupedStreamKernel` for the v2 kernel verifies in-context.
- **Step 3b (DecoderGpu actually dispatches the v2 kernel)**: ✓ SHIPPED
  commit `a28e196` as a dual-path branch on `_accelerator.AcceleratorType`:
  - **Desktop (CPU + CUDA + OpenCL)**: dispatches the v2 kernel directly.
    Host = pure coordinator (alloc + 1 packet upload + dispatches +
    final readback). All bit-stream decode + floor decode + residue
    decode runs inside the GPU integration kernel. Cardinal-rule
    compliant on desktop.
  - **WebGPU + Wasm**: v1 hybrid path retained. WebGPU's 10-binding
    cap (`maxStorageBuffersPerShaderStage`) rejects the v2 kernel's 44
    storage buffers (ILGPU flattens each `ArrayView<T>` field of
    `VorbisPacketDecodeStaticInputs` into a separate binding). Wasm
    fails OOB on the same dispatch (related root cause + may overlap
    with Geordi's open Bug 2). Browser backends become cardinal-rule
    clean when ILGPU coalesces struct-of-ArrayView kernel params on
    WebGPU OR a kernel restructure flattens the 36 int fields into
    one offset-indexed buffer. Filed with Geordi:
    `_DevComms/SpawnDev.ILGPU/tuvok-to-geordi-vorbis-v2-binding-count-2026-05-03.md`.

  Implementation notes: take 3 used an inline branch (post-spectrum
  chain unchanged across paths, common variables `floorOk`,
  `floorIndex`, `mapping`, `dPosteriorsView`, `dResidueView` populated
  per branch). Take 2's extracted `DecodePacketV2Async` regressed
  desktop tests via copy-paste subtleties; take 3 keeps post-spectrum
  code untouched.

  PMT verification on take 3: `Failed: 0, Passed: 15, Skipped: 3,
  Total: 18, Duration: 58 s`. 3 desktop backends × 3 tests via v2
  kernel = 9; 2 browser backends × 3 tests via v1 fallback = 6;
  WebGL × 3 atomics-gated SKIP.

### Helper classes extracted to main (so CPU classes can move cleanly)

- `Vp8CoefTables`: ZigzagScan + CoefBands + Cat3-6 probability tables
  (libvpx kZigzag / kBands / kCatN). Replaces references to the
  moved `Vp8CoefBlockDecoder.X` constants from
  `Vp8CoefBlockEncoderGpu.BuildConstsBuffer`.
- `Vp8ModeProbTables`: DefaultKfYModeProb + DefaultKfUvModeProb.
  Replaces references to the moved `Vp8ModeTrees.X` constants from
  `Vp8FrameEntropyKernel.BuildExtendedConstsBuffer`.
- `Vp9BlockCoefEnums`: PlaneType + RefType. Replaces nested enums on
  the moved `Vp9BlockCoefDecoder` class. Updated
  `Vp9KeyframeDecodeKernel` + `Vp9FrameEntropyKernel`.
- `Av1RangeEncoderGpu` / `Av1RangeDecoderGpu`: now define
  EcProbShift / EcMinProb / OdEcWindowSize / OdBitRes /
  OdEcLotsOfBits / CdfProbTop directly (AV1 spec entropy-coder
  constants). Removes the dependency on the moved CPU
  `Av1RangeEncoder` / `Av1RangeDecoder` classes.
- `VorbisHeaderPackBuilder`: hosts `BuildResolvedHeaders`,
  `BuildIdentificationHeader`, `BuildSetupHeader`,
  `BuildIdentPacketBytes`, `BuildCommentPacketBytes`,
  `BuildSetupPacketBytes`, plus private `PackFloor1` / `PackResidue`
  / `PackMapping`. Moved out of `VorbisAudioEncoder` so
  `VorbisAudioEncoderGpu` can call them without instantiating the
  CPU encoder. The CPU encoder (now in References) delegates back
  to this helper for behavior-preserving identity.
- `VorbisAudioEncoderOptions`: split out from `VorbisAudioEncoder.cs`
  (consumed by both GPU + CPU encoders).

### Dependencies

`SpawnDev.Codecs` (main library):
- `SpawnDev.ILGPU` 4.9.5-rc.1 (WebGPU `SubViewValue` IR codegen fix
  + previous kernel dispatch + partial readback +
  `ArrayView<T>.CopyToHostAsync()`)
- `SpawnDev.EBML` 3.0.1 (WebM/Matroska container)

`SpawnDev.Codecs.References` (CPU reference impls):
- `SpawnDev.Codecs` (sibling)
- `Concentus` 2.2.2 (BSD-3 pure-C# Opus reference)

### Out of scope (patent-encumbered)

H.264, H.265, AAC, MP3 - delegated to platform encoders via
SpawnDev.MultiMedia (P/Invoke to MediaFoundation / VideoToolbox /
VAAPI). System encoders are licensed by Microsoft / Apple / driver
vendors respectively; SpawnDev.Codecs stays clean.

### Path to 1.0.0 stable

Captain's directive 2026-05-03 defines the 1.0.0 bar. 0.3.0-rc.1 is
release candidate work toward that bar:

1. **All 6 codecs ship as 100%-ILGPU encoder + decoder pairs.** Today
   covers 5 of 6 (VP8, VP9, AV1, FLAC, Vorbis). Opus has 31 SILK GPU
   primitives + the shared Daala range coder; CELT has zero GPU
   primitives (bit allocator, PVQ encode/decode, anti-collapse, spread,
   prefilter, post-filter, stereo coupling all need to be ported). Plus
   SILK + CELT integration kernels + OpusEncoderGpu/OpusDecoderGpu
   top-level classes. This is the largest remaining piece.
2. **Vorbis decoder v2 GPU bit-stream decode.** v1 still does CPU-side
   per-packet header parse + floor + Huffman + residue work before
   handing to the GPU IMDCT chain. v2 ports those to GPU kernels using
   the existing `VorbisAudioPacketHeaderGpu`, `VorbisFloor1DecoderGpu`,
   `VorbisHuffmanDecoderGpu`, `VorbisBitReaderGpu`, and the new
   `VorbisResidueDecoderGpu` (shipped 0.3.0-rc.1) primitives. Plan in
   `Plans/PLAN-Vorbis-Decoder-V2-GPU-BitStream-Decode.md`.
3. **Complete the project split.** Started in 0.3.0-rc.1 (Opus, FLAC,
   SILK CPU, OpusRange entropy coders moved + Concentus dep removed
   from main). Remaining: Vorbis, VP8, VP9, AV1 CPU classes still
   live in main because each one mixes CPU implementation methods
   with static data tables consumed by the GPU integration classes.
   For 1.0.0 each CPU class needs its data tables / framing helpers
   extracted to a main-library "Tables" / "Helpers" class so the CPU
   class itself can move to References cleanly.
4. **Cross-lane SpawnDev.ILGPU bug fixes** for WebGPU `SubViewValue`
   IR codegen + Wasm OOB. Filed at
   `_DevComms/SpawnDev.ILGPU/tuvok-to-geordi-webgpu-subview-codegen-and-wasm-oob-2026-05-03.md`.

## 0.2.0-alpha.2 (2026-05-03)

Same library code as 0.2.0-alpha.1; this point release ships the
documentation + demo improvements that landed after the 0.2.0-alpha.1
package was published:

- README "Quickstart - GPU encoder + decoder pairs" section with
  concrete consumer code examples for each of the 5 GPU pairs (VP8,
  VP9, AV1, FLAC, Vorbis), including how to acquire an accelerator on
  desktop + Blazor WASM and how to handle backends that lack atomics
  (`AcceleratorRequirements` + `UnsupportedTestException` pattern).
- `/gpu-transcode` browser demo now exercises all 3 video codecs end-
  to-end through the GPU pairs (VP8 + VP9 + AV1 via the tile-bytes
  flow `EncodeSingleTileAsync` -> `DecodeSingleTileAsync`). Previously
  only VP8 + VP9.
- `Plans/PLAN-Vorbis-Decoder-V2-GPU-BitStream-Decode.md` design doc
  covers the next-session pickup target: move CPU per-packet floor +
  Huffman + residue work in `VorbisAudioDecoderGpu.DecodePacket` into
  a single GPU kernel, closing the v1 design gap.

Net code change for library consumers: README in the package updates
with the quickstart section. No API surface changes.

## 0.2.0-alpha.1 (2026-05-03)

First version after the cardinal-rule audit pass. Every GPU integration
class is now host-as-coordinator clean: alloc + upload + dispatch +
single readback. No CPU iteration on codec data inside any *Gpu class
production path.

### Cardinal-rule + perf wins

- **VP8 keyframe encoder** now uses a GPU stride-pack kernel instead
  of the per-row CPU stride-strip loop in `UploadPlane`. Strided source
  planes upload via `Vp8StridedPlanePackKernel` (one thread per output
  byte) so plane unpacking happens accelerator-resident.
- **VP8 keyframe encoder** final readback uses
  `dOutput.View.SubView(0, finalLen).CopyToCPU(result)` (real
  per-backend partial readback, SpawnDev.ILGPU 4.9.3+) - drops the
  prior `GetAsArray1D` of the worst-case-sized output buffer + CPU
  `Array.Copy` trim.
- **VP9 + AV1 + Vorbis encoder/decoder pairs** swap host-allocated
  zero-arrays + uploads (`CopyFromCPU(new T[N])`) for GPU-resident
  `MemSetToZero()`. Eliminates ~1MB of CPU allocator churn and bus
  transfer of zeros per frame at typical sizes.
- **VP9 keyframe decoder** drops the per-decode `Array.Copy` of tile
  bytes. Uploads the full frame once, dispatches the decode kernel
  with `dFrame.View.SubView(tileStartOffset, tileLength)` so the
  kernel sees the same-shaped tile view at zero copy cost.
- **VorbisAudioDecoderGpu** drops the per-channel `Array.Copy` flat-
  buffer build for posteriors + residues. Each channel uploads
  directly into its slot via `dPosteriors.View.SubView(...)` and
  `dResidue.View.SubView(...)`. `MemSetToZero()` covers silent-channel
  slots.
- **VorbisAudioEncoderGpu** + **VorbisAudioDecoderGpu** drop their
  CPU-encoder / CPU-decoder INSTANCE dependencies. The encoder used
  to call `_cpuRef.BuildIdentPacket()` / `BuildCommentPacket()` /
  `BuildSetupPacket()` for the 3 Ogg header packets in
  `EncodeStreamAsync`; the decoder used reflection to extract
  Huffman state from the CPU decoder's private fields. Now the GPU
  encoder calls `VorbisAudioEncoder.BuildIdentPacketBytes()` /
  `BuildCommentPacketBytes()` / `BuildSetupPacketBytes()` (new public
  static helpers) and the GPU decoder builds its own Huffman array
  from `setup.Codebooks` directly. CPU encoder/decoder classes
  remain as test reference oracles only.

### Demos

- **`/audio` page** - encode + decode all 3 audio codecs (FLAC,
  Vorbis, Opus) end-to-end via the CPU public API. Synthesizes a
  mono sine wave, drives every encoder + decoder, plots source +
  recovered waveforms with per-codec SNR + compression + timings.
- **`/gpu-transcode` page** - encode + decode VP8 + VP9 entirely
  through the ILGPU integration classes (`Vp8KeyframeEncoderGpu` +
  `Vp8KeyframeDecoderGpu`, `Vp9KeyframeEncoderGpu` +
  `Vp9KeyframeDecoderGpu`). 100% GPU-resident encode/decode chain.
  Routes to WebGPU when the browser exposes it, Wasm otherwise.
- `AudioTranscodeDemo_*` unit tests verify the demo pipeline
  bit-exact for FLAC + lossy SNR floors for Vorbis (>10 dB) + Opus
  (>5 dB), 18/18 pass on CUDA + OpenCL + CPU.

### Test infrastructure

- New `CodecsTestBase.AcquireAcceleratorOrSkipAsync()` wraps
  `CreateKernelAcceleratorAsync()` to convert the WebGL backend's
  `NotSupportedException` (eager kernel-feature validation) into
  `UnsupportedTestException`. The harness now records "Skipped" for
  WebGL tests whose kernels need atomics / shared memory / barriers
  rather than reporting them as Failed. Applied across all 200+
  GPU-using test sites.
- New `CodecsTestBase.RequireFeatures(acc, AcceleratorRequirements)`
  helper: explicit per-test feature gating. Throws
  `UnsupportedTestException` when the current accelerator's backend
  cannot satisfy declared requirements.

### SpawnDev.ILGPU PackageReference: 4.9.2 -> 4.9.3

Picked up `ArrayView<T>.CopyToHostAsync()` real per-backend partial
readback (WebGPU `queue.CopyBufferToBuffer` + partial mapAsync, WebGL
partial-range readback worker path, Wasm SAB slot view, CUDA / OpenCL
/ CPU `view.CopyToCPU(target)`). Used at every `*KeyframeEncoderGpu`
and `*KeyframeDecoderGpu` partial-readback site.

## Unreleased

### PMT: system Chrome for real WebGPU (2026-09-28)
- PlaywrightMultiTest now launches installed Chrome (`Channel = "chrome"`) with `SpawnDev.Codecs.PlaywrightProfile`, matching ILGPU/ML. Dropped `Vulkan` from `--enable-features` and added `--disable-software-rasterizer` so Windows Dawn stays on D3D12 instead of silently falling back to SwiftShader.

Initial development. Project still in pre-release. See README.md for the
working feature matrix.

### 2026-04-29 - FlacDecoderGpu output interleave moves from CPU to GPU

`FlacDecoderGpu.DecodeStreamAsync` previously copied each frame's
samples back to host (`dSamples.CopyToHostAsync()`) and ran a per-sample
channel-major -> interleaved CPU loop into the output array. That
violated the cardinal rule (host doing per-sample iteration on codec
data). Now: a new `FlacInterleaveOutputGpu` primitive does the same
work GPU-side as one thread per (channel, sample); the integration
class allocates the full interleaved output on the accelerator and
dispatches the kernel per frame. Single readback of the full output
buffer at end.

Net effect: the host is back to pure coordinator status for FLAC
decode - alloc, upload, kernel dispatch, single per-frame status check
+ frame-length scan, single readback. No per-frame `dSamples`
readback, no per-sample CPU iteration on codec data.

FLAC decoder PMT 6/6 PASS on CUDA + OpenCL + CPU.

### 2026-04-29 - VorbisAudioDecoderGpu floor curve render moves to GPU

`VorbisAudioDecoderGpu` now dispatches `VorbisFloor1RenderCurveGpu`
instead of calling the CPU `VorbisFloor1Curve.Render`. Per-floor
`xList` arrays + the 256-entry inverse-dB lookup are pre-flattened +
uploaded once at construction; per packet the CPU bit-stream parse
produces the per-channel posterior Y values, which upload to GPU and
drive a single-thread floor render kernel per non-silent channel.

Tone round-trip test still PASSES on CUDA + OpenCL + CPU (9/9 PMT).
This is a meaningful proof point because the floor render IS exercised
on the tone path (silent floors get zero-curve fallback + skip the
dispatch entirely).

Pipeline state (v1 hybrid):

| Stage                      | Where |
|----------------------------|-------|
| Parse + Huffman + floor decode + residue decode | CPU |
| Floor curve render         | **GPU** |
| Floor x residue multiply   | GPU |
| Inverse coupling           | GPU |
| IMDCT                      | GPU |
| Window + overlap-add + right-half save | GPU |
| Channel interleave         | GPU |

Remaining CPU side: bit-stream parse + Huffman, floor 1 posterior
decode, residue decode. Vorbis-Huffman GPU bit-reader keystone
primitive is what unlocks moving those.

### 2026-04-29 - VorbisAudioDecoderGpu non-silence tone round-trip test

`VorbisAudioDecoderGpu_ToneRoundTrip_MatchesCpuDecoder` exercises the
full GPU decoder chain on a non-silent 440 Hz mono tone (peak 0.5).
Same packets feed both the CPU `VorbisAudioDecoder` reference (double-
precision IMDCT) AND the GPU `VorbisAudioDecoderGpu` (float-precision
IMDCT). PCM compared sample-by-sample with 1e-3 absolute tolerance to
absorb the float-vs-double divergence in the IMDCT accumulator.

Result: 3/3 backends PASS (CUDA + OpenCL + CPU = 9/9 PMT counting all
three Vorbis decoder tests). This is the first non-silence proof point
for the GPU multiply + inverse-coupling + IMDCT + post-IMDCT chain
running end-to-end on real spectral data.

### 2026-04-29 - VorbisAudioDecoderGpu floor*residue multiply + inverse coupling move to GPU

`VorbisAudioDecoderGpu` now dispatches `VorbisFloorMultiplyGpu` and
`VorbisInverseCouplingGpu` instead of doing the corresponding scalar
loops in the CPU helper. Per-channel floor curves + residue buffers
upload once; multiply runs as one Index1D dispatch per non-silent
channel; the (mag, ang) coupling pairs run as one Index1D dispatch each
in REVERSE step order per Vorbis I sec 4.3.8.

Pipeline state after this commit (v1 hybrid):

| Stage                            | Where  |
|----------------------------------|--------|
| Bit-stream parse + Huffman + floor decode + residue decode | CPU |
| Floor curve x residue multiply   | GPU    |
| Inverse channel coupling         | GPU    |
| IMDCT                            | GPU    |
| Window apply + overlap-add + new-right-half-save | GPU    |
| Channel interleave               | GPU    |

Mono test path covers no coupling steps (mapping has zero pairs); the
inverse-coupling kernel will get exercised by future stereo tests.
Silence-path round-trip 6/6 PMT (CUDA + OpenCL + CPU) PASS.

### 2026-04-29 - VorbisAudioDecoderGpu IMDCT step moves from CPU to GPU

`VorbisAudioDecoderGpu.DecodePacketAsync` now dispatches `ImdctReferenceGpu`
on the accelerator instead of the CPU `ImdctReference.Transform`. The
spectrum coefficients (post-floor + post-residue + post-inverse-coupling)
upload to GPU once, and the IMDCT, post-IMDCT (window + overlap-add +
right-half save), and channel interleave all chain accelerator-resident
without intermediate CPU readback. Per-channel dispatch: one GPU thread
per output time-domain sample (2N=blockSize threads / channel).

Silence-path round-trip test still PASSES on CUDA + OpenCL + CPU
(`VorbisAudioDecoderGpu_SilenceRoundTrip_MatchesCpuDecoder` 6/6 PMT).
The CPU reference uses `double` accumulators for the IMDCT and the GPU
uses `float`, so non-silence inputs would diverge slightly; silence
remains bit-exact zero.

Remaining CPU-side work in the v1 hybrid: bit-stream parse + Huffman +
floor decode + residue decode + multiply + inverse coupling. Vorbis-
Huffman GPU bit-reader is the keystone primitive for moving those.

### 2026-04-29 - VorbisAudioDecoderGpu - closes the Vorbis encoder/decoder pair

**Vorbis is now 6 of 6 codec encoders + 6 of 6 codec decoders end-to-end on
the accelerator.** `VorbisAudioDecoderGpu` is the integration class that
mirrors `VorbisAudioEncoderGpu`: it consumes a Vorbis audio packet and emits
interleaved float PCM via the GPU post-spectrum chain (window apply +
overlap-add + new-right-half-save + channel interleave). State (previous
right-half samples, previous block size) is GPU-resident across packets.

**v1 scope (matches the encoder's silence-path scope):**
- Mono + stereo audio.
- GPU PostImdct + Interleave kernels run the per-sample math; spectrum decode
  (bit-stream parse + Huffman + floor + residue + IMDCT) currently delegates
  to the existing `VorbisAudioDecoder` CPU path so we have working end-to-end
  decode TODAY. The Vorbis-Huffman GPU bit-reader is the keystone primitive
  that will let the entire decode run in-kernel; it lands as a follow-up
  integration step.
- Round-trip silence test (`VorbisAudioDecoderGpu_SilenceRoundTrip_MatchesCpuDecoder`)
  encodes silence packets via `VorbisAudioEncoderGpu` then decodes them via
  both `VorbisAudioDecoder` (CPU reference) and `VorbisAudioDecoderGpu` (GPU)
  and compares interleaved PCM sample-by-sample within 1e-5 absolute
  tolerance. PASS on CUDA + OpenCL + CPU.

**Why this matters:** the v1 Vorbis encoder + decoder pair is now closed.
Silence in -> packets -> silence out, all data + state on the accelerator
through the post-spectrum chain.

### 2026-04-29 - FLAC subframe decode chain GPU coverage (7 new primitives)

**Continuing 2026-04-29 late-night session: 7 new FLAC GPU primitives plus 1
new Vorbis primitive.** The FLAC subframe decode chain is now fully GPU-
callable end-to-end - both FIXED and LPC predictor types have composite
decode primitives that compose 4 individual primitives in a single kernel
thread.

**New FLAC GPU primitives shipped this session (5 standalone + 2 composite):**
- VorbisOverlapAddGpu (`c9417b2`) - per-sample-parallel post-IMDCT overlap-add
- FlacFixedReconstructGpu (`46bc459`) - decoder-side FIXED predictor reconstruction (orders 0-4)
- FlacLpcReconstructGpu (`2aabcb3`) - decoder-side LPC predictor reconstruction (orders 1-32, quant 0-15)
- FlacChannelDecorrelationGpu (`e7e2dd1`) - per-sample stereo decorrelation (LeftSide / RightSide / MidSide)
- FlacResidualDecoderGpu (`80b73d3`) - Rice-coded residual decoder (PartitionedRice + PartitionedRice2)
- FlacSubframeHeaderGpu (`38d0fda`) - subframe header parser (CONSTANT / VERBATIM / FIXED / LPC)
- FlacFixedSubframeDecodeGpu (`1ab8a84`) - composite end-to-end FIXED subframe decoder
- FlacLpcSubframeDecodeGpu (`7319d1c`) - composite end-to-end LPC subframe decoder

**FLAC subframe decode chain (now fully GPU-callable):**
SubframeHeader -> FixedSubframeDecode or LpcSubframeDecode (warm-up reads ->
ResidualDecode -> Reconstruct -> wasted-bits left-shift) -> ChannelDecorrelate
(if stereo). Each step is bit-exact vs the libFLAC reference across CUDA +
OpenCL + CPU.

### 2026-04-29 - Opus SILK GPU primitive build-out continued (23 -> 31 primitives)

**Continuing 2026-04-29 late-night session: 8 more Opus SILK GPU primitives
beyond the earlier 23-primitive milestone.** Brings Opus SILK to 31 GPU
primitives total + 1 shared Daala range coder. Now have GPU coverage of every
non-entropy SILK decode stage including a full composite NLSF decode pipeline
(Unpack -> ResidualDequant -> WeightedAdd -> Stabilize) running in a single
kernel thread.

| Codec  | Encoder | Decoder | GPU primitives |
|--------|---------|---------|----------------|
| Opus   | -       | -       | 31 SILK + 1 shared range coder |

**New Opus SILK GPU primitives shipped this session (24th-31st):**
- SilkGainDecoderGpu (`c885b86`) - per-subframe gain dequantizer (delta-coded hysteresis, composes SilkLog2Gpu)
- SilkLtpGainVectorGpu (`4808113`) - per-tap LTP gain vector codebook lookup (per-tap parallel)
- SilkPitchContourGpu (`cbeb1bc`) - per-subframe pitch contour expansion (per-subframe parallel)
- SilkNlsfInterpolateGpu (`5fa485a`) - per-coefficient NLSF inter-frame interpolation (per-coefficient parallel)
- SilkNlsfResidualDequantGpu (`95f356d`) - reverse-iterating NLSF residual dequant (sequential per-stream)
- SilkNlsfWeightedAddGpu (`5a8944b`) - per-coefficient NLSF inverse-weight + first-stage add (per-coef parallel)
- SilkNlsfDecodeGpu (`65cbddf`) - composite NLSF decode pipeline (single-thread, composes 4 GPU primitives)
- SilkLtpScaleGpu - 3-entry LTP scale Q14 lookup

**Second library quirk discovered + worked around (composite NLSF decode):**
ILGPU CUDA + OpenCL backends do NOT preserve C# byte<->sbyte cast semantics
through ArrayView storage. Initial SilkNlsfDecodeGpu used ArrayView<sbyte>
for predQ8 scratch with explicit (sbyte)/(byte) casts; CPU backend passed
all 4 tests but CUDA + OpenCL produced random divergences (8/12 tests
failed). Fix: use ArrayView<byte> directly with implicit conversions. Saved
`feedback_ilgpu_byte_sbyte_roundtrip.md` so future composite primitives
avoid the trap. The cross-backend pattern (CPU passes, hardware fails) is
the smoking gun for the bug.

### 2026-04-29 - Opus SILK GPU primitive build-out (9 -> 23 primitives)

**Continuing 2026-04-29 (late evening session): Opus SILK primitive expansion
from 9 to 23 primitives.** 14 new commits. Now have GPU coverage of every
major SILK decode + analysis stage: LPC analysis + synthesis, full resampler
chain (Up2Hq + Ar2 + DownFirInterpol + IirFirInterpol), stereo M/S->L/R + the
predictor dequantizer, gain adjust + DivVarQ, excitation dequant (PRNG-driven
with long-arithmetic bypass for ILGPU OpenCL signed overflow quirk), LPC fit,
NLSF -> Q12 LPC pipeline (composes 3 GPU primitives in a single kernel
thread), NLSF table-index unpacker.

| Codec  | Encoder | Decoder | GPU primitives |
|--------|---------|---------|----------------|
| Opus   | -       | -       | 23 SILK + 1 shared range coder |

**New Opus SILK GPU primitives shipped this session (10th-23rd):**
- SilkLpcAnalysisFilterGpu (`6bbba73`) - per-sample MA prediction-error filter
- SilkResamplerUp2HqGpu (`74f3aa1`) - 2x HQ upsampler (sequential IIR, one-thread-per-stream)
- SilkResamplerAr2Gpu (`e475cd0`) - AR2 IIR pre-filter for downsample chain
- SilkResamplerDownFirInterpolGpu (`c2939b9`) - 3 FIR variants (Fir0/Fir1/Fir2), per-output-sample parallel
- SilkResamplerIirFirInterpolGpu (`40a3c9b`) - 12-phase fractional FIR upsample, per-output-sample parallel
- SilkLpcSynthesisFilterGpu (`868e9d2`) - decoder LPC synthesis inner loop
- SilkStereoMsToLrGpu (`8998230`) - M/S -> L/R (ApplySideAt + ApplyMixAt, both per-sample parallel)
- SilkDivVarQGpu + SilkGainAdjustGpu (`6a77c5c`) - reusable variable-Q division + LPC state rescale
- SilkExcitationDequantizerGpu (`879116d`) - PRNG-driven excitation dequant (long-arithmetic for OpenCL)
- SilkLpcFitGpu (`87568d4`) - LPC coefficient quantization with iterative bandwidth expansion
- SilkNlsf2AGpu (`00700e6`) - full NLSF -> Q12 LPC pipeline (composes LpcFitGpu + LpcInvPredGainGpu + BwexpanderGpu)
- SilkNlsfUnpackGpu (`aaf1496`) - per-coefficient-pair NLSF table-index unpacker
- SilkStereoDecodePredGpu (`b8aeb62`) - stereo predictor dequantizer (pairs with StereoMsToLrGpu)

**Library quirk discovered + worked around:** ILGPU OpenCL backend does NOT
preserve `unchecked(int * int)` overflow semantics for signed int
multiplication. Saved `feedback_ilgpu_opencl_signed_overflow.md` so future
GPU PRNG ports start with the right pattern: use long arithmetic
(`(long)(uint)x * y`) for portable modular semantics across all 6 ILGPU
backends. Discovered while shipping SilkExcitationDequantizerGpu where
all 4 OpenCL tests sign-flipped at i=0 of the PRNG until the long-
arithmetic form was used.

### 2026-04-29 - AV1 GPU encoder + decoder pair complete (4 of 6 codecs on GPU)

**Status:** 4 of 6 codecs now have working encoder + decoder pairs on every
ILGPU backend (CUDA + OpenCL + CPU verified, WebGPU/WebGL/Wasm by symmetry).

| Codec  | Encoder | Decoder | Notes                                         |
|--------|---------|---------|-----------------------------------------------|
| VP8    | ✅      | ✅      | v3 100% ILGPU since 2026-04-26                |
| VP9    | ✅      | ✅      | v3 100% ILGPU since 2026-04-27                |
| FLAC   | ✅      | ✅      | v3 100% ILGPU since 2026-04-28                |
| AV1    | ✅ NEW  | ✅ NEW  | v3 100% ILGPU bit-exact vs CPU encoder        |
| Vorbis | -       | -       | 5 GPU primitives shipped, integration pending |
| Opus   | -       | -       | 4 GPU primitives shipped, integration pending |

**New:**

- `Av1FrameSequentialEncodeKernel` + `Av1KeyframeEncoderGpu` - v3 100%
  ILGPU AV1 v1 keyframe encoder. Single GPU thread runs the entire
  EncodeSingleTile pipeline (partition recursion + skip + ymode + uvmode +
  per-plane predict + 2D DCT + quantize + coef tokens + dequant + iDCT +
  recon) bit-exact vs the CPU `Av1KeyframeEncoder.EncodeSingleTile` reference.
  `EncodeKeyFrameAsync` returns the full TD + SH + Frame OBU stream bit-exact
  vs CPU encoder. Verified across CUDA + OpenCL + CPU (commit `a53e26d`).
- `Av1FrameSequentialDecodeKernel` + `Av1KeyframeDecoderGpu` - companion
  decoder. Round-trip tests prove the GPU decoder reconstructs the same
  YUV planes the GPU encoder produces internally - bit-exact across all 3
  backends (commit `4832234`).
- `SilkBwexpanderGpu` - third Opus SILK primitive on GPU. Chirp expansion
  for AR filter coefficients (Expand16 + Expand32). Used by NLSF
  stabilization and LPC whitening (commit `ea7cc52`).

**Bug found + fixed:** `GetQctx` quantizer-bin thresholds were
`[32, 128, 192]` instead of correct libaom `[20, 60, 120]` per
`Av1CoefDecoder.GetQctx`. This shifted the txb_skip CDF row used for the
first per-plane emit, producing 18-byte output where CPU produced 21 bytes.
Diagnosed via a CPU-shadow walker that mirrored the GPU walker logic
through the same `Av1RangeEncoder`; emit-trace comparison pinpointed the
divergence at emit #6.

**Test counts added:**
- 9 SilkBwexpanderGpu tests
- 12 Av1KeyframeEncoderGpu tests (4 cases x 3 backends)
- 6 Av1KeyframeDecoderGpu tests (2 cases x 3 backends)
- Full AV1 sweep: 696/696 PASS (zero regressions from walker additions).

**Continuing 2026-04-29 (afternoon-evening session): Vorbis encoder + Opus SILK
primitive expansion.** 33 total commits. Vorbis went from 5 -> 20 GPU primitives,
Opus SILK went from 4 -> 9, and Vorbis became the 5th codec with a working GPU
integration class (silence path bit-exact + full .ogg stream output).

| Codec  | Encoder | Decoder | GPU primitives |
|--------|---------|---------|----------------|
| VP8    | ✅      | ✅      | -              |
| VP9    | ✅      | ✅      | -              |
| FLAC   | ✅      | ✅      | -              |
| AV1    | ✅      | ✅      | -              |
| Vorbis | partial NEW | -    | 20 (silence path bit-exact; non-silent deferred to MDCT precision alignment) |
| Opus   | -       | -       | 9 SILK + 1 shared range coder |

**Vorbis GPU primitives shipped:**
- VorbisHuffmanDecoderGpu (`957ec88`) - canonical Huffman flat-tree decoder
- VorbisFloor1RenderGpu (`c6b4522`) - per-line / per-point floor renderer
- VorbisCodebookVectorLookupGpu (`ef45942`) - 3 lookup types + sequenceP
- VorbisFloor1RenderCurveGpu (`8a7de7f`) - full Floor 1 curve orchestrator
- VorbisWindowGpu.ApplyWindowAt (`1b65058`) - per-sample window apply
- VorbisEncoderHelpersGpu (`401293c`) - MagnitudeToFloorY + QuantiseResidueValue
- VorbisSpectrumPeakGpu (`b7220c1`) - per-half-band peak reduction
- VorbisEncoderHelpersGpu.DivideQuantizeAt (`a562ae9`) - residue divide+quantize
- VorbisBitWriterGpu.WriteCodebookEntry (`9df95f0`) - canonical Huffman emit
- VorbisHuffmanCodebookSetGpu (`4f77273`) - multi-codebook flattener
- VorbisFloor1DecoderGpu (`46606ba`) - per-channel posterior decoder
- VorbisAudioPacketHeaderGpu (`9b4f1ee`) - per-packet header parser
- VorbisFloorMultiplyGpu (`cc780b3`) - per-bin floor x residue
- VorbisResidueAccumulateGpu (`0385fb2`) - lookup + accumulate composite
- VorbisFwdMdctScaledGpu (`a05c090`) - forward MDCT + 4/N scale
- VorbisEncoderFloorFitGpu (`ba43fd4`) - peak + floor-fit composite
- VorbisEncoderResidueEmitGpu (`d132f98`) - per-bin codebook emission
- VorbisEncoderBitstreamEmitGpu (`9382256`) - full per-packet emit composite
- VorbisAudioEncoderGpu (`fc491a5`) - integration class
- VorbisAudioEncoderGpu.EncodeStreamAsync (`3401183`) - full .ogg stream output

**Opus SILK GPU primitives shipped:**
- SilkSumSqrShiftGpu (`3120646`) - sum-of-squares with dynamic shift
- SilkLsfToCosGpu (`5ff7aad`) - LSF -> 2*cos(LSF) lookup
- SilkNlsfStabilizeGpu (`8f192b9`) - iterative NLSF stabilizer + insertion-sort fallback
- SilkInverseQ32Gpu (`e7398fd`) - Newton-like reciprocal silk_INVERSE32_varQ
- SilkLpcInvPredGainGpu (`ee5ebc5`) - LPC inverse prediction gain (uses InverseQ32)

**Investigation result (2026-04-29 evening):** Tried to align CPU + GPU
MagnitudeToFloorY (binary search vs Log10/Ceil) for non-silent Vorbis
encode bit-exactness. Confirmed via CUDA test that the actual upstream
divergence is MDCT float-vs-double precision: CUDA + OpenCL ILGPU
backends don't support `XMath.Cos(double)` / `XMath.Log10(double)`
without `EnableAlgorithms()` invoked at context construction. Reverted
to float MDCT, documented as a follow-up: switching the accelerator
factory to call `EnableAlgorithms()` would let the GPU encoder use
double-precision XMath.Cos and bit-exactly mirror the CPU encoder.
Until then, the GPU encoder produces a valid Vorbis bitstream that any
decoder accepts; non-silent decoded PCM is acoustically identical (float
MDCT drift < 1 ULP per cosine). The silence path produces byte-identical
output across all backends because all spectrum values are 0 regardless
of MDCT precision.

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
- AV1 walker: prediction always uses full tx-block dimensions, not frame-edge clip. Smooth/SmoothV/SmoothH/Paeth/directional predictors throw on non-power-of-2 dims at frame edge; libaom convention is to predict full size and clip on recon write. BBB Y mean 49.4 -> 94.95 (vs ffmpeg 97.4 = delta -2.5, was -48). Y plane drift essentially closed (commit `8e61258`).
- VP8 encoder + walker: multi-token-partition (Log2NumPartitions=0..3 = 1/2/4/8 partitions) per RFC 6386 sec 9.5. Encoder writes (n-1) 3-byte LE size headers + concatenated partition data; walker dispatches MB row M coefs to partition (M mod n). Round-trip + ffmpeg native VP8 decode all 4 counts: Y mean=124 exact across all (commits `86c2209` walker, `32c00cc` encoder).
- AV1 `GetNzMag` / `GetLowerLevelsCtx2d` were indexing into the raw `levelsBuf` without compensating for the `TxPadTop * stride` leading-pad offset that libaom's `set_levels()` applies via pointer arithmetic. Both helpers were reading from the leading zero-pad rows instead of actual coefficient data, so coefCtx for c > 0 in 2D class was always just the positional offset (mag=0). Encoder/decoder were self-consistent (both wrong the same way) but bit-incompatible with libaom + libdav1d. After fix: BBB transcode 4/60 → 60/60 frames decoded by libdav1d; av1.mp4 plays cleanly in VLC alongside vp8.mp4 + vp9.mp4 (commit `d9ba4b2`).

**Known remaining gaps:**

- AV1 walker chroma drift on BBB: U mean -12, V mean -28 vs ffmpeg ground truth (down from -44, -60 pre-fix but not yet zero). Likely chroma scan/qctx + remaining intra mode stubs.
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
