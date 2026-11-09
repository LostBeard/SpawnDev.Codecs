# PLAN: full VP9 decoder on the accelerator (inter frames, every profile-0 feature)

**Created:** 2026-11-08 (Geordi). **For:** Lore / SpawnGen Load Video (ComfyUI `LoadVideo` reads real .webm files),
item 3 of `_DevComms/2026-11-08-0919-Lore-TO-Geordi-codecs-for-comfyui-parity.md`. The encoder side (inter frames +
a crf knob for Save Video) is a separate plan that builds on this decoder.

## Source and licence
A port of **libvpx v1.17.0** (`vp9/decoder`, `vp9/common`, `vpx_dsp` C paths). libvpx is BSD-3-Clause plus Google's
royalty-free patent grant (LICENSE / PATENTS text read 2026-11-08); NOTICE.md already carries the libvpx entry.
**Never read FFmpeg's vp9 decoder (LGPL)**: FFmpeg is used only as a black box (`-c:v libvpx-vp9` decode, ffprobe).
ComfyUI (GPL-3) only as a black box: run its nodes, measure the files.

## Floor: libvpx conformance vectors
libvpx's own decoder test vectors (`test/test_vectors.cc`, from
`storage.googleapis.com/downloads.webmproject.org/test_data/libvpx/`, SHA-1 checked against `test/test-data.sha1`):
314 vectors, 26 MB, each with an `.md5` of every decoded frame (I420, cropped). Profile 0 (`vp90-2-*`): quantizers
00..63, every size 1x1..66x66 (02, 03), deltaq, bilinear / sharp filters, frame-parallel mode, tiles (08), odd sizes,
lf_deltas, show-existing-frame, intra-only frames, segmentation (15), skip, big superframes, resize with scaled
references (05, 14, 18, 21), SVC (22). Profiles 1..3 (`vp91/92/93`: 4:4:4, 10/12-bit) come after profile 0.
Done = every profile-0 vector's every frame MD5-equal on every backend. Then ComfyUI's own .webm files (generated
black-box from its Save WEBM / Save Video nodes) and the existing `Big_Buck_Bunny_180_10s.webm`.

## Architecture (host = coordinator)
Mirrors libvpx's row-MT decoder, which already separates **parse** (`parse_block`: mode info + dequantized
coefficients into buffers) from **recon** (`recon_block`) and loop filtering. Per frame:

1. **Host (metadata only):** container (WebM blocks / IVF frames, superframe index split), uncompressed header (frame
   type, show / refresh flags, ref slots, sizes, interp filter, loop filter + deltas, quant, segmentation params,
   tile counts), tile sizes. Buffer allocation and dispatch.
2. **Compressed header kernel (1 thread):** bool decoder; tx mode, coef / skip / mode / mv prob deltas into this
   frame's context on the device. The 4 saved frame contexts live on the device (reset rules as libvpx
   `setup_past_independence`).
3. **Parse kernel (1 thread per tile column, its tile rows in order):** partitions, mode info (`vp9_read_mode_info`:
   segment id with prediction from the previous segmentation map, skip, tx size, intra modes, ref frames, inter
   modes, MV candidate lists incl. the previous frame's MVs, MV decode, interp filter, sub-8x8 bmi), tokens ->
   dequantized coefficients + eobs, and symbol counts for adaptation (per column, summed afterwards). Writes the
   frame's MODE_INFO grid, MV buffer (for the next frame), segmentation map.
4. **Inter recon (parallel, per block x plane):** 8-tap regular / smooth / sharp and bilinear convolution
   (`vpx_convolve*`), compound averaging, sub-8x8 chroma MV averaging, edge extension equivalent to libvpx's
   `build_mc_border`, scaled references (`vp9_scale`, scaled convolve). Then inverse transform + add.
5. **Intra recon (wavefront over 64x64 superblocks, k = r + c):** intra blocks need only pixels above / left
   (libvpx `have_right = (aoff + txw) < bw`: above-right stays inside the block). Prediction + inverse transform
   (DCT / ADST 4..32, WHT for lossless) per transform block, in libvpx order inside the superblock.
6. **Loop filter (wavefront k = 2r + c, as libvpx's `sync_read`):** masks from mode info (`vp9_build_mask` /
   `vp9_setup_mask`), filter levels with ref / mode deltas and segment features, 4 / 8 / 16-wide filters, vertical
   then horizontal per superblock and plane.
7. **Backward adaptation (when refresh_frame_context and not error_resilient / frame_parallel):** coef, mode, tx,
   skip, mv probs from the counts (`vp9_adapt_*`), stored to the frame context slot.
8. **References:** frame buffers + MV buffers + seg maps pooled on the device; refresh slots; show_existing_frame;
   intra-only frames (profile-0 refs); frames are output cropped through the three-overload boundary (ArrayView1D /
   TypedArray / managed), never through .NET pixel loops.

Correct first: steps 3..7 start as straightforward kernels with libvpx's control flow; the parallel structure above is
already in the first version, finer parallelism (per transform block, per edge) is the performance pass after every
vector passes.

## Steps
- [x] S0 fixtures + harness: vectors into the repo (profile-0 core set embedded; full set as a Conformance category),
      per-frame MD5 test, WebM / IVF / superframe demux into frame packets.
- [x] S1 uncompressed header (all fields, every vector parses) + compressed header kernel + frame contexts.
- [ ] S2 parse kernel: keyframes / intra-only (vectors 00, 02, 03 first frames).
- [ ] S3 intra recon + inverse transforms + loop filter: keyframes MD5-exact (00 quantizer set frame 1, 02 / 03 sizes).
- [ ] S4 inter parse (refs, modes, MV prediction + decode, interp filters, compound, sub-8x8) + inter recon + adaptation:
      all frames of 00 / 02 / 03 / 06 / 07.
- [ ] S5 tiles (08), segmentation (15, deltaq), lf deltas, skip (19), superframes (20), show-existing (17), intra-only (16).
- [ ] S6 scaled references: resize vectors (05, 14, 18, 21), SVC (22).
- [ ] S7 every backend (CPU, Velocity, CUDA, OpenCL, WebGPU, Wasm; WebGL where its limits allow), ComfyUI files, BBB.
- [ ] S8 performance pass (frames / s vs libvpx single-thread and FFmpeg, logged in _TestStats).
- [ ] S9 public API + README + CHANGELOG; replace `Vp9KeyframeDecoderGpu`'s internals (it rejects sizes that are not
      multiples of 64) with this decoder.
- [ ] Then profiles 1..3 (4:4:4, 10 / 12-bit), and the encoder plan (inter frames, crf).
