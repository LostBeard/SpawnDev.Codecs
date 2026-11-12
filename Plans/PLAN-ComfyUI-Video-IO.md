# PLAN: ComfyUI video save / load parity (for SpawnGen)

Requested by Lore (SpawnGen) 2026-11-11; order approved by TJ ("Looks fine to me"). ComfyUI is GPL-3: black box only
(run its nodes, compare outputs; never read its source). FFmpeg is LGPL: never read or port its source; running the
ffmpeg / ffprobe tools as a black box is fine.

## What already exists (1.0.2-local.13)
- WebM muxer `WebMWriter` (VP8 / VP9 / AV1 + Opus / Vorbis), clean-room, layout as ffmpeg `-fflags +bitexact`.
- Keyframe encoders: VP8, VP9, AV1 (base q index, no CRF, no inter frames).
- Animated lossy WebP (`WebPLossyEncoderGpu.EncodeAnimationAsync`, libwebp WebPAnimEncoder port) and lossless.
- Full VP9 decoder `Vp9DecoderGpu` (inter frames, alpha, bit-exact with libvpx); `DecodeFileAsync` returns YUV planes.
- Opus decoder (byte-identical with libopus 1.5.2), Vorbis decoder.

## Out of scope
- H.264 (patent-encumbered; SpawnDev.MultiMedia's platform encoders). SpawnGen documents AV1 / VP9 as the substitute.

## Work, in order
1. [ ] **WebM load = ComfyUI LoadVideo + GetVideoComponents.** One call: file bytes -> RGB frames (N x H x W x 3),
   fps, optional alpha, audio (planar float, sample rate).
   - [x] Matroska reader: CodecPrivate, DefaultDuration, CodecDelay, SeekPreRoll, PixelWidth / Height, AlphaMode,
     SamplingFrequency, Channels, BitDepth on `MatroskaTrack`; DiscardPadding on `MatroskaFrame`.
   - [x] Black-box references from ComfyUI (Lore): IMAGE tensors, fps, AUDIO for the Vp9ComfyUI test files + one
     file with Opus audio (`D:\users\tj\Projects\_DevComms\codecs-load-refs`).
   - [x] The conversion rule, derived as a black box: `Docs/comfyui-loadvideo-yuv-to-rgb.md` (16-bit RGB, bilinear
     chroma, BT.601 limited; ComfyUI == `ffmpeg -sws_flags bilinear -pix_fmt gbrp16le` on all 108 frames).
   - [x] `Yuv420ToRgb16Gpu`: 108/108 ComfyUI frames and the odd-size oracle vectors bit-exact on CUDA, Velocity, CPU
     accelerator, WebGPU. A/B int32-split vs 64-bit (TJ: measure, don't assume): int32 kept (CUDA 1.9x, Velocity/CPU
     2-4%, WebGPU equal).
   - [x] fps rule (DefaultDuration -> smallest-denominator rational, matches ffprobe except 59.94 by 5e-6) and Opus
     trimming (pre-skip + DiscardPadding: ComfyUI's sample count). `WebMVideoLoaderGpu`.
   - [ ] Add Yuv420ToRgb16Gpu / WebMVideoLoaderGpu to the lazy-load guard; WebGL lane run (odd-size test).
   - [ ] AV1 inter-frame decode, so ComfyUI's own SaveVideo (AV1 + Opus) files load.
     Measured 2026-11-11: ComfyUI's AUDIO is bit-identical to ffmpeg's default (native) Opus decoder with pre-skip and
     end trimming (comfyui_webm_with_opus: 96000 samples x 2). Our OpusDecoderGpu is libopus-exact; libopus vs
     ComfyUI: max |diff| 1.54e-5 (about -96 dB), 185 of 192000 samples bit-equal. Bit-exact parity there would mean
     matching ffmpeg's own float Opus decoder (black box only); parked, reported to TJ.
   - [ ] Tests on every backend against the references.
2. [x] **MP4 (ISO BMFF) muxer** for AV1 (+ Opus), from the public specs (ISO/IEC 14496-12, AV1-ISOBMFF binding,
   Opus-in-ISOBMFF) and black-box ffmpeg `-c copy` output. `Mp4Writer`: byte-identical with ffmpeg on 10 fixtures
   (2026-11-12). Not covered yet: a track starting after 0 (empty edit: throws), colr, > 4 GiB (co64 / wide mdat are
   written but untested against ffmpeg), constant sample sizes.
   - Target layout (ffmpeg 8.1 `-c copy -fflags +bitexact`, AV1 + Opus, 2026-11-12): ftyp 32, free 8, mdat, moov
     (mvhd; video trak: tkhd, edts/elst, mdia/mdhd/hdlr/minf (vmhd, dinf/dref, stbl: stsd/av01 (av1C, colr, pasp,
     btrt), stts, stss, stsc, stsz, stco); audio trak: smhd, stsd/Opus (dOps, btrt), stts, stsc, stsz, stco, sgpd +
     sbgp (Opus roll recovery); udta/meta). Fixtures will be byte-identical targets like the WebMMux ones.
3. [ ] **VP9 inter frames in the encoder** (file size for 121-frame Wan clips).
