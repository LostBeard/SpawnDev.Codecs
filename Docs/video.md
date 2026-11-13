# Video codecs

Namespaces `SpawnDev.Codecs.Video.Vp8`, `.Vp9`, `.Av1`. Patent-clean through the AOMedia patent pledge. Inter frames
exist in two places only: the VP9 encoder (`Vp9EncoderGpu`) and the VP9 decoder (`Vp9DecoderGpu`). The VP8 and AV1
classes handle key frames (intra frames) only. Planes are 8-bit YUV 4:2:0, tightly packed unless a stride argument says
otherwise.

## VP8

| Class | What it does |
|-------|--------------|
| `Vp8KeyframeEncoderGpu` | Keyframes: prediction, forward DCT / WHT, quantization, reconstruction, bool-coded tokens and the frame header, all as kernels. One token partition. Any width and height (odd sizes pad on the device). |
| `Vp8FullKeyframeDecoderGpu` | Decodes the whole keyframe syntax libvpx / libwebp write: segmentation, per-segment quantizers and filter levels, all intra modes including B_PRED, multiple token partitions, the normal and simple loop filters. Identical planes to ffmpeg on 182 real files. Hostile input (truncated or corrupt partitions) throws `Vp8DecodeException` with the reason. |
| `Vp8KeyframeDecoderGpu` | The decoder matched to this library's own encoder (needs `baseQIndex`); use `Vp8FullKeyframeDecoderGpu` for files from elsewhere. |

```csharp
using var enc = new Vp8KeyframeEncoderGpu(accelerator);
byte[] frame = await enc.EncodeKeyFrameAsync(y, ySrcStride: width, u, uvSrcStride: width / 2, v, width, height, baseQIndex: 30);

using var dec = new Vp8FullKeyframeDecoderGpu(accelerator);
var picture = await dec.DecodeKeyFrameAsync(frame);   // picture.YPlane, UPlane, VPlane, Width, Height
```

On the device: `EncodeKeyFrameAsync(ArrayView1D y, ..., ArrayView1D output)` returns the length;
`DecodeKeyFrameAsync(ArrayView1D encoded, int length, ArrayView1D yOut, uOut, vOut)` writes caller-owned planes
(`GetLumaPlaneSize` / `GetChromaPlaneSize` size them). `ReadFrameTagAsync` reads the 10-byte header.

## VP9

| Class | What it does |
|-------|--------------|
| `Vp9EncoderGpu` | `new Vp9EncoderGpu(Accelerator accelerator)`; `EncodeFrameAsync(yPlane, uPlane, vPlane, width, height, output, baseQIndex = 30, forceKeyFrame = false)` (planes on the accelerator, returns `Vp9EncodedFrame(Length, IsKeyFrame)`) or the same with `ReadOnlyMemory<byte>` planes (returns the bytes and `IsKeyFrame`). A key frame, then frames predicted from the frame before, with a key frame every `KeyFrameInterval` frames (default 128). Motion search (`MotionSearch`, default on): three-level pyramid (+-38 pixels), half and quarter-pel refinement, merge with the neighbours' MVs. Inter frames quantize with a dead zone. Profile 0, 4:2:0, 8 bits, fixed quantizer. |
| `Vp9KeyframeEncoderGpu` | Keyframes: uncompressed and compressed headers, partition tree, 16x16 leaves with DC prediction, DCT, quantization, coefficient tokens, single tile. Any size. |
| `Vp9DecoderGpu` | `new Vp9DecoderGpu(Accelerator accelerator)`; `DecodePacketAsync(packet)` and `DecodeFileAsync(byte[] file)` (WebM / Matroska first V_VP9 track, or IVF). Profile 0 (8-bit 4:2:0), every libvpx profile-0 feature: inter frames, compound and scaled references, tiles, segmentation, lossless, hidden / show-existing frames, superframes, WebM alpha. Every frame of libvpx's 303 profile-0 conformance vectors decodes to libvpx v1.17.0's output bit for bit. |
| `Vp9KeyframeDecoderGpu` | Obsolete (marked `[Obsolete]`; use `Vp9DecoderGpu`, which also decodes `Vp9KeyframeEncoderGpu` output). Keyframes only; width and height must be multiples of 64. |

## AV1

| Class | What it does |
|-------|--------------|
| `Av1KeyframeEncoderGpu` | Keyframes only: temporal delimiter, sequence header and frame OBU (OBU framing and leb128 sizes written on the device), DC prediction, 16x16 blocks, forward transforms, quantization, the Daala range coder. Any picture size (edge superblocks coded with 8x8 leaves). 8-bit 4:2:0; no loop filter, CDEF or restoration. libdav1d decodes the output. |
| `Av1KeyframeDecoderGpu` | `DecodeSingleTileAsync(byte[] tileBytes, int width, int height, int baseQIndex = 32)` for single-tile key frames (raw tile bytes, not an OBU stream). Width and height must be multiples of 64. Managed overload only so far. No inter-frame decode. |

## Files: encode, load, mux

| Class | What it does |
|-------|--------------|
| `VideoFileEncoderGpu` | `new VideoFileEncoderGpu(Accelerator accelerator)`; `EncodeAsync(frames, width, height, fpsNum, fpsDen, audio = null, options = null)` returns the file bytes. RGB frames (ComfyUI IMAGE layout, float32 H x W x 3) to YUV 4:2:0 to VP9 or AV1 on the accelerator; optional audio (interleaved float) to Opus; muxed to MP4, WebM or MKV. Overloads: `IReadOnlyList<float[]>`, frames already on the accelerator (`ArrayView1D<float, Stride1D.Dense>` holding N x H x W x 3, plus `frameCount`), and in the browser `IReadOnlyList<Float32Array>` (returns a `Uint8Array`). VP8 is not offered. |
| `VideoFileEncodeOptions` | `Container` (`Mp4` default, `WebM`, `Mkv`), `Codec` (`Vp9` default, `Av1`), `QIndex` (1..255; null = 30 for VP9, 32 for AV1), `AudioBitrate`, `KeyFrameInterval` (VP9: null = 128, 1 = key frames only; AV1 writes key frames only), and `QIndexFromCrf(crf)` (libvpx / libaom quantizer_to_qindex). One fixed quantizer for every frame, no rate control: the same crf does not give the file size libvpx / libaom give. |
| `WebMVideoLoaderGpu` | `new WebMVideoLoaderGpu(Accelerator accelerator)`; `LoadAsync(byte[] file)` returns `LoadedVideo(Width, Height, FrameRate, Frames, Audio)`: a VP9 WebM to RGB frames bit-exact with ComfyUI's LoadVideo (all 108 frames of the three ComfyUI-written VP9 test files), the frame rate and the Opus audio (pre-skip and DiscardPadding trimmed). Only VP9 4:2:0 video loads; AV1 and VP8 video do not load yet (no inter-frame decoder for them), so ComfyUI's own SaveVideo (AV1) files do not load. |
| `Yuv420ToRgb16Gpu` | `ConvertToFloat(ArrayView<byte> planes, int width, int height, ArrayView<float> rgb)`: YUV 4:2:0 to RGB as ComfyUI's LoadVideo does (16-bit RGB, bilinear chroma, BT.601 limited). |
| `Mp4Writer` | `new Mp4Writer(Stream output, bool leaveOpen = false)`; `AddVideoTrack` (`Mp4Av1Track` or `Mp4Vp9Track`), `AddAudioTrack(Mp4OpusTrack)`, `WriteFrame(track, data, dts, duration, key)`, `Finish()`. ISO BMFF muxer for AV1 or VP9 video and Opus audio; byte-identical with `ffmpeg -c copy -fflags +bitexact` on 10 AV1 / Opus fixtures. Not covered: a track starting after 0 (throws), colr, files over 4 GiB (written but untested against ffmpeg). |
| `WebMWriter` | `new WebMWriter(Stream stream, WebMWriterOptions? options = null, bool leaveOpen = false)`; `AddVideoTrack`, `AddAudioTrack`, `WriteFrame(trackNumber, data, timestampMs, keyFrame, ...)`, `Finish()`. `WebMWriterOptions.DocType` is "webm" (default) or "matroska" (a valid .mkv with the same layout; not byte-identical with ffmpeg's matroska muxer). |

## Batches of frames

The encoders take a batch of frames in one call (`EncodeKeyFramesBatchAsync`): the frames sit in consecutive slots of
one buffer and every kernel runs once per step with one thread (or one thread group) per frame. One upload and one
readback per batch instead of one per frame. The output of a batch is identical to encoding the frames one by one.

## Limits and open work

- No rate control: every encoder takes one fixed quantizer (`baseQIndex` / `QIndex`).
- Encoders: VP8 and AV1 write key frames only; only VP9 (`Vp9EncoderGpu`) writes inter frames. VP9 key frames are DC
  prediction with 16x16 leaves only. VP8 encode is one token partition.
- VP9 encoder, not done yet (Plans/PLAN-VP9-Encoder-Inter-Frames.md): large skipped partitions (32x32 / 64x64),
  backward adaptation and previous-frame MV candidates, a loop filter on the reconstruction, intra blocks in inter frames,
  scene-cut key frames, high-precision MVs, golden / altref, tuning of the motion search's cost settings, and the
  browser's own VP9 decoder in the browser lanes. The Wasm and OpenCL lanes were not run for the inter-frame encoder.
  Lossless is not implemented. Pictures wider than `Vp9InterEntropy.MaxMiCols * 8` pixels throw `NotSupportedException`.
- Decoders: `Av1KeyframeDecoderGpu` decodes multiples of 64 pixels, key frames only; VP9 decode is profile 0 only
  (profiles 1..3, 4:4:4 and 10 / 12-bit, are not done); `Vp9DecoderGpu` is not yet checked on every backend, and its
  performance pass is open (Plans/PLAN-VP9-Decoder-Full.md, S7 and S8).
- Loading: `WebMVideoLoaderGpu` loads VP9 WebM only (AV1 inter-frame decode is open). Its audio differs from ComfyUI's
  by max |diff| 3.3e-4 (about -70 dB) in the CHANGELOG measurement.
- In the browser (WebGPU), the full WebGPU test lane with the VP9 decoder tests passes (1188 passed, 0 failed, 50 min;
  Chrome's GPU process peaked at 2.27 GB); WebGL still leaves the VP9 decoder tests out. On WebGPU the AV1 batch encode
  also differs from the single-frame encode; see [status.md](status.md).
- The GPU paths are checked bit for bit against CPU references: `Vp8KeyframeEncoder`, `Vp9KeyframeEncoder`, the
  walkers and `Av1Decoder` in `SpawnDev.Codecs.References`, and `Av1KeyframeEncoder` (still in the main library; moving
  it to References is open work).
