# Video codecs

Namespaces `SpawnDev.Codecs.Video.Vp8`, `.Vp9`, `.Av1`. Patent-clean through the AOMedia patent pledge. Every codec here
is **keyframe-only** today (intra frames); inter prediction is not implemented. Planes are 8-bit YUV 4:2:0, tightly
packed unless a stride argument says otherwise.

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
| `Vp9KeyframeEncoderGpu` | Keyframes: uncompressed and compressed headers, partition tree, 16x16 leaves with DC prediction, DCT, quantization, coefficient tokens, single tile. Any size. |
| `Vp9KeyframeDecoderGpu` | Keyframes from this encoder and similar streams. Width and height must be multiples of 64 (open: boundary superblocks). |

## AV1

| Class | What it does |
|-------|--------------|
| `Av1KeyframeEncoderGpu` | Keyframes: temporal delimiter, sequence header and frame OBU (OBU framing and leb128 sizes written on the device), DC prediction, 16x16 blocks, forward transforms, quantization, the Daala range coder. Width and height must be multiples of 64. libdav1d decodes the output. |
| `Av1KeyframeDecoderGpu` | `DecodeSingleTileAsync` for single-tile frames, multiples of 64. Managed overload only so far. |

## Batches of frames

The encoders take a batch of frames in one call (`EncodeKeyFramesBatchAsync`): the frames sit in consecutive slots of
one buffer and every kernel runs once per step with one thread (or one thread group) per frame. One upload and one
readback per batch instead of one per frame. The output of a batch is identical to encoding the frames one by one.

## Limits and open work

- Keyframes only; no inter frames, no rate control beyond the fixed `baseQIndex`.
- VP9 decode and AV1 encode / decode at multiples of 64 pixels; VP8 encode at one token partition.
- In the browser, the VP8 / VP9 / AV1 frame kernels are currently too large for Chrome's shader compiler (the GPU process
  is lost, or a compile takes minutes); they are being split into smaller stage kernels
  ([Plans/PLAN-Oversized-Kernels.md](../Plans/PLAN-Oversized-Kernels.md)). On WebGPU the AV1 batch encode also differs from
  the single-frame encode; see [status.md](status.md).
- The GPU paths are checked bit for bit against CPU references: `Vp8KeyframeEncoder`, `Vp9KeyframeEncoder`, the
  walkers and `Av1Decoder` in `SpawnDev.Codecs.References`, and `Av1KeyframeEncoder` (still in the main library; moving
  it to References is open work).
