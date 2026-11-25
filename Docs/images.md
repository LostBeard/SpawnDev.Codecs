# Image codecs

Namespace `SpawnDev.Codecs.Images` (and `.Png`, `.Jpeg`, `.WebP`, `.WebP.Lossy`, `.Gif`, `.Bmp`). Every class takes the
`Accelerator` in its constructor, compiles nothing there, loads its kernels once on its first async call (or in
`WarmUpAsync`, which every image codec has) and is reused across calls (one call at a time per instance). Every entry point has the three overloads described in [api.md](api.md); the examples below use the managed
one for brevity.

Pixels are tightly packed **RGBA8** (straight alpha, rows top to bottom) unless `PixelFormat.Rgba16` is requested
(16-bit samples, for example from a 16-bit PNG).

## Decode any format: `ImageDecoderGpu`

```csharp
using var images = new ImageDecoderGpu(accelerator);
ImageInfo info = await images.ReadInfoAsync(fileBytes);         // format, size, alpha, depth, frames, orientation
DecodedImage picture = await images.DecodeAsync(fileBytes);      // picture.Pixels = RGBA8
```

The format is recognized on the device from the file's signature (PNG, JPEG, WebP, GIF, BMP) and the call goes to the
matching decoder (`images.Png`, `images.Jpeg`, `images.WebP`, `images.Gif`, `images.Bmp` expose them for
format-specific calls). Unrecognized bytes throw `ImageDecodeException`.

## PNG

`PngDecoderGpu` decodes every colour type and bit depth (1 to 16 bits, palette, grey, grey + alpha, RGB, RGBA), tRNS,
Adam7 interlacing, and checks critical-chunk CRCs and the zlib Adler-32. Inflate runs on the accelerator
(`InflateGpu`). APNG files decode their default image; `ReadTextAsync` returns the tEXt / iTXt (and zTXt) chunks.

`PngEncoderGpu` compresses with `ZlibCompressorGpu`: the image is cut into independent deflate blocks that compress in
parallel, each block ends on a byte boundary, and every block gets its own IDAT chunk (so a decoder can start before the
end). Options:

| `PngEncodeOptions` | Default | |
|---|---|---|
| `CompressionLevel` | 2 | effort of the match finder |
| `ColorMode` | `Auto` | `Rgba`, `Rgb`, `Gray`, `GrayAlpha` (`Auto` picks the smallest exact one) |
| `Filter` | `Adaptive` | per-row filter choice, or a fixed filter |
| `InputFormat` | `Rgba8` | `Rgba16` writes 16-bit samples |
| `TextChunks` | null | key / value pairs written as tEXt (Latin-1) or iTXt (UTF-8) |
| `BlockSize` | `ZlibCompressorGpu.DefaultBlockSize` | bytes per independent deflate block |

Checked by an independent host PNG reader and by Pillow decoding our files to the source pixels.

## JPEG

`JpegDecoderGpu` decodes baseline, extended sequential and progressive (spectral selection and successive
approximation) JPEGs: grey, YCbCr in every common subsampling (4:4:4, 4:2:2, 4:2:0, 4:4:0, 4:1:1), Adobe CMYK / YCCK /
RGB, restart intervals, and reports the EXIF orientation (`JpegDecodeOptions.ApplyOrientation` rotates the output). It
follows libjpeg-turbo's arithmetic (ISLOW IDCT, fancy upsampling, fixed-point colour conversion), so the pixels are
**bit-exact** with libjpeg-turbo. Corrupt or truncated files throw `ImageDecodeException` with the reason.

`JpegEncoderGpu` follows libjpeg-turbo too (colour conversion, downsampling, ISLOW forward DCT, quantization, the
Annex K Huffman tables), so its files are **byte-identical** with libjpeg-turbo for the same pixels and settings:

| `JpegEncodeOptions` | Default | |
|---|---|---|
| `Quality` | 90 | 1 to 100 (libjpeg's quality scaling) |
| `Subsampling` | `Yuv420` | `Yuv444`, `Yuv422`, `Yuv420` |
| `Progressive` | false | progressive scan script |
| `OptimizeHuffman` | false | per-image Huffman tables |
| `RestartInterval` | 0 | MCUs between restart markers |
| `ExifOrientation` | 1 | written as an EXIF APP1 segment when not 1 |

## WebP

`WebPDecoderGpu` decodes lossless (VP8L) and lossy (VP8, with ALPH alpha) stills and animations, including animations
that mix lossy and lossless frames. Beyond `DecodeAsync`: `ReadFileInfoAsync`, `ReadMetadataAsync` (ICC, EXIF, XMP),
`ReadFramesAsync` (rectangles, durations, disposal, blending), `DecodeFrameAsync(k)` (the canvas while frame k is
shown) and `DecodeAlphaAsync`. Lossy decode is bit-exact with libwebp on 168 fixtures (the VP8 data goes through
`Vp8FullKeyframeDecoderGpu`, then libwebp's fancy upsampling and colour matrix in `WebPYuvToRgbaGpu`).

`WebPLosslessEncoderGpu` writes VP8L: palette, subtract-green, predictor and cross-colour transforms chosen by
estimated size, LZ77 with a colour cache, meta prefix codes for images whose tiles differ. The tokeniser
(`Vp8lTokenizeParallel`: every position's hash-chain candidates, best copy and colour-cache hit in parallel), the token
writer and the per-tile histograms run in parallel with the same bytes as the serial versions; on CUDA (RTX 4070) the
frames of a 512x352 q80 animation went from 4.26 s to 0.24 s each. `WebPEncodeOptions`:
`Quality` (effort, 0 to 100, default 75), `Exact` (default true: keep the RGB under transparent pixels), `IccProfile`,
`Exif`, `Xmp`. Also `EncodeAnimationAsync` (one RGBA view and one `ImageFrameInfo` per frame: rectangle, duration,
disposal, blending; `WebPAnimationEncodeOptions` adds the loop count and background colour) and
`EncodeAlphaAsync`, which writes the ALPH chunk payload of a lossy file (`WebPAlphaEncodeOptions`: filter none /
horizontal / vertical / gradient or automatic, compression, effort, `ExactFilterChoice`: with the automatic filter at
effort 50 and above the plane is coded with each of the four filters and the smallest kept (the default); false picks the
filter from a fast estimate and codes once, as libwebp's default does). The output decodes to exactly the input pixels in
libwebp and in our decoder.

A second `EncodeAnimationAsync(frames, width, height, durationsMs, options, animation)` takes full-canvas frames and makes
libwebp's lossless WebPAnimEncoder decisions: exact change rectangles, disposal by the smaller rectangle, blending when
it can reproduce the frame, dropped repeats, key frames between `KeyframeMin` and `KeyframeMax` (`WebPAnimEncoderOptions`;
null = Pillow's lossless defaults 9 / 17) and a lone frame written as a still. With `Exact = false` (libwebp's default) a
transparent pixel takes its predicted colour, so the RGB under alpha 0 is not preserved, and each tile's predictor mode
is chosen the way libwebp 1.6.0 does at method 4 (all 14 modes scored on the tile's residual histograms against the
histograms of the tiles before it); on 25 frames of a 512x352 clip that animation went from 3,772,426 to 3,415,440 bytes
(libwebp: 3,240,800). The VP8L bytes are ours, so these files are not byte-identical with libwebp's.

`WebPLossyEncoderGpu` is a port of libwebp 1.6.0's lossy encoder (methods 3 and 4). For the same pixels and settings
its files are **byte-identical** with libwebp's:

| `WebPLossyEncodeOptions` | Default | |
|---|---|---|
| `Quality` | 75 | 0 to 100 (Pillow passes 80) |
| `Method` | 4 | 3 or 4; other methods throw `NotSupportedException` (never a different file) |
| `Exact` | false | false runs libwebp's transparent-area cleanup |
| `Alpha` | null | how the ALPH chunk is coded (`WebPAlphaEncodeOptions`) |
| `FilterStrength` | 60 | libwebp's `filter_strength`; 0 writes VP8 profile 2 (byte-identical with libwebp at 0, 25, 60 and 100) |
| `IccProfile`, `Exif`, `Xmp` | null | written verbatim in libwebp's still layout (RIFF, VP8X, ICCP, ALPH, VP8, EXIF, XMP); pass EXIF without the `Exif\0\0` prefix, as Pillow's still save does |

A picture with transparency is written as RIFF + VP8X + ALPH + VP8: the VP8 chunk is byte-identical with libwebp's,
and the alpha is lossless (libwebp's `alpha_quality` 100). The ALPH bytes themselves differ from libwebp's (a different
lossless coder) and decode to the same plane. Not yet: methods 0 to 2 and 5 / 6 (trellis), more than one token
partition.

`WebPLossyEncoderGpu.EncodeAnimationAsync` (managed, `TypedArray` and `ArrayView1D` overloads) takes full-canvas RGBA8
frames and per-frame durations and makes libwebp 1.6.0's WebPAnimEncoder decisions frame for frame: change rectangles,
disposal, blending (similar 8x8 blocks made transparent), dropped frames whose durations merge into the previous one, key
frames between `KeyframeMin` and `KeyframeMax` (null = Pillow's lossy defaults 3 / 5) and a lone frame written as a still.
`WebPAnimEncoderOptions` also carries the loop count, the background colour and the ICCP / EXIF / XMP chunks (EXIF is
written exactly as given). Against Pillow 13 / libwebp 1.6.0 on 26 fixtures, 19 files are byte-identical and the 7 whose
frames carry transparent pixels are identical except for their ALPH chunks.

The mode decisions of the macroblock walk run in parallel: `PickBestIntra4`, `PickBestIntra16` and `PickBestUV` each use a
thread group per macroblock with a lane per mode, and one lane keeps libwebp's choice, so the files do not change. On
512x352 q80 frames (CUDA, RTX 4070) the Intra4 stage went from 334 to 87 ms per frame, Intra16 from 238 to 89 ms and UV
from 103 to 41 ms.

## GIF

`GifDecoderGpu` decodes 87a / 89a files: LZW on the accelerator (frames in parallel, in memory-bounded batches),
interlaced frames, transparency, disposal methods, and the NETSCAPE loop count. `DecodeFrameAsync(k)` returns the full
canvas while frame k is displayed; `ReadFramesAsync` lists the frames.

`GifEncoderGpu` quantizes (median-cut style palette, optional dithering, alpha threshold) and LZW-encodes stills and
animations. `GifEncodeOptions`: `MaxColors` (256), `Dither`, `AlphaThreshold` (128), `LoopCount` (0 = forever),
`FrameDelayMs` / `FrameDelaysMs`, `FrameDisposals`. Pillow decodes our files to the same pixels; the quantizer's PSNR is
compared with Pillow's median cut.

## BMP

`BmpDecoderGpu` and `BmpEncoderGpu` (`BmpEncodeOptions.Format`, default `Bgra32`). Checked against Pillow.

## PSD (layered, write)

`PsdEncoderGpu` writes a layered Photoshop file (version 1, 8 bits per channel, RGB, RLE): `PsdDocument` is the canvas size, the layers
(bottom first) and the merged image (`PsdComposite`, required). A `PsdLayer` has its name (kept whole: Unicode in a `luni` block), its
rectangle on the canvas (`Left` / `Top` may be negative and the layer may run past the canvas), `Width` x `Height`, RGBA8 pixels either as managed
bytes (`Rgba`) or as an accelerator view (`RgbaView`), a `PsdBlendMode`, an opacity and a visible flag. `PsdComposite.HasAlpha` writes the merged
image with an alpha channel (colours matted over white, the way Photoshop stores merged images, so a reader un-mattes them). Overloads:
`EncodeAsync(doc, outputView)` (length returned, output of at least `GetMaxEncodedSize(doc)` bytes), `EncodeAsync(doc)` to a managed array, and in a browser
`EncodeToUint8ArrayAsync(doc)`.

The 26 blend modes are Adobe's keys as ComfyUI writes them (measured); grain extract and grain merge, which the format has no key for, are written as
normal. Layers may be empty (a flat PSD), up to 8192 of them; canvas and layer sides up to 30,000. Read back pixel by pixel by psd-tools (layers, names,
rectangles off the canvas, opacity, visibility, blend keys, merged image); Pillow's PSD plugin is not the oracle because it reads alpha-first layers
in the wrong channel order. Not on WebGL (scatter stores).

## BCn textures

`BcnDecoderGpu` decodes BC1, BC3, BC4, BC5 and BC7 blocks to RGBA8 and `BcnEncoderGpu` encodes RGBA8 to BC1, BC3, BC4 and
BC5 (`BcnFormat`); `Bc7EncoderGpu` encodes BC7. `EncodeAsync(rgba, width, height, format)` and
`DecodeAsync(blocks, format, width, height)` have managed and `TypedArray` overloads; `Encode` / `Decode` take
`ArrayView1D` buffers and are synchronous launchers. Sizes that are not a multiple of 4 repeat the edge pixels.
The decoder is bit-exact with texture2ddecoder (29 fixtures). The BC1, BC3, BC4 and BC5 encoder is byte-identical with
etcpak 0.9.15 (19 fixtures); the BC7 encoder is a port of bc7enc at its default settings (CPU, Velocity and Wasm are
byte-identical with bc7enc on every block; CUDA, OpenCL and WebGPU are 94.5 to 99 % identical block for block, with the
PSNR never more than 0.01 dB lower).

## Byte compression

Namespace `SpawnDev.Codecs.Compression`. Each class has the managed, `TypedArray` and `ArrayView1D` overloads of
[api.md](api.md) and a `WarmUpAsync`. On WebGL only the shuffle filters run (the others need scatter stores).

| Class | What it does |
|-------|--------------|
| `ZlibCompressorGpu`, `DeflateGpu`, `InflateGpu` | zlib and raw DEFLATE (levels 1 to 4, and a Huffman-only level) in parallel blocks; .NET's `ZLibStream` inflates the output. PNG uses them |
| `Lz4FrameCompressorGpu`, `Lz4FrameDecompressorGpu` | The LZ4 frame format. The compressor is byte-identical with liblz4 1.9.4; the decoder reads every liblz4 configuration and throws `Lz4FormatException` with the reason on damaged frames |
| `SnappyCompressorGpu`, `SnappyDecompressorGpu` | Snappy, raw and framed (`SnappyFormat`). The compressor is byte-identical with the Rust snap crate; damaged streams throw `SnappyFormatException` |
| `GDeflateCompressorGpu`, `GDeflateDecompressorGpu` | GDeflate (DirectStorage tile streams); Microsoft's reference decoder decodes our streams and ours decodes the reference's. Damaged streams throw `GDeflateFormatException` |
| `ShuffleFilterGpu` | Byte shuffle and bitshuffle, both directions (`ShuffleAsync`, `UnshuffleAsync`, `BitShuffleAsync`, `BitUnshuffleAsync`); identical with c-blosc / numcodecs and the bitshuffle library |
| `WeightPackGpu` | Lossless packing of model weight tensors (Huffman-coded chunks that .NET's `DeflateStream` inflates) |
| `Crc32Gpu`, `Xxh32Gpu` | CRC-32 (PNG) and xxHash32 (LZ4 frames) helpers |

## Limits and open work

- Browser lanes: see [status.md](status.md). Lossless WebP encode and GIF are correct on WebGPU but slow there (many
  short launches with host waits); that is being reduced.
- Not implemented: AVIF (planned on top of the AV1 encoder), APNG frame encoding, JPEG arithmetic coding and lossless
  JPEG, HEIC (patent-encumbered, out of scope).
