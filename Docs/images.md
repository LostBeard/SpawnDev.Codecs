# Image codecs

Namespace `SpawnDev.Codecs.Images` (and `.Png`, `.Jpeg`, `.WebP`, `.WebP.Lossy`, `.Gif`, `.Bmp`). Every class takes the
`Accelerator` in its constructor, compiles its kernels once, and is reused across calls (one call at a time per
instance). Every entry point has the three overloads described in [api.md](api.md); the examples below use the managed
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
estimated size, LZ77 with a colour cache, meta prefix codes for images whose tiles differ. `WebPEncodeOptions`:
`Quality` (effort, 0 to 100, default 75), `Exact` (default true: keep the RGB under transparent pixels), `IccProfile`,
`Exif`, `Xmp`. Also `EncodeAnimationAsync` (one RGBA view and one `ImageFrameInfo` per frame: rectangle, duration,
disposal, blending; `WebPAnimationEncodeOptions` adds the loop count and background colour) and
`EncodeAlphaAsync`, which writes the ALPH chunk payload of a lossy file (`WebPAlphaEncodeOptions`: filter none /
horizontal / vertical / gradient or automatic, compression, effort). The output decodes to exactly the input pixels in
libwebp and in our decoder.

`WebPLossyEncoderGpu` is a port of libwebp 1.6.0's lossy encoder (methods 3 and 4). For the same pixels and settings
its files are **byte-identical** with libwebp's:

| `WebPLossyEncodeOptions` | Default | |
|---|---|---|
| `Quality` | 75 | 0 to 100 (Pillow passes 80) |
| `Method` | 4 | 3 or 4; other methods throw `NotSupportedException` (never a different file) |
| `Exact` | false | false runs libwebp's transparent-area cleanup |
| `Alpha` | null | how the ALPH chunk is coded (`WebPAlphaEncodeOptions`) |

A picture with transparency is written as RIFF + VP8X + ALPH + VP8: the VP8 chunk is byte-identical with libwebp's,
and the alpha is lossless (libwebp's `alpha_quality` 100). The ALPH bytes themselves differ from libwebp's (a different
lossless coder) and decode to the same plane. Not yet: methods 0 to 2 and 5 / 6 (trellis), more than one token
partition.

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

## Limits and open work

- Browser lanes: see [status.md](status.md). Lossless WebP encode and GIF are correct on WebGPU but slow there (many
  short launches with host waits); that is being reduced.
- Not implemented: AVIF (planned on top of the AV1 encoder), APNG frame encoding, JPEG arithmetic coding and lossless
  JPEG, HEIC (patent-encumbered, out of scope).
