# SpawnDev.Codecs documentation

| Page | What it covers |
|------|----------------|
| [Status](status.md) | Every codec by backend: verified, partial or open, with the evidence. Start here before relying on a codec. |
| [Images](images.md) | PNG, JPEG, WebP, GIF, BMP, layered PSD (write) and `ImageDecoderGpu`: API, options, limits, how each is checked; BCn textures and the byte compressors (zlib, LZ4, Snappy, GDeflate, shuffle filters) |
| [Video](video.md) | VP8, VP9 and AV1 encoders and decoders (VP9 with inter frames), file encode / load, MP4 and WebM muxers: API, frame batches, limits |
| [Audio](audio.md) | Vorbis, FLAC, MP3, Opus (including Ogg Opus files) and the audio containers |
| [API shape](api.md) | The three overloads every entry point has, buffer ownership, zero copy in the browser |
| [Architecture](architecture.md) | Host as coordinator, bounded launches, the browser shader-compile budget |
| [Testing](testing.md) | PlaywrightMultiTest, fixtures and their oracles, environment switches, red-checks, the DemoConsole benches |
| [ComfyUI LoadVideo conversion](comfyui-loadvideo-yuv-to-rgb.md) | The YUV 4:2:0 to RGB rule `Yuv420ToRgb16Gpu` implements, derived as a black box |

Design documents and open work live in [`Plans/`](../Plans/); every change is in the [CHANGELOG](../CHANGELOG.md).
