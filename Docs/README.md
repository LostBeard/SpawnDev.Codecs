# SpawnDev.Codecs documentation

| Page | What it covers |
|------|----------------|
| [Status](status.md) | Every codec by backend: verified, partial or open, with the evidence. Start here before relying on a codec. |
| [Images](images.md) | PNG, JPEG, WebP, GIF, BMP and `ImageDecoderGpu`: API, options, limits, how each is checked |
| [Video](video.md) | VP8, VP9 and AV1 keyframe encoders and decoders: API, frame batches, limits |
| [Audio](audio.md) | Vorbis, FLAC, Opus and the audio containers |
| [API shape](api.md) | The three overloads every entry point has, buffer ownership, zero copy in the browser |
| [Architecture](architecture.md) | Host as coordinator, bounded launches, the browser shader-compile budget |
| [Testing](testing.md) | PlaywrightMultiTest, fixtures and their oracles, environment switches, red-checks |

Design documents and open work live in [`Plans/`](../Plans/); every change is in the [CHANGELOG](../CHANGELOG.md).
