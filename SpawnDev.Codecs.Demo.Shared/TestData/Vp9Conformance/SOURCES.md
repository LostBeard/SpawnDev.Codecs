# VP9 test data

## Vp9Conformance / Vp9ConformanceLarge
libvpx's VP9 decoder test vectors, profile 0 (`vp90-2-*`), each with libvpx's `.md5` of every decoded frame (I420,
cropped; MD5 as libvpx `test/md5_helper.h`). From the WebM Project's test data,
`https://storage.googleapis.com/downloads.webmproject.org/test_data/libvpx/`, the list in libvpx v1.17.0
`test/test_vectors.cc`, every file SHA-1-checked against v1.17.0 `test/test-data.sha1`. libvpx is BSD-3-Clause
(see NOTICE.md). `Vp9ConformanceLarge` holds the four largest (1920x1080 loop filter, large scaling, subpixel, SVC
1280x720), not embedded.

Verified with `scripts/video/libvpx_ref` (libvpx built as plain C): all 303 vectors match their `.md5` on every frame.

## Vp9Synthetic
Header paths no vector reaches (absolute segment data; also delta data with every feature): the first frame of
vp90-2-00-quantizer-20 with only its uncompressed header re-packed by `scripts/video/make_vp9_header_fixtures.py`
(segmentation enabled, update_map 0 so the tile syntax is unchanged). `.md5` and header lines from libvpx_ref.

## libvpx_headers.tsv.gz
`libvpx_ref -h` of every frame of every file above (file name, tab, the H line), the reference for the header reader.

## libvpx_contexts.gz
`libvpx_ref -c` (hooked libvpx, see scripts/video/libvpx_ref/build_libvpx_ref.py PATCHES) for every frame: the frame
context the frame loads and the context after its compressed header (+ tx_mode, reference_mode, compound refs),
packed by scripts/video/libvpx_ref/pack_contexts.py. (Not `.bin.gz`: "bin" is a culture code, MSBuild would move the
resource into a satellite assembly.)

## Vp9ComfyUI
Files written by ComfyUI 0.42.0's SaveWEBM node (codec vp9, fps 24; crf 32 at 512x512 / 832x480 / 1024x576, crf 10
and 50 at 512x512, and an RGBA 512x320 batch with alpha), run as a black box through a local server (LoadVideo ->
GetVideoComponents -> SaveWEBM from lossless synthetic-motion sources; LoadImage -> JoinImageWithAlpha ->
RepeatImageBatch for RGBA). The `.md5` files are libvpx_ref's decode of the colour stream, in the vectors' format.
The RGBA file's alpha plane is a second VP9 stream in Matroska BlockAdditions (`alpha_mode=1`), not covered by its
`.md5`.

Measured (libvpx_ref -h): profile 0, 4:2:0 8-bit, one frame per packet (no hidden / alt-ref frames), single
reference prediction, no segmentation, frame_parallel_decoding_mode = 1 on every frame (no backward adaptation),
switchable interpolation filters, tile columns log2 1 at 512 / 832 wide and 2 at 1024, refresh flags 0x01 / 0x03 / 0xff.
