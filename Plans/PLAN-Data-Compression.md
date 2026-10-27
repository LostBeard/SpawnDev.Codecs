# PLAN: General data compression on the accelerator (Geordi, 2026-10-26)

TJ 10-26: audio, image and video are covered; general data compression is missing. Parallel compress / decompress on
the GPU serves ML weights, Gaussian splat scenes, textures and anything streamed between workers. Same rules as every
other codec here: ILGPU kernels on all six backends, host = coordinator, three overloads (managed / TypedArray /
ArrayView1D), no bulk bytes on the .NET heap, bit-exact against the reference implementation of each format.

## What a GPU wants from a compressor

A serial bitstream is the enemy. Every format below is chosen because its data splits into independent pieces:
- **Independent chunks** (LZ4 / Snappy / zstd blocks, our 64 KB Deflate blocks): one workgroup per chunk.
- **Interleaved lanes inside one chunk** (GDeflate: 32 Deflate-style lanes; interleaved rANS): 32 threads decode one chunk.
- **Per-element transforms** (shuffle, delta, bit-pack, quantise): one thread per element.
Formats with a single serial stream (plain gzip of a whole file, as SPZ v2 writes) decode on one thread; we read them,
but we write the parallel variant wherever the format allows it.

## What we already have (Compression/)

DeflateGpu (parallel independent 64 KB blocks + sync flush), InflateGpu, ZlibCompressorGpu, Crc32Gpu; Huffman code
building and bit writers in the WebP lossless and JPEG encoders. GZip is a 10-byte header + Deflate + CRC32 + size, so
gzip read/write is a wrapper over what exists.

## Formats, in build order

### C1. Filters (the building blocks; one thread per element)
Byte shuffle and bitshuffle (Blosc / HDF5 bitshuffle compatible), delta and XOR-delta, zigzag, frame-of-reference
bit-packing, run-length. On their own they compress nothing; in front of an entropy coder or LZ they are most of the
win on numeric arrays (float tensors, splat columns, vertex streams). This is what nvCOMP calls "Cascaded"
(RLE + delta + bit-pack). Truth: the Blosc / bitshuffle C libraries.

### C2. Chunked container + LZ4 (block and frame format)
LZ4 is the generic fast lossless format: byte-aligned tokens, no entropy stage, decode = copy loops. Compress = one
workgroup per independent block (hash-chain match search); decompress = one workgroup per block. Interop: the LZ4 frame
format (lz4 CLI, Hugging Face Xet chunks use LZ4). Truth: liblz4 (decode byte-exact; encode = output decodes with liblz4
and ratio within a stated bound of `lz4 -1`, since LZ4 encoders are not unique).
Snappy is the same shape with a simpler format (nvCOMP rates it best overall on GPUs); add it after LZ4 by reusing the
LZ4 match finder.

**Done 2026-10-26 (Snappy):** `Compression/Snappy/SnappyCompressorGpu` + `SnappyDecompressorGpu`, raw and framed.
Compressor byte-identical with the Rust snap crate (cramjam = python-snappy's backend; 28 inputs x 2 formats up to
40 MB; Google's C++ library is not the target, its encoder differs since 1.1.8). snap compresses 64 KB blocks with a
fresh table, so one thread per block, resumable like LZ4. Decoder: framed chunks are independent segments; a raw stream
is cut by a one-thread element scan into ~64 KB segments, and a segment whose copies reach before it (no 64 KB block
structure, e.g. any whole-stream encoder) decodes in order after the independent ones. Fixtures
`scripts/compression/make_snappy_fixtures.py` (snap streams, hand-built streams with every element form, damaged
streams). CRC-32C is `Crc32Gpu` over the Castagnoli table.

### C3. Entropy core: interleaved rANS
GPU-friendly entropy coder (32 interleaved states per chunk; Recoil-style split metadata for more parallel decode).
Used by C4 and by the zstd FSE stage later. Truth: round trip + an independent Python rANS for the format we define.

### C4. Lossless ML weights (ZipNN / DFloat11 family) - first big consumer win
BF16 / FP16 / FP32 tensors: split the exponent byte from sign + mantissa, entropy-code only the exponents (the
exponent of BF16 weights carries ~2.6 bits), store sign + mantissa raw (byte-grouped for FP32). Published results:
BF16 ~30-33% smaller, FP32 ~17%, bit-exact. Decode runs on the accelerator straight into the weight buffer at load, so
models.spawndev.com downloads and the OPFS cache shrink with no change to inference. Huffman (have) for the DFloat11
layout (GPU decode via small LUTs), rANS (C3) for our own container. Consumer: SpawnDev.ILGPU.ML loaders (Tuvok).

**Measured 2026-10-26 on our own weights** (scratchpad weights_entropy.py / planes_lz.py, 64-256 MB samples, same
sample for every column; ratio = compressed / original, lower is better):

| weights | per-plane order-0 | per-plane best of raw / order-0 / deflate | ZipNN (zstd) | zstd -3 whole |
|---|---|---|---|---|
| LTX DiT BF16 | 0.686 | 0.686 | 0.719 | 0.788 |
| LTX VAE F16 | 0.672 | 0.660 | 0.675 | 0.772 |
| SD-Turbo F32 | 0.797 | 0.757 | 0.771 | 0.853 |

Exponent-only coding (DFloat11) is right for BF16 (0.682) but loses on F16 / F32 (0.86 / 0.85): their low mantissa
bytes compress too (upcast weights). So C4 = byte grouping (`ShuffleFilterGpu.ShuffleAsync`, element size 2 / 4 =
one plane per byte position) + per plane and chunk the smallest of raw / Huffman-only deflate / deflate (our GPU
zlib), decoded by InflateGpu. Every plane is a standard zlib stream.

**Shipped 2026-10-26 as `WeightPackGpu`** (format "SDWP" v1). Real weights on CUDA (RTX 4070, 258-512 MB samples):
BF16 0.676, FP16 0.667, FP32 0.608; unpack 1.2-1.6 GB/s, pack 0.12-0.14 GB/s (`weightpack-bench` in the console
host; _TestStats "bench" records). Next, performance (the ratio is done):
- **Unpack** (what model loading pays): one InflateGpu thread per 64 KiB chunk decodes bit by bit (puff-style canonical
  walk). A table-driven Huffman decode (one lookup per symbol from a per-chunk LUT built in shared memory) and more
  chunks in flight should take it from ~1.5 GB/s to tens of GB/s. Measure first: which part is the decode loop.
  **10-26:** first-level tables in InflateGpu (9-bit literal/length, 7-bit distance, built per block in the stream's
  scratch; longer codes walk the canonical code; read-ahead bytes go back before a stored block and at the end):
  LTX DiT BF16 unpack 1.94 -> 4.05 GB/s on CUDA, same bytes. Next: shared-memory tables, more chunks in flight.
- **Pack**: waves of 64 single-thread chunks (DeflateGpu scratch is 570 KB per chunk, mostly the unused LZ77 hash and
  token arrays at the Huffman-only level). A dedicated Huffman-only encoder (histogram per chunk with many threads,
  code build per chunk, bit offsets by prefix sum, every thread writes its bytes) removes the scratch and the waves.
  **Done 10-26 (8f844bb):** count (256 threads, shared histogram) / plan (DeflateGpu.PlanHuffmanOnly) / emit (prefix-
  summed bit offsets, Atomic.Or into words) / layout / copy, waves of 1024 chunks. Same bytes as before (SHA-256 of a
  16 MB pack unchanged); LTX DiT BF16 pack 0.134 -> 3.57 GB/s on CUDA, unpack 8.79 GB/s. Green on CPU, CUDA, OpenCL,
  WebGPU (ILGPU 5.4.7-local.7+: shared-memory atomics in WGSL), Wasm.

### C5. GDeflate (Microsoft DirectStorage, Apache 2.0)
Deflate's ratio with 32-lane parallel decode of ONE stream: the Deflate symbols reordered into 32 interleaved
bitstreams. Reference CPU compressor/decompressor + HLSL GPU decoder in github.com/microsoft/DirectStorage (GDeflate/).
Encoder reuses DeflateGpu's LZ77 + Huffman; only the bit layout differs. Truth: the reference library (byte-exact
decode; encode decodes with the reference). Best general format when ratio matters more than LZ4's speed.

**Decoder done 2026-10-26:** `Compression/GDeflate/GDeflateDecompressorGpu` (tile stream: 8-byte header, offset table,
64 KB tiles; one thread per tile, a wave per launch). NOTE the alphabet is Deflate64 (the reference builds libdeflate
with DEFLATE64): distance symbols 30 / 31 (32769 / 49153 + 14 bits) and length symbol 285 = 3 + 16 extra bits; stored
blocks carry LEN only. Fixtures from the reference built from source (scratchpad gdeflate_cli: cmake + MSVC driver
around GDeflate::Compress / Decompress), `scripts/compression/make_gdeflate_fixtures.py`: 31 streams at levels 1 / 6 /
9 / 12 + damaged ones. A Python model of the decoder (scratchpad gd_model.py) found the Deflate64 detail.
**32 lanes per tile done (default):** the HLSL shader's shared-memory path (lane 0 parses headers serially; Huffman
rounds of one symbol per lane, ballot refills, prefix sums, cooperative copies). RTX 4070 vs Microsoft's CPU decoder on
31 threads: LTX BF16 512 MB 3.10 vs 2.01 GB/s, C# source 36 MB 6.17 vs 5.65 GB/s (thread-per-tile was 1.90 / 0.39).
**10-27:** occupancy-bound (measured: +10 KB of shared-memory padding per tile dropped 3.08 -> 1.94 GB/s), so the
tables are packed two 16-bit entries per int (~10.4 -> ~6 KB per tile): 4.09 GB/s on BF16 weights, 7.12 on source.
Halving the barriers per round (double-buffered votes, lane-private cursor) gained nothing on CUDA (a 32-lane group is
one warp) and was not kept; ILGPU has no portable ballot (warp size differs per backend). Then ~3.9 KB per tile
(9-bit literal table, 4-bit code lengths, copy slots outside the lanes kernel, one vote/scan area): 4.32 GB/s on
weights; source text (550 tiles, all resident at once) stays 7.13. Knock-out: building the literal table twice costs
~1 %, so parallel table builds are not worth it. Small inputs are now latency-bound per tile, not occupancy-bound.
**Encoder done 2026-10-26:** `GDeflateCompressorGpu` (DeflateGpu's parse + `DeflateGpu.GDeflate.cs` plan / emit:
Deflate64 long matches by merging same-distance matches, best of dynamic / static / stored, the reference writer's
packet reservation). `scripts/compression/check_gdeflate_encoder.py`: 65 streams (13 inputs x levels 0-4) all decoded
by Microsoft's reference; level 3 within -3 .. +5 % of the reference's level 6.
SPEED (a loss, recorded): one thread per tile is slow on a GPU: 512 MB of BF16 weights at 0.033 GB/s (level 1, waves of
512 tiles; 0.008 at the old 64) vs 0.066 GB/s for the reference on ONE CPU thread. To be worth it the encoder needs
parallel match finding inside a tile (per-position hash / match candidates, then a parse), not one serial LZ77 per tile.
**10-27, encoder in parallel (9d73c0c), byte-identical output:** knock-outs first (level 4, 64 MB: writer 68 %, plan
18 %, the rest parse). Writer: 32 lanes per tile (lane = sub-stream; prefix sums hand out symbols and place the packet
reservations of a dry pass). Plan: 256 lanes into shared histograms. Parse (DeflateGpu.ParallelParse): ParseStep's
chains do not depend on the parse (every position is inserted in order), so 32 lanes build them, one thread per
position searches, and a cheap walk per tile makes ParseStep's greedy / lazy choices. RTX 4070, 512 MB BF16: level 1
0.474 GB/s (was 0.033), level 3 0.463, level 4 4.81 (was 0.061); vs the reference on one CPU thread at level 6: 0.066.

### C6. Gaussian splat containers
- **SPZ** (Niantic, MIT): gzip of quantised columns. Read v2/v3 (+ v4 when its spec is out), write with our parallel
  Deflate. Replaces the CPU GZipStream in SpawnDev.ILGPU.ML Formats/SPZParser + SPZExporter (7 GZip/Deflate stream
  uses in ML today = bulk bytes on the .NET heap in the browser). Truth: the nianticlabs/spz C++ library.
- **SOG** (PlayCanvas): quantised attributes as lossless WebP images + meta.json, Morton order. We already have
  lossless WebP encode + decode on the accelerator; SOG = Morton sort + quantise + our WebP. Truth: splat-transform.

### C7. zstd decode (then encode)
Interop with .zst everywhere (datasets, checkpoints). FSE (tANS) + Huffman + LZ sequences; blocks are independent only
across frames, so parallelism = many frames / many chunks. Decode first; encode after C2/C3 are solid.

## Not in this plan: texture compression (BC1-7, ETC2, ASTC)
GPU texture formats are an IMAGE codec track (Images/Texture), planned separately: BC1/3/4/5/7 encode for desktop
(WebGPU `texture-compression-bc`), ETC2 / ASTC for mobile and Quest (`texture-compression-etc2` / `-astc`), decode
kernels for verification and for backends without the feature. Consumers: SpawnScene, VoxelEngine, LostSpawns (Quest).

## Gates (every format)
- Bit-exact decode against the reference library on a real corpus (Silesia-style text/binary, real model tensors from
  models.spawndev.com, real .spz / .sog scenes), at production sizes (100 MB+), not 4 KB toys.
- Encode output decodes with the reference library; ratio and throughput logged against the reference CLI in
  _TestStats (Rule 12), losses included.
- No launch grows with the input (resumable + LaunchPacer); CPU-only tests once; GPU lanes CUDA, OpenCL, WebGPU,
  WebGL (where scatter stores allow), Wasm.
