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

### C3. Entropy core: interleaved rANS
GPU-friendly entropy coder (32 interleaved states per chunk; Recoil-style split metadata for more parallel decode).
Used by C4 and by the zstd FSE stage later. Truth: round trip + an independent Python rANS for the format we define.

### C4. Lossless ML weights (ZipNN / DFloat11 family) - first big consumer win
BF16 / FP16 / FP32 tensors: split the exponent byte from sign + mantissa, entropy-code only the exponents (the
exponent of BF16 weights carries ~2.6 bits), store sign + mantissa raw (byte-grouped for FP32). Published results:
BF16 ~30-33% smaller, FP32 ~17%, bit-exact. Decode runs on the accelerator straight into the weight buffer at load, so
models.spawndev.com downloads and the OPFS cache shrink with no change to inference. Huffman (have) for the DFloat11
layout (GPU decode via small LUTs), rANS (C3) for our own container. Consumer: SpawnDev.ILGPU.ML loaders (Tuvok).

### C5. GDeflate (Microsoft DirectStorage, Apache 2.0)
Deflate's ratio with 32-lane parallel decode of ONE stream: the Deflate symbols reordered into 32 interleaved
bitstreams. Reference CPU compressor/decompressor + HLSL GPU decoder in github.com/microsoft/DirectStorage (GDeflate/).
Encoder reuses DeflateGpu's LZ77 + Huffman; only the bit layout differs. Truth: the reference library (byte-exact
decode; encode decodes with the reference). Best general format when ratio matters more than LZ4's speed.

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
