# Architecture

## The host is a pure coordinator

Every encoder and decoder runs entirely as ILGPU kernels. The host allocates buffers, uploads the input (when it is not
already on the device), launches kernels in order and reads back the finished output (or nothing, for the `ArrayView1D`
overloads). It does no codec math, no per-pixel or per-sample loops, no entropy coding and no bitstream assembly.

| Stage | Examples | Where |
|-------|----------|-------|
| Parallel | colour conversion, prediction, DCT / MDCT / WHT, quantization, loop filters, PNG filters, upsampling, candidate modes | kernels, one thread per pixel / block / candidate |
| Serial per stream | bool / range / Huffman coders, LZ77 and LZW, the libwebp RD loop, header writers | kernels, one thread per stream (several streams in parallel where the format allows: token partitions, deflate blocks, GIF frames) |
| Container | RIFF, PNG chunks and CRCs, JFIF segments, OBUs, Ogg pages | kernels, usually one thread per output byte |
| Coordination | buffers, launch order, the final readback | host |

Why: in Blazor WebAssembly the .NET runtime shares the page's thread, so CPU work freezes the page; every host / device
round trip costs latency on every backend; and one implementation on the device runs unchanged on all six backends.

## Bounded launches

A single launch that walks a whole picture or stream grows with the input. On a desktop GPU a long launch stalls the
compositor and can trip the operating system's GPU watchdog; in the browser it can lose the device. So every serial
stage whose work grows with the input is **resumable**: it processes a bounded chunk per launch (a macroblock row, a
few thousand tokens, one deflate block) and keeps its state (bit-writer position, coder state, cursors) in device
buffers between launches. The host knows the number of launches in advance or reads one done flag per batch; launch
loops go through `LaunchPacer`, which flushes and awaits at intervals so the browser's command queue stays short.

## The browser shader-compile budget

Browser backends compile each kernel at first use: WGSL goes through Tint and the platform's shader compiler (DXC or
FXC on Windows), GLSL through the WebGL driver. That compile cost grows superlinearly with the kernel, and Chrome
loses its GPU process on kernels that take too long. Measured on 2026-10-25 with an offline Dawn compile screen:

- The cost is driven by **constant-bound loop nests the compiler unrolls**, not by line count: a 13.8k-line kernel took
  134 s while a 8.9k-line one took 5 s. Twelve inlined prediction calls with 16x16 / 8x8 pixel loops cost 59 s alone;
  the same work written as one flat loop over all output samples compiles in 0.4 s and gives the same bytes.
- `[NoInlining]` helpers do not help: the shader compiler inlines every function into the entry point anyway.
- Large serial walkers are split into **stage kernels** at the codec's own seams (the lossy WebP RD loop is five
  kernels per macroblock), with the state between stages in device buffers.

The kernels over budget and the plan for each are in [Plans/PLAN-Oversized-Kernels.md](../Plans/PLAN-Oversized-Kernels.md).

## Library rules this code follows

- **Fix the library, not the consumer.** A bug found here in SpawnDev.ILGPU (the WGSL / GLSL code generators, copies,
  bindings) is fixed there, with a test, and this repo moves to the fixed version. Several ILGPU fixes of 2026-10 came
  from these codecs.
- **No CPU fallback branches.** The same kernels run on every backend; a backend that cannot run a codec (WebGL without
  atomics or scatter stores) skips it explicitly in the tests rather than taking a different path.
- **Correct first, then fast.** Each codec is first made bit-exact with its reference, then made parallel.
