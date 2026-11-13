using Microsoft.Extensions.DependencyInjection;
using SpawnDev.Codecs.DemoConsole;
using SpawnDev.Codecs.DemoConsole.UnitTests;
using SpawnDev.UnitTesting;

// Diagnostic commands (offline, no test harness). `wasm-vp9-dump` measures the VP9
// frame-entropy walker's EncodeFrameBody Wasm locals — the before/after for Geordi's
// CumulativeInlinedILBudget fix (ILGPU 4.9.16-local.1).
if (args.Length > 0 && args[0] == "wasm-vp9-dump")
    return await WasmVp9Dump.Run();

// `weightpack-bench <file.safetensors> [maxMB] [cuda|opencl|cpu] [reps]`: WeightPackGpu on real model weights.
if (args.Length > 0 && args[0] == "weightpack-bench")
    return await WeightPackBench.Run(args);

// `gdeflate-bench <file.gdf> <original> [cuda|opencl|cpu] [reps]`: GDeflateDecompressorGpu on a reference stream.
if (args.Length > 0 && args[0] == "gdeflate-bench")
    return await GDeflateBench.Run(args);

// `gdeflate-encode <in> <out.gdf> [level] [cpu|cuda|opencl]`: GDeflateCompressorGpu, for the reference check.
if (args.Length > 0 && args[0] == "gdeflate-encode")
    return await GDeflateEncode.Run(args);

// `zlib-encode <in> <out.zz> [level] [cpu|cuda|opencl] [time] [wave]`: ZlibCompressorGpu speed bench.
if (args.Length > 0 && args[0] == "zlib-encode")
    return await ZlibEncode.Run(args);

// `bcn-encode <in.rgba> <width> <height> <bc1|bc3|bc4|bc5|bc7> <out> [cpu|cuda|opencl] [time]`: BcnEncoderGpu speed bench.
if (args.Length > 0 && args[0] == "bcn-encode")
    return await BcnEncode.Run(args);

// `vp9-encode-bench <clip.webm> [--backend cpu|velocity|cuda|opencl] [--frames N] [--q 20,40,60] [--intervals 1,128]
// [--out dir]`: VideoFileEncoderGpu (VP9) on a real clip: video bytes, PSNR and speed per quantizer and key frame interval.
if (args.Length > 0 && args[0] == "vp9-encode-bench")
    return await Vp9EncodeBench.Run(args);

// `av1-encode-bench [--backend cpu|velocity|cuda|opencl] [--size 832x480] [--frames 12] [--warmup 2] [--q 40]`:
// Av1KeyframeEncoderGpu on a synthetic picture already on the accelerator: time per keyframe and encoded size.
if (args.Length > 0 && args[0] == "av1-encode-bench")
    return await Av1EncodeBench.Run(args);

// `vp9-key-bench [--backend cpu|velocity|cuda|opencl] [--size 512x512] [--frames 12] [--q 40] [--stages 1]`:
// Vp9KeyframeEncoderGpu on a synthetic picture already on the accelerator: time per key frame (and per stage).
if (args.Length > 0 && args[0] == "vp9-key-bench")
    return await Vp9KeyBench.Run(args);

// `vp9-psnr <source.webm> <encoded1.webm> [...] [--backend ...] [--frames N] [--label text]`: video bytes and PSNR (RGB
// and luma) of encoded WebM files against a source clip, e.g. libvpx's files of a clip beside this library's.
if (args.Length > 0 && args[0] == "vp9-psnr")
    return await Vp9PsnrCompare.Run(args);

// `dump-ptx|dump-glsl <Namespace.Type> <MethodName> <out>`: one kernel's PTX (for ptxas -v) or WebGL GLSL (+ its
// transform-feedback output count).
if (args.Length > 0 && (args[0] == "dump-ptx" || args[0] == "dump-glsl"))
    return DumpPtx.Run(args);

// Run SpawnDev.UnitTesting unit tests on desktop .NET via PlaywrightMultiTest harness.
try
{
    var services = new ServiceCollection();
    // One concrete test class per desktop ILGPU backend. Each currently inherits all
    // tests from CodecsTestBase; CELT kernel work will override per-backend accelerator.
    services.AddSingleton<CpuCodecsTests>();
    services.AddSingleton<CudaCodecsTests>();
    services.AddSingleton<OpenCLCodecsTests>();
    services.AddSingleton<VelocityCodecsTests>();
    var sp = services.BuildServiceProvider();
    var runner = new UnitTestRunner(sp, true);
    await ConsoleRunner.Run(args, runner);
}
catch (Exception ex)
{
    Console.Error.WriteLine(ex.Message);
    return 1;
}
return 0;
