// Verify Vp8ForwardDct4x4Kernel produces bit-exact output vs the CPU
// reference Vp8ForwardTransform.ShortFdct4x4 across N random blocks.

#:project ../../SpawnDev.Codecs/SpawnDev.Codecs.csproj
using System;
using System.Linq;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using SpawnDev.Codecs.Video.Vp8;

const int blockCount = 50;
const int N = blockCount * 16;

var rng = new Random(42);
var input = new short[N];
for (int i = 0; i < N; i++) input[i] = (short)rng.Next(-1024, 1024);

// CPU reference output.
var refOut = new short[N];
for (int b = 0; b < blockCount; b++)
{
    Vp8ForwardTransform.ShortFdct4x4(
        input.AsSpan(b * 16, 16), rowStrideShorts: 4,
        refOut.AsSpan(b * 16, 16));
}

// Kernel output via CPU ILGPU backend.
using var ctx = Context.CreateDefault();
using var accelerator = ctx.GetCPUDevice(0).CreateAccelerator(ctx);
using var kernel = new Vp8ForwardDct4x4Kernel(accelerator);

using var dIn = accelerator.Allocate1D<short>(N);
using var dOut = accelerator.Allocate1D<short>(N);
dIn.View.CopyFromCPU(input);
kernel.Run(dIn.View, dOut.View, blockCount);
accelerator.Synchronize();
var kernelOut = new short[N];
dOut.View.CopyToCPU(kernelOut);

int mismatches = 0;
for (int i = 0; i < N; i++)
{
    if (refOut[i] != kernelOut[i])
    {
        if (mismatches < 10)
            Console.WriteLine($"  block {i / 16} pos {i % 16}: ref={refOut[i]} kernel={kernelOut[i]}");
        mismatches++;
    }
}

Console.WriteLine();
if (mismatches == 0)
    Console.WriteLine($"=== Vp8ForwardDct4x4Kernel BIT-EXACT vs CPU reference ({N} shorts, {blockCount} blocks) ===");
else
{
    Console.WriteLine($"FAIL: {mismatches}/{N} mismatches");
    Environment.Exit(1);
}
