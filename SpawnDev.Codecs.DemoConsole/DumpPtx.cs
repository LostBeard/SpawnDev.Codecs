using System.Reflection;
using ILGPU;
using ILGPU.Backends.EntryPoints;
using ILGPU.Backends.PTX;
using ILGPU.Runtime.Cuda;
using SpawnDev.Codecs.Images.Texture;

namespace SpawnDev.Codecs.DemoConsole;

/// <summary>
/// Writes the PTX ILGPU generates for one (implicitly grouped) kernel of the library, for `ptxas -v` (registers, spills,
/// local memory) when a kernel is slower on CUDA than expected.
///
/// Run: SpawnDev.Codecs.DemoConsole dump-ptx Namespace.Type MethodName out.ptx
/// </summary>
internal static class DumpPtx
{
    public static int Run(string[] args)
    {
        if (args.Length < 4)
        {
            Console.WriteLine("usage: dump-ptx <Namespace.Type> <MethodName> <out.ptx>");
            return 2;
        }
        var type = typeof(BcnEncoderGpu).Assembly.GetType(args[1], throwOnError: true)!;
        var method = type.GetMethod(args[2], BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic)
            ?? throw new ArgumentException($"{args[1]}.{args[2]} not found");
        using var ctx = Context.Create(b => b.Cuda());
        using var acc = ctx.CreateCudaAccelerator(0);
        var kernel = (PTXCompiledKernel)acc.CompileKernel(EntryPointDescription.FromImplicitlyGroupedKernel(method));
        File.WriteAllText(args[3], kernel.PTXAssembly);
        Console.WriteLine($"{args[1]}.{args[2]}: {kernel.PTXAssembly.Length} chars of PTX -> {args[3]}");
        return 0;
    }
}
