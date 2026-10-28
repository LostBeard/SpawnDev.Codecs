using System.Reflection;
using System.Text.RegularExpressions;
using ILGPU;
using ILGPU.Backends.EntryPoints;
using ILGPU.Backends.PTX;
using ILGPU.Runtime.Cuda;
using SpawnDev.Codecs.Images.Texture;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.DemoConsole;

/// <summary>
/// Writes the code ILGPU generates for one (implicitly grouped) kernel of the library, to inspect it offline:
/// PTX for `ptxas -v` (registers, spills, local memory), or the WebGL GLSL with its transform-feedback output count
/// (each store site of an output view is one; a WebGL device captures at most
/// MAX_TRANSFORM_FEEDBACK_INTERLEAVED_COMPONENTS, 120 on the test machine).
///
/// Run: SpawnDev.Codecs.DemoConsole dump-ptx Namespace.Type MethodName out.ptx
///      SpawnDev.Codecs.DemoConsole dump-glsl Namespace.Type MethodName out.glsl
/// </summary>
internal static class DumpPtx
{
    public static int Run(string[] args)
    {
        if (args.Length < 4)
        {
            Console.WriteLine("usage: dump-ptx|dump-glsl <Namespace.Type> <MethodName> <out file>");
            return 2;
        }
        var type = typeof(BcnEncoderGpu).Assembly.GetType(args[1], throwOnError: true)!;
        var method = type.GetMethod(args[2], BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic)
            ?? throw new ArgumentException($"{args[1]}.{args[2]} not found");
        if (args[0] == "dump-glsl")
        {
            var gen = ShaderCompiler.Generate(method, CapabilityProfiles.WebGL2Baseline);
            File.WriteAllText(args[3], gen.Source);
            int outputs = Regex.Matches(gen.Source, @"^\s*(flat\s+)?out\s+\w+\s+\w+", RegexOptions.Multiline).Count;
            Console.WriteLine($"{args[1]}.{args[2]}: {gen.Source.Length} chars of GLSL, {outputs} transform-feedback outputs -> {args[3]}");
            return 0;
        }
        using var ctx = Context.Create(b => b.Cuda());
        using var acc = ctx.CreateCudaAccelerator(0);
        var kernel = (PTXCompiledKernel)acc.CompileKernel(EntryPointDescription.FromImplicitlyGroupedKernel(method));
        File.WriteAllText(args[3], kernel.PTXAssembly);
        Console.WriteLine($"{args[1]}.{args[2]}: {kernel.PTXAssembly.Length} chars of PTX -> {args[3]}");
        return 0;
    }
}
