// Cross-backend tests for the Opus range ENCODER GPU port (OpusRangeEncoderGpu). A random script of encoder operations
// (ec_encode, ec_encode_bin, ec_enc_bit_logp, ec_enc_icdf with 8- and 16-bit tables, ec_enc_uint, ec_enc_bits,
// ec_enc_patch_initial_bits, ec_enc_shrink) runs three ways: Concentus 2.2.2's internal EntropyCoder (the libopus port
// SpawnDev.Codecs.References uses as its oracle), the CPU OpusRangeEncoder, and the GPU kernel. The packets must be
// byte-identical and ec_tell / ec_tell_frac must agree after every operation, including the overflow (error) path.

using System.Reflection;
using System.Reflection.Emit;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Audio.Opus;
using SpawnDev.Codecs.EntropyCoders;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    /// <summary>Concentus' internal EntropyCoder (encoder side) as a callable oracle.</summary>
    private static class ConcentusEntropyOracle
    {
        private const BindingFlags Any = BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance;
        private static readonly Type Ec = typeof(Concentus.OpusCodecFactory).Assembly.GetType("Concentus.Common.EntropyCoder", true)!;

        public delegate void InitFn(object ec, uint size);
        public delegate void Enc3Fn(object ec, Span<byte> buf, uint a, uint b, uint c);
        public delegate void BitLogPFn(object ec, Span<byte> buf, int val, uint logp);
        public delegate void IcdfFn(object ec, Span<byte> buf, int s, byte[] icdf, int ptr, uint ftb);
        public delegate void Enc2Fn(object ec, Span<byte> buf, uint a, uint b);
        public delegate void DoneFn(object ec, Span<byte> buf);
        public delegate int TellFn(object ec);
        public delegate uint TellFracFn(object ec);

        private static MethodInfo M(string name, params Type[] args)
            => Ec.GetMethod(name, Any, null, args, null) ?? throw new MissingMethodException("Concentus.Common.EntropyCoder", name);

        // A DynamicMethod that casts the object to EntropyCoder and calls the instance method with the other arguments.
        private static T Bind<T>(MethodInfo target) where T : Delegate
        {
            var parameters = typeof(T).GetMethod("Invoke")!.GetParameters().Select(p => p.ParameterType).ToArray();
            var dm = new DynamicMethod("ec_" + target.Name, typeof(T).GetMethod("Invoke")!.ReturnType, parameters,
                typeof(ConcentusEntropyOracle).Module, true);
            var il = dm.GetILGenerator();
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Castclass, Ec);
            for (int i = 1; i < parameters.Length; i++) il.Emit(OpCodes.Ldarg, (short)i);
            il.Emit(OpCodes.Callvirt, target);
            il.Emit(OpCodes.Ret);
            return (T)dm.CreateDelegate(typeof(T));
        }

        public static object Create() => Activator.CreateInstance(Ec, true)!;
        public static readonly InitFn Init = Bind<InitFn>(M("enc_init", typeof(uint)));
        public static readonly Enc3Fn Encode = Bind<Enc3Fn>(M("encode", typeof(Span<byte>), typeof(uint), typeof(uint), typeof(uint)));
        public static readonly Enc3Fn EncodeBin = Bind<Enc3Fn>(M("encode_bin", typeof(Span<byte>), typeof(uint), typeof(uint), typeof(uint)));
        public static readonly BitLogPFn BitLogP = Bind<BitLogPFn>(M("enc_bit_logp", typeof(Span<byte>), typeof(int), typeof(uint)));
        public static readonly IcdfFn Icdf = Bind<IcdfFn>(M("enc_icdf", typeof(Span<byte>), typeof(int), typeof(byte[]), typeof(int), typeof(uint)));
        public static readonly Enc2Fn Uint = Bind<Enc2Fn>(M("enc_uint", typeof(Span<byte>), typeof(uint), typeof(uint)));
        public static readonly Enc2Fn Bits = Bind<Enc2Fn>(M("enc_bits", typeof(Span<byte>), typeof(uint), typeof(uint)));
        public static readonly Enc2Fn Patch = Bind<Enc2Fn>(M("enc_patch_initial_bits", typeof(Span<byte>), typeof(uint), typeof(uint)));
        public static readonly DoneFn Done = Bind<DoneFn>(M("enc_done", typeof(Span<byte>)));
        public static readonly TellFn Tell = Bind<TellFn>(M("tell"));
        public static readonly TellFracFn TellFrac = Bind<TellFracFn>(M("tell_frac"));
        public static int Error(object ec) => (int)Ec.GetField("error", Any)!.GetValue(ec)!;

        /// <summary>
        /// libopus <c>ec_enc_shrink</c>: memmove(buf + size - end_offs, buf + storage - end_offs, end_offs); storage = size.
        /// Concentus 2.2.2's enc_shrink passes MemMoveByte(buf, size - end_offs, storage - end_offs, end_offs), and
        /// MemMoveByte's order is (src, dst): it copies the wrong way and loses the raw-bit bytes. The oracle does the
        /// libopus move itself.
        /// </summary>
        public static void ShrinkLibopus(object ec, byte[] buf, uint size)
        {
            uint storage = (uint)Ec.GetField("storage", Any)!.GetValue(ec)!;
            uint endOffs = (uint)Ec.GetField("end_offs", Any)!.GetValue(ec)!;
            Buffer.BlockCopy(buf, (int)(storage - endOffs), buf, (int)(size - endOffs), (int)endOffs);
            Ec.GetField("storage", Any)!.SetValue(ec, size);
        }
    }

    /// <summary>A random valid inverse-CDF table: strictly decreasing from below <c>1 &lt;&lt; ftb</c> to 0.</summary>
    private static int[] RandomIcdf(Random rng, int symbols, int ftb)
    {
        int top = 1 << ftb;
        var cuts = new SortedSet<int>();
        while (cuts.Count < symbols - 1) cuts.Add(rng.Next(1, top));
        var icdf = cuts.Reverse().Append(0).ToArray();
        return icdf;
    }

    /// <summary>
    /// A random encoder script (<see cref="OpusRangeEncoderGpuTestKernel.OpStride"/> ints per op) over the 8-bit and
    /// 16-bit icdf tables, with one patch_initial_bits and, when <paramref name="withShrink"/>, one shrink whose size the
    /// CPU encoder picks (the bytes already written plus a margin).
    /// </summary>
    private static int[] RandomOpusEncoderScript(Random rng, int opCount, byte[] icdf8, int[] icdf8Starts, ushort[] icdf16,
        int[] icdf16Starts, int storage, bool withShrink)
    {
        var ops = new List<int>();
        var cpu = new OpusRangeEncoder(storage);
        int patchAt = rng.Next(12, opCount / 2), shrinkAt = withShrink ? rng.Next(opCount / 2, opCount - 8) : -1;
        for (int i = 0; i < opCount; i++)
        {
            int kind, a = 0, b = 0, c = 0;
            if (i == patchAt) { kind = OpusRangeEncoderGpuTestKernel.OpPatch; b = rng.Next(1, 9); a = rng.Next(1 << b); }
            else if (i == shrinkAt)
            {
                kind = OpusRangeEncoderGpuTestKernel.OpShrink;
                a = (int)(cpu.RangeBytes + cpu.EndBytes) + (storage - (int)(cpu.RangeBytes + cpu.EndBytes)) * 3 / 4;
            }
            else
            {
                kind = rng.Next(7);
                switch (kind)
                {
                    case OpusRangeEncoderGpuTestKernel.OpEncode:
                        c = rng.Next(2, 1 << 16); a = rng.Next(c); b = rng.Next(a + 1, c + 1); break;
                    case OpusRangeEncoderGpuTestKernel.OpEncodeBin:
                        c = rng.Next(1, 16); a = rng.Next(1 << c); b = rng.Next(a + 1, (1 << c) + 1); break;
                    case OpusRangeEncoderGpuTestKernel.OpBitLogP:
                        b = rng.Next(1, 16); a = rng.Next(4) == 0 ? 1 : 0; break;
                    case OpusRangeEncoderGpuTestKernel.OpIcdf:
                    {
                        int t = rng.Next(icdf8Starts.Length - 1);
                        b = icdf8Starts[t]; a = rng.Next(icdf8Starts[t + 1] - b); c = 8; break;
                    }
                    case OpusRangeEncoderGpuTestKernel.OpIcdf16:
                    {
                        int t = rng.Next(icdf16Starts.Length - 1);
                        b = icdf16Starts[t]; a = rng.Next(icdf16Starts[t + 1] - b); c = 15; break;
                    }
                    case OpusRangeEncoderGpuTestKernel.OpUint:
                    {
                        uint ft = rng.Next(3) switch { 0 => (uint)rng.Next(2, 257), 1 => (uint)rng.Next(257, 1 << 20), _ => (uint)rng.NextInt64(1 << 20, uint.MaxValue) };
                        a = (int)(uint)rng.NextInt64(ft); b = (int)ft; break;
                    }
                    default:
                        b = rng.Next(1, 26); a = (int)(uint)rng.NextInt64(1L << b); break;
                }
            }
            ops.Add(kind); ops.Add(a); ops.Add(b); ops.Add(c);
            ApplyOpusEncoderOp(cpu, kind, a, b, c, icdf8, icdf16);
        }
        return ops.ToArray();
    }

    private static void ApplyOpusEncoderOp(OpusRangeEncoder enc, int kind, int a, int b, int c, byte[] icdf8, ushort[] icdf16)
    {
        switch (kind)
        {
            case OpusRangeEncoderGpuTestKernel.OpEncode: enc.Encode((uint)a, (uint)b, (uint)c); break;
            case OpusRangeEncoderGpuTestKernel.OpEncodeBin: enc.EncodeBin((uint)a, (uint)b, c); break;
            case OpusRangeEncoderGpuTestKernel.OpBitLogP: enc.EncodeBitLogP(a, b); break;
            case OpusRangeEncoderGpuTestKernel.OpIcdf: enc.EncodeIcdf(a, icdf8.AsSpan(b), c); break;
            case OpusRangeEncoderGpuTestKernel.OpIcdf16: enc.EncodeIcdf16(a, icdf16.AsSpan(b), c); break;
            case OpusRangeEncoderGpuTestKernel.OpUint: enc.EncodeUint((uint)a, (uint)b); break;
            case OpusRangeEncoderGpuTestKernel.OpBits: enc.EncodeBits((uint)a, b); break;
            case OpusRangeEncoderGpuTestKernel.OpPatch: enc.PatchInitialBits((uint)a, b); break;
            case OpusRangeEncoderGpuTestKernel.OpShrink: enc.Shrink(a); break;
        }
    }

    /// <summary>The script through Concentus: packet bytes (the final storage), error, and tell / tell_frac per op.</summary>
    private static (byte[] Packet, int Error, int[] Tells) ConcentusEncodeScript(int[] ops, byte[] icdf8, ushort[] icdf16, int storage)
    {
        var ec = ConcentusEntropyOracle.Create();
        var buf = new byte[storage];
        ConcentusEntropyOracle.Init(ec, (uint)storage);
        int n = ops.Length / OpusRangeEncoderGpuTestKernel.OpStride;
        var tells = new int[2 * n];
        uint size = (uint)storage;
        for (int i = 0; i < n; i++)
        {
            int kind = ops[4 * i], a = ops[4 * i + 1], b = ops[4 * i + 2], c = ops[4 * i + 3];
            switch (kind)
            {
                case OpusRangeEncoderGpuTestKernel.OpEncode: ConcentusEntropyOracle.Encode(ec, buf, (uint)a, (uint)b, (uint)c); break;
                case OpusRangeEncoderGpuTestKernel.OpEncodeBin: ConcentusEntropyOracle.EncodeBin(ec, buf, (uint)a, (uint)b, (uint)c); break;
                case OpusRangeEncoderGpuTestKernel.OpBitLogP: ConcentusEntropyOracle.BitLogP(ec, buf, a, (uint)b); break;
                case OpusRangeEncoderGpuTestKernel.OpIcdf: ConcentusEntropyOracle.Icdf(ec, buf, a, icdf8, b, (uint)c); break;
                case OpusRangeEncoderGpuTestKernel.OpIcdf16:
                {
                    // Concentus has no 16-bit icdf; ec_enc_icdf(s) IS ec_encode_bin(ft - icdf[s - 1], ft - icdf[s], ftb).
                    uint ft = 1u << c;
                    uint fl = a > 0 ? ft - icdf16[b + a - 1] : 0u;
                    ConcentusEntropyOracle.EncodeBin(ec, buf, fl, ft - icdf16[b + a], (uint)c);
                    break;
                }
                case OpusRangeEncoderGpuTestKernel.OpUint: ConcentusEntropyOracle.Uint(ec, buf, (uint)a, (uint)b); break;
                case OpusRangeEncoderGpuTestKernel.OpBits: ConcentusEntropyOracle.Bits(ec, buf, (uint)a, (uint)b); break;
                case OpusRangeEncoderGpuTestKernel.OpPatch: ConcentusEntropyOracle.Patch(ec, buf, (uint)a, (uint)b); break;
                case OpusRangeEncoderGpuTestKernel.OpShrink: ConcentusEntropyOracle.ShrinkLibopus(ec, buf, (uint)a); size = (uint)a; break;
            }
            tells[2 * i] = ConcentusEntropyOracle.Tell(ec);
            tells[2 * i + 1] = (int)ConcentusEntropyOracle.TellFrac(ec);
        }
        ConcentusEntropyOracle.Done(ec, buf);
        return (buf.AsSpan(0, (int)size).ToArray(), ConcentusEntropyOracle.Error(ec), tells);
    }

    private async Task RunOpusRangeEncoderScriptTest(int seed, int opCount, int storage, bool withShrink, bool expectOverflow)
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            var rng = new Random(seed);
            // Eight 8-bit tables (2..32 symbols, ftb 8) and four 16-bit tables (2..64 symbols, ftb 15), packed.
            var t8 = Enumerable.Range(0, 8).Select(k => RandomIcdf(rng, 2 + rng.Next(31), 8)).ToArray();
            var t16 = Enumerable.Range(0, 4).Select(k => RandomIcdf(rng, 2 + rng.Next(63), 15)).ToArray();
            byte[] icdf8 = t8.SelectMany(t => t.Select(v => (byte)v)).ToArray();
            ushort[] icdf16 = t16.SelectMany(t => t.Select(v => (ushort)v)).ToArray();
            int[] starts8 = new int[t8.Length + 1], starts16 = new int[t16.Length + 1];
            for (int k = 0; k < t8.Length; k++) starts8[k + 1] = starts8[k] + t8[k].Length;
            for (int k = 0; k < t16.Length; k++) starts16[k + 1] = starts16[k] + t16[k].Length;

            int[] ops = RandomOpusEncoderScript(rng, opCount, icdf8, starts8, icdf16, starts16, storage, withShrink);
            var (refPacket, refError, refTells) = ConcentusEncodeScript(ops, icdf8, icdf16, storage);
            if ((refError != 0) != expectOverflow)
                throw new Exception($"test setup: Concentus error {refError}, expected overflow = {expectOverflow}");

            // The CPU encoder (SpawnDev.Codecs.References) on the same script.
            var cpu = new OpusRangeEncoder(storage);
            for (int i = 0; i < opCount; i++)
                ApplyOpusEncoderOp(cpu, ops[4 * i], ops[4 * i + 1], ops[4 * i + 2], ops[4 * i + 3], icdf8, icdf16);
            cpu.Done();
            EqualBytes(refPacket, cpu.ToArray(), "CPU OpusRangeEncoder packet vs Concentus");
            Equal(refError, cpu.Error, "CPU OpusRangeEncoder error vs Concentus");

            using var dOps = acc.Allocate1D(ops);
            using var dIcdf8 = acc.Allocate1D(icdf8);
            using var dIcdf16 = acc.Allocate1D(icdf16);
            using var dBuf = acc.Allocate1D<byte>(storage);
            using var dInfo = acc.Allocate1D<int>(OpusRangeEncoderGpuTestKernel.InfoHeader + 2 * opCount);
            using var kernel = new OpusRangeEncoderGpuTestKernel(acc);
            kernel.Run(dOps.View, opCount, dIcdf8.View, dIcdf16.View, dBuf.View, storage, dInfo.View);
            await acc.SynchronizeAsync();
            var info = await dInfo.View.CopyToHostAsync();
            var gpuBuf = await dBuf.View.CopyToHostAsync();

            Equal(refPacket.Length, info[0], "GPU final storage");
            Equal(refError, info[1], "GPU error vs Concentus");
            for (int i = 0; i < 2 * opCount; i++)
                if (info[OpusRangeEncoderGpuTestKernel.InfoHeader + i] != refTells[i])
                    throw new Exception($"op {i / 2} (kind {ops[4 * (i / 2)]}): GPU {(i % 2 == 0 ? "tell" : "tell_frac")} " +
                        $"{info[OpusRangeEncoderGpuTestKernel.InfoHeader + i]}, Concentus {refTells[i]}");
            EqualBytes(refPacket, gpuBuf.AsSpan(0, refPacket.Length).ToArray(), "GPU OpusRangeEncoderGpu packet vs Concentus");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod(Category = "NotOnWebGL")]
    public async Task OpusRangeEncoderGpu_RandomScript_ByteIdenticalWithConcentus()
        => await RunOpusRangeEncoderScriptTest(seed: 0x0E5C, opCount: 400, storage: 2400, withShrink: false, expectOverflow: false);

    [TestMethod(Category = "NotOnWebGL")]
    public async Task OpusRangeEncoderGpu_RandomScriptWithShrink_ByteIdenticalWithConcentus()
        => await RunOpusRangeEncoderScriptTest(seed: 0x0E5D, opCount: 400, storage: 2400, withShrink: true, expectOverflow: false);

    [TestMethod(Category = "NotOnWebGL")]
    public async Task OpusRangeEncoderGpu_Overflow_SameErrorAndBytesAsConcentus()
        => await RunOpusRangeEncoderScriptTest(seed: 0x0E5E, opCount: 120, storage: 24, withShrink: false, expectOverflow: true);
}
