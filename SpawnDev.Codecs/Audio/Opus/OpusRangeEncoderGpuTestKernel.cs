// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Test kernel for OpusRangeEncoderGpu: one thread runs a script of encoder operations (every ec_enc_* entry point),
// then Done(), and records ec_tell / ec_tell_frac after each operation, so the tests can compare the packet bytes and
// the bit accounting with Concentus' EntropyCoder and the CPU OpusRangeEncoder.

using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.Codecs.Audio.Opus;

/// <summary>Runs a script of <see cref="OpusRangeEncoderGpu"/> operations on the accelerator (tests).</summary>
public sealed class OpusRangeEncoderGpuTestKernel : IDisposable
{
    /// <summary>Ints per script operation: kind, a, b, c.</summary>
    public const int OpStride = 4;
    /// <summary><c>ec_encode(fl = a, fh = b, ft = c)</c>.</summary>
    public const int OpEncode = 0;
    /// <summary><c>ec_encode_bin(fl = a, fh = b, bits = c)</c>.</summary>
    public const int OpEncodeBin = 1;
    /// <summary><c>ec_enc_bit_logp(val = a, logp = b)</c>.</summary>
    public const int OpBitLogP = 2;
    /// <summary><c>ec_enc_icdf(s = a, icdf8 + b, ftb = c)</c>.</summary>
    public const int OpIcdf = 3;
    /// <summary>16-bit <c>ec_enc_icdf(s = a, icdf16 + b, ftb = c)</c>.</summary>
    public const int OpIcdf16 = 4;
    /// <summary><c>ec_enc_uint(fl = (uint)a, ft = (uint)b)</c>.</summary>
    public const int OpUint = 5;
    /// <summary><c>ec_enc_bits(fl = (uint)a, bits = b)</c>.</summary>
    public const int OpBits = 6;
    /// <summary><c>ec_enc_patch_initial_bits(val = a, nbits = b)</c>.</summary>
    public const int OpPatch = 7;
    /// <summary><c>ec_enc_shrink(size = a)</c>.</summary>
    public const int OpShrink = 8;

    /// <summary>Ints of <c>info</c> before the per-operation tell pairs: storage, error, offs, end_offs.</summary>
    public const int InfoHeader = 4;

    private readonly Action<Index1D, ArrayView<int>, int, ArrayView<byte>, ArrayView<ushort>, ArrayView<byte>, int, ArrayView<int>> _kernel;

    /// <summary>Compile the kernel on <paramref name="accelerator"/>.</summary>
    public OpusRangeEncoderGpuTestKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>, int, ArrayView<byte>, ArrayView<ushort>, ArrayView<byte>, int, ArrayView<int>>(ScriptKernel);
    }

    /// <summary>
    /// Run <paramref name="opCount"/> operations of <paramref name="ops"/> into <paramref name="outBuf"/> (a packet of
    /// <paramref name="storage"/> bytes), then Done(). <paramref name="info"/> receives the final storage, error, offs,
    /// end_offs, then ec_tell and ec_tell_frac after each operation (<see cref="InfoHeader"/> + 2 * opCount ints).
    /// </summary>
    public void Run(ArrayView<int> ops, int opCount, ArrayView<byte> icdf8, ArrayView<ushort> icdf16,
        ArrayView<byte> outBuf, int storage, ArrayView<int> info)
        => _kernel(1, ops, opCount, icdf8, icdf16, outBuf, storage, info);

    private static void ScriptKernel(Index1D index, ArrayView<int> ops, int opCount, ArrayView<byte> icdf8,
        ArrayView<ushort> icdf16, ArrayView<byte> buf, int storageIn, ArrayView<int> info)
    {
        if (index != 0) return;
        var st = OpusRangeEncoderGpu.Init();
        uint storage = (uint)storageIn;
        for (int i = 0; i < opCount; i++)
        {
            int kind = ops[i * OpStride];
            int a = ops[i * OpStride + 1];
            int b = ops[i * OpStride + 2];
            int c = ops[i * OpStride + 3];
            if (kind == OpEncode) OpusRangeEncoderGpu.Encode(ref st, buf, 0, storage, (uint)a, (uint)b, (uint)c);
            else if (kind == OpEncodeBin) OpusRangeEncoderGpu.EncodeBin(ref st, buf, 0, storage, (uint)a, (uint)b, c);
            else if (kind == OpBitLogP) OpusRangeEncoderGpu.EncodeBitLogP(ref st, buf, 0, storage, a, b);
            else if (kind == OpIcdf) OpusRangeEncoderGpu.EncodeIcdf(ref st, buf, 0, storage, a, icdf8, b, c);
            else if (kind == OpIcdf16) OpusRangeEncoderGpu.EncodeIcdf16(ref st, buf, 0, storage, a, icdf16, b, c);
            else if (kind == OpUint) OpusRangeEncoderGpu.EncodeUint(ref st, buf, 0, storage, (uint)a, (uint)b);
            else if (kind == OpBits) OpusRangeEncoderGpu.EncodeBits(ref st, buf, 0, storage, (uint)a, b);
            else if (kind == OpPatch) OpusRangeEncoderGpu.PatchInitialBits(ref st, buf, 0, (uint)a, b);
            else if (kind == OpShrink) storage = OpusRangeEncoderGpu.Shrink(ref st, buf, 0, storage, (uint)a);
            info[InfoHeader + 2 * i] = OpusRangeEncoderGpu.Tell(ref st);
            info[InfoHeader + 2 * i + 1] = (int)OpusRangeEncoderGpu.TellFrac(ref st);
        }
        OpusRangeEncoderGpu.Done(ref st, buf, 0, storage);
        info[0] = (int)storage;
        info[1] = st.Error;
        info[2] = (int)st.Offs;
        info[3] = (int)st.EndOffs;
    }

    /// <inheritdoc/>
    public void Dispose() { }
}
