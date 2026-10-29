// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt). See NOTICE.md for upstream attributions.
//
// The range-decoder operations CELT uses beyond OpusRangeDecoderGpu's (libopus celt/entdec.c ec_decode / ec_dec_update and
// celt/entcode.h ec_tell / ec_tell_frac), plus isqrt32 (celt/mathops.c), ported from Concentus 2.2.2 (BSD-3) EntropyCoder. They
// work on the same OpusRangeDecoderGpuState, so they interleave with its DecodeUint / DecodeBitLogP / DecodeIcdf calls.

using ILGPU;

namespace SpawnDev.Codecs.Audio.Opus.Celt;

/// <summary>CELT's extra range-decoder operations on <see cref="OpusRangeDecoderGpuState"/> (GPU-callable).</summary>
public static class CeltEntropyGpu
{
    /// <summary>The symbol frequency of a total of <paramref name="ft"/>, before <see cref="DecUpdate"/> (libopus ec_decode).</summary>
    public static uint Decode(ref OpusRangeDecoderGpuState state, uint ft)
    {
        state.Ext = state.Rng / ft;
        uint num = state.Val / state.Ext;
        return ft - EcMini(num + 1, ft);
    }

    /// <summary>The symbol frequency of a total of 2^<paramref name="bits"/>, before <see cref="DecUpdate"/> (libopus ec_decode_bin).</summary>
    public static uint DecodeBin(ref OpusRangeDecoderGpuState state, int bits)
    {
        state.Ext = state.Rng >> bits;
        uint num = state.Val / state.Ext;
        return (uint)(1 << bits) - EcMini(num + 1, (uint)(1 << bits));
    }

    /// <summary>Consume the symbol [<paramref name="fl"/>, <paramref name="fh"/>) of <paramref name="ft"/> (libopus ec_dec_update).</summary>
    public static void DecUpdate(ref OpusRangeDecoderGpuState state, ArrayView<byte> buf, int bufStart, uint storage, uint fl, uint fh, uint ft)
    {
        uint num = state.Ext * (ft - fh);
        state.Val -= num;
        state.Rng = ((fl != 0) ? (state.Ext * (fh - fl)) : (state.Rng - num));
        OpusRangeDecoderGpu.Normalize(ref state, buf, bufStart, storage);
    }

    /// <summary>Whole bits used so far (libopus ec_tell).</summary>
    public static int Tell(ref OpusRangeDecoderGpuState state)
    {
        return state.NBitsTotal - CeltFixed.EC_ILOG(state.Rng);
    }

    /// <summary>Bits used so far in 1/8 bit (libopus ec_tell_frac).</summary>
    public static uint TellFrac(ref OpusRangeDecoderGpuState state)
    {
        int num = state.NBitsTotal << 3;
        int num2 = CeltFixed.EC_ILOG(state.Rng);
        int num3 = (int)(state.Rng >> num2 - 16);
        uint num4 = (uint)((num3 >> 12) - 8);
        num4 += (((uint)num3 > Correction(num4)) ? 1u : 0u);
        num2 = (int)((num2 << 3) + num4);
        return (uint)(num - num2);
    }

    /// <summary>ec_tell_frac's correction table (libopus correction[8]).</summary>
    private static uint Correction(uint i)
    {
        switch (i)
        {
            case 0: return 35733u;
            case 1: return 38967u;
            case 2: return 42495u;
            case 3: return 46340u;
            case 4: return 50535u;
            case 5: return 55109u;
            case 6: return 60097u;
            default: return 65535u;
        }
    }

    /// <summary>Integer square root (libopus isqrt32).</summary>
    public static uint Isqrt32(uint val)
    {
        uint num = 0u;
        int num2 = CeltFixed.EC_ILOG(val) - 1 >> 1;
        uint num3 = (uint)(1 << num2);
        do
        {
            uint num4 = (num << 1) + num3 << num2;
            if (num4 <= val)
            {
                num += num3;
                val -= num4;
            }
            num3 >>= 1;
            num2--;
        }
        while (num2 >= 0);
        return num;
    }

    private static uint EcMini(uint a, uint b)
    {
        return a + ((b - a) & (uint)((b < a) ? (-1) : 0));
    }
}
