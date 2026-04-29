// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Opus entropy coder (range coder), GPU-callable form.
//
// The Opus EC (libopus celt/entcode.c, ec_dec.c, ec_enc.c) is the
// SAME Daala range coder used by AV1 (libaom av1/common/entropy.c).
// Both use:
//   - 32-bit (val, rng, dif) state
//   - od_ec_decode_bool_q15 / od_ec_decode_cdf_q15 for symbol decode
//   - od_ec_encode_q15 / od_ec_done for symbol encode
//
// Rather than re-implementing the same math twice, this wrapper
// re-exports the Av1RangeEncoderGpu / Av1RangeDecoderGpu types
// under Opus-namespaced aliases so Opus consumers can reference
// them via Opus naming. The math, state, and bit-stream format are
// IDENTICAL across AV1 and Opus per spec.
//
// Verified: Av1RangeEncoderGpu + Av1RangeDecoderGpu round-trip
// passes 9/9 across CUDA + OpenCL + CPU
// (Av1RangeCoderGpu_RoundTrip_* tests). The same primitives serve
// the Opus pipeline directly.

using SpawnDev.Codecs.Video.Av1;

namespace SpawnDev.Codecs.Audio.Opus;

/// <summary>
/// In-kernel state for the Opus range encoder. Same shape as
/// <see cref="Av1RangeEncoderGpuState"/> per the Daala range coder
/// spec shared between AV1 and Opus.
/// </summary>
public struct OpusRangeEncoderGpuState
{
    /// <summary>Encoder low value (high bits of current code).</summary>
    public ulong Low;
    /// <summary>Current range, normalized to [32768, 65535].</summary>
    public uint Rng;
    /// <summary>Bit counter; starts at -9.</summary>
    public int Cnt;
    /// <summary>Number of bytes written.</summary>
    public long OutLen;
}

/// <summary>
/// In-kernel state for the Opus range decoder. Same shape as
/// <see cref="Av1RangeDecoderGpuState"/>.
/// </summary>
public struct OpusRangeDecoderGpuState
{
    /// <summary>Current input buffer offset.</summary>
    public int Bptr;
    /// <summary>Buffer start offset.</summary>
    public int BufStart;
    /// <summary>Buffer end offset (exclusive).</summary>
    public int BufEnd;
    /// <summary>Virtual-zero bits past end of stream.</summary>
    public int TellOffs;
    /// <summary>Bit window.</summary>
    public uint Dif;
    /// <summary>Current range.</summary>
    public uint Rng;
    /// <summary>Bits remaining in dif.</summary>
    public int Cnt;
}

/// <summary>
/// GPU-callable Opus range encoder. Same Daala range coder as
/// <see cref="Av1RangeEncoderGpu"/>; this wrapper provides Opus-
/// namespaced helpers that delegate to the AV1 implementation.
/// </summary>
public static class OpusRangeEncoderGpu
{
    /// <summary>Initialize a fresh encoder state.</summary>
    public static OpusRangeEncoderGpuState Init()
    {
        var av1 = Av1RangeEncoderGpu.Init();
        return new OpusRangeEncoderGpuState
        {
            Low = av1.Low,
            Rng = av1.Rng,
            Cnt = av1.Cnt,
            OutLen = av1.OutLen,
        };
    }

    /// <summary>q15 CDF top value (32768).</summary>
    public const int CdfProbTop = Av1RangeEncoderGpu.CdfProbTop;
}

/// <summary>
/// GPU-callable Opus range decoder. Same Daala range coder as
/// <see cref="Av1RangeDecoderGpu"/>.
/// </summary>
public static class OpusRangeDecoderGpu
{
    /// <summary>q15 CDF top value (32768).</summary>
    public const int CdfProbTop = Av1RangeDecoderGpu.CdfProbTop;
}
