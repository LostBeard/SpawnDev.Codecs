// Tests for the Av1RangeDecoder + Av1RangeEncoder pair (EntropyCoders).
// Round-trips mixed boolean + CDF + raw-bit symbols and verifies decode
// equality. Catches regressions in the libaom-equivalent state machine
// (renormalize / refill / done flush / carry propagation).

using SpawnDev.Codecs.EntropyCoders;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Av1RangeCoder_BoolsAtVariedProbabilities_RoundTrip()
    {
        var enc = new Av1RangeEncoder();
        uint[] probs = { 16384, 8192, 24576, 4096, 28672, 2048, 30720 };
        var rng = new Random(42);
        var symbols = new int[64];
        for (int i = 0; i < symbols.Length; i++) symbols[i] = rng.Next(2);

        for (int i = 0; i < symbols.Length; i++)
            enc.EncodeBoolQ15(symbols[i], probs[i % probs.Length]);
        var bytes = enc.Done();

        var dec = new Av1RangeDecoder(bytes);
        for (int i = 0; i < symbols.Length; i++)
            Equal(symbols[i], dec.DecodeBoolQ15(probs[i % probs.Length]));
    }

    [TestMethod]
    public void Av1RangeCoder_FourSymbolCdf_RoundTrip()
    {
        // 4 equal-probability symbols: cumprob = [8192, 16384, 24576, 32768]
        // icdf = [32768 - cumprob[i]] = [24576, 16384, 8192, 0]
        ushort[] icdf = { 24576, 16384, 8192, 0 };
        var rng = new Random(123);
        var symbols = new int[40];
        for (int i = 0; i < symbols.Length; i++) symbols[i] = rng.Next(4);

        var enc = new Av1RangeEncoder();
        for (int i = 0; i < symbols.Length; i++) enc.EncodeCdfQ15(symbols[i], icdf, 4);
        var bytes = enc.Done();

        var dec = new Av1RangeDecoder(bytes);
        for (int i = 0; i < symbols.Length; i++)
            Equal(symbols[i], dec.DecodeCdfQ15(icdf, 4));
    }

    [TestMethod]
    public void Av1RangeCoder_RawBits_RoundTrip()
    {
        uint[] values = { 0xA5, 0x12, 0xFF, 0x00, 0x7F };
        var enc = new Av1RangeEncoder();
        foreach (var v in values) enc.EncodeBits(v, 8);
        var bytes = enc.Done();

        var dec = new Av1RangeDecoder(bytes);
        for (int i = 0; i < values.Length; i++)
            Equal(values[i], dec.DecodeBits(8));
    }

    [TestMethod]
    public void Av1RangeCoder_MixedOps_RoundTripStress()
    {
        // 200 mixed ops covering all three decode primitives - exercises the
        // state machine across more than a single 64-bit normalization window.
        ushort[] icdf4 = { 24576, 16384, 8192, 0 };
        var rng = new Random(2024);
        var ops = new (int kind, int sym, uint param)[200];
        for (int i = 0; i < ops.Length; i++)
        {
            int kind = rng.Next(3);
            int sym; uint param;
            switch (kind)
            {
                case 0: sym = rng.Next(2); param = (uint)rng.Next(1, 32768); break;
                case 1: sym = rng.Next(4); param = 4; break;
                case 2: sym = rng.Next(256); param = 8; break;
                default: sym = 0; param = 0; break;
            }
            ops[i] = (kind, sym, param);
        }

        var enc = new Av1RangeEncoder();
        foreach (var op in ops)
        {
            switch (op.kind)
            {
                case 0: enc.EncodeBoolQ15(op.sym, op.param); break;
                case 1: enc.EncodeCdfQ15(op.sym, icdf4, 4); break;
                case 2: enc.EncodeBits((uint)op.sym, 8); break;
            }
        }
        var bytes = enc.Done();

        var dec = new Av1RangeDecoder(bytes);
        for (int i = 0; i < ops.Length; i++)
        {
            var op = ops[i];
            int got = op.kind switch
            {
                0 => dec.DecodeBoolQ15(op.param),
                1 => dec.DecodeCdfQ15(icdf4, 4),
                2 => (int)dec.DecodeBits(8),
                _ => -1,
            };
            Equal(op.sym, got);
        }
    }

    [TestMethod]
    public void Av1RangeDecoder_InitialState_MatchesLibaom()
    {
        // libaom od_ec_dec_init: rng=0x8000, cnt starts at -15 then refill
        // brings count up to (8 - 9 - bits_in_first_byte) initial position.
        // We can't introspect those directly but we CAN verify Tell starts
        // near 0 (libaom contract is 1-bit-of-precision floor at init).
        var bytes = new byte[] { 0xFF, 0x00, 0xAA, 0x55 };
        var dec = new Av1RangeDecoder(bytes);
        // Tell at init should be small positive (libaom contract).
        var tell = dec.Tell;
        True(tell >= 0 && tell < 32, $"initial Tell {tell} out of expected [0, 32)");
    }
}
