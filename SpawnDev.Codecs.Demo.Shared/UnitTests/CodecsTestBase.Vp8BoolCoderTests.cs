// Tests for the Vp8BoolDecoder + Vp8BoolEncoder pair (VP8 boolean
// arithmetic coder, RFC 6386 sec 7). Round-trips bools and raw values
// and verifies decode equality. Specifically catches the
// "missing post-emit lowvalue shift" bug pattern that would let state
// drift after every byte emit.

using SpawnDev.Codecs.Video.Vp8;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp8BoolCoder_AlternatingBoolsAtProbHalf_RoundTrip()
    {
        // 40 alternating bools at prob 128 - every "got 0 expected 1" failure
        // pattern surfaces here. Original encoder bug failed at index 11.
        var enc = new Vp8BoolEncoder();
        var symbols = new int[40];
        for (int i = 0; i < symbols.Length; i++) symbols[i] = i & 1;
        foreach (var s in symbols) enc.EncodeBool(s, 128);
        var bytes = enc.Stop();

        var dec = new Vp8BoolDecoder(bytes);
        for (int i = 0; i < symbols.Length; i++)
            Equal(symbols[i], dec.DecodeBool(128));
    }

    [TestMethod]
    public void Vp8BoolCoder_BoolsAtSkewedProbabilities_RoundTrip()
    {
        // Wide probability range catches both rare-branch and common-branch paths.
        var enc = new Vp8BoolEncoder();
        int[] probs = { 128, 64, 200, 32, 240, 16, 250, 6 };
        var rng = new Random(42);
        var symbols = new int[200];
        for (int i = 0; i < symbols.Length; i++) symbols[i] = rng.Next(2);

        for (int i = 0; i < symbols.Length; i++)
            enc.EncodeBool(symbols[i], probs[i % probs.Length]);
        var bytes = enc.Stop();

        var dec = new Vp8BoolDecoder(bytes);
        for (int i = 0; i < symbols.Length; i++)
            Equal(symbols[i], dec.DecodeBool(probs[i % probs.Length]));
    }

    [TestMethod]
    public void Vp8BoolCoder_RawValues_RoundTrip()
    {
        int[] values = { 0xA5, 0x12, 0xFF, 0x00, 0x7F, 0x3C, 0xBA, 0x10 };
        var enc = new Vp8BoolEncoder();
        foreach (var v in values) enc.EncodeValue(v, 8);
        var bytes = enc.Stop();

        var dec = new Vp8BoolDecoder(bytes);
        for (int i = 0; i < values.Length; i++)
            Equal(values[i], dec.DecodeValue(8));
    }

    [TestMethod]
    public void Vp8BoolCoder_MixedOpsStress_RoundTrip()
    {
        // 1000 mixed operations - exercises the carry-propagation path
        // (long runs of 0xFF in the buffer require backward carry walks).
        var enc = new Vp8BoolEncoder();
        int[] probs = { 128, 200, 64, 32, 240 };
        var rng = new Random(2024);
        var ops = new (int kind, int sym, int param)[1000];
        for (int i = 0; i < ops.Length; i++)
        {
            int kind = rng.Next(2);
            int sym = kind == 0 ? rng.Next(2) : rng.Next(256);
            int param = kind == 0 ? probs[rng.Next(probs.Length)] : 0;
            ops[i] = (kind, sym, param);
        }
        foreach (var op in ops)
        {
            if (op.kind == 0) enc.EncodeBool(op.sym, op.param);
            else enc.EncodeValue(op.sym, 8);
        }
        var bytes = enc.Stop();

        var dec = new Vp8BoolDecoder(bytes);
        for (int i = 0; i < ops.Length; i++)
        {
            var op = ops[i];
            int got = op.kind == 0 ? dec.DecodeBool(op.param) : dec.DecodeValue(8);
            Equal(op.sym, got);
        }
    }
}
