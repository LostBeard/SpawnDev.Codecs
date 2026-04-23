using SpawnDev.Codecs.Audio.Opus.Silk;
using SpawnDev.Codecs.EntropyCoders;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Round-trip tests for <see cref="SilkShellCoder.Decode"/> against its own
/// <see cref="SilkShellCoder.Encode"/>. Exercises every total pulse count from
/// 0 to SILK_MAX_PULSES, plus several randomized distributions at higher counts.
/// </summary>
public abstract partial class CodecsTestBase
{
    // -------- Trivial boundaries --------

    [TestMethod]
    public void ShellCoder_ZeroPulses_AllZeros()
    {
        short[] zeros = new short[SilkConstants.SHELL_CODEC_FRAME_LENGTH];

        var enc = new OpusRangeEncoder(64);
        SilkShellCoder.Encode(enc, zeros, pulsesTotal: 0);
        enc.Done();

        short[] decoded = new short[SilkConstants.SHELL_CODEC_FRAME_LENGTH];
        var dec = new OpusRangeDecoder(enc.ToArray());
        SilkShellCoder.Decode(decoded, dec, pulsesTotal: 0);

        for (int i = 0; i < SilkConstants.SHELL_CODEC_FRAME_LENGTH; i++)
        {
            Equal((short)0, decoded[i], $"pos {i}");
        }
    }

    [TestMethod]
    public void ShellCoder_SinglePulseAtEachPosition_RoundTrip()
    {
        // For each of the 16 positions, put a single pulse there and verify it round-trips.
        for (int pos = 0; pos < SilkConstants.SHELL_CODEC_FRAME_LENGTH; pos++)
        {
            short[] pulses = new short[SilkConstants.SHELL_CODEC_FRAME_LENGTH];
            pulses[pos] = 1;

            var enc = new OpusRangeEncoder(64);
            SilkShellCoder.Encode(enc, pulses, pulsesTotal: 1);
            enc.Done();

            short[] decoded = new short[SilkConstants.SHELL_CODEC_FRAME_LENGTH];
            var dec = new OpusRangeDecoder(enc.ToArray());
            SilkShellCoder.Decode(decoded, dec, pulsesTotal: 1);

            for (int i = 0; i < SilkConstants.SHELL_CODEC_FRAME_LENGTH; i++)
            {
                Equal(pulses[i], decoded[i], $"single-pulse at {pos}, dec pos {i}");
            }
        }
    }

    // -------- Full range of pulse totals --------

    [TestMethod]
    public void ShellCoder_EvenlyDistributedPulses_AllTotals_RoundTrip()
    {
        // For each total in [0, 16], distribute as evenly as possible across 16 slots.
        for (int total = 0; total <= SilkConstants.SILK_MAX_PULSES; total++)
        {
            short[] pulses = new short[SilkConstants.SHELL_CODEC_FRAME_LENGTH];
            int baseValue = total / SilkConstants.SHELL_CODEC_FRAME_LENGTH;
            int remainder = total % SilkConstants.SHELL_CODEC_FRAME_LENGTH;
            for (int i = 0; i < SilkConstants.SHELL_CODEC_FRAME_LENGTH; i++)
            {
                pulses[i] = (short)(baseValue + (i < remainder ? 1 : 0));
            }

            var enc = new OpusRangeEncoder(64);
            SilkShellCoder.Encode(enc, pulses, pulsesTotal: total);
            enc.Done();

            short[] decoded = new short[SilkConstants.SHELL_CODEC_FRAME_LENGTH];
            var dec = new OpusRangeDecoder(enc.ToArray());
            SilkShellCoder.Decode(decoded, dec, pulsesTotal: total);

            for (int i = 0; i < SilkConstants.SHELL_CODEC_FRAME_LENGTH; i++)
            {
                Equal(pulses[i], decoded[i], $"total={total}, pos={i}");
            }
        }
    }

    // -------- Randomized distributions --------

    [TestMethod]
    public void ShellCoder_RandomizedDistributions_RoundTrip()
    {
        var rng = new Random(0x0EADBEEF);
        for (int trial = 0; trial < 200; trial++)
        {
            int total = rng.Next(0, SilkConstants.SILK_MAX_PULSES + 1);
            short[] pulses = new short[SilkConstants.SHELL_CODEC_FRAME_LENGTH];
            int remaining = total;
            for (int i = 0; i < SilkConstants.SHELL_CODEC_FRAME_LENGTH - 1 && remaining > 0; i++)
            {
                int take = rng.Next(0, remaining + 1);
                pulses[i] = (short)take;
                remaining -= take;
            }
            pulses[SilkConstants.SHELL_CODEC_FRAME_LENGTH - 1] = (short)remaining;

            var enc = new OpusRangeEncoder(64);
            SilkShellCoder.Encode(enc, pulses, pulsesTotal: total);
            enc.Done();

            short[] decoded = new short[SilkConstants.SHELL_CODEC_FRAME_LENGTH];
            var dec = new OpusRangeDecoder(enc.ToArray());
            SilkShellCoder.Decode(decoded, dec, pulsesTotal: total);

            for (int i = 0; i < SilkConstants.SHELL_CODEC_FRAME_LENGTH; i++)
            {
                if (pulses[i] != decoded[i])
                {
                    throw new Exception(
                        $"Trial {trial} total={total} pos={i}: expected {pulses[i]}, got {decoded[i]}. " +
                        $"Encoded=[{string.Join(",", pulses)}], Decoded=[{string.Join(",", decoded)}]");
                }
            }
        }
    }

    // -------- Argument validation --------

    [TestMethod]
    public void ShellCoder_Decode_NullRangeDecoder_Throws()
    {
        short[] pulses = new short[SilkConstants.SHELL_CODEC_FRAME_LENGTH];
        Throws<ArgumentNullException>(() => SilkShellCoder.Decode(pulses, null!, 4));
    }

    [TestMethod]
    public void ShellCoder_Decode_PulsesSpanTooSmall_Throws()
    {
        var enc = new OpusRangeEncoder(32);
        enc.Done();
        var dec = new OpusRangeDecoder(enc.ToArray());
        short[] pulses = new short[15]; // needs 16
        Throws<ArgumentException>(() => SilkShellCoder.Decode(pulses, dec, 0));
    }

    [TestMethod]
    public void ShellCoder_Encode_PulseSumMismatch_Throws()
    {
        var enc = new OpusRangeEncoder(64);
        short[] pulses = new short[SilkConstants.SHELL_CODEC_FRAME_LENGTH];
        pulses[0] = 2;
        Throws<ArgumentException>(() => SilkShellCoder.Encode(enc, pulses, pulsesTotal: 5));
    }

    // -------- Shell code tables sanity --------

    [TestMethod]
    public void ShellCoder_Offsets_HasExpectedShape()
    {
        Equal(SilkConstants.SILK_MAX_PULSES + 1, SilkShellCodeTables.Offsets.Length);
        Equal((byte)0, SilkShellCodeTables.Offsets[0]);
        Equal((byte)0, SilkShellCodeTables.Offsets[1]);
        Equal((byte)135, SilkShellCodeTables.Offsets[16]);
    }

    [TestMethod]
    public void ShellCoder_Tables_AllOneFiftyTwoBytes()
    {
        Equal(152, SilkShellCodeTables.Table0.Length);
        Equal(152, SilkShellCodeTables.Table1.Length);
        Equal(152, SilkShellCodeTables.Table2.Length);
        Equal(152, SilkShellCodeTables.Table3.Length);
    }
}
