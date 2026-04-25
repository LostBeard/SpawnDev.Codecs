// Tests for Vp9LoopFilterParamsParser (slice 186). Hand-encoded
// bitstreams exercise the spec layout from libvpx setup_loopfilter().

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    /// <summary>
    /// Encode a sequence of (value, bitWidth) pairs MSB-first into a
    /// byte array, padding the trailing partial byte with zeros. Mirror
    /// of the encoding side of Vp9BitReader.
    /// </summary>
    private static byte[] BitsToBytes(params (uint Value, int Bits)[] fields)
    {
        int totalBits = 0;
        foreach (var f in fields) totalBits += f.Bits;
        int totalBytes = (totalBits + 7) / 8;
        var bytes = new byte[totalBytes];
        int bytePos = 0;
        int bitPos = 0;  // bits already filled in the current byte, MSB-first.
        foreach (var (value, bits) in fields)
        {
            int remaining = bits;
            while (remaining > 0)
            {
                int avail = 8 - bitPos;
                int take = Math.Min(avail, remaining);
                int shift = remaining - take;
                uint chunk = (value >> shift) & ((1u << take) - 1);
                int destShift = avail - take;
                bytes[bytePos] |= (byte)(chunk << destShift);
                bitPos += take;
                if (bitPos == 8) { bitPos = 0; bytePos++; }
                remaining -= take;
            }
        }
        return bytes;
    }

    [TestMethod]
    public void Vp9LoopFilterParams_Disabled_ParsesMinimalLayout()
    {
        // filter_level=42 (6b), sharpness_level=3 (3b), mode_ref_delta_enabled=0 (1b)
        // Total 10 bits.
        var data = BitsToBytes(
            (42, 6),
            (3, 3),
            (0, 1));

        var p = Vp9LoopFilterParamsParser.Parse(data);

        Equal(42, p.FilterLevel);
        Equal(3, p.SharpnessLevel);
        Equal(false, p.ModeRefDeltaEnabled);
        Equal(false, p.ModeRefDeltaUpdate);
        Equal(0, p.RefDeltas.Length);
        Equal(0, p.ModeDeltas.Length);
    }

    [TestMethod]
    public void Vp9LoopFilterParams_EnabledWithoutUpdate_ParsesEnabledFlag()
    {
        // filter_level=10 (6b), sharpness_level=0 (3b),
        // mode_ref_delta_enabled=1 (1b), mode_ref_delta_update=0 (1b).
        var data = BitsToBytes(
            (10, 6),
            (0, 3),
            (1, 1),
            (0, 1));

        var p = Vp9LoopFilterParamsParser.Parse(data);

        Equal(10, p.FilterLevel);
        Equal(0, p.SharpnessLevel);
        Equal(true, p.ModeRefDeltaEnabled);
        Equal(false, p.ModeRefDeltaUpdate);
        // No deltas when no update.
        Equal(0, p.RefDeltas.Length);
        Equal(0, p.ModeDeltas.Length);
    }

    [TestMethod]
    public void Vp9LoopFilterParams_UpdateAllRefAndModeDeltas_PositiveValues()
    {
        // ref_deltas[0..3] all updated with +5, +10, +15, +20.
        // mode_deltas[0..1] all updated with +1, +2.
        // Each signed literal is 6 magnitude bits + 1 sign bit; sign=0 for positive.
        var data = BitsToBytes(
            (5, 6),  // filter_level
            (1, 3),  // sharpness_level
            (1, 1),  // mode_ref_delta_enabled
            (1, 1),  // mode_ref_delta_update
            // ref_deltas
            (1, 1), (5, 6), (0, 1),    // update[0]=1, mag=5, sign=0
            (1, 1), (10, 6), (0, 1),
            (1, 1), (15, 6), (0, 1),
            (1, 1), (20, 6), (0, 1),
            // mode_deltas
            (1, 1), (1, 6), (0, 1),
            (1, 1), (2, 6), (0, 1));

        var p = Vp9LoopFilterParamsParser.Parse(data);

        Equal(5, p.FilterLevel);
        Equal(1, p.SharpnessLevel);
        Equal(true, p.ModeRefDeltaEnabled);
        Equal(true, p.ModeRefDeltaUpdate);
        Equal(4, p.RefDeltas.Length);
        Equal(5, p.RefDeltas[0]!.Value);
        Equal(10, p.RefDeltas[1]!.Value);
        Equal(15, p.RefDeltas[2]!.Value);
        Equal(20, p.RefDeltas[3]!.Value);
        Equal(2, p.ModeDeltas.Length);
        Equal(1, p.ModeDeltas[0]!.Value);
        Equal(2, p.ModeDeltas[1]!.Value);
    }

    [TestMethod]
    public void Vp9LoopFilterParams_NegativeDeltas_SignBitFlipsValue()
    {
        // Just one ref delta = -7 (mag=7, sign=1).
        var data = BitsToBytes(
            (0, 6),  // filter_level
            (0, 3),  // sharpness_level
            (1, 1),  // mode_ref_delta_enabled
            (1, 1),  // mode_ref_delta_update
            // ref_deltas: only [0] is updated, rest skipped.
            (1, 1), (7, 6), (1, 1),  // update=1, mag=7, sign=1 -> -7
            (0, 1),                  // update[1]=0
            (0, 1),                  // update[2]=0
            (0, 1),                  // update[3]=0
            // mode_deltas: none updated.
            (0, 1),
            (0, 1));

        var p = Vp9LoopFilterParamsParser.Parse(data);

        Equal(true, p.ModeRefDeltaUpdate);
        Equal(-7, p.RefDeltas[0]!.Value);
        True(p.RefDeltas[1] == null);
        True(p.RefDeltas[2] == null);
        True(p.RefDeltas[3] == null);
        True(p.ModeDeltas[0] == null);
        True(p.ModeDeltas[1] == null);
    }

    [TestMethod]
    public void Vp9LoopFilterParams_PartialRefUpdate_OnlyMarkedSlotsCarryValues()
    {
        // ref_deltas: update [0]=+1, [2]=+3. Skip [1] and [3].
        // mode_deltas: update [1]=+5. Skip [0].
        var data = BitsToBytes(
            (15, 6),
            (2, 3),
            (1, 1),
            (1, 1),
            (1, 1), (1, 6), (0, 1),  // ref[0]=+1
            (0, 1),                  // ref[1] skip
            (1, 1), (3, 6), (0, 1),  // ref[2]=+3
            (0, 1),                  // ref[3] skip
            (0, 1),                  // mode[0] skip
            (1, 1), (5, 6), (0, 1)); // mode[1]=+5

        var p = Vp9LoopFilterParamsParser.Parse(data);

        Equal(15, p.FilterLevel);
        Equal(2, p.SharpnessLevel);
        Equal(1, p.RefDeltas[0]!.Value);
        True(p.RefDeltas[1] == null);
        Equal(3, p.RefDeltas[2]!.Value);
        True(p.RefDeltas[3] == null);
        True(p.ModeDeltas[0] == null);
        Equal(5, p.ModeDeltas[1]!.Value);
    }

    [TestMethod]
    public void Vp9LoopFilterParams_FilterLevelMaxValue()
    {
        var data = BitsToBytes(
            (63, 6),  // max filter_level
            (7, 3),   // max sharpness_level
            (0, 1));

        var p = Vp9LoopFilterParamsParser.Parse(data);

        Equal(63, p.FilterLevel);
        Equal(7, p.SharpnessLevel);
    }
}
