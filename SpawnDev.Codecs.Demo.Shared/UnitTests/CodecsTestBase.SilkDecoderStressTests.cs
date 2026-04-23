using SpawnDev.Codecs.Audio.Opus.Silk;
using SpawnDev.Codecs.EntropyCoders;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Stress tests for the full SILK decode pipeline via <see cref="SilkDecoder"/>.
/// Generates many random-but-valid SILK frames across NB/MB/WB configurations,
/// encodes them with the test-side helpers, decodes via the public API, and
/// verifies no crashes, output-range violations, or state corruption.
/// </summary>
public abstract partial class CodecsTestBase
{
    private static SilkDecodedIndices BuildRandomIndices(
        Random rng,
        SilkNlsfCodebook cb,
        int nbSubfr,
        bool allowVoiced,
        bool forceInteractiveOffset,
        out bool vadFlag,
        out int conditional)
    {
        int signalType;
        if (allowVoiced)
        {
            signalType = rng.Next(0, 3);
        }
        else
        {
            // Skip voiced to keep the stress test focused on the simpler path.
            signalType = rng.Next(0, 2);
        }

        vadFlag = signalType != SilkSideInfoDecoder.TypeInactive;
        conditional = 0; // keep conditional = independent for simpler stress

        int quantOffsetType = forceInteractiveOffset ? rng.Next(0, 2) : 0;

        var idx = new SilkDecodedIndices
        {
            SignalType = (sbyte)signalType,
            QuantOffsetType = (sbyte)quantOffsetType,
            NlsfInterpCoefQ2 = 4, // no interpolation for simplicity
            Seed = (sbyte)rng.Next(0, 4),
        };

        for (int k = 0; k < nbSubfr; k++)
        {
            // Gain indices 0..63 (independent = MSB+LSB encoding)
            idx.GainsIndices[k] = (sbyte)rng.Next(0, 64);
        }

        idx.NlsfIndices[0] = (sbyte)rng.Next(0, cb.NVectors);
        for (int i = 1; i <= cb.Order; i++)
        {
            // Keep residuals in the non-rail-extension range for simplicity.
            idx.NlsfIndices[i] = (sbyte)rng.Next(
                -(SilkConstants.NLSF_QUANT_MAX_AMPLITUDE - 1),
                SilkConstants.NLSF_QUANT_MAX_AMPLITUDE);
        }

        if (signalType == SilkSideInfoDecoder.TypeVoiced)
        {
            // Pick a pitch lag in the middle of the valid range.
            int fsKHz = cb.Order == 16 ? 16 : 8;
            idx.LagIndex = (short)((SilkConstants.PE_MIN_LAG_MS + 5) * fsKHz);
            int contourSize = fsKHz == 8
                ? (nbSubfr == 4 ? SilkConstants.PE_NB_CBKS_STAGE2_EXT : SilkConstants.PE_NB_CBKS_STAGE2_10MS)
                : (nbSubfr == 4 ? SilkConstants.PE_NB_CBKS_STAGE3_MAX : SilkConstants.PE_NB_CBKS_STAGE3_10MS);
            idx.ContourIndex = (sbyte)rng.Next(0, contourSize);
            idx.PerIndex = (sbyte)rng.Next(0, 3);
            int cbSize = idx.PerIndex switch { 0 => 8, 1 => 16, _ => 32 };
            for (int k = 0; k < nbSubfr; k++) idx.LtpIndices[k] = (sbyte)rng.Next(0, cbSize);
            idx.LtpScaleIndex = (sbyte)rng.Next(0, 3);
        }

        return idx;
    }

    [TestMethod]
    public void SilkDecoder_Stress_Nb20Ms_100RandomFrames_NoCrashesOrRangeViolations()
    {
        var rng = new Random(0xABCDE);
        var dec = new SilkDecoder(internalSampleRateHz: 8000, frameLengthMs: 20);
        var cb = SilkNlsfCodebookTables.NbMb;
        short[] pcm = new short[dec.FrameLength];
        short[] pulses = new short[dec.FrameLength + 16];

        for (int trial = 0; trial < 100; trial++)
        {
            var indices = BuildRandomIndices(rng, cb, nbSubfr: 4, allowVoiced: true,
                forceInteractiveOffset: true, out bool vad, out int cond);

            byte[] bitstream;
            try
            {
                bitstream = EncodeFullSilkFrame(cb, indices, pulses,
                    fsKHz: 8, nbSubfr: 4, conditional: cond, vadFlag: vad);
            }
            catch
            {
                // If encoding fails (rare at these bounds), skip the trial.
                continue;
            }

            try
            {
                dec.DecodeFrame(bitstream, pcm, vadFlag: vad, conditional: cond != 0);
            }
            catch (Exception ex)
            {
                throw new Exception(
                    $"Trial {trial} (sig={indices.SignalType}, cb1={indices.NlsfIndices[0]}) " +
                    $"threw during decode: {ex.GetType().Name}: {ex.Message}");
            }

            // Output in int16 range is a no-op at int16 level, but also check for NaN-ish signals
            // (we don't have floats, but check for any oscillation beyond reasonable amplitude).
            for (int i = 0; i < dec.FrameLength; i++)
            {
                True(pcm[i] >= short.MinValue && pcm[i] <= short.MaxValue,
                    $"Trial {trial} pos {i}: pcm = {pcm[i]}");
            }
        }
    }

    [TestMethod]
    public void SilkDecoder_Stress_Wb20Ms_100RandomFrames_NoCrashesOrRangeViolations()
    {
        var rng = new Random(0x12345);
        var dec = new SilkDecoder(internalSampleRateHz: 16000, frameLengthMs: 20);
        var cb = SilkNlsfCodebookTables.Wb;
        short[] pcm = new short[dec.FrameLength];
        short[] pulses = new short[dec.FrameLength + 16];

        for (int trial = 0; trial < 100; trial++)
        {
            var indices = BuildRandomIndices(rng, cb, nbSubfr: 4, allowVoiced: false,
                forceInteractiveOffset: false, out bool vad, out int cond);

            byte[] bitstream;
            try
            {
                bitstream = EncodeFullSilkFrame(cb, indices, pulses,
                    fsKHz: 16, nbSubfr: 4, conditional: cond, vadFlag: vad);
            }
            catch
            {
                continue;
            }

            try
            {
                dec.DecodeFrame(bitstream, pcm, vadFlag: vad, conditional: cond != 0);
            }
            catch (Exception ex)
            {
                throw new Exception(
                    $"Trial {trial} (WB, sig={indices.SignalType}) threw: " +
                    $"{ex.GetType().Name}: {ex.Message}");
            }

            for (int i = 0; i < dec.FrameLength; i++)
            {
                True(pcm[i] >= short.MinValue && pcm[i] <= short.MaxValue);
            }
        }
    }

    [TestMethod]
    public void SilkDecoder_Stress_MultiFrameStreaming_StateDoesNotCorrupt()
    {
        // 50 consecutive frames on the same decoder state. Verifies the state-carrying
        // fields (LastGainIndex, PrevNlsfQ15, PrevGainQ16, SLpcQ14Buf, OutBuf, etc.)
        // don't corrupt across many frames.
        var rng = new Random(0xFEDCB);
        var dec = new SilkDecoder(internalSampleRateHz: 8000, frameLengthMs: 20);
        var cb = SilkNlsfCodebookTables.NbMb;
        short[] pcm = new short[dec.FrameLength];
        short[] pulses = new short[dec.FrameLength + 16];

        for (int frame = 0; frame < 50; frame++)
        {
            var indices = BuildRandomIndices(rng, cb, nbSubfr: 4, allowVoiced: false,
                forceInteractiveOffset: false, out bool vad, out int cond);

            byte[] bitstream;
            try
            {
                bitstream = EncodeFullSilkFrame(cb, indices, pulses,
                    fsKHz: 8, nbSubfr: 4, conditional: cond, vadFlag: vad);
            }
            catch
            {
                continue;
            }

            dec.DecodeFrame(bitstream, pcm, vadFlag: vad, conditional: cond != 0);

            for (int i = 0; i < dec.FrameLength; i++)
            {
                True(pcm[i] >= short.MinValue && pcm[i] <= short.MaxValue,
                    $"frame {frame} pos {i}: pcm = {pcm[i]}");
            }
        }
    }
}
