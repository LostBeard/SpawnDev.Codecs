using SpawnDev.Codecs.Audio.Opus.Silk;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for <see cref="SilkStereoMsToLr.Apply"/> - the mid/side to left/right
/// conversion for stereo SILK output. Exercises zero-predictor + non-zero-predictor
/// cases and verifies state carries between frames.
/// </summary>
public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void StereoMsToLr_ConstantMidZeroSide_ConvergesToMidOnBothChannels()
    {
        // Input layout: x1[0..1] is state prefix (zero), x1[2..frameLength+1] is fresh mid data.
        // Libopus writes L output to x1[1..frameLength] and R output to x2[1..frameLength].
        // With constant mid = 1000 and zero side + zero predictors, after the low-pass filter
        // transient dies out, both L and R should converge to the mid value.
        var state = new SilkStereoState();
        int fsKHz = 16;
        int frameLength = 320;

        short[] x1 = new short[frameLength + 2];
        short[] x2 = new short[frameLength + 2];
        for (int i = 0; i < frameLength; i++) x1[i + 2] = 1000;

        SilkStereoMsToLr.Apply(state, x1, x2, new int[] { 0, 0 }, fsKHz, frameLength);

        // Check outputs past the filter transient region (~STEREO_INTERP_LEN_MS * fs_kHz + a few taps).
        int transient = SilkStereoMsToLr.StereoInterpLenMs * fsKHz + 10;
        for (int i = transient; i < frameLength; i++)
        {
            True(Math.Abs(x1[i + 1] - 1000) < 50, $"L[{i}] = {x1[i + 1]} should converge to ~1000");
            True(Math.Abs(x2[i + 1] - 1000) < 50, $"R[{i}] = {x2[i + 1]} should converge to ~1000");
        }
    }

    [TestMethod]
    public void StereoMsToLr_ZeroMidChannel_SideOnly_ProducesMirroredLR()
    {
        // Zero mid + non-zero side: L = 0 + side ~= side, R = 0 - side = -side. L + R = 0.
        var state = new SilkStereoState();
        int fsKHz = 16;
        int frameLength = 320;

        short[] x1 = new short[frameLength + 2]; // all zero
        short[] x2 = new short[frameLength + 2];
        for (int i = 0; i < frameLength; i++) x2[i + 2] = 500;

        SilkStereoMsToLr.Apply(state, x1, x2, new int[] { 0, 0 }, fsKHz, frameLength);

        // Past the transient, L + R = 2 * mid = 0.
        int transient = SilkStereoMsToLr.StereoInterpLenMs * fsKHz + 10;
        for (int i = transient; i < frameLength; i++)
        {
            True(x1[i + 1] >= short.MinValue && x1[i + 1] <= short.MaxValue);
            True(x2[i + 1] >= short.MinValue && x2[i + 1] <= short.MaxValue);
            int sumLR = x1[i + 1] + x2[i + 1];
            True(Math.Abs(sumLR) < 100, $"L + R should be ~0 (2*mid=0), got {sumLR} at pos {i}");
        }
    }

    [TestMethod]
    public void StereoMsToLr_StateCarriesAcrossFrames()
    {
        var state = new SilkStereoState();
        int fsKHz = 8;
        int frameLength = 160;

        short[] x1a = new short[frameLength + 2];
        short[] x2a = new short[frameLength + 2];
        for (int i = 0; i < frameLength; i++)
        {
            x1a[i + 1] = (short)(500 * Math.Sin(i * 0.1));
            x2a[i + 1] = (short)(200 * Math.Sin(i * 0.1));
        }

        SilkStereoMsToLr.Apply(state, x1a, x2a, new int[] { 1000, 500 }, fsKHz, frameLength);

        // After frame 1, state should carry last 2 samples and new predictors.
        Equal(1000, state.PredPrevQ13[0]);
        Equal(500, state.PredPrevQ13[1]);

        // Frame 2: just verify we can run again without crashing.
        short[] x1b = new short[frameLength + 2];
        short[] x2b = new short[frameLength + 2];
        for (int i = 0; i < frameLength; i++)
        {
            x1b[i + 1] = (short)(500 * Math.Sin((i + frameLength) * 0.1));
            x2b[i + 1] = (short)(200 * Math.Sin((i + frameLength) * 0.1));
        }
        SilkStereoMsToLr.Apply(state, x1b, x2b, new int[] { 800, 400 }, fsKHz, frameLength);

        // Predictors updated for frame 2.
        Equal(800, state.PredPrevQ13[0]);
        Equal(400, state.PredPrevQ13[1]);
    }

    [TestMethod]
    public void StereoMsToLr_Reset_ClearsAllState()
    {
        var state = new SilkStereoState();
        state.SMid[0] = 100;
        state.SSide[1] = -50;
        state.PredPrevQ13[0] = 1234;

        state.Reset();

        Equal((short)0, state.SMid[0]);
        Equal((short)0, state.SMid[1]);
        Equal((short)0, state.SSide[0]);
        Equal((short)0, state.SSide[1]);
        Equal(0, state.PredPrevQ13[0]);
        Equal(0, state.PredPrevQ13[1]);
    }

    [TestMethod]
    public void StereoMsToLr_NullState_Throws()
    {
        short[] x1 = new short[162];
        short[] x2 = new short[162];
        Throws<ArgumentNullException>(() =>
            SilkStereoMsToLr.Apply(null!, x1, x2, new int[] { 0, 0 }, 8, 160));
    }

    [TestMethod]
    public void StereoMsToLr_UndersizedBuffers_Throw()
    {
        var state = new SilkStereoState();
        short[] small = new short[100];
        short[] ok = new short[162];
        Throws<ArgumentException>(() =>
            SilkStereoMsToLr.Apply(state, small, ok, new int[] { 0, 0 }, 8, 160));
        Throws<ArgumentException>(() =>
            SilkStereoMsToLr.Apply(state, ok, small, new int[] { 0, 0 }, 8, 160));
    }
}
