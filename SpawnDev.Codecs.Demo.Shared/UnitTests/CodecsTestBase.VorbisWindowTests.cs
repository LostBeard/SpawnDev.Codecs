using SpawnDev.Codecs.Audio.Vorbis;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for <see cref="VorbisWindow"/>. Cover the canonical window shape,
/// Princen-Bradley complementarity, transition-window behaviour at long/short
/// block boundaries, and the overlap-add summation.
/// </summary>
public abstract partial class CodecsTestBase
{
    [TestMethod(Category = "CpuOnly")]
    public void VorbisWindow_Canonical_ProducesValuesInUnitInterval()
    {
        CpuOnlyTest();
        var w = VorbisWindow.GenerateCanonical(64);
        Equal(64, w.Length);
        for (int i = 0; i < w.Length; i++)
        {
            True(w[i] >= 0f && w[i] <= 1f, $"w[{i}] = {w[i]} outside [0, 1]");
        }
    }

    [TestMethod(Category = "CpuOnly")]
    public void VorbisWindow_Canonical_SymmetricAroundCenter()
    {
        CpuOnlyTest();
        // Window is symmetric: w[i] = w[n - 1 - i] to float precision.
        var w = VorbisWindow.GenerateCanonical(128);
        for (int i = 0; i < 128; i++)
        {
            float opposite = w[128 - 1 - i];
            True(Math.Abs(w[i] - opposite) < 1e-5f,
                $"symmetry: w[{i}]={w[i]}, w[{128 - 1 - i}]={opposite}");
        }
    }

    [TestMethod(Category = "CpuOnly")]
    public void VorbisWindow_Canonical_PrincenBradleyComplementarity()
    {
        CpuOnlyTest();
        // Vorbis window satisfies w[i]^2 + w[i + n/2]^2 = 1 (Princen-Bradley).
        int n = 256;
        var w = VorbisWindow.GenerateCanonical(n);
        int half = n / 2;
        for (int i = 0; i < half; i++)
        {
            float sum = w[i] * w[i] + w[i + half] * w[i + half];
            True(Math.Abs(sum - 1.0f) < 1e-5f,
                $"P-B at i={i}: {w[i]}^2 + {w[i + half]}^2 = {sum}, expected 1");
        }
    }

    [TestMethod(Category = "CpuOnly")]
    public void VorbisWindow_Canonical_CenterSamples_AtOneHalf()
    {
        CpuOnlyTest();
        // The midpoint samples evaluate to sin(pi/4) = 1/sqrt(2) approximately.
        int n = 64;
        var w = VorbisWindow.GenerateCanonical(n);
        // Samples at i = n/2 - 1 and i = n/2 are close to the peak of the
        // rising/falling half, not exactly 1, but both should be high.
        True(w[n / 2 - 1] > 0.98f, $"w[n/2-1] = {w[n / 2 - 1]} too low");
        True(w[n / 2] > 0.98f, $"w[n/2] = {w[n / 2]} too low");
    }

    [TestMethod(Category = "CpuOnly")]
    public void VorbisWindow_Transition_BothLong_MatchesCanonical()
    {
        CpuOnlyTest();
        var transition = VorbisWindow.GenerateTransition(longSize: 64, shortSize: 16, prevLong: true, nextLong: true);
        var canonical = VorbisWindow.GenerateCanonical(64);
        for (int i = 0; i < 64; i++)
            True(Math.Abs(transition[i] - canonical[i]) < 1e-5f,
                $"transition[{i}] = {transition[i]} != canonical[{i}] = {canonical[i]}");
    }

    [TestMethod(Category = "CpuOnly")]
    public void VorbisWindow_Transition_LeftShort_HasZeroPrefix()
    {
        CpuOnlyTest();
        // prev=short (Vorbis I sec 4.3.1): zeros before n/4 - short/4, the short
        // window's rise over [n/4 - short/4, n/4 + short/4), then ones up to the center.
        var w = VorbisWindow.GenerateTransition(64, 16, prevLong: false, nextLong: true);
        var shortWindow = VorbisWindow.GenerateCanonical(16);
        int zeroEnd = 64 / 4 - 16 / 4; // 12
        int halfShort = 8;
        for (int i = 0; i < zeroEnd; i++) Equal(0f, w[i]);
        for (int j = 0; j < halfShort; j++)
        {
            int i = zeroEnd + j;
            True(Math.Abs(w[i] - shortWindow[j]) < 1e-6f, $"rise sample {i} = {w[i]} != short window {shortWindow[j]}");
        }
        for (int i = zeroEnd + halfShort; i < 32; i++) Equal(1f, w[i]);
    }

    [TestMethod(Category = "CpuOnly")]
    public void VorbisWindow_Transition_RightShort_HasZeroSuffix()
    {
        CpuOnlyTest();
        // next=short: ones from the center to 3n/4 - short/4, the short window's
        // fall over [3n/4 - short/4, 3n/4 + short/4), then zeros.
        var w = VorbisWindow.GenerateTransition(64, 16, prevLong: true, nextLong: false);
        var shortWindow = VorbisWindow.GenerateCanonical(16);
        int fallStart = 3 * 64 / 4 - 16 / 4; // 44
        int fallEnd = 3 * 64 / 4 + 16 / 4; // 52
        for (int i = 32; i < fallStart; i++) Equal(1f, w[i]);
        for (int i = fallStart; i < fallEnd; i++)
            True(Math.Abs(w[i] - shortWindow[8 + i - fallStart]) < 1e-6f, $"fall sample {i} = {w[i]}");
        for (int i = fallEnd; i < 64; i++) Equal(0f, w[i]);
    }

    [TestMethod(Category = "CpuOnly")]
    public void VorbisWindow_OverlapAdd_PointwiseSum()
    {
        CpuOnlyTest();
        var prev = new float[] { 1f, 2f, 3f, 4f };
        var cur = new float[] { 10f, 20f, 30f, 40f };
        var output = new float[4];
        VorbisWindow.OverlapAdd(prev, cur, output);
        EqualFloats(new[] { 11f, 22f, 33f, 44f }, output);
    }

    [TestMethod(Category = "CpuOnly")]
    public void VorbisWindow_OverlapAdd_LengthMismatch_Throws()
    {
        CpuOnlyTest();
        Throws<ArgumentException>(() =>
            VorbisWindow.OverlapAdd(new float[4], new float[5], new float[4]));
    }

    [TestMethod(Category = "CpuOnly")]
    public void VorbisWindow_Canonical_LengthTooSmall_Throws()
    {
        CpuOnlyTest();
        Throws<ArgumentException>(() => VorbisWindow.GenerateCanonical(1));
    }
}
