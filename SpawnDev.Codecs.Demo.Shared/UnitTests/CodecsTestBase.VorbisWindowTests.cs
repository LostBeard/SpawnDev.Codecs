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
    [TestMethod]
    public void VorbisWindow_Canonical_ProducesValuesInUnitInterval()
    {
        var w = VorbisWindow.GenerateCanonical(64);
        Equal(64, w.Length);
        for (int i = 0; i < w.Length; i++)
        {
            True(w[i] >= 0f && w[i] <= 1f, $"w[{i}] = {w[i]} outside [0, 1]");
        }
    }

    [TestMethod]
    public void VorbisWindow_Canonical_SymmetricAroundCenter()
    {
        // Window is symmetric: w[i] = w[n - 1 - i] to float precision.
        var w = VorbisWindow.GenerateCanonical(128);
        for (int i = 0; i < 128; i++)
        {
            float opposite = w[128 - 1 - i];
            True(Math.Abs(w[i] - opposite) < 1e-5f,
                $"symmetry: w[{i}]={w[i]}, w[{128 - 1 - i}]={opposite}");
        }
    }

    [TestMethod]
    public void VorbisWindow_Canonical_PrincenBradleyComplementarity()
    {
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

    [TestMethod]
    public void VorbisWindow_Canonical_CenterSamples_AtOneHalf()
    {
        // The midpoint samples evaluate to sin(pi/4) = 1/sqrt(2) approximately.
        int n = 64;
        var w = VorbisWindow.GenerateCanonical(n);
        // Samples at i = n/2 - 1 and i = n/2 are close to the peak of the
        // rising/falling half, not exactly 1, but both should be high.
        True(w[n / 2 - 1] > 0.98f, $"w[n/2-1] = {w[n / 2 - 1]} too low");
        True(w[n / 2] > 0.98f, $"w[n/2] = {w[n / 2]} too low");
    }

    [TestMethod]
    public void VorbisWindow_Transition_BothLong_MatchesCanonical()
    {
        var transition = VorbisWindow.GenerateTransition(longSize: 64, shortSize: 16, prevLong: true, nextLong: true);
        var canonical = VorbisWindow.GenerateCanonical(64);
        for (int i = 0; i < 64; i++)
            True(Math.Abs(transition[i] - canonical[i]) < 1e-5f,
                $"transition[{i}] = {transition[i]} != canonical[{i}] = {canonical[i]}");
    }

    [TestMethod]
    public void VorbisWindow_Transition_LeftShort_HasZeroPrefix()
    {
        // prev=short -> the first (halfLong - halfShort) samples are zero,
        // then the short-window rise, then ones up to the center.
        var w = VorbisWindow.GenerateTransition(64, 16, prevLong: false, nextLong: true);
        int halfLong = 32;
        int halfShort = 8;
        int zeroEnd = halfLong - halfShort; // 24
        for (int i = 0; i < zeroEnd; i++) Equal(0f, w[i]);
        // Short window rise over samples [zeroEnd, zeroEnd + halfShort).
        for (int j = 0; j < halfShort; j++)
        {
            int i = zeroEnd + j;
            True(w[i] > 0f, $"rise sample {i} = {w[i]} should be > 0");
        }
    }

    [TestMethod]
    public void VorbisWindow_Transition_RightShort_HasZeroSuffix()
    {
        var w = VorbisWindow.GenerateTransition(64, 16, prevLong: true, nextLong: false);
        int halfLong = 32;
        int halfShort = 8;
        int fallEnd = halfLong + halfShort; // 40
        for (int i = fallEnd; i < 64; i++) Equal(0f, w[i]);
    }

    [TestMethod]
    public void VorbisWindow_OverlapAdd_PointwiseSum()
    {
        var prev = new float[] { 1f, 2f, 3f, 4f };
        var cur = new float[] { 10f, 20f, 30f, 40f };
        var output = new float[4];
        VorbisWindow.OverlapAdd(prev, cur, output);
        EqualFloats(new[] { 11f, 22f, 33f, 44f }, output);
    }

    [TestMethod]
    public void VorbisWindow_OverlapAdd_LengthMismatch_Throws()
    {
        Throws<ArgumentException>(() =>
            VorbisWindow.OverlapAdd(new float[4], new float[5], new float[4]));
    }

    [TestMethod]
    public void VorbisWindow_Canonical_LengthTooSmall_Throws()
    {
        Throws<ArgumentException>(() => VorbisWindow.GenerateCanonical(1));
    }
}
