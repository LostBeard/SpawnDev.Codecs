using SpawnDev.Codecs.Audio.Transforms;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for <see cref="ImdctReference"/>. The IMDCT is a linear
/// transform from N frequency coefficients to 2N time-domain samples;
/// these tests exercise its core invariants:
///   * zero input -> zero output
///   * length contract (2N output)
///   * linearity: T(a + b) = T(a) + T(b)
///   * single-tone input produces a deterministic cosine pattern
/// </summary>
public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Imdct_ZeroInput_ZeroOutput()
    {
        const int n = 16;
        var input = new float[n];
        var output = new float[2 * n];
        ImdctReference.Transform(input, output);
        for (int i = 0; i < 2 * n; i++) Equal(0.0f, output[i]);
    }

    [TestMethod]
    public void Imdct_OutputLength_IsTwoN()
    {
        var input = new float[32];
        var output = ImdctReference.Transform(input);
        Equal(64, output.Length);
    }

    [TestMethod]
    public void Imdct_Linearity_SumOfTransformsEqualsTransformOfSum()
    {
        const int n = 8;
        var a = new float[n];
        var b = new float[n];
        var rng = new Random(42);
        for (int i = 0; i < n; i++)
        {
            a[i] = (float)(rng.NextDouble() * 2 - 1);
            b[i] = (float)(rng.NextDouble() * 2 - 1);
        }
        float[] ta = ImdctReference.Transform(a);
        float[] tb = ImdctReference.Transform(b);
        var ab = new float[n];
        for (int i = 0; i < n; i++) ab[i] = a[i] + b[i];
        float[] tab = ImdctReference.Transform(ab);
        // ta[i] + tb[i] should equal tab[i] within float precision.
        for (int i = 0; i < 2 * n; i++)
        {
            float expected = ta[i] + tb[i];
            True(Math.Abs(expected - tab[i]) < 1e-4f,
                $"linearity mismatch at {i}: expected {expected}, got {tab[i]}");
        }
    }

    [TestMethod]
    public void Imdct_SingleCoefficient_ProducesCosinePattern()
    {
        // For a single coefficient X[0] = 1 with N=4:
        //   theta = (pi/4) * (n + 0.5 + 2) * 0.5 = (pi/8) * (n + 2.5)
        //   y[n] = cos(theta)
        const int n = 4;
        var input = new float[n] { 1, 0, 0, 0 };
        var output = ImdctReference.Transform(input);
        Equal(2 * n, output.Length);
        for (int i = 0; i < 2 * n; i++)
        {
            double expected = Math.Cos(Math.PI / 8 * (i + 2.5));
            True(Math.Abs(expected - output[i]) < 1e-4,
                $"sample {i}: expected {expected}, got {output[i]}");
        }
    }

    [TestMethod]
    public void Imdct_InputTooShort_Throws()
    {
        Throws<ArgumentException>(() => ImdctReference.Transform(new float[0], new float[0]));
    }

    [TestMethod]
    public void Imdct_WrongOutputLength_Throws()
    {
        Throws<ArgumentException>(() => ImdctReference.Transform(new float[8], new float[10]));
    }

    [TestMethod]
    public void Imdct_ScalingSanity_AllOnesInputProducesNonzeroOutput()
    {
        // An all-ones frequency input produces a sum-of-cosines output that's
        // not identically zero. Verify average magnitude is finite + non-zero.
        const int n = 32;
        var input = new float[n];
        for (int i = 0; i < n; i++) input[i] = 1.0f;
        var output = ImdctReference.Transform(input);
        double sumAbs = 0;
        for (int i = 0; i < 2 * n; i++)
        {
            True(float.IsFinite(output[i]), $"sample {i} must be finite.");
            sumAbs += Math.Abs(output[i]);
        }
        True(sumAbs > 0, "Sum-of-cosines output cannot be identically zero.");
    }

    [TestMethod]
    public void Imdct_SmallSizes_2_4_8_16_AllRun()
    {
        foreach (int n in new[] { 2, 4, 8, 16 })
        {
            var input = new float[n];
            for (int i = 0; i < n; i++) input[i] = (float)((i + 1) * 0.1);
            var output = ImdctReference.Transform(input);
            Equal(2 * n, output.Length);
            for (int i = 0; i < 2 * n; i++)
                True(float.IsFinite(output[i]), $"N={n} sample {i} not finite.");
        }
    }
}
