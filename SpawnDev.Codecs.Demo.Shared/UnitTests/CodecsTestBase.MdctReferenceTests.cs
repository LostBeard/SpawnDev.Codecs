using SpawnDev.Codecs.Audio.Transforms;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for <see cref="MdctReference"/> + the MDCT/IMDCT TDAC relationship.
/// </summary>
public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Mdct_ZeroInput_ZeroOutput()
    {
        var input = new float[16];
        var output = MdctReference.Transform(input);
        Equal(8, output.Length);
        for (int i = 0; i < output.Length; i++) Equal(0.0f, output[i]);
    }

    [TestMethod]
    public void Mdct_OutputLength_IsHalfInput()
    {
        var input = new float[32];
        var output = MdctReference.Transform(input);
        Equal(16, output.Length);
    }

    [TestMethod]
    public void Mdct_Linearity()
    {
        const int n = 8;
        var a = new float[2 * n];
        var b = new float[2 * n];
        var rng = new Random(7);
        for (int i = 0; i < 2 * n; i++)
        {
            a[i] = (float)(rng.NextDouble() * 2 - 1);
            b[i] = (float)(rng.NextDouble() * 2 - 1);
        }
        float[] ta = MdctReference.Transform(a);
        float[] tb = MdctReference.Transform(b);
        var ab = new float[2 * n];
        for (int i = 0; i < 2 * n; i++) ab[i] = a[i] + b[i];
        float[] tab = MdctReference.Transform(ab);
        for (int i = 0; i < n; i++)
        {
            float expected = ta[i] + tb[i];
            True(Math.Abs(expected - tab[i]) < 1e-3f,
                $"linearity mismatch at {i}: expected {expected}, got {tab[i]}");
        }
    }

    [TestMethod]
    public void Mdct_OddInputLength_Throws()
    {
        Throws<ArgumentException>(() => MdctReference.Transform(new float[9], new float[4]));
    }

    [TestMethod]
    public void Mdct_WrongOutputLength_Throws()
    {
        Throws<ArgumentException>(() => MdctReference.Transform(new float[16], new float[10]));
    }

    [TestMethod]
    public void Mdct_Imdct_Roundtrip_TdacSymmetry()
    {
        // Time-domain alias cancellation (TDAC) property: for one block x of
        // length 2N, IMDCT(MDCT(x)) yields aliased time samples y such that
        //   y[n]          = N * (-x[N - 1 - n] + x[n])          for n in [0, N)
        //   y[n]          = N * ( x[n]         - x[3N - 1 - n]) for n in [N, 2N)
        // Actually the cleanest invariant for a standalone block (no overlap-add)
        // is that a signal with the right symmetry is perfectly reconstructed.
        // Test a signal in the TDAC null space: symmetric on the first half and
        // antisymmetric on the second, such that the aliasing cancels.
        const int n = 4;
        // Build x with x[n] = 0 for n in [0, N), x[n] = c[n-N] for n in [N, 2N)
        // where c is arbitrary. After MDCT/IMDCT, the aliasing produces a
        // predictable scaled output.
        // For a more robust test, just verify the roundtrip is deterministic
        // and that MDCT(IMDCT(X)) = N * X for all frequency inputs.
        var freqInput = new float[n];
        var rng = new Random(11);
        for (int i = 0; i < n; i++) freqInput[i] = (float)(rng.NextDouble() * 2 - 1);

        float[] timeDomain = ImdctReference.Transform(freqInput);
        float[] recovered = MdctReference.Transform(timeDomain);

        // MDCT(IMDCT(X)) = N * X by the definitions used here.
        for (int k = 0; k < n; k++)
        {
            float expected = n * freqInput[k];
            True(Math.Abs(expected - recovered[k]) < 1e-3f,
                $"roundtrip X[{k}]: expected {expected} (= N * {freqInput[k]}), got {recovered[k]}");
        }
    }

    [TestMethod]
    public void Mdct_InputOf2_RunsWithoutError()
    {
        // Smallest useful input (N = 1).
        var input = new float[2] { 0.5f, -0.25f };
        var output = MdctReference.Transform(input);
        Equal(1, output.Length);
        True(float.IsFinite(output[0]));
    }

    [TestMethod]
    public void Mdct_LargerSizes_ProduceFiniteOutput()
    {
        foreach (int inputLen in new[] { 16, 32, 64, 128 })
        {
            var input = new float[inputLen];
            var rng = new Random(inputLen);
            for (int i = 0; i < inputLen; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);
            var output = MdctReference.Transform(input);
            Equal(inputLen / 2, output.Length);
            for (int i = 0; i < output.Length; i++)
                True(float.IsFinite(output[i]), $"len={inputLen} coef {i} not finite.");
        }
    }
}
