using SpawnDev.Codecs.Audio.Vorbis;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for <see cref="VorbisInverseCoupling"/>. Verifies each of the four
/// coupling quadrants (mag sign × ang sign) produces the expected M/A pair
/// per Vorbis I Section 4.3.8.2, plus the no-coupling and multi-step paths.
/// </summary>
public abstract partial class CodecsTestBase
{
    private static VorbisMappingConfig MakeMapping(int[] mag, int[] ang)
    {
        return new VorbisMappingConfig
        {
            Submaps = 1,
            CouplingMagnitudeChannels = mag,
            CouplingAngleChannels = ang,
            Mux = new[] { 0, 0 },
            SubmapFloor = new[] { 0 },
            SubmapResidue = new[] { 0 },
        };
    }

    [TestMethod]
    public void VorbisInverseCoupling_PositiveMag_PositiveAng_YieldsMsMinus()
    {
        // mag=5, ang=3 -> newM = 5, newA = 5 - 3 = 2.
        var magBuf = new float[] { 5f };
        var angBuf = new float[] { 3f };
        var spectra = new[] { magBuf, angBuf };
        VorbisInverseCoupling.Apply(spectra, MakeMapping(new[] { 0 }, new[] { 1 }));
        Equal(5f, magBuf[0]);
        Equal(2f, angBuf[0]);
    }

    [TestMethod]
    public void VorbisInverseCoupling_PositiveMag_NegativeAng_YieldsMsPlus()
    {
        // mag=5, ang=-3 -> newA = 5, newM = 5 + (-3) = 2.
        var magBuf = new float[] { 5f };
        var angBuf = new float[] { -3f };
        var spectra = new[] { magBuf, angBuf };
        VorbisInverseCoupling.Apply(spectra, MakeMapping(new[] { 0 }, new[] { 1 }));
        Equal(2f, magBuf[0]);
        Equal(5f, angBuf[0]);
    }

    [TestMethod]
    public void VorbisInverseCoupling_NegativeMag_PositiveAng()
    {
        // mag=-5, ang=3 -> newM = -5, newA = -5 + 3 = -2.
        var magBuf = new float[] { -5f };
        var angBuf = new float[] { 3f };
        var spectra = new[] { magBuf, angBuf };
        VorbisInverseCoupling.Apply(spectra, MakeMapping(new[] { 0 }, new[] { 1 }));
        Equal(-5f, magBuf[0]);
        Equal(-2f, angBuf[0]);
    }

    [TestMethod]
    public void VorbisInverseCoupling_NegativeMag_NegativeAng()
    {
        // mag=-5, ang=-3 -> newA = -5, newM = -5 - (-3) = -2.
        var magBuf = new float[] { -5f };
        var angBuf = new float[] { -3f };
        var spectra = new[] { magBuf, angBuf };
        VorbisInverseCoupling.Apply(spectra, MakeMapping(new[] { 0 }, new[] { 1 }));
        Equal(-2f, magBuf[0]);
        Equal(-5f, angBuf[0]);
    }

    [TestMethod]
    public void VorbisInverseCoupling_NoCouplingSteps_NoChange()
    {
        var ch0 = new float[] { 1.1f, 2.2f, 3.3f };
        var ch1 = new float[] { -1.1f, -2.2f, -3.3f };
        var mapping = MakeMapping(Array.Empty<int>(), Array.Empty<int>());
        VorbisInverseCoupling.Apply(new[] { ch0, ch1 }, mapping);
        Equal(1.1f, ch0[0]);
        Equal(-3.3f, ch1[2]);
    }

    [TestMethod]
    public void VorbisInverseCoupling_MultipleBins_AppliedElementwise()
    {
        // Each bin handled independently.
        var mag = new[] { 5f, -5f, 5f, -5f };
        var ang = new[] { 3f, 3f, -3f, -3f };
        var spectra = new[] { mag, ang };
        VorbisInverseCoupling.Apply(spectra, MakeMapping(new[] { 0 }, new[] { 1 }));
        EqualFloats(new[] { 5f, -5f, 2f, -2f }, mag);
        EqualFloats(new[] { 2f, -2f, 5f, -5f }, ang);
    }

    [TestMethod]
    public void VorbisInverseCoupling_MultipleSteps_AppliedInReverse()
    {
        // 2 coupling steps. Steps apply in LAST -> FIRST order.
        var ch0 = new float[] { 10f };
        var ch1 = new float[] { 2f };
        var ch2 = new float[] { 4f };
        var mapping = new VorbisMappingConfig
        {
            Submaps = 1,
            CouplingMagnitudeChannels = new[] { 0, 1 },
            CouplingAngleChannels = new[] { 1, 2 },
            Mux = new[] { 0, 0, 0 },
            SubmapFloor = new[] { 0 },
            SubmapResidue = new[] { 0 },
        };
        // Step 1 applied first (reverse): mag=ch1=2, ang=ch2=4
        //   mag>0, ang>0 -> newM=2, newA=2-4=-2 -> ch1=2, ch2=-2.
        // Step 0 applied second: mag=ch0=10, ang=ch1=2 (post-step1 value)
        //   mag>0, ang>0 -> newM=10, newA=10-2=8 -> ch0=10, ch1=8.
        VorbisInverseCoupling.Apply(new[] { ch0, ch1, ch2 }, mapping);
        Equal(10f, ch0[0]);
        Equal(8f, ch1[0]);
        Equal(-2f, ch2[0]);
    }

    private static void EqualFloats(float[] expected, float[] actual)
    {
        if (expected.Length != actual.Length)
            throw new Exception($"Length mismatch: expected {expected.Length}, got {actual.Length}");
        for (int i = 0; i < expected.Length; i++)
        {
            if (Math.Abs(expected[i] - actual[i]) > 1e-5f)
                throw new Exception($"At {i}: expected {expected[i]}, got {actual[i]}");
        }
    }
}
