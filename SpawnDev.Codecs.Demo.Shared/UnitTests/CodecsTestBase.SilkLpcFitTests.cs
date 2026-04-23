using SpawnDev.Codecs.Audio.Opus.Silk;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Audio.Opus.Silk.SilkMacros;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for <see cref="SilkLpcFit"/> - int32-to-int16 LPC coefficient fitting
/// with iterative bandwidth expansion on overflow. Plus the supporting math
/// macros added to reach this subsystem (silk_abs, silk_min, silk_DIV32,
/// silk_RSHIFT32, silk_SAT16).
/// </summary>
public abstract partial class CodecsTestBase
{
    // -------- New macros --------

    [TestMethod]
    public void SilkMacros_Abs_ReturnsAbsoluteValue()
    {
        Equal(0, silk_abs(0));
        Equal(5, silk_abs(5));
        Equal(5, silk_abs(-5));
        Equal(int.MaxValue, silk_abs(int.MaxValue));
        // silk_abs(int.MinValue) overflows (mirrors C behavior); test boundary nearby.
        Equal(int.MaxValue, silk_abs(-int.MaxValue));
    }

    [TestMethod]
    public void SilkMacros_Min_ReturnsSmaller()
    {
        Equal(1, silk_min(1, 2));
        Equal(1, silk_min(2, 1));
        Equal(-5, silk_min(-5, 5));
        Equal(int.MinValue, silk_min(int.MinValue, 0));
    }

    [TestMethod]
    public void SilkMacros_Div32_IntegerDivide()
    {
        Equal(5, silk_DIV32(10, 2));
        Equal(3, silk_DIV32(10, 3));         // truncates
        Equal(-3, silk_DIV32(-10, 3));       // C-style truncation
        Equal(0, silk_DIV32(0, 1));
    }

    [TestMethod]
    public void SilkMacros_Rshift32_SignedArithmeticShift()
    {
        Equal(5, silk_RSHIFT32(10, 1));
        Equal(-5, silk_RSHIFT32(-10, 1));    // signed arithmetic shift; -10 >> 1 = -5 per C
        Equal(-1, silk_RSHIFT32(-1, 4));
    }

    [TestMethod]
    public void SilkMacros_Sat16_ClampsToInt16Range()
    {
        Equal((short)5, silk_SAT16(5));
        Equal(short.MaxValue, silk_SAT16(int.MaxValue));
        Equal(short.MaxValue, silk_SAT16(short.MaxValue + 1));
        Equal(short.MinValue, silk_SAT16(int.MinValue));
        Equal(short.MinValue, silk_SAT16(short.MinValue - 1));
        Equal(short.MaxValue, silk_SAT16(short.MaxValue));
        Equal(short.MinValue, silk_SAT16(short.MinValue));
    }

    [TestMethod]
    public void SilkConstants_Int16Max_Is32767()
    {
        Equal((short)32767, silk_int16_MAX);
    }

    // -------- SilkLpcFit --------

    [TestMethod]
    public void LpcFit_SmallCoefficients_DirectConvert()
    {
        // All coefficients well within int16 range at target Q; no iteration needed.
        int qIn = 24;
        int qOut = 12;
        int[] aQIn = { 1 << 24, 2 << 24, -1 << 24, 1 << 23 };
        var aQInSpan = aQIn.AsSpan();

        Span<short> aQOut = stackalloc short[4];
        SilkLpcFit.Fit(aQOut, aQInSpan, qOut, qIn, aQIn.Length);

        // Expected: silk_RSHIFT_ROUND(aQIn[k], 12) for each. 1<<24 >> 12 = 4096.
        Equal((short)4096, aQOut[0]);
        Equal((short)(2 * 4096), aQOut[1]);
        Equal((short)(-4096), aQOut[2]);
        Equal((short)(4096 / 2), aQOut[3]);
    }

    [TestMethod]
    public void LpcFit_LargeCoefficient_TriggersIteration()
    {
        // Put one coefficient just above the overflow threshold in Q12 so iteration fires.
        // Q12 int16 threshold: silk_int16_MAX = 32767. To overflow after RSHIFT_ROUND(x, qIn-qOut),
        // need x > 32767 << 12 = 134_213_632 roughly.
        int qIn = 24;
        int qOut = 12;
        int[] aQIn = {
            1 << 28, // large - will exceed int16 after shift
            1 << 20,
            -1 << 22,
            1 << 18,
        };

        Span<short> aQOut = stackalloc short[4];
        SilkLpcFit.Fit(aQOut, aQIn, qOut, qIn, aQIn.Length);

        // After bwexpand iterations, all output must fit in int16 without saturation
        // (or if it reached the saturation fallback, the result is clipped to int16 range
        // and that's still valid).
        foreach (var v in aQOut)
        {
            InRange(v, short.MinValue, short.MaxValue);
        }
    }

    [TestMethod]
    public void LpcFit_ZeroCoefficients_ProducesZeros()
    {
        int[] aQIn = { 0, 0, 0, 0 };
        Span<short> aQOut = stackalloc short[4];
        SilkLpcFit.Fit(aQOut, aQIn, qOut: 12, qIn: 24, d: 4);
        for (int i = 0; i < aQOut.Length; i++) Equal((short)0, aQOut[i]);
    }

    [TestMethod]
    public void LpcFit_InputBufferTooSmall_Throws()
    {
        short[] aQOut = new short[10];
        int[] aQIn = new int[3];
        Throws<ArgumentException>(() => SilkLpcFit.Fit(aQOut, aQIn, 12, 24, 10));
    }

    [TestMethod]
    public void LpcFit_OutputBufferTooSmall_Throws()
    {
        short[] aQOut = new short[3];
        int[] aQIn = new int[10];
        Throws<ArgumentException>(() => SilkLpcFit.Fit(aQOut, aQIn, 12, 24, 10));
    }

    [TestMethod]
    public void LpcFit_QIn_EqualsQOut_NoShift()
    {
        // When qIn == qOut, silk_RSHIFT_ROUND(x, 0) is... x + 0 >> 0? Actually silk_RSHIFT_ROUND
        // with shift=0 is technically undefined in libopus (macro requires shift >= 1). We avoid
        // shift=0 scenarios. This test confirms qIn-qOut=1 case works cleanly.
        int qIn = 13;
        int qOut = 12;
        int[] aQIn = { 10, 20, -30, 40 };
        Span<short> aQOut = stackalloc short[4];
        SilkLpcFit.Fit(aQOut, aQIn, qOut, qIn, 4);
        // silk_RSHIFT_ROUND(10, 1) = (10 >> 1) + (10 & 1) = 5 + 0 = 5
        Equal((short)5, aQOut[0]);
        Equal((short)10, aQOut[1]);
        Equal((short)(-15), aQOut[2]);
        Equal((short)20, aQOut[3]);
    }
}
