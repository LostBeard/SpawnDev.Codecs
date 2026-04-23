// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Structural port of libopus silk/MacroDebug.h + silk/SigProc_FIX.h +
// silk/macros.h fixed-point helper macros into idiomatic C# static methods.
//
// Upstream Copyright (c) 2006-2011 Skype Limited. BSD 3-Clause.
// See NOTICE.md.

using System.Numerics;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// SILK fixed-point math helper functions. A small set of integer math primitives
/// used throughout the SILK decoder. libopus implements them as C macros so they
/// inline everywhere; here they are small static methods with the same semantics.
///
/// All functions follow libopus naming (snake_case with silk_ prefix) so a side-by-
/// side comparison with the C source is unambiguous.
/// </summary>
internal static class SilkMacros
{
    /// <summary>Maximum value representable as <c>opus_int32</c>.</summary>
    internal const int silk_int32_MAX = int.MaxValue;

    /// <summary>Signed multiply, low 16 bits of <paramref name="b32"/>, high-word of product.</summary>
    /// <returns><c>(a32 * (short)b32) &gt;&gt; 16</c></returns>
    internal static int silk_SMULWB(int a32, int b32) => (int)((long)a32 * (short)b32 >> 16);

    /// <summary>Signed multiply-accumulate: <paramref name="a32"/> plus the high-word of <paramref name="b32"/> times low-16 of <paramref name="c32"/>.</summary>
    /// <returns><c>a32 + ((b32 * (short)c32) &gt;&gt; 16)</c></returns>
    internal static int silk_SMLAWB(int a32, int b32, int c32) => a32 + (int)((long)b32 * (short)c32 >> 16);

    /// <summary>Signed multiply: low 16 bits of each operand.</summary>
    /// <returns><c>(short)a32 * (short)b32</c></returns>
    internal static int silk_SMULBB(int a32, int b32) => (short)a32 * (short)b32;

    /// <summary>Multiply-add: <c>a32 + b32 * c32</c>.</summary>
    internal static int silk_MLA(int a32, int b32, int c32) => a32 + b32 * c32;

    /// <summary>Signed multiply (32 bits x 32 bits -> 32 bits).</summary>
    internal static int silk_MUL(int a32, int b32) => a32 * b32;

    /// <summary>Left shift.</summary>
    internal static int silk_LSHIFT(int a, int shift) => a << shift;

    /// <summary>Arithmetic right shift for signed integers.</summary>
    internal static int silk_RSHIFT(int a, int shift) => a >> shift;

    /// <summary>Add then right-shift: <c>a + (b &gt;&gt; shift)</c>.</summary>
    internal static int silk_ADD_RSHIFT32(int a, int b, int shift) => a + (b >> shift);

    /// <summary>Add then left-shift: <c>a + (b &lt;&lt; shift)</c>.</summary>
    internal static int silk_ADD_LSHIFT32(int a, int b, int shift) => a + (b << shift);

    /// <summary>Clamp <paramref name="x"/> into the inclusive interval <c>[lo, hi]</c>.</summary>
    internal static int silk_LIMIT_int(int x, int lo, int hi) => x < lo ? lo : (x > hi ? hi : x);

    /// <summary>Minimum of two 32-bit integers.</summary>
    internal static int silk_min_int(int a, int b) => a < b ? a : b;

    /// <summary>Maximum of two 32-bit integers.</summary>
    internal static int silk_max_int(int a, int b) => a > b ? a : b;

    /// <summary>Minimum of two 32-bit integers (same as <see cref="silk_min_int"/>; libopus uses both spellings).</summary>
    internal static int silk_min_32(int a, int b) => a < b ? a : b;

    /// <summary>
    /// Rounded right-shift. Libopus defines this as:
    ///   shift == 1: <c>(a &gt;&gt; 1) + (a &amp; 1)</c>
    ///   shift &gt; 1: <c>((a &gt;&gt; (shift - 1)) + 1) &gt;&gt; 1</c>
    /// Rounds half away from zero for positive input.
    /// </summary>
    internal static int silk_RSHIFT_ROUND(int a, int shift) =>
        shift == 1
            ? (a >> 1) + (a & 1)
            : ((a >> (shift - 1)) + 1) >> 1;

    /// <summary>Clamp 32-bit value to inclusive <c>[lo, hi]</c>.</summary>
    internal static int silk_LIMIT_32(int x, int lo, int hi) => x < lo ? lo : (x > hi ? hi : x);

    /// <summary>Saturating 16-bit add: clamps result to <c>[short.MinValue, short.MaxValue]</c>.</summary>
    internal static short silk_ADD_SAT16(short a, short b)
    {
        int sum = a + b;
        if (sum > short.MaxValue) return short.MaxValue;
        if (sum < short.MinValue) return short.MinValue;
        return (short)sum;
    }

    /// <summary>
    /// Insertion sort (ascending) for a span of <see cref="short"/> values.
    /// Libopus <c>silk_insertion_sort_increasing_all_values_int16</c>. Best case O(n),
    /// worst case O(n^2), but the SILK caller typically hands in already-almost-sorted data.
    /// </summary>
    internal static void silk_insertion_sort_increasing_all_values_int16(Span<short> a)
    {
        for (int i = 1; i < a.Length; i++)
        {
            short value = a[i];
            int j = i - 1;
            while (j >= 0 && value < a[j])
            {
                a[j + 1] = a[j];
                j--;
            }
            a[j + 1] = value;
        }
    }

    /// <summary>
    /// Decomposes a positive 32-bit integer into (leading zeros, fractional bits in Q7).
    /// <para>
    /// For input 0 the result is (32, 0) per libopus <c>silk_CLZ_FRAC</c> behavior
    /// (BitOperations.LeadingZeroCount(0) == 32).
    /// </para>
    /// </summary>
    /// <param name="inVal">Input value (typically positive).</param>
    /// <param name="lz">Output: leading zero count.</param>
    /// <param name="fracQ7">Output: fractional part in Q7.</param>
    internal static void silk_CLZ_FRAC(int inVal, out int lz, out int fracQ7)
    {
        lz = BitOperations.LeadingZeroCount((uint)inVal);
        // Shift inVal left by lz so the MSB is bit 31, then take bits 30..24 (the 7 bits just below the MSB).
        fracQ7 = (int)(((uint)inVal << lz >> 24) & 0x7F);
    }
}
