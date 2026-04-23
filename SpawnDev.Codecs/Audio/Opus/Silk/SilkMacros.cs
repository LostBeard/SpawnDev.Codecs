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

    /// <summary>Minimum value representable as <c>opus_int32</c>.</summary>
    internal const int silk_int32_MIN = int.MinValue;

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

    /// <summary>Count leading zeros of a 32-bit value. Returns 32 for input 0 (matches libopus <c>silk_CLZ32</c>).</summary>
    internal static int silk_CLZ32(int x) => BitOperations.LeadingZeroCount((uint)x);

    /// <summary>Maximum int16 value, matching libopus <c>silk_int16_MAX</c>.</summary>
    internal const short silk_int16_MAX = short.MaxValue;

    /// <summary>Absolute value of a 32-bit integer.</summary>
    internal static int silk_abs(int x) => x < 0 ? -x : x;

    /// <summary>Minimum of two int32 (libopus <c>silk_min</c> variant; shorthand for <see cref="silk_min_int"/>).</summary>
    internal static int silk_min(int a, int b) => a < b ? a : b;

    /// <summary>Signed 32-bit division.</summary>
    internal static int silk_DIV32(int a, int b) => a / b;

    /// <summary>32-bit right shift alias (libopus <c>silk_RSHIFT32</c>).</summary>
    internal static int silk_RSHIFT32(int a, int shift) => a >> shift;

    /// <summary>Saturate a 32-bit signed value to int16 range.</summary>
    internal static short silk_SAT16(int x) =>
        x > short.MaxValue ? short.MaxValue : (x < short.MinValue ? short.MinValue : (short)x);

    /// <summary>Count leading zeros of a 32-bit value (same as <see cref="silk_CLZ32"/>; some libopus code uses 32-bit spelling).</summary>
    internal static int silk_max_32(int a, int b) => a > b ? a : b;

    /// <summary>
    /// Overflow-wrapping 32-bit add: performs <c>a + b</c> in unsigned arithmetic and
    /// reinterprets the bit pattern as signed. Matches libopus <c>silk_ADD32_ovflw</c>.
    /// </summary>
    internal static int silk_ADD32_ovflw(int a, int b) => (int)((uint)a + (uint)b);

    /// <summary>
    /// Overflow-wrapping 32-bit subtract. Matches libopus <c>silk_SUB32_ovflw</c>.
    /// </summary>
    internal static int silk_SUB32_ovflw(int a, int b) => (int)((uint)a - (uint)b);

    /// <summary>Plain 32-bit add alias matching libopus <c>silk_ADD32</c>.</summary>
    internal static int silk_ADD32(int a, int b) => a + b;

    /// <summary>
    /// Overflow-wrapping signed multiply-accumulate byte-by-byte: <c>a + (short)b * (short)c</c>
    /// with overflow wrapping. Matches libopus <c>silk_SMLABB_ovflw</c>.
    /// </summary>
    internal static int silk_SMLABB_ovflw(int a, int b, int c) => silk_ADD32_ovflw(a, silk_SMULBB(b, c));

    /// <summary>
    /// Unsigned add-then-right-shift: <c>a + (b &gt;&gt; shift)</c> treating both operands as <see cref="uint"/>.
    /// Matches libopus <c>silk_ADD_RSHIFT_uint</c>.
    /// </summary>
    internal static uint silk_ADD_RSHIFT_uint(uint a, uint b, int shift) => a + (b >> shift);

    /// <summary>
    /// Signed multiply word-by-word high. Libopus macro form:
    /// <c>silk_MLA(silk_SMULWB(a, b), a, silk_RSHIFT_ROUND(b, 16))</c>.
    /// Used by bwexpander_32 and a few other SILK utilities.
    /// </summary>
    internal static int silk_SMULWW(int a32, int b32) =>
        silk_MLA(silk_SMULWB(a32, b32), a32, silk_RSHIFT_ROUND(b32, 16));

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

    // ----- 64-bit and saturating helpers used by LPC stability / NLSF-to-LPC math -----

    /// <summary>32-bit subtraction alias (libopus <c>silk_SUB32</c>).</summary>
    internal static int silk_SUB32(int a, int b) => a - b;

    /// <summary>Left-shift alias matching libopus <c>silk_LSHIFT32</c>.</summary>
    internal static int silk_LSHIFT32(int a, int shift) => a << shift;

    /// <summary>
    /// 32 x 32 -&gt; 64 signed multiply (libopus <c>silk_SMULL</c>).
    /// </summary>
    internal static long silk_SMULL(int a32, int b32) => (long)a32 * b32;

    /// <summary>
    /// Signed multiply, high 32 bits of the 64-bit product.
    /// Matches libopus <c>silk_SMMUL(a32, b32) = (int)(silk_SMULL(a32, b32) &gt;&gt; 32)</c>.
    /// </summary>
    internal static int silk_SMMUL(int a32, int b32) => (int)((long)a32 * b32 >> 32);

    /// <summary>
    /// 64-bit rounded right shift (libopus <c>silk_RSHIFT_ROUND64</c>).
    ///   shift == 1: <c>(a &gt;&gt; 1) + (a &amp; 1)</c>
    ///   shift &gt; 1: <c>((a &gt;&gt; (shift - 1)) + 1) &gt;&gt; 1</c>
    /// </summary>
    internal static long silk_RSHIFT_ROUND64(long a, int shift) =>
        shift == 1
            ? (a >> 1) + (a & 1)
            : ((a >> (shift - 1)) + 1) >> 1;

    /// <summary>
    /// Saturating 32-bit subtraction: returns <paramref name="a"/> - <paramref name="b"/> clamped to int32 range.
    /// Matches libopus <c>silk_SUB_SAT32</c>.
    /// </summary>
    internal static int silk_SUB_SAT32(int a, int b)
    {
        long diff = (long)a - b;
        if (diff > int.MaxValue) return int.MaxValue;
        if (diff < int.MinValue) return int.MinValue;
        return (int)diff;
    }

    /// <summary>
    /// Saturating 32-bit left shift: clamps overflow to <c>int.MaxValue</c> / <c>int.MinValue</c>.
    /// Matches libopus <c>silk_LSHIFT_SAT32</c>. Behavior for negative shift is not defined
    /// (libopus callers only invoke with non-negative shift).
    /// </summary>
    internal static int silk_LSHIFT_SAT32(int a, int shift)
    {
        if (shift == 0) return a;
        if (shift >= 32)
        {
            if (a > 0) return int.MaxValue;
            if (a < 0) return int.MinValue;
            return 0;
        }
        long val = (long)a << shift;
        if (val > int.MaxValue) return int.MaxValue;
        if (val < int.MinValue) return int.MinValue;
        return (int)val;
    }

    /// <summary>
    /// Signed multiply-accumulate using "word x word" high-word product.
    /// Matches libopus <c>silk_SMLAWW(a, b, c) = a + silk_SMULWW(b, c)</c>.
    /// </summary>
    internal static int silk_SMLAWW(int a32, int b32, int c32) => a32 + silk_SMULWW(b32, c32);

    /// <summary>
    /// 32/16 signed division alias. Libopus uses <c>silk_DIV32_16</c> where <c>b</c> is expected
    /// to fit in 16 bits; semantically identical to plain integer division for 32-bit inputs.
    /// </summary>
    internal static int silk_DIV32_16(int a32, int b32) => a32 / b32;

    /// <summary>
    /// Computes an approximation to <c>(1 &lt;&lt; Qres) / b32</c> as a signed 32-bit value.
    /// Matches libopus <c>silk_INVERSE32_varQ</c> in silk/Inlines.h. Uses two Newton-like
    /// refinement steps (initial 14-bit-precision divide, then a correction via
    /// <c>SMLAWW</c>). Requires <paramref name="b32"/> != 0 and <paramref name="Qres"/> &gt; 0.
    /// </summary>
    internal static int silk_INVERSE32_varQ(int b32, int Qres)
    {
        int b_headrm = silk_CLZ32(silk_abs(b32)) - 1;
        int b32_nrm = silk_LSHIFT(b32, b_headrm);

        int b32_inv = silk_DIV32_16(silk_int32_MAX >> 2, silk_RSHIFT(b32_nrm, 16));

        int result = silk_LSHIFT(b32_inv, 16);

        int err_Q32 = silk_LSHIFT((1 << 29) - silk_SMULWB(b32_nrm, b32_inv), 3);

        result = silk_SMLAWW(result, err_Q32, b32_inv);

        int lshift = 61 - b_headrm - Qres;
        if (lshift <= 0)
        {
            return silk_LSHIFT_SAT32(result, -lshift);
        }
        else
        {
            if (lshift < 32)
            {
                return silk_RSHIFT(result, lshift);
            }
            else
            {
                return 0;
            }
        }
    }

    /// <summary>
    /// 32-bit fractional multiply in variable Q: <c>(int)((a * b) rounded-right-shift Q)</c>
    /// using 64-bit intermediate. Matches libopus <c>MUL32_FRAC_Q</c> in silk/LPC_inv_pred_gain.c.
    /// </summary>
    internal static int silk_MUL32_FRAC_Q(int a32, int b32, int Q) =>
        (int)silk_RSHIFT_ROUND64(silk_SMULL(a32, b32), Q);

    // ----- Decode-core support helpers -----

    /// <summary>
    /// Overflow-wrapping multiply-accumulate: <c>a + b * c</c> with unsigned multiplication
    /// so overflow wraps bit-exactly. Matches libopus <c>silk_MLA_ovflw</c>.
    /// </summary>
    internal static int silk_MLA_ovflw(int a, int b, int c) =>
        silk_ADD32_ovflw(a, (int)((uint)b * (uint)c));

    /// <summary>
    /// SILK pseudo-random generator step. Matches libopus
    /// <c>silk_RAND(seed) = silk_MLA_ovflw(RAND_INCREMENT, seed, RAND_MULTIPLIER)</c>.
    /// </summary>
    internal static int silk_RAND(int seed) =>
        silk_MLA_ovflw(SilkConstants.RAND_INCREMENT, seed, SilkConstants.RAND_MULTIPLIER);

    /// <summary>
    /// Saturating 32-bit add: clamps <c>a + b</c> to <c>[int.MinValue, int.MaxValue]</c>.
    /// Matches libopus <c>silk_ADD_SAT32</c>.
    /// </summary>
    internal static int silk_ADD_SAT32(int a, int b)
    {
        long sum = (long)a + b;
        if (sum > int.MaxValue) return int.MaxValue;
        if (sum < int.MinValue) return int.MinValue;
        return (int)sum;
    }

    /// <summary>Subtract-then-left-shift: <c>a - (b &lt;&lt; shift)</c>. Matches libopus <c>silk_SUB_LSHIFT32</c>.</summary>
    internal static int silk_SUB_LSHIFT32(int a, int b, int shift) => a - (b << shift);

    /// <summary>Overflow-wrapping left shift (uses unsigned arithmetic). Matches libopus <c>silk_LSHIFT_ovflw</c>.</summary>
    internal static int silk_LSHIFT_ovflw(int a, int shift) => (int)((uint)a << shift);

    /// <summary>
    /// Variable-Q 32-bit division: returns a Q<paramref name="Qres"/> approximation to
    /// <paramref name="a32"/> / <paramref name="b32"/>. Matches libopus
    /// <c>silk_DIV32_varQ</c> (Inlines.h). Requires <paramref name="b32"/> != 0 and
    /// <paramref name="Qres"/> &gt;= 0.
    /// </summary>
    internal static int silk_DIV32_varQ(int a32, int b32, int Qres)
    {
        int a_headrm = silk_CLZ32(silk_abs(a32)) - 1;
        int a32_nrm = silk_LSHIFT(a32, a_headrm);

        int b_headrm = silk_CLZ32(silk_abs(b32)) - 1;
        int b32_nrm = silk_LSHIFT(b32, b_headrm);

        int b32_inv = silk_DIV32_16(silk_int32_MAX >> 2, silk_RSHIFT(b32_nrm, 16));

        int result = silk_SMULWB(a32_nrm, b32_inv);

        // Residual: a_nrm - (b_nrm * result) scaled back to Q(a_headrm). Overflow is fine -
        // libopus guarantees the final value stays small. NOTE: earlier libopus versions
        // used silk_SUB_LSHIFT32 here with an extra <<3 shift; the current (xiph/opus main)
        // uses silk_SUB32_ovflw with no second shift - we follow current.
        a32_nrm = silk_SUB32_ovflw(a32_nrm, silk_LSHIFT_ovflw(silk_SMMUL(b32_nrm, result), 3));

        result = silk_SMLAWB(result, a32_nrm, b32_inv);

        int lshift = 29 + a_headrm - b_headrm - Qres;
        if (lshift < 0) return silk_LSHIFT_SAT32(result, -lshift);
        if (lshift < 32) return silk_RSHIFT(result, lshift);
        return 0;
    }
}
