// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Structural port of libopus silk/stereo_MS_to_LR.c. Converts mid+side PCM
// samples produced by two independent SILK channel decodes back into left/right
// output, applying per-frame predictor interpolation across the first
// STEREO_INTERP_LEN_MS milliseconds to avoid audible predictor discontinuities.
//
// Upstream Copyright (c) 2006-2011 Skype Limited. BSD 3-Clause. See NOTICE.md.

using static SpawnDev.Codecs.Audio.Opus.Silk.SilkMacros;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// Persistent state for stereo mid/side -&gt; L/R conversion. Holds the last
/// two samples of each channel (for filter history) and the previous frame's
/// predictor coefficients (for inter-frame interpolation).
/// </summary>
internal sealed class SilkStereoState
{
    /// <summary>Last 2 mid-channel samples from the previous frame.</summary>
    public readonly short[] SMid = new short[2];

    /// <summary>Last 2 side-channel samples from the previous frame.</summary>
    public readonly short[] SSide = new short[2];

    /// <summary>Previous frame's mid/side predictors in Q13.</summary>
    public readonly int[] PredPrevQ13 = new int[2];

    /// <summary>Reset state for a fresh stream.</summary>
    public void Reset()
    {
        SMid[0] = SMid[1] = 0;
        SSide[0] = SSide[1] = 0;
        PredPrevQ13[0] = PredPrevQ13[1] = 0;
    }
}

/// <summary>
/// Mid/side -&gt; left/right conversion for stereo SILK frames. Matches libopus
/// <c>silk_stereo_MS_to_LR</c> bit-exactly.
/// </summary>
internal static class SilkStereoMsToLr
{
    /// <summary>Duration in milliseconds of the predictor-interpolation region. Libopus <c>STEREO_INTERP_LEN_MS = 8</c>.</summary>
    internal const int StereoInterpLenMs = 8;

    /// <summary>
    /// Convert mid/side samples to left/right in place. After the call:
    /// <list type="bullet">
    /// <item><paramref name="x1"/> (mid input) -&gt; left output (samples <c>[2, frameLength+2)</c>).</item>
    /// <item><paramref name="x2"/> (side input) -&gt; right output (samples <c>[2, frameLength+2)</c>).</item>
    /// </list>
    /// Both buffers include 2 prefix samples of state at indices 0 and 1; the function
    /// reads and updates the state via <paramref name="state"/> and writes the output
    /// in-place at index <c>n + 1</c> for <c>n in [0, frameLength)</c>.
    /// </summary>
    /// <param name="state">Persistent stereo state.</param>
    /// <param name="x1">Mid-channel buffer, length &gt;= frameLength + 2. [0..1] seeded from previous frame; written with L output.</param>
    /// <param name="x2">Side-channel buffer, length &gt;= frameLength + 2. [0..1] seeded from previous frame; written with R output.</param>
    /// <param name="predQ13">Current frame's 2 Q13 mid/side predictors.</param>
    /// <param name="fsKHz">Internal SILK sample rate in kHz (8, 12, or 16).</param>
    /// <param name="frameLength">SILK frame length in samples.</param>
    internal static void Apply(
        SilkStereoState state,
        Span<short> x1,
        Span<short> x2,
        ReadOnlySpan<int> predQ13,
        int fsKHz,
        int frameLength)
    {
        if (state is null) throw new ArgumentNullException(nameof(state));
        if (x1.Length < frameLength + 2) throw new ArgumentException("x1 too short", nameof(x1));
        if (x2.Length < frameLength + 2) throw new ArgumentException("x2 too short", nameof(x2));
        if (predQ13.Length < 2) throw new ArgumentException("predQ13 needs 2 values", nameof(predQ13));

        // Prefix samples from persisted state.
        x1[0] = state.SMid[0];
        x1[1] = state.SMid[1];
        x2[0] = state.SSide[0];
        x2[1] = state.SSide[1];
        // Persist the trailing state for the next frame.
        state.SMid[0] = x1[frameLength];
        state.SMid[1] = x1[frameLength + 1];
        state.SSide[0] = x2[frameLength];
        state.SSide[1] = x2[frameLength + 1];

        // Predictor interpolation across the first STEREO_INTERP_LEN_MS * fs_kHz samples.
        int pred0Q13 = state.PredPrevQ13[0];
        int pred1Q13 = state.PredPrevQ13[1];
        int interpLen = StereoInterpLenMs * fsKHz;
        int denomQ16 = silk_DIV32_16(1 << 16, interpLen);
        int delta0Q13 = silk_RSHIFT_ROUND(silk_SMULBB(predQ13[0] - state.PredPrevQ13[0], denomQ16), 16);
        int delta1Q13 = silk_RSHIFT_ROUND(silk_SMULBB(predQ13[1] - state.PredPrevQ13[1], denomQ16), 16);

        for (int n = 0; n < interpLen; n++)
        {
            pred0Q13 += delta0Q13;
            pred1Q13 += delta1Q13;
            int sum = silk_LSHIFT(silk_ADD_LSHIFT32((int)x1[n] + (int)x1[n + 2], (int)x1[n + 1], 1), 9);
            sum = silk_SMLAWB(silk_LSHIFT((int)x2[n + 1], 8), sum, pred0Q13);
            sum = silk_SMLAWB(sum, silk_LSHIFT((int)x1[n + 1], 11), pred1Q13);
            x2[n + 1] = silk_SAT16(silk_RSHIFT_ROUND(sum, 8));
        }

        // Remaining samples use the final (current-frame) predictors without interpolation.
        pred0Q13 = predQ13[0];
        pred1Q13 = predQ13[1];
        for (int n = interpLen; n < frameLength; n++)
        {
            int sum = silk_LSHIFT(silk_ADD_LSHIFT32((int)x1[n] + (int)x1[n + 2], (int)x1[n + 1], 1), 9);
            sum = silk_SMLAWB(silk_LSHIFT((int)x2[n + 1], 8), sum, pred0Q13);
            sum = silk_SMLAWB(sum, silk_LSHIFT((int)x1[n + 1], 11), pred1Q13);
            x2[n + 1] = silk_SAT16(silk_RSHIFT_ROUND(sum, 8));
        }
        state.PredPrevQ13[0] = predQ13[0];
        state.PredPrevQ13[1] = predQ13[1];

        // Convert mid + side -> L + R via L = mid + side, R = mid - side (both saturating).
        for (int n = 0; n < frameLength; n++)
        {
            int sum = x1[n + 1] + (int)x2[n + 1];
            int diff = x1[n + 1] - (int)x2[n + 1];
            x1[n + 1] = silk_SAT16(sum);
            x2[n + 1] = silk_SAT16(diff);
        }
    }
}
