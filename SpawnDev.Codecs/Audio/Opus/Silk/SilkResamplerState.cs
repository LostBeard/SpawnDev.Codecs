// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// C# equivalent of libopus silk_resampler_state_struct. Persists the
// resampler's IIR / FIR history + delay buffer + active configuration
// between successive Apply() calls in a streaming resampler.
//
// Upstream Copyright (c) 2006-2011 Skype Limited. BSD 3-Clause. See NOTICE.md.

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// Resampler state: IIR + FIR history buffers, delay line, configuration fields.
/// Populated by <see cref="SilkResampler.Init"/>; updated in place by
/// <see cref="SilkResampler.Apply"/>.
/// </summary>
internal sealed class SilkResamplerState
{
    /// <summary>IIR history buffer. Only the first <see cref="FirOrder"/> entries are live.</summary>
    public readonly int[] SIir = new int[SilkResamplerConstants.MAX_IIR_ORDER];

    /// <summary>FIR history buffer (int32 variant for upsample paths).</summary>
    public readonly int[] SFirI32 = new int[SilkResamplerConstants.MAX_FIR_ORDER];

    /// <summary>FIR history buffer (int16 variant for downsample paths).</summary>
    public readonly short[] SFirI16 = new short[SilkResamplerConstants.MAX_FIR_ORDER];

    /// <summary>Delay-compensation buffer used to equalize total delay across modes.</summary>
    public readonly short[] DelayBuf = new short[SilkResamplerConstants.DELAY_BUF_SIZE];

    /// <summary>Which dispatch path to use: one of <see cref="SilkResamplerConstants"/>' <c>USE_*</c> constants.</summary>
    public int ResamplerFunction;

    /// <summary>Per-call batch size (<see cref="FsInKHz"/> * <see cref="SilkResamplerConstants.MAX_BATCH_SIZE_MS"/>).</summary>
    public int BatchSize;

    /// <summary>Inverse sample-rate ratio in Q16 (input/output in Q16 for interpolation).</summary>
    public int InvRatioQ16;

    /// <summary>FIR filter order (for downsample variants).</summary>
    public int FirOrder;

    /// <summary>Number of polyphase fractions (for downsample variants).</summary>
    public int FirFracs;

    /// <summary>Input sample rate in kHz.</summary>
    public int FsInKHz;

    /// <summary>Output sample rate in kHz.</summary>
    public int FsOutKHz;

    /// <summary>Initial delay offset written into the delay buffer on each call.</summary>
    public int InputDelay;

    /// <summary>Filter coefficient table for the selected downsample variant (null for other paths).</summary>
    public short[]? Coefs;

    /// <summary>Reset all buffers + scalars to zero. Called at the start of <see cref="SilkResampler.Init"/>.</summary>
    public void Clear()
    {
        Array.Clear(SIir, 0, SIir.Length);
        Array.Clear(SFirI32, 0, SFirI32.Length);
        Array.Clear(SFirI16, 0, SFirI16.Length);
        Array.Clear(DelayBuf, 0, DelayBuf.Length);
        ResamplerFunction = 0;
        BatchSize = 0;
        InvRatioQ16 = 0;
        FirOrder = 0;
        FirFracs = 0;
        FsInKHz = 0;
        FsOutKHz = 0;
        InputDelay = 0;
        Coefs = null;
    }
}
