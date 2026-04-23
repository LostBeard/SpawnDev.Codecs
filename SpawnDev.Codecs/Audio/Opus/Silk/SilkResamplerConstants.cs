// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Structural port of libopus silk/resampler_structs.h + silk/resampler_private.h +
// silk/resampler_rom.h selected constants to clean C#. Used by the SilkResampler
// family.
//
// Upstream Copyright (c) 2006-2011 Skype Limited. BSD 3-Clause. See NOTICE.md.

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>SILK resampler-specific constants, ported from libopus resampler headers.</summary>
internal static class SilkResamplerConstants
{
    /// <summary>Maximum IIR order used by any resampler variant. Libopus <c>SILK_RESAMPLER_MAX_IIR_ORDER = 6</c>.</summary>
    internal const int MAX_IIR_ORDER = 6;

    /// <summary>Maximum FIR order used by any resampler variant. Libopus <c>SILK_RESAMPLER_MAX_FIR_ORDER = 36</c>.</summary>
    internal const int MAX_FIR_ORDER = 36;

    /// <summary>Size of the delay compensation buffer. Libopus <c>delayBuf[96]</c>.</summary>
    internal const int DELAY_BUF_SIZE = 96;

    /// <summary>Resampler batch size in milliseconds. Libopus <c>RESAMPLER_MAX_BATCH_SIZE_MS = 10</c>.</summary>
    internal const int MAX_BATCH_SIZE_MS = 10;

    /// <summary>Maximum sample rate in kHz supported by the resampler. Libopus <c>RESAMPLER_MAX_FS_KHZ = 48</c>.</summary>
    internal const int MAX_FS_KHZ = 48;

    /// <summary>Maximum batch size in samples. <c>MAX_BATCH_SIZE_MS * MAX_FS_KHZ = 480</c>.</summary>
    internal const int MAX_BATCH_SIZE_IN = MAX_BATCH_SIZE_MS * MAX_FS_KHZ;

    /// <summary>FIR order for 3/4 and 2/3 down-sample filters. Libopus <c>RESAMPLER_DOWN_ORDER_FIR0 = 18</c>.</summary>
    internal const int DOWN_ORDER_FIR0 = 18;

    /// <summary>FIR order for 1/2 down-sample filter. Libopus <c>RESAMPLER_DOWN_ORDER_FIR1 = 24</c>.</summary>
    internal const int DOWN_ORDER_FIR1 = 24;

    /// <summary>FIR order for 1/3, 1/4, 1/6 down-sample filters. Libopus <c>RESAMPLER_DOWN_ORDER_FIR2 = 36</c>.</summary>
    internal const int DOWN_ORDER_FIR2 = 36;

    /// <summary>FIR order for 12-tap polyphase filter. Libopus <c>RESAMPLER_ORDER_FIR_12 = 8</c>.</summary>
    internal const int ORDER_FIR_12 = 8;

    // ---- Dispatch IDs for resampler_function field ----

    /// <summary>Identity (input rate == output rate); delay-buffer copy only.</summary>
    internal const int USE_COPY = 0;

    /// <summary>Exact 2x upsample path. Libopus <c>USE_silk_resampler_private_up2_HQ_wrapper</c>.</summary>
    internal const int USE_UP2_HQ_WRAPPER = 1;

    /// <summary>Arbitrary upsample via IIR + FIR polyphase.</summary>
    internal const int USE_IIR_FIR = 2;

    /// <summary>Arbitrary downsample via FIR.</summary>
    internal const int USE_DOWN_FIR = 3;
}
