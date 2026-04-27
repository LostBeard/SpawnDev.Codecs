// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
// See NOTICE.md for upstream attributions.
//
// CELT decoder per-stream state. Mirrors libopus celt/celt_decoder.c
// `OpusCustomDecoder` and Concentus Celt/Structs/CELTDecoder.cs.
//
// This file currently scaffolds the future hand-port. The runtime CELT path
// in CeltDecoder.cs delegates to Concentus and does NOT consult this state
// type. When the per-module hand-port replaces the Concentus delegation,
// this state is what CeltDecoder will own and pass through every internal
// stage (decode_lost, comb_filter, deemphasis, etc.).
//
// Upstream references:
//   - libopus celt/celt_decoder.c (BSD-3, Xiph.Org)
//   - Concentus Celt/Structs/CELTDecoder.cs (BSD-3)

namespace SpawnDev.Codecs.Audio.Opus.Celt;

/// <summary>
/// Per-stream CELT decoder state. Persists across frames within a single
/// decoder instance. Cleared on construction and on
/// <see cref="CeltDecoder.ResetState"/>.
///
/// Currently a placeholder: the running CELT path in <see cref="CeltDecoder"/>
/// uses Concentus internally and therefore stores its state inside the
/// Concentus decoder instance. This type exists so the future hand-port has
/// a place to live without re-shaping the public API.
/// </summary>
internal sealed class CeltDecoderState
{
    /// <summary>Number of channels in this stream (1 or 2).</summary>
    internal int Channels;

    /// <summary>Number of channels emitted to the output buffer.</summary>
    internal int OutputChannels;

    /// <summary>Downsample factor: 1, 2, 3, or 6 (for 48k -> 8/12/16/24/48 kHz output).</summary>
    internal int Downsample;

    /// <summary>First band to decode (always 0 for standard Opus).</summary>
    internal int Start;

    /// <summary>Last band + 1 to decode (effective bandwidth limit per TOC).</summary>
    internal int End;

    /// <summary>Range coder rng tail value at the end of the most recent frame; carried for state integrity checks.</summary>
    internal uint Rng;

    /// <summary>Last-frame error code (0 = OK).</summary>
    internal int Error;

    /// <summary>Most recent pitch index detected by the PLC pitch search.</summary>
    internal int LastPitchIndex;

    /// <summary>Number of consecutive lost frames concealed.</summary>
    internal int LossCount;

    /// <summary>Post-filter pitch period (samples) for current and previous frame.</summary>
    internal int PostfilterPeriod;
    internal int PostfilterPeriodOld;

    /// <summary>Post-filter gain (Q15) for current and previous frame.</summary>
    internal int PostfilterGain;
    internal int PostfilterGainOld;

    /// <summary>Post-filter tap selector (0/1/2) for current and previous frame.</summary>
    internal int PostfilterTapset;
    internal int PostfilterTapsetOld;

    /// <summary>2-tap pre-emphasis filter memory (one entry per channel).</summary>
    internal readonly int[] PreemphMemD = new int[2];

    /// <summary>
    /// Per-channel decode buffer. Size = channels * (DECODE_BUFFER_SIZE + overlap).
    /// Holds the running synthesis history; the most recent N samples are the
    /// output that gets de-emphasized into the caller's PCM buffer.
    /// </summary>
    internal int[][]? DecodeMem;

    /// <summary>Per-channel LPC coefficients used during PLC. Size = channels * LPC_ORDER.</summary>
    internal int[][]? Lpc;

    /// <summary>Last-decoded log-energies per band; size = 2 * NbEBands (always 2 channels worth, mono uses [0..NbEBands)).</summary>
    internal int[]? OldEBands;

    /// <summary>Last-decoded log-energies (one frame back); size = 2 * NbEBands.</summary>
    internal int[]? OldLogE;

    /// <summary>Last-decoded log-energies (two frames back); size = 2 * NbEBands.</summary>
    internal int[]? OldLogE2;

    /// <summary>Background log-energy floor used by the PLC; size = 2 * NbEBands.</summary>
    internal int[]? BackgroundLogE;

    /// <summary>Reset all dynamic state for a fresh decode.</summary>
    internal void Reset()
    {
        Channels = 0;
        OutputChannels = 0;
        Downsample = 0;
        Start = 0;
        End = 0;
        PartialReset();
    }

    /// <summary>
    /// Reset state that is regenerated on every packet (rng, post-filter,
    /// energies). Static config (channel count, downsample factor) is preserved.
    /// Mirrors libopus OPUS_RESET_STATE for the CELT half.
    /// </summary>
    internal void PartialReset()
    {
        Rng = 0;
        Error = 0;
        LastPitchIndex = 0;
        LossCount = 0;
        PostfilterPeriod = 0;
        PostfilterPeriodOld = 0;
        PostfilterGain = 0;
        PostfilterGainOld = 0;
        PostfilterTapset = 0;
        PostfilterTapsetOld = 0;
        PreemphMemD[0] = 0;
        PreemphMemD[1] = 0;
        DecodeMem = null;
        Lpc = null;
        OldEBands = null;
        OldLogE = null;
        OldLogE2 = null;
        BackgroundLogE = null;
    }
}
