// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Structural port of constants from libFLAC's format.h and stream_decoder.c
// for the FLAC (Free Lossless Audio Codec) decoder.
//
// Upstream Copyright (c) 2000-2009 Josh Coalson, 2011-2023 Xiph.Org Foundation.
// BSD 3-Clause. See NOTICE.md.

namespace SpawnDev.Codecs.Audio.Flac;

/// <summary>
/// FLAC format constants. Values match libFLAC's format.h.
/// </summary>
internal static class FlacConstants
{
    /// <summary>The 4-byte "fLaC" stream marker at the start of every FLAC stream.</summary>
    internal static readonly byte[] StreamMarker = { (byte)'f', (byte)'L', (byte)'a', (byte)'C' };

    /// <summary>14-bit FLAC frame sync code: <c>0x3FFE</c>.</summary>
    internal const int FrameSyncCode = 0x3FFE;

    /// <summary>Maximum block size in samples. Libflac <c>FLAC__MAX_BLOCK_SIZE = 65535</c>.</summary>
    internal const int MaxBlockSize = 65535;

    /// <summary>Maximum channels supported. Libflac <c>FLAC__MAX_CHANNELS = 8</c>.</summary>
    internal const int MaxChannels = 8;

    /// <summary>Maximum LPC order. Libflac <c>FLAC__MAX_LPC_ORDER = 32</c>.</summary>
    internal const int MaxLpcOrder = 32;

    /// <summary>Maximum fixed predictor order. Libflac <c>FLAC__MAX_FIXED_ORDER = 4</c>.</summary>
    internal const int MaxFixedOrder = 4;

    // ----- Metadata block types -----

    /// <summary>STREAMINFO metadata block type.</summary>
    internal const int MetadataStreamInfo = 0;

    /// <summary>PADDING metadata block type.</summary>
    internal const int MetadataPadding = 1;

    /// <summary>APPLICATION metadata block type.</summary>
    internal const int MetadataApplication = 2;

    /// <summary>SEEKTABLE metadata block type.</summary>
    internal const int MetadataSeekTable = 3;

    /// <summary>VORBIS_COMMENT metadata block type.</summary>
    internal const int MetadataVorbisComment = 4;

    /// <summary>CUESHEET metadata block type.</summary>
    internal const int MetadataCuesheet = 5;

    /// <summary>PICTURE metadata block type.</summary>
    internal const int MetadataPicture = 6;

    // ----- Subframe types -----

    /// <summary>CONSTANT subframe: single sample value replicated.</summary>
    internal const int SubframeConstant = 0;

    /// <summary>VERBATIM subframe: raw uncompressed samples.</summary>
    internal const int SubframeVerbatim = 1;

    /// <summary>FIXED subframe: fixed-predictor residual coding (order 0-4).</summary>
    internal const int SubframeFixed = 8;

    /// <summary>LPC subframe: linear-predictor residual coding (order 1-32).</summary>
    internal const int SubframeLpc = 32;

    // ----- Channel assignment modes (frame header) -----

    /// <summary>Independent channels (no inter-channel prediction).</summary>
    internal const int ChannelAssignmentIndependent = 0;

    /// <summary>Left + side stereo.</summary>
    internal const int ChannelAssignmentLeftSide = 8;

    /// <summary>Right + side stereo.</summary>
    internal const int ChannelAssignmentRightSide = 9;

    /// <summary>Mid + side stereo.</summary>
    internal const int ChannelAssignmentMidSide = 10;

    // ----- Residual coding methods -----

    /// <summary>PARTITIONED_RICE residual coding (partitions with 4-bit parameter).</summary>
    internal const int ResidualCodingPartitionedRice = 0;

    /// <summary>PARTITIONED_RICE2 residual coding (partitions with 5-bit parameter).</summary>
    internal const int ResidualCodingPartitionedRice2 = 1;

    /// <summary>Escape parameter signaling verbatim residuals (4-bit rice parameter).</summary>
    internal const int RiceParameterEscape = 15;

    /// <summary>Escape parameter signaling verbatim residuals (5-bit rice parameter).</summary>
    internal const int Rice2ParameterEscape = 31;
}
