// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// 4-byte FLAC metadata block header. See RFC 9639 Section 8 / libFLAC format.h
// FLAC__StreamMetadata (the first 4 bytes).

namespace SpawnDev.Codecs.Audio.Flac;

/// <summary>
/// 4-byte metadata block header: 1-bit last-block flag, 7-bit block type, 24-bit
/// payload length in bytes.
/// </summary>
public readonly record struct FlacMetadataBlockHeader(bool IsLast, int BlockType, int LengthBytes);
