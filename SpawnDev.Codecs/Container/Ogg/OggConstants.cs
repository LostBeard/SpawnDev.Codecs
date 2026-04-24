// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Ogg container constants per RFC 3533.

namespace SpawnDev.Codecs.Container.Ogg;

/// <summary>Ogg container constants.</summary>
internal static class OggConstants
{
    /// <summary>4-byte capture pattern at the start of every Ogg page: "OggS".</summary>
    internal static readonly byte[] CapturePattern = { (byte)'O', (byte)'g', (byte)'g', (byte)'S' };

    /// <summary>Current Ogg stream structure version; always 0.</summary>
    internal const byte Version = 0;

    /// <summary>Header-type flag: page is continuation of a packet from the previous page.</summary>
    internal const byte HeaderTypeContinuation = 0x01;

    /// <summary>Header-type flag: page is the beginning of stream (BOS).</summary>
    internal const byte HeaderTypeBeginningOfStream = 0x02;

    /// <summary>Header-type flag: page is the end of stream (EOS).</summary>
    internal const byte HeaderTypeEndOfStream = 0x04;

    /// <summary>Ogg page headers are 27 fixed bytes + the segment table.</summary>
    internal const int FixedHeaderLength = 27;
}
