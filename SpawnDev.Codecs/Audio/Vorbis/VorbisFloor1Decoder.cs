// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Vorbis floor type 1 posterior-value decoder per Vorbis I Section 7.2.3.
// Given the per-packet bit stream and the floor config + the setup's
// codebooks, reads the Y-coordinate list for this channel's floor curve,
// or returns null when the "nonzero" flag indicates a silent floor.

namespace SpawnDev.Codecs.Audio.Vorbis;

internal static class VorbisFloor1Decoder
{
    // Vorbis I Table 7.2.4: endpoint range bits per multiplier.
    private static readonly int[] EndpointBits = { 0, 8, 7, 7, 6 };

    /// <summary>
    /// Decode the floor 1 Y coordinates for one channel. Returns <c>null</c>
    /// if the packet signalled a silent floor (nonzero flag = 0).
    /// </summary>
    /// <param name="reader">Bit reader positioned at the floor data in the audio packet.</param>
    /// <param name="config">Floor 1 configuration from the setup header.</param>
    /// <param name="decoders">Pre-built Huffman decoders indexed by setup codebook index.</param>
    internal static int[]? Decode(
        ref VorbisBitReader reader,
        VorbisFloor1Config config,
        VorbisHuffmanDecoder[] decoders)
    {
        bool nonzero = reader.ReadBit() != 0;
        if (!nonzero) return null;

        int endpointBits = EndpointBits[config.Multiplier];
        var y = new int[config.XList.Length];
        y[0] = (int)reader.ReadBits(endpointBits);
        y[1] = (int)reader.ReadBits(endpointBits);

        int yIndex = 2;
        for (int partition = 0; partition < config.Partitions; partition++)
        {
            int cls = config.PartitionClassList[partition];
            int cdim = config.ClassDimensions[cls];
            int cbits = config.ClassSubclasses[cls];
            int csub = (1 << cbits) - 1;
            int cval = 0;

            if (cbits > 0)
            {
                int masterbook = config.ClassMasterbooks[cls];
                if (masterbook < 0 || masterbook >= decoders.Length)
                    throw new InvalidDataException(
                        $"Floor 1 class {cls} masterbook index {masterbook} out of range.");
                cval = decoders[masterbook].Decode(ref reader);
            }

            for (int j = 0; j < cdim; j++)
            {
                int book = config.ClassSubclassBooks[cls][cval & csub];
                cval >>= cbits;
                if (book >= 0)
                {
                    if (book >= decoders.Length)
                        throw new InvalidDataException(
                            $"Floor 1 subclass book index {book} out of range.");
                    y[yIndex++] = decoders[book].Decode(ref reader);
                }
                else
                {
                    y[yIndex++] = 0;
                }
            }
        }
        return y;
    }
}
