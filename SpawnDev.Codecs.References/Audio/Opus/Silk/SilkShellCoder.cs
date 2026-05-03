// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Structural port of libopus silk/shell_coder.c to clean C#. The shell coder
// decomposes the magnitudes of a 16-sample pulse block via a balanced binary
// tree: at each split, the decoder reads one side of the split from an iCDF
// (sized by the parent's pulse count) and derives the other side.
//
// Upstream Copyright (c) 2006-2011 Skype Limited. BSD 3-Clause. See NOTICE.md.

using SpawnDev.Codecs.EntropyCoders;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// Shell decoder and matching encoder. The decoder reads a series of balanced
/// binary splits of a 16-pulse block from the range decoder and materializes
/// the unsigned per-sample pulse magnitudes. The encoder performs the inverse
/// tree walk and is used by tests (and, later, by a SILK encoder).
/// </summary>
internal static class SilkShellCoder
{
    /// <summary>
    /// Decode one split of a pulse block: given that <paramref name="parent"/> pulses
    /// reach this node, read the count that goes into the left child from the iCDF at
    /// <c>shellTable[ShellCodeTables.Offsets[parent]..]</c>, and derive the right child.
    /// </summary>
    private static void DecodeSplit(
        OpusRangeDecoder rangeDec,
        int parent,
        byte[] shellTable,
        out short leftChild,
        out short rightChild)
    {
        if (parent > 0)
        {
            int left = rangeDec.DecodeIcdf(
                shellTable.AsSpan(SilkShellCodeTables.Offsets[parent]),
                8);
            leftChild = (short)left;
            rightChild = (short)(parent - left);
        }
        else
        {
            leftChild = 0;
            rightChild = 0;
        }
    }

    /// <summary>
    /// Mirrors <see cref="DecodeSplit"/>. Writes the left-child pulse count to the
    /// bitstream (and the right-child count is implicit as <c>parent - left</c>).
    /// </summary>
    private static void EncodeSplit(
        OpusRangeEncoder rangeEnc,
        int parent,
        int leftChild,
        byte[] shellTable)
    {
        if (parent > 0)
        {
            rangeEnc.EncodeIcdf(leftChild,
                shellTable.AsSpan(SilkShellCodeTables.Offsets[parent]),
                8);
        }
    }

    /// <summary>
    /// Decode the unsigned magnitudes of one 16-sample shell block given the total
    /// pulse count <paramref name="pulsesTotal"/>. Matches libopus <c>silk_shell_decoder</c>.
    /// </summary>
    /// <param name="pulses">Output: 16 unsigned magnitudes.</param>
    /// <param name="rangeDec">Range decoder positioned at the start of the shell block.</param>
    /// <param name="pulsesTotal">Total pulses for the block (typically in <c>[0, 16]</c>).</param>
    internal static void Decode(Span<short> pulses, OpusRangeDecoder rangeDec, int pulsesTotal)
    {
        if (rangeDec is null) throw new ArgumentNullException(nameof(rangeDec));
        if (pulses.Length < SilkConstants.SHELL_CODEC_FRAME_LENGTH)
            throw new ArgumentException(
                $"pulses too small (need {SilkConstants.SHELL_CODEC_FRAME_LENGTH}).",
                nameof(pulses));

        Span<short> pulses3 = stackalloc short[2];
        Span<short> pulses2 = stackalloc short[4];
        Span<short> pulses1 = stackalloc short[8];

        DecodeSplit(rangeDec, pulsesTotal, SilkShellCodeTables.Table3, out pulses3[0], out pulses3[1]);

        DecodeSplit(rangeDec, pulses3[0], SilkShellCodeTables.Table2, out pulses2[0], out pulses2[1]);

        DecodeSplit(rangeDec, pulses2[0], SilkShellCodeTables.Table1, out pulses1[0], out pulses1[1]);
        DecodeSplit(rangeDec, pulses1[0], SilkShellCodeTables.Table0, out pulses[0], out pulses[1]);
        DecodeSplit(rangeDec, pulses1[1], SilkShellCodeTables.Table0, out pulses[2], out pulses[3]);

        DecodeSplit(rangeDec, pulses2[1], SilkShellCodeTables.Table1, out pulses1[2], out pulses1[3]);
        DecodeSplit(rangeDec, pulses1[2], SilkShellCodeTables.Table0, out pulses[4], out pulses[5]);
        DecodeSplit(rangeDec, pulses1[3], SilkShellCodeTables.Table0, out pulses[6], out pulses[7]);

        DecodeSplit(rangeDec, pulses3[1], SilkShellCodeTables.Table2, out pulses2[2], out pulses2[3]);

        DecodeSplit(rangeDec, pulses2[2], SilkShellCodeTables.Table1, out pulses1[4], out pulses1[5]);
        DecodeSplit(rangeDec, pulses1[4], SilkShellCodeTables.Table0, out pulses[8], out pulses[9]);
        DecodeSplit(rangeDec, pulses1[5], SilkShellCodeTables.Table0, out pulses[10], out pulses[11]);

        DecodeSplit(rangeDec, pulses2[3], SilkShellCodeTables.Table1, out pulses1[6], out pulses1[7]);
        DecodeSplit(rangeDec, pulses1[6], SilkShellCodeTables.Table0, out pulses[12], out pulses[13]);
        DecodeSplit(rangeDec, pulses1[7], SilkShellCodeTables.Table0, out pulses[14], out pulses[15]);
    }

    /// <summary>
    /// Encode the unsigned magnitudes of one 16-sample shell block. Inverse of
    /// <see cref="Decode"/>. Reads all 16 magnitudes and emits the same range-coder
    /// splits that <see cref="Decode"/> reads back.
    /// </summary>
    /// <param name="rangeEnc">Range encoder to write to.</param>
    /// <param name="pulses">16 unsigned magnitudes. Their sum must equal
    /// <paramref name="pulsesTotal"/>.</param>
    /// <param name="pulsesTotal">Total pulses for the block.</param>
    internal static void Encode(OpusRangeEncoder rangeEnc, ReadOnlySpan<short> pulses, int pulsesTotal)
    {
        if (rangeEnc is null) throw new ArgumentNullException(nameof(rangeEnc));
        if (pulses.Length < SilkConstants.SHELL_CODEC_FRAME_LENGTH)
            throw new ArgumentException(
                $"pulses too small (need {SilkConstants.SHELL_CODEC_FRAME_LENGTH}).",
                nameof(pulses));

        Span<int> pulses3 = stackalloc int[2];
        Span<int> pulses2 = stackalloc int[4];
        Span<int> pulses1 = stackalloc int[8];

        // Build the intermediate tiers by pairwise summation.
        for (int i = 0; i < 8; i++) pulses1[i] = pulses[2 * i] + pulses[2 * i + 1];
        for (int i = 0; i < 4; i++) pulses2[i] = pulses1[2 * i] + pulses1[2 * i + 1];
        for (int i = 0; i < 2; i++) pulses3[i] = pulses2[2 * i] + pulses2[2 * i + 1];

        if (pulses3[0] + pulses3[1] != pulsesTotal)
            throw new ArgumentException(
                $"pulses sum ({pulses3[0] + pulses3[1]}) != pulsesTotal ({pulsesTotal}).",
                nameof(pulses));

        EncodeSplit(rangeEnc, pulsesTotal, pulses3[0], SilkShellCodeTables.Table3);

        EncodeSplit(rangeEnc, pulses3[0], pulses2[0], SilkShellCodeTables.Table2);

        EncodeSplit(rangeEnc, pulses2[0], pulses1[0], SilkShellCodeTables.Table1);
        EncodeSplit(rangeEnc, pulses1[0], pulses[0], SilkShellCodeTables.Table0);
        EncodeSplit(rangeEnc, pulses1[1], pulses[2], SilkShellCodeTables.Table0);

        EncodeSplit(rangeEnc, pulses2[1], pulses1[2], SilkShellCodeTables.Table1);
        EncodeSplit(rangeEnc, pulses1[2], pulses[4], SilkShellCodeTables.Table0);
        EncodeSplit(rangeEnc, pulses1[3], pulses[6], SilkShellCodeTables.Table0);

        EncodeSplit(rangeEnc, pulses3[1], pulses2[2], SilkShellCodeTables.Table2);

        EncodeSplit(rangeEnc, pulses2[2], pulses1[4], SilkShellCodeTables.Table1);
        EncodeSplit(rangeEnc, pulses1[4], pulses[8], SilkShellCodeTables.Table0);
        EncodeSplit(rangeEnc, pulses1[5], pulses[10], SilkShellCodeTables.Table0);

        EncodeSplit(rangeEnc, pulses2[3], pulses1[6], SilkShellCodeTables.Table1);
        EncodeSplit(rangeEnc, pulses1[6], pulses[12], SilkShellCodeTables.Table0);
        EncodeSplit(rangeEnc, pulses1[7], pulses[14], SilkShellCodeTables.Table0);
    }
}
