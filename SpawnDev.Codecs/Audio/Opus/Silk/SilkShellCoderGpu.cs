// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-callable port of SilkShellCoder.Decode (libopus silk/shell_coder.c).
// Decodes the unsigned pulse magnitudes for one 16-sample shell block via
// a balanced 4-level binary tree. At each split node:
//   left = OpusRangeDecoderGpu.DecodeIcdf(table[Offsets[parent]..])
//   right = parent - left
//
// Tree levels use different tables (Table0 for final 2-leaf splits,
// Table3 for the root 2-way split of the whole block). Tree walk is
// fully unrolled - no recursion. 15 split calls total per shell block.
//
// Sequential per-stream because every DecodeIcdf advances the shared
// range decoder state. One thread per stream.
//
// Upstream Copyright (c) 2006-2011 Skype Limited. BSD 3-Clause.

using ILGPU;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// Body-struct kernel parameter for SilkShellCoderGpu callers: the four per-tree-level shell code tables + the
/// offsets table packed into one byte buffer (element offsets in the *At fields; <see cref="PackedDeviceBuffer{T}"/>),
/// one storage binding instead of five.
/// </summary>
public struct SilkShellCoderTables
{
    /// <summary>The five tables.</summary>
    public ArrayView<byte> Bytes;
    /// <summary>silk_shell_code_table_offsets - 17 entries, byte offset
    /// into each Table[N] for pulse count p in [0, 16].</summary>
    public int OffsetsAt;
    /// <summary>silk_shell_code_table0 - leaf-level table.</summary>
    public int Table0At;
    /// <summary>silk_shell_code_table1.</summary>
    public int Table1At;
    /// <summary>silk_shell_code_table2.</summary>
    public int Table2At;
    /// <summary>silk_shell_code_table3 - root-level table.</summary>
    public int Table3At;

    /// <summary>
    /// Appends the standard libopus shell code tables to <paramref name="bytes"/>; set <see cref="Bytes"/> to the
    /// uploaded buffer.
    /// </summary>
    public static SilkShellCoderTables AddStandard(PackedDeviceBuffer<byte> bytes) => new SilkShellCoderTables
    {
        OffsetsAt = bytes.Add(SilkShellCodeTables.Offsets),
        Table0At = bytes.Add(SilkShellCodeTables.Table0),
        Table1At = bytes.Add(SilkShellCodeTables.Table1),
        Table2At = bytes.Add(SilkShellCodeTables.Table2),
        Table3At = bytes.Add(SilkShellCodeTables.Table3),
    };
}

/// <summary>
/// GPU-callable shell-block pulse decoder. Mirror of
/// `SilkShellCoder.Decode`.
/// </summary>
public static class SilkShellCoderGpu
{
    /// <summary>SilkConstants.SHELL_CODEC_FRAME_LENGTH = 16.</summary>
    public const int ShellCodecFrameLength = 16;

    /// <summary>
    /// Decode 16 pulse magnitudes for one shell block. Output written to
    /// <paramref name="pulsesOut"/>[<paramref name="pulsesOutBase"/>..+16).
    /// </summary>
    public static void Decode(
        ref OpusRangeDecoderGpuState state,
        ArrayView<byte> buf, int bufStart, uint storage,
        SilkShellCoderTables tables,
        int pulsesTotal,
        ArrayView<short> pulsesOut, long pulsesOutBase)
    {
        // Tier-3: split the whole block into two halves of 8.
        int p3a, p3b;
        DecodeSplit(ref state, buf, bufStart, storage,
            tables.Bytes, tables.Table3At, tables.OffsetsAt, pulsesTotal, out p3a, out p3b);

        // Tier-2: each tier-3 half splits into two quarters of 4.
        int p2a, p2b, p2c, p2d;
        DecodeSplit(ref state, buf, bufStart, storage,
            tables.Bytes, tables.Table2At, tables.OffsetsAt, p3a, out p2a, out p2b);
        // Note: libopus emits the RIGHT-half tier-2 split AFTER finishing
        // both tier-1+0 levels of the LEFT-half tier-2. Tree-walk order
        // matches CPU SilkShellCoder.Decode line-for-line.

        // Tier-1: each tier-2 quarter splits into two pairs of 2.
        int p1a, p1b;
        DecodeSplit(ref state, buf, bufStart, storage,
            tables.Bytes, tables.Table1At, tables.OffsetsAt, p2a, out p1a, out p1b);

        // Tier-0: each tier-1 pair splits into two singletons.
        int leaf0, leaf1;
        DecodeSplit(ref state, buf, bufStart, storage,
            tables.Bytes, tables.Table0At, tables.OffsetsAt, p1a, out leaf0, out leaf1);
        pulsesOut[pulsesOutBase + 0] = (short)leaf0;
        pulsesOut[pulsesOutBase + 1] = (short)leaf1;
        DecodeSplit(ref state, buf, bufStart, storage,
            tables.Bytes, tables.Table0At, tables.OffsetsAt, p1b, out leaf0, out leaf1);
        pulsesOut[pulsesOutBase + 2] = (short)leaf0;
        pulsesOut[pulsesOutBase + 3] = (short)leaf1;

        int p1c, p1d;
        DecodeSplit(ref state, buf, bufStart, storage,
            tables.Bytes, tables.Table1At, tables.OffsetsAt, p2b, out p1c, out p1d);
        DecodeSplit(ref state, buf, bufStart, storage,
            tables.Bytes, tables.Table0At, tables.OffsetsAt, p1c, out leaf0, out leaf1);
        pulsesOut[pulsesOutBase + 4] = (short)leaf0;
        pulsesOut[pulsesOutBase + 5] = (short)leaf1;
        DecodeSplit(ref state, buf, bufStart, storage,
            tables.Bytes, tables.Table0At, tables.OffsetsAt, p1d, out leaf0, out leaf1);
        pulsesOut[pulsesOutBase + 6] = (short)leaf0;
        pulsesOut[pulsesOutBase + 7] = (short)leaf1;

        // Right-half tier-2.
        DecodeSplit(ref state, buf, bufStart, storage,
            tables.Bytes, tables.Table2At, tables.OffsetsAt, p3b, out p2c, out p2d);

        int p1e, p1f;
        DecodeSplit(ref state, buf, bufStart, storage,
            tables.Bytes, tables.Table1At, tables.OffsetsAt, p2c, out p1e, out p1f);
        DecodeSplit(ref state, buf, bufStart, storage,
            tables.Bytes, tables.Table0At, tables.OffsetsAt, p1e, out leaf0, out leaf1);
        pulsesOut[pulsesOutBase + 8] = (short)leaf0;
        pulsesOut[pulsesOutBase + 9] = (short)leaf1;
        DecodeSplit(ref state, buf, bufStart, storage,
            tables.Bytes, tables.Table0At, tables.OffsetsAt, p1f, out leaf0, out leaf1);
        pulsesOut[pulsesOutBase + 10] = (short)leaf0;
        pulsesOut[pulsesOutBase + 11] = (short)leaf1;

        int p1g, p1h;
        DecodeSplit(ref state, buf, bufStart, storage,
            tables.Bytes, tables.Table1At, tables.OffsetsAt, p2d, out p1g, out p1h);
        DecodeSplit(ref state, buf, bufStart, storage,
            tables.Bytes, tables.Table0At, tables.OffsetsAt, p1g, out leaf0, out leaf1);
        pulsesOut[pulsesOutBase + 12] = (short)leaf0;
        pulsesOut[pulsesOutBase + 13] = (short)leaf1;
        DecodeSplit(ref state, buf, bufStart, storage,
            tables.Bytes, tables.Table0At, tables.OffsetsAt, p1h, out leaf0, out leaf1);
        pulsesOut[pulsesOutBase + 14] = (short)leaf0;
        pulsesOut[pulsesOutBase + 15] = (short)leaf1;
    }

    /// <summary>
    /// Decode one shell-tree split: read the iCDF entry for the left child,
    /// derive the right child as parent - left.
    /// </summary>
    private static void DecodeSplit(
        ref OpusRangeDecoderGpuState state,
        ArrayView<byte> buf, int bufStart, uint storage,
        ArrayView<byte> bytes, int shellTableAt, int offsetsAt,
        int parent,
        out int leftChild, out int rightChild)
    {
        if (parent > 0)
        {
            int tableOffset = bytes[offsetsAt + parent];
            int left = OpusRangeDecoderGpu.DecodeIcdf(
                ref state, buf, bufStart, storage,
                bytes, shellTableAt + tableOffset, 8);
            leftChild = left;
            rightChild = parent - left;
        }
        else
        {
            leftChild = 0;
            rightChild = 0;
        }
    }
}
