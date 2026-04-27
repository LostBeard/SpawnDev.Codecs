// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 partition type enum + constants. Mirrors libaom av1/common/enums.h
// PARTITION_TYPE + related defines.
//
// Spec reference: AV1 Bitstream and Decoding Process Specification sec 6.4.4
// (Partition syntax) and sec 9.3 (Partition CDFs).

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>
/// AV1 block partition type. Encodes how a parent block is split into
/// children. Matches libaom <c>PARTITION_TYPE</c> ordering exactly.
/// </summary>
public enum Av1PartitionType : byte
{
    /// <summary>PARTITION_NONE: no split, the block is decoded as-is.</summary>
    None = 0,
    /// <summary>PARTITION_HORZ: horizontal split (top/bottom halves).</summary>
    Horz = 1,
    /// <summary>PARTITION_VERT: vertical split (left/right halves).</summary>
    Vert = 2,
    /// <summary>PARTITION_SPLIT: 2x2 split into four equal sub-blocks.</summary>
    Split = 3,
    /// <summary>PARTITION_HORZ_A: HORZ split, top half further split into 2 quarters.</summary>
    HorzA = 4,
    /// <summary>PARTITION_HORZ_B: HORZ split, bottom half further split into 2 quarters.</summary>
    HorzB = 5,
    /// <summary>PARTITION_VERT_A: VERT split, left half further split into 2 quarters.</summary>
    VertA = 6,
    /// <summary>PARTITION_VERT_B: VERT split, right half further split into 2 quarters.</summary>
    VertB = 7,
    /// <summary>PARTITION_HORZ_4: 4:1 horizontal split (four narrow horizontal stripes).</summary>
    Horz4 = 8,
    /// <summary>PARTITION_VERT_4: 4:1 vertical split (four narrow vertical stripes).</summary>
    Vert4 = 9,
    /// <summary>Sentinel: number of extended partition types (10).</summary>
    ExtPartitionTypes = 10,
    /// <summary>PARTITION_INVALID: uninitialized / out-of-range marker.</summary>
    Invalid = 255,
}

/// <summary>AV1 partitioning constants (mirrors libaom av1/common/enums.h).</summary>
public static class Av1PartitionConstants
{
    /// <summary>Number of basic (non-extended) partition types: NONE, HORZ, VERT, SPLIT.</summary>
    public const int PartitionTypes = 4;
    /// <summary>Number of extended partition types (adds HORZ_A/B, VERT_A/B, HORZ_4, VERT_4).</summary>
    public const int ExtPartitionTypes = 10;
    /// <summary>Number of partition probability models per block size (libaom PARTITION_PLOFFSET).</summary>
    public const int PartitionPlaneOffset = 4;
    /// <summary>Number of distinct block sizes that have a partition CDF (libaom PARTITION_BLOCK_SIZES).</summary>
    public const int PartitionBlockSizes = 5;
    /// <summary>Total partition contexts = PartitionBlockSizes * PartitionPlaneOffset = 20.</summary>
    public const int PartitionContexts = PartitionBlockSizes * PartitionPlaneOffset;
}
