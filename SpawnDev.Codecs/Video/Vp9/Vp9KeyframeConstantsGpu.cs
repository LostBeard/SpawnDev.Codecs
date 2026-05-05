// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Unified constant tables for VP9 v1 keyframe encoder + decoder GPU
// kernels. Packs every prob/scan/neighbor/lookup table the entropy
// kernel and decode kernel need into two big buffers (one byte, one
// ushort) that the host uploads once per accelerator and reuses
// across every frame.
//
// Why pack everything: ILGPU's Action&lt;&gt; entry-point arg budget is
// 15. A from-scratch VP9 entropy kernel needs probs + scan tables +
// neighbor tables + coef-prob defaults + cat probs + pareto8 + etc.
// Naively that's a dozen separate ArrayViews. Pack them into one
// byte buffer + one ushort buffer with offset constants, and the
// kernel signature stays well under the budget.
//
// Byte buffer layout (5048 bytes total):
//   [0..3142]    Vp9BlockCoefEncoderGpu consts (3143 bytes - reuses
//                that class's BuildConstsBuffer output exactly so
//                Vp9BlockCoefEncoderGpu.EncodeBlock works unchanged
//                with offsets into this section).
//   [3143..4042] Vp9KfYModeProbs (900 bytes - 10 above x 10 left x 9 nodes)
//   [4043..4132] Vp9IntraModeProbs.KfUvModeProbs (90 bytes - 10 yMode x 9 nodes)
//   [4133..4180] Vp9PartitionProbs.KfPartitionProbs (48 bytes - 4 sizeIdx x 4 splitState x 3 nodes)
//   [4181..4183] Vp9SkipProbs.DefaultProbs (3 bytes - one per skipContext)
//   [4184..4615] Vp9CoefProbs.DefaultCoefProbs8x8 (432 bytes)
//   [4616..5047] Vp9CoefProbs.DefaultCoefProbs16x16 (432 bytes)
//
// Ushort buffer layout (960 entries = 1920 bytes total):
//   [0..63]      Vp9ScanTables.DefaultScan8x8 (64)
//   [64..319]    Vp9ScanTables.DefaultScan16x16 (256)
//   [320..447]   Vp9NeighborTables default scan 8x8 (128 = 64*2)
//   [448..959]   Vp9NeighborTables default scan 16x16 (512 = 256*2)
//
// V1 simplifications: only Tx8x8 (chroma) + Tx16x16 (luma) tables
// are packed. Tx4x4 and Tx32x32 will be added in a follow-up if a
// codec stage needs them.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// Builder + offset constants for the unified VP9 v1 keyframe
/// encoder/decoder constant buffers.
/// </summary>
public static class Vp9KeyframeConstantsGpu
{
    // === Byte buffer offsets ===

    /// <summary>Offset of the Vp9BlockCoefEncoderGpu consts section.</summary>
    public const int CoefConstsOffset = 0;
    /// <summary>Length of the Vp9BlockCoefEncoderGpu consts section.</summary>
    public const int CoefConstsLength = Vp9BlockCoefEncoderGpu.ConstsTotalBytes; // 3143

    /// <summary>Offset of Vp9KfYModeProbs (900 bytes).</summary>
    public const int KfYModeProbsOffset = CoefConstsOffset + CoefConstsLength;
    /// <summary>Length of Vp9KfYModeProbs.</summary>
    public const int KfYModeProbsLength = 900;

    /// <summary>Offset of Vp9IntraModeProbs.KfUvModeProbs (90 bytes).</summary>
    public const int KfUvModeProbsOffset = KfYModeProbsOffset + KfYModeProbsLength;
    /// <summary>Length of KfUvModeProbs.</summary>
    public const int KfUvModeProbsLength = 90;

    /// <summary>Offset of Vp9PartitionProbs.KfPartitionProbs (48 bytes).</summary>
    public const int KfPartitionProbsOffset = KfUvModeProbsOffset + KfUvModeProbsLength;
    /// <summary>Length of KfPartitionProbs.</summary>
    public const int KfPartitionProbsLength = 48;

    /// <summary>Offset of Vp9SkipProbs.DefaultProbs (3 bytes).</summary>
    public const int SkipProbsOffset = KfPartitionProbsOffset + KfPartitionProbsLength;
    /// <summary>Length of SkipProbs.</summary>
    public const int SkipProbsLength = 3;

    /// <summary>Offset of Vp9CoefProbs.DefaultCoefProbs4x4 (432 bytes).</summary>
    public const int CoefProbs4x4Offset = SkipProbsOffset + SkipProbsLength;
    /// <summary>Length of DefaultCoefProbs4x4.</summary>
    public const int CoefProbs4x4Length = 432;

    /// <summary>Offset of Vp9CoefProbs.DefaultCoefProbs8x8 (432 bytes).</summary>
    public const int CoefProbs8x8Offset = CoefProbs4x4Offset + CoefProbs4x4Length;
    /// <summary>Length of DefaultCoefProbs8x8.</summary>
    public const int CoefProbs8x8Length = 432;

    /// <summary>Offset of Vp9CoefProbs.DefaultCoefProbs16x16 (432 bytes).</summary>
    public const int CoefProbs16x16Offset = CoefProbs8x8Offset + CoefProbs8x8Length;
    /// <summary>Length of DefaultCoefProbs16x16.</summary>
    public const int CoefProbs16x16Length = 432;

    /// <summary>Total byte buffer size.</summary>
    public const int ByteConstsTotalBytes = CoefProbs16x16Offset + CoefProbs16x16Length;

    // === Ushort buffer offsets ===

    /// <summary>Offset of DefaultScan4x4 in the ushort buffer (16 ushorts).</summary>
    public const int Scan4x4Offset = 0;
    /// <summary>Length of DefaultScan4x4 (16).</summary>
    public const int Scan4x4Length = 16;

    /// <summary>Offset of DefaultScan8x8 (64 ushorts).</summary>
    public const int Scan8x8Offset = Scan4x4Offset + Scan4x4Length;
    /// <summary>Length of DefaultScan8x8 (64).</summary>
    public const int Scan8x8Length = 64;

    /// <summary>Offset of DefaultScan16x16 (256 ushorts).</summary>
    public const int Scan16x16Offset = Scan8x8Offset + Scan8x8Length;
    /// <summary>Length of DefaultScan16x16 (256).</summary>
    public const int Scan16x16Length = 256;

    /// <summary>Offset of Default-scan Neighbors4x4 (32 ushorts = 16*2).</summary>
    public const int Neighbors4x4Offset = Scan16x16Offset + Scan16x16Length;
    /// <summary>Length of Neighbors4x4.</summary>
    public const int Neighbors4x4Length = 32;

    /// <summary>Offset of Default-scan Neighbors8x8 (128 ushorts).</summary>
    public const int Neighbors8x8Offset = Neighbors4x4Offset + Neighbors4x4Length;
    /// <summary>Length of Neighbors8x8 (128 = 64*2).</summary>
    public const int Neighbors8x8Length = 128;

    /// <summary>Offset of Default-scan Neighbors16x16 (512 ushorts).</summary>
    public const int Neighbors16x16Offset = Neighbors8x8Offset + Neighbors8x8Length;
    /// <summary>Length of Neighbors16x16 (512 = 256*2).</summary>
    public const int Neighbors16x16Length = 512;

    /// <summary>Total ushort buffer size.</summary>
    public const int UshortConstsTotalEntries = Neighbors16x16Offset + Neighbors16x16Length;

    /// <summary>
    /// Build the byte constants buffer for upload. Caller materialises
    /// once per accelerator and reuses across every frame.
    /// </summary>
    public static byte[] BuildByteConstsBuffer()
    {
        var buf = new byte[ByteConstsTotalBytes];

        // Coef encoder/decoder consts (band tables + ptEnergyClass +
        // pareto8 + cat probs).
        var coefConsts = Vp9BlockCoefEncoderGpu.BuildConstsBuffer();
        Array.Copy(coefConsts, 0, buf, CoefConstsOffset, CoefConstsLength);

        Array.Copy(Vp9IntraModeProbs.KfYModeProbs, 0, buf, KfYModeProbsOffset, KfYModeProbsLength);
        Array.Copy(Vp9IntraModeProbs.KfUvModeProbs, 0, buf, KfUvModeProbsOffset, KfUvModeProbsLength);
        Array.Copy(Vp9PartitionProbs.KfPartitionProbs, 0, buf, KfPartitionProbsOffset, KfPartitionProbsLength);
        Array.Copy(Vp9SkipProbs.DefaultProbs, 0, buf, SkipProbsOffset, SkipProbsLength);
        Array.Copy(Vp9CoefProbs.DefaultCoefProbs4x4, 0, buf, CoefProbs4x4Offset, CoefProbs4x4Length);
        Array.Copy(Vp9CoefProbs.DefaultCoefProbs8x8, 0, buf, CoefProbs8x8Offset, CoefProbs8x8Length);
        Array.Copy(Vp9CoefProbs.DefaultCoefProbs16x16, 0, buf, CoefProbs16x16Offset, CoefProbs16x16Length);

        return buf;
    }

    /// <summary>
    /// Build the ushort constants buffer for upload. Caller materialises
    /// once per accelerator and reuses across every frame.
    /// </summary>
    public static ushort[] BuildUshortConstsBuffer()
    {
        var buf = new ushort[UshortConstsTotalEntries];

        Array.Copy(Vp9ScanTables.DefaultScan4x4, 0, buf, Scan4x4Offset, Scan4x4Length);
        Array.Copy(Vp9ScanTables.DefaultScan8x8, 0, buf, Scan8x8Offset, Scan8x8Length);
        Array.Copy(Vp9ScanTables.DefaultScan16x16, 0, buf, Scan16x16Offset, Scan16x16Length);

        var n4 = Vp9NeighborTables.GetNeighbors4x4(Vp9ScanType.Default);
        var n8 = Vp9NeighborTables.GetNeighbors8x8(Vp9ScanType.Default);
        var n16 = Vp9NeighborTables.GetNeighbors16x16(Vp9ScanType.Default);
        Array.Copy(n4, 0, buf, Neighbors4x4Offset, Neighbors4x4Length);
        Array.Copy(n8, 0, buf, Neighbors8x8Offset, Neighbors8x8Length);
        Array.Copy(n16, 0, buf, Neighbors16x16Offset, Neighbors16x16Length);

        return buf;
    }
}
