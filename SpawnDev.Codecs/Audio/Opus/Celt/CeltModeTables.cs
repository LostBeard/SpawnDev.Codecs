// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt). See NOTICE.md for upstream attributions.
//
// GENERATED (scratchpad celt/gen_mode_tables.py): the static CELT mode 48000 Hz / 960 / overlap 120 (libopus
// static_modes_fixed.h), values from Concentus 2.2.2 Tables.cs and Rate.cs (BSD-3). CeltModeGpu packs them into one int view.

using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.Codecs.Audio.Opus.Celt;

/// <summary>The tables of the static 48 kHz CELT mode (host copies; <see cref="CeltModeGpu.Upload"/> packs them).</summary>
public static class CeltModeTables
{
    /// <summary>Band count (libopus nbEBands).</summary>
    public const int NbEBands = 21;
    /// <summary>Allocation steps (libopus nbAllocVectors).</summary>
    public const int NbAllocVectors = 11;
    /// <summary>Largest LM (frame = 120 &lt;&lt; LM samples).</summary>
    public const int MaxLM = 3;

    /// <summary>Band edges in 2.5 ms MDCT bins (libopus eband5ms), 22 entries for 21 bands.</summary>
    public static readonly short[] EBands =
    {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 34, 40, 48, 60,
        78, 100,
    };

    /// <summary>Bits per band for the 11 allocation steps, 1/32 bit per bin (libopus band_allocation), 11 x 21.</summary>
    public static readonly byte[] AllocVectors =
    {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 90, 80, 75, 69, 63, 56, 49, 40, 34, 29, 20, 18, 10, 0, 0, 0, 0, 0, 0,
        0, 0, 110, 100, 90, 84, 78, 71, 65, 58, 51, 45, 39, 32, 26, 20, 12, 0, 0, 0,
        0, 0, 0, 118, 110, 103, 93, 86, 80, 75, 70, 65, 59, 53, 47, 40, 31, 23, 15, 4,
        0, 0, 0, 0, 126, 119, 112, 104, 95, 89, 83, 78, 72, 66, 60, 54, 47, 39, 32, 25,
        17, 12, 1, 0, 0, 134, 127, 120, 114, 103, 97, 91, 85, 78, 72, 66, 60, 54, 47, 41,
        35, 29, 23, 16, 10, 1, 144, 137, 130, 124, 113, 107, 101, 95, 88, 82, 76, 70, 64, 57,
        51, 45, 39, 33, 26, 15, 1, 152, 145, 138, 132, 123, 117, 111, 105, 98, 92, 86, 80, 74,
        67, 61, 55, 49, 43, 36, 20, 1, 162, 155, 148, 142, 133, 127, 121, 115, 108, 102, 96, 90,
        84, 77, 71, 65, 59, 53, 46, 30, 1, 172, 165, 158, 152, 143, 137, 131, 125, 118, 112, 106,
        100, 94, 87, 81, 75, 69, 63, 56, 45, 20, 200, 200, 200, 200, 200, 200, 200, 200, 198, 193,
        188, 183, 178, 173, 168, 163, 158, 153, 148, 129, 104,
    };

    /// <summary>log2 of each band size in Q3 (libopus logN400).</summary>
    public static readonly short[] LogN =
    {
        0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 16, 16, 16, 21, 21, 24, 29, 34,
        36,
    };

    /// <summary>Pulse-cache row start per (LM + 1, band), -1 = none (libopus cache_index50), 5 x 21.</summary>
    public static readonly short[] CacheIndex =
    {
        -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 41, 41, 41, 82, 82, 123, 164, 200,
        222, 0, 0, 0, 0, 0, 0, 0, 0, 41, 41, 41, 41, 123, 123, 123, 164, 164, 240, 266,
        283, 295, 41, 41, 41, 41, 41, 41, 41, 41, 123, 123, 123, 123, 240, 240, 240, 266, 266, 305,
        318, 328, 336, 123, 123, 123, 123, 123, 123, 123, 123, 240, 240, 240, 240, 305, 305, 305, 318, 318,
        343, 351, 358, 364, 240, 240, 240, 240, 240, 240, 240, 240, 305, 305, 305, 305, 343, 343, 343, 351,
        351, 370, 376, 382, 387,
    };

    /// <summary>Pulse cache: bits for K pulses, row 0 = max K (libopus cache_bits50).</summary>
    public static readonly byte[] CacheBits =
    {
        40, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
        7, 40, 15, 23, 28, 31, 34, 36, 38, 39, 41, 42, 43, 44, 45, 46, 47, 47, 49, 50,
        51, 52, 53, 54, 55, 55, 57, 58, 59, 60, 61, 62, 63, 63, 65, 66, 67, 68, 69, 70,
        71, 71, 40, 20, 33, 41, 48, 53, 57, 61, 64, 66, 69, 71, 73, 75, 76, 78, 80, 82,
        85, 87, 89, 91, 92, 94, 96, 98, 101, 103, 105, 107, 108, 110, 112, 114, 117, 119, 121, 123,
        124, 126, 128, 40, 23, 39, 51, 60, 67, 73, 79, 83, 87, 91, 94, 97, 100, 102, 105, 107,
        111, 115, 118, 121, 124, 126, 129, 131, 135, 139, 142, 145, 148, 150, 153, 155, 159, 163, 166, 169,
        172, 174, 177, 179, 35, 28, 49, 65, 78, 89, 99, 107, 114, 120, 126, 132, 136, 141, 145, 149,
        153, 159, 165, 171, 176, 180, 185, 189, 192, 199, 205, 211, 216, 220, 225, 229, 232, 239, 245, 251,
        21, 33, 58, 79, 97, 112, 125, 137, 148, 157, 166, 174, 182, 189, 195, 201, 207, 217, 227, 235,
        243, 251, 17, 35, 63, 86, 106, 123, 139, 152, 165, 177, 187, 197, 206, 214, 222, 230, 237, 250,
        25, 31, 55, 75, 91, 105, 117, 128, 138, 146, 154, 161, 168, 174, 180, 185, 190, 200, 208, 215,
        222, 229, 235, 240, 245, 255, 16, 36, 65, 89, 110, 128, 144, 159, 173, 185, 196, 207, 217, 226,
        234, 242, 250, 11, 41, 74, 103, 128, 151, 172, 191, 209, 225, 241, 255, 9, 43, 79, 110, 138,
        163, 186, 207, 227, 246, 12, 39, 71, 99, 123, 144, 164, 182, 198, 214, 228, 241, 253, 9, 44,
        81, 113, 142, 168, 192, 214, 235, 255, 7, 49, 90, 127, 160, 191, 220, 247, 6, 51, 95, 134,
        170, 203, 234, 7, 47, 87, 123, 155, 184, 212, 237, 6, 52, 97, 137, 174, 208, 240, 5, 57,
        106, 151, 192, 231, 5, 59, 111, 158, 202, 243, 5, 55, 103, 147, 187, 224, 5, 60, 113, 161,
        206, 248, 4, 65, 122, 175, 224, 4, 67, 127, 182, 234,
    };

    /// <summary>Per-band bit caps per (LM, stereo) (libopus cache_caps50), 4 x 2 x 21.</summary>
    public static readonly byte[] CacheCaps =
    {
        224, 224, 224, 224, 224, 224, 224, 224, 160, 160, 160, 160, 185, 185, 185, 178, 178, 168, 134, 61,
        37, 224, 224, 224, 224, 224, 224, 224, 224, 240, 240, 240, 240, 207, 207, 207, 198, 198, 183, 144,
        66, 40, 160, 160, 160, 160, 160, 160, 160, 160, 185, 185, 185, 185, 193, 193, 193, 183, 183, 172,
        138, 64, 38, 240, 240, 240, 240, 240, 240, 240, 240, 207, 207, 207, 207, 204, 204, 204, 193, 193,
        180, 143, 66, 40, 185, 185, 185, 185, 185, 185, 185, 185, 193, 193, 193, 193, 193, 193, 193, 183,
        183, 172, 138, 65, 39, 207, 207, 207, 207, 207, 207, 207, 207, 204, 204, 204, 204, 201, 201, 201,
        188, 188, 176, 141, 66, 40, 193, 193, 193, 193, 193, 193, 193, 193, 193, 193, 193, 193, 194, 194,
        194, 184, 184, 173, 139, 65, 39, 204, 204, 204, 204, 204, 204, 204, 204, 201, 201, 201, 201, 198,
        198, 198, 187, 187, 175, 140, 66, 40,
    };

    /// <summary>log2(n) in 1/8 bits for the intensity / skip reservations (libopus LOG2_FRAC_TABLE).</summary>
    public static readonly byte[] Log2FracTable =
    {
        0, 8, 13, 16, 19, 21, 23, 24, 26, 27, 28, 29, 30, 31, 32, 32, 33, 34, 34, 35,
        36, 36, 37, 37,
    };

}

/// <summary>
/// The static 48 kHz CELT mode on the accelerator: every <see cref="CeltModeTables"/> table widened to int in one view
/// (<see cref="Data"/>), each at its offset. Pass by value to kernels.
/// </summary>
public struct CeltModeGpu
{
    /// <summary>All tables, widened to int.</summary>
    public ArrayView<int> Data;
    /// <summary>Offset of <see cref="CeltModeTables.EBands"/> in <see cref="Data"/>.</summary>
    public int EBandsAt;
    /// <summary>Offset of <see cref="CeltModeTables.AllocVectors"/> in <see cref="Data"/>.</summary>
    public int AllocVectorsAt;
    /// <summary>Offset of <see cref="CeltModeTables.LogN"/> in <see cref="Data"/>.</summary>
    public int LogNAt;
    /// <summary>Offset of <see cref="CeltModeTables.CacheIndex"/> in <see cref="Data"/>.</summary>
    public int CacheIndexAt;
    /// <summary>Offset of <see cref="CeltModeTables.CacheBits"/> in <see cref="Data"/>.</summary>
    public int CacheBitsAt;
    /// <summary>Offset of <see cref="CeltModeTables.CacheCaps"/> in <see cref="Data"/>.</summary>
    public int CacheCapsAt;
    /// <summary>Offset of <see cref="CeltModeTables.Log2FracTable"/> in <see cref="Data"/>.</summary>
    public int Log2FracTableAt;

    /// <summary>Upload the tables; the returned buffer backs <see cref="Data"/> (dispose it after the last kernel).</summary>
    public static (CeltModeGpu Mode, MemoryBuffer1D<int, Stride1D.Dense> Buffer) Upload(Accelerator accelerator)
    {
        var all = new List<int>();
        var mode = new CeltModeGpu();
        mode.EBandsAt = all.Count; foreach (var v in CeltModeTables.EBands) all.Add(v);
        mode.AllocVectorsAt = all.Count; foreach (var v in CeltModeTables.AllocVectors) all.Add(v);
        mode.LogNAt = all.Count; foreach (var v in CeltModeTables.LogN) all.Add(v);
        mode.CacheIndexAt = all.Count; foreach (var v in CeltModeTables.CacheIndex) all.Add(v);
        mode.CacheBitsAt = all.Count; foreach (var v in CeltModeTables.CacheBits) all.Add(v);
        mode.CacheCapsAt = all.Count; foreach (var v in CeltModeTables.CacheCaps) all.Add(v);
        mode.Log2FracTableAt = all.Count; foreach (var v in CeltModeTables.Log2FracTable) all.Add(v);
        var buffer = accelerator.Allocate1D(all.ToArray());
        mode.Data = buffer.View;
        return (mode, buffer);
    }

    /// <summary><see cref="CeltModeTables.EBands"/>[i].</summary>
    public int EBandsOf(int i) => Data[EBandsAt + i];
    /// <summary><see cref="CeltModeTables.AllocVectors"/>[i].</summary>
    public int AllocVectorsOf(int i) => Data[AllocVectorsAt + i];
    /// <summary><see cref="CeltModeTables.LogN"/>[i].</summary>
    public int LogNOf(int i) => Data[LogNAt + i];
    /// <summary><see cref="CeltModeTables.CacheIndex"/>[i].</summary>
    public int CacheIndexOf(int i) => Data[CacheIndexAt + i];
    /// <summary><see cref="CeltModeTables.CacheBits"/>[i].</summary>
    public int CacheBitsOf(int i) => Data[CacheBitsAt + i];
    /// <summary><see cref="CeltModeTables.CacheCaps"/>[i].</summary>
    public int CacheCapsOf(int i) => Data[CacheCapsAt + i];
    /// <summary><see cref="CeltModeTables.Log2FracTable"/>[i].</summary>
    public int Log2FracTableOf(int i) => Data[Log2FracTableAt + i];
}
