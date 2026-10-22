// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// The PNG decode pipeline shares ONE small int buffer ("meta") between its kernels: the chunk-scan kernel
// fills the header facts, the host adds the derived layout (Adam7 pass table, sizes) after reading the few
// scalars it needs to size allocations, and the later kernels read it. Slot numbers live here so a kernel
// and the host cannot disagree about them.

namespace SpawnDev.Codecs.Images.Png;

/// <summary>Why a PNG could not be decoded (slot <see cref="PngMeta.Error"/>).</summary>
public enum PngError
{
    /// <summary>No error.</summary>
    None = 0,
    /// <summary>The 8-byte PNG signature is missing.</summary>
    BadSignature = 1,
    /// <summary>The file ends inside a chunk, or before IEND.</summary>
    Truncated = 2,
    /// <summary>The first chunk is not a 13-byte IHDR.</summary>
    BadHeader = 3,
    /// <summary>More chunks than the scan table holds.</summary>
    TooManyChunks = 4,
    /// <summary>No IDAT chunk.</summary>
    NoImageData = 5,
    /// <summary>A critical chunk has a bad CRC.</summary>
    BadCrc = 6,
    /// <summary>Unsupported colour type / bit depth / compression / filter / interlace value.</summary>
    Unsupported = 7,
    /// <summary>A palette image has no (or a too short) PLTE chunk.</summary>
    BadPalette = 8,
    /// <summary>A scanline has an invalid filter type.</summary>
    BadFilter = 9,
    /// <summary>The inflated data is shorter or longer than the image needs.</summary>
    BadImageDataSize = 10,
}

/// <summary>Slot indices of the shared PNG metadata buffer.</summary>
public static class PngMeta
{
    /// <summary>Total ints a metadata buffer must hold.</summary>
    public const int Size = 96;

    // ---- written by PngChunkScanKernel ----
    /// <summary><see cref="PngError"/> raised by the scan (0 = none).</summary>
    public const int Error = 0;
    /// <summary>Number of chunks recorded in the chunk table.</summary>
    public const int ChunkCount = 1;
    /// <summary>Total bytes of all IDAT payloads (the zlib stream length).</summary>
    public const int IdatTotal = 2;
    /// <summary>Number of IDAT chunks.</summary>
    public const int IdatCount = 3;
    /// <summary>IHDR width.</summary>
    public const int Width = 4;
    /// <summary>IHDR height.</summary>
    public const int Height = 5;
    /// <summary>IHDR bit depth.</summary>
    public const int BitDepth = 6;
    /// <summary>IHDR colour type.</summary>
    public const int ColorType = 7;
    /// <summary>IHDR interlace method.</summary>
    public const int Interlace = 8;
    /// <summary>IHDR compression method.</summary>
    public const int Compression = 9;
    /// <summary>IHDR filter method.</summary>
    public const int FilterMethod = 10;
    /// <summary>File offset of the PLTE payload (-1 = none).</summary>
    public const int PlteOffset = 11;
    /// <summary>PLTE payload length in bytes.</summary>
    public const int PlteLength = 12;
    /// <summary>File offset of the tRNS payload (-1 = none).</summary>
    public const int TrnsOffset = 13;
    /// <summary>tRNS payload length in bytes.</summary>
    public const int TrnsLength = 14;
    /// <summary>1 when IEND was seen.</summary>
    public const int IendSeen = 15;
    /// <summary>acTL num_frames (0 = not an APNG).</summary>
    public const int ApngFrames = 16;
    /// <summary>acTL num_plays.</summary>
    public const int ApngPlays = 17;
    /// <summary>1-based index of the first critical chunk with a bad CRC (0 = all good). Written by the CRC kernel.</summary>
    public const int BadCrcChunk = 18;
    /// <summary>1 when an sRGB chunk is present.</summary>
    public const int HasSrgb = 19;

    // ---- written by the host after it has read the scalars above ----
    /// <summary>Total bytes the inflated, filtered scanlines occupy (all Adam7 passes).</summary>
    public const int RawSize = 24;
    /// <summary>Bits per pixel of the file (channels * bitDepth).</summary>
    public const int BitsPerPixel = 25;
    /// <summary>Filter byte distance: max(1, bitsPerPixel / 8).</summary>
    public const int FilterBytesPerPixel = 26;
    /// <summary>Number of Adam7 passes with at least one pixel (1 when not interlaced).</summary>
    public const int PassCount = 27;
    /// <summary>1 when the tRNS chunk is a colour key (gray / RGB images).</summary>
    public const int HasColorKey = 28;
    /// <summary>Output <see cref="PixelFormat"/> (0 = Rgba8, 1 = Rgba16).</summary>
    public const int OutputFormat = 30;

    /// <summary>First slot the host writes after the scan (everything from here up to <see cref="Size"/>).</summary>
    public const int HostWrittenStart = 24;

    /// <summary>First slot of the pass table: per pass 8 ints (see <see cref="PassOffset"/> .. <see cref="PassYStep"/>).</summary>
    public const int PassTable = 40;
    /// <summary>Ints per pass-table entry.</summary>
    public const int PassStride = 8;
    /// <summary>Entry field: byte offset of the pass's first scanline in the inflated buffer.</summary>
    public const int PassOffset = 0;
    /// <summary>Entry field: pass width in pixels.</summary>
    public const int PassWidth = 1;
    /// <summary>Entry field: pass height in pixels.</summary>
    public const int PassHeight = 2;
    /// <summary>Entry field: bytes per scanline of the pass (excluding the filter byte).</summary>
    public const int PassRowBytes = 3;
    /// <summary>Entry field: x of the pass's first pixel in the image.</summary>
    public const int PassXStart = 4;
    /// <summary>Entry field: y of the pass's first pixel in the image.</summary>
    public const int PassYStart = 5;
    /// <summary>Entry field: x step.</summary>
    public const int PassXStep = 6;
    /// <summary>Entry field: y step.</summary>
    public const int PassYStep = 7;
}
