// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// The JPEG decode pipeline shares ONE small int buffer ("meta") between its kernels, exactly like the PNG
// pipeline (PngMeta): the marker-scan kernel fills the header facts, the host derives the layout (block
// grids, plane offsets, upsampling modes) from the few scalars it reads back and writes it into the
// host-written region, and every later kernel reads it. Slot numbers live here so a kernel and the host
// cannot disagree about them.

namespace SpawnDev.Codecs.Images.Jpeg;

/// <summary>Why a JPEG could not be decoded (slot <see cref="JpegMeta.Error"/>).</summary>
public enum JpegError
{
    /// <summary>No error.</summary>
    None = 0,
    /// <summary>The file does not start with the SOI marker.</summary>
    NotJpeg = 1,
    /// <summary>The file ends inside a marker segment (headers cut off).</summary>
    TruncatedHeaders = 2,
    /// <summary>A byte that should start a marker does not, or a marker is out of place.</summary>
    BadMarker = 3,
    /// <summary>A scan appears before any frame header, or there is no frame header at all.</summary>
    NoFrame = 4,
    /// <summary>A frame type the decoder does not implement (arithmetic, lossless, hierarchical); detail = marker.</summary>
    UnsupportedFrame = 5,
    /// <summary>A sample precision other than 8 bits; detail = the precision.</summary>
    UnsupportedPrecision = 6,
    /// <summary>The frame header is invalid (zero size, bad component count or sampling factors, duplicate ids).</summary>
    BadFrameHeader = 7,
    /// <summary>A quantization table segment is malformed.</summary>
    BadQuantTable = 8,
    /// <summary>A Huffman table segment is malformed or describes an impossible code.</summary>
    BadHuffmanTable = 9,
    /// <summary>A scan header is invalid (unknown component, bad spectral / approximation parameters).</summary>
    BadScanHeader = 10,
    /// <summary>The file has a frame header but no scan.</summary>
    NoImageData = 11,
    /// <summary>The entropy-coded data ends before the image is complete.</summary>
    EntropyTruncated = 12,
    /// <summary>The entropy-coded data contains a bit pattern that is not a code of the table.</summary>
    BadHuffmanCode = 13,
    /// <summary>A zero run walks past the last coefficient of a block.</summary>
    BadCoefficientRun = 14,
    /// <summary>Restart markers are missing, extra or out of sequence; detail = expected / found.</summary>
    BadRestart = 15,
    /// <summary>A scan uses a Huffman table that was never defined; detail = table slot.</summary>
    MissingHuffmanTable = 16,
    /// <summary>A component uses a quantization table that was never defined; detail = table id.</summary>
    MissingQuantTable = 17,
    /// <summary>A component sampling ratio that is not an integer multiple (libjpeg: fractional sampling).</summary>
    FractionalSampling = 18,
    /// <summary>The image needs more memory than the decoder can address.</summary>
    TooLarge = 19,
    /// <summary>More scans or Huffman table sets than the decoder tables hold.</summary>
    TooManyScans = 20,
    /// <summary>A progressive file ends without an EOI marker, so later scans may be missing.</summary>
    MissingEoi = 21,
    /// <summary>A component was never completely covered by the scans.</summary>
    IncompleteScans = 22,
}

/// <summary>Slot indices of the shared JPEG metadata buffer.</summary>
public static class JpegMeta
{
    /// <summary>Total ints a metadata buffer must hold.</summary>
    public const int Size = 4096;

    /// <summary>Maximum number of scans the scan table holds.</summary>
    public const int MaxScans = 128;

    /// <summary>Maximum number of distinct Huffman table sets (a set is recorded when a scan starts after a DHT).</summary>
    public const int MaxHuffSets = 64;

    // ---- written by the marker-scan kernel -----------------------------------------------------------------------
    /// <summary><see cref="JpegError"/> (0 = none). Later kernels may overwrite it with a decode error.</summary>
    public const int Error = 0;
    /// <summary>Extra detail for the error (marker, precision, table slot, ...).</summary>
    public const int ErrorDetail = 1;
    /// <summary>Frame width.</summary>
    public const int Width = 2;
    /// <summary>Frame height.</summary>
    public const int Height = 3;
    /// <summary>Sample precision (always 8 for a decodable file).</summary>
    public const int Precision = 4;
    /// <summary>Number of components (1, 3 or 4).</summary>
    public const int NumComps = 5;
    /// <summary>0 baseline (SOF0), 1 extended sequential (SOF1), 2 progressive (SOF2).</summary>
    public const int SofType = 6;
    /// <summary>1 when a JFIF APP0 segment was seen.</summary>
    public const int JfifSeen = 7;
    /// <summary>1 when an Adobe APP14 segment was seen.</summary>
    public const int AdobeSeen = 8;
    /// <summary>Adobe transform byte (valid when <see cref="AdobeSeen"/> is 1).</summary>
    public const int AdobeTransform = 9;
    /// <summary>EXIF orientation 1..8 (1 when absent or invalid).</summary>
    public const int Orientation = 10;
    /// <summary>Number of scans recorded in the scan table.</summary>
    public const int ScanCount = 11;
    /// <summary>1 when the EOI marker was reached.</summary>
    public const int EoiSeen = 12;
    /// <summary>Largest horizontal sampling factor.</summary>
    public const int MaxH = 13;
    /// <summary>Largest vertical sampling factor.</summary>
    public const int MaxV = 14;
    /// <summary>1 once a frame header was parsed.</summary>
    public const int SofSeen = 15;
    /// <summary>Current restart interval in MCUs (from the latest DRI).</summary>
    public const int RestartInterval = 16;
    /// <summary>Number of committed Huffman table sets.</summary>
    public const int HuffSetCount = 17;
    /// <summary>1 while the working Huffman tables differ from the last committed set.</summary>
    public const int HuffDirty = 18;
    /// <summary>First component id seen in the frame header (used by the colour space rules).</summary>
    public const int FirstCompId = 19;
    /// <summary>Resumable marker walk (ScanMarkersKernel): the position to continue from, the phase (0 = start, 1 = running,
    /// 2 = finished), whether the EXIF orientation was found.</summary>
    public const int ScanPos = 20, ScanPhase = 21, ScanExifDone = 22;

    // ---- host-written -----------------------------------------------------------------------------------------------
    /// <summary>First slot the host writes after the scan.</summary>
    public const int HostWrittenStart = 112;
    /// <summary>Output <see cref="PixelFormat"/> (0 = Rgba8, 1 = Rgba16).</summary>
    public const int OutputFormat = 112;
    /// <summary>Colour transform: see <see cref="ColorGray"/> .. <see cref="ColorYcck"/>.</summary>
    public const int ColorSpace = 113;
    /// <summary>Total number of 8x8 blocks over all components (padded grids).</summary>
    public const int TotalBlocks = 114;
    /// <summary>Image width as laid out in the output (after optional orientation).</summary>
    public const int OutWidth = 115;
    /// <summary>Image height as laid out in the output.</summary>
    public const int OutHeight = 116;
    /// <summary>EXIF orientation the orientation kernel applies (1 = none).</summary>
    public const int ApplyOrientation = 117;

    /// <summary>Colour transform: single component.</summary>
    public const int ColorGray = 0;
    /// <summary>Colour transform: YCbCr to RGB.</summary>
    public const int ColorYcc = 1;
    /// <summary>Colour transform: components are already RGB.</summary>
    public const int ColorRgb = 2;
    /// <summary>Colour transform: four components, CMYK (Adobe-inverted) to RGB.</summary>
    public const int ColorCmyk = 3;
    /// <summary>Colour transform: four components, YCCK to CMYK to RGB.</summary>
    public const int ColorYcck = 4;

    // ---- components --------------------------------------------------------------------------------------------------
    /// <summary>First slot of the frame component table written by the scan kernel (<see cref="CompInStride"/> ints each).</summary>
    public const int CompIn = 32;
    /// <summary>Ints per scan-kernel component entry.</summary>
    public const int CompInStride = 8;
    /// <summary>Entry field: component id.</summary>
    public const int CiId = 0;
    /// <summary>Entry field: horizontal sampling factor.</summary>
    public const int CiH = 1;
    /// <summary>Entry field: vertical sampling factor.</summary>
    public const int CiV = 2;
    /// <summary>Entry field: quantization table selector.</summary>
    public const int CiTq = 3;
    /// <summary>Entry field: 1 once the quantization table was latched at the component's first scan.</summary>
    public const int CiLatched = 4;

    /// <summary>First slot of the per-component layout the host writes (<see cref="CompLayoutStride"/> ints each).</summary>
    public const int CompLayout = 128;
    /// <summary>Ints per layout entry.</summary>
    public const int CompLayoutStride = 16;
    /// <summary>Layout field: byte offset of the component's sample plane.</summary>
    public const int ClPlaneOffset = 0;
    /// <summary>Layout field: plane row stride in bytes (blocks per line * 8).</summary>
    public const int ClPlaneStride = 1;
    /// <summary>Layout field: downsampled width (libjpeg downsampled_width).</summary>
    public const int ClDownW = 2;
    /// <summary>Layout field: downsampled height.</summary>
    public const int ClDownH = 3;
    /// <summary>Layout field: upsampling mode, see <see cref="UpFull"/> .. <see cref="UpReplicate"/>.</summary>
    public const int ClUpMode = 4;
    /// <summary>Layout field: horizontal replication factor (maxH / h).</summary>
    public const int ClHr = 5;
    /// <summary>Layout field: vertical replication factor (maxV / v).</summary>
    public const int ClVr = 6;
    /// <summary>Layout field: index of the component's first block in the global block numbering.</summary>
    public const int ClBlockPrefix = 7;
    /// <summary>Layout field: blocks per line in the padded grid.</summary>
    public const int ClBlocksPerLine = 8;
    /// <summary>Layout field: block rows in the padded grid.</summary>
    public const int ClBlockRows = 9;
    /// <summary>Layout field: element offset of the component's coefficients.</summary>
    public const int ClCoefOffset = 10;
    /// <summary>Layout field: blocks per line that carry image data (ceil(downW / 8)).</summary>
    public const int ClWidthInBlocks = 11;
    /// <summary>Layout field: block rows that carry image data.</summary>
    public const int ClHeightInBlocks = 12;
    /// <summary>Layout field: horizontal sampling factor.</summary>
    public const int ClH = 13;
    /// <summary>Layout field: vertical sampling factor.</summary>
    public const int ClV = 14;

    /// <summary>Upsampling: component is full size.</summary>
    public const int UpFull = 0;
    /// <summary>Upsampling: libjpeg h2v1 fancy (triangle) filter.</summary>
    public const int UpH2V1 = 1;
    /// <summary>Upsampling: libjpeg h1v2 fancy filter.</summary>
    public const int UpH1V2 = 2;
    /// <summary>Upsampling: libjpeg h2v2 fancy filter.</summary>
    public const int UpH2V2 = 3;
    /// <summary>Upsampling: pixel replication by integral factors.</summary>
    public const int UpReplicate = 4;

    // ---- scan table --------------------------------------------------------------------------------------------------
    /// <summary>First slot of the scan table (<see cref="ScanStride"/> ints per scan).</summary>
    public const int Scans = 256;
    /// <summary>Ints per scan entry.</summary>
    public const int ScanStride = 24;
    /// <summary>Scan field: number of components in the scan.</summary>
    public const int ScNs = 0;
    /// <summary>Scan field: first of four frame-component indices (slots 1..4).</summary>
    public const int ScComp = 1;
    /// <summary>Scan field: first of four packed (dcTable &lt;&lt; 4 | acTable) selectors (slots 5..8).</summary>
    public const int ScTables = 5;
    /// <summary>Scan field: spectral selection start.</summary>
    public const int ScSs = 9;
    /// <summary>Scan field: spectral selection end.</summary>
    public const int ScSe = 10;
    /// <summary>Scan field: successive approximation high bit.</summary>
    public const int ScAh = 11;
    /// <summary>Scan field: successive approximation low bit.</summary>
    public const int ScAl = 12;
    /// <summary>Scan field: file offset of the first entropy-coded byte.</summary>
    public const int ScEcsStart = 13;
    /// <summary>Scan field: file offset just past the last entropy-coded byte (the terminating marker).</summary>
    public const int ScEcsEnd = 14;
    /// <summary>Scan field: restart interval in MCUs (0 = none).</summary>
    public const int ScRestartInterval = 15;
    /// <summary>Scan field: Huffman table set index.</summary>
    public const int ScHuffSet = 16;
    /// <summary>Scan field: number of RSTn markers inside the entropy-coded data.</summary>
    public const int ScRstCount = 17;

    // ---- latched quantization tables (natural order, one per frame component) -----------------------------------------
    /// <summary>First slot of the latched quantization tables: 64 ints per frame component.</summary>
    public const int Quant = Scans + MaxScans * ScanStride;       // 3328
    /// <summary>Working quantization tables as DQT defines them (natural order, 4 x 64 ints).</summary>
    public const int QuantWork = Quant + 4 * 64;                   // 3584
    /// <summary>Four flags: quantization table slot defined.</summary>
    public const int QuantDefined = QuantWork + 4 * 64;            // 3840

    /// <summary>Ints per Huffman table in the raw (DHT as written) form: defined flag, 16 counts, 256 symbols.</summary>
    public const int RawTableStride = 288;
    /// <summary>Ints per Huffman table set: 8 tables (DC 0..3 then AC 0..3).</summary>
    public const int RawSetStride = 8 * RawTableStride;

    /// <summary>Ints per derived Huffman table: 512 lookahead, 18 maxcode, 18 valoffset, 256 symbols, flags.</summary>
    public const int HuffStride = 1024;
    /// <summary>Derived table field: 9-bit lookahead (code length &lt;&lt; 8 | symbol), 0 when the code is longer.</summary>
    public const int HuffLookup = 0;
    /// <summary>Derived table field: largest code of each length (index = length), -1 when none.</summary>
    public const int HuffMaxCode = 512;
    /// <summary>Derived table field: symbol index minus first code of each length.</summary>
    public const int HuffValOffset = 530;
    /// <summary>Derived table field: symbols in code order.</summary>
    public const int HuffSymbols = 548;
    /// <summary>Derived table field: 1 when the table is defined.</summary>
    public const int HuffDefined = 1023;

    /// <summary>Ints of per-segment decoder state saved between chunked decode dispatches.</summary>
    public const int SegStateStride = 16;
}
