// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// The GIF decode pipeline shares ONE int buffer ("meta") between its kernels: a small header block followed by
// the frame table. The scan kernels fill both (the header block also holds the resumable walk state); the host
// reads them back (a few dozen ints plus 16 ints per frame), validates them with 64-bit arithmetic, and the LZW /
// compositing kernels read them.

namespace SpawnDev.Codecs.Images.Gif;

/// <summary>Why a GIF could not be decoded (slot <see cref="GifMeta.Error"/>).</summary>
public enum GifError
{
    /// <summary>No error.</summary>
    None = 0,
    /// <summary>The file does not start with GIF87a / GIF89a.</summary>
    BadSignature = 1,
    /// <summary>The file ends inside a header, colour table, extension or image block.</summary>
    Truncated = 2,
    /// <summary>A block starts with a byte that is not an extension (0x21), image (0x2C) or trailer (0x3B) introducer.</summary>
    BadBlock = 3,
    /// <summary>More frames than <see cref="GifMeta.MaxFrames"/>.</summary>
    TooManyFrames = 4,
}

/// <summary>Slot indices of the shared GIF metadata buffer.</summary>
public static class GifMeta
{
    /// <summary>Largest number of frames a file may hold.</summary>
    public const int MaxFrames = 16384;

    /// <summary>Ints in the header block (the frame table starts right after).</summary>
    public const int Size = 32;

    /// <summary><see cref="GifError"/> raised by the scan (0 = none).</summary>
    public const int Error = 0;
    /// <summary>Logical screen width.</summary>
    public const int Width = 1;
    /// <summary>Logical screen height.</summary>
    public const int Height = 2;
    /// <summary>File offset of the global colour table (-1 = none).</summary>
    public const int GlobalPaletteOffset = 3;
    /// <summary>Entries in the global colour table.</summary>
    public const int GlobalPaletteCount = 4;
    /// <summary>Logical screen descriptor background colour index.</summary>
    public const int BackgroundIndex = 5;
    /// <summary>Number of frames found.</summary>
    public const int FrameCount = 6;
    /// <summary>Netscape loop count (0 = forever), or -1 when no loop extension precedes the first image.</summary>
    public const int LoopCount = 7;
    /// <summary>File length in bytes.</summary>
    public const int FileLength = 8;
    /// <summary>File offset of the byte that made the scan fail.</summary>
    public const int ErrorOffset = 9;

    // ---- block walk state of the resumable scan (GifKernels.ScanStepKernel); ScanDone = 1 once the walk ended ----
    internal const int ScanPhase = 16;
    internal const int ScanPos = 17;
    internal const int ScanChainPos = 18;
    internal const int ScanPayload = 19;
    internal const int ScanHasGce = 20;
    internal const int ScanGceDisposal = 21;
    internal const int ScanGceTransFlag = 22;
    internal const int ScanGceTransIndex = 23;
    internal const int ScanGceDelay = 24;
    internal const int ScanFrames = 25;
    internal const int ScanErr = 26;
    internal const int ScanDone = 27;

    /// <summary>First slot of the frame table.</summary>
    public const int FrameTable = Size;
    /// <summary>Ints per frame entry.</summary>
    public const int FrameStride = 16;

    // ---- frame entry fields (relative to FrameTable + index * FrameStride) ----
    /// <summary>Left edge on the canvas.</summary>
    public const int FrameX = 0;
    /// <summary>Top edge on the canvas.</summary>
    public const int FrameY = 1;
    /// <summary>Frame width.</summary>
    public const int FrameW = 2;
    /// <summary>Frame height.</summary>
    public const int FrameH = 3;
    /// <summary>Bit 0: interlaced. Bit 1: has a transparent colour index.</summary>
    public const int FrameFlags = 4;
    /// <summary>Transparent colour index (valid when flag bit 1 is set).</summary>
    public const int FrameTransparentIndex = 5;
    /// <summary>Delay in hundredths of a second.</summary>
    public const int FrameDelayCs = 6;
    /// <summary>Disposal method as stored (0..7).</summary>
    public const int FrameDisposal = 7;
    /// <summary>File offset of the colour table in force (local, else global; -1 = none).</summary>
    public const int FramePaletteOffset = 8;
    /// <summary>Entries in that colour table.</summary>
    public const int FramePaletteCount = 9;
    /// <summary>LZW minimum code size.</summary>
    public const int FrameMinCodeSize = 10;
    /// <summary>File offset of the first data sub-block's length byte.</summary>
    public const int FrameDataOffset = 11;
    /// <summary>1 when the colour table in force is the identity grayscale ramp (entry i = i, i, i for every i).</summary>
    public const int FramePaletteIdentity = 12;
    /// <summary>Payload bytes of the frame's LZW data sub-blocks (bounds how many pixels the frame can describe).</summary>
    public const int FrameDataBytes = 13;

    /// <summary>Total ints a metadata buffer must hold.</summary>
    public const int BufferInts = FrameTable + MaxFrames * FrameStride;
}
