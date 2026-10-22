// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// The WebP decode pipeline shares ONE small int buffer ("meta") between its kernels, the same shape as
// PngMeta: the RIFF scan kernel fills the container facts, the VP8L decode kernel fills the bitstream facts
// (transform list, pool use), and the host reads the few scalars it needs to size buffers and dispatch the
// inverse transforms. Slot numbers live here so a kernel and the host cannot disagree about them.

namespace SpawnDev.Codecs.Images.WebP;

/// <summary>Why a WebP could not be decoded (slots <see cref="WebPMeta.Error"/> / <see cref="WebPMeta.DecodeError"/>).</summary>
public enum WebPError
{
    /// <summary>No error.</summary>
    None = 0,
    /// <summary>The RIFF / WEBP signature is missing.</summary>
    NotWebP = 1,
    /// <summary>The RIFF size or a chunk size runs past the end of the file.</summary>
    Truncated = 2,
    /// <summary>A chunk is malformed (wrong size, unexpected first chunk).</summary>
    BadChunk = 3,
    /// <summary>The image data is a lossy VP8 chunk (not supported yet).</summary>
    LossyUnsupported = 4,
    /// <summary>The VP8L header is invalid (signature, version).</summary>
    BadHeader = 5,
    /// <summary>No VP8L / VP8 image data (or an animation without frames).</summary>
    NoImageData = 6,
    /// <summary>The VP8X canvas size differs from the image size, or a frame is outside the canvas.</summary>
    CanvasMismatch = 7,
    /// <summary>More ANMF frames than the frame table holds.</summary>
    TooManyFrames = 8,
    /// <summary>The canvas or image has more pixels than 2^31 - 1.</summary>
    TooLarge = 9,
    /// <summary>A transform appears twice.</summary>
    BadTransform = 10,
    /// <summary>The colour cache size is outside 1..11.</summary>
    BadCacheBits = 11,
    /// <summary>A prefix (Huffman) code is empty, over-subscribed, incomplete or a symbol did not decode.</summary>
    BadPrefixCode = 12,
    /// <summary>A backward reference reaches before the first pixel or past the last.</summary>
    BadBackReference = 13,
    /// <summary>The bitstream ended before the image was complete.</summary>
    TruncatedBitstream = 14,
    /// <summary>A colour cache index outside the cache.</summary>
    BadCacheIndex = 15,
    /// <summary>The prefix-code pool was too small; the host grows it to <see cref="WebPMeta.PoolNeeded"/> and decodes again.</summary>
    PoolTooSmall = 16,
    /// <summary>A code length repeat runs past the alphabet or max_symbol exceeds it.</summary>
    BadCodeLengths = 17,
}

/// <summary>Slot indices of the shared WebP metadata buffer.</summary>
public static class WebPMeta
{
    /// <summary>Total ints a metadata buffer must hold.</summary>
    public const int Size = 128;

    // ---- written by WebPScanKernel ----
    /// <summary><see cref="WebPError"/> raised by the scan (0 = none).</summary>
    public const int Error = 0;
    /// <summary>Effective end of the RIFF data (min of RIFF size + 8 and the file length).</summary>
    public const int FileEnd = 1;
    /// <summary>VP8X flag byte (0 for a simple file).</summary>
    public const int Flags = 2;
    /// <summary>1 when a VP8X chunk is present.</summary>
    public const int HasVp8x = 3;
    /// <summary>VP8X canvas width (or the image width of a simple file).</summary>
    public const int CanvasWidth = 4;
    /// <summary>VP8X canvas height (or the image height of a simple file).</summary>
    public const int CanvasHeight = 5;
    /// <summary>1 when the VP8X animation flag is set.</summary>
    public const int IsAnimated = 6;
    /// <summary>Number of ANMF frames recorded in the frame table (1 for a still).</summary>
    public const int FrameCount = 7;
    /// <summary>ANIM loop count (0 = forever).</summary>
    public const int LoopCount = 8;
    /// <summary>ANIM background colour as stored (little-endian B, G, R, A bytes read as a uint).</summary>
    public const int BackgroundColor = 9;
    /// <summary>Still image data kind: 0 none, 1 VP8L, 2 VP8, 3 ALPH + VP8.</summary>
    public const int ImageKind = 10;
    /// <summary>File offset of the still image's VP8L / VP8 payload.</summary>
    public const int ImageOffset = 11;
    /// <summary>Length of the still image's payload.</summary>
    public const int ImageLength = 12;
    /// <summary>Width of the still image (the first frame for an animation).</summary>
    public const int Width = 13;
    /// <summary>Height of the still image (the first frame for an animation).</summary>
    public const int Height = 14;
    /// <summary>The VP8L header's alpha_is_used bit of the still image.</summary>
    public const int AlphaBit = 15;
    /// <summary>File offset of the ICCP payload (-1 = none).</summary>
    public const int IccOffset = 16;
    /// <summary>ICCP payload length.</summary>
    public const int IccLength = 17;
    /// <summary>File offset of the EXIF payload (-1 = none).</summary>
    public const int ExifOffset = 18;
    /// <summary>EXIF payload length.</summary>
    public const int ExifLength = 19;
    /// <summary>File offset of the XMP payload (-1 = none).</summary>
    public const int XmpOffset = 20;
    /// <summary>XMP payload length.</summary>
    public const int XmpLength = 21;
    /// <summary>EXIF orientation 1..8 (1 when absent).</summary>
    public const int Orientation = 22;
    /// <summary>1 when an ALPH chunk was seen next to a still VP8 image.</summary>
    public const int HasAlph = 23;
    /// <summary>Number of top-level chunks walked.</summary>
    public const int ChunkCount = 24;
    /// <summary>Final HasAlpha fact (libwebp semantics: VP8L bit for lossless stills, VP8X flag / ALPH otherwise).</summary>
    public const int HasAlpha = 25;

    // ---- written by Vp8lDecodeKernel ----
    /// <summary><see cref="WebPError"/> raised by the VP8L decode (0 = none).</summary>
    public const int DecodeError = 32;
    /// <summary>Number of transforms read.</summary>
    public const int TransformCount = 33;
    /// <summary>Ints the prefix-code pool needs (valid with <see cref="WebPError.PoolTooSmall"/>).</summary>
    public const int PoolNeeded = 34;
    /// <summary>Pool ints in use right now (decode scratch).</summary>
    public const int PoolUsed = 35;
    /// <summary>Colour cache bits of the main image.</summary>
    public const int CacheBits = 36;
    /// <summary>Meta prefix bits (0 = single prefix group).</summary>
    public const int MetaBits = 37;
    /// <summary>Number of prefix groups declared by the entropy image.</summary>
    public const int Groups = 38;
    /// <summary>Number of prefix groups actually referenced.</summary>
    public const int GroupsUsed = 39;
    /// <summary>Width of the decoded main image (before the inverse transforms).</summary>
    public const int PackedWidth = 40;
    /// <summary>Bit position reached, for diagnostics.</summary>
    public const int BitsConsumed = 41;

    /// <summary>First transform record; <see cref="TransformStride"/> ints each.</summary>
    public const int Transforms = 48;
    /// <summary>Ints per transform record.</summary>
    public const int TransformStride = 8;
    /// <summary>Record field: 0 predictor, 1 cross colour, 2 subtract green, 3 colour indexing.</summary>
    public const int TrKind = 0;
    /// <summary>Record field: block size bits (predictor / cross colour) or pixel bundling width bits (colour indexing).</summary>
    public const int TrBits = 1;
    /// <summary>Record field: int offset of the transform data in the work buffer.</summary>
    public const int TrData = 2;
    /// <summary>Record field: image width in effect when the transform was read.</summary>
    public const int TrWidth = 3;
    /// <summary>Record field: palette size (colour indexing).</summary>
    public const int TrCount = 4;

    /// <summary>Transform kinds (spec numbering).</summary>
    public const int KindPredictor = 0, KindCrossColor = 1, KindSubtractGreen = 2, KindColorIndexing = 3;

    // ---- frame table (written by the scan kernel), WebPFrameTable ints per ANMF frame ----
    /// <summary>Ints per frame record.</summary>
    public const int FrameStride = 12;
    /// <summary>Frame field: x offset on the canvas.</summary>
    public const int FrX = 0;
    /// <summary>Frame field: y offset.</summary>
    public const int FrY = 1;
    /// <summary>Frame field: width.</summary>
    public const int FrW = 2;
    /// <summary>Frame field: height.</summary>
    public const int FrH = 3;
    /// <summary>Frame field: duration in ms.</summary>
    public const int FrDuration = 4;
    /// <summary>Frame field: ANMF flag byte (bit 0 dispose to background, bit 1 do not blend).</summary>
    public const int FrFlags = 5;
    /// <summary>Frame field: payload kind (1 VP8L, 2 VP8, 3 ALPH + VP8).</summary>
    public const int FrKind = 6;
    /// <summary>Frame field: payload file offset.</summary>
    public const int FrOffset = 7;
    /// <summary>Frame field: payload length.</summary>
    public const int FrLength = 8;
    /// <summary>Frame field: 1 when the frame has alpha (VP8L alpha_is_used or an ALPH chunk).</summary>
    public const int FrHasAlpha = 9;
}
