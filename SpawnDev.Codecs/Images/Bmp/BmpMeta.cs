// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// The BMP decode pipeline shares ONE small int buffer ("meta") between its kernels, like the PNG decoder: the
// header-scan kernel fills the raw header facts, the host validates them with 64-bit arithmetic and adds the
// derived layout (row stride, channel shifts / widths, alpha mode), and the pixel kernels read both.

namespace SpawnDev.Codecs.Images.Bmp;

/// <summary>Why a BMP could not be decoded (slot <see cref="BmpMeta.Error"/>).</summary>
public enum BmpError
{
    /// <summary>No error.</summary>
    None = 0,
    /// <summary>The file does not start with "BM".</summary>
    BadSignature = 1,
    /// <summary>The file ends inside the file header, the DIB header, the channel masks or the palette.</summary>
    TruncatedHeader = 2,
    /// <summary>The DIB header size is not one of the supported layouts.</summary>
    UnsupportedHeader = 3,
    /// <summary>An RLE stream ended in the middle of an instruction.</summary>
    RleTruncated = 4,
    /// <summary>An RLE stream draws outside the image.</summary>
    RleOutOfBounds = 5,
}

/// <summary>Slot indices of the shared BMP metadata buffer.</summary>
public static class BmpMeta
{
    /// <summary>Total ints a metadata buffer must hold.</summary>
    public const int Size = 48;

    // ---- written by the scan kernel (raw facts from the headers) ----
    /// <summary><see cref="BmpError"/> raised by the scan or the RLE kernel (0 = none).</summary>
    public const int Error = 0;
    /// <summary>BITMAPFILEHEADER.bfOffBits: file offset of the pixel data.</summary>
    public const int OffBits = 1;
    /// <summary>DIB header size in bytes (12, 40, 52, 56, 108 or 124).</summary>
    public const int HeaderSize = 2;
    /// <summary>Width as stored (BITMAPCOREHEADER: unsigned 16 bit).</summary>
    public const int Width = 3;
    /// <summary>Height as stored: negative = top-down.</summary>
    public const int Height = 4;
    /// <summary>Colour planes (must be 1).</summary>
    public const int Planes = 5;
    /// <summary>Bits per pixel.</summary>
    public const int Bpp = 6;
    /// <summary>Compression: 0 RGB, 1 RLE8, 2 RLE4, 3 BITFIELDS, 4 JPEG, 5 PNG, 6 ALPHABITFIELDS.</summary>
    public const int Compression = 7;
    /// <summary>biSizeImage (0 for BI_RGB is legal).</summary>
    public const int ImageSize = 8;
    /// <summary>biClrUsed.</summary>
    public const int ClrUsed = 9;
    /// <summary>File offset where the palette starts.</summary>
    public const int PaletteOffset = 10;
    /// <summary>Red channel mask (bit pattern of the uint), from the file when present.</summary>
    public const int MaskR = 11;
    /// <summary>Green channel mask.</summary>
    public const int MaskG = 12;
    /// <summary>Blue channel mask.</summary>
    public const int MaskB = 13;
    /// <summary>Alpha channel mask.</summary>
    public const int MaskA = 14;
    /// <summary>File length in bytes.</summary>
    public const int FileLength = 15;
    /// <summary>1 when the file carries channel masks (BITFIELDS, or a V2+ header).</summary>
    public const int FileHasMasks = 16;

    // ---- written by the host after it has validated the facts above ----
    /// <summary>First slot the host writes (everything from here up to <see cref="Size"/>).</summary>
    public const int HostWrittenStart = 24;
    /// <summary>Bytes per pixel row in the file (4-byte aligned). Unused for RLE.</summary>
    public const int Stride = 24;
    /// <summary>1 when rows are stored top to bottom.</summary>
    public const int TopDown = 25;
    /// <summary>Palette entries usable (0 for bpp above 8).</summary>
    public const int PaletteCount = 26;
    /// <summary>Bytes per palette entry (3 for BITMAPCOREHEADER, else 4).</summary>
    public const int EntrySize = 27;
    /// <summary>Red channel right shift.</summary>
    public const int ShiftR = 28;
    /// <summary>Red channel width in bits.</summary>
    public const int BitsR = 29;
    /// <summary>Green channel right shift.</summary>
    public const int ShiftG = 30;
    /// <summary>Green channel width in bits.</summary>
    public const int BitsG = 31;
    /// <summary>Blue channel right shift.</summary>
    public const int ShiftB = 32;
    /// <summary>Blue channel width in bits.</summary>
    public const int BitsB = 33;
    /// <summary>Alpha channel right shift.</summary>
    public const int ShiftA = 34;
    /// <summary>Alpha channel width in bits (0 = no alpha channel).</summary>
    public const int BitsA = 35;
    /// <summary>1 when the image has an alpha channel (explicit mask, or the 4th byte of 32-bit BI_RGB).</summary>
    public const int HasAlphaChannel = 36;
    /// <summary>Written by the alpha-scan kernel: 1 when any pixel has a non-zero alpha value.</summary>
    public const int AlphaSeen = 37;
    /// <summary>Absolute image height.</summary>
    public const int AbsHeight = 38;
    /// <summary>Effective red mask: the file's, or the BI_RGB default (0x7C00 for 16 bit, 0x00FF0000 for 32 bit).</summary>
    public const int EffMaskR = 39;
    /// <summary>Effective green mask.</summary>
    public const int EffMaskG = 40;
    /// <summary>Effective blue mask.</summary>
    public const int EffMaskB = 41;
    /// <summary>Effective alpha mask (0 = none).</summary>
    public const int EffMaskA = 42;
}
