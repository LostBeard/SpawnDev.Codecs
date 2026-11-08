// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// ICCP / EXIF / XMP chunks around a lossy still, in the layout libwebp's WebPMuxAssemble writes for one image (what
// Pillow's still save produces when given icc_profile / exif / xmp): RIFF, VP8X (ICC 0x20, alpha 0x10, EXIF 0x08,
// XMP 0x04, canvas size), ICCP, [ALPH], VP8, EXIF, XMP. A still with none of them keeps the simple RIFF + VP8 file (or
// the encoder's own VP8X + ALPH + VP8 when it has transparency). The payloads are written verbatim: Pillow's still save
// strips an "Exif\0\0" prefix in Python before libwebp sees the bytes, so pass the stripped bytes for the same file.
// The chunk headers and the copy of the image chunks are kernels (WebPAnimKernels); the host uploads the caller's blobs.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Images.WebP.Lossy;

public sealed partial class WebPLossyEncoderGpu
{
    private const int FourccIccp = 0x50434349, FourccExif = 0x46495845, FourccXmp = 0x20504D58;

    private static bool HasMetadata(WebPLossyEncodeOptions o) => o.IccProfile is { Length: > 0 } || o.Exif is { Length: > 0 } || o.Xmp is { Length: > 0 };

    private static byte[]? Blob(byte[]? b) => b is { Length: > 0 } ? b : null;

    private static int StillFlags(bool hasAlpha, byte[]? icc, byte[]? exif, byte[]? xmp) =>
        (hasAlpha ? 0x10 : 0) | (icc != null ? 0x20 : 0) | (exif != null ? 0x08 : 0) | (xmp != null ? 0x04 : 0);

    /// <summary>Length of the still WebPMuxAssemble writes for one encoded image of <paramref name="fileLength"/> bytes.</summary>
    private static long StillFileLength(int fileLength, bool hasAlpha, byte[]? icc, byte[]? exif, byte[]? xmp)
    {
        if (StillFlags(hasAlpha, icc, exif, xmp) == 0) return fileLength;
        return 30 + ChunkLength(icc) + (fileLength - (hasAlpha ? 30 : 12)) + ChunkLength(exif) + ChunkLength(xmp);
    }

    /// <summary>Upload ICC, EXIF, XMP (in that order) into the blob buffer the chunk kernels read.</summary>
    private async Task UploadBlobsAsync(byte[]? icc, byte[]? exif, byte[]? xmp)
    {
        var a = _anim ??= new AnimBuffers(_accelerator);
        int total = (icc?.Length ?? 0) + (exif?.Length ?? 0) + (xmp?.Length ?? 0);
        var blobs = await a.Blobs.EnsureDrainedAsync(Math.Max(1, total));
        int at = 0;
        foreach (var b in new[] { icc, exif, xmp })
        {
            if (b == null) continue;
            DeviceTransfer.Upload(blobs.SubView(at, b.Length), b);
            at += b.Length;
        }
    }

    /// <summary>EXIF then XMP chunks at <paramref name="at"/>; their payloads follow the ICC one in the blob buffer.</summary>
    private void WriteMetadataTail(ArrayView1D<byte, Stride1D.Dense> o, int at, int blob, byte[]? exif, byte[]? xmp)
    {
        var a = _anim!;
        if (exif != null)
        {
            a.BlobChunk(ChunkLength(exif), o, a.Blobs.View, at, FourccExif, blob, exif.Length);
            at += ChunkLength(exif);
            blob += exif.Length;
        }
        if (xmp != null)
            a.BlobChunk(ChunkLength(xmp), o, a.Blobs.View, at, FourccXmp, blob, xmp.Length);
    }

    /// <summary>
    /// The still file of one encoded image (<paramref name="file"/>, a complete RIFF of <paramref name="fileLength"/>
    /// bytes) with the metadata chunks, into <paramref name="o"/>. The blobs must already be uploaded.
    /// </summary>
    private void WriteStillFile(ArrayView1D<byte, Stride1D.Dense> o, ArrayView1D<byte, Stride1D.Dense> file, int fileLength, bool hasAlpha,
        int width, int height, byte[]? icc, byte[]? exif, byte[]? xmp)
    {
        var a = _anim!;
        int flags = StillFlags(hasAlpha, icc, exif, xmp);
        if (flags == 0)
        {
            a.Copy(fileLength, o, file, 0, 0);   // the encoder's own file (RIFF + VP8)
            return;
        }
        long length = StillFileLength(fileLength, hasAlpha, icc, exif, xmp);
        a.FileHeader(30, o, (int)length - 8, flags, width, height);
        int p = 30, blob = 0;
        if (icc != null)
        {
            a.BlobChunk(ChunkLength(icc), o, a.Blobs.View, p, FourccIccp, blob, icc.Length);
            p += ChunkLength(icc);
            blob += icc.Length;
        }
        int skip = hasAlpha ? 30 : 12;
        a.Copy(fileLength - skip, o, file, p, skip);
        WriteMetadataTail(o, p + fileLength - skip, blob, exif, xmp);
    }

    /// <summary>
    /// The still file with its metadata chunks: the encoded image goes to a scratch buffer first, then the container
    /// is written around its chunks into <paramref name="ensureOutput"/>'s view. Returns the file length.
    /// </summary>
    private async Task<int> EncodeStillWithMetadataAsync(ArrayView1D<byte, Stride1D.Dense> rgba, int width, int height, WebPLossyEncodeOptions o,
        Func<long, ValueTask<ArrayView1D<byte, Stride1D.Dense>>> ensureOutput)
    {
        byte[]? icc = Blob(o.IccProfile), exif = Blob(o.Exif), xmp = Blob(o.Xmp);
        var (simpleLength, alphaLength) = await EncodeToScratchAsync(rgba, width, height, o);
        int fileLength = FileLength(simpleLength, alphaLength);
        bool hasAlpha = alphaLength >= 0;
        var a = _anim ??= new AnimBuffers(_accelerator);
        a.ResetPool();
        var scratch = a.RentFile();
        var fileView = await scratch.EnsureDrainedAsync(fileLength);
        Assemble(fileView, fileLength, alphaLength);
        await UploadBlobsAsync(icc, exif, xmp);
        long length = StillFileLength(fileLength, hasAlpha, icc, exif, xmp);
        if (length > int.MaxValue) throw new InvalidOperationException("The file is too large.");
        var output = await ensureOutput(length);
        WriteStillFile(output, fileView, fileLength, hasAlpha, width, height, icc, exif, xmp);
        await _accelerator.SynchronizeAsync();
        return (int)length;
    }
}
