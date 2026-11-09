// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// ICCP / EXIF / XMP chunks around a lossy still, in the layout libwebp's WebPMuxAssemble writes for one image (what
// Pillow's still save produces when given icc_profile / exif / xmp): RIFF, VP8X (ICC 0x20, alpha 0x10, EXIF 0x08,
// XMP 0x04, canvas size), ICCP, [ALPH], VP8, EXIF, XMP. A still with none of them keeps the encoder's own file. The
// payloads are written verbatim: Pillow's still save strips an "Exif\0\0" prefix in Python before libwebp sees the
// bytes, so pass the stripped bytes for the same file. Chunk headers and the copy of the image chunks are kernels.

using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.Codecs.Images.WebP.Lossy;

public sealed partial class WebPLossyEncoderGpu
{
    private static bool HasMetadata(WebPLossyEncodeOptions o) => o.IccProfile is { Length: > 0 } || o.Exif is { Length: > 0 } || o.Xmp is { Length: > 0 };

    private static byte[]? Blob(byte[]? b) => b is { Length: > 0 } ? b : null;

    /// <summary>
    /// The still file with its metadata chunks: the encoded image goes to a scratch buffer first, then the container is
    /// written around its chunks into <paramref name="ensureOutput"/>'s view. Returns the file length.
    /// </summary>
    private async Task<int> EncodeStillWithMetadataAsync(ArrayView1D<byte, Stride1D.Dense> rgba, int width, int height, WebPLossyEncodeOptions o,
        Func<long, ValueTask<ArrayView1D<byte, Stride1D.Dense>>> ensureOutput)
    {
        byte[]? icc = Blob(o.IccProfile), exif = Blob(o.Exif), xmp = Blob(o.Xmp);
        var anim = Anim;
        // The container kernels (UploadBlobsAsync, WriteStillFile) belong to the animation encoder: load them first.
        await anim.EnsureKernelsAsync();
        anim.ResetPool();
        var scratch = anim.RentFile();
        var e = await FrameEncoder(o)(rgba, width, height, false, scratch);
        await anim.UploadBlobsAsync(icc, exif, xmp);
        long length = WebPAnimEncoder.StillFileLength(e.Length, e.HasAlpha, e.ImageChunksStart, icc, exif, xmp);
        if (length > int.MaxValue) throw new InvalidOperationException("The file is too large.");
        var output = await ensureOutput(length);
        anim.WriteStillFile(output, scratch.View, e.Length, e.HasAlpha, e.ImageChunksStart, width, height, icc, exif, xmp);
        await _accelerator.SynchronizeAsync();
        return (int)length;
    }
}
