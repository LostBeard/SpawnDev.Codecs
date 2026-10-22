// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// YUV 4:2:0 -> RGBA8 exactly as libwebp's default decode produces it (what Pillow, browsers and dwebp show):
//   - the "fancy" triangle chroma upsampler (upsampling.c: UpsampleRgbaLinePair): every output pixel blends its
//     four nearest chroma samples with 9/3/3/1 weights, rounding as the packed-integer original does;
//   - the fixed-point YUV -> RGB matrix of yuv.h (YUV_FIX2 = 6: R = clip((19077*y + 26149*v - 14234) >> 6) with the
//     per-term >> 8 that VP8YUVToR uses, G and B likewise).
// One thread per output pixel; the planes never visit the host. Used after the VP8 keyframe decoder for lossy WebP,
// and usable on its own for any 4:2:0 picture that should look like libwebp's.
//
// Plane layout: Y is width x height, U and V are ((width + 1) / 2) x ((height + 1) / 2), all tightly packed.

using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.Codecs.Images.Yuv;

/// <summary>Converts YUV 4:2:0 planes on the accelerator to RGBA8 bit-exactly like libwebp (fancy upsampling).</summary>
public sealed class WebPYuvToRgbaGpu
{
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, int, int> _convert;

    /// <summary>Compile the kernel.</summary>
    public WebPYuvToRgbaGpu(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _convert = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, int, int>(ConvertKernel);
    }

    /// <summary>Size in bytes of the U (or V) plane for a picture.</summary>
    public static long ChromaPlaneSize(int width, int height) => (long)((width + 1) >> 1) * ((height + 1) >> 1);

    /// <summary>
    /// Queue the conversion; synchronize the accelerator before reading <paramref name="rgba"/>. Alpha is written as
    /// 255 (an ALPH plane is applied by the WebP decoder afterwards).
    /// </summary>
    /// <param name="yPlane">Luma, <c>width * height</c> bytes.</param>
    /// <param name="uPlane">Cb, <see cref="ChromaPlaneSize"/> bytes.</param>
    /// <param name="vPlane">Cr, <see cref="ChromaPlaneSize"/> bytes.</param>
    /// <param name="width">Picture width in pixels (&gt; 0).</param>
    /// <param name="height">Picture height in pixels (&gt; 0).</param>
    /// <param name="rgba">Receives <c>width * height * 4</c> bytes.</param>
    public void Convert(
        ArrayView<byte> yPlane, ArrayView<byte> uPlane, ArrayView<byte> vPlane,
        int width, int height, ArrayView<byte> rgba)
    {
        if (width <= 0) throw new ArgumentOutOfRangeException(nameof(width));
        if (height <= 0) throw new ArgumentOutOfRangeException(nameof(height));
        long pixels = (long)width * height;
        if (pixels > int.MaxValue / 4) throw new ArgumentOutOfRangeException(nameof(width), "picture too large");
        long chroma = ChromaPlaneSize(width, height);
        if (yPlane.Length < pixels) throw new ArgumentException("yPlane too short.", nameof(yPlane));
        if (uPlane.Length < chroma) throw new ArgumentException("uPlane too short.", nameof(uPlane));
        if (vPlane.Length < chroma) throw new ArgumentException("vPlane too short.", nameof(vPlane));
        if (rgba.Length < pixels * 4) throw new ArgumentException("rgba too short.", nameof(rgba));
        _convert((int)pixels, yPlane, uPlane, vPlane, rgba, width, height);
    }

    // libwebp yuv.h
    private static int MultHi(int v, int coeff) => (v * coeff) >> 8;

    private static int Clip8(int v)
    {
        // ((v & ~YUV_MASK2) == 0) ? (v >> YUV_FIX2) : (v < 0) ? 0 : 255, with YUV_MASK2 = (256 << 6) - 1
        int inRange = (v & ~16383) == 0 ? 1 : 0;
        int shifted = v >> 6;
        int clamped = v < 0 ? 0 : 255;
        return inRange != 0 ? shifted : clamped;
    }

    private static void ConvertKernel(
        Index1D index, ArrayView<byte> yPlane, ArrayView<byte> uPlane, ArrayView<byte> vPlane,
        ArrayView<byte> rgba, int width, int height)
    {
        int px = index % width;
        int py = index / width;
        int cw = (width + 1) >> 1;
        int ch = (height + 1) >> 1;

        // Row pairing of EmitFancyRGB: row 0 alone (chroma rows 0,0); then rows (2k-1, 2k) share chroma rows (k-1, k);
        // an even height ends with a lone row that reuses the last chroma row for both.
        bool top = true;
        int rowTop = 0;
        int rowCur = 0;
        if (py > 0)
        {
            int k = (py + 1) >> 1;
            top = (py & 1) == 1;
            rowTop = k - 1;
            rowCur = k < ch - 1 ? k : ch - 1;
        }

        int lastPair = (width - 1) >> 1;
        int u;
        int v;
        if (px == 0)
        {
            int tlU = uPlane[rowTop * cw];
            int lU = uPlane[rowCur * cw];
            int tlV = vPlane[rowTop * cw];
            int lV = vPlane[rowCur * cw];
            u = top ? (3 * tlU + lU + 2) >> 2 : (3 * lU + tlU + 2) >> 2;
            v = top ? (3 * tlV + lV + 2) >> 2 : (3 * lV + tlV + 2) >> 2;
        }
        else
        {
            int x = (px + 1) >> 1;
            bool trailing = x > lastPair;   // even width, last pixel: only one chroma column left
            int xs = trailing ? lastPair : x;
            int xl = trailing ? lastPair : x - 1;
            int tlU = uPlane[rowTop * cw + xl];
            int tU = uPlane[rowTop * cw + xs];
            int lU = uPlane[rowCur * cw + xl];
            int cU = uPlane[rowCur * cw + xs];
            int tlV = vPlane[rowTop * cw + xl];
            int tV = vPlane[rowTop * cw + xs];
            int lV = vPlane[rowCur * cw + xl];
            int cV = vPlane[rowCur * cw + xs];

            int avgU = tlU + tU + lU + cU + 8;
            int diag12U = (avgU + 2 * (tU + lU)) >> 3;
            int diag03U = (avgU + 2 * (tlU + cU)) >> 3;
            int avgV = tlV + tV + lV + cV + 8;
            int diag12V = (avgV + 2 * (tV + lV)) >> 3;
            int diag03V = (avgV + 2 * (tlV + cV)) >> 3;

            bool odd = (px & 1) == 1;   // pixel 2x-1 (odd) or 2x (even)
            int uTrail = top ? (3 * tlU + lU + 2) >> 2 : (3 * lU + tlU + 2) >> 2;
            int vTrail = top ? (3 * tlV + lV + 2) >> 2 : (3 * lV + tlV + 2) >> 2;
            int uMain;
            int vMain;
            if (odd)
            {
                uMain = top ? (diag12U + tlU) >> 1 : (diag03U + lU) >> 1;
                vMain = top ? (diag12V + tlV) >> 1 : (diag03V + lV) >> 1;
            }
            else
            {
                uMain = top ? (diag03U + tU) >> 1 : (diag12U + cU) >> 1;
                vMain = top ? (diag03V + tV) >> 1 : (diag12V + cV) >> 1;
            }
            u = trailing ? uTrail : uMain;
            v = trailing ? vTrail : vMain;
        }

        int y = yPlane[index];
        int yy = MultHi(y, 19077);
        int r = Clip8(yy + MultHi(v, 26149) - 14234);
        int g = Clip8(yy - MultHi(u, 6419) - MultHi(v, 13320) + 8708);
        int b = Clip8(yy + MultHi(u, 33050) - 17685);

        int o = index * 4;
        rgba[o] = (byte)r;
        rgba[o + 1] = (byte)g;
        rgba[o + 2] = (byte)b;
        rgba[o + 3] = 255;
    }
}
