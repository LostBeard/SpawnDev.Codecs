// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// EXIF orientation from a TIFF block, on the accelerator: shared by the WebP (EXIF chunk) and PNG (eXIf chunk) chunk
// scans. The block may start with JPEG's "Exif\0\0" prefix (WebP writers often keep it) or not (PNG's eXIf, WebP's
// own form).

using ILGPU;

namespace SpawnDev.Codecs.Images;

internal static class ExifTiff
{
    /// <summary>Little-endian 16-bit read honouring the TIFF byte order flag (1 = little endian).</summary>
    private static int Tiff16(ArrayView<byte> f, int p, int little)
        => little == 1 ? (f[p] | (f[p + 1] << 8)) : ((f[p] << 8) | f[p + 1]);

    private static int Tiff32(ArrayView<byte> f, int p, int little)
        => little == 1
            ? (f[p] | (f[p + 1] << 8) | (f[p + 2] << 16) | (f[p + 3] << 24))
            : ((f[p] << 24) | (f[p + 1] << 16) | (f[p + 2] << 8) | f[p + 3]);

    /// <summary>EXIF orientation (tag 0x0112) of a TIFF block at <paramref name="p"/> (optionally behind "Exif\0\0"); 1 when absent or invalid.</summary>
    public static int Orientation(ArrayView<byte> f, int p, int len)
    {
        int result = 1;
        int start = p;
        int length = len;
        if (len >= 6)
        {
            bool hasPrefix = (f[p] == 'E') & (f[p + 1] == 'x') & (f[p + 2] == 'i') & (f[p + 3] == 'f')
                             & (f[p + 4] == 0) & (f[p + 5] == 0);
            if (hasPrefix)
            {
                start = p + 6;
                length = len - 6;
            }
        }
        if (length >= 8)
        {
            int little = 0;
            bool ok = true;
            bool isLittle = (f[start] == 'I') & (f[start + 1] == 'I');
            bool isBig = (f[start] == 'M') & (f[start + 1] == 'M');
            if (isLittle) little = 1;
            else if (!isBig) ok = false;
            if (ok)
            {
                int magic = Tiff16(f, start + 2, little);
                int ifd = Tiff32(f, start + 4, little);
                bool ifdOk = (magic == 42) & (ifd >= 8) & (ifd + 2 <= length);
                if (ifdOk)
                {
                    int count = Tiff16(f, start + ifd, little);
                    int e = 0;
                    int found = 0;
                    bool go = count > 0;
                    while (go)
                    {
                        int ep = ifd + 2 + e * 12;
                        if (ep + 12 > length)
                        {
                            found = 1;
                        }
                        else if (Tiff16(f, start + ep, little) == 0x0112)
                        {
                            found = 1;
                            int type = Tiff16(f, start + ep + 2, little);
                            int cnt = Tiff32(f, start + ep + 4, little);
                            int v = Tiff16(f, start + ep + 8, little);
                            bool valid = (type == 3) & (cnt == 1) & (v >= 1) & (v <= 8);
                            if (valid) result = v;
                        }
                        e++;
                        go = (e < count) & (found == 0);
                    }
                }
            }
        }
        return result;
    }
}
