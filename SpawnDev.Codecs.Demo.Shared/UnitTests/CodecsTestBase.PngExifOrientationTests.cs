// PNG eXIf orientation: ImageInfo.Orientation reports the EXIF orientation of an eXIf chunk (a TIFF block, no "Exif\0\0"
// prefix), as the JPEG and WebP decoders report theirs; the pixels are returned as stored. SpawnGen's Load Image turns
// the picture by it (Lore, 2026-11-20: a PNG with orientation 6 came back unturned, Orientation 1). Every orientation in
// both TIFF byte orders, before and after IDAT, plus blocks that must read as 1 (truncated, wrong magic, no tag).

using SpawnDev.Codecs.Images;
using SpawnDev.Codecs.Images.Png;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod(Category = "NotOnWebGL")]
    public async Task PngDecoderGpu_ExifChunk_ReportsOrientation()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            const int w = 9, h = 5;
            var src = PicturePixels(w, h, 0x0E1F, opaque: true, gray: false);
            using var enc = new PngEncoderGpu(acc);
            byte[] plain = await enc.EncodeAsync(src, w, h);
            using var dec = new PngDecoderGpu(acc);
            var failures = new List<string>();
            async Task Check(string what, byte[] file, int want)
            {
                var r = await dec.DecodeAsync(file);
                if (r.Info.Orientation != want) failures.Add($"{what}: Orientation {r.Info.Orientation}, expected {want}");
                if (!r.Pixels.AsSpan().SequenceEqual(src)) failures.Add($"{what}: pixels changed");
                var info = await dec.ReadInfoAsync(file);
                if (info.Orientation != want) failures.Add($"{what}: ReadInfoAsync Orientation {info.Orientation}, expected {want}");
            }
            await Check("no eXIf", plain, 1);
            for (int o = 1; o <= 8; o++)
                foreach (bool little in new[] { true, false })
                    foreach (bool afterIdat in new[] { false, true })
                        await Check($"orientation {o} {(little ? "II" : "MM")} {(afterIdat ? "after" : "before")} IDAT",
                            WithChunk(plain, "eXIf", Tiff(little, o, withTag: true), afterIdat), o);
            await Check("eXIf without the orientation tag", WithChunk(plain, "eXIf", Tiff(true, 6, withTag: false), false), 1);
            await Check("eXIf with a bad magic", WithChunk(plain, "eXIf", BadMagic(Tiff(true, 6, true)), false), 1);
            await Check("eXIf truncated", WithChunk(plain, "eXIf", Tiff(false, 6, true)[..10], false), 1);
            await Check("eXIf orientation 9 (invalid)", WithChunk(plain, "eXIf", Tiff(true, 9, true), false), 1);
            // a writer that kept JPEG's APP1 prefix: tolerated (the shared parser skips "Exif\0\0")
            var prefixed = new byte[] { (byte)'E', (byte)'x', (byte)'i', (byte)'f', 0, 0 }.Concat(Tiff(true, 3, true)).ToArray();
            await Check("eXIf with an Exif\\0\\0 prefix", WithChunk(plain, "eXIf", prefixed, false), 3);
            True(failures.Count == 0, string.Join("\n", failures));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    // A TIFF block with one IFD: an ImageDescription tag first (so the orientation is not the first entry), then (optionally)
    // Orientation (0x0112, SHORT, count 1).
    private static byte[] Tiff(bool little, int orientation, bool withTag)
    {
        var b = new List<byte>();
        void U16(int v) { if (little) { b.Add((byte)v); b.Add((byte)(v >> 8)); } else { b.Add((byte)(v >> 8)); b.Add((byte)v); } }
        void U32(int v) { if (little) { U16(v & 0xFFFF); U16((v >> 16) & 0xFFFF); } else { U16((v >> 16) & 0xFFFF); U16(v & 0xFFFF); } }
        b.Add((byte)(little ? 'I' : 'M')); b.Add((byte)(little ? 'I' : 'M'));
        U16(42); U32(8);
        int entries = withTag ? 2 : 1;
        U16(entries);
        U16(0x010E); U16(2); U32(4); b.Add((byte)'a'); b.Add((byte)'b'); b.Add((byte)'c'); b.Add(0);   // ImageDescription "abc"
        if (withTag) { U16(0x0112); U16(3); U32(1); U16(orientation); U16(0); }
        U32(0);   // no next IFD
        return b.ToArray();
    }

    private static byte[] BadMagic(byte[] tiff)
    {
        var t = (byte[])tiff.Clone();
        t[2] = 43;   // II, 43 instead of 42
        return t;
    }

    // The PNG with a chunk inserted right after IHDR, or right before IEND.
    private static byte[] WithChunk(byte[] png, string type, byte[] data, bool beforeIend)
    {
        int at = beforeIend ? png.Length - 12 : 8 + 25;
        var t = System.Text.Encoding.ASCII.GetBytes(type);
        var chunk = new byte[12 + data.Length];
        chunk[0] = (byte)(data.Length >> 24); chunk[1] = (byte)(data.Length >> 16); chunk[2] = (byte)(data.Length >> 8); chunk[3] = (byte)data.Length;
        Buffer.BlockCopy(t, 0, chunk, 4, 4);
        Buffer.BlockCopy(data, 0, chunk, 8, data.Length);
        uint crc = PngCrc(chunk, 4, 4 + data.Length);
        chunk[8 + data.Length] = (byte)(crc >> 24); chunk[9 + data.Length] = (byte)(crc >> 16);
        chunk[10 + data.Length] = (byte)(crc >> 8); chunk[11 + data.Length] = (byte)crc;
        return png[..at].Concat(chunk).Concat(png[at..]).ToArray();
    }

    private static uint PngCrc(byte[] b, int off, int len)
    {
        uint c = 0xFFFFFFFFu;
        for (int i = off; i < off + len; i++)
        {
            c ^= b[i];
            for (int k = 0; k < 8; k++) c = (c & 1) != 0 ? 0xEDB88320u ^ (c >> 1) : c >> 1;
        }
        return c ^ 0xFFFFFFFFu;
    }
}
