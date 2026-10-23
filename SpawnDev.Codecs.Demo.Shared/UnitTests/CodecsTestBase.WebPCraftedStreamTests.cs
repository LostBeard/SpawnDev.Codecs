// Hand-built VP8L streams for the decoder's bitstream-level error paths. The streams are assembled bit by bit in the
// test from the WebP Lossless Bitstream Specification (RFC 9649), so each one is valid except for the single thing it is
// named after; a VALID control built the same way must decode, otherwise the negative tests would prove nothing.

using SpawnDev.Codecs.Images;
using SpawnDev.Codecs.Images.WebP;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    /// <summary>Least-significant-bit-first bit writer producing a simple lossless WebP file.</summary>
    private sealed class Vp8lCraft
    {
        private readonly List<byte> _bytes = new();
        private int _acc, _bits;

        public void Put(int value, int n)
        {
            for (int i = 0; i < n; i++)
            {
                _acc |= ((value >> i) & 1) << _bits;
                if (++_bits == 8) { _bytes.Add((byte)_acc); _acc = 0; _bits = 0; }
            }
        }

        public byte[] ToFile(int width, int height)
        {
            var body = new Vp8lCraft();
            body.Put(0x2F, 8);
            body.Put(width - 1, 14);
            body.Put(height - 1, 14);
            body.Put(0, 1);
            body.Put(0, 3);
            var payload = new List<byte>(body._bytes);
            payload.AddRange(_bytes);
            if (_bits > 0) payload.Add((byte)_acc);
            int size = payload.Count;
            if ((size & 1) == 1) payload.Add(0);
            var file = new List<byte>();
            file.AddRange("RIFF"u8.ToArray());
            file.AddRange(BitConverter.GetBytes(4 + 8 + payload.Count));
            file.AddRange("WEBP"u8.ToArray());
            file.AddRange("VP8L"u8.ToArray());
            file.AddRange(BitConverter.GetBytes(size));
            file.AddRange(payload);
            return file.ToArray();
        }

        /// <summary>No transform, no colour cache, no meta prefix codes.</summary>
        public void PlainImageStart()
        {
            Put(0, 1);
            Put(0, 1);
            Put(0, 1);
        }

        /// <summary>A prefix code with exactly one symbol 0 (zero bits per pixel), simple form.</summary>
        public void OneSymbolCode()
        {
            Put(1, 1);
            Put(0, 1);
            Put(0, 1);
            Put(0, 1);
        }

        /// <summary>Green code with two symbols, literal 0 ('0') and length symbol 256 ('1'), in the code-length-code form.</summary>
        public void GreenCodeLiteralAndCopy()
        {
            Put(0, 1);              // normal form
            Put(0, 4);              // 4 code-length code lengths: symbols 17, 18, 0, 1
            Put(0, 3); Put(1, 3); Put(0, 3); Put(1, 3);
            Put(0, 1);              // max_symbol = alphabet size
            Put(0, 1);                          // symbol 0: length 1 (token "1" = code 0)
            Put(1, 1); Put(127, 7);             // symbols 1..138: 138 zeros
            Put(1, 1); Put(106, 7);             // symbols 139..255: 117 zeros
            Put(0, 1);                          // symbol 256: length 1
            Put(1, 1); Put(12, 7);              // symbols 257..279: 23 zeros
        }
    }

    private static Vp8lCraft CraftedCopyStream(int literals, int copies)
    {
        var c = new Vp8lCraft();
        c.PlainImageStart();
        c.GreenCodeLiteralAndCopy();
        c.OneSymbolCode(); // red
        c.OneSymbolCode(); // blue
        c.OneSymbolCode(); // alpha
        c.OneSymbolCode(); // distance: symbol 0 = plane code 1 = the pixel above
        for (int i = 0; i < literals; i++) c.Put(0, 1);
        for (int i = 0; i < copies; i++) c.Put(1, 1);
        return c;
    }

    [TestMethod]
    public async Task WebPDecoderGpu_CraftedStream_ValidControl_Decodes()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var dec = new WebPDecoderGpu(acc);
            // 4x2: four literals, then four copies of the pixel above (distance code 1 = one row up)
            var image = await dec.DecodeAsync(CraftedCopyStream(4, 4).ToFile(4, 2));
            Equal(4, image.Info.Width, "width");
            Equal(2, image.Info.Height, "height");
            EqualBytes(new byte[32], image.Pixels, "all-zero pixels");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod]
    public async Task WebPDecoderGpu_CraftedStream_BackwardReferenceBeforeTheFirstPixel_IsRejected()
    {
        // 4x1: one literal, then a copy from one row up = 4 pixels back, but only 1 pixel exists
        await AssertWebPRejectedAsync(CraftedCopyStream(1, 1).ToFile(4, 1), "invalid backward reference", "copy reaches before the image");
    }

    [TestMethod]
    public async Task WebPDecoderGpu_CraftedStream_TransformUsedTwice_IsRejected()
    {
        var c = new Vp8lCraft();
        c.Put(1, 1); c.Put(2, 2);   // subtract green
        c.Put(1, 1); c.Put(2, 2);   // subtract green again
        await AssertWebPRejectedAsync(c.ToFile(4, 4), "transform is used twice", "duplicate transform");
    }

    [TestMethod]
    public async Task WebPDecoderGpu_CraftedStream_ColourCacheSize12_IsRejected()
    {
        var c = new Vp8lCraft();
        c.Put(0, 1);                // no transform
        c.Put(1, 1); c.Put(12, 4);  // colour cache of 12 bits (valid range 1..11)
        await AssertWebPRejectedAsync(c.ToFile(4, 4), "colour cache size", "cache bits 12");
    }

    [TestMethod]
    public async Task WebPDecoderGpu_CraftedStream_OverSubscribedPrefixCode_IsRejected()
    {
        var c = new Vp8lCraft();
        c.PlainImageStart();
        c.Put(0, 1);                // normal form
        c.Put(0, 4);                // 4 code-length code lengths
        c.Put(1, 3); c.Put(1, 3); c.Put(1, 3); c.Put(1, 3); // four symbols of length 1: more than a binary tree holds
        await AssertWebPRejectedAsync(c.ToFile(4, 4), "invalid prefix", "over-subscribed code-length code");
    }

    [TestMethod]
    public async Task WebPDecoderGpu_CraftedStream_IncompletePrefixCode_IsRejected()
    {
        var c = new Vp8lCraft();
        c.PlainImageStart();
        c.Put(0, 1);                // normal form
        c.Put(0, 4);
        c.Put(0, 3); c.Put(0, 3); c.Put(1, 3); c.Put(2, 3); // symbols 0 and 1 with lengths 1 and 2: Kraft sum 3/4
        await AssertWebPRejectedAsync(c.ToFile(4, 4), "invalid prefix", "incomplete code-length code");
    }
}
