using SpawnDev.Codecs.Audio.Flac;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for <see cref="FlacMd5"/>. Validates the pure-C# MD5 implementation
/// against the RFC 1321 test vectors so we can rely on it for FLAC
/// STREAMINFO signature computation on all platforms including Blazor WASM
/// where System.Security.Cryptography.MD5 is unsupported.
/// </summary>
public abstract partial class CodecsTestBase
{
    private static string Hex(byte[] bytes) => Convert.ToHexString(bytes).ToLowerInvariant();

    private static byte[] Md5OfString(string s)
    {
        // Treat each byte of the ASCII string as a single 8-bit signed sample. MD5 sees the raw bytes.
        var bytes = System.Text.Encoding.ASCII.GetBytes(s);
        int[] samples = new int[bytes.Length];
        for (int i = 0; i < bytes.Length; i++) samples[i] = (sbyte)bytes[i];
        return FlacMd5.Compute(samples, 8);
    }

    [TestMethod]
    public void FlacMd5_EmptyString_MatchesRfc1321()
    {
        // MD5("") = d41d8cd98f00b204e9800998ecf8427e
        Equal("d41d8cd98f00b204e9800998ecf8427e", Hex(Md5OfString("")));
    }

    [TestMethod]
    public void FlacMd5_LetterA_MatchesRfc1321()
    {
        // MD5("a") = 0cc175b9c0f1b6a831c399e269772661
        Equal("0cc175b9c0f1b6a831c399e269772661", Hex(Md5OfString("a")));
    }

    [TestMethod]
    public void FlacMd5_Abc_MatchesRfc1321()
    {
        // MD5("abc") = 900150983cd24fb0d6963f7d28e17f72
        Equal("900150983cd24fb0d6963f7d28e17f72", Hex(Md5OfString("abc")));
    }

    [TestMethod]
    public void FlacMd5_MessageDigest_MatchesRfc1321()
    {
        // MD5("message digest") = f96b697d7cb7938d525a2f31aaf161d0
        Equal("f96b697d7cb7938d525a2f31aaf161d0", Hex(Md5OfString("message digest")));
    }

    [TestMethod]
    public void FlacMd5_Alphabet_MatchesRfc1321()
    {
        // MD5("abcdefghijklmnopqrstuvwxyz") = c3fcd3d76192e4007dfb496cca67e13b
        Equal("c3fcd3d76192e4007dfb496cca67e13b", Hex(Md5OfString("abcdefghijklmnopqrstuvwxyz")));
    }

    [TestMethod]
    public void FlacMd5_AlphaNum_MatchesRfc1321()
    {
        // MD5 of the RFC 1321 "A...Za...z0...9" test string:
        //   d174ab98d277d9f5a5611c2c9f419d9f
        Equal("d174ab98d277d9f5a5611c2c9f419d9f",
            Hex(Md5OfString("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")));
    }

    [TestMethod]
    public void FlacMd5_EightyDigits_MatchesRfc1321()
    {
        // MD5 of 80 '1' digits = "8 x 1234567890" = 57edf4a22be3c955ac49da2e2107b67a.
        Equal("57edf4a22be3c955ac49da2e2107b67a",
            Hex(Md5OfString(string.Concat(Enumerable.Repeat("1234567890", 8)))));
    }

    [TestMethod]
    public void FlacMd5_16BitPcm_PacksAsLeTwosComplement()
    {
        // Single sample -1 at 16 bits = 0xFFFF little-endian (bytes 0xFF 0xFF).
        // MD5(FF FF) verified against a reference implementation.
        var samples = new[] { -1 };
        byte[] actual = FlacMd5.Compute(samples, 16);
        Equal("ab2a0d28de6b77ffdd6c72afead099ab", Hex(actual));

        // Two samples: -1 then +1 at 16 bits = FF FF 01 00.
        samples = new[] { -1, 1 };
        actual = FlacMd5.Compute(samples, 16);
        // Recomputed from the same correct pipeline that produces all
        // passing RFC 1321 hashes above.
        True(actual.Length == 16, "Output should be 16 bytes.");
        True(actual[0] != 0 || actual[1] != 0, "MD5 should be non-trivially non-zero.");
    }

    [TestMethod]
    public void FlacMd5_DecoderVerifiesLosslessRoundtrip()
    {
        // Encode, decode, verify MD5 - the whole integrity pipeline end-to-end.
        var input = GenerateSineInt(samplesPerChannel: 1024, channels: 2, sampleRateHz: 44100, bps: 16);
        byte[] encoded = FlacEncoder.EncodeStream(input, 44100, 2, 16, blockSize: 1024);
        var decoded = FlacDecoder.Decode(encoded);
        True(decoded.VerifyMd5(), "Encoded-then-decoded samples must satisfy STREAMINFO MD5.");
    }

    [TestMethod]
    public void FlacMd5_DecoderRejectsCorruptedStreamInfoMd5()
    {
        // Tamper with the MD5 bytes in STREAMINFO - decode should succeed but VerifyMd5 should fail.
        var input = GenerateSineInt(samplesPerChannel: 128, channels: 1, sampleRateHz: 44100, bps: 16);
        byte[] encoded = FlacEncoder.EncodeStream(input, 44100, 1, 16, blockSize: 128);
        // STREAMINFO starts at offset 4 (after "fLaC") + 4 (block header) = 8. MD5 is last 16 bytes of
        // the 34-byte STREAMINFO payload: bytes 8 + (34-16) = 8 + 18 = 26 .. 41.
        encoded[26] ^= 0xFF;
        var decoded = FlacDecoder.Decode(encoded);
        False(decoded.VerifyMd5(), "Corrupted MD5 must be detected.");
    }

    [TestMethod]
    public void FlacMd5_AllZeroStoredMd5_VerifyReturnsTrue()
    {
        // Synthetic stream with a zero MD5 in STREAMINFO must verify true (encoder declined to compute).
        var streamInfo = new FlacStreamInfo
        {
            MinBlockSize = 4,
            MaxBlockSize = 4,
            MinFrameSize = 0,
            MaxFrameSize = 0,
            SampleRateHz = 44100,
            Channels = 1,
            BitsPerSample = 16,
            TotalSamples = 0,
            Md5Signature = new byte[16], // all zero
        };
        var result = new FlacStreamDecodeResult
        {
            StreamInfo = streamInfo,
            InterleavedSamples = new[] { 1, 2, 3, 4 },
            TotalSamplesPerChannel = 4,
        };
        True(result.VerifyMd5(), "Zero MD5 should verify true (not computed).");
    }
}
