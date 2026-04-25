// Av1ObuWriter round-trip tests against real BBB AV1 OBU streams.
// Parses every OBU out of bbb_180_2s.ivf (60 frames, 148 OBUs total),
// re-emits each through Av1ObuWriter, and verifies the emitted bytes
// match the source bit-exactly.
//
// This proves the writer math (header byte composition, extension
// byte, LEB128 size encoding) is the exact inverse of the parser.
// The first AV1 emit path in pure .NET that round-trips a real-world
// stream from a libaom-av1 encoder.

using SpawnDev.Codecs.Container.Ivf;
using SpawnDev.Codecs.Video.Av1;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Av1ObuWriter_Leb128_RoundTripCanonicalValues()
    {
        long[] values = { 0, 1, 0x7F, 0x80, 0xFF, 0x3FFF, 0x4000, 0x1FFFFF, 0x200000, int.MaxValue };
        foreach (var v in values)
        {
            int len = Av1ObuWriter.Leb128Length(v);
            var buf = new byte[len];
            int written = Av1ObuWriter.WriteLeb128(buf, v);
            Equal(len, written);

            long readBack = ReadLeb128(buf, out int read);
            Equal(len, read);
            Equal(v, readBack);
        }
    }

    [TestMethod]
    public void Av1ObuWriter_Leb128_KnownEncodings()
    {
        // 0x80 LEB128 = [0x80, 0x01].
        var buf2 = new byte[2];
        int n = Av1ObuWriter.WriteLeb128(buf2, 0x80);
        Equal(2, n);
        Equal(0x80, (int)buf2[0]);
        Equal(0x01, (int)buf2[1]);

        // 0x3FFF = 16383 = [0xFF, 0x7F].
        var buf3 = new byte[2];
        n = Av1ObuWriter.WriteLeb128(buf3, 0x3FFF);
        Equal(2, n);
        Equal(0xFF, (int)buf3[0]);
        Equal(0x7F, (int)buf3[1]);
    }

    [TestMethod]
    public void Av1ObuWriter_BbbFirstFrame_RoundTripsBitExact()
    {
        var bytes = LoadAv1Fixture();
        var firstFrame = IvfReader.EnumerateFrames(bytes).First();
        int total = 0;

        foreach (var obu in Av1ObuParser.EnumerateObus(firstFrame.Data))
        {
            byte[] reEmitted = Av1ObuWriter.EmitObu(obu, firstFrame.Data);

            // Reconstruct the source slice for this OBU - from the start of
            // the OBU header byte through the end of its payload.
            int hdrLen = 1
                + (obu.HasExtension ? 1 : 0)
                + (obu.HasSizeField ? Av1ObuWriter.Leb128Length(obu.PayloadLength) : 0);
            int srcStart = obu.PayloadOffset - hdrLen;
            int srcLen = hdrLen + obu.PayloadLength;
            var sourceSlice = firstFrame.Data.Slice(srcStart, srcLen).Span;

            Equal(srcLen, reEmitted.Length);
            for (int i = 0; i < srcLen; i++)
            {
                Equal(sourceSlice[i], reEmitted[i]);
            }
            total++;
        }

        True(total >= 3, $"expected several OBUs in first frame; got {total}");
    }

    [TestMethod]
    public void Av1ObuWriter_BbbAllFrames_RoundTripsBitExact()
    {
        var bytes = LoadAv1Fixture();
        int frames = 0;
        int obus = 0;
        int totalBytes = 0;

        foreach (var ivfFrame in IvfReader.EnumerateFrames(bytes))
        {
            frames++;
            foreach (var obu in Av1ObuParser.EnumerateObus(ivfFrame.Data))
            {
                obus++;
                byte[] reEmitted = Av1ObuWriter.EmitObu(obu, ivfFrame.Data);
                int hdrLen = 1
                    + (obu.HasExtension ? 1 : 0)
                    + (obu.HasSizeField ? Av1ObuWriter.Leb128Length(obu.PayloadLength) : 0);
                int srcStart = obu.PayloadOffset - hdrLen;
                int srcLen = hdrLen + obu.PayloadLength;
                var sourceSlice = ivfFrame.Data.Slice(srcStart, srcLen).Span;
                totalBytes += srcLen;

                Equal(srcLen, reEmitted.Length);
                for (int i = 0; i < srcLen; i++)
                {
                    if (sourceSlice[i] != reEmitted[i])
                        throw new Exception(
                            $"frame {frames} OBU {obu.Type} byte {i}: "
                            + $"src 0x{sourceSlice[i]:X2} vs emit 0x{reEmitted[i]:X2}");
                }
            }
        }

        Equal(60, frames);
        True(obus > 100, $"expected >100 OBUs across 60 frames; got {obus}");
        True(totalBytes > 50_000, $"expected substantial OBU bytes; got {totalBytes}");
    }

    [TestMethod]
    public void Av1ObuWriter_BbbFirstFrame_ConcatenatedRoundTripMatchesSource()
    {
        // Stronger end-to-end check: concatenate every re-emitted OBU and
        // verify the resulting buffer is byte-identical to the original
        // IVF frame payload.
        var bytes = LoadAv1Fixture();
        var firstFrame = IvfReader.EnumerateFrames(bytes).First();

        using var ms = new MemoryStream();
        foreach (var obu in Av1ObuParser.EnumerateObus(firstFrame.Data))
        {
            byte[] re = Av1ObuWriter.EmitObu(obu, firstFrame.Data);
            ms.Write(re, 0, re.Length);
        }
        byte[] emitted = ms.ToArray();
        byte[] source = firstFrame.Data.ToArray();

        Equal(source.Length, emitted.Length);
        for (int i = 0; i < source.Length; i++)
        {
            if (source[i] != emitted[i])
                throw new Exception(
                    $"first-frame concat mismatch at byte {i}: "
                    + $"src 0x{source[i]:X2} vs emit 0x{emitted[i]:X2}");
        }
    }

    private static long ReadLeb128(ReadOnlySpan<byte> data, out int bytesRead)
    {
        long value = 0;
        int shift = 0;
        for (int i = 0; i < 8 && i < data.Length; i++)
        {
            byte b = data[i];
            value |= (long)(b & 0x7F) << shift;
            shift += 7;
            if ((b & 0x80) == 0)
            {
                bytesRead = i + 1;
                return value;
            }
        }
        throw new InvalidDataException("LEB128 truncated or too long.");
    }
}
