// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 in ISO BMFF (the AV1 codec ISO media file format binding, AOMedia): the AV1CodecConfigurationRecord ('av1C') from a
// sequence header OBU, and the sample form of a temporal unit (temporal delimiter OBUs removed; everything else, the
// sequence header of key frames included, kept as is). Matches what ffmpeg writes (black box: the Mp4Mux fixtures).
// Host-side container bookkeeping: it reads OBU headers and the sequence header's fields, no picture data.

using SpawnDev.Codecs.Video.Av1;

namespace SpawnDev.Codecs.Container.Mp4;

/// <summary>AV1 ISOBMFF helpers for <see cref="Mp4Writer"/>.</summary>
public static class Av1Isobmff
{
    private const int TemporalDelimiter = 2, SequenceHeader = 1;

    /// <summary>One OBU's span in a buffer: where it starts, its whole length, its type and its payload.</summary>
    private readonly record struct Span(int Start, int Length, int Type, int PayloadOffset, int PayloadLength, bool HasSize);

    private static List<Span> Obus(ReadOnlySpan<byte> data)
    {
        var list = new List<Span>();
        int pos = 0;
        while (pos < data.Length)
        {
            int start = pos;
            byte h = data[pos++];
            int type = (h >> 3) & 15;
            bool ext = (h & 4) != 0, hasSize = (h & 2) != 0;
            if (ext) pos++;
            long size;
            if (hasSize)
            {
                size = 0;
                for (int i = 0; ; i++)
                {
                    if (pos >= data.Length || i == 8) throw new InvalidDataException("AV1: truncated OBU size.");
                    byte b = data[pos++];
                    size |= (long)(b & 0x7F) << (7 * i);
                    if ((b & 0x80) == 0) break;
                }
            }
            else size = data.Length - pos;
            if (size > data.Length - pos) throw new InvalidDataException("AV1: OBU size past the end.");
            list.Add(new Span(start, pos + (int)size - start, type, pos, (int)size, hasSize));
            pos += (int)size;
        }
        return list;
    }

    /// <summary>
    /// The av1C record for the stream whose temporal unit <paramref name="temporalUnit"/> carries a sequence header OBU:
    /// marker / version 0x81, seq_profile and seq_level_idx[0], tier / bit depth / monochrome / chroma subsampling /
    /// chroma sample position, no initial presentation delay, then the sequence header OBU as config OBU.
    /// </summary>
    /// <exception cref="InvalidDataException">No sequence header OBU in <paramref name="temporalUnit"/>.</exception>
    public static byte[] BuildAv1C(ReadOnlySpan<byte> temporalUnit)
    {
        foreach (var o in Obus(temporalUnit))
        {
            if (o.Type != SequenceHeader) continue;
            var sh = Av1SequenceHeaderParser.Parse(temporalUnit.Slice(o.PayloadOffset, o.PayloadLength));
            var obu = temporalUnit.Slice(o.Start, o.Length);
            if (!o.HasSize) throw new NotSupportedException("AV1: a sequence header OBU without obu_size (Annex B / low overhead without sizes).");
            var rec = new byte[4 + obu.Length];
            rec[0] = 0x81;                                                       // marker 1, version 1
            rec[1] = (byte)((sh.SeqProfile << 5) | (sh.SeqLevelIdx0 & 31));
            rec[2] = (byte)((sh.SeqTier0 << 7) | ((sh.BitDepth > 8 ? 1 : 0) << 6) | ((sh.BitDepth == 12 ? 1 : 0) << 5)
                | ((sh.Monochrome ? 1 : 0) << 4) | (sh.SubsamplingX << 3) | (sh.SubsamplingY << 2) | (sh.ChromaSamplePosition & 3));
            rec[3] = 0;                                                          // no initial_presentation_delay
            obu.CopyTo(rec.AsSpan(4));
            return rec;
        }
        throw new InvalidDataException("AV1: no sequence header OBU in the temporal unit.");
    }

    /// <summary>The ISOBMFF sample of a temporal unit: its OBUs without the temporal delimiters.</summary>
    public static byte[] ToSample(ReadOnlySpan<byte> temporalUnit)
    {
        var obus = Obus(temporalUnit);
        var sample = new byte[obus.Where(o => o.Type != TemporalDelimiter).Sum(o => o.Length)];
        int at = 0;
        foreach (var o in obus)
        {
            if (o.Type == TemporalDelimiter) continue;
            temporalUnit.Slice(o.Start, o.Length).CopyTo(sample.AsSpan(at));
            at += o.Length;
        }
        return sample;
    }
}
