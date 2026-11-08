// Tests for the Ogg Opus writer (Container/Ogg/OggOpusWriter.cs, a port of FFmpeg 8.1's ogg muxer) and for Ogg Opus
// encoding with OpusEncoderGpu at the settings FFmpeg's libopus wrapper uses.
//
// Fixtures (scripts/audio/make_ogg_opus_fixtures.py, TestData/OggOpus): FFmpeg + libopus files of stereo / mono 48 kHz
// and stereo 24 kHz sources at 64..320 kbps (bit-exact mode), with the packets each contains.
//
// What must hold:
//   * the writer, given FFmpeg's packets (pts from -pre-skip in 20 ms steps, the last packet's duration from the final
//     granule), writes FFmpeg's file BYTE FOR BYTE: OpusHead, OpusTags, page packing (1 s pages, 255 segments),
//     granules, end-of-stream flag, Ogg CRCs;
//   * OggOpusEncoder (OpusEncoderGpu at FFmpeg's libopus settings + the writer) writes FFmpeg's file layout: the same
//     packet count, page granules, page flags and OpusHead / OpusTags; its audio decodes close to the source. The
//     packets themselves differ: FFmpeg's libopus is the float build (tonality analysis from complexity 7), ours is
//     libopus 1.5.2's fixed-point CELT; the test reports how many packets happen to match;
//   * the float input path converts as libswresample does (lrintf(x * 32768), clipped).

using System.Text.Json;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.Codecs.Audio.Opus;
using SpawnDev.Codecs.Container.Ogg;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private const string OggOpusPrefix = "SpawnDev.Codecs.Demo.Shared.TestData.OggOpus.";

    private static List<string> OggOpusCases() =>
        System.Text.Encoding.UTF8.GetString(ReadEmbedded(OggOpusPrefix + "manifest.txt")).Split('\n')
            .Select(l => l.Trim()).Where(l => l.Length > 0).ToList();

    /// <summary>Vendor and fields of an OpusTags packet.</summary>
    private static (string Vendor, List<KeyValuePair<string, string>> Tags) ParseOpusTags(byte[] t)
    {
        int p = 8;
        int vl = BitConverter.ToInt32(t, p); p += 4;
        string vendor = System.Text.Encoding.UTF8.GetString(t, p, vl); p += vl;
        int n = BitConverter.ToInt32(t, p); p += 4;
        var tags = new List<KeyValuePair<string, string>>();
        for (int i = 0; i < n; i++)
        {
            int l = BitConverter.ToInt32(t, p); p += 4;
            string f = System.Text.Encoding.UTF8.GetString(t, p, l); p += l;
            int eq = f.IndexOf('=');
            tags.Add(new KeyValuePair<string, string>(f[..eq], f[(eq + 1)..]));
        }
        return (vendor, tags);
    }

    [TestMethod(Category = "CpuOnly")]
    public async Task OggOpusWriter_ByteIdenticalWithFfmpeg()
    {
        await Task.Yield();
        var failures = new List<string>();
        foreach (var name in OggOpusCases())
        {
            var truth = ReadEmbedded(OggOpusPrefix + name + ".opus");
            using var json = JsonDocument.Parse(ReadEmbedded(OggOpusPrefix + name + ".json"));
            var d = json.RootElement;
            int preSkip = d.GetProperty("preSkip").GetInt32();
            var (vendor, tags) = ParseOpusTags(Convert.FromHexString(d.GetProperty("tags").GetString()!));
            var packets = d.GetProperty("packets").EnumerateArray().ToList();
            long finalGranule = packets[^1][2].GetInt64();
            var ms = new MemoryStream();
            using (var w = new OggOpusWriter(ms, new OggOpusWriterOptions
            {
                Channels = d.GetProperty("channels").GetInt32(), PreSkip = preSkip,
                InputSampleRate = d.GetProperty("inputRate").GetInt32(), Vendor = vendor, Tags = tags,
            }, leaveOpen: true))
            {
                for (int k = 0; k < packets.Count; k++)
                {
                    long pts = -preSkip + 960L * k;
                    int duration = k < packets.Count - 1 ? 960 : (int)(finalGranule - preSkip - pts);
                    w.WritePacket(truth.AsSpan(packets[k][0].GetInt32(), packets[k][1].GetInt32()), pts, duration);
                }
            }
            var ours = ms.ToArray();
            if (!ours.AsSpan().SequenceEqual(truth))
                failures.Add($"{name}: {ours.Length} bytes vs FFmpeg {truth.Length}, first difference at byte {FirstDifference(ours, truth)}");
        }
        True(failures.Count == 0, string.Join("\n", failures));
    }

    /// <summary>Each page of an Ogg file: (granule, flags, sequence number).</summary>
    private static List<(long Granule, int Flags, int Seq)> OggPages(byte[] b)
    {
        var pages = new List<(long, int, int)>();
        int p = 0;
        while (p < b.Length)
        {
            int nseg = b[p + 26], body = 0;
            for (int i = 0; i < nseg; i++) body += b[p + 27 + i];
            pages.Add((BitConverter.ToInt64(b, p + 6), b[p + 5], BitConverter.ToInt32(b, p + 18)));
            p += 27 + nseg + body;
        }
        return pages;
    }

    /// <summary>The packets of an Ogg file (header packets included).</summary>
    private static List<byte[]> OggPackets(byte[] b)
    {
        var packets = new List<byte[]>();
        var cur = new List<byte>();
        int p = 0;
        while (p < b.Length)
        {
            int nseg = b[p + 26], at = p + 27 + nseg;
            for (int i = 0; i < nseg; i++)
            {
                int s = b[p + 27 + i];
                cur.AddRange(b.AsSpan(at, s).ToArray());
                at += s;
                if (s < 255) { packets.Add(cur.ToArray()); cur.Clear(); }
            }
            p = at;
        }
        return packets;
    }

    [TestMethod(Timeout = 900000, Category = "NotOnWebGL")]
    public async Task OggOpusEncoder_FfmpegLayout_DecodesCloseToSource()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            var failures = new List<string>();
            var report = new List<string>();
            var cache = new Dictionary<string, short[]>();
            foreach (var name in OggOpusCases())
            {
                var truth = ReadEmbedded(OggOpusPrefix + name + ".opus");
                using var json = JsonDocument.Parse(ReadEmbedded(OggOpusPrefix + name + ".json"));
                var d = json.RootElement;
                int rate = d.GetProperty("rate").GetInt32(), ch = d.GetProperty("channels").GetInt32();
                int complexity = d.GetProperty("complexity").GetInt32();
                string src = d.GetProperty("source").GetString()!;
                if (!cache.TryGetValue(src, out var pcm))
                {
                    var raw = Gunzip(ReadEmbedded(OggOpusPrefix + src + ".s16.gz"));
                    pcm = new short[raw.Length / 2];
                    Buffer.BlockCopy(raw, 0, pcm, 0, raw.Length);
                    cache[src] = pcm;
                }
                var (vendor, tags) = ParseOpusTags(Convert.FromHexString(d.GetProperty("tags").GetString()!));
                var ours = await OggOpusEncoder.EncodeAsync(acc, pcm, rate, ch, new OggOpusEncodeOptions
                {
                    Bitrate = d.GetProperty("bitrate").GetInt32(), Complexity = Math.Min(complexity, 9), Vendor = vendor, Tags = tags,
                });

                // FFmpeg's layout: the same pages (granule, flags, sequence) and header packets
                var op = OggPages(ours);
                var tp = OggPages(truth);
                if (!op.SequenceEqual(tp))
                {
                    failures.Add($"{name}: pages {string.Join(" ", op.Select(x => x.Granule + "/" + x.Flags))} vs FFmpeg "
                        + string.Join(" ", tp.Select(x => x.Granule + "/" + x.Flags)));
                    continue;
                }
                var ok = OggPackets(ours);
                var tk = OggPackets(truth);
                if (ok.Count != tk.Count) { failures.Add($"{name}: {ok.Count} packets vs FFmpeg {tk.Count}"); continue; }
                if (!ok[0].AsSpan().SequenceEqual(tk[0]) || !ok[1].AsSpan().SequenceEqual(tk[1]))
                {
                    failures.Add($"{name}: OpusHead / OpusTags differ");
                    continue;
                }
                int same = 0;
                for (int k = 2; k < ok.Count; k++) if (ok[k].AsSpan().SequenceEqual(tk[k])) same++;

                // the audio: decode (at the source rate), drop the pre-skip, compare with the source
                using var dec = new OpusDecoderGpu(acc, rate, ch);
                var outPcm = new List<float>();
                for (int k = 2; k < ok.Count; k++) outPcm.AddRange(await dec.DecodePacketAsync(ok[k]));
                int skip = 312 / (48000 / rate) * ch;
                double sig = 0, err = 0;
                for (int i = 0; i < pcm.Length; i++)
                {
                    double x = pcm[i] / 32768.0, y = outPcm[skip + i];
                    sig += x * x;
                    err += (x - y) * (x - y);
                }
                double snr = 10 * Math.Log10(sig / Math.Max(err, 1e-30));
                report.Add($"{name} {ours.Length}B vs {truth.Length}B, {same}/{ok.Count - 2} packets equal, SNR {snr:F1} dB");
                if (snr < 12) failures.Add($"{name}: decoded audio SNR {snr:F1} dB vs the source");
            }
            Console.WriteLine(string.Join("\n", report));
            True(failures.Count == 0, string.Join("\n", failures));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OggOpusEncoder_FloatInput_ConvertsAsLibswresample()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            // the conversion itself, edge cases included (round half to even, clipping, x86 integer indefinite)
            float[] x = { 0f, 0.5f / 32768, 1.5f / 32768, 2.5f / 32768, -0.5f / 32768, -1.5f / 32768, -2.5f / 32768, 0.7f / 32768,
                -0.7f / 32768, 1f, -1f, 2f, -2f, 32766.5f / 32768, -32767.5f / 32768, float.NaN, float.PositiveInfinity,
                float.NegativeInfinity, 70000f, 0.25f };
            short[] want = { 0, 0, 2, 2, 0, -2, -2, 1, -1, 32767, -32768, 32767, -32768, 32766, -32768, -32768, -32768, -32768,
                -32768, 8192 };
            using var dIn = acc.Allocate1D<float>(x.Length);
            dIn.View.CopyFromCPU(x);
            using var dOut = acc.Allocate1D<short>(x.Length);
            var k = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<short>>(PcmConvertKernels.FloatToS16Kernel);
            k(x.Length, dIn.View, dOut.View);
            var got = await dOut.CopyToHostAsync();
            var bad = Enumerable.Range(0, x.Length).Where(i => got[i] != want[i]).Select(i => $"{x[i]} -> {got[i]} (want {want[i]})").ToList();
            True(bad.Count == 0, string.Join(", ", bad));

            // float input of exact int16 values encodes to the same file as the int16 input
            var raw = Gunzip(ReadEmbedded(OggOpusPrefix + "mono48.s16.gz"));
            var pcm = new short[raw.Length / 2];
            Buffer.BlockCopy(raw, 0, pcm, 0, raw.Length);
            var o = new OggOpusEncodeOptions { Bitrate = 64000 };
            var a = await OggOpusEncoder.EncodeAsync(acc, pcm, 48000, 1, o);
            var b = await OggOpusEncoder.EncodeAsync(acc, pcm.Select(v => v / 32768f).ToArray(), 48000, 1, o);
            True(a.AsSpan().SequenceEqual(b), $"float input file differs at byte {FirstDifference(a, b)}");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }
}
