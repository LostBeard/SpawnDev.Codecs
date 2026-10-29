// OpusEncoderGpu's Opus layer vs LIBOPUS 1.5.2 (fixed point) on real streams (TestData/OpusEnc,
// scripts/audio/make_opus_encoder_fixtures.py). libopus encoded each stream CELT-only; its instrumented build recorded the
// exact input and CTLs of every celt_encode_with_ec call (.celtin). Here the encoder runs with the CELT frame encode
// replaced by the libopus CELT bytes of that frame, so everything else is checked exactly: every CELT call's int16 input
// (delay compensation, DC reject / VOIP high-pass, stereo width fade), its CTLs (byte budget, bitrate, VBR flags, end band,
// stream channels, prediction), and every packet's bytes (TOC, code 1 / 3 framing, CBR padding).

using System.Text;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Audio.Opus;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private const string OpusEncFixturePrefix = "SpawnDev.Codecs.Demo.Shared.TestData.OpusEnc.";

    private sealed record OpusEncCeltCall(int[] Header, short[] Pcm);

    private sealed record OpusEncFixture(string Name, int Rate, int Channels, int FrameUs, string Application, int Bitrate, bool Vbr,
        bool Cvbr, int Complexity, string Bandwidth, int ForceChannels, List<byte[]> Packets, short[] Input, List<OpusEncCeltCall> Calls);

    private static OpusEncFixture LoadOpusEncFixture(string name)
    {
        string manifest = Encoding.UTF8.GetString(ReadEmbedded(OpusEncFixturePrefix + "manifest.tsv"));
        var c = manifest.Split('\n').Select(l => l.TrimEnd('\r').Split('\t')).First(r => r[0] == name);
        byte[] stream = ReadEmbedded(OpusEncFixturePrefix + name + ".opusp");
        var packets = new List<byte[]>();
        for (int i = 0; i < stream.Length;)
        {
            int len = stream[i] | (stream[i + 1] << 8);
            packets.Add(stream.AsSpan(i + 2, len).ToArray());
            i += 2 + len;
        }
        byte[] raw = ReadEmbedded(OpusEncFixturePrefix + name + ".pcmin");
        var input = new short[raw.Length / 2];
        Buffer.BlockCopy(raw, 0, input, 0, raw.Length);
        byte[] celtin = ReadEmbedded(OpusEncFixturePrefix + name + ".celtin");
        var calls = new List<OpusEncCeltCall>();
        for (int at = 0; at < celtin.Length;)
        {
            var header = new int[20];
            Buffer.BlockCopy(celtin, at, header, 0, 80);
            var pcm = new short[header[1] * header[2]];
            Buffer.BlockCopy(celtin, at + 80, pcm, 0, pcm.Length * 2);
            calls.Add(new OpusEncCeltCall(header, pcm));
            at += 80 + pcm.Length * 2;
        }
        return new OpusEncFixture(name, int.Parse(c[1]), int.Parse(c[2]), int.Parse(c[3]), c[4], int.Parse(c[5]), c[6] == "1", c[7] == "1",
            int.Parse(c[8]), c[9], int.Parse(c[11]), packets, input, calls);
    }

    private async Task RunOpusEncoderLibopusLayer(string name)
    {
        var fx = LoadOpusEncFixture(name);
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            var app = fx.Application switch
            {
                "voip" => OpusEncoderApplication.Voip, "lowdelay" => OpusEncoderApplication.RestrictedLowDelay, _ => OpusEncoderApplication.Audio,
            };
            using var enc = new OpusEncoderGpu(acc, fx.Rate, fx.Channels, app)
            {
                Bitrate = fx.Bitrate, Vbr = fx.Vbr, VbrConstraint = fx.Cvbr, Complexity = fx.Complexity,
                Bandwidth = fx.Bandwidth switch
                {
                    "NB" => OpusBandwidth.Narrowband, "WB" => OpusBandwidth.Wideband, "SWB" => OpusBandwidth.Superwideband,
                    "FB" => OpusBandwidth.Fullband, _ => null,
                },
                ForceChannels = fx.ForceChannels == -1000 ? null : fx.ForceChannels,
            };
            int packet = 0, frameInPacket = 0, calls = 0;
            string Where(int call) => $"{name} packet {packet} CELT call {call}";
            enc.CeltEncode = async call =>
            {
                if (call.Call >= fx.Calls.Count) throw new Exception($"{Where(call.Call)}: libopus made only {fx.Calls.Count} CELT calls");
                calls++;
                var want = fx.Calls[call.Call];
                int[] h = want.Header;
                void Same(string what, int got, int expected)
                {
                    if (got != expected) throw new Exception($"{Where(call.Call)}: {what} {got} vs libopus {expected}");
                }
                Same("frame_size", call.FrameSize, h[1]);
                Same("CC", call.Channels, h[2]);
                Same("nbCompressedBytes", call.NbCompressedBytes, h[3]);
                Same("standalone", 0, h[5]);
                Same("bitrate", call.Bitrate, h[6]);
                Same("vbr", call.Vbr ? 1 : 0, h[7]);
                Same("cvbr", call.ConstrainedVbr ? 1 : 0, h[8]);
                Same("complexity", call.Complexity, h[9]);
                Same("start", call.StartBand, h[10]);
                Same("end", call.EndBand, h[11]);
                Same("stream channels", call.StreamChannels, h[12]);
                Same("lsb_depth", call.LsbDepth, h[13]);
                Same("disable_pf", call.DisablePrefilter ? 1 : 0, h[14]);
                Same("force_intra", call.ForceIntra ? 1 : 0, h[15]);
                Same("loss_rate", call.LossRate, h[18]);
                var pcm = await call.Pcm.SubView(0, want.Pcm.Length).CopyToHostAsync();
                for (int i = 0; i < pcm.Length; i++)
                    if (pcm[i] != want.Pcm[i])
                        throw new Exception($"{Where(call.Call)}: CELT input sample {i / call.Channels} ch {i % call.Channels}: GPU {pcm[i]} vs libopus {want.Pcm[i]}");
                // Stand in for the CELT encoder: libopus' CELT bytes of this frame.
                if (!OpusPacketParser.TryParse(fx.Packets[packet], false, out var parsed, out var error) || parsed is null)
                    throw new Exception($"{name} packet {packet}: libopus packet does not parse ({error})");
                byte[] celt = parsed.Frames[frameInPacket++].ToArray();
                if (celt.Length > call.NbCompressedBytes) throw new Exception($"{Where(call.Call)}: libopus frame {celt.Length} B > budget");
                if (celt.Length > 0) call.Out.SubView(call.OutAt, celt.Length).CopyFromCPU(celt);
                call.Length.SubView(call.LengthAt, 1).CopyFromCPU(new[] { celt.Length });
            };
            int frameSize = (int)((long)fx.Rate * fx.FrameUs / 1_000_000);
            for (packet = 0; packet < fx.Packets.Count; packet++)
            {
                frameInPacket = 0;
                var got = await enc.EncodeAsync(fx.Input.AsMemory(packet * frameSize * fx.Channels, frameSize * fx.Channels), frameSize, 1500);
                var want = fx.Packets[packet];
                if (got.Length != want.Length) throw new Exception($"{name} packet {packet}: {got.Length} B vs libopus {want.Length} B");
                for (int i = 0; i < got.Length; i++)
                    if (got[i] != want[i])
                        throw new Exception($"{name} packet {packet} byte {i}: GPU {got[i]:X2} vs libopus {want[i]:X2}");
            }
            Equal(fx.Calls.Count, calls, $"{name} CELT calls");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusLayer_Cbr20Mono64kC5() => await RunOpusEncoderLibopusLayer("enc_cbr20_mono_64k_c5");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusLayer_Cbr20Mono64kC5Speech() => await RunOpusEncoderLibopusLayer("enc_cbr20_mono_64k_c5_speech");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusLayer_Cbr10Mono48kC5() => await RunOpusEncoderLibopusLayer("enc_cbr10_mono_48k_c5");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusLayer_Cbr5Mono48kC5() => await RunOpusEncoderLibopusLayer("enc_cbr5_mono_48k_c5");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusLayer_Cbr2p5Mono64kC5() => await RunOpusEncoderLibopusLayer("enc_cbr2p5_mono_64k_c5");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusLayer_Cbr40Mono64kC5() => await RunOpusEncoderLibopusLayer("enc_cbr40_mono_64k_c5");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusLayer_Cbr60Mono64kC5() => await RunOpusEncoderLibopusLayer("enc_cbr60_mono_64k_c5");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusLayer_Cbr20Stereo96kC5() => await RunOpusEncoderLibopusLayer("enc_cbr20_stereo_96k_c5");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusLayer_Cbr10Stereo64kC5() => await RunOpusEncoderLibopusLayer("enc_cbr10_stereo_64k_c5");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusLayer_Cbr5Stereo128kC5() => await RunOpusEncoderLibopusLayer("enc_cbr5_stereo_128k_c5");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusLayer_Cbr2p5Stereo96kC5() => await RunOpusEncoderLibopusLayer("enc_cbr2p5_stereo_96k_c5");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusLayer_Cbr20Stereo24kC5Low() => await RunOpusEncoderLibopusLayer("enc_cbr20_stereo_24k_c5_low");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusLayer_Cbr20Mono12kC5Low() => await RunOpusEncoderLibopusLayer("enc_cbr20_mono_12k_c5_low");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusLayer_Cbr20Stereo64kC0() => await RunOpusEncoderLibopusLayer("enc_cbr20_stereo_64k_c0");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusLayer_Cbr20Stereo64kC1() => await RunOpusEncoderLibopusLayer("enc_cbr20_stereo_64k_c1");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusLayer_Cbr20Stereo64kC2() => await RunOpusEncoderLibopusLayer("enc_cbr20_stereo_64k_c2");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusLayer_Cbr20Stereo64kC3() => await RunOpusEncoderLibopusLayer("enc_cbr20_stereo_64k_c3");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusLayer_Cbr20Stereo64kC4() => await RunOpusEncoderLibopusLayer("enc_cbr20_stereo_64k_c4");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusLayer_Cbr20Stereo64kC6() => await RunOpusEncoderLibopusLayer("enc_cbr20_stereo_64k_c6");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusLayer_Cbr20Stereo64kC7() => await RunOpusEncoderLibopusLayer("enc_cbr20_stereo_64k_c7");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusLayer_Cbr20Stereo64kC8() => await RunOpusEncoderLibopusLayer("enc_cbr20_stereo_64k_c8");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusLayer_Cbr20Stereo64kC9() => await RunOpusEncoderLibopusLayer("enc_cbr20_stereo_64k_c9");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusLayer_Vbr20Mono64kC5() => await RunOpusEncoderLibopusLayer("enc_vbr20_mono_64k_c5");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusLayer_Cvbr20Mono64kC5() => await RunOpusEncoderLibopusLayer("enc_cvbr20_mono_64k_c5");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusLayer_Vbr20Stereo96kC9() => await RunOpusEncoderLibopusLayer("enc_vbr20_stereo_96k_c9");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusLayer_Cvbr10Stereo64kC7() => await RunOpusEncoderLibopusLayer("enc_cvbr10_stereo_64k_c7");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusLayer_Vbr20Mono32kC5Speech() => await RunOpusEncoderLibopusLayer("enc_vbr20_mono_32k_c5_speech");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusLayer_Cbr20Mono24kRate16kC5() => await RunOpusEncoderLibopusLayer("enc_cbr20_mono_24k_rate16k_c5");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusLayer_Cbr20Stereo48kRate24kC5() => await RunOpusEncoderLibopusLayer("enc_cbr20_stereo_48k_rate24k_c5");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusLayer_Cbr20Mono64kC5Bwswb() => await RunOpusEncoderLibopusLayer("enc_cbr20_mono_64k_c5_bwswb");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusLayer_Lowdelay20Stereo64kC5() => await RunOpusEncoderLibopusLayer("enc_lowdelay20_stereo_64k_c5");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusLayer_Lowdelay5Mono32kVbrC5() => await RunOpusEncoderLibopusLayer("enc_lowdelay5_mono_32k_vbr_c5");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusLayer_Voip20Mono32kC5() => await RunOpusEncoderLibopusLayer("enc_voip20_mono_32k_c5");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusLayer_Voip10Stereo48kCvbrC6() => await RunOpusEncoderLibopusLayer("enc_voip10_stereo_48k_cvbr_c6");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusLayer_Cbr20Stereo12kC5Tomono() => await RunOpusEncoderLibopusLayer("enc_cbr20_stereo_12k_c5_tomono");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusLayer_Vbr40Mono48kC5() => await RunOpusEncoderLibopusLayer("enc_vbr40_mono_48k_c5");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusLayer_Vbr60Stereo64kC7() => await RunOpusEncoderLibopusLayer("enc_vbr60_stereo_64k_c7");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusLayer_Cvbr120Mono32kC5() => await RunOpusEncoderLibopusLayer("enc_cvbr120_mono_32k_c5");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusLayer_Cbr100Stereo48kC5() => await RunOpusEncoderLibopusLayer("enc_cbr100_stereo_48k_c5");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusLayer_Cbr2p5Mono6kToosmall() => await RunOpusEncoderLibopusLayer("enc_cbr2p5_mono_6k_toosmall");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusLayer_Vbr2p5Stereo6kToosmall() => await RunOpusEncoderLibopusLayer("enc_vbr2p5_stereo_6k_toosmall");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusLayer_Cbr20Mono16kRate8kC5() => await RunOpusEncoderLibopusLayer("enc_cbr20_mono_16k_rate8k_c5");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusLayer_Vbr10Stereo32kRate12kC5() => await RunOpusEncoderLibopusLayer("enc_vbr10_stereo_32k_rate12k_c5");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusLayer_Cbr20Stereo32kC5Forcemono() => await RunOpusEncoderLibopusLayer("enc_cbr20_stereo_32k_c5_forcemono");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusLayer_Default20Stereo() => await RunOpusEncoderLibopusLayer("enc_default20_stereo");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusLayer_Bmax20MonoCbrC5() => await RunOpusEncoderLibopusLayer("enc_bmax20_mono_cbr_c5");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusLayer_Vbr40Stereo96kC5Gap() => await RunOpusEncoderLibopusLayer("enc_vbr40_stereo_96k_c5_gap");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusLayer_Vbr60Mono64kC5Gap() => await RunOpusEncoderLibopusLayer("enc_vbr60_mono_64k_c5_gap");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusLayer_Cbr20Mono64kC5Gap() => await RunOpusEncoderLibopusLayer("enc_cbr20_mono_64k_c5_gap");
}
