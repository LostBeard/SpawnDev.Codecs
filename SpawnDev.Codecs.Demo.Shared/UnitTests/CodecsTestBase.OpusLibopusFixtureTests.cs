// OpusDecoderGpu vs LIBOPUS 1.5.2 (fixed point) on real streams (TestData/Opus, scripts/audio/make_opus_fixtures.py):
// libopus encoded each stream and decoded it, including lost packets (concealment) and DTX; every packet's GPU output
// must equal libopus' int16 PCM exactly (the GPU float output is that int16 / 32768). The CPU OpusDecoder is checked on
// the stream it can decode (no loss, no DTX), so the libopus fixtures are a second, independent oracle beside Concentus.

using System.Text;
using SpawnDev.Codecs.Audio.Opus;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private const string OpusFixturePrefix = "SpawnDev.Codecs.Demo.Shared.TestData.Opus.";

    private sealed record OpusFixture(string Name, int ApiRate, int ApiChannels, int PacketMs, List<byte[]?> Packets, short[] Pcm);

    private static OpusFixture LoadOpusFixture(string name)
    {
        string manifest = Encoding.UTF8.GetString(ReadEmbedded(OpusFixturePrefix + "manifest.tsv"));
        var row = manifest.Split('\n').Select(l => l.TrimEnd('\r').Split('\t')).First(c => c[0] == name);
        byte[] stream = ReadEmbedded(OpusFixturePrefix + name + ".opusp");
        var packets = new List<byte[]?>();
        for (int i = 0; i < stream.Length;)
        {
            int len = stream[i] | (stream[i + 1] << 8);
            i += 2;
            if (len == 0xFFFF) { packets.Add(null); continue; }
            packets.Add(stream.AsSpan(i, len).ToArray());
            i += len;
        }
        byte[] raw = ReadEmbedded(OpusFixturePrefix + name + ".pcm16");
        var pcm = new short[raw.Length / 2];
        Buffer.BlockCopy(raw, 0, pcm, 0, raw.Length);
        return new OpusFixture(name, int.Parse(row[1]), int.Parse(row[2]), int.Parse(row[3]), packets, pcm);
    }

    private async Task RunOpusLibopusFixture(string name)
    {
        var fx = LoadOpusFixture(name);
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var gpu = new OpusDecoderGpu(acc, fx.ApiRate, fx.ApiChannels);
            int at = 0, lost = 0, dtx = 0;
            for (int p = 0; p < fx.Packets.Count; p++)
            {
                var pkt = fx.Packets[p];
                float[] got;
                if (pkt is null) { got = await gpu.DecodeLostAsync(fx.ApiRate * fx.PacketMs / 1000); lost++; }
                else { got = await gpu.DecodePacketAsync(pkt); if (pkt.Length == 1) dtx++; }
                for (int i = 0; i < got.Length; i++)
                {
                    if (at + i >= fx.Pcm.Length) throw new Exception($"{name} packet {p}: more samples than libopus ({fx.Pcm.Length})");
                    float want = fx.Pcm[at + i] / 32768.0f;
                    if (got[i] != want)
                        throw new Exception($"{name} packet {p} ({(pkt is null ? "LOST" : pkt.Length == 1 ? "DTX" : $"{pkt.Length} B")}): " +
                            $"sample {i / fx.ApiChannels} ch {i % fx.ApiChannels}: GPU {got[i] * 32768} vs libopus {fx.Pcm[at + i]}");
                }
                at += got.Length;
            }
            Equal(fx.Pcm.Length, at, $"{name} total samples");
            if (name.Contains("loss") && lost == 0) throw new Exception($"{name}: no lost packet");
            if (name.Contains("dtx") && dtx == 0) throw new Exception($"{name}: no DTX packet");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusDecoderGpu_Libopus_SilkWb20MonoLoss() => await RunOpusLibopusFixture("silk_wb20_mono_48k_loss");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusDecoderGpu_Libopus_SilkMb20StereoLoss() => await RunOpusLibopusFixture("silk_mb20_stereo_48k_loss");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusDecoderGpu_Libopus_SilkNb60MonoToStereoLoss() => await RunOpusLibopusFixture("silk_nb60_mono_16kst_loss");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusDecoderGpu_Libopus_SilkWb40SwitchLoss() => await RunOpusLibopusFixture("silk_wb40_switch_24k_loss");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusDecoderGpu_Libopus_SilkWb20Dtx() => await RunOpusLibopusFixture("silk_wb20_dtx_16k");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusDecoderGpu_Libopus_Silk10msClean() => await RunOpusLibopusFixture("silk_10ms_mono_8k_clean");

    /// <summary>The CPU OpusDecoder (References) on the clean libopus stream: equal to libopus.</summary>
    [TestMethod(Category = "CpuOnly")]
    public async Task OpusDecoder_Cpu_Libopus_Silk10msClean()
    {
        CpuOnlyTest();
        var fx = LoadOpusFixture("silk_10ms_mono_8k_clean");
        var cpu = OpusCodec.CreateDecoder(new OpusDecoderConfig { SampleRateHz = fx.ApiRate, ChannelCount = fx.ApiChannels });
        try
        {
            int at = 0;
            for (int p = 0; p < fx.Packets.Count; p++)
            {
                var outBuf = new float[OpusDecoderGpu.MaxSamplesPerChannel * fx.ApiChannels];
                int n = await cpu.DecodePacketAsync(fx.Packets[p]!, outBuf);
                for (int i = 0; i < n * fx.ApiChannels; i++)
                    if (outBuf[i] != fx.Pcm[at + i] / 32768.0f)
                        throw new Exception($"packet {p} sample {i}: CPU {outBuf[i] * 32768} vs libopus {fx.Pcm[at + i]}");
                at += n * fx.ApiChannels;
            }
            Equal(fx.Pcm.Length, at, "total samples");
        }
        finally { await cpu.DisposeAsync(); }
    }
}
