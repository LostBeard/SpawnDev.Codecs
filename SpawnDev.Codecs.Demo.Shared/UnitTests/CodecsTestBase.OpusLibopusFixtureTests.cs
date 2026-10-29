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

    private sealed record OpusFixture(string Name, int ApiRate, int ApiChannels, int PacketUs, List<byte[]?> Packets, short[] Pcm, uint[] Ranges);

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
        byte[] rngRaw = ReadEmbedded(OpusFixturePrefix + name + ".rng");
        var ranges = new uint[rngRaw.Length / 4];
        Buffer.BlockCopy(rngRaw, 0, ranges, 0, rngRaw.Length);
        return new OpusFixture(name, int.Parse(row[1]), int.Parse(row[2]), int.Parse(row[3]), packets, pcm, ranges);
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
                if (pkt is null) { got = await gpu.DecodeLostAsync((int)((long)fx.ApiRate * fx.PacketUs / 1_000_000)); lost++; }
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
                uint rng = await gpu.GetFinalRangeAsync();
                if (rng != fx.Ranges[p])
                    throw new Exception($"{name} packet {p}: final range {rng:X8} vs libopus {fx.Ranges[p]:X8} (PCM equal so far)");
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

    // CELT-only and hybrid streams (the names carry "Celt" / "Hybrid": the WebGPU lane leaves them out until the CELT band
    // kernel is split; see PLAN-Opus-GPU-Pair).
    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusDecoderGpu_Libopus_CeltFb20StereoLoss() => await RunOpusLibopusFixture("celt_fb20_stereo_96k_loss");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusDecoderGpu_Libopus_CeltFb20MonoLowRate() => await RunOpusLibopusFixture("celt_fb20_mono_12k_low");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusDecoderGpu_Libopus_CeltFb10MonoLoss() => await RunOpusLibopusFixture("celt_fb10_mono_48k_loss");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusDecoderGpu_Libopus_CeltFb10StereoLowRate() => await RunOpusLibopusFixture("celt_fb10_stereo_24k_low");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusDecoderGpu_Libopus_CeltFb5Stereo() => await RunOpusLibopusFixture("celt_fb5_stereo_128k");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusDecoderGpu_Libopus_CeltFb5MonoLoss() => await RunOpusLibopusFixture("celt_fb5_mono_32k_loss");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusDecoderGpu_Libopus_CeltFb2p5Mono() => await RunOpusLibopusFixture("celt_fb2p5_mono_64k");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusDecoderGpu_Libopus_CeltFb2p5StereoLoss() => await RunOpusLibopusFixture("celt_fb2p5_stereo_96k_loss");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusDecoderGpu_Libopus_CeltWb20Mono16k() => await RunOpusLibopusFixture("celt_wb20_mono_16k");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusDecoderGpu_Libopus_CeltSwb20Stereo24k() => await RunOpusLibopusFixture("celt_swb20_stereo_24k");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusDecoderGpu_Libopus_HybridSwb20MonoLoss() => await RunOpusLibopusFixture("hybrid_swb20_mono_24k_loss");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusDecoderGpu_Libopus_HybridFb20StereoLoss() => await RunOpusLibopusFixture("hybrid_fb20_stereo_48k_loss");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusDecoderGpu_Libopus_HybridFb10Mono() => await RunOpusLibopusFixture("hybrid_fb10_mono_48k");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusDecoderGpu_Libopus_HybridSwb10StereoLowRate() => await RunOpusLibopusFixture("hybrid_swb10_stereo_48k_low");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusDecoderGpu_Libopus_HybridCeltModeSwitchesLoss() => await RunOpusLibopusFixture("switch_modes_48k_stereo_loss");

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
