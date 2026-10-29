// SilkPacketDecoderGpu (SILK decode of Opus frames on the accelerator) vs the CPU OpusDecoder on real streams: Concentus
// encodes a varied signal SILK-only (NB / MB / WB, 10 / 20 / 40 / 60 ms, mono / stereo, in-band FEC so LBRR frames are
// present, channel and bandwidth switches mid-stream); every packet decodes on the GPU and must equal the CPU decoder's
// float output exactly, and the CPU decoder must match Concentus' decoder (the oracle) within the cross-validation
// tolerance, so the streams are real and the CPU path is the right reference.

using System.Runtime.InteropServices;
using Concentus;
using OpusApplication = Concentus.Enums.OpusApplication;
using CBandwidth = Concentus.Enums.OpusBandwidth;
using CMode = Concentus.Enums.OpusMode;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Audio.Opus;
using SpawnDev.Codecs.Audio.Opus.Silk;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    /// <summary>A speech-like test signal: a few amplitude-modulated partials with a wandering pitch and a little noise.</summary>
    private static float[] SilkTestSignal(int rateHz, int channels, int samples, int seed)
    {
        var rng = new Random(seed);
        var pcm = new float[samples * channels];
        for (int ch = 0; ch < channels; ch++)
        {
            double f0 = 140 + 40 * ch, phase = 0;
            for (int i = 0; i < samples; i++)
            {
                double t = (double)i / rateHz;
                double pitch = f0 * (1 + 0.15 * Math.Sin(2 * Math.PI * 1.3 * t));
                phase += 2 * Math.PI * pitch / rateHz;
                double env = 0.5 + 0.5 * Math.Sin(2 * Math.PI * 3.1 * t + ch);
                double v = 0;
                for (int h = 1; h <= 6; h++) v += Math.Sin(h * phase) / h;
                v = 0.25 * env * v + 0.01 * (rng.NextDouble() * 2 - 1);
                // A quiet stretch in the middle (VAD off, mid-only stereo frames).
                if (i > samples * 2 / 5 && i < samples / 2) v *= 0.002;
                pcm[i * channels + ch] = (float)v;
            }
        }
        return pcm;
    }

    private sealed record SilkStreamCase(string Label, int EncRateHz, int EncChannels, CBandwidth Bandwidth, int FrameMs,
        int ApiRateHz, int ApiChannels, bool Fec, int Packets, int Bitrate,
        Func<int, int>? ForceChannelsAt = null, Func<int, CBandwidth>? BandwidthAt = null, int ForceChannels = -1000);

    private async Task RunSilkPacketDecoderGpuStream(SilkStreamCase sc)
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            int frameSamples = sc.EncRateHz * sc.FrameMs / 1000;
            float[] signal = SilkTestSignal(sc.EncRateHz, sc.EncChannels, frameSamples * sc.Packets, seed: sc.Label.Length);
            // One encoder per bandwidth: SILK changes its internal rate only through a seconds-long transition, so a
            // bandwidth switch is built by interleaving encoders. The decoder sees exactly a TOC bandwidth change.
            var encoders = new Dictionary<CBandwidth, IOpusEncoder>();
            IOpusEncoder EncoderFor(CBandwidth bw)
            {
                if (encoders.TryGetValue(bw, out var e)) return e;
                e = OpusCodecFactory.CreateEncoder(sc.EncRateHz, sc.EncChannels, OpusApplication.OPUS_APPLICATION_VOIP);
                e.ForceMode = CMode.MODE_SILK_ONLY;
                e.Bitrate = sc.Bitrate;
                e.UseInbandFEC = sc.Fec;
                e.PacketLossPercent = sc.Fec ? 20 : 0;
                e.MaxBandwidth = bw;
                e.Bandwidth = bw;
                return encoders[bw] = e;
            }
            using var oracle = OpusCodecFactory.CreateDecoder(sc.ApiRateHz, sc.ApiChannels);
            var cpu = OpusCodec.CreateDecoder(new OpusDecoderConfig { SampleRateHz = sc.ApiRateHz, ChannelCount = sc.ApiChannels });
            using var gpu = new SilkPacketDecoderGpu(acc, sc.ApiRateHz, sc.ApiChannels);
            int maxOut = sc.ApiRateHz * 60 / 1000 * sc.ApiChannels;
            using var dOut = acc.Allocate1D<float>(maxOut);
            using var dPacket = acc.Allocate1D<byte>(1275);
            var packetBuf = new byte[1275];
            int lbrrPackets = 0, stereoPackets = 0, monoPackets = 0;
            var internalRates = new HashSet<int>();
            try
            {
                for (int p = 0; p < sc.Packets; p++)
                {
                    var enc = EncoderFor(sc.BandwidthAt?.Invoke(p) ?? sc.Bandwidth);
                    enc.ForceChannels = sc.ForceChannelsAt?.Invoke(p) ?? sc.ForceChannels; // -1000 = OPUS_AUTO
                    int bytes = enc.Encode(signal.AsSpan(p * frameSamples * sc.EncChannels, frameSamples * sc.EncChannels), frameSamples, packetBuf, packetBuf.Length);
                    if (bytes <= 0) throw new Exception($"Concentus encoder returned {bytes}");
                    var packetBytes = packetBuf.AsSpan(0, bytes).ToArray();

                    if (!OpusPacketParser.TryParse(packetBytes, selfDelimited: false, out var packet, out var err) || packet is null)
                        throw new Exception($"packet {p}: parse failed: {err}");
                    int samplesPerFrame = packet.GetSamplesPerFrame(sc.ApiRateHz);
                    int total = samplesPerFrame * packet.FrameCount;

                    var oracleOut = new float[maxOut];
                    int oracleSamples = oracle.Decode(packetBytes, oracleOut.AsSpan(), sc.ApiRateHz * 60 / 1000);
                    var cpuOut = new float[total * sc.ApiChannels];
                    int cpuSamples = await cpu.DecodePacketAsync(packetBytes, cpuOut);
                    Equal(oracleSamples, cpuSamples, $"{sc.Label} packet {p} sample count");
                    for (int i = 0; i < cpuOut.Length; i++)
                        if (Math.Abs(cpuOut[i] - oracleOut[i]) > 1e-5f)
                            throw new Exception($"{sc.Label} packet {p}: CPU OpusDecoder vs Concentus at {i}: {cpuOut[i]} vs {oracleOut[i]}");

                    // GPU: each Opus frame of the packet through SilkPacketDecoderGpu.
                    dPacket.View.SubView(0, bytes).CopyFromCPU(packetBytes);
                    int internalKHz = packet.Toc.Bandwidth switch
                    {
                        OpusBandwidth.Narrowband => 8,
                        OpusBandwidth.Mediumband => 12,
                        _ => 16,
                    };
                    internalRates.Add(internalKHz);
                    int ms = packet.Toc.GetSamplesPerFrame(48000) / 48;
                    (int silkMs, int silkCount) = ms switch { 10 => (10, 1), 20 => (20, 1), 40 => (20, 2), _ => (20, 3) };
                    int streamChannels = packet.Toc.IsStereo ? 2 : 1;
                    if (streamChannels == 2) stereoPackets++; else monoPackets++;
                    for (int fr = 0; fr < packet.FrameCount; fr++)
                    {
                        if (!MemoryMarshal.TryGetArray(packet.Frames[fr], out var seg)) throw new Exception("frame not array-backed");
                        gpu.Decode(dPacket.View, seg.Offset, seg.Count, internalKHz, silkMs, silkCount, streamChannels,
                            dOut.View, (long)fr * samplesPerFrame * sc.ApiChannels);
                    }
                    await acc.SynchronizeAsync();
                    var gpuOut = await dOut.View.SubView(0, cpuOut.Length).CopyToHostAsync();
                    for (int i = 0; i < cpuOut.Length; i++)
                        if (gpuOut[i] != cpuOut[i])
                            throw new Exception($"{sc.Label} packet {p} ({ms} ms, {internalKHz} kHz, {streamChannels} ch): GPU vs CPU at sample {i} " +
                                $"(channel {i % sc.ApiChannels}): {gpuOut[i] * 32768} vs {cpuOut[i] * 32768}");
                    var flags = await gpu.WorkView.SubView(SilkPacketDecoderGpu.WLbrrFlag, 2).CopyToHostAsync();
                    if (flags[0] != 0 || flags[1] != 0) lbrrPackets++;
                }
            }
            finally
            {
                await cpu.DisposeAsync();
                foreach (var e in encoders.Values) e.Dispose();
            }
            if (sc.EncChannels == 2 && stereoPackets == 0) throw new Exception($"{sc.Label}: no stereo packet ({monoPackets} mono)");
            if (sc.Fec && lbrrPackets == 0) throw new Exception($"{sc.Label}: no packet carried LBRR frames");
            if (sc.ForceChannelsAt is not null && (stereoPackets == 0 || monoPackets == 0))
                throw new Exception($"{sc.Label}: the stream never switched channels ({stereoPackets} stereo, {monoPackets} mono packets)");
            if (sc.BandwidthAt is not null && internalRates.Count < 2)
                throw new Exception($"{sc.Label}: the stream never switched bandwidth (internal rates {string.Join(",", internalRates)})");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task SilkPacketDecoderGpu_Wb20ms_Mono_Fec_To48k_MatchesCpu()
        => await RunSilkPacketDecoderGpuStream(new("WB-20ms-mono-48k", 16000, 1, CBandwidth.OPUS_BANDWIDTH_WIDEBAND, 20, 48000, 1, true, 30, 20000));

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task SilkPacketDecoderGpu_Mb20ms_Stereo_Fec_To48kStereo_MatchesCpu()
        => await RunSilkPacketDecoderGpuStream(new("MB-20ms-stereo-48k", 12000, 2, CBandwidth.OPUS_BANDWIDTH_MEDIUMBAND, 20, 48000, 2, true, 30, 48000, ForceChannels: 2));

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task SilkPacketDecoderGpu_Nb60ms_Mono_To16kStereo_MatchesCpu()
        => await RunSilkPacketDecoderGpuStream(new("NB-60ms-mono-16k-stereo", 8000, 1, CBandwidth.OPUS_BANDWIDTH_NARROWBAND, 60, 16000, 2, true, 12, 24000));

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task SilkPacketDecoderGpu_Wb40ms_StereoMonoSwitches_To24k_MatchesCpu()
        => await RunSilkPacketDecoderGpuStream(new("WB-40ms-switch-24k", 16000, 2, CBandwidth.OPUS_BANDWIDTH_WIDEBAND, 40, 24000, 2, true, 18, 24000,
            ForceChannelsAt: p => (p / 5) % 2 == 0 ? 2 : 1));

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task SilkPacketDecoderGpu_10ms_BandwidthSwitches_To8k_MatchesCpu()
        => await RunSilkPacketDecoderGpuStream(new("10ms-bw-switch-8k", 16000, 1, CBandwidth.OPUS_BANDWIDTH_WIDEBAND, 10, 8000, 1, false, 48, 16000,
            BandwidthAt: p => ((p / 8) % 3) switch { 0 => CBandwidth.OPUS_BANDWIDTH_WIDEBAND, 1 => CBandwidth.OPUS_BANDWIDTH_NARROWBAND, _ => CBandwidth.OPUS_BANDWIDTH_MEDIUMBAND }));
}
