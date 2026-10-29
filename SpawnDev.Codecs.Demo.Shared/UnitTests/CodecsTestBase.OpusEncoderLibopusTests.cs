// OpusEncoderGpu end to end vs LIBOPUS 1.5.2 (fixed point, CELT-only) on the TestData/OpusEnc streams: the Opus layer
// AND the CELT frame encoder (CeltEncoderGpu) on the accelerator; every packet must equal libopus' byte for byte and the
// encoder's final range after it must equal libopus' OPUS_GET_FINAL_RANGE. The names carry "Celt": the WebGPU lane leaves
// them out until the CELT encoder is cleared there (PLAN-Opus-GPU-Pair).

using SpawnDev.Codecs.Audio.Opus;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private async Task RunOpusEncoderLibopus(string name)
    {
        var fx = LoadOpusEncFixture(name);
        byte[] rngRaw = ReadEmbedded(OpusEncFixturePrefix + name + ".rng");
        var ranges = new uint[rngRaw.Length / 4];
        Buffer.BlockCopy(rngRaw, 0, ranges, 0, rngRaw.Length);
        Equal(fx.Packets.Count, ranges.Length, $"{name} ranges");
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var enc = CreateFixtureEncoder(acc, fx);
            int frameSize = (int)((long)fx.Rate * fx.FrameUs / 1_000_000);
            for (int p = 0; p < fx.Packets.Count; p++)
            {
                var got = await enc.EncodePacketAsync(fx.Input.AsMemory(p * frameSize * fx.Channels, frameSize * fx.Channels), frameSize, 1500);
                var want = fx.Packets[p];
                int diff = -1;
                for (int i = 0; i < Math.Min(got.Length, want.Length) && diff < 0; i++)
                    if (got[i] != want[i]) diff = i;
                if (diff >= 0 || got.Length != want.Length)
                    throw new Exception($"{name} packet {p}: {got.Length} B vs libopus {want.Length} B" +
                        (diff >= 0 ? $", first difference at byte {diff}: GPU {got[diff]:X2} vs libopus {want[diff]:X2}" : ""));
                uint rng = await enc.GetFinalRangeAsync();
                if (rng != ranges[p])
                    throw new Exception($"{name} packet {p}: final range {rng:X8} vs libopus {ranges[p]:X8} (packet bytes equal)");
            }
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusCelt_Cbr20Mono64kC5() => await RunOpusEncoderLibopus("enc_cbr20_mono_64k_c5");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusCelt_Cbr20Mono64kC5Speech() => await RunOpusEncoderLibopus("enc_cbr20_mono_64k_c5_speech");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusCelt_Cbr10Mono48kC5() => await RunOpusEncoderLibopus("enc_cbr10_mono_48k_c5");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusCelt_Cbr5Mono48kC5() => await RunOpusEncoderLibopus("enc_cbr5_mono_48k_c5");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusCelt_Cbr2p5Mono64kC5() => await RunOpusEncoderLibopus("enc_cbr2p5_mono_64k_c5");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusCelt_Cbr40Mono64kC5() => await RunOpusEncoderLibopus("enc_cbr40_mono_64k_c5");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusCelt_Cbr60Mono64kC5() => await RunOpusEncoderLibopus("enc_cbr60_mono_64k_c5");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusCelt_Cbr20Stereo96kC5() => await RunOpusEncoderLibopus("enc_cbr20_stereo_96k_c5");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusCelt_Cbr10Stereo64kC5() => await RunOpusEncoderLibopus("enc_cbr10_stereo_64k_c5");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusCelt_Cbr5Stereo128kC5() => await RunOpusEncoderLibopus("enc_cbr5_stereo_128k_c5");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusCelt_Cbr2p5Stereo96kC5() => await RunOpusEncoderLibopus("enc_cbr2p5_stereo_96k_c5");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusCelt_Cbr20Stereo24kC5Low() => await RunOpusEncoderLibopus("enc_cbr20_stereo_24k_c5_low");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusCelt_Cbr20Mono12kC5Low() => await RunOpusEncoderLibopus("enc_cbr20_mono_12k_c5_low");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusCelt_Cbr20Stereo64kC0() => await RunOpusEncoderLibopus("enc_cbr20_stereo_64k_c0");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusCelt_Cbr20Stereo64kC1() => await RunOpusEncoderLibopus("enc_cbr20_stereo_64k_c1");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusCelt_Cbr20Stereo64kC2() => await RunOpusEncoderLibopus("enc_cbr20_stereo_64k_c2");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusCelt_Cbr20Stereo64kC3() => await RunOpusEncoderLibopus("enc_cbr20_stereo_64k_c3");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusCelt_Cbr20Stereo64kC4() => await RunOpusEncoderLibopus("enc_cbr20_stereo_64k_c4");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusCelt_Cbr20Stereo64kC6() => await RunOpusEncoderLibopus("enc_cbr20_stereo_64k_c6");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusCelt_Cbr20Stereo64kC7() => await RunOpusEncoderLibopus("enc_cbr20_stereo_64k_c7");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusCelt_Cbr20Stereo64kC8() => await RunOpusEncoderLibopus("enc_cbr20_stereo_64k_c8");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusCelt_Cbr20Stereo64kC9() => await RunOpusEncoderLibopus("enc_cbr20_stereo_64k_c9");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusCelt_Vbr20Mono64kC5() => await RunOpusEncoderLibopus("enc_vbr20_mono_64k_c5");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusCelt_Cvbr20Mono64kC5() => await RunOpusEncoderLibopus("enc_cvbr20_mono_64k_c5");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusCelt_Vbr20Stereo96kC9() => await RunOpusEncoderLibopus("enc_vbr20_stereo_96k_c9");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusCelt_Cvbr10Stereo64kC7() => await RunOpusEncoderLibopus("enc_cvbr10_stereo_64k_c7");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusCelt_Vbr20Mono32kC5Speech() => await RunOpusEncoderLibopus("enc_vbr20_mono_32k_c5_speech");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusCelt_Cbr20Mono24kRate16kC5() => await RunOpusEncoderLibopus("enc_cbr20_mono_24k_rate16k_c5");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusCelt_Cbr20Stereo48kRate24kC5() => await RunOpusEncoderLibopus("enc_cbr20_stereo_48k_rate24k_c5");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusCelt_Cbr20Mono64kC5Bwswb() => await RunOpusEncoderLibopus("enc_cbr20_mono_64k_c5_bwswb");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusCelt_Lowdelay20Stereo64kC5() => await RunOpusEncoderLibopus("enc_lowdelay20_stereo_64k_c5");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusCelt_Lowdelay5Mono32kVbrC5() => await RunOpusEncoderLibopus("enc_lowdelay5_mono_32k_vbr_c5");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusCelt_Voip20Mono32kC5() => await RunOpusEncoderLibopus("enc_voip20_mono_32k_c5");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusCelt_Voip10Stereo48kCvbrC6() => await RunOpusEncoderLibopus("enc_voip10_stereo_48k_cvbr_c6");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusCelt_Cbr20Stereo12kC5Tomono() => await RunOpusEncoderLibopus("enc_cbr20_stereo_12k_c5_tomono");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusCelt_Vbr40Mono48kC5() => await RunOpusEncoderLibopus("enc_vbr40_mono_48k_c5");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusCelt_Vbr60Stereo64kC7() => await RunOpusEncoderLibopus("enc_vbr60_stereo_64k_c7");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusCelt_Cvbr120Mono32kC5() => await RunOpusEncoderLibopus("enc_cvbr120_mono_32k_c5");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusCelt_Cbr100Stereo48kC5() => await RunOpusEncoderLibopus("enc_cbr100_stereo_48k_c5");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusCelt_Cbr2p5Mono6kToosmall() => await RunOpusEncoderLibopus("enc_cbr2p5_mono_6k_toosmall");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusCelt_Vbr2p5Stereo6kToosmall() => await RunOpusEncoderLibopus("enc_vbr2p5_stereo_6k_toosmall");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusCelt_Cbr20Mono16kRate8kC5() => await RunOpusEncoderLibopus("enc_cbr20_mono_16k_rate8k_c5");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusCelt_Vbr10Stereo32kRate12kC5() => await RunOpusEncoderLibopus("enc_vbr10_stereo_32k_rate12k_c5");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusCelt_Cbr20Stereo32kC5Forcemono() => await RunOpusEncoderLibopus("enc_cbr20_stereo_32k_c5_forcemono");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusCelt_Default20Stereo() => await RunOpusEncoderLibopus("enc_default20_stereo");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusCelt_Bmax20MonoCbrC5() => await RunOpusEncoderLibopus("enc_bmax20_mono_cbr_c5");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusCelt_Vbr40Stereo96kC5Gap() => await RunOpusEncoderLibopus("enc_vbr40_stereo_96k_c5_gap");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusCelt_Vbr60Mono64kC5Gap() => await RunOpusEncoderLibopus("enc_vbr60_mono_64k_c5_gap");

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task OpusEncoderGpu_LibopusCelt_Cbr20Mono64kC5Gap() => await RunOpusEncoderLibopus("enc_cbr20_mono_64k_c5_gap");
}
