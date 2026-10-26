using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Audio.Vorbis;
using SpawnDev.ILGPU;
using SpawnDev.SpawnJS.JSObjects;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// The three entry points of <see cref="VorbisAudioDecoderGpu"/> (managed, ArrayView1D, browser TypedArray) decode the
/// same libvorbis streams to the same samples, bit for bit, packet by packet (two decoder instances in step: each keeps
/// its own overlap state). The ArrayView1D overload reads a packet from a longer device buffer and writes only the frames
/// it returns into a longer, poisoned PCM view.
/// </summary>
public abstract partial class CodecsTestBase
{
    [TestMethod(Timeout = 600000)]
    public async Task VorbisAudioDecoderGpu_ArrayViewOverload_MatchesManaged_WritesOnlyItsFrames()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            var failures = new List<string>();
            int packetsChecked = 0;
            foreach (var fx in LoadVorbisFixtures())
            {
                var (ident, setup, audio) = ReadVorbisPackets(fx.Ogg);
                using var managedDec = new VorbisAudioDecoderGpu(acc, ident, setup);
                using var viewDec = new VorbisAudioDecoderGpu(acc, ident, setup);
                int cap = viewDec.MaxSamplesPerPacket;
                const int tail = 37;
                using var dPcm = acc.Allocate1D<float>(cap + tail);
                int maxPacket = audio.Max(p => p.Length);
                using var dPacket = acc.Allocate1D<byte>(maxPacket + 11);
                var poison = new float[cap + tail];
                System.Array.Fill(poison, -12345.5f);
                for (int k = 0; k < audio.Length; k++)
                {
                    float[] expected = await managedDec.DecodePacketAsync(audio[k]);
                    var padded = new byte[dPacket.Length];
                    System.Array.Fill(padded, (byte)0xC3);
                    audio[k].CopyTo(padded, 0);
                    DeviceTransfer.Upload(dPacket.View, padded);
                    dPcm.View.CopyFromCPU(poison);
                    int frames = await viewDec.DecodePacketAsync(dPacket.View, audio[k].Length, dPcm.View);
                    float[] got = await dPcm.View.CopyToHostAsync();
                    int n = frames * ident.AudioChannels;
                    packetsChecked++;
                    if (n != expected.Length) { failures.Add($"{fx.Name} packet {k}: {n} samples, managed {expected.Length}"); break; }
                    int bad = -1;
                    for (int i = 0; i < n && bad < 0; i++)
                        if (BitConverter.SingleToInt32Bits(got[i]) != BitConverter.SingleToInt32Bits(expected[i])) bad = i;
                    if (bad >= 0) { failures.Add($"{fx.Name} packet {k}: sample {bad} {got[bad]:G9} vs managed {expected[bad]:G9}"); break; }
                    for (int i = n; i < got.Length && bad < 0; i++) if (got[i] != -12345.5f) bad = i;
                    if (bad >= 0) { failures.Add($"{fx.Name} packet {k}: wrote past its {n} samples (index {bad})"); break; }
                }
            }
            True(failures.Count == 0, string.Join("\n", failures));
            True(packetsChecked > 100, $"only {packetsChecked} packets checked");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod]
    public async Task VorbisAudioDecoderGpu_ArrayViewOverload_RejectsAShortPcmView()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            var fx = LoadVorbisFixtures()[0];
            var (ident, setup, audio) = ReadVorbisPackets(fx.Ogg);
            using var dec = new VorbisAudioDecoderGpu(acc, ident, setup);
            using var dPacket = acc.Allocate1D<byte>(audio[0].Length);
            DeviceTransfer.Upload(dPacket.View, audio[0]);
            using var dShort = acc.Allocate1D<float>(dec.MaxSamplesPerPacket - 1);
            bool threw = false;
            try { await dec.DecodePacketAsync(dPacket.View, audio[0].Length, dShort.View); }
            catch (ArgumentException) { threw = true; }
            True(threw, "a PCM view shorter than MaxSamplesPerPacket must be rejected");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod(Timeout = 600000)]
    public async Task VorbisAudioDecoderGpu_TypedArrayOverload_MatchesManaged()
    {
        if (!OperatingSystem.IsBrowser())
            throw new UnsupportedTestException("TypedArray overloads need a browser accelerator.");
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            await RunVorbisTypedArrayMatchesManaged(acc);
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [System.Runtime.Versioning.SupportedOSPlatform("browser")]
    private static async Task RunVorbisTypedArrayMatchesManaged(Accelerator acc)
    {
        var failures = new List<string>();
        foreach (var fx in LoadVorbisFixtures().Take(2))
        {
            var (ident, setup, audio) = ReadVorbisPackets(fx.Ogg);
            using var managedDec = new VorbisAudioDecoderGpu(acc, ident, setup);
            using var jsDec = new VorbisAudioDecoderGpu(acc, ident, setup);
            for (int k = 0; k < audio.Length; k++)
            {
                float[] expected = await managedDec.DecodePacketAsync(audio[k]);
                using var jsPacket = new Uint8Array(audio[k].Length);
                jsPacket.Set(audio[k]);
                using var js = await jsDec.DecodePacketAsync(jsPacket);
                float[] got = (float[])js;
                if (got.Length != expected.Length) { failures.Add($"{fx.Name} packet {k}: {got.Length} samples, managed {expected.Length}"); break; }
                int bad = -1;
                for (int i = 0; i < got.Length && bad < 0; i++)
                    if (BitConverter.SingleToInt32Bits(got[i]) != BitConverter.SingleToInt32Bits(expected[i])) bad = i;
                if (bad >= 0) { failures.Add($"{fx.Name} packet {k}: sample {bad} differs"); break; }
            }
        }
        True(failures.Count == 0, string.Join("\n", failures));
    }
}
