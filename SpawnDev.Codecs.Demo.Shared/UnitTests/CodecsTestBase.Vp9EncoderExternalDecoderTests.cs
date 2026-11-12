// Vp9EncoderGpu against an independent decoder. Our own Vp9DecoderGpu agreeing with the encoder's reconstruction is not
// enough: both could share one wrong reading of the VP9 specification. This test muxes the encoder's packets into a WebM
// file and has FFmpeg (run as a black-box executable) decode it; FFmpeg's pictures must equal the encoder's reconstruction
// byte for byte, for every frame, with the motion search on and off, at one size that is not a multiple of 8 in height and one in neither axis.

using SpawnDev.Codecs.Container.Matroska;
using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod(Timeout = 1800000, Category = "NotOnWebGL,GiantShader,DesktopOnly")]
    public async Task Vp9EncoderGpu_FfmpegDecodesWhatTheEncoderReconstructs()
    {
        FfmpegOracle.RequireAvailable();
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            const int frames = 8, q = 40;
            var failures = new List<string>();
            var report = new List<string>();
            foreach (var (w, h) in new[] { (176, 100), (70, 50) })
                foreach (bool search in new[] { true, false })
                {
                    string tag = $"{w}x{h} search {(search ? "on" : "off")}";
                    int cw = (w + 1) / 2, ch = (h + 1) / 2;
                    var recons = new List<byte[]>();
                    var packets = new List<byte[]>();
                    var keys = new List<bool>();
                    using (var enc = new Vp9EncoderGpu(acc) { MotionSearch = search })
                        for (int n = 0; n < frames; n++)
                        {
                            var y = new byte[w * h]; var u = new byte[cw * ch]; var v = new byte[cw * ch];
                            for (int r = 0; r < h; r++)
                                for (int c = 0; c < w; c++) y[r * w + c] = Vp9TestTexture(c - 3 * n + 64, r - 2 * n + 64);
                            for (int r = 0; r < ch; r++)
                                for (int c = 0; c < cw; c++)
                                {
                                    u[r * cw + c] = (byte)(64 + Vp9TestTexture(2 * c - 3 * n + 300, 2 * r - 2 * n + 300) / 2);
                                    v[r * cw + c] = (byte)(64 + Vp9TestTexture(2 * c - 3 * n + 600, 2 * r - 2 * n + 600) / 2);
                                }
                            var (bytes, isKey) = await enc.EncodeFrameAsync(y, u, v, w, h, q);
                            packets.Add(bytes); keys.Add(isKey);
                            recons.Add(await enc.ReadReconAsync());
                        }
                    var file = new MemoryStream();
                    using (var mux = new WebMWriter(file, leaveOpen: true))
                    {
                        int track = mux.AddVideoTrack(new WebMVideoTrack { CodecId = "V_VP9", Width = w, Height = h, DefaultDurationNs = 62_500_000 });
                        for (int n = 0; n < frames; n++)
                            mux.WriteFrame(track, packets[n], (1000L * n + 8) / 16, keys[n], 63);   // 16 fps
                        mux.Finish();
                    }
                    var decoded = FfmpegOracle.DecodeVideoI420(file.ToArray(), ".webm", w, h);
                    if (decoded.Count != frames) { failures.Add($"{tag}: ffmpeg decoded {decoded.Count} frames, expected {frames}"); continue; }
                    for (int n = 0; n < frames; n++)
                    {
                        int first = -1, count = 0;
                        for (int i = 0; i < Math.Min(recons[n].Length, decoded[n].Length); i++)
                            if (recons[n][i] != decoded[n][i]) { if (first < 0) first = i; count++; }
                        if (first >= 0 || recons[n].Length != decoded[n].Length)
                            failures.Add($"{tag} frame {n}: ffmpeg != encoder reconstruction, first differing byte {first} of {recons[n].Length}, " +
                                $"{count} bytes differ (lengths {decoded[n].Length} / {recons[n].Length})");
                    }
                    int total = 0;
                    foreach (var p in packets) total += p.Length;
                    report.Add($"{tag}: {total} B");
                }
            Console.WriteLine($"{acc.AcceleratorType}: ffmpeg decode of {frames} frames at q{q}: " + string.Join(" | ", report));
            True(failures.Count == 0, string.Join("\n", failures));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }
}
