// Vp9EncoderGpu in tiles (Vp9TileGrid: tile columns / rows, each tile its own bool coder; port of branch
// vp9-multi-tile-entropy with libvpx's tile boundaries). For several grids, key and inter frames:
//   - the parallel entropy and the serial walkers code the same bytes (both paths are tile-aware);
//   - the frame header carries the grid (Vp9HeaderReader reads log2 tile columns / rows back);
//   - Vp9DecoderGpu (bit-exact with libvpx) decodes every frame to the encoder's reconstruction;
//   - FFmpeg (black box, desktop) decodes the same files to the same pictures.
// 1920 wide in 4 tile columns is the case the branch's tile helper got wrong (it put the first boundary at superblock
// 8, libvpx at 7); 1912 is not a multiple of 64; 640x360 uses 4 tile rows.

using SpawnDev.Codecs.Container.Matroska;
using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.Codecs.Video.Vp9.Decoder;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    /// <summary>(width, height, log2 tile columns requested, log2 tile rows requested)</summary>
    private static readonly (int W, int H, int Cols, int Rows)[] Vp9TileCases =
    {
        (512, 512, 1, 0), (640, 360, 1, 2), (70, 50, -1, 2), (1920, 1080, 2, 0), (1912, 1080, -1, 1),
    };

    /// <summary>Where byte <paramref name="i"/> of a tight I420 picture is: plane, x, y (and the superblock column).</summary>
    private static string Vp9TilePixel(int w, int h, int i)
    {
        int cw = (w + 1) / 2, ch = (h + 1) / 2;
        if (i < 0) return "a length difference";
        if (i < w * h) return $"Y x={i % w} y={i / w} (superblock column {i % w / 64})";
        i -= w * h;
        string plane = i < cw * ch ? "U" : "V";
        if (i >= cw * ch) i -= cw * ch;
        return $"{plane} x={i % cw} y={i / cw} (superblock column {i % cw / 32})";
    }

    private static (byte[] Y, byte[] U, byte[] V) Vp9TileTestFrame(int w, int h, int n)
    {
        int cw = (w + 1) / 2, ch = (h + 1) / 2;
        var y = new byte[w * h]; var u = new byte[cw * ch]; var v = new byte[cw * ch];
        for (int r = 0; r < h; r++)
            for (int c = 0; c < w; c++) y[r * w + c] = Vp9TestTexture(c - 3 * n + 64, r - 2 * n + 64);
        for (int r = 0; r < ch; r++)
            for (int c = 0; c < cw; c++)
            {
                u[r * cw + c] = (byte)(64 + Vp9TestTexture(2 * c - 3 * n + 300, 2 * r - 2 * n + 300) / 2);
                v[r * cw + c] = (byte)(64 + Vp9TestTexture(2 * c - 3 * n + 600, 2 * r - 2 * n + 600) / 2);
            }
        return (y, u, v);
    }

    [TestMethod(Timeout = 1800000, Category = "NotOnWebGL,GiantShader")]
    public async Task Vp9EncoderGpu_Tiles_BothEntropyPathsAndDecoderAgree()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, AtomicKernels);
        try
        {
            const int frames = 3, q = 40;
            var failures = new List<string>();
            foreach (var (w, h, cols, rows) in Vp9TileCases)
            {
                var grid = Vp9TileGrid.For(w, h, cols, rows);
                string tag = $"{w}x{h} {grid.TileCols}x{grid.TileRows} tiles";
                using var serial = new Vp9EncoderGpu(acc) { TileColumnsLog2 = cols, TileRowsLog2 = rows, SerialEntropy = true };
                using var parallel = new Vp9EncoderGpu(acc) { TileColumnsLog2 = cols, TileRowsLog2 = rows, SerialEntropy = false };
                using var dec = new Vp9DecoderGpu(acc);
                long total = 0;
                for (int n = 0; n < frames; n++)
                {
                    var (y, u, v) = Vp9TileTestFrame(w, h, n);
                    byte[] a, b; bool aKey, bKey;
                    string stage = "serial";
                    try
                    {
                        (a, aKey) = await serial.EncodeFrameAsync(y, u, v, w, h, q);
                        stage = "parallel";
                        (b, bKey) = await parallel.EncodeFrameAsync(y, u, v, w, h, q);
                    }
                    catch (Exception ex)
                    {
                        failures.Add($"{tag} frame {n}: the {stage} encode threw {ex.GetType().Name}: {ex.Message}");
                        break;
                    }
                    total += b.Length;
                    string f = $"{tag} frame {n} ({(bKey ? "key" : "inter")})";
                    if (aKey != bKey || a.Length != b.Length) { failures.Add($"{f}: parallel {b.Length} B, serial {a.Length} B"); break; }
                    int first = -1;
                    for (int i = 0; i < a.Length && first < 0; i++) if (a[i] != b[i]) first = i;
                    if (first >= 0) { failures.Add($"{f}: byte {first} of {a.Length} differs between the entropy paths"); break; }
                    if (bKey)
                    {
                        // a key frame's header stands alone (inter headers need the reference bookkeeping of a decode)
                        var info = new Vp9HeaderReader().ReadFrame(b);
                        if (info.Log2TileCols != grid.Log2Cols || info.Log2TileRows != grid.Log2Rows)
                        {
                            failures.Add($"{f}: header says {1 << info.Log2TileCols}x{1 << info.Log2TileRows} tiles");
                            break;
                        }
                    }
                    var recon = await parallel.ReadReconAsync();
                    Vp9VideoFrame? shown;
                    try { shown = await dec.DecodePacketAsync(b); }
                    catch (Exception ex) { failures.Add($"{f}: the decoder threw {ex.GetType().Name}: {ex.Message}"); break; }
                    if (shown == null) { failures.Add($"{f}: the decoder shows no picture"); break; }
                    int diff = shown.Planes.Length != recon.Length ? 0 : -1;
                    for (int i = 0; i < recon.Length && diff < 0; i++) if (recon[i] != shown.Planes[i]) diff = i;
                    if (diff >= 0) { failures.Add($"{f}: decoder != encoder reconstruction at {Vp9TilePixel(w, h, diff)}"); break; }
                }
                Console.WriteLine($"[report] VP9 tiles {tag}: {frames} frames {total} B");
            }
            True(failures.Count == 0, string.Join("\n", failures));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod(Timeout = 1800000, Category = "NotOnWebGL,GiantShader,DesktopOnly")]
    public async Task Vp9EncoderGpu_Tiles_FfmpegDecodesWhatTheEncoderReconstructs()
    {
        FfmpegOracle.RequireAvailable();
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            const int frames = 4, q = 40;
            var failures = new List<string>();
            foreach (var (w, h, cols, rows) in Vp9TileCases)
            {
                var grid = Vp9TileGrid.For(w, h, cols, rows);
                string tag = $"{w}x{h} {grid.TileCols}x{grid.TileRows} tiles";
                var recons = new List<byte[]>();
                var packets = new List<byte[]>();
                var keys = new List<bool>();
                using (var enc = new Vp9EncoderGpu(acc) { TileColumnsLog2 = cols, TileRowsLog2 = rows })
                    for (int n = 0; n < frames; n++)
                    {
                        var (y, u, v) = Vp9TileTestFrame(w, h, n);
                        var (bytes, isKey) = await enc.EncodeFrameAsync(y, u, v, w, h, q);
                        packets.Add(bytes); keys.Add(isKey);
                        recons.Add(await enc.ReadReconAsync());
                    }
                var file = new MemoryStream();
                using (var mux = new WebMWriter(file, leaveOpen: true))
                {
                    int track = mux.AddVideoTrack(new WebMVideoTrack { CodecId = "V_VP9", Width = w, Height = h, DefaultDurationNs = 62_500_000 });
                    for (int n = 0; n < frames; n++) mux.WriteFrame(track, packets[n], (1000L * n + 8) / 16, keys[n], 63);
                    mux.Finish();
                }
                var decoded = FfmpegOracle.DecodeVideoI420(file.ToArray(), ".webm", w, h);
                if (decoded.Count != frames) { failures.Add($"{tag}: ffmpeg decoded {decoded.Count} frames, expected {frames}"); continue; }
                for (int n = 0; n < frames; n++)
                {
                    int first = -1;
                    for (int i = 0; i < Math.Min(recons[n].Length, decoded[n].Length) && first < 0; i++) if (recons[n][i] != decoded[n][i]) first = i;
                    if (first >= 0 || recons[n].Length != decoded[n].Length)
                        failures.Add($"{tag} frame {n}: ffmpeg != encoder reconstruction at {Vp9TilePixel(w, h, first)}");
                }
            }
            True(failures.Count == 0, string.Join("\n", failures));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }
}
