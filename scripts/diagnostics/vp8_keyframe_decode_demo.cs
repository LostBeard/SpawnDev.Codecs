// VP8 KEYFRAME DECODE DEMO - end-to-end integration of the inverse pipeline.
//
// Steps:
//   1. ffmpeg encodes a small testsrc pattern to VP8 keyframe-only IVF
//   2. Parse the IVF, get the first frame's bytes
//   3. Run the walker: tag -> header -> per-MB mode/coef/dequant/IDCT/predict
//      to produce a reconstructed YUV420 plane set
//   4. ffmpeg decodes the SAME IVF to raw YUV420 ground truth
//   5. Print mean / range / first-row sample comparison
//
// Loop filter is OUT OF SCOPE for this slice - the walker output will be
// slightly blocky compared to ffmpeg, but it should still be recognizable
// (mean / range should be close, first-row samples should show the same
// gross color pattern with small per-pixel differences).

#:project SpawnDev.Codecs/SpawnDev.Codecs.csproj
using System;
using System.Diagnostics;
using System.IO;
using SpawnDev.Codecs.Container.Ivf;
using SpawnDev.Codecs.Video.Vp8;

const int Width = 64, Height = 64, Fps = 30;
string ffmpegPath = "C:\\Users\\TJ\\AppData\\Local\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\\ffmpeg-8.1-full_build\\bin\\ffmpeg.exe";
string tempIvf = Path.Combine(Path.GetTempPath(), "spawndev_vp8_dec_demo.ivf");
string tempYuv = Path.Combine(Path.GetTempPath(), "spawndev_vp8_dec_demo.yuv");

// === Step 1: encode a 64x64 1-second testsrc pattern as VP8 keyframe-only ===
RunFfmpeg($"-y -f lavfi -i testsrc=size={Width}x{Height}:rate={Fps}:duration=1 " +
          $"-c:v libvpx -keyint_min 1 -g 1 -auto-alt-ref 0 " +
          $"-f ivf \"{tempIvf}\"");

var ivfBytes = File.ReadAllBytes(tempIvf);
var ivfFrames = IvfReader.EnumerateFrames(ivfBytes).Take(2).ToArray();
if (ivfFrames.Length == 0) { Console.WriteLine("FAIL: no frames in IVF"); Environment.Exit(1); }
var firstFrame = ivfFrames[0].Data.ToArray();
Console.WriteLine($"Encoded {Width}x{Height} test pattern, IVF size = {ivfBytes.Length}, first frame = {firstFrame.Length} bytes");

// === Step 2: parse frame tag ===
var tag = Vp8FrameTagParser.Parse(firstFrame.AsSpan());
Console.WriteLine($"Frame tag: keyframe={tag.IsKeyFrame}, dims={tag.Width}x{tag.Height}, firstPartSize={tag.FirstPartitionSize}");
if (!tag.IsKeyFrame) { Console.WriteLine("FAIL: not a keyframe"); Environment.Exit(1); }

// === Step 3: parse compressed frame header ===
int firstPartOffset = 10; // 3-byte tag + 7-byte key extension
int firstPartLen = tag.FirstPartitionSize;
byte[] firstPart = new byte[firstPartLen];
Buffer.BlockCopy(firstFrame, firstPartOffset, firstPart, 0, firstPartLen);
var bd = new Vp8BoolDecoder(firstPart);
var hdr = Vp8FrameHeaderParser.ParseKeyFrameHeader(bd);
Console.WriteLine($"Frame header: log2NumPartitions={hdr.Log2NumPartitions}, baseQ={hdr.Quantizer.BaseQIndex}, " +
                  $"filterLevel={hdr.LoopFilter.FilterLevel}, mbNoSkip={hdr.MbNoSkipCoeffEnabled}, probSkipFalse={hdr.ProbSkipFalse}");

// === Step 4: walk all macroblocks ===
var fb = new Vp8FrameBuffer(tag.Width!.Value, tag.Height!.Value);
var ec = new Vp8EntropyContexts(fb.MbCols);

// Token partition starts AFTER the first partition. For Log2NumPartitions=0
// (single token partition), it is the rest of the frame data.
int tokenPartOffset = firstPartOffset + firstPartLen;
int tokenPartLen = firstFrame.Length - tokenPartOffset;
byte[] tokenPart = new byte[tokenPartLen];
Buffer.BlockCopy(firstFrame, tokenPartOffset, tokenPart, 0, tokenPartLen);
Console.WriteLine($"Token partition: {tokenPartLen} bytes");

Vp8KeyframeWalker.Decode(tag, hdr, bd, tokenPart, fb, ec);

Console.WriteLine($"Walker complete. Y plane = {fb.YPlane.Length} bytes (stride={fb.YStride}), UV planes = {fb.UPlane.Length} each (stride={fb.UvStride})");

// === Step 5: ffmpeg decode for ground truth ===
// Decode to a flat YUV420p file. Width/height match the encode.
RunFfmpeg($"-y -i \"{tempIvf}\" -pix_fmt yuv420p -frames:v 1 -f rawvideo \"{tempYuv}\"");
var ffmpegYuv = File.ReadAllBytes(tempYuv);
int yPlaneSize = Width * Height;
int uvPlaneSize = (Width / 2) * (Height / 2);
int expectedSize = yPlaneSize + 2 * uvPlaneSize;
Console.WriteLine($"ffmpeg YUV file: got {ffmpegYuv.Length} bytes, expected {expectedSize}");
if (ffmpegYuv.Length < expectedSize) { Console.WriteLine("FAIL: ffmpeg YUV too short"); Environment.Exit(1); }

byte[] ffY = new byte[yPlaneSize];
byte[] ffU = new byte[uvPlaneSize];
byte[] ffV = new byte[uvPlaneSize];
Buffer.BlockCopy(ffmpegYuv, 0, ffY, 0, yPlaneSize);
Buffer.BlockCopy(ffmpegYuv, yPlaneSize, ffU, 0, uvPlaneSize);
Buffer.BlockCopy(ffmpegYuv, yPlaneSize + uvPlaneSize, ffV, 0, uvPlaneSize);

// === Step 6: compare statistics ===
PrintStats("ffmpeg Y", ffY, Width, Height, Width);
PrintWalkerStats("walker Y", fb.YPlane, Width, Height, fb.YStride);
PrintStats("ffmpeg U", ffU, Width / 2, Height / 2, Width / 2);
PrintWalkerStats("walker U", fb.UPlane, Width / 2, Height / 2, fb.UvStride);
PrintStats("ffmpeg V", ffV, Width / 2, Height / 2, Width / 2);
PrintWalkerStats("walker V", fb.VPlane, Width / 2, Height / 2, fb.UvStride);
Console.WriteLine();

// First row of Y plane: 16-byte sample comparison.
Console.WriteLine($"First-row Y samples (first 16 bytes):");
Console.WriteLine($"  ffmpeg : {string.Join(" ", ffY.Take(16).Select(b => b.ToString("D3")))}");
Console.WriteLine($"  walker : {string.Join(" ", Enumerable.Range(0, 16).Select(c => fb.YPlane[c].ToString("D3")))}");

// Per-pixel diff stats (Y plane only).
long sumAbsDiff = 0;
long sumSqDiff = 0;
int maxAbs = 0;
for (int r = 0; r < Height; r++)
    for (int c = 0; c < Width; c++)
    {
        int ff = ffY[r * Width + c];
        int wk = fb.YPlane[r * fb.YStride + c];
        int d = wk - ff;
        if (d < 0) d = -d;
        sumAbsDiff += d;
        sumSqDiff += d * d;
        if (d > maxAbs) maxAbs = d;
    }
double mae = (double)sumAbsDiff / (Width * Height);
double mse = (double)sumSqDiff / (Width * Height);
Console.WriteLine();
Console.WriteLine($"Y plane diff: MAE = {mae:F2}, MSE = {mse:F2}, max abs diff = {maxAbs}");

// Same for U + V.
double[] uvMae = new double[2];
int[] uvMax = new int[2];
for (int p = 0; p < 2; p++)
{
    byte[] ff = p == 0 ? ffU : ffV;
    byte[] wkPlane = p == 0 ? fb.UPlane : fb.VPlane;
    long sad = 0;
    int mx = 0;
    int w = Width / 2, h = Height / 2;
    for (int r = 0; r < h; r++)
        for (int c = 0; c < w; c++)
        {
            int dv = wkPlane[r * fb.UvStride + c] - ff[r * w + c];
            if (dv < 0) dv = -dv;
            sad += dv;
            if (dv > mx) mx = dv;
        }
    uvMae[p] = (double)sad / (w * h);
    uvMax[p] = mx;
}
Console.WriteLine($"U plane diff: MAE = {uvMae[0]:F2}, max abs diff = {uvMax[0]}");
Console.WriteLine($"V plane diff: MAE = {uvMae[1]:F2}, max abs diff = {uvMax[1]}");

Console.WriteLine();
Console.WriteLine("=== VP8 KEYFRAME DECODE DEMO COMPLETE ===");
Console.WriteLine("Walker decoded a real libvpx-encoded keyframe end-to-end.");
Console.WriteLine($"Note: loop filter is OUT OF SCOPE for this slice; per-pixel MAE will reflect the LF skip.");

void PrintStats(string label, byte[] plane, int width, int height, int stride)
{
    long sum = 0;
    int min = 255, max = 0;
    for (int r = 0; r < height; r++)
        for (int c = 0; c < width; c++)
        {
            int v = plane[r * stride + c];
            sum += v;
            if (v < min) min = v;
            if (v > max) max = v;
        }
    double mean = (double)sum / (width * height);
    Console.WriteLine($"  {label,-12}: mean = {mean:F2}, range = [{min}, {max}]");
}

void PrintWalkerStats(string label, byte[] plane, int width, int height, int stride)
    => PrintStats(label, plane, width, height, stride);

void RunFfmpeg(string args)
{
    var p = new Process { StartInfo = new ProcessStartInfo(ffmpegPath, args) { RedirectStandardError = true, UseShellExecute = false, CreateNoWindow = true } };
    p.Start();
    p.WaitForExit();
    if (p.ExitCode != 0)
    {
        Console.WriteLine($"ffmpeg failed: {p.StandardError.ReadToEnd()}");
        Environment.Exit(1);
    }
}
