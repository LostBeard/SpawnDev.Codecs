// VP8 multi-token-partition decode check. ffmpeg/libvpx encodes BBB at
// 256x144 with multiple token partitions (Log2NumPartitions > 0); our
// Vp8KeyframeWalker rejected those before today's fix. Verify the
// walker now decodes them and the result is reasonable vs ffmpeg.

#:project ../../SpawnDev.Codecs/SpawnDev.Codecs.csproj
using System;
using System.Diagnostics;
using System.IO;
using SpawnDev.Codecs.Container.Ivf;
using SpawnDev.Codecs.Video.Vp8;

const int W = 256, H = 144;
string ffmpeg = "C:\\Users\\TJ\\AppData\\Local\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\\ffmpeg-8.1-full_build\\bin\\ffmpeg.exe";
string outDir = Path.Combine(Path.GetTempPath(), "vp8_multi_part");
Directory.CreateDirectory(outDir);

// Encode our OWN multi-partition VP8 frames at each log2NumPartitions and
// round-trip them through Vp8KeyframeWalker. This proves the encoder +
// walker symmetric multi-partition path works.
string yuvIn = Path.Combine(outDir, "src.yuv");
RunFfmpeg($"-y -f lavfi -i testsrc=size={W}x{H}:rate=30:duration=1 -frames:v 1 -f rawvideo -pix_fmt yuv420p \"{yuvIn}\"");

var srcBytes = File.ReadAllBytes(yuvIn);
int yLen = W * H;
int uvLen = (W / 2) * (H / 2);
var ySrc = srcBytes[0..yLen];
var uSrc = srcBytes[yLen..(yLen + uvLen)];
var vSrc = srcBytes[(yLen + uvLen)..(yLen + 2 * uvLen)];

foreach (int log2Parts in new[] { 0, 1, 2, 3 })
{
    int npart = 1 << log2Parts;
    var encoded = Vp8KeyframeEncoder.EncodeKeyFrame(
        ySrc, W, uSrc, W / 2, vSrc, W, H,
        baseQIndex: 30, log2NumPartitions: log2Parts);
    string ivf = Path.Combine(outDir, $"vp8_p{npart}_ours.ivf");
    using (var fs = File.Create(ivf))
    {
        var w = new IvfWriter(fs, "VP80", W, H, frameRate: 1, timeScale: 30, numFrames: 0, leaveOpen: true);
        w.WriteFrame(encoded, 0); w.Finish();
    }

    // Parse via our walker.
    var frame = encoded;
    var tag = Vp8FrameTagParser.Parse(frame.AsSpan());
    int firstPartOffset = 10;
    int firstPartLen = tag.FirstPartitionSize;
    var firstPart = new byte[firstPartLen];
    Buffer.BlockCopy(frame, firstPartOffset, firstPart, 0, firstPartLen);
    var bd = new Vp8BoolDecoder(firstPart);
    var hdr = Vp8FrameHeaderParser.ParseKeyFrameHeader(bd);
    int tokenOffset = firstPartOffset + firstPartLen;
    var tokenBytes = new byte[frame.Length - tokenOffset];
    Buffer.BlockCopy(frame, tokenOffset, tokenBytes, 0, tokenBytes.Length);

    var fb = new Vp8FrameBuffer(tag.Width!.Value, tag.Height!.Value);
    var ec = new Vp8EntropyContexts(fb.MbCols);
    string verdict = "OK";
    string detail = "";
    try
    {
        Vp8KeyframeWalker.Decode(tag, hdr, bd, tokenBytes, fb, ec);
        long ySum = 0; int yMin = 255, yMax = 0;
        for (int row = 0; row < H; row++)
            for (int col = 0; col < W; col++)
            {
                int v = fb.YPlane[row * fb.YStride + col];
                ySum += v; if (v < yMin) yMin = v; if (v > yMax) yMax = v;
            }
        detail = $"Y mean={ySum / (W * H)} range=[{yMin},{yMax}]";
    }
    catch (Exception ex)
    {
        verdict = "FAIL";
        detail = ex.Message;
    }

    Console.WriteLine($"Log2NumPartitions={log2Parts} ({npart} partitions): hdr says {hdr.Log2NumPartitions} | {verdict} | {detail}");
}

// Also verify ffmpeg's libvpx-vp8 decoder accepts our multi-partition output.
Console.WriteLine();
Console.WriteLine("ffmpeg native VP8 decode of our multi-partition outputs:");
foreach (int log2Parts in new[] { 0, 1, 2, 3 })
{
    int npart = 1 << log2Parts;
    string ivf = Path.Combine(outDir, $"vp8_p{npart}_ours.ivf");
    string yuv = Path.Combine(outDir, $"vp8_p{npart}_ours.yuv");
    var p = Process.Start(new ProcessStartInfo(ffmpeg,
        $"-y -i \"{ivf}\" -f rawvideo -pix_fmt yuv420p \"{yuv}\"")
    { RedirectStandardError = true, UseShellExecute = false, CreateNoWindow = true })!;
    p.StandardError.ReadToEnd();
    p.WaitForExit();
    string verdict = p.ExitCode == 0 ? "OK" : "REJECT";
    string detail = "";
    if (p.ExitCode == 0)
    {
        var dec = File.ReadAllBytes(yuv);
        long ySum = 0; for (int i = 0; i < yLen; i++) ySum += dec[i];
        detail = $"Y mean={ySum / yLen}";
    }
    Console.WriteLine($"  p{npart}: {verdict} | {detail}");
}

void RunFfmpeg(string args)
{
    var p = Process.Start(new ProcessStartInfo(ffmpeg, args) { RedirectStandardError = true, UseShellExecute = false, CreateNoWindow = true })!;
    p.StandardError.ReadToEnd();
    p.WaitForExit();
    if (p.ExitCode != 0) throw new Exception($"ffmpeg failed: {args}");
}
