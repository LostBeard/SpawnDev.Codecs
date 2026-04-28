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

// Encode a 256x144 testsrc keyframe via libvpx with token-partitions=4
// (Log2NumPartitions=2). We use ffmpeg's --token-partitions option
// indirectly by setting -threads 4 on libvpx (which auto-picks 4
// partitions for parallel decode).
string yuvIn = Path.Combine(outDir, "src.yuv");
RunFfmpeg($"-y -f lavfi -i testsrc=size={W}x{H}:rate=30:duration=1 -frames:v 1 -f rawvideo -pix_fmt yuv420p \"{yuvIn}\"");

foreach (int log2Parts in new[] { 0, 1, 2, 3 })
{
    int npart = 1 << log2Parts;
    string ivf = Path.Combine(outDir, $"vp8_p{npart}.ivf");
    // libvpx --token-parts=N maps to log2 of N. ffmpeg lacks a direct flag
    // so we use -threads which libvpx interprets as token-partition hint
    // for n_threads <= 4 (n_partitions = next-pow-2-le-threads).
    int threads = npart;
    RunFfmpeg($"-y -f rawvideo -pix_fmt yuv420p -s {W}x{H} -i \"{yuvIn}\" -c:v libvpx -threads {threads} -keyint_min 1 -g 1 -auto-alt-ref 0 -frames:v 1 -f ivf \"{ivf}\"");

    // Parse via our walker.
    var ivfBytes = File.ReadAllBytes(ivf);
    var firstFrame = IvfReader.EnumerateFrames(ivfBytes).First();
    var frame = firstFrame.Data.ToArray();
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

    Console.WriteLine($"Log2NumPartitions={log2Parts} ({npart} partitions, ffmpeg threads={threads}): hdr says {hdr.Log2NumPartitions} | {verdict} | {detail}");
}

void RunFfmpeg(string args)
{
    var p = Process.Start(new ProcessStartInfo(ffmpeg, args) { RedirectStandardError = true, UseShellExecute = false, CreateNoWindow = true })!;
    p.StandardError.ReadToEnd();
    p.WaitForExit();
    if (p.ExitCode != 0) throw new Exception($"ffmpeg failed: {args}");
}
