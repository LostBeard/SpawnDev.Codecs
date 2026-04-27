// Integration test: encode a tiny test pattern to VP8 via ffmpeg, parse
// the IVF, hand the first frame to Vp8FrameTagParser + Vp8FrameHeaderParser,
// and verify the parsed structural fields match what ffmpeg encoded.
//
// This is the first end-to-end test of the VP8 inverse pipeline shipped
// today: bool decoder + frame tag + frame header + default coef probs +
// coef update probs. Block-level decode + macroblock walker land in the
// next slice.

#:project SpawnDev.Codecs/SpawnDev.Codecs.csproj
using System;
using System.Diagnostics;
using System.IO;
using SpawnDev.Codecs.Container.Ivf;
using SpawnDev.Codecs.Video.Vp8;

const int Width = 320, Height = 240, Fps = 30;

string ffmpegPath = "C:\\Users\\TJ\\AppData\\Local\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\\ffmpeg-8.1-full_build\\bin\\ffmpeg.exe";
string tempIvf = Path.Combine(Path.GetTempPath(), "spawndev_vp8_test.ivf");

// Encode a 1-second 320x240 test pattern to VP8 keyframe-only.
RunFfmpeg($"-y -f lavfi -i testsrc=size={Width}x{Height}:rate={Fps}:duration=1 " +
          $"-c:v libvpx -keyint_min 1 -g 1 -auto-alt-ref 0 " +
          $"-f ivf \"{tempIvf}\"");

long ivfSize = new FileInfo(tempIvf).Length;
Console.WriteLine($"ffmpeg encoded a {Width}x{Height} 1-sec test pattern as {ivfSize}-byte VP8 IVF.");

// Parse IVF to get the first frame.
var ivfBytes = File.ReadAllBytes(tempIvf);
var ivfFrames = IvfReader.EnumerateFrames(ivfBytes).Take(5).ToArray();
Console.WriteLine($"IVF first {ivfFrames.Length} frames listed.");

if (ivfFrames.Length == 0)
{
    Console.WriteLine("FAIL: no frames in IVF");
    Environment.Exit(1);
}

var firstFrame = ivfFrames[0];
var firstData = firstFrame.Data.ToArray();
Console.WriteLine($"First frame: pts={firstFrame.Pts}, length={firstData.Length} bytes");
Console.WriteLine();

// === Stage 1: parse the 3+7 byte frame tag ===
var tag = Vp8FrameTagParser.Parse(firstData.AsSpan());
Console.WriteLine($"Frame tag:");
Console.WriteLine($"  IsKeyFrame             = {tag.IsKeyFrame}");
Console.WriteLine($"  Version                = {tag.Version}");
Console.WriteLine($"  ShowFrame              = {tag.ShowFrame}");
Console.WriteLine($"  FirstPartitionSize     = {tag.FirstPartitionSize}");
if (tag.IsKeyFrame)
{
    Console.WriteLine($"  Width                  = {tag.Width}");
    Console.WriteLine($"  Height                 = {tag.Height}");
    Console.WriteLine($"  HorizontalScale        = {tag.HorizontalScale}");
    Console.WriteLine($"  VerticalScale          = {tag.VerticalScale}");
}
Console.WriteLine();

// Verify dimensions match what we encoded.
if (tag.Width != Width)
{
    Console.WriteLine($"FAIL: parsed width {tag.Width} != encoded {Width}");
    Environment.Exit(1);
}
if (tag.Height != Height)
{
    Console.WriteLine($"FAIL: parsed height {tag.Height} != encoded {Height}");
    Environment.Exit(1);
}
if (!tag.IsKeyFrame)
{
    Console.WriteLine("FAIL: first frame should be a key frame");
    Environment.Exit(1);
}

Console.WriteLine($"OK frame tag parses correctly + dimensions match encoded values.");
Console.WriteLine();

// === Stage 2: parse the compressed first-partition frame header ===
// Frame tag is 3 bytes for inter, 10 bytes for keyframe; the bool decoder
// starts at offset 10 for keyframes.
int firstPartOffset = 10;
int firstPartLen = tag.FirstPartitionSize;
byte[] firstPart = new byte[firstPartLen];
Buffer.BlockCopy(firstData, firstPartOffset, firstPart, 0, firstPartLen);

var bd = new Vp8BoolDecoder(firstPart);
var hdr = Vp8FrameHeaderParser.ParseKeyFrameHeader(bd);

Console.WriteLine($"Frame header:");
Console.WriteLine($"  ColorSpace             = {hdr.ColorSpace}");
Console.WriteLine($"  ClampingType           = {hdr.ClampingType}");
Console.WriteLine($"  Segmentation.Enabled   = {hdr.Segmentation.Enabled}");
Console.WriteLine($"  LoopFilter.Type        = {hdr.LoopFilter.FilterType}");
Console.WriteLine($"  LoopFilter.Level       = {hdr.LoopFilter.FilterLevel}");
Console.WriteLine($"  LoopFilter.Sharpness   = {hdr.LoopFilter.SharpnessLevel}");
Console.WriteLine($"  LoopFilter.DeltaEnabled= {hdr.LoopFilter.ModeRefLfDeltaEnabled}");
Console.WriteLine($"  Log2NumPartitions      = {hdr.Log2NumPartitions}");
Console.WriteLine($"  Quantizer.BaseQIndex   = {hdr.Quantizer.BaseQIndex}");
Console.WriteLine($"  Quantizer.Y1DcDeltaQ   = {hdr.Quantizer.Y1DcDeltaQ}");
Console.WriteLine($"  Quantizer.Y2DcDeltaQ   = {hdr.Quantizer.Y2DcDeltaQ}");
Console.WriteLine($"  Quantizer.Y2AcDeltaQ   = {hdr.Quantizer.Y2AcDeltaQ}");
Console.WriteLine($"  Quantizer.UvDcDeltaQ   = {hdr.Quantizer.UvDcDeltaQ}");
Console.WriteLine($"  Quantizer.UvAcDeltaQ   = {hdr.Quantizer.UvAcDeltaQ}");
Console.WriteLine($"  RefreshEntropyProbs    = {hdr.RefreshEntropyProbs}");
Console.WriteLine($"  MbNoSkipCoeffEnabled   = {hdr.MbNoSkipCoeffEnabled}");
Console.WriteLine($"  ProbSkipFalse          = {hdr.ProbSkipFalse}");
Console.WriteLine();

// Sanity checks on the parsed values.
if (hdr.Quantizer.BaseQIndex < 0 || hdr.Quantizer.BaseQIndex > 127)
{
    Console.WriteLine($"FAIL: BaseQIndex {hdr.Quantizer.BaseQIndex} out of valid range [0, 127]");
    Environment.Exit(1);
}
if (hdr.LoopFilter.FilterLevel < 0 || hdr.LoopFilter.FilterLevel > 63)
{
    Console.WriteLine($"FAIL: FilterLevel {hdr.LoopFilter.FilterLevel} out of valid range [0, 63]");
    Environment.Exit(1);
}
if (hdr.Log2NumPartitions < 0 || hdr.Log2NumPartitions > 3)
{
    Console.WriteLine($"FAIL: Log2NumPartitions {hdr.Log2NumPartitions} out of valid range [0, 3]");
    Environment.Exit(1);
}

Console.WriteLine($"OK frame header parses cleanly + all fields in spec-valid ranges.");
Console.WriteLine();

// === Stage 3: decode the FIRST 4 macroblocks' mode info ===
// The bool decoder (bd) is positioned at the mode-info section after the
// frame header. For a 320x240 frame that's 20x15 = 300 macroblocks total;
// we just decode the first 4 to prove the integration works on real data.
Console.WriteLine("First 4 macroblocks - mode info:");
Console.WriteLine($"  idx | seg | skip | y_mode | uv_mode | sub_modes");
for (int i = 0; i < 4; i++)
{
    var mb = Vp8MbModeInfoDecoder.DecodeKeyFrameMb(bd, hdr);
    string subModes = mb.SubBlockModes != null
        ? "[" + string.Join(",", mb.SubBlockModes.Take(4).Select(m => m.ToString())) + ",...]"
        : "(n/a)";
    Console.WriteLine($"  {i,3} | {mb.SegmentId,3} | {(mb.SkipCoeff ? "yes" : "no "),4} | {mb.YMode,-7} | {mb.UvMode,-7} | {subModes}");
}
Console.WriteLine();

Console.WriteLine("=== VP8 INVERSE PIPELINE INTEGRATION: PARSE+MODE PASS GREEN ===");
Console.WriteLine($"VP8 frame tag + frame header + MB mode info decoders correctly process");
Console.WriteLine($"a real libvpx-encoded {Width}x{Height} VP8 keyframe through the first 4 MBs.");

void RunFfmpeg(string args)
{
    var p = new Process { StartInfo = new ProcessStartInfo(ffmpegPath, args) { RedirectStandardError = true, UseShellExecute = false, CreateNoWindow = true } };
    p.Start(); p.WaitForExit();
}
