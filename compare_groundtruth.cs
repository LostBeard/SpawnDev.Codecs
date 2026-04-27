// Compare ffmpeg-decoded ground truth with our decoder output for
// BBB top-left 16x16 Y plane.

using System;
using System.IO;

byte[] data = File.ReadAllBytes("bbb_frame0.yuv");
const int W = 320;
Console.WriteLine("ffmpeg ground truth - top-left 16x16 Y:");
int min = 255, max = 0, sum = 0;
for (int r = 0; r < 16; r++)
{
    for (int c = 0; c < 16; c++)
    {
        byte v = data[r * W + c];
        if (v < min) min = v;
        if (v > max) max = v;
        sum += v;
        Console.Write($"{v,4}");
    }
    Console.WriteLine();
}
Console.WriteLine($"min={min}, max={max}, mean={sum / 256}");
