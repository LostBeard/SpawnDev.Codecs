// Round-trip test for Vp8FrameTagWriter + Vp8FrameTagParser.

#:project SpawnDev.Codecs/SpawnDev.Codecs.csproj
using System;
using SpawnDev.Codecs.Video.Vp8;

int totalChecks = 0;
int totalFails = 0;

void Test(string name, Vp8FrameTag input)
{
    var bytes = Vp8FrameTagWriter.WriteTag(input);
    var parsed = Vp8FrameTagParser.Parse(bytes);

    bool ok =
        parsed.IsKeyFrame == input.IsKeyFrame &&
        parsed.Version == input.Version &&
        parsed.ShowFrame == input.ShowFrame &&
        parsed.FirstPartitionSize == input.FirstPartitionSize &&
        parsed.Width == input.Width &&
        parsed.Height == input.Height &&
        parsed.HorizontalScale == input.HorizontalScale &&
        parsed.VerticalScale == input.VerticalScale;

    totalChecks++;
    if (!ok)
    {
        totalFails++;
        Console.WriteLine($"  FAIL {name}: input={input}, parsed={parsed}");
    }
    else
    {
        Console.WriteLine($"  OK   {name,-30}: bytes={bytes.Length}");
    }
}

Test("KeyFrame_320x240_v0", new Vp8FrameTag
{
    IsKeyFrame = true,
    Version = Vp8Version.Bicubic,
    ShowFrame = true,
    FirstPartitionSize = 100,
    Width = 320,
    Height = 240,
    HorizontalScale = 0,
    VerticalScale = 0,
});

Test("KeyFrame_1920x1080", new Vp8FrameTag
{
    IsKeyFrame = true,
    Version = Vp8Version.BilinearSimpleLoopFilter,
    ShowFrame = true,
    FirstPartitionSize = 50000,
    Width = 1920,
    Height = 1080,
    HorizontalScale = 0,
    VerticalScale = 0,
});

Test("KeyFrame_Hidden", new Vp8FrameTag
{
    IsKeyFrame = true,
    Version = Vp8Version.NoReconNoLoopFilter,
    ShowFrame = false,
    FirstPartitionSize = 1,
    Width = 16,
    Height = 16,
    HorizontalScale = 1,
    VerticalScale = 2,
});

Test("InterFrame_v1_Show", new Vp8FrameTag
{
    IsKeyFrame = false,
    Version = Vp8Version.BilinearSimpleLoopFilter,
    ShowFrame = true,
    FirstPartitionSize = 50,
});

Test("InterFrame_v0_Hide", new Vp8FrameTag
{
    IsKeyFrame = false,
    Version = Vp8Version.Bicubic,
    ShowFrame = false,
    FirstPartitionSize = 524287,  // 2^19 - 1, max
});

Test("KeyFrame_MaxDims", new Vp8FrameTag
{
    IsKeyFrame = true,
    Version = Vp8Version.Bicubic,
    ShowFrame = true,
    FirstPartitionSize = 0,
    Width = 0x3FFF,   // max 14-bit
    Height = 0x3FFF,
    HorizontalScale = 3,
    VerticalScale = 3,
});

Console.WriteLine();
Console.WriteLine($"VP8 frame tag round-trip: {totalChecks - totalFails}/{totalChecks} pass");
if (totalFails == 0)
    Console.WriteLine("VP8 frame tag writer + parser round-trip BIT-EXACT.");
else
    Environment.Exit(1);
