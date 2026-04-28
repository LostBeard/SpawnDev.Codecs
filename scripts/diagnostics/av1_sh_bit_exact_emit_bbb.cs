// Build a BBB-matching SH config and verify our writer emits bytes
// IDENTICAL to libaom-av1's source SH. The strongest possible
// validation that our writer is spec-equivalent: same input config
// produces the same bitstream as the reference encoder.

#:project SpawnDev.Codecs/SpawnDev.Codecs.csproj
using System;
using System.IO;
using System.Linq;
using SpawnDev.Codecs.Container.Ivf;
using SpawnDev.Codecs.Video.Av1;

var bytes = File.ReadAllBytes("SpawnDev.Codecs.Demo.Shared/TestData/bbb_180_2s.ivf");
var first = IvfReader.EnumerateFrames(bytes).First();

byte[] sourceSh = Array.Empty<byte>();
foreach (var obu in Av1ObuParser.EnumerateObus(first.Data))
{
    if (obu.Type == Av1ObuType.SequenceHeader)
    {
        sourceSh = first.Data.Slice(obu.PayloadOffset, obu.PayloadLength).ToArray();
        break;
    }
}

Console.WriteLine();
Console.WriteLine("============================================================");
Console.WriteLine("  AV1 SH writer: emit BBB-equivalent bytes from config");
Console.WriteLine("============================================================");
Console.WriteLine();
Console.WriteLine($"BBB source SH ({sourceSh.Length} bytes):");
Console.WriteLine($"  {string.Join(" ", sourceSh.Select(b => b.ToString("X2")))}");

// Build config matching what we observed from BBB's actual bits
// (see inspect_bbb_sh.cs output).
var cfg = new Av1SequenceHeaderConfig
{
    SeqProfile = 0,
    SeqLevelIdx0 = 0,
    MaxFrameWidth = 320,
    MaxFrameHeight = 180,
    BitDepth = 8,
    Monochrome = false,
    SubsamplingX = 1,
    SubsamplingY = 1,
    ColorRangeFull = false,

    Use128x128Superblock = false,
    EnableFilterIntra = true,
    EnableIntraEdgeFilter = true,
    EnableInterintraCompound = false,
    EnableMaskedCompound = true,
    EnableWarpedMotion = true,
    EnableDualFilter = false,
    EnableOrderHint = true,
    EnableJntComp = false,
    EnableRefFrameMvs = true,
    OrderHintBitsMinus1 = 6,
    SeqChooseScreenContentTools = true,
    SeqChooseIntegerMv = true,
    EnableSuperres = false,
    EnableCdef = true,
    EnableRestoration = false,
    ColorDescriptionPresent = true,
    ColorPrimaries = 2,           // AOM_CICP_CP_UNSPECIFIED
    TransferCharacteristics = 2,  // AOM_CICP_TC_UNSPECIFIED
    MatrixCoefficients = 5,       // AOM_CICP_MC_BT_709 (libaom default)
    ChromaSamplePosition = 0,
    SeparateUvDeltas = false,
    FilmGrainParamsPresent = false,
};
var our = Av1SequenceHeaderWriter.EmitPayload(cfg);
Console.WriteLine();
Console.WriteLine($"Our SH ({our.Length} bytes):");
Console.WriteLine($"  {string.Join(" ", our.Select(b => b.ToString("X2")))}");
Console.WriteLine();

if (our.Length != sourceSh.Length)
{
    Console.WriteLine($"  Length differs: source={sourceSh.Length}, ours={our.Length}");
    return;
}
int mismatch = 0, firstMismatch = -1;
for (int i = 0; i < our.Length; i++)
{
    if (our[i] != sourceSh[i])
    {
        if (firstMismatch < 0) firstMismatch = i;
        mismatch++;
    }
}
if (mismatch == 0)
{
    Console.WriteLine("============================================================");
    Console.WriteLine($"  BIT-EXACT: our writer emitted IDENTICAL bytes to libaom-av1.");
    Console.WriteLine($"  {our.Length}/{our.Length} bytes match.");
    Console.WriteLine("============================================================");
}
else
{
    Console.WriteLine($"  Mismatches: {mismatch}/{our.Length} bytes (first at {firstMismatch})");
    Console.WriteLine($"  source bits: {string.Join("", sourceSh.Select(b => Convert.ToString(b, 2).PadLeft(8, '0')))}");
    Console.WriteLine($"  ours bits:   {string.Join("", our.Select(b => Convert.ToString(b, 2).PadLeft(8, '0')))}");
}
