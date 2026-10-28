// Tests for Vp9PartitionProbsParser (slice 216).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod(Category = "CpuOnly")]
    public void Vp9PartitionProbsParser_NoUpdates_LeavesTableUntouched()
    {
        CpuOnlyTest();
        var data = new byte[16];
        var reader = new Vp9BoolDecoder(data, 0, data.Length);

        var probs = (byte[])Vp9PartitionProbs.KfPartitionProbs.Clone();
        var snapshot = (byte[])probs.Clone();

        Vp9PartitionProbsParser.Read(probs, reader);

        for (int i = 0; i < probs.Length; i++) Equal(snapshot[i], probs[i]);
    }

    [TestMethod(Category = "CpuOnly")]
    public void Vp9PartitionProbsParser_RejectsUndersizedBuffer()
    {
        CpuOnlyTest();
        var data = new byte[16];
        var reader = new Vp9BoolDecoder(data, 0, data.Length);
        Throws<ArgumentException>(() =>
            Vp9PartitionProbsParser.Read(new byte[47], reader));
    }

    [TestMethod(Category = "CpuOnly")]
    public void Vp9PartitionProbsParser_RejectsNullArgs()
    {
        CpuOnlyTest();
        var data = new byte[16];
        var reader = new Vp9BoolDecoder(data, 0, data.Length);
        Throws<ArgumentNullException>(() =>
            Vp9PartitionProbsParser.Read(null!, reader));
        Throws<ArgumentNullException>(() =>
            Vp9PartitionProbsParser.Read(new byte[48], null!));
    }
}
