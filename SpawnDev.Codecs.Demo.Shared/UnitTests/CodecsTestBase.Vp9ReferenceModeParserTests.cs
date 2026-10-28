// Tests for Vp9ReferenceModeParser (slice 220).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod(Category = "CpuOnly")]
    public void Vp9ReferenceModeParser_CompoundReferenceAllowed_AllSameBias_ReturnsFalse()
    {
        CpuOnlyTest();
        Equal(false, Vp9ReferenceModeParser.CompoundReferenceAllowed(false, false, false));
        Equal(false, Vp9ReferenceModeParser.CompoundReferenceAllowed(true, true, true));
    }

    [TestMethod(Category = "CpuOnly")]
    public void Vp9ReferenceModeParser_CompoundReferenceAllowed_AnyDifferentBias_ReturnsTrue()
    {
        CpuOnlyTest();
        Equal(true, Vp9ReferenceModeParser.CompoundReferenceAllowed(false, false, true));
        Equal(true, Vp9ReferenceModeParser.CompoundReferenceAllowed(true, false, false));
        Equal(true, Vp9ReferenceModeParser.CompoundReferenceAllowed(false, true, false));
    }

    [TestMethod(Category = "CpuOnly")]
    public void Vp9ReferenceModeParser_CompoundNotAllowed_AlwaysSingleReference()
    {
        CpuOnlyTest();
        // No bits read; arithmetic decoder still has a buffer for init.
        var data = new byte[8];
        var reader = new Vp9BoolDecoder(data, 0, data.Length);

        var mode = Vp9ReferenceModeParser.Read(reader, compoundReferenceAllowed: false);
        Equal(Vp9ReferenceMode.SingleReference, mode);
    }

    [TestMethod(Category = "CpuOnly")]
    public void Vp9ReferenceModeParser_CompoundAllowed_ZeroBuffer_ReadsSingleReference()
    {
        CpuOnlyTest();
        // Buffer of all zeros -> arithmetic ReadBit() returns 0.
        var data = new byte[8];
        var reader = new Vp9BoolDecoder(data, 0, data.Length);

        var mode = Vp9ReferenceModeParser.Read(reader, compoundReferenceAllowed: true);
        Equal(Vp9ReferenceMode.SingleReference, mode);
    }

    [TestMethod(Category = "CpuOnly")]
    public void Vp9ReferenceModeParser_RejectsNullReader()
    {
        CpuOnlyTest();
        Throws<ArgumentNullException>(() =>
            Vp9ReferenceModeParser.Read(null!, true));
    }
}
