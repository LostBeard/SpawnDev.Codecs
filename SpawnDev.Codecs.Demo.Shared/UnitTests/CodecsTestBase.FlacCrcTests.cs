using SpawnDev.Codecs.Audio.Flac;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for <see cref="FlacCrc"/>. FLAC uses:
/// <list type="bullet">
/// <item>CRC-8 (polynomial 0x07, init 0, no reflection, no xor-out) for frame headers.</item>
/// <item>CRC-16 (polynomial 0x8005, init 0, no reflection, no xor-out) for frame footers.</item>
/// </list>
/// Both variants have well-known standard "check" values computed over the
/// ASCII byte sequence "123456789". Those values below are the industry-standard
/// test vectors for CRC-8/SMBUS and CRC-16/UMTS aka CRC-16/BUYPASS, which are
/// exactly the variants FLAC specifies.
/// </summary>
public abstract partial class CodecsTestBase
{
    // -------- CRC-8 --------

    [TestMethod]
    public void FlacCrc_Compute8_EmptyInput_IsZero()
    {
        Equal((byte)0x00, InvokeCompute8(Array.Empty<byte>()));
    }

    [TestMethod]
    public void FlacCrc_Compute8_Standard_123456789_IsF4()
    {
        // "123456789" is the universal CRC check string; CRC-8/SMBUS (= FLAC CRC-8) = 0xF4.
        byte[] input = System.Text.Encoding.ASCII.GetBytes("123456789");
        Equal((byte)0xF4, InvokeCompute8(input));
    }

    [TestMethod]
    public void FlacCrc_Compute8_SingleZero_IsZero()
    {
        // CRC of a single 0-byte under this variant is 0 (init=0, 0 XOR 0 = 0, table[0]=0).
        Equal((byte)0x00, InvokeCompute8(new byte[] { 0x00 }));
    }

    [TestMethod]
    public void FlacCrc_Compute8_Deterministic()
    {
        // Same input twice produces the same result.
        byte[] input = { 0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE };
        byte a = InvokeCompute8(input);
        byte b = InvokeCompute8(input);
        Equal(a, b);
    }

    [TestMethod]
    public void FlacCrc_Compute8_InputSensitive()
    {
        // Changing one bit changes the CRC.
        byte[] a = { 0xDE, 0xAD, 0xBE, 0xEF };
        byte[] b = { 0xDE, 0xAD, 0xBE, 0xEE };
        True(InvokeCompute8(a) != InvokeCompute8(b),
            "CRC-8 should differ when input differs.");
    }

    // -------- CRC-16 --------

    [TestMethod]
    public void FlacCrc_Compute16_EmptyInput_IsZero()
    {
        Equal((ushort)0x0000, InvokeCompute16(Array.Empty<byte>()));
    }

    [TestMethod]
    public void FlacCrc_Compute16_Standard_123456789_IsFEE8()
    {
        // CRC-16/UMTS aka CRC-16/BUYPASS (= FLAC CRC-16) check = 0xFEE8.
        byte[] input = System.Text.Encoding.ASCII.GetBytes("123456789");
        Equal((ushort)0xFEE8, InvokeCompute16(input));
    }

    [TestMethod]
    public void FlacCrc_Compute16_SingleZero_IsZero()
    {
        Equal((ushort)0x0000, InvokeCompute16(new byte[] { 0x00 }));
    }

    [TestMethod]
    public void FlacCrc_Compute16_Deterministic()
    {
        byte[] input = { 0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE };
        ushort a = InvokeCompute16(input);
        ushort b = InvokeCompute16(input);
        Equal(a, b);
    }

    [TestMethod]
    public void FlacCrc_Compute16_InputSensitive()
    {
        byte[] a = { 0xDE, 0xAD, 0xBE, 0xEF };
        byte[] b = { 0xDE, 0xAD, 0xBE, 0xEE };
        True(InvokeCompute16(a) != InvokeCompute16(b),
            "CRC-16 should differ when input differs.");
    }

    // FlacCrc is internal, so access helper lives here in the shared-test assembly
    // (which has InternalsVisibleTo permission) rather than in TestHelpers
    // (kept codec-agnostic).
    private static byte InvokeCompute8(ReadOnlySpan<byte> data) => FlacCrc.Compute8(data);
    private static ushort InvokeCompute16(ReadOnlySpan<byte> data) => FlacCrc.Compute16(data);
}
