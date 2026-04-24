using SpawnDev.Codecs.Audio.Vorbis;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// End-to-end test of <see cref="VorbisSetupHeaderParser"/>. Builds a full
/// minimal setup packet byte stream from all the sub-component writers used
/// in the prior slices and verifies every section parses correctly.
/// </summary>
public abstract partial class CodecsTestBase
{
    private static byte[] BuildMinimalSetupPacket(int audioChannels)
    {
        // Header: 1 byte packet-type 0x05 + 6 bytes "vorbis"
        var prefix = new byte[] { 0x05, (byte)'v', (byte)'o', (byte)'r', (byte)'b', (byte)'i', (byte)'s' };
        var w = new VorbisTestWriter();

        // ---- 1 codebook: minimal 2-entry all-length-1, no lookup ----
        w.Write(0, 8);                 // codebook_count - 1 = 0 -> 1
        // codebook #0
        w.Write(0x564342, 24);         // sync
        w.Write(1, 16);                // dimensions
        w.Write(2, 24);                // entries
        w.Write(0, 1);                 // ordered = no
        w.Write(0, 1);                 // sparse = no
        w.Write(0, 5); w.Write(0, 5);  // lengths 1 and 1 (writing 0 means length-1=0)
        w.Write(0, 4);                 // lookup_type = 0

        // ---- 1 time entry, value 0 ----
        w.Write(0, 6);                 // time_count - 1 = 0 -> 1
        w.Write(0, 16);                // placeholder

        // ---- 1 floor 1 config, minimal ----
        w.Write(0, 6);                 // floor_count - 1 = 0 -> 1
        w.Write(1, 16);                // floor_type = 1
        w.Write(1, 5);                 // partitions
        w.Write(0, 4);                 // partition_class_list[0]
        w.Write(0, 3);                 // class 0 dimensions - 1 = 0 -> 1
        w.Write(0, 2);                 // class 0 subclasses = 0
        w.Write(1, 8);                 // subclass book 0 = 0 (written as 0+1)
        w.Write(0, 2);                 // multiplier - 1 = 0 -> 1
        w.Write(4, 4);                 // rangebits = 4
        w.Write(7, 4);                 // xlist[2] = 7

        // ---- 1 residue, type 0, minimal ----
        w.Write(0, 6);                 // residue_count - 1 = 0 -> 1
        w.Write(0, 16);                // residue_type = 0
        w.Write(0, 24); w.Write(128, 24); w.Write(31, 24); // begin/end/partition_size-1
        w.Write(0, 6);                 // 1 classification
        w.Write(0, 8);                 // classbook
        w.Write(1, 3); w.Write(0, 1);  // cascade = 0b001
        w.Write(0, 8);                 // books[0][0]

        // ---- 1 mapping type 0, mono, single submap ----
        w.Write(0, 6);                 // mapping_count - 1 = 0 -> 1
        w.Write(0, 16);                // mapping_type = 0
        w.Write(0, 1);                 // submaps flag = no
        w.Write(0, 1);                 // coupling flag = no
        w.Write(0, 2);                 // reserved
        w.Write(0, 8); w.Write(0, 8); w.Write(0, 8); // submap 0: reserved/floor/residue

        // ---- 1 mode ----
        w.Write(0, 6);                 // mode_count - 1 = 0 -> 1
        w.Write(0, 1);                 // blockflag = short
        w.Write(0, 16); w.Write(0, 16); // window + transform
        w.Write(0, 8);                 // mapping index
        w.Write(1, 1);                 // framing flag

        byte[] body = w.ToArray();
        var full = new byte[prefix.Length + body.Length];
        Array.Copy(prefix, full, prefix.Length);
        Array.Copy(body, 0, full, prefix.Length, body.Length);
        return full;
    }

    [TestMethod]
    public void VorbisSetupHeader_Minimal_ParsesEverySection()
    {
        byte[] data = BuildMinimalSetupPacket(audioChannels: 1);
        var setup = VorbisSetupHeaderParser.Parse(data, audioChannels: 1);
        Equal(1, setup.Codebooks.Length);
        Equal(2, setup.Codebooks[0].Entries);
        Equal(1, setup.Floors.Length);
        Equal(1, setup.Floors[0].Partitions);
        Equal(1, setup.Residues.Length);
        Equal(VorbisResidueType.Type0, setup.Residues[0].Type);
        Equal(1, setup.Mappings.Length);
        Equal(1, setup.Mappings[0].Submaps);
        Equal(1, setup.Modes.Length);
        False(setup.Modes[0].BlockFlag);
    }

    [TestMethod]
    public void VorbisSetupHeader_BadPacketType_Throws()
    {
        byte[] data = BuildMinimalSetupPacket(1);
        data[0] = 0x04;
        bool threw = false;
        try { VorbisSetupHeaderParser.Parse(data, 1); } catch (InvalidDataException) { threw = true; }
        True(threw);
    }

    [TestMethod]
    public void VorbisSetupHeader_BadMagic_Throws()
    {
        byte[] data = BuildMinimalSetupPacket(1);
        data[2] = (byte)'X';
        bool threw = false;
        try { VorbisSetupHeaderParser.Parse(data, 1); } catch (InvalidDataException) { threw = true; }
        True(threw);
    }

    [TestMethod]
    public void VorbisSetupHeader_FloorType0_ThrowsNotSupported()
    {
        // Swap the floor_type 16-bit field value from 1 to 0. We don't rebuild the
        // rest of the packet because the parser bails right after reading type.
        // Easier: build a custom packet up to the floor_type field.
        var prefix = new byte[] { 0x05, (byte)'v', (byte)'o', (byte)'r', (byte)'b', (byte)'i', (byte)'s' };
        var w = new VorbisTestWriter();
        // Skip through codebook + time like the minimal parser.
        w.Write(0, 8);
        w.Write(0x564342, 24); w.Write(1, 16); w.Write(2, 24);
        w.Write(0, 1); w.Write(0, 1); w.Write(0, 5); w.Write(0, 5); w.Write(0, 4);
        w.Write(0, 6); w.Write(0, 16);
        w.Write(0, 6);          // 1 floor
        w.Write(0, 16);         // floor_type = 0 -> NotSupported
        byte[] body = w.ToArray();
        var full = new byte[prefix.Length + body.Length];
        Array.Copy(prefix, full, prefix.Length);
        Array.Copy(body, 0, full, prefix.Length, body.Length);
        bool threw = false;
        try { VorbisSetupHeaderParser.Parse(full, 1); } catch (NotSupportedException) { threw = true; }
        True(threw);
    }
}
