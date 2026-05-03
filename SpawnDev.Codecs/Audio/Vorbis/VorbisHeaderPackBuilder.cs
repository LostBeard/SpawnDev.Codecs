// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VorbisHeaderPackBuilder - public static helpers for constructing the
// 3 Vorbis Ogg header packets (identification, comment, setup) from
// records, plus the resolved (Identification + Setup) record pair from
// VorbisAudioEncoderOptions.
//
// Lives in main library so GPU integration classes (VorbisAudioEncoderGpu)
// can produce the per-stream header bytes without depending on a CPU
// VorbisAudioEncoder instance. The CPU encoder (in
// SpawnDev.Codecs.References) delegates its header construction to these
// statics so behavior stays bit-exact.
//
// Built per Captain's 2026-05-03 architectural directive: ZERO external
// dependencies in main library + CPU encoders/decoders move to References.

namespace SpawnDev.Codecs.Audio.Vorbis;

/// <summary>
/// Static helpers for constructing Vorbis identification + comment +
/// setup headers from encoder options or pre-built header records.
/// All methods are pure - no instance state.
/// </summary>
public static class VorbisHeaderPackBuilder
{
    /// <summary>
    /// Codebook index for the residue classbook (1 used entry, dim 1, length 1).
    /// Built first in the codebook list.
    /// </summary>
    public const int ClassBookIndex = 0;

    /// <summary>
    /// Codebook index for the residue VQ codebook
    /// (<see cref="ResidueBookEntries"/> entries, dim 1, lookup type 1).
    /// </summary>
    public const int ResidueBookIndex = 1;

    /// <summary>
    /// Number of residue VQ codebook entries. 1024 entries spread across
    /// [-2, +2] gives step = 4/1024 = 0.0039, ~0.4% relative quantisation
    /// per residue value (matches libvorbis SNR at default quality).
    /// </summary>
    public const int ResidueBookEntries = 1024;

    /// <summary>
    /// Residue VQ codebook covers a normalised range. Set to 2.0f so the
    /// codebook absorbs under-fits where the local spectrum magnitude
    /// exceeds the chosen floor endpoint.
    /// </summary>
    public const float ResidueRange = 2.0f;

    /// <summary>
    /// Build the identification header AND resolved setup header from
    /// encoder options. Resolves residue End/PartitionSize so the result
    /// is the EXACT pair the decoder will see.
    /// </summary>
    public static (VorbisIdentificationHeader Identification, VorbisSetupHeader Setup)
        BuildResolvedHeaders(VorbisAudioEncoderOptions opts)
    {
        ArgumentNullException.ThrowIfNull(opts);
        var ident = BuildIdentificationHeader(opts);
        var rawSetup = BuildSetupHeader(opts.BlockSize);
        int half = opts.BlockSize / 2;
        var resolvedResidues = new VorbisResidueConfig[rawSetup.Residues.Length];
        for (int i = 0; i < rawSetup.Residues.Length; i++)
            resolvedResidues[i] = rawSetup.Residues[i] with { End = half, PartitionSize = half };
        var setup = rawSetup with { Residues = resolvedResidues };
        return (ident, setup);
    }

    /// <summary>
    /// Build the identification header from encoder options.
    /// </summary>
    public static VorbisIdentificationHeader BuildIdentificationHeader(VorbisAudioEncoderOptions opts)
    {
        ArgumentNullException.ThrowIfNull(opts);
        return new VorbisIdentificationHeader
        {
            VorbisVersion = 0,
            AudioChannels = opts.Channels,
            SampleRateHz = opts.SampleRateHz,
            BitrateMaximum = 0,
            BitrateNominal = 0,
            BitrateMinimum = 0,
            BlockSize0 = opts.BlockSize,
            BlockSize1 = opts.BlockSize,
        };
    }

    /// <summary>
    /// Build the (unresolved) setup header for the given block size.
    /// Caller resolves residue End/PartitionSize per stream.
    /// </summary>
    public static VorbisSetupHeader BuildSetupHeader(int blockSize)
    {
        int halfBlock = blockSize / 2;
        // Floor X[1] must reach the end of the spectrum so the rendered
        // floor curve interpolates across the entire band. Pick the smallest
        // RangeBits such that 1<<RangeBits >= halfBlock; clamp to spec max
        // (4-bit field, so RangeBits<=15).
        int floorRangeBits = 1;
        while ((1 << floorRangeBits) < halfBlock) floorRangeBits++;
        if (floorRangeBits > 15) floorRangeBits = 15;
        int floorX1 = Math.Min(1 << floorRangeBits, halfBlock);

        // Codebook 0: classbook with 1 used entry (length 1, code 0).
        // Decoder requires entries >= 1 and dimensions >= 1. With 1 entry, the
        // only valid codeword is 0 bits long; but VorbisHuffman special-cases
        // single-entry as code=0 with the entry's actual length. We use length 1
        // so writer + reader agree.
        var classBook = new VorbisCodebook
        {
            Dimensions = 1,
            Entries = 2, // 2 entries so the marker tree has both children populated
            Ordered = false,
            Sparse = false,
            Lengths = new[] { 1, 1 },
            LookupType = 0,
            MinValue = 0,
            DeltaValue = 0,
            ValueBits = 0,
            SequenceP = false,
            Multiplicands = Array.Empty<int>(),
        };

        // Codebook 1: residue VQ with ResidueBookEntries entries, dim 1, lookup type 1.
        // We anchor entry N/2 at value 0 exactly so noise-gated bins decode to
        // silence rather than to half-step noise:
        //   mindel = -(N/2) * delta = -ResidueRange
        // Entries 0..N-1 then cover [-ResidueRange, +ResidueRange - step].
        var residueMultiplicands = new int[ResidueBookEntries];
        for (int i = 0; i < ResidueBookEntries; i++) residueMultiplicands[i] = i;
        var residueLengths = new int[ResidueBookEntries];
        // Uniform fixed-length code per entry: log2(entries) bits each.
        int residueCodeLen = (int)Math.Round(Math.Log2(ResidueBookEntries));
        for (int i = 0; i < ResidueBookEntries; i++) residueLengths[i] = residueCodeLen;
        int residueValueBits = 0;
        while ((1 << residueValueBits) < ResidueBookEntries) residueValueBits++;
        double residueDelta = 2.0 * ResidueRange / ResidueBookEntries;
        double residueMin = -ResidueRange;
        var residueBook = new VorbisCodebook
        {
            Dimensions = 1,
            Entries = ResidueBookEntries,
            Ordered = false,
            Sparse = false,
            Lengths = residueLengths,
            LookupType = 1,
            MinValue = residueMin,
            DeltaValue = residueDelta,
            ValueBits = residueValueBits,
            SequenceP = false,
            Multiplicands = residueMultiplicands,
        };

        var floor1 = new VorbisFloor1Config
        {
            Partitions = 0,
            PartitionClassList = Array.Empty<int>(),
            ClassDimensions = Array.Empty<int>(),
            ClassSubclasses = Array.Empty<int>(),
            ClassMasterbooks = Array.Empty<int>(),
            ClassSubclassBooks = Array.Empty<int[]>(),
            Multiplier = 1,
            RangeBits = floorRangeBits,
            XList = new int[] { 0, floorX1 },
        };

        var residue = new VorbisResidueConfig
        {
            Type = VorbisResidueType.Type1,
            Begin = 0,
            // End/PartitionSize are filled in per-stream by BuildResolvedHeaders.
            End = 0,
            PartitionSize = 0,
            Classifications = 1,
            Classbook = ClassBookIndex,
            Cascade = new int[] { 1 },
            Books = new int[][] { new int[] { ResidueBookIndex, -1, -1, -1, -1, -1, -1, -1 } },
        };

        var mapping = new VorbisMappingConfig
        {
            Submaps = 1,
            CouplingMagnitudeChannels = Array.Empty<int>(),
            CouplingAngleChannels = Array.Empty<int>(),
            Mux = new int[] { 0 },
            SubmapFloor = new int[] { 0 },
            SubmapResidue = new int[] { 0 },
        };

        var mode = new VorbisModeConfig
        {
            BlockFlag = false,
            WindowType = 0,
            TransformType = 0,
            Mapping = 0,
        };

        return new VorbisSetupHeader
        {
            Codebooks = new[] { classBook, residueBook },
            Floors = new[] { floor1 },
            Residues = new[] { residue },
            Mappings = new[] { mapping },
            Modes = new[] { mode },
        };
    }

    /// <summary>
    /// Build the Vorbis identification (header packet 1) bytes from the
    /// supplied header.
    /// </summary>
    public static byte[] BuildIdentPacketBytes(VorbisIdentificationHeader ident)
    {
        if (ident is null) throw new ArgumentNullException(nameof(ident));
        var bytes = new byte[30];
        bytes[0] = 0x01;
        var magic = new byte[] { (byte)'v', (byte)'o', (byte)'r', (byte)'b', (byte)'i', (byte)'s' };
        for (int i = 0; i < 6; i++) bytes[1 + i] = magic[i];
        // version = 0 (already)
        bytes[11] = (byte)ident.AudioChannels;
        WriteInt32Le(bytes, 12, ident.SampleRateHz);
        WriteInt32Le(bytes, 16, ident.BitrateMaximum);
        WriteInt32Le(bytes, 20, ident.BitrateNominal);
        WriteInt32Le(bytes, 24, ident.BitrateMinimum);
        int log0 = Log2(ident.BlockSize0);
        int log1 = Log2(ident.BlockSize1);
        bytes[28] = (byte)((log1 << 4) | log0);
        bytes[29] = 0x01; // framing flag
        return bytes;
    }

    /// <summary>
    /// Build the Vorbis comment (header packet 2) bytes for the supplied
    /// vendor string with no user comments.
    /// </summary>
    public static byte[] BuildCommentPacketBytes(string vendor)
    {
        if (vendor is null) throw new ArgumentNullException(nameof(vendor));
        var vendorBytes = System.Text.Encoding.UTF8.GetBytes(vendor);
        var bytes = new byte[7 + 4 + vendorBytes.Length + 4 + 1];
        bytes[0] = 0x03;
        var magic = new byte[] { (byte)'v', (byte)'o', (byte)'r', (byte)'b', (byte)'i', (byte)'s' };
        for (int i = 0; i < 6; i++) bytes[1 + i] = magic[i];
        WriteUInt32Le(bytes, 7, (uint)vendorBytes.Length);
        Array.Copy(vendorBytes, 0, bytes, 11, vendorBytes.Length);
        WriteUInt32Le(bytes, 11 + vendorBytes.Length, 0u); // 0 user comments
        bytes[15 + vendorBytes.Length] = 0x01; // framing flag
        return bytes;
    }

    /// <summary>
    /// Build the Vorbis setup (header packet 3) bytes from the supplied
    /// resolved setup header.
    /// </summary>
    public static byte[] BuildSetupPacketBytes(VorbisSetupHeader setup)
    {
        if (setup is null) throw new ArgumentNullException(nameof(setup));
        var writer = new VorbisBitWriter();
        writer.WriteBits((uint)setup.Codebooks.Length - 1u, 8);
        for (int i = 0; i < setup.Codebooks.Length; i++)
            VorbisCodebookEncoder.Pack(writer, setup.Codebooks[i]);

        // Time count - 1 = 0 (one entry, value 0).
        writer.WriteBits(0u, 6);
        writer.WriteBits(0u, 16);

        // Floor count - 1.
        writer.WriteBits((uint)setup.Floors.Length - 1u, 6);
        for (int i = 0; i < setup.Floors.Length; i++)
        {
            writer.WriteBits(1u, 16); // floor type 1
            PackFloor1(writer, setup.Floors[i]);
        }

        // Residue count - 1.
        writer.WriteBits((uint)setup.Residues.Length - 1u, 6);
        for (int i = 0; i < setup.Residues.Length; i++)
        {
            writer.WriteBits((uint)setup.Residues[i].Type, 16);
            PackResidue(writer, setup.Residues[i]);
        }

        // Mapping count - 1.
        writer.WriteBits((uint)setup.Mappings.Length - 1u, 6);
        for (int i = 0; i < setup.Mappings.Length; i++)
        {
            writer.WriteBits(0u, 16); // mapping type 0
            PackMapping(writer, setup.Mappings[i]);
        }

        // Mode count - 1.
        writer.WriteBits((uint)setup.Modes.Length - 1u, 6);
        for (int i = 0; i < setup.Modes.Length; i++)
        {
            writer.WriteBit(setup.Modes[i].BlockFlag ? 1u : 0u);
            writer.WriteBits(0u, 16); // window type
            writer.WriteBits(0u, 16); // transform type
            writer.WriteBits((uint)setup.Modes[i].Mapping, 8);
        }

        // Framing flag.
        writer.WriteBit(1u);

        var bitBytes = writer.ToArray();
        var output = new byte[7 + bitBytes.Length];
        output[0] = 0x05;
        var magic = new byte[] { (byte)'v', (byte)'o', (byte)'r', (byte)'b', (byte)'i', (byte)'s' };
        for (int i = 0; i < 6; i++) output[1 + i] = magic[i];
        Array.Copy(bitBytes, 0, output, 7, bitBytes.Length);
        return output;
    }

    private static void PackFloor1(VorbisBitWriter writer, VorbisFloor1Config cfg)
    {
        writer.WriteBits((uint)cfg.Partitions, 5);
        for (int i = 0; i < cfg.Partitions; i++)
            writer.WriteBits((uint)cfg.PartitionClassList[i], 4);
        for (int c = 0; c < cfg.ClassDimensions.Length; c++)
        {
            writer.WriteBits((uint)(cfg.ClassDimensions[c] - 1), 3);
            writer.WriteBits((uint)cfg.ClassSubclasses[c], 2);
            if (cfg.ClassSubclasses[c] != 0)
                writer.WriteBits((uint)cfg.ClassMasterbooks[c], 8);
            int subCount = 1 << cfg.ClassSubclasses[c];
            for (int j = 0; j < subCount; j++)
                writer.WriteBits((uint)(cfg.ClassSubclassBooks[c][j] + 1), 8);
        }
        writer.WriteBits((uint)(cfg.Multiplier - 1), 2);
        writer.WriteBits((uint)cfg.RangeBits, 4);
        for (int i = 2; i < cfg.XList.Length; i++)
            writer.WriteBits((uint)cfg.XList[i], cfg.RangeBits);
    }

    private static void PackResidue(VorbisBitWriter writer, VorbisResidueConfig cfg)
    {
        writer.WriteBits((uint)cfg.Begin, 24);
        writer.WriteBits((uint)cfg.End, 24);
        writer.WriteBits((uint)(cfg.PartitionSize - 1), 24);
        writer.WriteBits((uint)(cfg.Classifications - 1), 6);
        writer.WriteBits((uint)cfg.Classbook, 8);
        for (int i = 0; i < cfg.Classifications; i++)
        {
            int low = cfg.Cascade[i] & 0x07;
            int high = (cfg.Cascade[i] >> 3) & 0x1F;
            writer.WriteBits((uint)low, 3);
            writer.WriteBit(high != 0 ? 1u : 0u);
            if (high != 0) writer.WriteBits((uint)high, 5);
        }
        for (int i = 0; i < cfg.Classifications; i++)
        {
            for (int j = 0; j < 8; j++)
            {
                if (((cfg.Cascade[i] >> j) & 1) != 0)
                    writer.WriteBits((uint)cfg.Books[i][j], 8);
            }
        }
    }

    private static void PackMapping(VorbisBitWriter writer, VorbisMappingConfig cfg)
    {
        bool submapsFlag = cfg.Submaps > 1;
        writer.WriteBit(submapsFlag ? 1u : 0u);
        if (submapsFlag) writer.WriteBits((uint)(cfg.Submaps - 1), 4);
        bool couplingFlag = cfg.CouplingMagnitudeChannels.Length > 0;
        writer.WriteBit(couplingFlag ? 1u : 0u);
        // (No coupling in our minimal mapping.)
        writer.WriteBits(0u, 2); // reserved
        if (cfg.Submaps > 1)
            for (int i = 0; i < cfg.Mux.Length; i++)
                writer.WriteBits((uint)cfg.Mux[i], 4);
        for (int j = 0; j < cfg.Submaps; j++)
        {
            writer.WriteBits(0u, 8); // submap reserved/time placeholder
            writer.WriteBits((uint)cfg.SubmapFloor[j], 8);
            writer.WriteBits((uint)cfg.SubmapResidue[j], 8);
        }
    }

    private static void WriteInt32Le(byte[] dest, int offset, int value)
    {
        for (int i = 0; i < 4; i++) dest[offset + i] = (byte)((uint)value >> (8 * i));
    }

    private static void WriteUInt32Le(byte[] dest, int offset, uint value)
    {
        for (int i = 0; i < 4; i++) dest[offset + i] = (byte)(value >> (8 * i));
    }

    private static int Log2(int v)
    {
        int r = 0;
        while ((1 << r) < v) r++;
        return r;
    }
}
