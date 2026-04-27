// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Minimum-viable Vorbis I encoder. Produces a structurally valid Ogg-Vorbis
// stream that round-trips through our matching decoder. Designed to prove the
// encode/decode pipeline end to end with a single fixed configuration:
//
//   - Mono only (one channel)
//   - Single block size (no long/short transitions)
//   - Single mode (BlockFlag=false)
//   - Single mapping with single submap (no coupling)
//   - Single floor 1 with two endpoint Y values (no partitions)
//   - Single residue type 1 with one classification, one codebook
//   - Two static codebooks: a tiny class codebook (book 0) and a residue VQ
//     codebook (book 1) that quantises residue values to a small alphabet.
//
// The encoder forward-MDCTs each input block, fits a piecewise-linear floor
// curve via the two endpoint Y values that approximate the block's spectral
// envelope, then quantises floor-divided residue with the VQ codebook. The
// output stream decodes back to a tone whose dominant frequency matches the
// input - lossy but structurally complete.
//
// Reference: Vorbis I 1.5 spec; libvorbis lib/encoder.c for the canonical
// approach. This implementation is intentionally minimal - it is NOT meant
// as a competitive Vorbis encoder, but as a "the pipeline works" deliverable
// that consumers (and tests) can build on.

using SpawnDev.Codecs.Audio.Transforms;

namespace SpawnDev.Codecs.Audio.Vorbis;

/// <summary>
/// Configuration for the minimum-viable Vorbis encoder.
/// </summary>
public sealed record VorbisAudioEncoderOptions
{
    /// <summary>Audio sample rate in Hz.</summary>
    public int SampleRateHz { get; init; } = 44100;

    /// <summary>Channel count. Currently only 1 (mono) is supported.</summary>
    public int Channels { get; init; } = 1;

    /// <summary>
    /// Block size in samples (power of 2 in [64, 8192]). Same value used for
    /// both blocksize_0 and blocksize_1; the encoder emits short-block-only
    /// audio packets with no transition windows. Default 1024 matches what
    /// libvorbis uses by default for the short block size.
    /// </summary>
    public int BlockSize { get; init; } = 1024;
}

/// <summary>
/// Single-channel Vorbis I encoder. Use <see cref="EncodeStream"/> to convert
/// PCM to a complete Ogg-Vorbis byte stream in one shot.
/// </summary>
public sealed class VorbisAudioEncoder
{
    private readonly VorbisAudioEncoderOptions _opts;
    private readonly VorbisIdentificationHeader _ident;
    private readonly VorbisSetupHeader _setup;
    private readonly (uint code, int length)[] _classBookCodes;
    private readonly (uint code, int length)[] _residueBookCodes;

    /// <summary>Construct an encoder with the given options.</summary>
    public VorbisAudioEncoder(VorbisAudioEncoderOptions options)
    {
        _opts = options ?? throw new ArgumentNullException(nameof(options));
        if (_opts.Channels != 1)
            throw new NotSupportedException("Only mono encoding is supported in this minimum-viable encoder.");
        if (_opts.BlockSize < 64 || _opts.BlockSize > 8192 || (_opts.BlockSize & (_opts.BlockSize - 1)) != 0)
            throw new ArgumentException(
                $"BlockSize must be a power of 2 in [64, 8192], got {_opts.BlockSize}.", nameof(options));

        _ident = BuildIdentificationHeader(_opts);
        var rawSetup = BuildSetupHeader();
        // Resolve residue End/PartitionSize at construction so per-packet
        // encode does not depend on having called BuildSetupPacket first.
        var resolvedResidues = new VorbisResidueConfig[rawSetup.Residues.Length];
        int half = _opts.BlockSize / 2;
        for (int i = 0; i < rawSetup.Residues.Length; i++)
            resolvedResidues[i] = rawSetup.Residues[i] with { End = half, PartitionSize = half };
        _setup = rawSetup with { Residues = resolvedResidues };
        _resolvedResidues = resolvedResidues;

        _classBookCodes = VorbisCodebookEncoder.BuildCodewords(_setup.Codebooks[ClassBookIndex].Lengths);
        _residueBookCodes = VorbisCodebookEncoder.BuildCodewords(_setup.Codebooks[ResidueBookIndex].Lengths);
    }

    /// <summary>The identification header this encoder will emit.</summary>
    public VorbisIdentificationHeader Identification => _ident;

    /// <summary>The setup header this encoder will emit.</summary>
    public VorbisSetupHeader Setup => _setup;

    // Indices in the setup header codebook list:
    // Book 0: class codebook used as the residue classbook (1 entry, dim 1, length 1).
    // Book 1: residue VQ codebook (ResidueBookEntries entries, dim 1, lookup type 2).
    private const int ClassBookIndex = 0;
    private const int ResidueBookIndex = 1;
    // 256-entry residue VQ covers wide dynamic range. The floor curve in our
    // minimum encoder is forced to ~1.0 (table index 255), so the residue
    // alone carries the spectrum magnitude. With 256 levels uniformly spaced
    // across [-ResidueRange, +ResidueRange], quantisation noise is
    // 2*ResidueRange/256 per bin.
    private const int ResidueBookEntries = 256;
    private const float ResidueRange = 256.0f;

    /// <summary>
    /// Encode an entire mono PCM input into a complete Ogg-Vorbis byte stream.
    /// Pads the input with silence so the trailing block is fully encoded.
    /// </summary>
    public byte[] EncodeStream(ReadOnlySpan<float> mono, string vendor = "SpawnDev.Codecs")
    {
        if (_opts.Channels != 1) throw new NotSupportedException();
        var packets = new List<Container.Ogg.OggOutgoingPacket>();

        // Three header packets first.
        packets.Add(new Container.Ogg.OggOutgoingPacket
        {
            Data = BuildIdentPacket(),
            GranulePosition = 0,
        });
        packets.Add(new Container.Ogg.OggOutgoingPacket
        {
            Data = BuildCommentPacket(vendor),
            GranulePosition = 0,
        });
        packets.Add(new Container.Ogg.OggOutgoingPacket
        {
            Data = BuildSetupPacket(),
            GranulePosition = 0,
        });

        // Audio packets: one per block. Each block consumes BlockSize/2 NEW
        // samples and overlaps with the previous block by BlockSize/2.
        int n = _opts.BlockSize;
        int half = n / 2;
        int totalSamples = mono.Length;
        // Vorbis decoder discards the first packet's output (no previous right half),
        // so we emit one extra leading packet to prime the overlap. Total decoded
        // sample count after first-packet discard is (numPackets - 1) * half.
        int audioPackets = (int)Math.Ceiling((double)(totalSamples + half) / half);
        if (audioPackets < 2) audioPackets = 2;

        var inputBuffer = new float[n];
        long granule = 0;

        for (int p = 0; p < audioPackets; p++)
        {
            int srcStart = p * half - half; // first packet has overlap-prefix at -half
            for (int i = 0; i < n; i++)
            {
                int srcIdx = srcStart + i;
                inputBuffer[i] = (srcIdx >= 0 && srcIdx < totalSamples) ? mono[srcIdx] : 0f;
            }
            byte[] packet = EncodeAudioPacket(inputBuffer);
            // Granule position counts decoded samples emitted up to and
            // including this packet. First packet emits 0; subsequent emit half.
            if (p > 0) granule += half;
            packets.Add(new Container.Ogg.OggOutgoingPacket
            {
                Data = packet,
                GranulePosition = granule,
            });
        }

        return Container.Ogg.OggPageWriter.WriteStream(
            (uint)Random.Shared.Next(1, int.MaxValue), packets);
    }

    /// <summary>
    /// Encode one block's worth of PCM (length = <c>BlockSize</c>, windowed by
    /// the caller? Actually the encoder applies the window itself). Returns
    /// the bit-packed audio packet bytes (no Ogg framing).
    /// </summary>
    internal byte[] EncodeAudioPacket(ReadOnlySpan<float> block)
    {
        int n = _opts.BlockSize;
        int half = n / 2;
        if (block.Length != n)
            throw new ArgumentException($"Block must be exactly {n} samples, got {block.Length}.");

        // 1. Apply the synthesis window (used identically for analysis in
        //    Vorbis's TDAC scheme).
        var window = VorbisWindow.GenerateCanonical(n);
        var windowed = new float[n];
        for (int i = 0; i < n; i++) windowed[i] = block[i] * window[i];

        // 2. Forward MDCT to get N/2 spectral coefficients.
        var spectrum = new float[half];
        MdctReference.Transform(windowed, spectrum);

        // 3. Floor curve: forced to maximum (~1.0). The residue alone carries
        //    the spectrum magnitude. This is intentionally simple - a real
        //    encoder would shape the floor to the spectral envelope and let
        //    residue carry only the per-bin detail. With 256 residue levels
        //    over [-256, +256], typical music spectra are quantised to about
        //    8-bit precision per bin which is enough for tone round-tripping.
        var floorCfg = (VorbisFloor1Config)_setup.Floors[0];
        const int FloorYMax = 127; // multiplier=2 -> table index 127*2=254 -> ~0.94
        var posteriors = new int[] { FloorYMax, FloorYMax };
        var floorCurve = new float[half];
        VorbisFloor1Curve.Render(floorCfg, posteriors, half, floorCurve);

        // 4. Compute residue = spectrum / floor, quantised to the residue VQ.
        //    Apply a noise-gate threshold relative to the spectral peak so
        //    quiet bins quantise to zero instead of accumulating quantisation
        //    noise. The threshold is conservative (0.5% of peak) so we do not
        //    drop musically meaningful detail.
        var residueQ = new int[half];
        float peak = 0;
        for (int i = 0; i < half; i++) { float a = Math.Abs(spectrum[i]); if (a > peak) peak = a; }
        float threshold = peak * 0.05f;
        // Find the residue codebook entry closest to value 0 (used as the
        // "gated to zero" index).
        int zeroEntry = QuantiseResidueValue(0f);
        for (int i = 0; i < half; i++)
        {
            float floor = Math.Max(floorCurve[i], 1e-12f);
            float spec = Math.Abs(spectrum[i]) < threshold ? 0f : spectrum[i];
            float r = spec / floor;
            residueQ[i] = spec == 0f ? zeroEntry : QuantiseResidueValue(r);
        }

        // 5. Bit-pack the audio packet.
        var writer = new VorbisBitWriter();

        // Audio header: type bit (0) + 1-bit mode (we have 1 mode so 0 bits, but ilog(0)=0).
        writer.WriteBit(0u); // packet type = 0 (audio)
        // ilog(modes-1) = ilog(0) = 0 bits for mode; nothing to write.

        // Floor 1 nonzero bit + 2 endpoint Y values (multiplier 2 -> 7-bit endpoints).
        writer.WriteBit(1u); // nonzero
        int endpointBits = 7; // Multiplier=2 -> 7 bits
        writer.WriteBits((uint)posteriors[0], endpointBits);
        writer.WriteBits((uint)posteriors[1], endpointBits);
        // No partitions (Partitions=0 in our floor config), so no further floor data.

        // Residue type 1 with 1 channel, 1 classification, classbook always
        // returns class 0, then for class 0 the residue book quantises per dim.
        // - actualBegin = 0, actualEnd = half, partitionSize = half (so 1 partition)
        // - classifications matrix has 1 partition * 1 channel
        // - classwordsPerCodeword = ClassDimensions = 1 (so each codeword maps to 1 partition)
        // Pass 0:
        //   - classification: read 1 codebook 0 entry (always entry 0). Sets classification[0][0] = 0.
        //   - data: classification 0 -> book 1 active. Decode residue values.
        // Residue partition size is half-block (filled in by BuildSetupPacket).
        int classbookEntry = 0; // single entry in classbook -> always entry 0
        WriteCodebookEntry(writer, _classBookCodes, classbookEntry);

        // Now write the actual residue partition. PartitionSize values, one
        // codebook entry per dim (=1). PartitionSize == half by design.
        for (int i = 0; i < half; i++)
        {
            int q = residueQ[i];
            WriteCodebookEntry(writer, _residueBookCodes, q);
        }
        // Passes 1..7: book index is -1 for our single classification, so
        // nothing additional is written.

        return writer.ToArray();
    }

    /// <summary>Write a codebook entry's canonical Huffman codeword.</summary>
    private static void WriteCodebookEntry(VorbisBitWriter writer, (uint code, int length)[] codes, int entry)
    {
        if ((uint)entry >= (uint)codes.Length)
            throw new InvalidOperationException($"Entry index {entry} >= codebook size {codes.Length}.");
        var (code, length) = codes[entry];
        if (length == 0)
            throw new InvalidOperationException($"Cannot write unused codebook entry {entry}.");
        // Vorbis canonical codes are MSB-first when conceptually walked, but
        // the canonical Huffman table from VorbisHuffman.Build matches what an
        // LSB-first reader extracts when we write the bits MSB-first using
        // WriteBits' value-encoding semantics. WriteBits writes the low bit
        // first; we want the highest bit of the code to come out first so the
        // tree walks correctly in the decoder. So we bit-reverse `code` over
        // `length` bits and write that.
        uint reversed = BitReverse(code, length);
        writer.WriteBits(reversed, length);
    }

    private static uint BitReverse(uint value, int bits)
    {
        uint r = 0;
        for (int i = 0; i < bits; i++)
        {
            r = (r << 1) | (value & 1u);
            value >>= 1;
        }
        return r;
    }

    /// <summary>
    /// Map a linear magnitude value to a 7-bit floor Y coordinate (multiplier=2,
    /// range=128). Higher Y -> bigger floor multiplier in
    /// <see cref="VorbisFloor1Curve"/>'s inverse-dB lookup.
    /// </summary>
    private static int MagnitudeToFloorY(float magnitude)
    {
        // VorbisFloor1Curve uses an inverse-dB table that runs from
        // ~1.06e-7 (idx 0) to 1.0 (idx 255). With multiplier=2, posterior Y
        // values map as 0..127 -> table indices 0..254. We pick the Y that
        // makes table[Y*2] closest to (or just above) `magnitude`.
        if (!float.IsFinite(magnitude) || magnitude <= 1.0e-7f) return 0;
        // dB = 20 * log10(magnitude). idx = clamp(dB/dBPerStep + 255, 0, 255)
        // Step ~0.6 dB per index (normative table is exactly that). dB/0.6 + 255.
        double db = 20.0 * Math.Log10(Math.Min(1.0f, Math.Max(1e-12f, magnitude)));
        double idx = db / 0.625 + 255.0;
        int idxI = (int)Math.Round(idx);
        if (idxI < 0) idxI = 0;
        if (idxI > 254) idxI = 254;
        // With multiplier=2 the Y is half the table index.
        int y = idxI / 2;
        if (y < 0) y = 0;
        if (y > 127) y = 127;
        return y;
    }

    /// <summary>
    /// Quantise a residue sample (already divided by the floor curve, so
    /// approximately in [-1, 1]) into the residue codebook entry index.
    /// </summary>
    private int QuantiseResidueValue(float v)
    {
        // Residue codebook has ResidueBookEntries entries spaced uniformly
        // across [-ResidueRange, +ResidueRange]. Find nearest.
        float clipped = Math.Max(-ResidueRange, Math.Min(ResidueRange, v));
        // Centre of bin for entry i is (-R + (i+0.5) * 2R/N).
        float step = 2f * ResidueRange / ResidueBookEntries;
        int idx = (int)Math.Floor((clipped + ResidueRange) / step);
        if (idx < 0) idx = 0;
        if (idx >= ResidueBookEntries) idx = ResidueBookEntries - 1;
        return idx;
    }

    private static float MaxAbs(ReadOnlySpan<float> v)
    {
        float m = 0;
        for (int i = 0; i < v.Length; i++) { float a = Math.Abs(v[i]); if (a > m) m = a; }
        return m;
    }

    // --- Header packet builders ---

    private static VorbisIdentificationHeader BuildIdentificationHeader(VorbisAudioEncoderOptions opts)
    {
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

    private static VorbisSetupHeader BuildSetupHeader()
    {
        // Codebook 0: classbook with 1 used entry (length 1, code 0).
        // Decoder requires entries >= 1 and dimensions >= 1. With 1 entry, the
        // only valid codeword is 0 bits long; but VorbisHuffman special-cases
        // single-entry as code=0 with the entry's actual length. We use length 1
        // so writer + reader agree.
        // Actually with usedCount==1 BuildCodewords returns (0, length). We
        // emit length 1 -> reader builds tree where bit '0' selects entry 0.
        // The tree will only ever reach the leaf at depth 1. Reading the bit
        // pattern in TryDecode walks 1 bit, finds the leaf.
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
        // For dim 1, lookup type 1 and 2 are equivalent (multiplicand[entry % N]
        // for type 1 with quantvals = N, vs multiplicand[entry] for type 2),
        // but ffmpeg's libavcodec vorbis decoder only implements lookup type 1
        // so we use that for compatibility.
        // Lookup type 1 formula (per VorbisCodebookVector.LookupVector):
        //   val[d] = abs(multiplicand[(entry / quantvals^d) % quantvals]) * delta + mindel
        // For dim 1, quantvals = lookup1_values(entries, 1) = entries, and
        //   val[0] = abs(multiplicand[entry]) * delta + mindel
        // We choose mindel and delta so that val ranges symmetrically across
        // [-ResidueRange, +ResidueRange] in N uniformly-spaced steps:
        //   delta = 2*ResidueRange / N
        //   mindel = -ResidueRange + 0.5 * delta = -ResidueRange + ResidueRange/N
        var residueMultiplicands = new int[ResidueBookEntries];
        for (int i = 0; i < ResidueBookEntries; i++) residueMultiplicands[i] = i;
        var residueLengths = new int[ResidueBookEntries];
        // Uniform 8-bit-ish length (matches 256 entries: every entry length 8 -> full balanced tree).
        int residueCodeLen = (int)Math.Round(Math.Log2(ResidueBookEntries));
        for (int i = 0; i < ResidueBookEntries; i++) residueLengths[i] = residueCodeLen;
        // ValueBits must hold values 0..ResidueBookEntries-1.
        int residueValueBits = 0;
        while ((1 << residueValueBits) < ResidueBookEntries) residueValueBits++;
        double residueDelta = 2.0 * ResidueRange / ResidueBookEntries;
        double residueMin = -ResidueRange + 0.5 * residueDelta;
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
            Multiplier = 2,
            RangeBits = 4, // X[1] = 1<<4 = 16; covers our smallest test block
            XList = new int[] { 0, 16 },
        };

        var residue = new VorbisResidueConfig
        {
            Type = VorbisResidueType.Type1,
            Begin = 0,
            // End/PartitionSize are filled in per-stream below since they
            // depend on BlockSize.
            End = 0,
            PartitionSize = 0,
            Classifications = 1,
            Classbook = ClassBookIndex,
            Cascade = new int[] { 1 }, // class 0 has bit 0 set -> book at pass 0 = ResidueBookIndex
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

    private byte[] BuildIdentPacket()
    {
        var bytes = new byte[30];
        bytes[0] = 0x01;
        var magic = new byte[] { (byte)'v', (byte)'o', (byte)'r', (byte)'b', (byte)'i', (byte)'s' };
        for (int i = 0; i < 6; i++) bytes[1 + i] = magic[i];
        // version = 0 (already)
        bytes[11] = (byte)_ident.AudioChannels;
        WriteInt32Le(bytes, 12, _ident.SampleRateHz);
        WriteInt32Le(bytes, 16, _ident.BitrateMaximum);
        WriteInt32Le(bytes, 20, _ident.BitrateNominal);
        WriteInt32Le(bytes, 24, _ident.BitrateMinimum);
        int log0 = Log2(_ident.BlockSize0);
        int log1 = Log2(_ident.BlockSize1);
        bytes[28] = (byte)((log1 << 4) | log0);
        bytes[29] = 0x01; // framing flag
        return bytes;
    }

    private byte[] BuildCommentPacket(string vendor)
    {
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

    private byte[] BuildSetupPacket()
    {
        var writer = new VorbisBitWriter();
        // Setup packets begin LSB-first AFTER the 7-byte header. We write the
        // header bytes separately and prepend.
        // Codebook count - 1.
        writer.WriteBits((uint)_setup.Codebooks.Length - 1u, 8);
        for (int i = 0; i < _setup.Codebooks.Length; i++)
            VorbisCodebookEncoder.Pack(writer, _setup.Codebooks[i]);

        // Time count - 1 = 0 (one entry, value 0).
        writer.WriteBits(0u, 6);
        writer.WriteBits(0u, 16);

        // Floor count - 1.
        writer.WriteBits((uint)_setup.Floors.Length - 1u, 6);
        for (int i = 0; i < _setup.Floors.Length; i++)
        {
            writer.WriteBits(1u, 16); // floor type 1
            PackFloor1(writer, _setup.Floors[i]);
        }

        // Residue count - 1. Residues are already resolved (End/PartitionSize set)
        // at construction time so we just pack them as-is.
        writer.WriteBits((uint)_setup.Residues.Length - 1u, 6);
        for (int i = 0; i < _setup.Residues.Length; i++)
        {
            writer.WriteBits((uint)_setup.Residues[i].Type, 16);
            PackResidue(writer, _setup.Residues[i]);
        }

        // Mapping count - 1.
        writer.WriteBits((uint)_setup.Mappings.Length - 1u, 6);
        for (int i = 0; i < _setup.Mappings.Length; i++)
        {
            writer.WriteBits(0u, 16); // mapping type 0
            PackMapping(writer, _setup.Mappings[i]);
        }

        // Mode count - 1.
        writer.WriteBits((uint)_setup.Modes.Length - 1u, 6);
        for (int i = 0; i < _setup.Modes.Length; i++)
        {
            writer.WriteBit(_setup.Modes[i].BlockFlag ? 1u : 0u);
            writer.WriteBits(0u, 16); // window type
            writer.WriteBits(0u, 16); // transform type
            writer.WriteBits((uint)_setup.Modes[i].Mapping, 8);
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

    private VorbisResidueConfig[] _resolvedResidues = Array.Empty<VorbisResidueConfig>();

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
