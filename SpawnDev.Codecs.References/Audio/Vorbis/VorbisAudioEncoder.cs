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

// VorbisAudioEncoderOptions moved to main library (used by both this CPU
// reference encoder and VorbisAudioEncoderGpu).

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

        // Header construction is the same logic VorbisHeaderPackBuilder runs.
        // Centralize so the CPU encoder + VorbisAudioEncoderGpu agree byte-for-byte.
        (_ident, _setup) = VorbisHeaderPackBuilder.BuildResolvedHeaders(_opts);
        _resolvedResidues = _setup.Residues;

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
    // Residue VQ covers a normalised range. Because the floor curve is now
    // fitted to the spectrum envelope per block (see EncodeAudioPacket step
    // 3), residue = spectrum / floor stays in roughly [-1, +1]. We size the
    // codebook to cover slightly past +/- 1 to absorb under-fits where the
    // local spectrum magnitude exceeds the chosen floor endpoint.
    //
    // 1024 entries spread across [-2, +2] gives step = 4/1024 = 0.0039,
    // ~0.4% relative quantisation per residue value. That matches the SNR
    // libvorbis achieves with its production residue books on real music
    // (~21 dB on BBB at default quality).
    //
    // The residue VQ is encoded with a fixed-length code per entry
    // (ceil(log2(entries)) bits) so the in-stream cost of one block is
    // halfBlock * residueCodeLen bits. With halfBlock=512 and 10-bit codes
    // that's ~640 bytes per audio packet; the resulting .ogg fits well
    // within typical Vorbis bitrate budgets for the BBB benchmark.
    private const int ResidueBookEntries = 1024;
    private const float ResidueRange = 2.0f;

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
        //    Apply the 4/N normalization on the encoder side per libvorbis
        //    convention (lib/mdct.c sets lookup->scale = 4.f/n in mdct_init
        //    and applies it inside mdct_forward; mdct_backward is unscaled).
        //    Without this, our spectrum would be N/4 times libvorbis's, and
        //    third-party decoders (ffmpeg/libavcodec) would emit at N/4 the
        //    intended amplitude (deafening / clipping). MdctReference itself
        //    stays the literal direct formula; the scale lives at the Vorbis
        //    boundary so the same reference transform can be reused by codecs
        //    that put the normalization on the inverse side.
        var spectrum = new float[half];
        MdctReference.Transform(windowed, spectrum);
        float forwardScale = 4f / n;
        for (int i = 0; i < half; i++) spectrum[i] *= forwardScale;

        // 3. Floor curve: fit a piecewise-linear envelope to the actual spectrum
        //    per block so residue = spectrum / floor stays in a normalised
        //    range. The floor configuration uses 2 endpoints (X=0 and
        //    X=halfBlock); we pick endpoint Y values from the magnitude of
        //    the spectrum in the lower and upper halves of the band.
        //
        //    Per the inverse-dB lookup table (Vorbis I Section 10.1, used by
        //    VorbisFloor1Curve.Render via multiplier=1), Y in [0, 255] maps
        //    to floor magnitude in [1.06e-7, 1.0]. We choose Y so that the
        //    floor is just above the local spectrum peak: the residue value
        //    spectrum / floor then falls within roughly [-1, +1] and the
        //    1024-entry residue codebook over [-2, +2] quantises it with
        //    ~0.4% relative precision per bin.
        //
        //    Without per-block floor tracking the residue is dominated by
        //    quantisation noise on real music (BBB block spectrum peaks are
        //    ~0.005-0.01, far smaller than the codebook step that would
        //    cover [-4, +4]/256 = 0.031). Tracking the envelope is the
        //    fundamental purpose of floor 1 in Vorbis - this is what
        //    libvorbis lib/floor1.c does (much more elaborately) per packet.
        var floorCfg = (VorbisFloor1Config)_setup.Floors[0];
        // Spectrum magnitudes per half-band for envelope estimation.
        int splitBin = half / 2;
        float specPeakLow = 0, specPeakHigh = 0;
        for (int i = 0; i < splitBin; i++) { float a = Math.Abs(spectrum[i]); if (a > specPeakLow) specPeakLow = a; }
        for (int i = splitBin; i < half; i++) { float a = Math.Abs(spectrum[i]); if (a > specPeakHigh) specPeakHigh = a; }
        // Pick a slight headroom factor so the floor sits a touch above the
        // peak; this keeps the residue values comfortably inside the codebook
        // range even when bins fluctuate around the local peak.
        const float FloorHeadroom = 1.25f;
        int posteriorLow = MagnitudeToFloorY(specPeakLow * FloorHeadroom);
        int posteriorHigh = MagnitudeToFloorY(specPeakHigh * FloorHeadroom);
        // Guard against fully-silent blocks - emit a tiny floor so residue
        // = 0 / floor = 0 is well-defined, and the bitstream stays valid.
        if (posteriorLow < 1) posteriorLow = 1;
        if (posteriorHigh < 1) posteriorHigh = 1;
        var posteriors = new int[] { posteriorLow, posteriorHigh };
        var floorCurve = new float[half];
        VorbisFloor1Curve.Render(floorCfg, posteriors, half, floorCurve);

        // 4. Compute residue = spectrum / floor, quantised to the residue VQ.
        //    Bins below a small fraction of the local floor quantise to the
        //    zero-anchored entry to avoid emitting half-step noise on
        //    inaudible content. With per-block floor tracking the threshold
        //    is relative to the FLOOR (not the spectrum peak) so that quiet
        //    high-frequency detail rides the high-band floor correctly.
        var residueQ = new int[half];
        int zeroEntry = QuantiseResidueValue(0f);
        for (int i = 0; i < half; i++)
        {
            float floor = Math.Max(floorCurve[i], 1e-12f);
            float r = spectrum[i] / floor;
            // Noise-gate: residue magnitudes below half of one quantisation
            // step round to the zero entry exactly via QuantiseResidueValue,
            // so no extra threshold is needed here.
            residueQ[i] = QuantiseResidueValue(r);
        }

        // 5. Bit-pack the audio packet.
        var writer = new VorbisBitWriter();

        // Audio header: type bit (0) + 1-bit mode (we have 1 mode so 0 bits, but ilog(0)=0).
        writer.WriteBit(0u); // packet type = 0 (audio)
        // ilog(modes-1) = ilog(0) = 0 bits for mode; nothing to write.

        // Floor 1 nonzero bit + 2 endpoint Y values. Endpoint bit width
        // depends on the floor multiplier: 1->8, 2->7, 3->7, 4->6 (Vorbis I
        // Table 7.2.4). Our floor uses multiplier=1 (full 256-step range)
        // so each endpoint is 8 bits.
        writer.WriteBit(1u); // nonzero
        int endpointBits = floorCfg.Multiplier switch { 1 => 8, 2 => 7, 3 => 7, 4 => 6, _ => 8 };
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
    /// Map a linear magnitude value to an 8-bit floor Y coordinate
    /// (multiplier=1, range=256). Higher Y -> bigger floor magnitude in
    /// <see cref="VorbisFloor1Curve"/>'s inverse-dB lookup.
    /// </summary>
    private static int MagnitudeToFloorY(float magnitude)
    {
        // VorbisFloor1Curve uses an inverse-dB table (Vorbis I Section 10.1)
        // that runs from ~1.065e-7 (Y=0) to 1.0 (Y=255), spanning 139.45 dB
        // across 256 steps -> 0.547 dB per Y, or 0.02735 in log10 units.
        // With multiplier=1, posterior Y values map directly to table
        // indices 0..255. We pick the Y that makes table[Y] >= magnitude
        // (ceil rather than round) so residue r = spectrum / floor stays
        // bounded in [-1, +1].
        if (!float.IsFinite(magnitude) || magnitude <= 1.0649863e-7f) return 0;
        if (magnitude >= 1.0f) return 255;
        // log10 / 0.02735 + 255: solves for Y given table approximation.
        double idx = Math.Log10(magnitude) / 0.02735 + 255.0;
        int y = (int)Math.Ceiling(idx);
        if (y < 0) y = 0;
        if (y > 255) y = 255;
        return y;
    }

    /// <summary>
    /// Quantise a residue sample (already divided by the floor curve) into the
    /// residue codebook entry index. Codebook layout is anchored so entry
    /// <c>ResidueBookEntries/2</c> decodes to exactly 0; entry <c>i</c> decodes
    /// to <c>(i - N/2) * step</c> where <c>step = 2R/N</c>.
    /// </summary>
    private int QuantiseResidueValue(float v)
    {
        // Codebook is anchored: entry 0 = -ResidueRange, entry N/2 = 0,
        // entry N-1 = +ResidueRange - step. Find nearest entry.
        float step = 2f * ResidueRange / ResidueBookEntries;
        int half = ResidueBookEntries / 2;
        // Inverse of decode formula val = (i - half) * step:
        //   i = round(v / step) + half
        int idx = (int)Math.Round(v / step) + half;
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
    //
    // All header construction (BuildIdentificationHeader, BuildSetupHeader,
    // BuildIdentPacketBytes, BuildCommentPacketBytes, BuildSetupPacketBytes,
    // PackFloor1, PackResidue, PackMapping, plus the WriteInt32Le / WriteUInt32Le
    // / Log2 helpers) moved to `VorbisHeaderPackBuilder` (main library) so
    // `VorbisAudioEncoderGpu` can call them without instantiating a CPU
    // encoder. Behavior unchanged - the constructor + instance methods below
    // delegate to the helper class.

    internal byte[] BuildIdentPacket() => VorbisHeaderPackBuilder.BuildIdentPacketBytes(_ident);

    internal byte[] BuildCommentPacket(string vendor) => VorbisHeaderPackBuilder.BuildCommentPacketBytes(vendor);

    internal byte[] BuildSetupPacket() => VorbisHeaderPackBuilder.BuildSetupPacketBytes(_setup);

    private VorbisResidueConfig[] _resolvedResidues = Array.Empty<VorbisResidueConfig>();
}
