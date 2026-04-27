// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Stateful Vorbis audio-packet decoder. Composes:
//   1. VorbisAudioPacketHeaderParser (mode + block size + window flags)
//   2. VorbisFloor1Decoder per channel (Y posteriors)
//   3. VorbisFloor1Curve per channel (posterior -> spectral envelope)
//   4. VorbisResidueDecoder per submap (residue vectors per channel)
//   5. Multiply floor curve x residue -> spectral coefficients
//   6. VorbisInverseCoupling (undoes M/S-like decorrelation)
//   7. ImdctReference per channel -> 2N time-domain samples
//   8. VorbisWindow.GenerateCanonical x the IMDCT output (elementwise)
//   9. VorbisWindow.OverlapAdd with the previous packet's right half
// The first packet produces NO audio output (no previous right half);
// subsequent packets emit blockSize/2 samples interleaved across channels.

using SpawnDev.Codecs.Audio.Transforms;

namespace SpawnDev.Codecs.Audio.Vorbis;

/// <summary>
/// Stateful Vorbis audio-packet decoder. Construct once per stream with the
/// parsed identification and setup headers; feed audio packets via
/// <see cref="DecodePacket"/>. Maintains the per-channel overlap-add buffer
/// between packets.
/// </summary>
public sealed class VorbisAudioDecoder
{
    private readonly VorbisIdentificationHeader _ident;
    private readonly VorbisSetupHeader _setup;
    private readonly VorbisHuffmanDecoder[] _huffmanDecoders;
    private readonly float[]?[] _previousRightHalf;
    private int _previousBlockSize = 0;

    /// <summary>Construct a Vorbis audio decoder from the parsed stream headers.</summary>
    public VorbisAudioDecoder(VorbisIdentificationHeader ident, VorbisSetupHeader setup)
    {
        _ident = ident ?? throw new ArgumentNullException(nameof(ident));
        _setup = setup ?? throw new ArgumentNullException(nameof(setup));
        // Pre-build Huffman decoders once per codebook (re-used across packets).
        _huffmanDecoders = new VorbisHuffmanDecoder[setup.Codebooks.Length];
        for (int i = 0; i < setup.Codebooks.Length; i++)
        {
            var cb = setup.Codebooks[i];
            _huffmanDecoders[i] = new VorbisHuffmanDecoder(VorbisHuffman.Build(cb.Lengths));
        }
        _previousRightHalf = new float[ident.AudioChannels][];
    }

    /// <summary>
    /// Decode one audio packet into interleaved float PCM samples. Returns
    /// the number of sample frames written (the first packet returns 0 since
    /// it only primes the overlap-add state).
    /// </summary>
    public int DecodePacket(ReadOnlySpan<byte> packet, Span<float> interleavedOut)
    {
        if (packet.Length == 0) return 0;

        var bitReader = new VorbisBitReader(packet);
        var header = VorbisAudioPacketHeaderParser.ParseFromReader(ref bitReader, _setup, _ident);
        int blockSize = header.BlockSize;
        int halfBlock = blockSize / 2;
        int channels = _ident.AudioChannels;

        // ----- Per-channel floor decode -----
        var mapping = _setup.Mappings[_setup.Modes[header.ModeNumber].Mapping];
        var floorOk = new bool[channels];
        var floorCurves = new float[channels][];
        for (int ch = 0; ch < channels; ch++)
        {
            int submap = mapping.Mux[ch];
            int floorIdx = mapping.SubmapFloor[submap];
            var floorCfg = _setup.Floors[floorIdx];
            int[]? posterior = VorbisFloor1Decoder.Decode(ref bitReader, floorCfg, _huffmanDecoders);
            floorOk[ch] = posterior is not null;
            if (floorOk[ch])
            {
                var curve = new float[halfBlock];
                VorbisFloor1Curve.Render(floorCfg, posterior!, halfBlock, curve);
                floorCurves[ch] = curve;
            }
            else
            {
                floorCurves[ch] = new float[halfBlock]; // zeros
            }
        }

        // ----- Residue decode -----
        // do_not_decode flag: true when both channels in a coupling pair have
        // silent floors. Simplified: per-channel doNotDecode == !floorOk.
        var doNotDecode = new bool[channels];
        for (int ch = 0; ch < channels; ch++) doNotDecode[ch] = !floorOk[ch];

        var residueBuffers = new float[channels][];
        for (int ch = 0; ch < channels; ch++) residueBuffers[ch] = new float[halfBlock];

        // One residue call per submap; each submap's residue covers the channels
        // whose mux points to it. For simplicity we run one residue pass for
        // submap 0 across all channels (the common single-submap case).
        for (int s = 0; s < mapping.Submaps; s++)
        {
            int residueIdx = mapping.SubmapResidue[s];
            var residueCfg = _setup.Residues[residueIdx];
            // Collect the per-channel slices that belong to this submap.
            int membersInSubmap = 0;
            for (int ch = 0; ch < channels; ch++) if (mapping.Mux[ch] == s) membersInSubmap++;
            if (membersInSubmap == 0) continue;
            var subBuffers = new float[membersInSubmap][];
            var subSkip = new bool[membersInSubmap];
            var channelMap = new int[membersInSubmap];
            int k = 0;
            for (int ch = 0; ch < channels; ch++)
            {
                if (mapping.Mux[ch] == s)
                {
                    subBuffers[k] = residueBuffers[ch];
                    subSkip[k] = doNotDecode[ch];
                    channelMap[k] = ch;
                    k++;
                }
            }
            VorbisResidueDecoder.Decode(
                ref bitReader, residueCfg, _huffmanDecoders, _setup.Codebooks,
                subBuffers, subSkip, halfBlock);
        }

        // ----- Multiply floor curve by residue per channel -----
        var spectra = new float[channels][];
        for (int ch = 0; ch < channels; ch++)
        {
            var spec = new float[halfBlock];
            if (floorOk[ch])
            {
                for (int i = 0; i < halfBlock; i++)
                    spec[i] = floorCurves[ch][i] * residueBuffers[ch][i];
            }
            spectra[ch] = spec;
        }

        // ----- Inverse channel coupling -----
        VorbisInverseCoupling.Apply(spectra, mapping);

        // ----- IMDCT + window per channel -----
        // Our ImdctReference is the literal unscaled inverse (matches the
        // direct-formula MDCT in MdctReference). The MDCT->IMDCT round-trip
        // for these reference impls produces N/4 times the original signal.
        // Vorbis I expects unity round-trip, so we scale IMDCT output by 4/N.
        // libvorbis bakes this normalisation into its FFT-based MDCT; our
        // reference impls keep the bare formula and do the scaling at the
        // decoder boundary so MdctReference and ImdctReference stay literal.
        var timeDomain = new float[channels][];
        var window = VorbisWindow.GenerateCanonical(blockSize);
        float imdctScale = 4f / blockSize;
        for (int ch = 0; ch < channels; ch++)
        {
            var td = new float[blockSize];
            ImdctReference.Transform(spectra[ch], td);
            for (int i = 0; i < blockSize; i++) td[i] *= window[i] * imdctScale;
            timeDomain[ch] = td;
        }

        // ----- Overlap-add against previous packet's right halves -----
        bool havePrevious = _previousBlockSize == blockSize
            && _previousRightHalf[0] is not null;
        int outFrames = 0;
        if (havePrevious)
        {
            outFrames = halfBlock;
            if (interleavedOut.Length < outFrames * channels)
                throw new ArgumentException(
                    $"Output buffer too small: {interleavedOut.Length} < {outFrames * channels}.");
            for (int n = 0; n < outFrames; n++)
            {
                for (int ch = 0; ch < channels; ch++)
                {
                    interleavedOut[n * channels + ch] =
                        _previousRightHalf[ch]![n] + timeDomain[ch][n];
                }
            }
        }

        // Save current packet's right half for next time.
        for (int ch = 0; ch < channels; ch++)
        {
            if (_previousRightHalf[ch] is null || _previousRightHalf[ch]!.Length != halfBlock)
                _previousRightHalf[ch] = new float[halfBlock];
            for (int n = 0; n < halfBlock; n++)
                _previousRightHalf[ch]![n] = timeDomain[ch][halfBlock + n];
        }
        _previousBlockSize = blockSize;

        return outFrames;
    }
}

