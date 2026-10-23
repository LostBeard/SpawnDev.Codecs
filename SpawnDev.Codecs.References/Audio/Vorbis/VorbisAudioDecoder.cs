// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Stateful Vorbis audio-packet decoder (Vorbis I sec 4.3). Composes:
//   1. VorbisAudioPacketHeaderParser (mode + block size + window flags)
//   2. VorbisFloor1Decoder per channel (Y posteriors; null = floor unused)
//   3. Nonzero vector propagate: a coupling step with one used floor decodes both residues (sec 4.3.4)
//   4. VorbisResidueDecoder per submap (residue vectors per channel)
//   5. VorbisInverseCoupling on the RESIDUE vectors (sec 4.3.6)
//   6. Floor curve x residue -> spectral coefficients (an unused floor gives zeros) (sec 4.3.7)
//   7. ImdctReference per channel -> blockSize time-domain samples
//   8. VorbisWindow.Generate (the block's window, shaped by its neighbours' sizes) x the IMDCT output
//   9. Overlap-add with the previous block (sec 4.3.8): a packet returns the samples from the previous
//      block's centre to this block's centre, prevBlock/4 + curBlock/4 frames.
// The first packet produces NO audio output (no previous block).

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
    // Windowed right half of the previous block per channel ([blockSize/2, blockSize) of its output).
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

    /// <summary>Most sample frames one packet can return (two long blocks: blockSize1 / 2).</summary>
    public int MaxFramesPerPacket => _ident.BlockSize1 / 2;

    /// <summary>
    /// Decode one audio packet into interleaved float PCM samples. Returns
    /// the number of sample frames written: 0 for the first packet (it only
    /// primes the overlap-add state), else previousBlockSize / 4 + blockSize / 4.
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
        }

        // ----- Nonzero vector propagate (sec 4.3.4) -----
        // A channel's residue is decoded unless its floor is unused; a coupling step whose magnitude OR angle
        // floor is used decodes BOTH residues (the inverse coupling needs both vectors).
        var doNotDecode = new bool[channels];
        for (int ch = 0; ch < channels; ch++) doNotDecode[ch] = !floorOk[ch];
        for (int i = 0; i < mapping.CouplingMagnitudeChannels.Length; i++)
        {
            int mag = mapping.CouplingMagnitudeChannels[i], ang = mapping.CouplingAngleChannels[i];
            if (!doNotDecode[mag] || !doNotDecode[ang]) { doNotDecode[mag] = false; doNotDecode[ang] = false; }
        }

        var residueBuffers = new float[channels][];
        for (int ch = 0; ch < channels; ch++) residueBuffers[ch] = new float[halfBlock];

        // ----- Residue decode: one pass per submap over the channels whose mux points to it -----
        for (int s = 0; s < mapping.Submaps; s++)
        {
            int residueIdx = mapping.SubmapResidue[s];
            var residueCfg = _setup.Residues[residueIdx];
            int membersInSubmap = 0;
            for (int ch = 0; ch < channels; ch++) if (mapping.Mux[ch] == s) membersInSubmap++;
            if (membersInSubmap == 0) continue;
            var subBuffers = new float[membersInSubmap][];
            var subSkip = new bool[membersInSubmap];
            int k = 0;
            for (int ch = 0; ch < channels; ch++)
            {
                if (mapping.Mux[ch] == s)
                {
                    subBuffers[k] = residueBuffers[ch];
                    subSkip[k] = doNotDecode[ch];
                    k++;
                }
            }
            VorbisResidueDecoder.Decode(
                ref bitReader, residueCfg, _huffmanDecoders, _setup.Codebooks,
                subBuffers, subSkip, halfBlock);
        }

        // ----- Inverse channel coupling on the residue vectors (sec 4.3.6) -----
        VorbisInverseCoupling.Apply(residueBuffers, mapping);

        // ----- Floor curve x residue per channel (sec 4.3.7); an unused floor zeroes the channel -----
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

        // ----- IMDCT + window per channel -----
        // Per libvorbis convention (lib/mdct.c), the 4/N normalisation lives
        // on the FORWARD MDCT (encoder side), and mdct_backward is unscaled.
        // We follow that convention: VorbisAudioEncoder applies 4/N to its
        // MdctReference output, and here we feed that libvorbis-scaled
        // spectrum directly into ImdctReference with no extra scale.
        // The window's slopes follow the overlapping neighbours (sec 4.3.1): a long block's left / right slope is
        // short when the previous / next window flag says so; a short block always overlaps with short slopes.
        int bs0 = _ident.BlockSize0;
        int leftN = header.IsLongBlock && !header.PreviousWindowLong ? bs0 : blockSize;
        int rightN = header.IsLongBlock && !header.NextWindowLong ? bs0 : blockSize;
        var window = VorbisWindow.Generate(blockSize, leftN, rightN);
        var timeDomain = new float[channels][];
        for (int ch = 0; ch < channels; ch++)
        {
            var td = new float[blockSize];
            ImdctReference.Transform(spectra[ch], td);
            for (int i = 0; i < blockSize; i++) td[i] *= window[i];
            timeDomain[ch] = td;
        }

        // ----- Overlap-add with the previous block (sec 4.3.8) -----
        // The packet returns [previous centre, current centre): prevN/4 + n/4 frames. Output frame t is the
        // previous block's sample prevN/2 + t plus the current block's sample t - prevN/4 + n/4 (the blocks are
        // placed so the previous block's 3/4 point meets the current block's 1/4 point); outside a block's
        // range its contribution is zero (its window is zero there).
        int outFrames = 0;
        if (_previousRightHalf[0] is not null)
        {
            int prevN = _previousBlockSize;
            int prevHalf = prevN / 2;
            int curShift = prevN / 4 - blockSize / 4;
            outFrames = prevN / 4 + blockSize / 4;
            if (interleavedOut.Length < outFrames * channels)
                throw new ArgumentException(
                    $"Output buffer too small: {interleavedOut.Length} < {outFrames * channels}.");
            for (int ch = 0; ch < channels; ch++)
            {
                var prev = _previousRightHalf[ch]!;
                var td = timeDomain[ch];
                for (int t = 0; t < outFrames; t++)
                {
                    int c = t - curShift;
                    float v = t < prevHalf ? prev[t] : 0f;
                    if (c >= 0) v += td[c];
                    interleavedOut[t * channels + ch] = v;
                }
            }
        }

        // Save current block's right half for next time.
        for (int ch = 0; ch < channels; ch++)
        {
            if (_previousRightHalf[ch] is null || _previousRightHalf[ch]!.Length != halfBlock)
                _previousRightHalf[ch] = new float[halfBlock];
            Array.Copy(timeDomain[ch], halfBlock, _previousRightHalf[ch]!, 0, halfBlock);
        }
        _previousBlockSize = blockSize;

        return outFrames;
    }
}
