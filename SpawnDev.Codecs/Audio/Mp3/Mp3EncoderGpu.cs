// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt). See NOTICE.md for upstream attributions.
//
// MP3 (MPEG-1 / 2 / 2.5 Layer III) encoder on the accelerator, written from ISO/IEC 11172-3 / 13818-3 (no LAME code;
// see Plans/PLAN-MP3.md). This stage: constant bit rate, long blocks, left / right stereo or mono, every Layer III
// sample rate (8000..48000 Hz), each frame holding its own main data (no bit reservoir yet), quantization noise flat
// across the spectrum (no psychoacoustic model yet). The files are valid MP3 that every decoder plays.
//
// Host work (pure coordinator): upload the PCM, compute the frame sizes of the bit rate (padding rule), dispatch
// Mp3EncodeKernels, read back the file.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using K = SpawnDev.Codecs.Audio.Mp3.Mp3EncodeKernels;

namespace SpawnDev.Codecs.Audio.Mp3;

/// <summary>Settings of <see cref="Mp3EncoderGpu"/>.</summary>
public sealed record Mp3EncodeOptions
{
    /// <summary>Constant bit rate in kbps: MPEG-1 (32 / 44.1 / 48 kHz) 32..320, MPEG-2 / 2.5 (8..24 kHz) 8..160. Ignored
    /// when <see cref="VbrQuality"/> is set.</summary>
    public int BitrateKbps { get; init; } = 128;

    /// <summary>Variable bit rate at quality 0 (best, ComfyUI's "V0") .. 9 (smallest), or null for constant bit rate.
    /// Each granule gets the fewest bits that keep its quantization noise within the psychoacoustic masks, scaled from
    /// 6 dB below them (0) to 6 dB above (9); each frame takes the lowest bit rate that holds it.</summary>
    public int? VbrQuality { get; init; }

    /// <summary>Joint stereo with mid / side coding of the frames whose channels are alike. Default true.</summary>
    public bool MidSide { get; init; } = true;

    /// <summary>Write an ID3v2.4 tag first (the tags below as TXXX frames, then TSSE). Default true.</summary>
    public bool WriteId3 { get; init; } = true;

    /// <summary>User text fields written as ID3v2.4 TXXX frames, in order (ComfyUI: prompt, workflow).</summary>
    public IReadOnlyList<KeyValuePair<string, string>>? Tags { get; init; }

    /// <summary>The ID3v2.4 TSSE (encoder settings) text. Default "Lavf62.12.102", the value ComfyUI's PyAV writes.</summary>
    public string Id3Encoder { get; init; } = "Lavf62.12.102";

    /// <summary>Write the Info / Xing frame (frame count, seek table, encoder delay and padding for gapless decoding).
    /// Default true.</summary>
    public bool WriteInfoFrame { get; init; } = true;

    /// <summary>The 9-byte encoder string of the Info frame's LAME-format extension (padded with NULs). Default "Lavf",
    /// the value ComfyUI's PyAV files carry (Lore measured; the ffmpeg command-line tool writes "Lavc62.28").</summary>
    public string InfoEncoder { get; init; } = "Lavf";
}

/// <summary>MP3 encoder on the accelerator (see the file header).</summary>
public sealed class Mp3EncoderGpu : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, long, int, long> _polyphase;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, long> _matrix;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int, long> _mdct;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, long> _alias;
    private readonly Action<Index1D, ArrayView<long>, ArrayView<int>, ArrayView<int>, ArrayView<float>, long, long, int, int, int, int, int, long> _layout;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<int>, ArrayView<float>, int> _pe;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, long, int> _musicCrc;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<long>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<long>> _infoFrame;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _crcTable;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<int>, ArrayView<float>, ArrayView<float>, ArrayView<float>, float, int> _psy;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<float>, ArrayView<int>, int, int, int> _quantize;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<float>, int, int> _vbrQuantize;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<long>, ArrayView<int>, ArrayView<long>, ArrayView<int>, int, int, int, int, int, int, long> _vbrLayout;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<byte>, int, int> _writeGranule;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<byte>, ArrayView<long>, ArrayView<int>, ArrayView<byte>, ArrayView<int>, int, int, int, int, int> _writeFrame;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<int>, int, int, int> _midSide;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<float>, long> _toFloat;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _consts;

    /// <summary>Samples between an input sample and the same sample in a decoder's output (481 + 576).</summary>
    public const int CodecDelay = 1057;

    /// <summary>An encoder bound to <paramref name="accelerator"/>.</summary>
    public Mp3EncoderGpu(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        var a = accelerator;
        _polyphase = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, long, int, long>(K.PolyphaseKernel);
        _matrix = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, long>(K.MatrixKernel);
        _mdct = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int, long>(K.MdctKernel);
        _alias = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, long>(K.AliasReduceKernel);
        _layout = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<long>, ArrayView<int>, ArrayView<int>, ArrayView<float>, long, long, int, int, int, int, int, long>(K.FrameLayoutKernel);
        _pe = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<int>, ArrayView<float>, int>(K.PerceptualEntropyKernel);
        _musicCrc = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, long, int>(K.MusicCrcKernel);
        _infoFrame = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<long>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<long>>(K.InfoFrameKernel);
        var crc = new int[256];
        for (int i = 0; i < 256; i++)
        {
            int c = i;
            for (int k = 0; k < 8; k++) c = (c & 1) != 0 ? (c >> 1) ^ 0xA001 : c >> 1;
            crc[i] = c;
        }
        _crcTable = a.Allocate1D<int>(256);
        _crcTable.View.CopyFromCPU(crc);
        _psy = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<int>, ArrayView<float>, ArrayView<float>, ArrayView<float>, float, int>(K.PsyKernel);
        _quantize = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<float>, ArrayView<int>, int, int, int>(K.QuantizeKernel);
        _vbrQuantize = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<float>, int, int>(K.VbrQuantizeKernel);
        _vbrLayout = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<long>, ArrayView<int>, ArrayView<long>, ArrayView<int>, int, int, int, int, int, int, long>(K.VbrLayoutKernel);
        _writeGranule = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<byte>, int, int>(K.WriteGranuleKernel);
        _writeFrame = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<byte>, ArrayView<long>, ArrayView<int>, ArrayView<byte>, ArrayView<int>, int, int, int, int, int>(K.WriteFrameKernel);
        _midSide = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<int>, int, int, int>(K.MidSideKernel);
        _toFloat = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<float>, long>(Int32ToFloatKernel);

        // the constant tables (setup data, computed once per encoder)
        var f = new float[K.FloatConsts];
        for (int i = 0; i < 512; i++) f[K.FWin + i] = Mp3Tables.WindowQ16[i] / 65536f / 32f;
        for (int k = 0; k < 32; k++)
            for (int i = 0; i < 64; i++) f[K.FMat + k * 64 + i] = (float)Math.Cos((2 * k + 1) * (i - 16) * Math.PI / 64);
        for (int k = 0; k < 18; k++)
            // the MDCT / IMDCT pair of the standard (both unnormalised) gains n / 4 = 9 for the 36-point long block
            for (int i = 0; i < 36; i++) f[K.FCos + k * 36 + i] = (float)(Math.Cos(Math.PI / 72 * (2 * i + 19) * (2 * k + 1)) / 9.0);
        for (int i = 0; i < 36; i++) f[K.FSin + i] = (float)Math.Sin(Math.PI / 36 * (i + 0.5));
        for (int i = 0; i < 8; i++)
        {
            double c = Mp3Tables.AliasC[i], n = Math.Sqrt(1 + c * c);
            f[K.FCs + i] = (float)(1 / n);
            f[K.FCa + i] = (float)(c / n);
        }
        _consts = a.Allocate1D<float>(K.FloatConsts);
        _consts.View.CopyFromCPU(f);
    }

    /// <summary>int32 full scale (PyAV's s32 samples) to float in [-1, 1).</summary>
    private static void Int32ToFloatKernel(Index1D i, ArrayView<int> src, ArrayView<float> dst, long first)
    {
        long at = first + i;
        dst[at] = src[at] / 2147483648f;
    }

    private static (int Version, int RateIndex, bool Lsf) RateInfo(int sampleRate) => sampleRate switch
    {
        44100 => (3, 0, false), 48000 => (3, 1, false), 32000 => (3, 2, false),
        22050 => (2, 0, true), 24000 => (2, 1, true), 16000 => (2, 2, true),
        11025 => (0, 0, true), 12000 => (0, 1, true), 8000 => (0, 2, true),
        _ => throw new ArgumentOutOfRangeException(nameof(sampleRate), "MP3 rates: 8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000."),
    };

    /// <summary>Encode interleaved float PCM (full scale +-1) to an MP3 file.</summary>
    public async Task<byte[]> EncodeAsync(ReadOnlyMemory<float> pcm, int channels, int sampleRate, Mp3EncodeOptions? options = null)
    {
        using var d = _accelerator.Allocate1D<float>(Math.Max(1, pcm.Length));
        if (pcm.Length > 0) d.View.SubView(0, pcm.Length).CopyFromCPU(pcm.ToArray());
        return await EncodeDeviceAsync(d.View.SubView(0, Math.Max(1, pcm.Length)), pcm.Length / Math.Max(1, channels), channels, sampleRate, options);
    }

    /// <summary>Encode interleaved int32 PCM (full scale, as PyAV hands libmp3lame s32 samples) to an MP3 file.</summary>
    public async Task<byte[]> EncodeAsync(ReadOnlyMemory<int> pcm, int channels, int sampleRate, Mp3EncodeOptions? options = null)
    {
        using var di = _accelerator.Allocate1D<int>(Math.Max(1, pcm.Length));
        if (pcm.Length > 0) di.View.SubView(0, pcm.Length).CopyFromCPU(pcm.ToArray());
        using var d = _accelerator.Allocate1D<float>(Math.Max(1, pcm.Length));
        if (pcm.Length > 0) _toFloat(pcm.Length, di.View, d.View, 0);
        return await EncodeDeviceAsync(d.View.SubView(0, Math.Max(1, pcm.Length)), pcm.Length / Math.Max(1, channels), channels, sampleRate, options);
    }

    private async Task<byte[]> EncodeDeviceAsync(ArrayView<float> pcm, long samples, int channels, int sampleRate, Mp3EncodeOptions? options)
    {
        if (channels is < 1 or > 2) throw new NotSupportedException("MP3 carries 1 or 2 channels.");
        var o = options ?? new Mp3EncodeOptions();
        var (version, rateIndex, lsf) = RateInfo(sampleRate);
        bool vbr = o.VbrQuality is not null;
        if (vbr && o.VbrQuality is < 0 or > 9) throw new ArgumentOutOfRangeException(nameof(options), "VbrQuality is 0..9.");
        int brBase = lsf ? 16 : 0;
        int bitrateIndex = vbr ? 0 : Array.IndexOf(Mp3Tables.BitratesKbps, o.BitrateKbps, brBase, 15) - brBase;
        if (!vbr && bitrateIndex < 1) throw new ArgumentOutOfRangeException(nameof(options), $"{o.BitrateKbps} kbps is not a Layer III bit rate at {sampleRate} Hz.");
        int granulesPerFrame = lsf ? 1 : 2;
        int frameSamples = 576 * granulesPerFrame;
        int sideBytes = lsf ? (channels == 1 ? 9 : 17) : (channels == 1 ? 17 : 32);
        int coeff = lsf ? 72000 : 144000;
        // a sample reaches the decoder's output CodecDelay samples later (polyphase analysis + synthesis 481, MDCT
        // overlap 576): enough frames that the last input sample comes out
        long frames = (samples + CodecDelay + frameSamples - 1) / frameSamples;
        int granules = (int)(frames * granulesPerFrame);
        int slots = granules * 18;
        int units = granules * channels;
        // the Info frame: CBR one unpadded frame of the stream's rate; VBR the lowest rate whose frame holds the tag
        int infoIndex = bitrateIndex;
        if (vbr)
            for (infoIndex = 1; infoIndex < 14; infoIndex++)
                if ((long)coeff * Mp3Tables.BitratesKbps[brBase + infoIndex] / sampleRate >= 4 + sideBytes + 156) break;
        int infoBytes = o.WriteInfoFrame ? (int)((long)coeff * Mp3Tables.BitratesKbps[brBase + infoIndex] / sampleRate) : 0;
        long num = (long)coeff * (vbr ? 0 : o.BitrateKbps);   // CBR bytes per frame = num / rate
        var a = _accelerator;

        using var tables = a.Allocate1D<int>(K.IntConsts);
        tables.View.CopyFromCPU(IntTables(sampleRate));
        using var bitrates = a.Allocate1D<int>(15);
        bitrates.View.CopyFromCPU(Mp3Tables.BitratesKbps.AsSpan(brBase, 15).ToArray());
        using var y = a.Allocate1D<float>((long)channels * slots * 64);
        using var sub = a.Allocate1D<float>((long)channels * slots * 32);
        using var xr = a.Allocate1D<float>((long)units * 576);
        using var ix = a.Allocate1D<int>((long)units * 576);
        using var side = a.Allocate1D<int>((long)units * K.SideInts);
        using var slotBuf = a.Allocate1D<byte>((long)units * K.SlotBytes);
        using var psy = a.Allocate1D<float>(K.PsyConsts);
        psy.View.CopyFromCPU(PsyTables(sampleRate));
        using var psyScratch = a.Allocate1D<float>((long)units * 44);
        using var xmin = a.Allocate1D<float>((long)units * 22);
        using var work = a.Allocate1D<int>((long)units * K.WorkInts);
        using var frameOffsets = a.Allocate1D<long>(frames + 1);
        using var frameInfo = a.Allocate1D<int>(Math.Max(1, frames * 2));
        using var vmeta = a.Allocate1D<long>(K.VMeta);
        using var ms = a.Allocate1D<int>(Math.Max(1, frames));
        ms.MemSetToZero();
        side.MemSetToZero();

        var pacer = new LaunchPacer(a);
        const int Chunk = 1 << 18;
        async Task Launch(long count, Action<int, long> launch)
        {
            for (long first = 0; first < count; first += Chunk)
            {
                launch((int)Math.Min(Chunk, count - first), first);
                await pacer.LaunchedAsync();
            }
        }
        await Launch((long)channels * slots * 64, (n, f) => _polyphase(n, pcm, _consts.View, y.View, channels, samples, slots, f));
        await Launch((long)channels * slots * 32, (n, f) => _matrix(n, y.View, _consts.View, sub.View, f));
        await Launch((long)channels * granules * 32, (n, f) => _mdct(n, sub.View, _consts.View, xr.View, granules, slots, channels, f));
        await Launch((long)units * 31 * 8, (n, f) => _alias(n, xr.View, _consts.View, f));

        // masks: CBR at the model's threshold; VBR scaled by quality, 6 dB below (0) .. 6 dB above (9)
        float maskScale = vbr ? (float)Math.Pow(10, (-6 + 12.0 * o.VbrQuality!.Value / 9) / 10) : 1f;
        const int UnitChunk = 1024;
        for (int first = 0; first < units; first += UnitChunk)
        {
            int n = Math.Min(UnitChunk, units - first);
            _psy(n, xr.View, tables.View, psy.View, psyScratch.View, xmin.View, maskScale, first);
            await pacer.LaunchedAsync();
        }
        if (channels == 2)
        {
            for (int first = 0; first < frames; first += UnitChunk)
            {
                int n = (int)Math.Min(UnitChunk, frames - first);
                _midSide(n, xr.View, xmin.View, ms.View, granulesPerFrame, o.MidSide ? 1 : 0, first);
                await pacer.LaunchedAsync();
            }
        }
        long totalBytes;
        if (!vbr)
        {
            using var pe = a.Allocate1D<float>(units);
            for (int first = 0; first < units; first += UnitChunk)
            {
                _pe(Math.Min(UnitChunk, units - first), xr.View, xmin.View, tables.View, pe.View, first);
                await pacer.LaunchedAsync();
            }
            _layout((int)frames + 1, frameOffsets.View, frameInfo.View, side.View, pe.View, frames, num, sampleRate, bitrateIndex, sideBytes,
                granulesPerFrame, channels, infoBytes);
            for (int first = 0; first < units; first += UnitChunk)
            {
                _quantize(Math.Min(UnitChunk, units - first), xr.View, ix.View, side.View, tables.View, xmin.View, work.View, lsf ? 1 : 0, 0, first);
                await pacer.LaunchedAsync();
            }
            totalBytes = infoBytes + frames * num / sampleRate;
        }
        else
        {
            for (int first = 0; first < units; first += UnitChunk)
            {
                _vbrQuantize(Math.Min(UnitChunk, units - first), xr.View, ix.View, side.View, tables.View, xmin.View, lsf ? 1 : 0, first);
                await pacer.LaunchedAsync();
            }
            _vbrLayout(1, side.View, frameOffsets.View, frameInfo.View, vmeta.View, bitrates.View, (int)frames, sampleRate, coeff,
                sideBytes, granulesPerFrame, channels, infoBytes);
            // frames over the highest rate: their granules re-quantized to its budget (rare)
            for (int first = 0; first < units; first += UnitChunk)
            {
                _quantize(Math.Min(UnitChunk, units - first), xr.View, ix.View, side.View, tables.View, xmin.View, work.View, lsf ? 1 : 0, 1, first);
                await pacer.LaunchedAsync();
            }
            await a.SynchronizeAsync();
            totalBytes = infoBytes + (await vmeta.View.SubView(K.VAudioBytes, 1).CopyToHostAsync())[0];
        }
        using var output = a.Allocate1D<byte>(Math.Max(1, totalBytes));
        for (int first = 0; first < units; first += UnitChunk)
        {
            _writeGranule(Math.Min(UnitChunk, units - first), ix.View, side.View, tables.View, slotBuf.View, lsf ? 1 : 0, first);
            await pacer.LaunchedAsync();
        }
        for (int first = 0; first < frames; first += UnitChunk)
        {
            _writeFrame((int)Math.Min(UnitChunk, frames - first), side.View, slotBuf.View, frameOffsets.View, frameInfo.View, output.View,
                ms.View, channels, lsf ? 1 : 0, version, rateIndex, first);
            await pacer.LaunchedAsync();
        }
        if (o.WriteInfoFrame)
        {
            // CRC-16 of the audio frames in bounded launches, then the Info frame (seek table, delay / padding, CRCs)
            using var crcState = a.Allocate1D<int>(1);
            crcState.MemSetToZero();
            const int CrcChunk = 1 << 20;
            for (long from = infoBytes; from < totalBytes; from += CrcChunk)
            {
                _musicCrc(1, output.View, _crcTable.View, crcState.View, from, (int)Math.Min(CrcChunk, totalBytes - from));
                await pacer.LaunchedAsync();
            }
            var name = new byte[9];
            System.Text.Encoding.ASCII.GetBytes(o.InfoEncoder ?? "").AsSpan(0, Math.Min(9, (o.InfoEncoder ?? "").Length)).CopyTo(name);
            using var nameView = a.Allocate1D<byte>(9);
            nameView.View.CopyFromCPU(name);
            // gapless: a decoder drops delay + 529 samples at the start (our latency is CodecDelay) and padding - 529 at the
            // end, so delay + samples + padding covers every frame (the convention LAME's own files follow)
            int delay = CodecDelay - 529;
            int padding = (int)(frames * frameSamples - samples - delay);
            var ip = new long[K.InfoParams];
            ip[K.InfoVersion] = version; ip[K.InfoRate] = rateIndex; ip[K.InfoBitrate] = infoIndex; ip[K.InfoChannels] = channels;
            ip[K.InfoSide] = sideBytes; ip[K.InfoBytes] = infoBytes; ip[K.InfoFrames] = frames; ip[K.InfoTotal] = totalBytes;
            ip[K.InfoVbr] = vbr ? 1 : 0; ip[K.InfoDelay] = delay; ip[K.InfoPadding] = padding;
            using var ipView = a.Allocate1D<long>(K.InfoParams);
            ipView.View.CopyFromCPU(ip);
            _infoFrame(1, output.View, frameOffsets.View, nameView.View, _crcTable.View, crcState.View, ipView.View);
        }
        await a.SynchronizeAsync();
        var audio = await output.View.SubView(0, totalBytes).CopyToHostAsync();
        if (!o.WriteId3) return audio;
        var id3 = Id3Tag(o);
        var file = new byte[id3.Length + audio.Length];
        id3.CopyTo(file, 0);
        audio.CopyTo(file, id3.Length);
        return file;
    }

    /// <summary>
    /// The ID3v2.4 tag (metadata, not codec data): header (version 4.0, flags 0, syncsafe size), a TXXX frame per tag
    /// (UTF-8, description NUL value NUL), TSSE (UTF-8, text NUL), 10 bytes of padding.
    /// </summary>
    private static byte[] Id3Tag(Mp3EncodeOptions o)
    {
        var body = new MemoryStream();
        void Frame(string id, byte[] content)
        {
            body.Write(System.Text.Encoding.ASCII.GetBytes(id));
            body.Write(Syncsafe(content.Length));
            body.WriteByte(0);
            body.WriteByte(0);
            body.Write(content);
        }
        var utf8 = System.Text.Encoding.UTF8;
        foreach (var kv in o.Tags ?? Array.Empty<KeyValuePair<string, string>>())
        {
            var c = new MemoryStream();
            c.WriteByte(3);
            c.Write(utf8.GetBytes(kv.Key));
            c.WriteByte(0);
            c.Write(utf8.GetBytes(kv.Value));
            c.WriteByte(0);
            Frame("TXXX", c.ToArray());
        }
        {
            var c = new MemoryStream();
            c.WriteByte(3);
            c.Write(utf8.GetBytes(o.Id3Encoder ?? ""));
            c.WriteByte(0);
            Frame("TSSE", c.ToArray());
        }
        body.Write(new byte[10]);
        var tag = new MemoryStream();
        tag.Write("ID3"u8);
        tag.WriteByte(4);
        tag.WriteByte(0);
        tag.WriteByte(0);
        tag.Write(Syncsafe((int)body.Length));
        body.Position = 0;
        body.CopyTo(tag);
        return tag.ToArray();
    }

    private static byte[] Syncsafe(int v) => new[] { (byte)((v >> 21) & 0x7F), (byte)((v >> 14) & 0x7F), (byte)((v >> 7) & 0x7F), (byte)(v & 0x7F) };

    /// <summary>
    /// The psychoacoustic model's per-band constants for a rate: the Bark of each long band's centre (Zwicker:
    /// 13 atan(0.00076 f) + 3.5 atan((f / 7500)^2)) and the threshold of hearing (Terhardt: 3.64 f^-0.8 - 6.5
    /// e^(-0.6 (f - 3.3)^2) + 0.001 f^4 dB SPL, f in kHz; the lowest over the band's lines) as an MDCT energy, with a
    /// full-scale sine (MDCT energy about 1 per granule) at 96 dB SPL.
    /// </summary>
    private static float[] PsyTables(int sampleRate)
    {
        var t = new float[K.PsyConsts];
        int r = Mp3Tables.BandRateIndex(sampleRate);
        int lo = 0;
        for (int b = 0; b < 22; b++)
        {
            int hi = lo + Mp3Tables.LongBandWidths[r * 22 + b];
            double fc = (lo + hi) / 2.0 * sampleRate / 1152.0;
            t[K.PBark + b] = (float)(13 * Math.Atan(0.00076 * fc) + 3.5 * Math.Atan(Math.Pow(fc / 7500, 2)));
            double athMin = double.MaxValue;
            for (int i = lo; i < hi; i++)
            {
                double f = Math.Max((i + 0.5) * sampleRate / 1152.0, 20) / 1000.0;
                double ath = 3.64 * Math.Pow(f, -0.8) - 6.5 * Math.Exp(-0.6 * (f - 3.3) * (f - 3.3)) + 0.001 * Math.Pow(f, 4);
                athMin = Math.Min(athMin, ath);
            }
            t[K.PAth + b] = (float)Math.Pow(10, (athMin - 96) / 10);
            lo = hi;
        }
        return t;
    }

    private static int[] IntTables(int sampleRate)
    {
        var t = new int[K.IntConsts];
        for (int i = 0; i < 8192; i++)
        {
            t[K.IHCode + i] = (int)Mp3HuffmanTables.Codes[i];
            t[K.IHLen + i] = Mp3HuffmanTables.Lengths[i];
        }
        for (int i = 0; i < 32; i++)
        {
            t[K.ILinbits + i] = Mp3HuffmanTables.Linbits[i];
            t[K.IMax + i] = Mp3HuffmanTables.MaxValue[i];
            t[K.IC1Code + i] = (int)Mp3HuffmanTables.Count1Codes[i];
            t[K.IC1Len + i] = Mp3HuffmanTables.Count1Lengths[i];
        }
        int r = Mp3Tables.BandRateIndex(sampleRate);
        int b = 0;
        t[K.IBounds] = 0;
        for (int i = 0; i < 22; i++)
        {
            b += Mp3Tables.LongBandWidths[r * 22 + i];
            t[K.IBounds + i + 1] = b;
        }
        return t;
    }

    /// <summary>Release accelerator-bound resources.</summary>
    public void Dispose()
    {
        _consts.Dispose();
        _crcTable.Dispose();
    }
}
