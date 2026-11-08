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
    /// <summary>Constant bit rate in kbps: MPEG-1 (32 / 44.1 / 48 kHz) 32..320, MPEG-2 / 2.5 (8..24 kHz) 8..160.</summary>
    public int BitrateKbps { get; init; } = 128;
}

/// <summary>MP3 encoder on the accelerator (see the file header).</summary>
public sealed class Mp3EncoderGpu : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, long, int, long> _polyphase;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, long> _matrix;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int, long> _mdct;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, long> _alias;
    private readonly Action<Index1D, ArrayView<long>, ArrayView<int>, ArrayView<int>, long, long, int, int, int, int, int> _layout;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<int>, ArrayView<float>, ArrayView<float>, ArrayView<float>, float, int> _psy;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<float>, ArrayView<int>, int, int> _quantize;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<byte>, int, int> _writeGranule;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<byte>, ArrayView<long>, ArrayView<int>, ArrayView<byte>, int, int, int, int, int> _writeFrame;
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
        _layout = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<long>, ArrayView<int>, ArrayView<int>, long, long, int, int, int, int, int>(K.FrameLayoutKernel);
        _psy = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<int>, ArrayView<float>, ArrayView<float>, ArrayView<float>, float, int>(K.PsyKernel);
        _quantize = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<float>, ArrayView<int>, int, int>(K.QuantizeKernel);
        _writeGranule = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<byte>, int, int>(K.WriteGranuleKernel);
        _writeFrame = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<byte>, ArrayView<long>, ArrayView<int>, ArrayView<byte>, int, int, int, int, int>(K.WriteFrameKernel);
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
        int bitrateIndex = Array.IndexOf(Mp3Tables.BitratesKbps, o.BitrateKbps, lsf ? 16 : 0, 15) - (lsf ? 16 : 0);
        if (bitrateIndex < 1) throw new ArgumentOutOfRangeException(nameof(options), $"{o.BitrateKbps} kbps is not a Layer III bit rate at {sampleRate} Hz.");
        int granulesPerFrame = lsf ? 1 : 2;
        int frameSamples = 576 * granulesPerFrame;
        int sideBytes = lsf ? (channels == 1 ? 9 : 17) : (channels == 1 ? 17 : 32);
        // a sample reaches the decoder's output CodecDelay samples later (polyphase analysis + synthesis 481, MDCT
        // overlap 576): enough frames that the last input sample comes out
        long frames = (samples + CodecDelay + frameSamples - 1) / frameSamples;
        int granules = (int)(frames * granulesPerFrame);
        int slots = granules * 18;
        long num = (lsf ? 72000L : 144000L) * o.BitrateKbps;   // bytes per frame = num / rate
        long totalBytes = frames * num / sampleRate;
        int units = granules * channels;
        var a = _accelerator;

        using var tables = a.Allocate1D<int>(K.IntConsts);
        tables.View.CopyFromCPU(IntTables(sampleRate));
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
        using var output = a.Allocate1D<byte>(Math.Max(1, totalBytes));

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
        _layout((int)frames + 1, frameOffsets.View, frameInfo.View, side.View, frames, num, sampleRate, bitrateIndex, sideBytes,
            granulesPerFrame, channels);
        await Launch((long)channels * slots * 64, (n, f) => _polyphase(n, pcm, _consts.View, y.View, channels, samples, slots, f));
        await Launch((long)channels * slots * 32, (n, f) => _matrix(n, y.View, _consts.View, sub.View, f));
        await Launch((long)channels * granules * 32, (n, f) => _mdct(n, sub.View, _consts.View, xr.View, granules, slots, channels, f));
        await Launch((long)units * 31 * 8, (n, f) => _alias(n, xr.View, _consts.View, f));
        const int UnitChunk = 1024;
        for (int first = 0; first < units; first += UnitChunk)
        {
            int n = Math.Min(UnitChunk, units - first);
            _psy(n, xr.View, tables.View, psy.View, psyScratch.View, xmin.View, 1f, first);
            _quantize(n, xr.View, ix.View, side.View, tables.View, xmin.View, work.View, lsf ? 1 : 0, first);
            await pacer.LaunchedAsync();
            _writeGranule(n, ix.View, side.View, tables.View, slotBuf.View, lsf ? 1 : 0, first);
            await pacer.LaunchedAsync();
        }
        for (int first = 0; first < frames; first += UnitChunk)
        {
            _writeFrame((int)Math.Min(UnitChunk, frames - first), side.View, slotBuf.View, frameOffsets.View, frameInfo.View, output.View,
                channels, lsf ? 1 : 0, version, rateIndex, first);
            await pacer.LaunchedAsync();
        }
        await a.SynchronizeAsync();
        return await output.View.SubView(0, totalBytes).CopyToHostAsync();
    }

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
    public void Dispose() => _consts.Dispose();
}
