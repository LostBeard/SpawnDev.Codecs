// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// FLAC stream encoder on the accelerator (RFC 9639): any sample rate FLAC can carry, 1..8 channels, 4..24 bits per
// sample, any length, compression levels 0..8. Written from the RFC and the textbook linear prediction / Rice coding
// literature; the kernels are in FlacEncodeKernels (see its header for the stages).
//
// Levels: 0..2 code each subframe with the best fixed polynomial predictor (orders up to 3, 4, 4) in blocks of about
// 27 ms; 3..8 add LPC (windowed autocorrelation, Levinson-Durbin, 15-bit coefficients; orders up to 6, 8, 8, 10, 12, 12)
// in blocks of about 105 ms, with more Rice partition orders searched as the level rises. Every level picks per
// subframe the cheapest of constant / verbatim / fixed / LPC by bit count and, for stereo, the cheapest of independent /
// left-side / right-side / mid-side coding per frame. The block size is the largest of FLAC's header-coded sizes (192,
// 576 * 2^k, 256 * 2^k) not above that duration, at most 4608 up to 48 kHz and 16384 above (the streamable subset).
//
// The stream: "fLaC", STREAMINFO (block sizes, smallest / largest frame, rate, channels, bits, sample count, MD5 of the
// samples), VORBIS_COMMENT (vendor, the tags in order, then WAVEFORMATEXTENSIBLE_CHANNEL_MASK when the options give a
// channel mask other than FLAC's default layout for the channel count), PADDING (Options.Padding zero bytes; none when
// 0), then the frames (fixed block size, frame numbers).
//
// Host work (pure coordinator): validate, build the metadata blocks (strings and stream parameters; the frame sizes and
// the MD5 are patched in by a kernel), upload the samples, dispatch the kernels chunk by chunk (bounded launches through
// LaunchPacer), read back the length (and the file for the managed / TypedArray overloads).

using System.Runtime.InteropServices;
using System.Runtime.Versioning;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.SpawnJS.JSObjects;
using K = SpawnDev.Codecs.Audio.Flac.FlacEncodeKernels;
using W = SpawnDev.Codecs.Audio.Flac.FlacMd5Kernels;

namespace SpawnDev.Codecs.Audio.Flac;

/// <summary>How <see cref="FlacEncoderGpu.EncodeAsync(ReadOnlyMemory{int}, int, int, int, FlacEncodeOptions?)"/> codes a stereo pair.</summary>
public enum FlacChannelMode
{
    /// <summary>Per frame, whichever of the four codings is smallest.</summary>
    Auto,
    /// <summary>Left and right coded as they are.</summary>
    Independent,
    /// <summary>Left and side (left - right).</summary>
    LeftSide,
    /// <summary>Side (left - right) and right.</summary>
    RightSide,
    /// <summary>Mid ((left + right) >> 1) and side (left - right).</summary>
    MidSide,
}

/// <summary>Settings of the FLAC stream encoder (<see cref="FlacEncoderGpu"/>.EncodeAsync).</summary>
public sealed record FlacEncodeOptions
{
    /// <summary>0 (fastest) .. 8 (smallest). See the file header of FlacEncoderGpu.Encode.cs.</summary>
    public int CompressionLevel { get; init; } = 5;
    /// <summary>Samples per channel per frame (16..65535); null: the level's duration at the sample rate.</summary>
    public int? BlockSize { get; init; }
    /// <summary>Stereo decorrelation (two channels only).</summary>
    public FlacChannelMode ChannelMode { get; init; } = FlacChannelMode.Auto;
    /// <summary>Bytes of the PADDING block (0: no PADDING block).</summary>
    public int Padding { get; init; } = 8192;
    /// <summary>The VORBIS_COMMENT vendor string.</summary>
    public string Vendor { get; init; } = "ffmpeg";
    /// <summary>VORBIS_COMMENT fields, written in order as "KEY=value".</summary>
    public IReadOnlyList<KeyValuePair<string, string>>? Tags { get; init; }
    /// <summary>
    /// WAVEFORMATEXTENSIBLE channel mask of the input. When it differs from FLAC's default layout for the channel count
    /// it is recorded as the WAVEFORMATEXTENSIBLE_CHANNEL_MASK tag (after <see cref="Tags"/>).
    /// </summary>
    public int? ChannelMask { get; init; }
}

public sealed partial class FlacEncoderGpu
{
    private StreamKernels? _stream;

    /// <summary>The stream encoder's kernels and tables, loaded on first use.</summary>
    private sealed class StreamKernels : IDisposable
    {
        public Action<Index1D, ArrayView<float>, ArrayView<int>> Window = null!;
        public Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<long>, int> Prep = null!;
        public Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<float>, ArrayView<double>, int> Autocorr = null!;
        public Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<double>, ArrayView<double>, ArrayView<int>> Lpc = null!;
        public Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<long>, int> PartSum = null!;
        public Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<long>, ArrayView<long>, ArrayView<int>, int> PredCost = null!;
        public Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<long>, ArrayView<int>, ArrayView<long>, int> Subframe = null!;
        public Action<Index1D, ArrayView<int>, ArrayView<long>, ArrayView<int>, int> FrameSize = null!;
        public Action<Index1D, ArrayView<int>, ArrayView<long>, ArrayView<long>, int> Offsets = null!;
        public Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<long>, ArrayView<int>, ArrayView<byte>, int> WriteFrame = null!;
        public Action<Index1D, ArrayView<long>, ArrayView<uint>, ArrayView<byte>, int> Patch = null!;
        public Action<Index1D, ArrayView<byte>, ArrayView<byte>, long> Copy = null!;
        public Action<Index1D, ArrayView<int>, ArrayView<uint>, ArrayView<uint>, int, long, long, int> Md5Blocks = null!;
        public Action<Index1D, ArrayView<int>, ArrayView<uint>, ArrayView<uint>, int, long> Md5Final = null!;
        public readonly MemoryBuffer1D<int, Stride1D.Dense> Crc;
        public readonly MemoryBuffer1D<uint, Stride1D.Dense> Md5Tables;

        /// <summary>Tables only; <see cref="LoadAsync"/> loads the kernels.</summary>
        public StreamKernels(Accelerator a)
        {
            Crc = a.Allocate1D<int>(512);
            Crc.View.CopyFromCPU(K.BuildCrcTables());
            Md5Tables = a.Allocate1D<uint>(128);
            Md5Tables.View.CopyFromCPU(W.BuildTables());
        }

        /// <summary>Load every kernel (awaited loads: each compile yields to the browser's event loop first).</summary>
        public async Task LoadAsync(Accelerator a)
        {
            Window = await a.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<float>, ArrayView<int>>(K.WindowKernel);
            Prep = await a.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<long>, int>(K.PrepKernel);
            Autocorr = await a.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<float>, ArrayView<double>, int>(K.AutocorrKernel);
            Lpc = await a.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<double>, ArrayView<double>, ArrayView<int>>(K.LpcKernel);
            PartSum = await a.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<long>, int>(K.PartSumKernel);
            PredCost = await a.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<long>, ArrayView<long>, ArrayView<int>, int>(K.PredCostKernel);
            Subframe = await a.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<long>, ArrayView<int>, ArrayView<long>, int>(K.SubframeKernel);
            FrameSize = await a.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<int>, ArrayView<long>, ArrayView<int>, int>(K.FrameSizeKernel);
            Offsets = await a.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<int>, ArrayView<long>, ArrayView<long>, int>(K.OffsetsKernel);
            WriteFrame = await a.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<long>, ArrayView<int>, ArrayView<byte>, int>(K.WriteFrameKernel);
            Patch = await a.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<long>, ArrayView<uint>, ArrayView<byte>, int>(K.PatchStreamInfoKernel);
            Copy = await a.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<byte>, ArrayView<byte>, long>(K.CopyBytesKernel);
            Md5Blocks = await a.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<int>, ArrayView<uint>, ArrayView<uint>, int, long, long, int>(W.Md5BlocksKernel);
            Md5Final = await a.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<int>, ArrayView<uint>, ArrayView<uint>, int, long>(W.Md5FinalKernel);
        }

        public void Dispose()
        {
            Crc.Dispose();
            Md5Tables.Dispose();
        }
    }

    private Task<StreamKernels>? _streamLoad;

    /// <summary>The stream encoder's kernels, loaded once on first use (awaited, so a browser page keeps running).</summary>
    private Task<StreamKernels> KernelsAsync()
    {
        if (_streamLoad == null || _streamLoad.IsFaulted || _streamLoad.IsCanceled) _streamLoad = LoadStreamKernelsAsync();
        return _streamLoad;
    }

    private async Task<StreamKernels> LoadStreamKernelsAsync()
    {
        var k = new StreamKernels(_accelerator);
        try { await k.LoadAsync(_accelerator); }
        catch { k.Dispose(); throw; }
        return _stream = k;
    }

    /// <summary>Per level: block duration (ms), highest fixed order, highest LPC order (0: none), highest Rice partition order.</summary>
    private static readonly (int Ms, int Fixed, int Lpc, int Po)[] Levels =
    {
        (27, 3, 0, 4), (27, 4, 0, 4), (27, 4, 0, 6),
        (105, 4, 6, 6), (105, 4, 8, 6), (105, 4, 8, 8), (105, 4, 10, 8), (105, 4, 12, 8), (105, 4, 12, 8),
    };

    private static readonly int[] HeaderBlockSizes = { 192, 256, 512, 576, 1024, 1152, 2048, 2304, 4096, 4608, 8192, 16384, 32768 };

    /// <summary>FLAC's default channel layout per channel count (WAVEFORMATEXTENSIBLE masks), index = channels.</summary>
    private static readonly int[] DefaultChannelMasks = { 0, 0x4, 0x3, 0x7, 0x33, 0x607, 0x60F, 0x70F, 0x63F };

    /// <summary>The block size a level uses at a sample rate (see the file header).</summary>
    public static int LevelBlockSize(int compressionLevel, int sampleRate)
    {
        if (compressionLevel is < 0 or > 8) throw new ArgumentOutOfRangeException(nameof(compressionLevel), "FLAC compression levels are 0..8.");
        long target = (long)sampleRate * Levels[compressionLevel].Ms / 1000;
        int cap = sampleRate <= 48000 ? 4608 : 16384;
        int best = 192;
        foreach (int b in HeaderBlockSizes)
            if (b <= target && b <= cap && b > best) best = b;
        return best;
    }

    private static int RateCode(int rate) => rate switch
    {
        88200 => 1, 176400 => 2, 192000 => 3, 8000 => 4, 16000 => 5, 22050 => 6, 24000 => 7, 32000 => 8, 44100 => 9,
        48000 => 10, 96000 => 11,
        _ when rate % 1000 == 0 && rate / 1000 <= 255 => 12,
        _ when rate <= 65535 => 13,
        _ when rate % 10 == 0 && rate / 10 <= 65535 => 14,
        _ => 0,
    };

    private static int BpsCode(int bps) => bps switch { 8 => 1, 12 => 2, 16 => 4, 20 => 5, 24 => 6, _ => 0 };

    /// <summary>The validated encode parameters and the metadata blocks.</summary>
    private sealed record Plan(int Channels, int Rate, int Bps, long Samples, int Block, int Frames, int LastN, int Level,
        int Stereo, int Cands, byte[] Header, long Bound);

    private static Plan MakePlan(long interleavedLength, int channels, int sampleRate, int bitsPerSample, FlacEncodeOptions? options)
    {
        var o = options ?? new FlacEncodeOptions();
        if (channels is < 1 or > 8) throw new ArgumentOutOfRangeException(nameof(channels), "FLAC carries 1..8 channels.");
        if (sampleRate is < 1 or > 1048575) throw new ArgumentOutOfRangeException(nameof(sampleRate), "FLAC sample rates are 1..1048575 Hz.");
        if (bitsPerSample is < 4 or > 24) throw new ArgumentOutOfRangeException(nameof(bitsPerSample), "This encoder takes 4..24 bits per sample.");
        if (o.CompressionLevel is < 0 or > 8) throw new ArgumentOutOfRangeException(nameof(options), "FLAC compression levels are 0..8.");
        if (o.BlockSize is { } bsz && (bsz < 16 || bsz > 65535)) throw new ArgumentOutOfRangeException(nameof(options), "FLAC block sizes are 16..65535.");
        if (o.Padding is < 0 or > 0xFFFFFF) throw new ArgumentOutOfRangeException(nameof(options), "Padding is 0..16777215 bytes.");
        if (interleavedLength % channels != 0)
            throw new ArgumentException($"{interleavedLength} interleaved samples is not a multiple of {channels} channels.");
        long samples = interleavedLength / channels;
        if (samples >= 1L << 36) throw new ArgumentException("FLAC streams hold fewer than 2^36 samples per channel.");
        int block = o.BlockSize ?? LevelBlockSize(o.CompressionLevel, sampleRate);
        long frames64 = (samples + block - 1) / block;
        if (frames64 > int.MaxValue) throw new ArgumentException("Too many frames.");
        int frames = (int)frames64;
        int lastN = frames == 0 ? block : (int)(samples - (long)(frames - 1) * block);
        int stereo = -1;
        if (channels == 2)
            stereo = o.ChannelMode switch
            {
                FlacChannelMode.Auto => 0, FlacChannelMode.LeftSide => 1, FlacChannelMode.RightSide => 2, FlacChannelMode.MidSide => 3,
                _ => -1,
            };
        int cands = stereo >= 0 ? 4 : channels;

        // metadata blocks (RFC 9639 section 8)
        var h = new List<byte>(64 + o.Padding);
        h.AddRange("fLaC"u8.ToArray());
        bool hasPadding = o.Padding > 0;
        // STREAMINFO: frame sizes (bytes 12..17 of the file) and MD5 (26..41) are patched by PatchStreamInfoKernel
        h.Add(0x00); h.Add(0x00); h.Add(0x00); h.Add(34);
        h.Add((byte)(block >> 8)); h.Add((byte)block);
        h.Add((byte)(block >> 8)); h.Add((byte)block);
        for (int i = 0; i < 6; i++) h.Add(0);
        ulong packed = ((ulong)sampleRate << 44) | ((ulong)(channels - 1) << 41) | ((ulong)(bitsPerSample - 1) << 36) | (ulong)samples;
        for (int i = 7; i >= 0; i--) h.Add((byte)(packed >> (8 * i)));
        for (int i = 0; i < 16; i++) h.Add(0);
        // VORBIS_COMMENT
        var fields = new List<byte[]>();
        if (o.Tags != null)
            foreach (var t in o.Tags) fields.Add(System.Text.Encoding.UTF8.GetBytes(t.Key + "=" + t.Value));
        if (o.ChannelMask is { } mask && mask != DefaultChannelMasks[channels])
            fields.Add(System.Text.Encoding.UTF8.GetBytes("WAVEFORMATEXTENSIBLE_CHANNEL_MASK=0x" + mask.ToString("x")));
        var vendor = System.Text.Encoding.UTF8.GetBytes(o.Vendor ?? "");
        int vcLength = 4 + vendor.Length + 4;
        foreach (var fld in fields) vcLength += 4 + fld.Length;
        if (vcLength > 0xFFFFFF) throw new ArgumentException("The vendor string and tags exceed a metadata block (16 MiB).");
        h.Add((byte)(hasPadding ? 0x04 : 0x84));
        h.Add((byte)(vcLength >> 16)); h.Add((byte)(vcLength >> 8)); h.Add((byte)vcLength);
        void Le32(int v) { h.Add((byte)v); h.Add((byte)(v >> 8)); h.Add((byte)(v >> 16)); h.Add((byte)(v >> 24)); }
        Le32(vendor.Length);
        h.AddRange(vendor);
        Le32(fields.Count);
        foreach (var fld in fields) { Le32(fld.Length); h.AddRange(fld); }
        if (hasPadding)
        {
            h.Add(0x81);
            h.Add((byte)(o.Padding >> 16)); h.Add((byte)(o.Padding >> 8)); h.Add((byte)o.Padding);
            h.AddRange(new byte[o.Padding]);
        }
        var header = h.ToArray();

        // the frames can never be larger than this: every subframe is at most its verbatim coding
        long FrameBound(long n) => 16 + 2 + 1 + (channels * (8 + n * (bitsPerSample + 1)) + 7) / 8;
        long bound = header.Length;
        if (frames > 0) bound += (frames - 1) * FrameBound(block) + FrameBound(lastN);
        return new Plan(channels, sampleRate, bitsPerSample, samples, block, frames, lastN, o.CompressionLevel, stereo, cands, header, bound);
    }

    /// <summary>
    /// Compile and load every kernel the stream encoder uses (encodes a short built-in stereo signal at level 8), so the
    /// first real encode does not wait for them. On browser backends the first compile of the shaders can take tens of
    /// seconds; call this early.
    /// </summary>
    public async Task WarmUpAsync()
    {
        // a deterministic 2-channel test signal: two tones plus a little integer noise
        const int n = 6000;
        var pcm = new int[n * 2];
        uint seed = 12345;
        for (int i = 0; i < n; i++)
        {
            seed = seed * 1664525 + 1013904223;
            int noise = (int)(seed >> 26) - 32;
            pcm[2 * i] = (int)(8000 * Math.Sin(i * 0.0627)) + noise;
            pcm[2 * i + 1] = (int)(6000 * Math.Sin(i * 0.0411 + 1)) - noise;
        }
        await EncodeAsync(pcm, 2, 44100, 16, new FlacEncodeOptions { CompressionLevel = 8 });
    }

    /// <summary>
    /// Encode interleaved integer PCM (channel 0[0], channel 1[0], ...; each sample in the signed
    /// <paramref name="bitsPerSample"/>-bit range) to a complete FLAC file.
    /// </summary>
    /// <exception cref="ArgumentException">A sample is outside the bit range, or a parameter is invalid.</exception>
    public async Task<byte[]> EncodeAsync(ReadOnlyMemory<int> interleaved, int channels, int sampleRate, int bitsPerSample,
        FlacEncodeOptions? options = null)
    {
        var plan = MakePlan(interleaved.Length, channels, sampleRate, bitsPerSample, options);
        using var input = _accelerator.Allocate1D<int>(Math.Max(1, interleaved.Length));
        if (interleaved.Length > 0)
            input.View.SubView(0, interleaved.Length).CopyFromCPU(ref MemoryMarshal.GetReference(interleaved.Span), interleaved.Length);
        using var output = _accelerator.Allocate1D<byte>(plan.Bound);
        long length = await EncodeCoreAsync(input.View, plan, options ?? new FlacEncodeOptions(), output.View);
        return await DeviceTransfer.DownloadAsync(output.View, length);
    }

    /// <summary>
    /// Encode interleaved integer PCM held in a JS Int32Array to a FLAC file in a new Uint8Array (the samples and the file
    /// never enter the .NET heap); the caller disposes the result.
    /// </summary>
    [SupportedOSPlatform("browser")]
    public async Task<Uint8Array> EncodeAsync(Int32Array interleaved, int channels, int sampleRate, int bitsPerSample,
        FlacEncodeOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(interleaved);
        long count = interleaved.ByteLength / sizeof(int);
        var plan = MakePlan(count, channels, sampleRate, bitsPerSample, options);
        using var input = _accelerator.Allocate1D<int>(Math.Max(1, count));
        if (count > 0) DeviceTransfer.Upload(input.View, interleaved);
        using var output = _accelerator.Allocate1D<byte>(plan.Bound);
        long length = await EncodeCoreAsync(input.View, plan, options ?? new FlacEncodeOptions(), output.View);
        return await DeviceTransfer.DownloadUint8ArrayAsync(output.View, length);
    }

    /// <summary>
    /// Encode interleaved integer PCM already on the accelerator into <paramref name="output"/> (the file starts at its
    /// first byte; bytes past the file are not touched). Returns the file's length.
    /// </summary>
    /// <exception cref="ArgumentException"><paramref name="output"/> is smaller than the file (the message gives the size), a
    /// sample is outside the bit range, or a parameter is invalid.</exception>
    public async Task<int> EncodeAsync(ArrayView1D<int, Stride1D.Dense> interleaved, int channels, int sampleRate, int bitsPerSample,
        ArrayView1D<byte, Stride1D.Dense> output, FlacEncodeOptions? options = null)
    {
        var plan = MakePlan(interleaved.Length, channels, sampleRate, bitsPerSample, options);
        var o = options ?? new FlacEncodeOptions();
        if (output.Length >= plan.Bound)
            return checked((int)await EncodeCoreAsync(interleaved, plan, o, output));
        // the file's size is known only once the frames are coded: code into a buffer that always fits, then copy
        using var work = _accelerator.Allocate1D<byte>(plan.Bound);
        long length = await EncodeCoreAsync(interleaved, plan, o, work.View);
        if (length > output.Length)
            throw new ArgumentException($"The FLAC file is {length} bytes; the output view holds {output.Length}.", nameof(output));
        var k = await KernelsAsync();
        var pacer = new LaunchPacer(_accelerator);
        const int CopyChunk = 1 << 22;
        for (long first = 0; first < length; first += CopyChunk)
        {
            k.Copy((int)Math.Min(CopyChunk, length - first), work.View, output, first);
            await pacer.LaunchedAsync();
        }
        await _accelerator.SynchronizeAsync();
        return (int)length;
    }

    /// <summary>Encode into <paramref name="target"/> (at least plan.Bound bytes); returns the file length.</summary>
    private async Task<long> EncodeCoreAsync(ArrayView1D<int, Stride1D.Dense> input, Plan plan, FlacEncodeOptions o,
        ArrayView1D<byte, Stride1D.Dense> target)
    {
        var a = _accelerator;
        var k = await KernelsAsync();
        var lv = Levels[plan.Level];
        int maxLpc = lv.Lpc;
        int preds = 5 + maxLpc;
        int cands = plan.Cands;
        int fc = Math.Clamp((1 << 17) / plan.Block, 1, 1024);
        fc = Math.Max(1, Math.Min(fc, plan.Frames));

        var p = new int[K.PrmInts];
        p[K.PChannels] = plan.Channels; p[K.PBps] = plan.Bps; p[K.PCands] = cands; p[K.PBlock] = plan.Block;
        p[K.PLastN] = plan.LastN; p[K.PFrames] = plan.Frames; p[K.PMaxFixed] = lv.Fixed; p[K.PMaxLpc] = maxLpc;
        p[K.PMaxPo] = lv.Po; p[K.PPrecision] = 15; p[K.PRateCode] = RateCode(plan.Rate); p[K.PRate] = plan.Rate;
        p[K.PBpsCode] = BpsCode(plan.Bps); p[K.PStereo] = plan.Stereo; p[K.PHeaderLen] = plan.Header.Length; p[K.PPreds] = preds;

        using var prm = a.Allocate1D<int>(K.PrmInts);
        prm.View.CopyFromCPU(p);
        using var meta = a.Allocate1D<long>(K.MetaLongs);
        meta.View.CopyFromCPU(new long[] { 0, long.MaxValue, 0, 0 });
        using var sub = a.Allocate1D<int>((long)fc * cands * K.SubInts);
        using var acf = a.Allocate1D<double>((long)fc * cands * K.Lags);
        using var scratch = a.Allocate1D<double>((long)fc * cands * 2 * K.Lags);
        using var lpcQ = a.Allocate1D<int>((long)fc * cands * K.LpcInts);
        using var sums = a.Allocate1D<long>((long)fc * cands * preds * K.MaxParts);
        using var cost = a.Allocate1D<long>((long)fc * cands * preds);
        using var costPo = a.Allocate1D<int>((long)fc * cands * preds);
        using var subBits = a.Allocate1D<long>((long)fc * cands);
        using var frameInfo = a.Allocate1D<int>((long)fc * K.FrameInts);
        using var offsets = a.Allocate1D<long>(fc);
        using var win = a.Allocate1D<float>(maxLpc > 0 ? plan.Block + plan.LastN : 1);
        using var md5 = a.Allocate1D<uint>(W.Md5Size);
        md5.View.SubView(0, 4).CopyFromCPU(new uint[] { 0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476 });

        DeviceTransfer.Upload(target, plan.Header);
        // an empty input still needs a bindable view (no kernel reads it: no frames, MD5 of zero bytes)
        using var empty = input.Length == 0 ? a.Allocate1D<int>(1) : null;
        var inView = empty != null ? empty.View : input;
        var pacer = new LaunchPacer(a);
        if (maxLpc > 0 && plan.Frames > 0)
        {
            k.Window(plan.Block + plan.LastN, win.View, prm.View);
            await pacer.LaunchedAsync();
        }
        for (int first = 0; first < plan.Frames; first += fc)
        {
            int count = Math.Min(fc, plan.Frames - first);
            int subs = count * cands;
            k.Prep(subs, inView, prm.View, sub.View, meta.View, first);
            await pacer.LaunchedAsync();
            if (maxLpc > 0)
            {
                k.Autocorr(subs * K.Lags, inView, prm.View, sub.View, win.View, acf.View, first);
                await pacer.LaunchedAsync();
                k.Lpc(subs, prm.View, sub.View, acf.View, scratch.View, lpcQ.View);
                await pacer.LaunchedAsync();
            }
            // without LPC (levels 0..2) preds is 5: only the fixed predictors, which never read lpcQ
            k.PartSum(subs * preds * K.MaxParts, inView, prm.View, sub.View, lpcQ.View, sums.View, first);
            await pacer.LaunchedAsync();
            k.PredCost(subs * preds, prm.View, sub.View, sums.View, cost.View, costPo.View, first);
            await pacer.LaunchedAsync();
            k.Subframe(subs, inView, prm.View, sub.View, lpcQ.View, cost.View, costPo.View, subBits.View, first);
            await pacer.LaunchedAsync();
            k.FrameSize(count, prm.View, subBits.View, frameInfo.View, first);
            await pacer.LaunchedAsync();
            k.Offsets(1, frameInfo.View, offsets.View, meta.View, count);
            await pacer.LaunchedAsync();
            k.WriteFrame(count, inView, prm.View, sub.View, frameInfo.View, offsets.View, k.Crc.View, target, first);
            await pacer.LaunchedAsync();
        }

        // MD5 of the samples (little-endian, ceil(bits / 8) bytes each, interleaved)
        int bytesPerSample = (plan.Bps + 7) / 8;
        long totalBytes = plan.Samples * plan.Channels * bytesPerSample;
        long blocks = totalBytes / 64;
        for (long b = 0; b < blocks; b += W.Md5BlocksPerLaunch)
        {
            k.Md5Blocks(1, inView, md5.View, k.Md5Tables.View, bytesPerSample, totalBytes, b, (int)Math.Min(W.Md5BlocksPerLaunch, blocks - b));
            await pacer.LaunchedAsync();
        }
        k.Md5Final(1, inView, md5.View, k.Md5Tables.View, bytesPerSample, totalBytes);
        k.Patch(1, meta.View, md5.View, target, plan.Frames);
        await a.SynchronizeAsync();
        var m = await meta.CopyToHostAsync();
        if (m[K.MRangeError] != 0)
            throw new ArgumentException($"A sample is outside the signed {plan.Bps}-bit range.");
        return plan.Header.Length + m[K.MRun];
    }
}
