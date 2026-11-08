// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// FlacEncoderGpu.EncodeAsync: a complete .flac file of any length, sample rate and channel count (1..8) at 16 or 24 bits,
// the way FFmpeg 8.1 writes one (libavcodec/flacenc.c at a compression level, then libavformat's FLAC muxer:
// "fLaC", STREAMINFO with the MD5, VORBIS_COMMENT, PADDING) - what PyAV / ComfyUI's Save Audio produce. See
// FlacStreamKernels.cs for the kernels and for where the LPC analysis differs from FFmpeg (integer, so every backend
// writes the same bytes; levels 0..2 use no LPC and match FFmpeg's decisions exactly).
//
// The host coordinates: it uploads the samples (or takes them on the accelerator), dispatches the stages, reads back
// the file size and, for the managed / TypedArray overloads, the file. The Vorbis comment payload (vendor string and
// the caller's tags) is metadata the host serializes and uploads, like the EXIF blob of the WebP encoders.
//
// The older EncodeStreamAsync (verbatim-only, 4096-sample multiples, 44.1 kHz / 16-bit) is unchanged.

using System.Runtime.Versioning;
using System.Text;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.SpawnJS.JSObjects;
using K = SpawnDev.Codecs.Audio.Flac.FlacStreamKernels;
using W = SpawnDev.Codecs.Audio.Flac.FlacStreamWriteKernels;

namespace SpawnDev.Codecs.Audio.Flac;

/// <summary>Settings of <see cref="FlacEncoderGpu.EncodeAsync(ReadOnlyMemory{int}, int, int, int, FlacEncodeOptions?)"/>.</summary>
public sealed record FlacEncodeOptions
{
    /// <summary>FFmpeg compression level 0..8 (default 5, FFmpeg's and PyAV's): block time, predictor kind and orders, partitions.</summary>
    public int CompressionLevel { get; init; } = 5;

    /// <summary>Block size in samples (16..65535; null = FFmpeg's choice for the rate and level).</summary>
    public int? BlockSize { get; init; }

    /// <summary>Stereo decorrelation: null = estimated per frame (FFmpeg's default), 0 independent, 1 left/side, 2 right/side, 3 mid/side.</summary>
    public int? ChannelMode { get; init; }

    /// <summary>PADDING block size in bytes (FFmpeg's default 8192; 0 = no padding block).</summary>
    public int Padding { get; init; } = 8192;

    /// <summary>Vorbis comment vendor string (FFmpeg: "ffmpeg" in bit-exact mode, else "Lavf62.x.100").</summary>
    public string Vendor { get; init; } = "ffmpeg";

    /// <summary>Vorbis comment fields, in order, written as "KEY=value" (null = none).</summary>
    public IReadOnlyList<KeyValuePair<string, string>>? Tags { get; init; }

    /// <summary>
    /// The channel layout as a WAVEFORMATEXTENSIBLE mask (null = FLAC's default order). As FFmpeg's muxer, a layout FLAC
    /// does not carry natively (anything but mono, stereo, 3.0, quad, 5.0, 5.1, 6.1, 7.1) is written as the comment
    /// WAVEFORMATEXTENSIBLE_CHANNEL_MASK=0x... after the tags (FFmpeg's 5- and 6-channel defaults are 0x37 and 0x3f).
    /// </summary>
    public ulong? ChannelMask { get; init; }
}

public sealed partial class FlacEncoderGpu
{
    private StreamState? _stream;

    // FFmpeg flac_encode_init's per-level defaults, levels 0..8
    private static readonly int[] LevelBlockTime = { 27, 27, 27, 105, 105, 105, 105, 105, 105 };
    private static readonly int[] LevelLpcType = { K.LpcFixed, K.LpcFixed, K.LpcFixed, K.LpcLevinson, K.LpcLevinson, K.LpcLevinson, K.LpcLevinson, K.LpcLevinson, K.LpcLevinson };
    private static readonly int[] LevelMinOrder = { 2, 0, 0, 1, 1, 1, 1, 1, 1 };
    private static readonly int[] LevelMaxOrder = { 3, 4, 4, 6, 8, 8, 8, 8, 12 };
    private static readonly int[] LevelOrderMethod = { K.OrderEst, K.OrderEst, K.OrderEst, K.OrderEst, K.OrderEst, K.OrderEst, K.Order4Level, K.OrderLog, K.Order4Level };
    private static readonly int[] LevelMinPorder = { 2, 2, 0, 0, 0, 0, 0, 0, 0 };
    private static readonly int[] LevelMaxPorder = { 2, 2, 3, 3, 3, 8, 8, 8, 8 };
    private static readonly int[] RateTable = { 0, 88200, 176400, 192000, 8000, 16000, 22050, 24000, 32000, 44100, 48000, 96000 };
    private static readonly int[] BlockSizeTable = { 0, 192, 576, 1152, 2304, 4608, 0, 0, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768 };

    /// <summary>Kernels and buffers of EncodeAsync, created on its first use.</summary>
    private sealed class StreamState : IDisposable
    {
        public readonly Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, long> Deinterleave;
        public readonly Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>> StereoMode, Decorrelate;
        public readonly Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>> Wasted;
        public readonly Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<double>, ArrayView<int>> LpcAnalysis;
        public readonly Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<long>, ArrayView<int>, ArrayView<int>, int> Choose;
        public readonly Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>> FrameBits;
        public readonly Action<Index1D, ArrayView<int>, ArrayView<int>> Offsets;
        public readonly Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<byte>, ArrayView<int>> WriteFrame;
        public readonly Action<Index1D, ArrayView<int>, ArrayView<uint>, ArrayView<uint>, int, long, long, int> Md5Blocks;
        public readonly Action<Index1D, ArrayView<int>, ArrayView<uint>, ArrayView<uint>, int, long> Md5Final;
        public readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, int, int, ArrayView<int>, ArrayView<uint>> Header;

        public readonly StageBuffer<int> Input, X, R, Frames, Subs, Lpc, Parms;
        public readonly StageBuffer<long> Sums;
        public readonly StageBuffer<double> LpcD;
        public readonly StageBuffer<byte> Comment, Output;
        public readonly MemoryBuffer1D<int, Stride1D.Dense> Meta;
        public readonly MemoryBuffer1D<uint, Stride1D.Dense> Md5, Md5Tables;

        public StreamState(Accelerator a)
        {
            Deinterleave = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, long>(K.DeinterleaveKernel);
            StereoMode = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>>(K.StereoModeKernel);
            Decorrelate = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>>(K.DecorrelateKernel);
            Wasted = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>>(K.WastedBitsKernel);
            LpcAnalysis = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<double>, ArrayView<int>>(K.LpcAnalysisKernel);
            Choose = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<long>, ArrayView<int>, ArrayView<int>, int>(K.ChooseKernel);
            FrameBits = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>>(W.FrameBitsKernel);
            Offsets = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>>(W.OffsetsKernel);
            WriteFrame = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<byte>, ArrayView<int>>(W.WriteFrameKernel);
            Md5Blocks = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<uint>, ArrayView<uint>, int, long, long, int>(W.Md5BlocksKernel);
            Md5Final = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<uint>, ArrayView<uint>, int, long>(W.Md5FinalKernel);
            Header = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<byte>, int, int, ArrayView<int>, ArrayView<uint>>(W.HeaderKernel);
            Input = new StageBuffer<int>(a); X = new StageBuffer<int>(a); R = new StageBuffer<int>(a); Frames = new StageBuffer<int>(a);
            Subs = new StageBuffer<int>(a); Lpc = new StageBuffer<int>(a); Parms = new StageBuffer<int>(a); Sums = new StageBuffer<long>(a); LpcD = new StageBuffer<double>(a);
            Comment = new StageBuffer<byte>(a); Output = new StageBuffer<byte>(a);
            Meta = a.Allocate1D<int>(K.MSize);
            Md5 = a.Allocate1D<uint>(W.Md5Size);
            Md5Tables = a.Allocate1D<uint>(128);
            Md5Tables.View.CopyFromCPU(BuildMd5Tables());
        }

        private static uint[] BuildMd5Tables()
        {
            var t = new uint[128];
            for (int i = 0; i < 64; i++) t[i] = (uint)(long)Math.Floor(Math.Abs(Math.Sin(i + 1)) * 4294967296.0);
            int[] s = { 7, 12, 17, 22, 5, 9, 14, 20, 4, 11, 16, 23, 6, 10, 15, 21 };
            for (int i = 0; i < 64; i++) t[64 + i] = (uint)s[(i / 16) * 4 + (i & 3)];
            return t;
        }

        public void Dispose()
        {
            Input.Dispose(); X.Dispose(); R.Dispose(); Frames.Dispose(); Subs.Dispose(); Lpc.Dispose(); Parms.Dispose(); Sums.Dispose(); LpcD.Dispose();
            Comment.Dispose(); Output.Dispose(); Meta.Dispose(); Md5.Dispose(); Md5Tables.Dispose();
        }
    }

    /// <summary>select_blocksize: the largest table block size not longer than the level's block time (at least 192).</summary>
    private static int SelectBlockSize(int sampleRate, int blockTimeMs)
    {
        int blockSize = BlockSizeTable[1];
        int target = (int)((long)sampleRate * blockTimeMs / 1000);
        for (int i = 0; i < 16; i++)
            if (target >= BlockSizeTable[i] && BlockSizeTable[i] > blockSize) blockSize = BlockSizeTable[i];
        return blockSize;
    }

    /// <summary>The sample rate's frame-header code and its explicit value (flac_encode_init).</summary>
    private static (int Code, int Value) RateCode(int rate)
    {
        for (int i = 1; i < 12; i++) if (RateTable[i] == rate) return (i, 0);
        if (rate % 1000 == 0 && rate < 255000) return (12, rate / 1000);
        if (rate % 10 == 0 && rate < 655350) return (14, rate / 10);
        if (rate < 65535) return (13, rate);
        if (rate < 1048576) return (0, 0);
        throw new ArgumentOutOfRangeException(nameof(rate), $"{rate} Hz is not supported by FLAC.");
    }

    private static int MaxFrameSize(int n, int channels, int bps)
    {
        int count = 16 + channels * ((7 + bps + 7) / 8);
        if (channels == 2) count += ((2 * bps + 1) * n + 7) / 8;
        else count += (channels * bps * n + 7) / 8;
        return count + 2;
    }

    /// <summary>ff_vorbiscomment_write: vendor length + vendor, field count, each field's length + "KEY=value" (little-endian).</summary>
    private static byte[] VorbisComment(FlacEncodeOptions o)
    {
        var ms = new MemoryStream();
        void Le32(int v) { ms.WriteByte((byte)v); ms.WriteByte((byte)(v >> 8)); ms.WriteByte((byte)(v >> 16)); ms.WriteByte((byte)(v >> 24)); }
        var vendor = Encoding.UTF8.GetBytes(o.Vendor ?? "");
        Le32(vendor.Length);
        ms.Write(vendor);
        var tags = new List<KeyValuePair<string, string>>(o.Tags ?? System.Array.Empty<KeyValuePair<string, string>>());
        ulong[] native = { 0x4, 0x3, 0x7, 0x33, 0x607, 0x60F, 0x70F, 0x63F };   // ff_flac_is_native_layout
        if (o.ChannelMask is ulong mask && System.Array.IndexOf(native, mask) < 0)
            tags.Add(new KeyValuePair<string, string>("WAVEFORMATEXTENSIBLE_CHANNEL_MASK", "0x" + mask.ToString("x")));
        Le32(tags.Count);
        foreach (var kv in tags)
        {
            var field = Encoding.UTF8.GetBytes(kv.Key + "=" + kv.Value);
            Le32(field.Length);
            ms.Write(field);
        }
        return ms.ToArray();
    }

    private static void ValidateStream(long samplesPerChannel, int channels, int sampleRate, int bitsPerSample, FlacEncodeOptions o)
    {
        if (channels < 1 || channels > 8) throw new ArgumentOutOfRangeException(nameof(channels), "FLAC carries 1..8 channels.");
        if (bitsPerSample != 16 && bitsPerSample != 24)
            throw new NotSupportedException("FLAC encode: 16 or 24 bits per sample (FFmpeg's sample formats s16 / s32 as 24-bit).");
        if (sampleRate < 1) throw new ArgumentOutOfRangeException(nameof(sampleRate));
        RateCode(sampleRate);
        if (o.CompressionLevel < 0 || o.CompressionLevel > 8) throw new NotSupportedException("Compression levels 0..8 are supported.");
        if (o.BlockSize is int bs && (bs < 16 || bs > 65535)) throw new ArgumentOutOfRangeException(nameof(o), "BlockSize must be 16..65535.");
        if (o.ChannelMode is int cm && (cm < 0 || cm > 3 || channels != 2)) throw new ArgumentOutOfRangeException(nameof(o), "ChannelMode 0..3 needs 2 channels.");
        if (o.Padding < 0 || o.Padding >= (1 << 24)) throw new ArgumentOutOfRangeException(nameof(o), "Padding must be 0..2^24 - 1.");
        if (samplesPerChannel < 1) throw new ArgumentException("No samples to encode.");
        if (samplesPerChannel >= (1L << 36)) throw new ArgumentException("Too many samples for FLAC's 36-bit sample count.");
    }

    /// <summary>
    /// Encode interleaved PCM (channel 0[0], channel 1[0], ..., each a signed <paramref name="bitsPerSample"/>-bit value in
    /// an int) into a complete .flac file, as FFmpeg's FLAC encoder and muxer write it. Any length, any rate FLAC can
    /// carry, 1..8 channels, 16 or 24 bits. Returns the file (a terminal CPU sink).
    /// </summary>
    public async Task<byte[]> EncodeAsync(ReadOnlyMemory<int> interleavedSamples, int channels, int sampleRate, int bitsPerSample,
        FlacEncodeOptions? options = null)
    {
        var o = options ?? new FlacEncodeOptions();
        if (channels < 1 || interleavedSamples.Length % channels != 0)
            throw new ArgumentException("The sample count must be a multiple of the channel count.", nameof(interleavedSamples));
        long perChannel = interleavedSamples.Length / channels;
        ValidateStream(perChannel, channels, sampleRate, bitsPerSample, o);
        var s = _stream ??= new StreamState(_accelerator);
        var input = await s.Input.EnsureDrainedAsync(interleavedSamples.Length);
        input.SubView(0, interleavedSamples.Length).CopyFromCPU(
            ref System.Runtime.InteropServices.MemoryMarshal.GetReference(interleavedSamples.Span), interleavedSamples.Length);
        int length = await EncodeCoreAsync(s, input, perChannel, channels, sampleRate, bitsPerSample, o, need => s.Output.EnsureDrainedAsync(need));
        return await DeviceTransfer.DownloadAsync(s.Output.View, length);
    }

    /// <summary>Encode interleaved PCM held in a JS Int32Array (browser); the file comes back as a JS Uint8Array.</summary>
    [SupportedOSPlatform("browser")]
    public async Task<Uint8Array> EncodeAsync(Int32Array interleavedSamples, int channels, int sampleRate, int bitsPerSample,
        FlacEncodeOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(interleavedSamples);
        var o = options ?? new FlacEncodeOptions();
        long count = interleavedSamples.Length;
        if (channels < 1 || count % channels != 0)
            throw new ArgumentException("The sample count must be a multiple of the channel count.", nameof(interleavedSamples));
        long perChannel = count / channels;
        ValidateStream(perChannel, channels, sampleRate, bitsPerSample, o);
        var s = _stream ??= new StreamState(_accelerator);
        var input = await s.Input.EnsureDrainedAsync(count);
        DeviceTransfer.Upload(input.SubView(0, count), interleavedSamples);
        int length = await EncodeCoreAsync(s, input, perChannel, channels, sampleRate, bitsPerSample, o, need => s.Output.EnsureDrainedAsync(need));
        return await DeviceTransfer.DownloadUint8ArrayAsync(s.Output.View, length);
    }

    /// <summary>
    /// Encode interleaved PCM already on the accelerator into <paramref name="output"/>; returns the file length. Throws
    /// <see cref="ArgumentException"/> naming the size when <paramref name="output"/> is too small.
    /// </summary>
    public async Task<int> EncodeAsync(ArrayView1D<int, Stride1D.Dense> interleavedSamples, int channels, int sampleRate, int bitsPerSample,
        ArrayView1D<byte, Stride1D.Dense> output, FlacEncodeOptions? options = null)
    {
        var o = options ?? new FlacEncodeOptions();
        if (channels < 1 || interleavedSamples.Length % channels != 0)
            throw new ArgumentException("The sample count must be a multiple of the channel count.", nameof(interleavedSamples));
        long perChannel = interleavedSamples.Length / channels;
        ValidateStream(perChannel, channels, sampleRate, bitsPerSample, o);
        var s = _stream ??= new StreamState(_accelerator);
        return await EncodeCoreAsync(s, interleavedSamples, perChannel, channels, sampleRate, bitsPerSample, o,
            need => need <= output.Length ? new ValueTask<ArrayView1D<byte, Stride1D.Dense>>(output)
                : throw new ArgumentException($"output must hold {need} bytes.", nameof(output)));
    }

    private async Task<int> EncodeCoreAsync(StreamState s, ArrayView1D<int, Stride1D.Dense> input, long perChannel, int channels,
        int sampleRate, int bitsPerSample, FlacEncodeOptions o, Func<long, ValueTask<ArrayView1D<byte, Stride1D.Dense>>> ensureOutput)
    {
        int level = o.CompressionLevel;
        int blockSize = o.BlockSize ?? SelectBlockSize(sampleRate, LevelBlockTime[level]);
        long frames64 = (perChannel + blockSize - 1) / blockSize;
        if (frames64 * channels > int.MaxValue / K.SubStride) throw new ArgumentException("The stream is too long to encode in one call.");
        int frames = (int)frames64;
        int lastN = (int)(perChannel - (frames64 - 1) * blockSize);
        var (srCode, srValue) = RateCode(sampleRate);
        int lpcType = LevelLpcType[level];
        int minOrder = LevelMinOrder[level], maxOrder = LevelMaxOrder[level];
        if (lpcType == K.LpcFixed)
        {
            if (minOrder > 4) minOrder = 4;
            if (maxOrder > 4) maxOrder = 4;
        }
        var comment = VorbisComment(o);
        int framesStart = 4 + 38 + 4 + comment.Length + (o.Padding > 0 ? 4 + o.Padding : 0);

        var meta = new int[K.MSize];
        meta[K.MChannels] = channels; meta[K.MBlockSize] = blockSize; meta[K.MFrames] = frames;
        meta[K.MTotalLo] = (int)(perChannel & 0xFFFFFFFF); meta[K.MTotalHi] = (int)(perChannel >> 32);
        meta[K.MBps] = bitsPerSample; meta[K.MBpsCode] = bitsPerSample == 16 ? 4 : 6;
        meta[K.MSrCode0] = srCode; meta[K.MSrCode1] = srValue; meta[K.MRate] = sampleRate;
        meta[K.MMinOrder] = minOrder; meta[K.MMaxOrder] = maxOrder; meta[K.MLpcType] = lpcType;
        meta[K.MOrderMethod] = LevelOrderMethod[level]; meta[K.MMinPorder] = LevelMinPorder[level]; meta[K.MMaxPorder] = LevelMaxPorder[level];
        meta[K.MPrecision] = 15; meta[K.MChMode] = o.ChannelMode ?? -1;
        meta[K.MFramesStart] = framesStart; meta[K.MMaxFrameInit] = MaxFrameSize(blockSize, channels, bitsPerSample); meta[K.MLastN] = lastN;
        s.Meta.View.CopyFromCPU(meta);

        long slots = (long)frames * channels * blockSize;
        int subframes = frames * channels;
        var x = await s.X.EnsureDrainedAsync(slots);
        var r = await s.R.EnsureDrainedAsync(slots);
        var fr = await s.Frames.EnsureDrainedAsync((long)frames * K.FrameStride);
        var subs = await s.Subs.EnsureDrainedAsync((long)subframes * K.SubStride);
        var lpc = await s.Lpc.EnsureDrainedAsync((long)subframes * K.LpcInts);
        var lpcD = await s.LpcD.EnsureDrainedAsync((long)subframes * K.LpcDoubles);
        var parms = await s.Parms.EnsureDrainedAsync((long)subframes * K.MaxPartitions);
        var sums = await s.Sums.EnsureDrainedAsync((long)subframes * K.ScratchLongs);
        var commentView = await s.Comment.EnsureDrainedAsync(Math.Max(1, comment.Length));
        commentView.SubView(0, comment.Length).CopyFromCPU(comment);
        x.SubView(0, slots).MemSetToZero();
        subs.SubView(0, (long)subframes * K.SubStride).MemSetToZero();
        var m = s.Meta.View;

        var pacer = new LaunchPacer(_accelerator);
        long total = perChannel * channels;
        const int Chunk = 1 << 22;
        for (long first = 0; first < total; first += Chunk)
        {
            int count = (int)Math.Min(Chunk, total - first);
            s.Deinterleave(count, input, x, m, first);
            await pacer.LaunchedAsync();
        }
        s.StereoMode(frames, x, fr, m);
        if (channels == 2) s.Decorrelate(frames * blockSize, x, fr, m);
        s.Wasted(frames, x, fr, subs, m);
        s.LpcAnalysis(subframes, x, fr, subs, lpc, lpcD, m);
        s.Choose(subframes, x, r, fr, subs, lpc, sums, parms, m, 0);
        s.FrameBits(frames, fr, subs, m);
        s.Choose(subframes, x, r, fr, subs, lpc, sums, parms, m, 1);   // frames marked for the verbatim fallback only
        s.FrameBits(frames, fr, subs, m);
        s.Offsets(1, fr, m);
        await _accelerator.SynchronizeAsync();
        var back = await m.CopyToHostAsync();
        int length = back[K.MTotalBytes];

        var output = await ensureOutput(length);
        output.SubView(0, length).MemSetToZero();
        s.WriteFrame(frames, r, fr, subs, parms, output, m);

        // MD5 of the samples as FFmpeg hashes them (little-endian, 2 or 3 bytes each)
        int bytesPerSample = bitsPerSample / 8;
        long totalBytes = total * bytesPerSample;
        s.Md5.View.SubView(0, 4).CopyFromCPU(new uint[] { 0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476 });
        long blocks = totalBytes / 64;
        for (long b = 0; b < blocks; b += W.Md5BlocksPerLaunch)
        {
            int count = (int)Math.Min(W.Md5BlocksPerLaunch, blocks - b);
            s.Md5Blocks(1, input, s.Md5.View, s.Md5Tables.View, bytesPerSample, totalBytes, b, count);
            await pacer.LaunchedAsync();
        }
        s.Md5Final(1, input, s.Md5.View, s.Md5Tables.View, bytesPerSample, totalBytes);
        s.Header(framesStart, output, commentView, comment.Length, o.Padding, m, s.Md5.View);
        await _accelerator.SynchronizeAsync();
        return length;
    }
}
