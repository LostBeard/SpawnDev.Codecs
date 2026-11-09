// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// FLAC decoder (RFC 9639) on the accelerator: any conforming stream - fixed or variable block size, 4..32 bits per
// sample, 1..8 channels, every subframe and residual coding, wasted bits, every stereo decorrelation - as written by
// libFLAC, FFmpeg and others. ComfyUI's Load Audio reads FLAC through PyAV / FFmpeg; this decodes to the same samples.
//
// Host work (pure coordinator): parse the metadata blocks (an ID3v2 tag before "fLaC" is skipped; STREAMINFO,
// VORBIS_COMMENT), upload the file once, dispatch FlacStreamDecodeKernels (frame search, frame chain, one thread per frame
// decode, decorrelation + interleave, MD5 of the samples), read back the frame count and the samples. Frames are checked
// by their CRC-8 and CRC-16 as libFLAC checks them; a damaged frame comes out as silence of its length and the samples
// after it keep their time (DecodeResult reports how many).

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using K = SpawnDev.Codecs.Audio.Flac.FlacStreamDecodeKernels;
using W = SpawnDev.Codecs.Audio.Flac.FlacMd5Kernels;

namespace SpawnDev.Codecs.Audio.Flac;

/// <summary>A decoded FLAC stream.</summary>
public sealed record FlacDecodeResult
{
    /// <summary>Sample rate in Hz (STREAMINFO).</summary>
    public required int SampleRateHz { get; init; }
    /// <summary>Channel count.</summary>
    public required int Channels { get; init; }
    /// <summary>Bits per sample (4..32): the samples are integers in that range.</summary>
    public required int BitsPerSample { get; init; }
    /// <summary>Samples per channel.</summary>
    public required long TotalSamplesPerChannel { get; init; }
    /// <summary>Interleaved samples (channel 0[0], channel 1[0], ...), in the stream's integer range
    /// (<see cref="FlacDecoderGpu.DecodeAsync"/>; empty from <see cref="FlacDecoderGpu.DecodeFloatAsync"/>).</summary>
    public int[] InterleavedSamples { get; init; } = Array.Empty<int>();
    /// <summary>Float32 samples, planar [channel][sample] (<see cref="FlacDecoderGpu.DecodeFloatAsync"/>; else null).</summary>
    public float[]? PlanarFloat { get; init; }
    /// <summary>True when STREAMINFO holds an MD5 of the samples (not all zero) and it was checked.</summary>
    public required bool Md5Checked { get; init; }
    /// <summary>True when the decoded samples' MD5 equals STREAMINFO's (false when not checked).</summary>
    public required bool Md5Matches { get; init; }
    /// <summary>Frames decoded.</summary>
    public required int Frames { get; init; }
    /// <summary>Frames whose CRCs passed but whose subframes could not be decoded (each comes out as silence).</summary>
    public required int BadFrames { get; init; }
    /// <summary>Samples per channel of silence where frames were lost (damaged headers or CRC-16 failures).</summary>
    public required long LostSamples { get; init; }
    /// <summary>The VORBIS_COMMENT vendor string, or null.</summary>
    public string? Vendor { get; init; }
    /// <summary>The VORBIS_COMMENT fields, in order ("KEY", "value"), or empty.</summary>
    public IReadOnlyList<KeyValuePair<string, string>> Tags { get; init; } = Array.Empty<KeyValuePair<string, string>>();
}

/// <summary>FLAC decoder on the accelerator (see the file header).</summary>
public sealed class FlacDecoderGpu : IDisposable
{
    private readonly Accelerator _accelerator;
    // Kernels load lazily (EnsureKernelsAsync, awaited by every entry point): the constructor compiles nothing, so
    // creating a decoder never holds a browser page's thread, and each compile yields to the event loop first.
    private Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, long> _findSync = null!;
    private Action<Index1D, ArrayView<byte>, ArrayView<int>> _countSync = null!;
    private Action<Index1D, ArrayView<int>, ArrayView<int>> _syncOffsets = null!;
    private Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>> _compactSync = null!;
    private Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int> _frameEnd = null!;
    private Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<long>, ArrayView<int>> _walk = null!;
    private Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<long>, ArrayView<int>, ArrayView<int>, ArrayView<long>, ArrayView<int>, ArrayView<int>, int> _decodeFrame = null!;
    private Action<Index1D, ArrayView<long>, ArrayView<int>, ArrayView<int>, ArrayView<long>, ArrayView<int>, ArrayView<int>, long> _interleave = null!;
    private Action<Index1D, ArrayView<int>, ArrayView<uint>, ArrayView<uint>, int, long, long, int> _md5Blocks = null!;
    private Action<Index1D, ArrayView<int>, ArrayView<uint>, ArrayView<uint>, int, long> _md5Final = null!;
    private Action<Index1D, ArrayView<int>, ArrayView<float>, int, long, int, long> _toFloat = null!;
    private Task? _kernelsLoad;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _crcTable;
    private readonly MemoryBuffer1D<uint, Stride1D.Dense> _md5Tables;

    private const int ByteChunk = 1 << 22, FrameChunk = 4096, SampleChunk = 1 << 22;

    /// <summary>A decoder bound to <paramref name="accelerator"/>. Compiles no kernels (see <see cref="WarmUpAsync"/>).</summary>
    public FlacDecoderGpu(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        var a = accelerator;
        _crcTable = a.Allocate1D<int>(256);
        var t = new int[256];
        for (int i = 0; i < 256; i++)
        {
            int c = i << 8;
            for (int k = 0; k < 8; k++) c = (c & 0x8000) != 0 ? ((c << 1) ^ 0x8005) & 0xFFFF : (c << 1) & 0xFFFF;
            t[i] = c;
        }
        _crcTable.View.CopyFromCPU(t);
        _md5Tables = a.Allocate1D<uint>(128);
        _md5Tables.View.CopyFromCPU(W.BuildTables());
    }

    /// <summary>Load every kernel once (awaited loads: each compile yields to the browser's event loop first).</summary>
    private Task EnsureKernelsAsync()
    {
        if (_kernelsLoad == null || _kernelsLoad.IsFaulted || _kernelsLoad.IsCanceled) _kernelsLoad = LoadKernelsAsync();
        return _kernelsLoad;
    }

    private async Task LoadKernelsAsync()
    {
        var a = _accelerator;
        _findSync = await a.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, long>(K.FindSyncKernel);
        _countSync = await a.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<byte>, ArrayView<int>>(K.CountSyncKernel);
        _syncOffsets = await a.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<int>, ArrayView<int>>(K.SyncOffsetsKernel);
        _compactSync = await a.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>>(K.CompactSyncKernel);
        _frameEnd = await a.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int>(K.FrameEndKernel);
        _walk = await a.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<long>, ArrayView<int>>(K.WalkKernel);
        _decodeFrame = await a.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<long>, ArrayView<int>, ArrayView<int>, ArrayView<long>, ArrayView<int>, ArrayView<int>, int>(K.DecodeFrameKernel);
        _interleave = await a.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<long>, ArrayView<int>, ArrayView<int>, ArrayView<long>, ArrayView<int>, ArrayView<int>, long>(K.InterleaveKernel);
        _md5Blocks = await a.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<int>, ArrayView<uint>, ArrayView<uint>, int, long, long, int>(W.Md5BlocksKernel);
        _md5Final = await a.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<int>, ArrayView<uint>, ArrayView<uint>, int, long>(W.Md5FinalKernel);
        _toFloat = await a.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<int>, ArrayView<float>, int, long, int, long>(ToFloatKernel);
    }

    private sealed record StreamInfo(int MinBlock, int MaxBlock, int MaxFrame, int Rate, int Channels, int Bps, long Total,
        byte[] Md5, int AudioStart, string? Vendor, List<KeyValuePair<string, string>> Tags);

    /// <summary>The metadata blocks: STREAMINFO and VORBIS_COMMENT; where the frames start.</summary>
    private static StreamInfo ParseMetadata(ReadOnlySpan<byte> f)
    {
        int pos = 0;
        // an ID3v2 tag before the stream (libFLAC skips it too)
        if (f.Length >= 10 && f[0] == 'I' && f[1] == 'D' && f[2] == '3')
        {
            int size = (f[6] & 0x7F) << 21 | (f[7] & 0x7F) << 14 | (f[8] & 0x7F) << 7 | (f[9] & 0x7F);
            pos = 10 + size + ((f[5] & 0x10) != 0 ? 10 : 0);
        }
        if (f.Length < pos + 4 + 38 || f[pos] != 'f' || f[pos + 1] != 'L' || f[pos + 2] != 'a' || f[pos + 3] != 'C')
            throw new InvalidDataException("Not a FLAC stream (no fLaC marker).");
        pos += 4;
        StreamInfo? si = null;
        string? vendor = null;
        var tags = new List<KeyValuePair<string, string>>();
        while (true)
        {
            if (pos + 4 > f.Length) throw new InvalidDataException("FLAC metadata runs past the end of the file.");
            bool last = (f[pos] & 0x80) != 0;
            int type = f[pos] & 0x7F;
            int len = f[pos + 1] << 16 | f[pos + 2] << 8 | f[pos + 3];
            pos += 4;
            if (pos + len > f.Length) throw new InvalidDataException("FLAC metadata block runs past the end of the file.");
            var b = f.Slice(pos, len);
            if (type == 0 && len >= 34)
            {
                int minBlock = b[0] << 8 | b[1], maxBlock = b[2] << 8 | b[3];
                int maxFrame = b[7] << 16 | b[8] << 8 | b[9];
                int rate = b[10] << 12 | b[11] << 4 | b[12] >> 4;
                int channels = ((b[12] >> 1) & 7) + 1;
                int bps = ((b[12] & 1) << 4 | b[13] >> 4) + 1;
                long total = (long)(b[13] & 15) << 32 | (long)b[14] << 24 | (long)b[15] << 16 | (long)b[16] << 8 | b[17];
                si = new StreamInfo(minBlock, maxBlock, maxFrame, rate, channels, bps, total, b.Slice(18, 16).ToArray(), 0, null, tags);
            }
            else if (type == 4 && len >= 8)
            {
                // Vorbis comment: little-endian lengths
                int p = 0;
                int vl = BitConverter.ToInt32(b.Slice(p, 4)); p += 4;
                if (vl >= 0 && p + vl <= len)
                {
                    vendor = System.Text.Encoding.UTF8.GetString(b.Slice(p, vl)); p += vl;
                    if (p + 4 <= len)
                    {
                        int n = BitConverter.ToInt32(b.Slice(p, 4)); p += 4;
                        for (int i = 0; i < n && p + 4 <= len; i++)
                        {
                            int l = BitConverter.ToInt32(b.Slice(p, 4)); p += 4;
                            if (l < 0 || p + l > len) break;
                            string field = System.Text.Encoding.UTF8.GetString(b.Slice(p, l)); p += l;
                            int eq = field.IndexOf('=');
                            tags.Add(eq < 0 ? new(field, "") : new(field[..eq], field[(eq + 1)..]));
                        }
                    }
                }
            }
            pos += len;
            if (last) break;
        }
        if (si is null) throw new InvalidDataException("FLAC stream without STREAMINFO.");
        if (si.Bps < 4 || si.Bps > 32) throw new InvalidDataException($"FLAC bits per sample {si.Bps} is outside 4..32.");
        return si with { AudioStart = pos, Vendor = vendor };
    }

    /// <summary>A 64-sample stereo 16-bit FLAC file (one LPC frame, FFmpeg level 5, metadata trimmed) for <see cref="WarmUpAsync"/>.</summary>
    private const string WarmUpFile =
        "ZkxhQ4AAACISABIAAACbAACbCsRC8AAAAECODGdxc8e8G0RiBWZwZIMt//hpmAA/CUoAAACJAIBgVdAwGBdLmMiaDjqWnU+izf5dhKvekL92xU/xxos8YGBgSCBwSYMS5GuzS2F5UdGJ08WPB6p17NGlx1suRMGRQ+JnycAAACpAU0B6AJ0gu8DVAOgcxeZRZ1f+2jo4/7QsTV/ZoAyGK6MXbVtNJJPaf58tm6RzCDHDiBhAgogpDqqLStJbzySzSSWgics=";

    /// <summary>
    /// Compile and load every kernel <see cref="DecodeAsync"/> and <see cref="DecodeFloatAsync"/> use, then decode a tiny
    /// built-in file both ways, so the first real decode does not wait for them. On browser backends the first compile
    /// of the shaders can take tens of seconds; call this early, for example when a page shows a node that loads audio.
    /// The loads are awaited one kernel at a time and yield to the browser's event loop before each compile (WebGPU builds
    /// the pipelines asynchronously), so the page keeps painting and answering input meanwhile.
    /// </summary>
    public async Task WarmUpAsync()
    {
        await EnsureKernelsAsync();
        var file = Convert.FromBase64String(WarmUpFile);
        await DecodeAsync(file);
        await DecodeFloatAsync(file);
    }

    /// <summary>Decode a FLAC file to interleaved integer samples (see the file header).</summary>
    public Task<FlacDecodeResult> DecodeAsync(ReadOnlyMemory<byte> file) => DecodeCoreAsync(file, toFloat: false);

    /// <summary>
    /// Decode a FLAC file to float32, planar [channel][sample] (<see cref="FlacDecodeResult.PlanarFloat"/>), as ComfyUI's
    /// Load Audio gets it through PyAV / FFmpeg: FFmpeg's FLAC decoder shifts each sample into int16 (8..16 bits) or
    /// int32 (17..32 bits), and ComfyUI converts that int to float32 (round to nearest even) and divides by 2^15 or 2^31.
    /// Done on the accelerator; only the floats come back.
    /// </summary>
    public Task<FlacDecodeResult> DecodeFloatAsync(ReadOnlyMemory<byte> file) => DecodeCoreAsync(file, toFloat: true);

    /// <summary>One thread per sample: interleaved int -> planar float32 (see <see cref="DecodeFloatAsync"/>).</summary>
    private static void ToFloatKernel(Index1D index, ArrayView<int> interleaved, ArrayView<float> planar, int channels, long n, int bps, long first)
    {
        long i = first + index;
        long s = i / channels;
        int c = (int)(i - s * channels);
        float v = bps <= 16
            ? (float)(interleaved[i] << (16 - bps)) / 32768f
            : (float)(interleaved[i] << (32 - bps)) / 2147483648f;
        planar[(long)c * n + s] = v;
    }

    private async Task<FlacDecodeResult> DecodeCoreAsync(ReadOnlyMemory<byte> file, bool toFloat)
    {
        var si = ParseMetadata(file.Span);
        await EnsureKernelsAsync();
        int channels = si.Channels, bps = si.Bps;
        var audio = file[si.AudioStart..];
        long length = audio.Length;
        var a = _accelerator;

        using var data = a.Allocate1D<byte>(Math.Max(1, length));
        if (length > 0) data.View.SubView(0, length).CopyFromCPU(audio.ToArray());
        var dataView = data.View.SubView(0, Math.Max(1, length));
        using var meta = a.Allocate1D<int>(K.MetaSize);
        int syncBlocks = (int)Math.Max(1, (length + K.SyncBlock - 1) / K.SyncBlock);
        var m0 = new int[K.MetaSize];
        m0[K.MChannels] = channels; m0[K.MBps] = bps; m0[K.MRate] = si.Rate; m0[K.MMaxFrame] = si.MaxFrame;
        m0[K.MSyncBlocks] = syncBlocks;
        m0[K.MBlockSize] = si.MinBlock == si.MaxBlock ? si.MinBlock : 0;
        meta.View.CopyFromCPU(m0);

        // 1-2. frame header candidates, in file order
        using var flags = a.Allocate1D<byte>(Math.Max(1, length));
        var pacer = new LaunchPacer(a);
        for (long first = 0; first < length; first += ByteChunk)
        {
            _findSync((int)Math.Min(ByteChunk, length - first), dataView, flags.View.SubView(0, Math.Max(1, length)), meta.View, first);
            await pacer.LaunchedAsync();
        }
        var flagView = flags.View.SubView(0, Math.Max(1, length));
        using var blockCounts = a.Allocate1D<int>(syncBlocks);
        if (length > 0)
        {
            _countSync(syncBlocks, flagView, blockCounts.View);
            _syncOffsets(1, blockCounts.View, meta.View);
        }
        await a.SynchronizeAsync();
        var mHost = await meta.CopyToHostAsync();
        int candidates = mHost[K.MCandidates];
        using var cands = a.Allocate1D<int>(Math.Max(1, candidates));
        using var ends = a.Allocate1D<int>(Math.Max(1, candidates));
        using var frameOffsets = a.Allocate1D<int>(Math.Max(1, candidates));
        using var frameEnds = a.Allocate1D<int>(Math.Max(1, candidates));
        using var frameSamples = a.Allocate1D<long>(candidates + 1);
        if (candidates > 0)
        {
            _compactSync(syncBlocks, flagView, blockCounts.View, cands.View);
            // 3. where each candidate's frame ends (CRC-16)
            for (int first = 0; first < candidates; first += FrameChunk)
            {
                _frameEnd(Math.Min(FrameChunk, candidates - first), dataView, cands.View, _crcTable.View, ends.View, meta.View, first);
                await pacer.LaunchedAsync();
            }
        }
        // 4. the frame chain
        _walk(1, dataView, cands.View, ends.View, frameOffsets.View, frameEnds.View, frameSamples.View, meta.View);
        await a.SynchronizeAsync();
        mHost = await meta.CopyToHostAsync();
        int frames = mHost[K.MFrames];
        long walked = (await frameSamples.View.SubView(frames, 1).CopyToHostAsync())[0];
        long total = Math.Max(walked, si.Total);

        // 5. frames -> channel-major regions
        using var buf = a.Allocate1D<int>(Math.Max(1, walked * channels));
        using var wide = a.Allocate1D<long>(bps == 32 && channels == 2 ? Math.Max(1, walked) : 1);
        using var frameInfo = a.Allocate1D<int>(Math.Max(1, frames * K.FrameInts));
        using var coefs = a.Allocate1D<int>(Math.Max(1, (long)frames * K.MaxChannels * K.MaxLpcOrder));
        for (int first = 0; first < frames; first += FrameChunk)
        {
            _decodeFrame(Math.Min(FrameChunk, frames - first), dataView, frameOffsets.View, frameEnds.View, frameSamples.View, frameInfo.View,
                buf.View, wide.View, coefs.View, meta.View, first);
            await pacer.LaunchedAsync();
        }
        // 6. decorrelation + interleave
        using var output = a.Allocate1D<int>(Math.Max(1, total * channels));
        for (long first = 0; first < total; first += SampleChunk)
        {
            _interleave((int)Math.Min(SampleChunk, total - first), frameSamples.View, frameInfo.View, buf.View, wide.View,
                output.View, meta.View, first);
            await pacer.LaunchedAsync();
        }

        // MD5 of the samples (little-endian, ceil(bps / 8) bytes each), when STREAMINFO has one
        bool md5Checked = si.Md5.Any(x => x != 0);
        bool md5Matches = false;
        if (md5Checked)
        {
            using var md5 = a.Allocate1D<uint>(W.Md5Size);
            md5.View.SubView(0, 4).CopyFromCPU(new uint[] { 0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476 });
            int bytesPerSample = (bps + 7) / 8;
            long totalBytes = total * channels * bytesPerSample;
            long blocks = totalBytes / 64;
            var outView = output.View.SubView(0, Math.Max(1, total * channels));
            for (long b = 0; b < blocks; b += W.Md5BlocksPerLaunch)
            {
                _md5Blocks(1, outView, md5.View, _md5Tables.View, bytesPerSample, totalBytes, b, (int)Math.Min(W.Md5BlocksPerLaunch, blocks - b));
                await pacer.LaunchedAsync();
            }
            _md5Final(1, outView, md5.View, _md5Tables.View, bytesPerSample, totalBytes);
            await a.SynchronizeAsync();
            var digest = await md5.View.SubView(W.Md5Digest, 16).CopyToHostAsync();
            md5Matches = true;
            for (int i = 0; i < 16; i++) md5Matches &= digest[i] == si.Md5[i];
        }

        await a.SynchronizeAsync();
        mHost = await meta.CopyToHostAsync();
        int[] samples = Array.Empty<int>();
        float[]? planar = null;
        if (toFloat)
        {
            using var floats = a.Allocate1D<float>(Math.Max(1, total * channels));
            for (long first = 0; first < total * channels; first += SampleChunk)
            {
                _toFloat((int)Math.Min(SampleChunk, total * channels - first), output.View, floats.View, channels, total, bps, first);
                await pacer.LaunchedAsync();
            }
            await a.SynchronizeAsync();
            planar = total > 0 ? await floats.View.SubView(0, total * channels).CopyToHostAsync() : Array.Empty<float>();
        }
        else if (total > 0) samples = await output.View.SubView(0, total * channels).CopyToHostAsync();
        return new FlacDecodeResult
        {
            SampleRateHz = si.Rate,
            Channels = channels,
            BitsPerSample = bps,
            TotalSamplesPerChannel = total,
            InterleavedSamples = samples,
            PlanarFloat = planar,
            Md5Checked = md5Checked,
            Md5Matches = md5Matches,
            Frames = frames,
            BadFrames = mHost[K.MBadFrames],
            LostSamples = mHost[K.MGapSamples] + (total - walked),
            Vendor = si.Vendor,
            Tags = si.Tags,
        };
    }

    /// <summary>Decode a complete FLAC byte stream (compatibility form of <see cref="DecodeAsync"/>).</summary>
    public async Task<FlacGpuDecodeResult> DecodeStreamAsync(byte[] flacBytes)
    {
        ArgumentNullException.ThrowIfNull(flacBytes);
        var r = await DecodeAsync(flacBytes);
        return new FlacGpuDecodeResult
        {
            SampleRateHz = r.SampleRateHz,
            Channels = r.Channels,
            BitsPerSample = r.BitsPerSample,
            BlockSize = ParseMetadata(flacBytes).MaxBlock,
            TotalSamplesPerChannel = (int)r.TotalSamplesPerChannel,
            InterleavedSamples = r.InterleavedSamples,
        };
    }

    /// <summary>Release accelerator-bound resources.</summary>
    public void Dispose()
    {
        _crcTable.Dispose();
        _md5Tables.Dispose();
    }
}

/// <summary>Result of <see cref="FlacDecoderGpu.DecodeStreamAsync"/>.</summary>
public sealed record FlacGpuDecodeResult
{
    /// <summary>Sample rate in Hz.</summary>
    public required int SampleRateHz { get; init; }
    /// <summary>Channel count.</summary>
    public required int Channels { get; init; }
    /// <summary>Bits per sample.</summary>
    public required int BitsPerSample { get; init; }
    /// <summary>STREAMINFO's largest block size.</summary>
    public required int BlockSize { get; init; }
    /// <summary>Total samples per channel across the stream.</summary>
    public required int TotalSamplesPerChannel { get; init; }
    /// <summary>Interleaved PCM samples: channel 0[0], channel 1[0], ...</summary>
    public required int[] InterleavedSamples { get; init; }
}
