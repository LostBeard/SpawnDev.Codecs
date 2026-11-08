// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt). See NOTICE.md for upstream attributions.
//
// MP3 (MPEG-1 / 2 / 2.5 Layer III) decoder on the accelerator (port of minimp3, CC0 / public domain,
// github.com/lieff/minimp3): float output as minimp3 built with MINIMP3_FLOAT_OUTPUT gives it (sample / 32768, not
// clipped), bit for bit on clean streams (see Mp3DecodeKernels). ComfyUI's Load Audio decodes MP3 through PyAV / FFmpeg's
// own decoder (not ported, LGPL); the tests report how close the two are.
//
// Host work (pure coordinator): find the container tags (an ID3v2 tag at the start, its TXXX / text frames read for
// Tags; ID3v1 and APEv2 at the end, skipped as minimp3_ex skips them), upload the file once, dispatch
// Mp3DecodeKernels (walk, side info, reservoir, gather; then per batch of frames granule decode, stereo, IMDCT,
// synthesis), read back a few counts and the samples. Gapless: when the first frame is a Xing / Info frame with a
// LAME-format extension, its encoder delay + 529 samples are dropped at the start and its padding - 529 at the end of the
// frames decoded (FFmpeg's convention; Mp3EncoderGpu writes its files for it). The Info frame is not audio.
// A damaged or missing frame comes out as zero lines (silence after the previous frame's overlap) and keeps its time.
// Layer I / II streams are not decoded (NotSupportedException).

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using K = SpawnDev.Codecs.Audio.Mp3.Mp3DecodeKernels;

namespace SpawnDev.Codecs.Audio.Mp3;

/// <summary>A decoded MP3 stream.</summary>
public sealed record Mp3DecodeResult
{
    /// <summary>Sample rate in Hz.</summary>
    public required int SampleRateHz { get; init; }
    /// <summary>Channel count (1 or 2).</summary>
    public required int Channels { get; init; }
    /// <summary>Samples per channel (after the gapless trim).</summary>
    public required long TotalSamplesPerChannel { get; init; }
    /// <summary>Interleaved float32 samples (channel 0[0], channel 1[0], ...), nominally in [-1, 1]
    /// (<see cref="Mp3DecoderGpu.DecodeAsync"/>; empty from <see cref="Mp3DecoderGpu.DecodeFloatAsync"/>).</summary>
    public float[] Interleaved { get; init; } = Array.Empty<float>();
    /// <summary>Float32 samples, planar [channel][sample] (<see cref="Mp3DecoderGpu.DecodeFloatAsync"/>; else null).</summary>
    public float[]? PlanarFloat { get; init; }
    /// <summary>Audio frames (the Info frame not counted; frames lost to damage counted).</summary>
    public required int Frames { get; init; }
    /// <summary>Frames that came out as zero lines: lost in damage, rejected side info, or main data that was not
    /// there (bit reservoir reaching before a loss).</summary>
    public required int LostFrames { get; init; }
    /// <summary>Samples per channel dropped at the start (the Info frame's encoder delay + 529), 0 without one.</summary>
    public required int StartTrim { get; init; }
    /// <summary>The Info frame's padding - 529 (samples per channel dropped at the end when positive), 0 without one.</summary>
    public required int EndTrim { get; init; }
    /// <summary>ID3v2 text frames in order: TXXX as (description, value), other T*** frames as (frame id, value).</summary>
    public IReadOnlyList<KeyValuePair<string, string>> Tags { get; init; } = Array.Empty<KeyValuePair<string, string>>();
}

/// <summary>MP3 decoder on the accelerator (port of minimp3, CC0; see the file header).</summary>
public sealed class Mp3DecoderGpu : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _ints;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _floats;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int> _walk;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int> _sideInfo;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, int, int> _reservoir;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<byte>, ArrayView<int>, int> _gather;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<float>, ArrayView<int>, ArrayView<int>, ArrayView<float>, ArrayView<int>, int, int> _granule;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<float>, ArrayView<int>, ArrayView<int>, ArrayView<float>, ArrayView<int>, int, int> _stereo;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<int>, ArrayView<float>, ArrayView<int>, int, int> _imdctPrep;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<int>, int, int> _imdctOut;
    private readonly Action<Index1D, ArrayView<float>, long, long> _carry;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<int>, int> _dct;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<int>, long, long, long, int, long> _synth;

    /// <summary>Walker steps per launch, launches between meta reads.</summary>
    private const int WalkBudget = 1 << 14, WalkLaunchesPerRead = 8;
    /// <summary>Frames per launch of the per-frame stages and of the reservoir pass; frames per batch of the decode.</summary>
    private const int FrameChunk = 4096, ReservoirChunk = 4096, BatchFrames = 1024;
    /// <summary>Threads per launch of the per-line stages.</summary>
    private const int ThreadChunk = 1 << 20;

    /// <summary>A decoder bound to <paramref name="accelerator"/>.</summary>
    public Mp3DecoderGpu(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        var a = accelerator;
        _walk = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int>(K.WalkKernel);
        _sideInfo = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int>(K.SideInfoKernel);
        _reservoir = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, int, int>(K.ReservoirKernel);
        _gather = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<byte>, ArrayView<int>, int>(K.GatherKernel);
        _granule = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<float>, ArrayView<int>, ArrayView<int>, ArrayView<float>, ArrayView<int>, int, int>(K.GranuleKernel);
        _stereo = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<float>, ArrayView<int>, ArrayView<int>, ArrayView<float>, ArrayView<int>, int, int>(K.StereoKernel);
        _imdctPrep = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<int>, ArrayView<float>, ArrayView<int>, int, int>(K.ImdctPrepKernel);
        _imdctOut = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<int>, int, int>(K.ImdctOutKernel);
        _carry = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, long, long>(K.CarryKernel);
        _dct = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<int>, int>(K.DctKernel);
        _synth = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<int>, long, long, long, int, long>(K.SynthKernel);
        _ints = a.Allocate1D<int>(Mp3DecodeTables.IntCount);
        _ints.View.CopyFromCPU(Mp3DecodeTables.Ints);
        _floats = a.Allocate1D<float>(Mp3DecodeTables.FloatCount);
        _floats.View.CopyFromCPU(Mp3DecodeTables.Floats);
    }

    /// <summary>A 3-frame stereo 44.1 kHz MP3 (Info frame + 2 audio frames, FFmpeg / libmp3lame 32k) for <see cref="WarmUpAsync"/>.</summary>
    private const string WarmUpFile =
        "//tQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAASW5mbwAAAA8AAAADAAACCQCZmZmZmZmZmZmZmZmZmZmZmZmZmZmZmZmZmZmZmZmZmZnMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMz///////////////////////////////////////////8AAAAATGF2YzYyLjI4AAAAAAAAAAAAAAAAJAKjAAAAAAAAAgmjxhB/AAAAAAAAAAAAAAAAAAAAAP/7EGQAAAB5BtOFMAAKAAANIKAAAQQIM0oZoQAAAAA0gwAAABLEszjOBABAGhMfu2+Hh5eYQ8JQFaVgCjDWAYaDxF0bYXXGhq//fCgPgINcKgr9inAG4ABjjGZjRj+jjFYdlVNTZBw///sSZAoD8I8G1adgAAgAAA0g4AABAiAbTLWgACAAADSCgAAEdPMOgaaKZ0Cut14nG43nAAvDDDDAAAAAyFCUw55oJjcV30TS6TGd7v383lk/gYV/HixfAx34VUcDAIlrkSJGyJFFhsXp//sQZBsAAUMQ1c5goAAAAA0gwAAAAawbPhxgACgAADSDgAAETEFNRTMuMTAwVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVU=";

    /// <summary>
    /// Compile and load every kernel <see cref="DecodeAsync"/> and <see cref="DecodeFloatAsync"/> use (decodes a tiny
    /// built-in file both ways), so the first real decode does not wait for them. On browser backends the first compile
    /// of the shaders can take tens of seconds; call this early, for example when a page shows a node that loads audio.
    /// </summary>
    public async Task WarmUpAsync()
    {
        var file = Convert.FromBase64String(WarmUpFile);
        await DecodeAsync(file);
        await DecodeFloatAsync(file);
    }

    /// <summary>Decode an MP3 file to interleaved float32 samples (see the file header).</summary>
    public Task<Mp3DecodeResult> DecodeAsync(ReadOnlyMemory<byte> file) => DecodeCoreAsync(file, planar: false);

    /// <summary>
    /// Decode an MP3 file to float32, planar [channel][sample] (<see cref="Mp3DecodeResult.PlanarFloat"/>): the same values
    /// as <see cref="DecodeAsync"/>, laid out as ComfyUI's Load Audio hands its tensor on (the decoder's float output,
    /// unchanged). Written in that layout on the accelerator; only the floats come back.
    /// </summary>
    public Task<Mp3DecodeResult> DecodeFloatAsync(ReadOnlyMemory<byte> file) => DecodeCoreAsync(file, planar: true);

    // ------------------------------------------------------------------ container (tags)

    /// <summary>Where the MPEG frames are: after an ID3v2 tag, before an ID3v1 / APEv2 tag (minimp3_ex's rules).</summary>
    private static (int Start, int End, List<KeyValuePair<string, string>> Tags) ParseContainer(ReadOnlySpan<byte> f)
    {
        int start = 0, end = f.Length;
        var tags = new List<KeyValuePair<string, string>>();
        if (f.Length >= 10 && f[0] == 'I' && f[1] == 'D' && f[2] == '3'
            && (f[5] & 15) == 0 && (f[6] & 0x80) == 0 && (f[7] & 0x80) == 0 && (f[8] & 0x80) == 0 && (f[9] & 0x80) == 0)
        {
            long size = ((f[6] & 0x7F) << 21 | (f[7] & 0x7F) << 14 | (f[8] & 0x7F) << 7 | (f[9] & 0x7F)) + 10L;
            if ((f[5] & 16) != 0) size += 10;
            start = (int)Math.Min(size, f.Length);
            try { ReadId3v2Text(f[..start], tags); }
            catch (Exception) { /* a damaged tag only loses its text */ }
        }
        if (end - start >= 128 && f[end - 128] == 'T' && f[end - 127] == 'A' && f[end - 126] == 'G')
        {
            end -= 128;
            if (end - start >= 227 && f[end - 227] == 'T' && f[end - 226] == 'A' && f[end - 225] == 'G' && f[end - 224] == '+') end -= 227;
        }
        if (end - start > 32 && System.Text.Encoding.ASCII.GetString(f.Slice(end - 32, 8)) == "APETAGEX")
        {
            end -= 32;
            int tagSize = BitConverter.ToInt32(f.Slice(end + 8 + 4, 4));
            if (tagSize >= 0 && end - start >= tagSize) end -= tagSize;
        }
        return (start, end, tags);
    }

    private static int SyncSafe(ReadOnlySpan<byte> b) => (b[0] & 0x7F) << 21 | (b[1] & 0x7F) << 14 | (b[2] & 0x7F) << 7 | (b[3] & 0x7F);

    /// <summary>The text frames of an ID3v2.3 / 2.4 tag (unsynchronised tags are skipped).</summary>
    private static void ReadId3v2Text(ReadOnlySpan<byte> tag, List<KeyValuePair<string, string>> tags)
    {
        int ver = tag[3], flags = tag[5];
        if ((ver != 3 && ver != 4) || (flags & 0x80) != 0) return;
        int p = 10, end = tag.Length - ((flags & 0x10) != 0 ? 10 : 0);
        if ((flags & 0x40) != 0) p += ver == 4 ? SyncSafe(tag.Slice(p, 4)) : 4 + System.Buffers.Binary.BinaryPrimitives.ReadInt32BigEndian(tag.Slice(p, 4));
        while (p + 10 <= end)
        {
            if (tag[p] == 0) break;   // padding
            string id = System.Text.Encoding.ASCII.GetString(tag.Slice(p, 4));
            int size = ver == 4 ? SyncSafe(tag.Slice(p + 4, 4)) : System.Buffers.Binary.BinaryPrimitives.ReadInt32BigEndian(tag.Slice(p + 4, 4));
            int fflags = tag[p + 9];
            p += 10;
            if (size < 0 || p + size > end) break;
            var body = tag.Slice(p, size);
            p += size;
            if (id[0] != 'T' || body.Length < 1) continue;
            if (ver == 4 && (fflags & 0x0E) != 0) continue;   // compressed / encrypted / unsynchronised frame
            if (ver == 3 && (fflags & 0xC0) != 0) continue;   // compressed / encrypted frame
            if (ver == 4 && (fflags & 0x01) != 0)
            {
                if (body.Length < 5) continue;
                body = body[4..];   // data length indicator
            }
            int enc = body[0];
            var text = body[1..];
            if (id == "TXXX")
            {
                int cut = TextEnd(text, enc);
                string desc = DecodeText(text[..cut], enc);
                int skip = Math.Min(text.Length, cut + (enc == 1 || enc == 2 ? 2 : 1));
                tags.Add(new(desc, DecodeText(text[skip..], enc)));
            }
            else tags.Add(new(id, DecodeText(text, enc)));
        }
    }

    /// <summary>Where the first NUL terminator of a text in encoding <paramref name="enc"/> starts (or the length).</summary>
    private static int TextEnd(ReadOnlySpan<byte> t, int enc)
    {
        if (enc == 1 || enc == 2)
        {
            for (int i = 0; i + 1 < t.Length; i += 2)
                if (t[i] == 0 && t[i + 1] == 0) return i;
            return t.Length;
        }
        int z = t.IndexOf((byte)0);
        return z < 0 ? t.Length : z;
    }

    private static string DecodeText(ReadOnlySpan<byte> t, int enc)
    {
        string s = enc switch
        {
            0 => System.Text.Encoding.Latin1.GetString(t),
            1 => t.Length >= 2 && t[0] == 0xFE && t[1] == 0xFF
                ? System.Text.Encoding.BigEndianUnicode.GetString(t[2..])
                : System.Text.Encoding.Unicode.GetString(t.Length >= 2 && t[0] == 0xFF && t[1] == 0xFE ? t[2..] : t),
            2 => System.Text.Encoding.BigEndianUnicode.GetString(t),
            _ => System.Text.Encoding.UTF8.GetString(t),
        };
        return s.TrimEnd('\0');
    }

    // ------------------------------------------------------------------ decode

    private async Task<Mp3DecodeResult> DecodeCoreAsync(ReadOnlyMemory<byte> file, bool planar)
    {
        var (start, end, tags) = ParseContainer(file.Span);
        int length = end - start;
        if (length < 4) throw new InvalidDataException("No MPEG audio frames found.");
        if (length > int.MaxValue - (1 << 20)) throw new NotSupportedException("MP3 files of 2 GB or more are not supported.");
        var a = _accelerator;
        var pacer = new LaunchPacer(a);

        // the file once (the frames are a view into it)
        using var upload = a.Allocate1D<byte>(file.Length);
        if (System.Runtime.InteropServices.MemoryMarshal.TryGetArray(file, out var seg) && seg.Offset == 0 && seg.Count == seg.Array!.Length)
            upload.View.CopyFromCPU(seg.Array);
        else
            upload.View.CopyFromCPU(file.ToArray());
        var data = upload.View.SubView(start, length);

        // 1. walk the frames
        int cap = length / 12 + 16;
        using var meta = a.Allocate1D<int>(K.MetaSize);
        var m0 = new int[K.MetaSize];
        m0[K.MLen] = length;
        m0[K.MCap] = cap;
        meta.View.CopyFromCPU(m0);
        using var fOff = a.Allocate1D<int>(cap);
        using var fBytes = a.Allocate1D<int>(cap);
        using var fMain = a.Allocate1D<int>(cap);
        using var scratch = a.Allocate1D<int>(K.SideInts);
        int[] m;
        while (true)
        {
            for (int i = 0; i < WalkLaunchesPerRead; i++)
            {
                _walk(1, data, fOff.View, fBytes.View, fMain.View, meta.View, _ints.View, scratch.View, WalkBudget);
                await pacer.LaunchedAsync();
            }
            await a.SynchronizeAsync();
            m = await meta.CopyToHostAsync();
            if (m[K.MState] == K.StateDone) break;
        }
        if (m[K.MRate] == 0) throw new InvalidDataException("No MPEG audio frames found.");
        if (m[K.MLayer] != 3) throw new NotSupportedException($"MPEG audio Layer {m[K.MLayer]} is not decoded (Layer III only).");
        int frames = m[K.MFrames], nch = m[K.MChannels], ngr = m[K.MGranules], rate = m[K.MRate];
        int streamBytes = m[K.MStream];

        // gapless: delay + 529 dropped at the start, padding - 529 (when positive) at the end of the frames decoded, as
        // FFmpeg trims. minimp3_ex's mp3dec_load_buf cuts to the Info frame's frame count instead: the same length when
        // the count is right; some muxers write a wrong count (FFmpeg's, for MediaFoundation's mp3_mf, writes a third of
        // it), where a cut to it would drop most of the audio.
        long decoded = (long)frames * ngr * 576;
        long skip = 0, total = decoded;
        int startTrim = 0, endTrim = 0;
        if (m[K.MXing] > 0)
        {
            startTrim = m[K.MXDelay];
            endTrim = m[K.MXPadding];
            skip = startTrim;
            total = Math.Max(0, decoded - skip - Math.Max(0, endTrim));
        }

        int lostFrames = 0;
        float[] samples = Array.Empty<float>();
        if (frames > 0)
        {
            // 2-4. side info, reservoir, main-data stream
            using var side = a.Allocate1D<int>((long)frames * K.SideInts);
            for (int first = 0; first < frames; first += FrameChunk)
            {
                _sideInfo(Math.Min(FrameChunk, frames - first), data, fOff.View, fBytes.View, side.View, meta.View, _ints.View, first);
                await pacer.LaunchedAsync();
            }
            for (int first = 0; first < frames; first += ReservoirChunk)
            {
                _reservoir(1, fMain.View, side.View, meta.View, first, Math.Min(ReservoirChunk, frames - first));
                await pacer.LaunchedAsync();
            }
            using var stream = a.Allocate1D<byte>(Math.Max(1, streamBytes));
            for (int first = 0; first < frames; first += FrameChunk)
            {
                _gather(Math.Min(FrameChunk, frames - first), data, fOff.View, fBytes.View, fMain.View, stream.View, meta.View, first);
                await pacer.LaunchedAsync();
            }

            // 5-10. per batch of frames
            int batch = Math.Min(BatchFrames, frames);
            long gMax = (long)batch * ngr;
            using var spec = a.Allocate1D<float>(gMax * nch * 576);
            using var xbuf = a.Allocate1D<float>(gMax * nch * 576);
            using var tbuf = a.Allocate1D<float>(gMax * nch * 576);
            using var ist = a.Allocate1D<int>(gMax * nch * K.IstInts);
            using var ovl = a.Allocate1D<float>((gMax + 1) * nch * 288);
            using var dct = a.Allocate1D<float>((K.HistorySlots + gMax * 18) * nch * 32);
            ovl.View.MemSetToZero();
            dct.View.MemSetToZero();
            using var output = a.Allocate1D<float>(Math.Max(1, total * nch));
            for (int f0 = 0; f0 < frames; f0 += batch)
            {
                int bf = Math.Min(batch, frames - f0);
                int gB = bf * ngr;
                long firstSample = (long)f0 * ngr * 576;
                if (firstSample - skip >= total) break;
                await Launch(gB * nch, (n, first) => _granule(n, stream.View, side.View, spec.View, ist.View, _ints.View, _floats.View, meta.View, f0, (int)first), pacer);
                if (nch == 2)
                    await Launch(gB, (n, first) => _stereo(n, side.View, spec.View, ist.View, _ints.View, _floats.View, meta.View, f0, (int)first), pacer);
                await Launch(gB * nch * 32, (n, first) => _imdctPrep(n, side.View, spec.View, xbuf.View, ovl.View, _ints.View, _floats.View, meta.View, f0, (int)first), pacer);
                await Launch(gB * nch * 32, (n, first) => _imdctOut(n, side.View, xbuf.View, ovl.View, tbuf.View, _floats.View, meta.View, f0, (int)first), pacer);
                _carry(nch * 288, ovl.View, (long)gB * nch * 288, 0L);
                await pacer.LaunchedAsync();
                await Launch(gB * 18 * nch, (n, first) => _dct(n, tbuf.View, dct.View, _floats.View, meta.View, (int)first), pacer);
                await Launch((long)gB * 18 * nch * 32, (n, first) => _synth(n, dct.View, output.View, _floats.View, meta.View, firstSample, skip, total, planar ? 1 : 0, first), pacer);
                _carry(K.HistorySlots * nch * 32, dct.View, (long)gB * 18 * nch * 32, 0L);
                await pacer.LaunchedAsync();
            }
            await a.SynchronizeAsync();
            m = await meta.CopyToHostAsync();
            lostFrames = m[K.MUndecoded];
            if (total > 0) samples = await output.View.SubView(0, total * nch).CopyToHostAsync();
        }
        return new Mp3DecodeResult
        {
            SampleRateHz = rate,
            Channels = nch,
            TotalSamplesPerChannel = total,
            Interleaved = planar ? Array.Empty<float>() : samples,
            PlanarFloat = planar ? samples : null,
            Frames = frames,
            LostFrames = lostFrames,
            StartTrim = startTrim,
            EndTrim = endTrim,
            Tags = tags,
        };
    }

    /// <summary>Launch <paramref name="count"/> threads in bounded launches (<see cref="ThreadChunk"/> each).</summary>
    private static async ValueTask Launch(long count, Action<int, long> launch, LaunchPacer pacer)
    {
        for (long first = 0; first < count; first += ThreadChunk)
        {
            launch((int)Math.Min(ThreadChunk, count - first), first);
            await pacer.LaunchedAsync();
        }
    }

    /// <summary>Release accelerator-bound resources.</summary>
    public void Dispose()
    {
        _ints.Dispose();
        _floats.Dispose();
    }
}
