// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt). See NOTICE.md for upstream attributions.
//
// VP9 decoder on the accelerator (profile 0: 8-bit 4:2:0, every libvpx profile-0 feature: inter frames, compound and
// scaled references, tiles, segmentation, lossless, hidden / show-existing frames, superframes). Every frame of
// libvpx's 303 profile-0 conformance vectors decodes to libvpx v1.17.0's output bit for bit
// (Plans/PLAN-VP9-Decoder-Full.md, CodecsTestBase.Vp9DecoderVectorTests.cs).
//
// Host = coordinator: container packets, frame headers, superframe index and tile sizes (metadata) on the host; the
// compressed header, tile parse, reconstruction, loop filter and backward adaptation are kernels; pictures stay on the
// accelerator until the caller takes them (one readback per output frame for the managed overload, none for ArrayView).
// One output frame per packet, as libvpx: the packet's last shown frame (a show_existing_frame shows a reference).

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp9.Decoder;

/// <summary>Size and chroma layout of a decoded VP9 picture.</summary>
/// <param name="Width">Luma width (crop size).</param>
/// <param name="Height">Luma height.</param>
/// <param name="SubsamplingX">1 for 4:2:0 / 4:2:2.</param>
/// <param name="SubsamplingY">1 for 4:2:0.</param>
public readonly record struct Vp9PictureFormat(int Width, int Height, int SubsamplingX, int SubsamplingY)
{
    /// <summary>Chroma plane width ((Width + SubsamplingX) &gt;&gt; SubsamplingX).</summary>
    public int ChromaWidth => (Width + SubsamplingX) >> SubsamplingX;
    /// <summary>Chroma plane height.</summary>
    public int ChromaHeight => (Height + SubsamplingY) >> SubsamplingY;
    /// <summary>Bytes of the tight planar picture: Y (Width x Height), then U, then V (ChromaWidth x ChromaHeight each).</summary>
    public long PlanarBytes => (long)Width * Height + 2L * ChromaWidth * ChromaHeight;
}

/// <summary>A decoded VP9 picture: tight planar 8-bit Y, U, V (I420 for 4:2:0) in <see cref="Planes"/>.</summary>
public sealed record Vp9VideoFrame(Vp9PictureFormat Format, byte[] Planes);

/// <summary>VP9 decoder whose whole pipeline runs on an ILGPU accelerator.</summary>
public sealed class Vp9DecoderGpu : IDisposable
{
    private readonly Accelerator _acc;
    private readonly Vp9DecodePipeline _pipe;
    private MemoryBuffer1D<byte, Stride1D.Dense>? _packet;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, int, int, int, int, int, int, long, long, long> _pack;

    /// <summary>A decoder on <paramref name="accelerator"/>.</summary>
    public Vp9DecoderGpu(Accelerator accelerator)
    {
        _acc = accelerator ?? throw new ArgumentNullException(nameof(accelerator));
        _pipe = new Vp9DecodePipeline(_acc);
        _pack = _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<byte>, int, int, int, int, int, int, long, long, long>(
            Vp9OutputKernels.PackPlanesKernel);
    }

    /// <summary>Decode the packets of a WebM / Matroska (first V_VP9 track) or IVF file; one picture per shown packet.</summary>
    public async IAsyncEnumerable<Vp9VideoFrame> DecodeFileAsync(byte[] file)
    {
        foreach (byte[] packet in Vp9Packets.ReadFile(file))
        {
            var frame = await DecodePacketAsync(packet);
            if (frame != null) yield return frame;
        }
    }

    /// <summary>Decode one packet (a WebM block or IVF frame; superframes included). Returns its picture, or null when the
    /// packet shows nothing (hidden frames only).</summary>
    public async Task<Vp9VideoFrame?> DecodePacketAsync(ReadOnlyMemory<byte> packet)
    {
        int show = await DecodePacketCoreAsync(packet);
        if (show < 0) return null;
        var format = Format(show);
        using var dst = _acc.Allocate1D<byte>(format.PlanarBytes);
        Pack(show, dst.View);
        return new Vp9VideoFrame(format, await DeviceTransfer.DownloadAsync(dst.View, format.PlanarBytes));
    }

    /// <summary>
    /// Decode one packet and write its picture (tight planar Y, U, V; see <see cref="Vp9PictureFormat.PlanarBytes"/>) into
    /// <paramref name="dst"/> on the accelerator. Returns the picture's format, or null when the packet shows nothing.
    /// </summary>
    public async Task<Vp9PictureFormat?> DecodePacketAsync(ReadOnlyMemory<byte> packet, ArrayView1D<byte, Stride1D.Dense> dst)
    {
        int show = await DecodePacketCoreAsync(packet);
        if (show < 0) return null;
        var format = Format(show);
        if (dst.Length < format.PlanarBytes) throw new ArgumentException($"dst holds {dst.Length} bytes, the picture needs {format.PlanarBytes}.", nameof(dst));
        Pack(show, dst);
        return format;
    }

    private async Task<int> DecodePacketCoreAsync(ReadOnlyMemory<byte> packet)
    {
        if (packet.IsEmpty) return -1;
        if (_packet == null || _packet.Length < packet.Length)
        {
            _packet?.Dispose();
            _packet = _acc.Allocate1D<byte>(Math.Max(packet.Length, 1 << 16));
        }
        DeviceTransfer.Upload(_packet.View, packet.Span);   // the compressed packet (necessary input)
        int show = -1;
        foreach (var (offset, length) in Vp9Packets.Split(packet.Span))
        {
            var h = await _pipe.DecodeFrameAsync(packet.Slice(offset, length), _packet.View, offset);
            if (h.ShowExistingFrame) show = h.FrameToShowBuffer;
            else if (h.ShowFrame) show = h.NewBuffer;
        }
        return show;
    }

    private Vp9PictureFormat Format(int buffer)
    {
        var l = _pipe.Frames.Layout(buffer);
        return new Vp9PictureFormat(l.Width, l.Height, l.SsX, l.SsY);
    }

    private void Pack(int buffer, ArrayView<byte> dst)
    {
        var l = _pipe.Frames.Layout(buffer);
        long b = _pipe.Frames.Offset(buffer);
        var f = Format(buffer);
        _pack(checked((int)f.PlanarBytes), _pipe.Frames.Pool, dst, l.Width, l.Height, f.ChromaWidth, f.ChromaHeight, l.YStride, l.UvStride,
            b, b + l.UOffset, b + l.VOffset);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _pipe.Dispose();
        _packet?.Dispose();
    }
}
