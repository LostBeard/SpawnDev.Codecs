// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP8 keyframe decoder, FULL keyframe syntax, 100% accelerator resident. Decodes what real encoders (libwebp, libvpx)
// write: B_PRED with all ten sub-modes and their contexts, all 16x16 and chroma modes, segmentation (map + per-segment
// quantizer and loop-filter level, absolute or delta), mb_no_coeff_skip, 1/2/4/8 token partitions with the partition size
// table, per-frame coefficient probability updates, quantizer deltas, and the in-loop deblocking filter (normal and
// simple, sharpness, mode/ref deltas). The v1 decoder (Vp8KeyframeDecoderGpu) only decodes what OUR encoder emits.
//
// Pipeline (host = pure coordinator: allocate, upload the frame, dispatch, read back the meta block):
//   1. header kernel   one thread: frame header -> meta block, coefficient probabilities, dequantizers, partition table
//   2. mode kernel     one thread per launch, one macroblock row per launch: segment / skip / modes -> macroblock records
//   3. token kernel    one thread per launch, one macroblock row per launch: DCT tokens -> raw coefficients
//   4. recon kernel    one thread per macroblock, one launch per wavefront diagonal d = col + 2 * row (see the kernel)
//   5. loop filter     one thread per macroblock, one launch per wavefront diagonal (see the kernel)
//   6. crop            the macroblock grid is cropped to the signalled size on the accelerator
// Entropy decoding is sequential by nature (a single GPU thread, chunked per macroblock row so no launch runs long);
// the pixel stages are parallel across the wavefront. Truncated or corrupt input throws Vp8DecodeException with the
// specific reason: the bool decoders detect reads past the end of their partition on the accelerator.
//
// API: three overloads funnelling into the ArrayView1D one (see Vp8KeyframeDecoderGpu): ArrayView1D (frame already on the
// accelerator, planes written into caller views), managed (ReadOnlyMemory in, byte[] planes out), browser TypedArray
// (JS bytes in, JS Uint8Array planes out, nothing enters the .NET heap).

using System.Runtime.Versioning;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.SpawnJS.JSObjects;

namespace SpawnDev.Codecs.Video.Vp8;

/// <summary>Full-syntax VP8 keyframe decoder on the accelerator.</summary>
/// <remarks>Not thread-safe: one in-flight decode per instance (the staging buffer is shared across calls).</remarks>
public sealed class Vp8FullKeyframeDecoderGpu : IDisposable
{
    private readonly Accelerator _accelerator;

    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>,
        ArrayView<int>, ArrayView<int>> _header;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<byte>, ArrayView<byte>, int> _modes;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<byte>, ArrayView<byte>, ArrayView<short>,
        ArrayView<byte>, ArrayView<byte>, int, int, int> _tokens;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<short>, ArrayView<int>,
        int, int> _recon;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<int>,
        int, int> _loopFilter;
    private readonly Vp8StridedPlanePackKernel _crop;

    // Constants uploaded once per accelerator.
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _dcQ;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _acQ;
    private readonly MemoryBuffer1D<byte, Stride1D.Dense> _defaultProbs;
    private readonly MemoryBuffer1D<byte, Stride1D.Dense> _updateProbs;
    private readonly MemoryBuffer1D<byte, Stride1D.Dense> _tables;

    // Grow-only staging for the encoded bytes of the managed and TypedArray overloads.
    private readonly StageBuffer<byte> _stageIn;

    /// <summary>Compile the kernels and upload the constant tables onto <paramref name="accelerator"/>.</summary>
    public Vp8FullKeyframeDecoderGpu(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _header = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<byte>,
            ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>>(Vp8FullHeaderKernel.HeaderKernel);
        _modes = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<byte>,
            ArrayView<byte>, int>(Vp8FullModeKernel.ModeKernel);
        _tokens = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<byte>,
            ArrayView<byte>, ArrayView<short>, ArrayView<byte>, ArrayView<byte>, int, int, int>(Vp8FullTokenKernel.TokenKernel);
        _recon = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>,
            ArrayView<short>, ArrayView<int>, int, int>(Vp8FullReconKernel.ReconKernel);
        _loopFilter = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>,
            ArrayView<int>, int, int>(Vp8FullLoopFilterKernel.LoopFilterKernel);
        _crop = new Vp8StridedPlanePackKernel(accelerator);
        _stageIn = new StageBuffer<byte>(accelerator);

        _dcQ = accelerator.Allocate1D<int>(128);
        _acQ = accelerator.Allocate1D<int>(128);
        _defaultProbs = accelerator.Allocate1D<byte>(4 * 264);
        _updateProbs = accelerator.Allocate1D<byte>(4 * 264);
        _tables = accelerator.Allocate1D<byte>(Vp8FullTables.Total);
        _dcQ.View.CopyFromCPU(Vp8FrameSetupKernel.BuildDcQLookup());
        _acQ.View.CopyFromCPU(Vp8FrameSetupKernel.BuildAcQLookup());
        _defaultProbs.View.CopyFromCPU(Vp8FrameSetupKernel.BuildDefaultCoefProbs());
        _updateProbs.View.CopyFromCPU(Vp8FrameSetupKernel.BuildUpdateCoefProbs());
        _tables.View.CopyFromCPU(Vp8FullTables.Build());
    }

    /// <summary>Luma plane size in bytes for a <paramref name="width"/> x <paramref name="height"/> picture.</summary>
    public static long GetLumaPlaneSize(int width, int height) => (long)width * height;

    /// <summary>Chroma plane size in bytes (each of Cb, Cr) for a <paramref name="width"/> x <paramref name="height"/> picture.</summary>
    public static long GetChromaPlaneSize(int width, int height) => (long)((width + 1) / 2) * ((height + 1) / 2);

    // ------------------------------------------------------------------
    // Frame tag (the 10 uncompressed bytes: plain integer fields per RFC 6386 section 9.1)
    // ------------------------------------------------------------------

    private static Vp8FrameTag ParseTag(ReadOnlySpan<byte> h)
    {
        if (h.Length < 10) throw Vp8DecodeException.From(Vp8DecodeError.TooShort);
        if ((h[0] & 1) != 0) throw Vp8DecodeException.From(Vp8DecodeError.NotKeyFrame);
        if (h[3] != 0x9D || h[4] != 0x01 || h[5] != 0x2A) throw Vp8DecodeException.From(Vp8DecodeError.BadStartCode);
        var tag = Vp8FrameTagParser.Parse(h.Slice(0, 10));
        if (tag.Width is not > 0 || tag.Height is not > 0) throw Vp8DecodeException.From(Vp8DecodeError.ZeroSize);
        if ((int)tag.Version > 3) throw Vp8DecodeException.From(Vp8DecodeError.UnsupportedVersion);
        return tag;
    }

    /// <summary>
    /// Read and validate the uncompressed header of an encoded keyframe that is on the accelerator (a 10-byte readback).
    /// Use the returned size to allocate the output planes.
    /// </summary>
    /// <exception cref="Vp8DecodeException">The header is not a decodable keyframe header.</exception>
    public async Task<Vp8FrameTag> ReadFrameTagAsync(ArrayView1D<byte, Stride1D.Dense> encoded)
    {
        if (encoded.Length < 10) throw Vp8DecodeException.From(Vp8DecodeError.TooShort);
        byte[] header = await DeviceTransfer.DownloadAsync(encoded, 10);
        return ParseTag(header);
    }

    // ------------------------------------------------------------------
    // ArrayView1D overload: the implementation
    // ------------------------------------------------------------------

    /// <summary>
    /// Decode an encoded VP8 keyframe that is already on the accelerator into caller-owned plane views. Only the 10-byte
    /// header and the 128-int meta block cross to the host.
    /// </summary>
    /// <param name="encoded">The encoded keyframe (may be longer than <paramref name="encodedLength"/>).</param>
    /// <param name="encodedLength">Number of valid bytes in <paramref name="encoded"/>.</param>
    /// <param name="yOut">Receives the luma plane, tightly packed; at least <see cref="GetLumaPlaneSize"/> bytes.</param>
    /// <param name="uOut">Receives the Cb plane, tightly packed; at least <see cref="GetChromaPlaneSize"/> bytes.</param>
    /// <param name="vOut">Receives the Cr plane, tightly packed; at least <see cref="GetChromaPlaneSize"/> bytes.</param>
    /// <returns>The parsed frame tag (width, height, ...).</returns>
    /// <exception cref="Vp8DecodeException">The frame is truncated or corrupt; <see cref="Vp8DecodeException.Error"/> says how.</exception>
    public async Task<Vp8FrameTag> DecodeKeyFrameAsync(
        ArrayView1D<byte, Stride1D.Dense> encoded, int encodedLength,
        ArrayView1D<byte, Stride1D.Dense> yOut,
        ArrayView1D<byte, Stride1D.Dense> uOut,
        ArrayView1D<byte, Stride1D.Dense> vOut)
    {
        if (encodedLength < 10) throw Vp8DecodeException.From(Vp8DecodeError.TooShort);
        if (encodedLength > encoded.Length) throw new ArgumentOutOfRangeException(nameof(encodedLength));
        var tag = await ReadFrameTagAsync(encoded);
        if (10L + tag.FirstPartitionSize > encodedLength) throw Vp8DecodeException.From(Vp8DecodeError.FirstPartitionExceedsFrame);
        await DecodeCoreAsync(encoded, encodedLength, tag, yOut, uOut, vOut);
        return tag;
    }

    private async Task DecodeCoreAsync(
        ArrayView1D<byte, Stride1D.Dense> encoded, int encodedLength, Vp8FrameTag tag,
        ArrayView1D<byte, Stride1D.Dense> yOut,
        ArrayView1D<byte, Stride1D.Dense> uOut,
        ArrayView1D<byte, Stride1D.Dense> vOut)
    {
        int width = tag.Width!.Value;
        int height = tag.Height!.Value;
        int mbCols = (width + 15) / 16;
        int mbRows = (height + 15) / 16;
        int workWidth = mbCols * 16, workHeight = mbRows * 16;
        int uvWorkWidth = mbCols * 8, uvWorkHeight = mbRows * 8;
        int uvWidth = (width + 1) / 2, uvHeight = (height + 1) / 2;
        long mbCount = (long)mbCols * mbRows;

        long yBytes = GetLumaPlaneSize(width, height);
        long uvBytes = GetChromaPlaneSize(width, height);
        if (yOut.Length < yBytes) throw new ArgumentException($"yOut must hold {yBytes} bytes.", nameof(yOut));
        if (uOut.Length < uvBytes) throw new ArgumentException($"uOut must hold {uvBytes} bytes.", nameof(uOut));
        if (vOut.Length < uvBytes) throw new ArgumentException($"vOut must hold {uvBytes} bytes.", nameof(vOut));

        // Host-known metadata: the tiny blob the header kernel starts from.
        var hostMeta = new int[Vp8FullMeta.Size];
        hostMeta[Vp8FullMeta.Width] = width;
        hostMeta[Vp8FullMeta.Height] = height;
        hostMeta[Vp8FullMeta.MbCols] = mbCols;
        hostMeta[Vp8FullMeta.MbRows] = mbRows;
        hostMeta[Vp8FullMeta.FrameLength] = encodedLength;
        hostMeta[Vp8FullMeta.FirstPartLen] = tag.FirstPartitionSize;

        using var dMeta = _accelerator.Allocate1D<int>(Vp8FullMeta.Size);
        using var dBd = _accelerator.Allocate1D<int>(Vp8FullMeta.BdCount * Vp8FullMeta.BdStateInts);
        using var dProbs = _accelerator.Allocate1D<byte>(4 * 264);
        using var dRecs = _accelerator.Allocate1D<byte>(mbCount * Vp8MbRecord.Size);
        using var dCoeffs = _accelerator.Allocate1D<short>(mbCount * Vp8FullTokenKernel.CoeffsPerMb);
        using var dAbove = _accelerator.Allocate1D<byte>((long)mbCols * 9 + 9);   // above context + the left context of the token kernel
        dMeta.View.CopyFromCPU(hostMeta);
        dBd.View.MemSetToZero();
        dRecs.View.MemSetToZero();
        dCoeffs.View.MemSetToZero();
        dAbove.View.MemSetToZero();

        // A picture that is already macroblock-aligned decodes straight into the caller's planes; otherwise the grid is
        // decoded into working planes and cropped on the accelerator.
        bool aligned = width == workWidth && height == workHeight;
        MemoryBuffer1D<byte, Stride1D.Dense>? wY = null, wU = null, wV = null;
        try
        {
            ArrayView1D<byte, Stride1D.Dense> yT, uT, vT;
            if (aligned)
            {
                yT = yOut.SubView(0, yBytes);
                uT = uOut.SubView(0, uvBytes);
                vT = vOut.SubView(0, uvBytes);
            }
            else
            {
                wY = _accelerator.Allocate1D<byte>((long)workWidth * workHeight);
                wU = _accelerator.Allocate1D<byte>((long)uvWorkWidth * uvWorkHeight);
                wV = _accelerator.Allocate1D<byte>((long)uvWorkWidth * uvWorkHeight);
                yT = wY.View; uT = wU.View; vT = wV.View;
            }
            yT.MemSetToZero();
            uT.MemSetToZero();
            vT.MemSetToZero();

            ArrayView<byte> enc = encoded;
            _header(1, enc, dMeta.View, dBd.View, dProbs.View, _defaultProbs.View, _updateProbs.View, _dcQ.View, _acQ.View);
            for (int row = 0; row < mbRows; row++)
                _modes(1, enc, dMeta.View, dBd.View, dRecs.View, _tables.View, row);
            for (int row = 0; row < mbRows; row++)
                for (int col = 0; col < mbCols; col += Vp8FullTokenKernel.MbsPerLaunch)
                    _tokens(1, enc, dMeta.View, dBd.View, dRecs.View, dProbs.View, dCoeffs.View, dAbove.View, _tables.View, row, col, col + Vp8FullTokenKernel.MbsPerLaunch);

            int diagonals = mbCols + Vp8FullMeta.WaveRowWeight * (mbRows - 1);
            for (int d = 0; d < diagonals; d++)
            {
                WaveRange(d, mbCols, mbRows, out int rowLo, out int count);
                if (count <= 0) continue;                // a single-column picture has no macroblock on odd diagonals
                _recon(count, yT, uT, vT, dRecs.View, dCoeffs.View, dMeta.View, d, rowLo);
            }
            for (int d = 0; d < diagonals; d++)
            {
                WaveRange(d, mbCols, mbRows, out int rowLo, out int count);
                if (count <= 0) continue;                // a single-column picture has no macroblock on odd diagonals
                _loopFilter(count, yT, uT, vT, dRecs.View, dMeta.View, d, rowLo);
            }

            if (!aligned)
            {
                _crop.Run(wY!.View, 0, workWidth, yOut, 0, width, height);
                _crop.Run(wU!.View, 0, uvWorkWidth, uOut, 0, uvWidth, uvHeight);
                _crop.Run(wV!.View, 0, uvWorkWidth, vOut, 0, uvWidth, uvHeight);
            }

            // Drain before the scoped buffers dispose: on the browser backends a dispatch that still references a freed
            // buffer reads freed memory.
            await _accelerator.SynchronizeAsync();
            int[] meta = await dMeta.View.SubView(0, 1).CopyToHostAsync();
            if (meta[Vp8FullMeta.Error] != 0) throw Vp8DecodeException.From((Vp8DecodeError)meta[Vp8FullMeta.Error]);
        }
        finally
        {
            wY?.Dispose();
            wU?.Dispose();
            wV?.Dispose();
        }
    }

    /// <summary>
    /// The wavefront diagonal a macroblock is reconstructed and loop-filtered on. A macroblock needs its left, top, top-left
    /// and top-right neighbours finished, so all of those must be on EARLIER diagonals, and two macroblocks on one diagonal
    /// must not touch each other's pixels (public so the schedule can be property-tested; see the recon and loop filter kernels).
    /// </summary>
    public static int WaveDiagonal(int mbRow, int mbCol) => mbCol + Vp8FullMeta.WaveRowWeight * mbRow;

    /// <summary>The macroblock rows on wavefront diagonal <paramref name="d"/> (column = d - WaveRowWeight * row): rows rowLo .. rowLo + count - 1.</summary>
    public static void GetWaveRows(int d, int mbCols, int mbRows, out int rowLo, out int count) => WaveRange(d, mbCols, mbRows, out rowLo, out count);

    private static void WaveRange(int d, int mbCols, int mbRows, out int rowLo, out int count)
    {
        int lo = d - (mbCols - 1);
        rowLo = lo > 0 ? (lo + Vp8FullMeta.WaveRowWeight - 1) / Vp8FullMeta.WaveRowWeight : 0;
        int rowHi = Math.Min(mbRows - 1, d / Vp8FullMeta.WaveRowWeight);
        count = rowHi - rowLo + 1;
    }

    /// <summary>Number of wavefront launches for a picture of the given macroblock size.</summary>
    public static int GetWaveCount(int mbCols, int mbRows) => mbCols + Vp8FullMeta.WaveRowWeight * (mbRows - 1);

    // ------------------------------------------------------------------
    // Managed overload
    // ------------------------------------------------------------------

    /// <summary>
    /// Decode an encoded VP8 keyframe to YUV planes (managed in, managed out: a terminal CPU source/sink; in a browser prefer
    /// the <see cref="TypedArray"/> overload).
    /// </summary>
    /// <exception cref="Vp8DecodeException">The frame is truncated or corrupt.</exception>
    public async Task<Vp8KeyframeDecoderGpu.DecodedFrame> DecodeKeyFrameAsync(ReadOnlyMemory<byte> encoded)
    {
        if (encoded.Length < 10) throw Vp8DecodeException.From(Vp8DecodeError.TooShort);
        var tag = ParseTag(encoded.Span.Slice(0, 10));
        if (10L + tag.FirstPartitionSize > encoded.Length) throw Vp8DecodeException.From(Vp8DecodeError.FirstPartitionExceedsFrame);
        _stageIn.Ensure(encoded.Length);
        DeviceTransfer.Upload(_stageIn.View, encoded.Span);

        int width = tag.Width!.Value, height = tag.Height!.Value;
        using var dY = _accelerator.Allocate1D<byte>(GetLumaPlaneSize(width, height));
        using var dU = _accelerator.Allocate1D<byte>(GetChromaPlaneSize(width, height));
        using var dV = _accelerator.Allocate1D<byte>(GetChromaPlaneSize(width, height));
        await DecodeCoreAsync(_stageIn.View, encoded.Length, tag, dY.View, dU.View, dV.View);

        var yPlane = await DeviceTransfer.DownloadAsync(dY.View, dY.Length);
        var uPlane = await DeviceTransfer.DownloadAsync(dU.View, dU.Length);
        var vPlane = await DeviceTransfer.DownloadAsync(dV.View, dV.Length);
        return new Vp8KeyframeDecoderGpu.DecodedFrame(yPlane, uPlane, vPlane, width, height);
    }

    // ------------------------------------------------------------------
    // TypedArray overload (browser: bulk bytes stay in JS)
    // ------------------------------------------------------------------

    /// <summary>
    /// Decode an encoded VP8 keyframe held in a JS typed array (browser accelerators only). The bytes go to the GPU with
    /// <c>CopyFromJS</c> and the planes come back as JS <see cref="Uint8Array"/>s; none of it enters the .NET heap. The
    /// caller disposes the result.
    /// </summary>
    /// <exception cref="NotSupportedException">The accelerator is not a browser accelerator.</exception>
    /// <exception cref="Vp8DecodeException">The frame is truncated or corrupt.</exception>
    [SupportedOSPlatform("browser")]
    public async Task<Vp8KeyframeDecoderGpu.DecodedFrameJS> DecodeKeyFrameAsync(TypedArray encoded)
    {
        ArgumentNullException.ThrowIfNull(encoded);
        long length = encoded.ByteLength;
        if (length < 10) throw Vp8DecodeException.From(Vp8DecodeError.TooShort);
        _stageIn.Ensure(length);
        DeviceTransfer.Upload(_stageIn.View, encoded);

        var tag = await ReadFrameTagAsync(_stageIn.View);
        if (10L + tag.FirstPartitionSize > length) throw Vp8DecodeException.From(Vp8DecodeError.FirstPartitionExceedsFrame);
        int width = tag.Width!.Value, height = tag.Height!.Value;
        using var dY = _accelerator.Allocate1D<byte>(GetLumaPlaneSize(width, height));
        using var dU = _accelerator.Allocate1D<byte>(GetChromaPlaneSize(width, height));
        using var dV = _accelerator.Allocate1D<byte>(GetChromaPlaneSize(width, height));
        await DecodeCoreAsync(_stageIn.View, (int)length, tag, dY.View, dU.View, dV.View);

        var y = await DeviceTransfer.DownloadUint8ArrayAsync(dY.View, dY.Length);
        var u = await DeviceTransfer.DownloadUint8ArrayAsync(dU.View, dU.Length);
        var v = await DeviceTransfer.DownloadUint8ArrayAsync(dV.View, dV.Length);
        return new Vp8KeyframeDecoderGpu.DecodedFrameJS(y, u, v, width, height);
    }

    /// <summary>Release the staging buffer and constant tables.</summary>
    public void Dispose()
    {
        _crop.Dispose();
        _dcQ.Dispose();
        _acQ.Dispose();
        _defaultProbs.Dispose();
        _updateProbs.Dispose();
        _tables.Dispose();
        _stageIn.Dispose();
    }
}
