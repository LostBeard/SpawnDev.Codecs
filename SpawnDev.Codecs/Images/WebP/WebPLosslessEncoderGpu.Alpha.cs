// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// ALPH chunk encoder: the alpha plane of a lossy WebP (VP8X + ALPH + VP8). The plane is predicted by one of the four
// filters (none / horizontal / vertical / gradient, chosen by a cost estimate on the accelerator unless the caller asks
// for one), then either stored raw or compressed with the VP8L coder this class already is (a green-only image, no
// 5-byte VP8L header). Pre-processing is level 0: the plane is reproduced exactly. The output is the chunk PAYLOAD (header
// byte + data): a lossy encoder wraps it in "ALPH" + size + padding; WebPDecoderGpu.DecodeAlphaAsync reads it back.
//
// Three overloads, funnelling into the ArrayView1D one, like every entry point of this class.

using System.Runtime.Versioning;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.SpawnJS.JSObjects;

namespace SpawnDev.Codecs.Images.WebP;

public sealed partial class WebPLosslessEncoderGpu
{
    /// <summary>From this effort on, <see cref="WebPAlphaFilter.Auto"/> encodes with every filter and keeps the smallest; below it a quick estimate picks.</summary>
    private const int AlphaExactFilterTrialQuality = 50;

    /// <summary>
    /// Upper bound on the size of the ALPH payload of a <paramref name="width"/> x <paramref name="height"/> plane, for sizing the
    /// output buffer of <see cref="EncodeAlphaAsync(ArrayView1D{byte, Stride1D.Dense}, int, int, ArrayView1D{byte, Stride1D.Dense}, WebPAlphaEncodeOptions?)"/>.
    /// </summary>
    public static int GetMaxAlphaEncodedSize(int width, int height)
    {
        ValidateSize(width, height);
        long size = 1 + (long)width * height; // a longer coded stream is replaced by the raw plane
        if (size > int.MaxValue) throw new ArgumentOutOfRangeException(nameof(width), "Plane is too large to encode.");
        return (int)size;
    }

    /// <summary>
    /// Encode an alpha plane that is already on the accelerator into an ALPH chunk payload in a caller-owned view. Only
    /// scalars cross to the host.
    /// </summary>
    /// <param name="alpha">One byte per pixel, rows top to bottom, at least width x height bytes.</param>
    /// <param name="width">Width in pixels (1..16384).</param>
    /// <param name="height">Height in pixels (1..16384).</param>
    /// <param name="output">Receives the payload; at least <see cref="GetMaxAlphaEncodedSize"/> bytes.</param>
    /// <param name="options">Encoder options (null = defaults).</param>
    /// <returns>The length of the payload in <paramref name="output"/>.</returns>
    public async Task<int> EncodeAlphaAsync(
        ArrayView1D<byte, Stride1D.Dense> alpha, int width, int height,
        ArrayView1D<byte, Stride1D.Dense> output, WebPAlphaEncodeOptions? options = null)
    {
        ValidateSize(width, height);
        options ??= new WebPAlphaEncodeOptions();
        if (options.Quality < 0 || options.Quality > 100) throw new ArgumentOutOfRangeException(nameof(options), "Quality must be 0..100.");
        if (!Enum.IsDefined(options.Filter)) throw new ArgumentOutOfRangeException(nameof(options), "Unknown alpha filter.");
        long pixels = (long)width * height;
        if (alpha.Length < pixels)
            throw new ArgumentException($"alpha must hold {pixels} bytes for {width}x{height}.", nameof(alpha));
        int maxSize = GetMaxAlphaEncodedSize(width, height);
        if (output.Length < maxSize)
            throw new ArgumentException($"output must hold {maxSize} bytes (GetMaxAlphaEncodedSize) for {width}x{height}.", nameof(output));

        var hostMeta = new int[WebPEncMeta.Size];
        // the coded stream may use at most the raw size (1 header byte + the plane): longer, and the plane is stored raw instead
        hostMeta[WebPEncMeta.OutputCapacity] = 1 + width * height;
        hostMeta[WebPEncMeta.PayloadStart] = 1; // after the ALPH header byte
        hostMeta[WebPEncMeta.AlphaFilterRequest] = (int)options.Filter;
        hostMeta[WebPEncMeta.Width] = width;
        hostMeta[WebPEncMeta.Height] = height;

        int n = (int)pixels;
        var meta = _meta.View;
        var plane = alpha;
        var rowCost = _rowCost.Ensure(4L * height);

        if (!options.Compress)
        {
            // a prediction filter cannot make a stored plane smaller: Auto means None here
            if (options.Filter == WebPAlphaFilter.Auto) hostMeta[WebPEncMeta.AlphaFilterRequest] = (int)WebPAlphaFilter.None;
            meta.CopyFromCPU(hostMeta);
            _kernels.AlphaPickFilter(plane, rowCost, meta, height);
            _kernels.AlphaRaw(plane, output, meta, n);
            _kernels.FinalizeAlpha(output, meta, 1);
        }
        else
        {
            var job = new StreamJob(width, height, options.Quality, a => _kernels.AlphaToArgb(plane, a, meta, n), Plane: true,
                AfterMeta: () => _kernels.AlphaPickFilter(plane, rowCost, meta, height));
            if (options.Filter == WebPAlphaFilter.Auto && options.Quality >= AlphaExactFilterTrialQuality)
            {
                // Exact choice: measure the stream each filter produces and keep the filter with the smallest one. The estimate
                // the filter pick kernel makes (used below this effort) cannot see LZ77 runs, which decide masks and flat areas.
                // A measuring pass runs the whole encoder with a writable capacity of 0: the bit writer still counts every byte
                // (and flags the overflow, which is expected here), but stores nothing, so the caller's buffer only ever sees the
                // final stream and no stale bytes of a longer trial are left behind it.
                int realCapacity = hostMeta[WebPEncMeta.OutputCapacity];
                int bestFilter = 0;
                int bestLength = int.MaxValue;
                hostMeta[WebPEncMeta.OutputCapacity] = 0;
                for (int f = 0; f < 4; f++)
                {
                    hostMeta[WebPEncMeta.AlphaFilterRequest] = f;
                    await EncodeStreamAsync(job, hostMeta, output);
                    await _accelerator.SynchronizeAsync();
                    var trial = await meta.CopyToHostAsync();
                    if (trial[WebPEncMeta.PayloadLength] < bestLength)
                    {
                        bestLength = trial[WebPEncMeta.PayloadLength];
                        bestFilter = f;
                    }
                }
                hostMeta[WebPEncMeta.OutputCapacity] = realCapacity;
                hostMeta[WebPEncMeta.AlphaFilterRequest] = bestFilter;
            }
            await EncodeStreamAsync(job, hostMeta, output);
            _kernels.FinalizeAlpha(output, meta, 0);
            await _accelerator.SynchronizeAsync();
            var m = await meta.CopyToHostAsync();
            if (m[WebPEncMeta.Error] != 0 || m[WebPEncMeta.Total] >= 1 + pixels)
            {
                // the coder did not beat storing the plane (its stream overflowed the raw size): method 0, like libwebp
                // stored raw, a filter only costs the decoder time: an automatic choice becomes None (an explicit one is honoured)
                if (options.Filter == WebPAlphaFilter.Auto) meta.SubView(WebPEncMeta.AlphaFilter, 1).CopyFromCPU(new[] { 0 });
                _kernels.AlphaRaw(plane, output, meta, n);
                _kernels.FinalizeAlpha(output, meta, 1);
            }
        }
        await _accelerator.SynchronizeAsync();

        var result = await meta.CopyToHostAsync();
        if (result[WebPEncMeta.Error] != 0 || result[WebPEncMeta.Total] > output.Length)
            throw new InvalidOperationException(
                $"ALPH encode overflowed the output buffer (needed {result[WebPEncMeta.Total]} bytes, GetMaxAlphaEncodedSize said {maxSize}).");
        return result[WebPEncMeta.Total];
    }

    /// <summary>
    /// Encode a managed alpha plane. Returns the ALPH payload as a managed array (a terminal CPU sink; in a browser prefer
    /// the <see cref="TypedArray"/> overload).
    /// </summary>
    public async Task<byte[]> EncodeAlphaAsync(ReadOnlyMemory<byte> alpha, int width, int height, WebPAlphaEncodeOptions? options = null)
    {
        ValidateSize(width, height);
        long needed = (long)width * height;
        if (alpha.Length < needed) throw new ArgumentException($"alpha must hold {needed} bytes for {width}x{height}.", nameof(alpha));
        _stageIn.Ensure(needed);
        DeviceTransfer.Upload(_stageIn.View, alpha.Span.Slice(0, (int)needed));
        _stageOut.Ensure(GetMaxAlphaEncodedSize(width, height));
        int length = await EncodeAlphaAsync(_stageIn.View, width, height, _stageOut.View, options);
        return await DeviceTransfer.DownloadAsync(_stageOut.View, length);
    }

    /// <summary>
    /// Encode an alpha plane held in a JS typed array (browser accelerators only). The bytes go to the GPU with
    /// <c>CopyFromJS</c> and the payload comes back as a JS <see cref="Uint8Array"/>; none of it enters the .NET heap.
    /// The caller disposes the result.
    /// </summary>
    [SupportedOSPlatform("browser")]
    public async Task<Uint8Array> EncodeAlphaAsync(TypedArray alpha, int width, int height, WebPAlphaEncodeOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(alpha);
        ValidateSize(width, height);
        long needed = (long)width * height;
        if (alpha.ByteLength < needed) throw new ArgumentException($"alpha must hold {needed} bytes for {width}x{height}.", nameof(alpha));
        _stageIn.Ensure(needed);
        DeviceTransfer.Upload(_stageIn.View, alpha);
        _stageOut.Ensure(GetMaxAlphaEncodedSize(width, height));
        int length = await EncodeAlphaAsync(_stageIn.View, width, height, _stageOut.View, options);
        return await DeviceTransfer.DownloadUint8ArrayAsync(_stageOut.View, length);
    }
}
