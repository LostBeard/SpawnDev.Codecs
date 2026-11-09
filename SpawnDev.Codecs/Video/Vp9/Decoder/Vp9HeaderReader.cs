// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt). See NOTICE.md for upstream attributions.
//
// VP9 uncompressed frame header + the decoder's frame bookkeeping, host side (metadata: sizes, flags, slot numbers;
// the coded data is the accelerator's). A port of libvpx v1.17.0 (BSD-3-Clause):
//   vp9/decoder/vp9_decodeframe.c  read_uncompressed_header, read_bitdepth_colorspace_sampling, setup_frame_size(_with_refs),
//                                  setup_loopfilter, setup_quantization, setup_segmentation(_dequant), setup_tile_info,
//                                  and the start of vp9_decode_frame (use_prev_frame_mvs)
//   vp9/decoder/vp9_decoder.c      vp9_receive_compressed_data + swap_frame_buffers (reference map, last_* state)
//   vp9/common/vp9_entropymode.c   vp9_setup_past_independence (the host half: deltas, segment features, sign bias)
//   vp9/common/vp9_tile_common.c   vp9_get_tile_n_bits
// The frame contexts and segmentation maps live on the accelerator; Vp9FrameInfo tells the decoder which of them
// this frame resets (Vp9ContextReset) and which context it loads and stores.

namespace SpawnDev.Codecs.Video.Vp9.Decoder;

/// <summary>libvpx INTERP_FILTER values (the frame header's switchable choice is <see cref="Switchable"/>).</summary>
public enum Vp9Interp : byte { EightTap = 0, EightTapSmooth = 1, EightTapSharp = 2, Bilinear = 3, Switchable = 4 }

/// <summary>Which frame contexts <c>vp9_setup_past_independence</c> resets to the defaults for this frame.</summary>
public enum Vp9ContextReset : byte { None = 0, All = 1, One = 2 }

/// <summary>Loop filter parameters of a frame (deltas persist between frames, as in libvpx).</summary>
public sealed record Vp9LoopFilterInfo(int Level, int Sharpness, bool DeltaEnabled, bool DeltaUpdate, int[] RefDeltas, int[] ModeDeltas);

/// <summary>Segmentation parameters of a frame (persist between frames unless updated, as in libvpx).</summary>
public sealed record Vp9SegmentationInfo(bool Enabled, bool UpdateMap, bool UpdateData, bool TemporalUpdate, bool AbsDelta,
    byte[] TreeProbs, byte[] PredProbs, int[] FeatureMask, int[,] FeatureData);

/// <summary>One frame's uncompressed header plus the decoder state libvpx derives from it.</summary>
public sealed record Vp9FrameInfo
{
    public int Profile { get; init; }
    public int BitDepth { get; init; }
    public bool ShowExistingFrame { get; init; }
    /// <summary>Frame buffer shown by a show_existing_frame frame.</summary>
    public int FrameToShowBuffer { get; init; } = -1;
    /// <summary>0 = key frame, 1 = inter frame (libvpx FRAME_TYPE).</summary>
    public int FrameType { get; init; }
    /// <summary>libvpx cm->last_frame_type: the previous decoded frame's type (0 = key).</summary>
    public int LastFrameType { get; init; }
    public bool ShowFrame { get; init; }
    public bool ErrorResilient { get; init; }
    public bool IntraOnly { get; init; }
    /// <summary>Key frame or intra-only frame (libvpx frame_is_intra_only).</summary>
    public bool IsIntraOnlyFrame => FrameType == 0 || IntraOnly;
    public int ResetFrameContext { get; init; }
    public int ColorSpace { get; init; }
    public int ColorRange { get; init; }
    public int SubsamplingX { get; init; }
    public int SubsamplingY { get; init; }
    public int RefreshFrameFlags { get; init; }
    /// <summary>Frame buffers of LAST, GOLDEN, ALTREF (inter frames).</summary>
    public int[] RefBuffers { get; init; } = { -1, -1, -1 };
    /// <summary>ref_frame_sign_bias indexed by MV_REFERENCE_FRAME (0 intra, 1 last, 2 golden, 3 altref).</summary>
    public int[] SignBias { get; init; } = new int[4];
    public int Width { get; init; }
    public int Height { get; init; }
    public int RenderWidth { get; init; }
    public int RenderHeight { get; init; }
    public int MiCols { get; init; }
    public int MiRows { get; init; }
    public bool AllowHighPrecisionMv { get; init; }
    public Vp9Interp InterpFilter { get; init; }
    public bool RefreshFrameContext { get; init; }
    public bool FrameParallel { get; init; }
    /// <summary>The frame context this frame loads (and stores when <see cref="RefreshFrameContext"/>).</summary>
    public int FrameContextIdx { get; init; }
    /// <summary>vp9_setup_past_independence ran: segment maps cleared, the current context set to the defaults.</summary>
    public bool PastIndependence { get; init; }
    public Vp9ContextReset ContextReset { get; init; }
    /// <summary>The context reset when <see cref="ContextReset"/> is <see cref="Vp9ContextReset.One"/> (the header's index).</summary>
    public int ContextResetIdx { get; init; }
    public required Vp9LoopFilterInfo LoopFilter { get; init; }
    public int BaseQIndex { get; init; }
    public int YDcDeltaQ { get; init; }
    public int UvDcDeltaQ { get; init; }
    public int UvAcDeltaQ { get; init; }
    public bool Lossless { get; init; }
    public required Vp9SegmentationInfo Segmentation { get; init; }
    /// <summary>Per segment (8; only [0] when segmentation is off): y dc, y ac, uv dc, uv ac.</summary>
    public required short[,] Dequant { get; init; }
    public int Log2TileCols { get; init; }
    public int Log2TileRows { get; init; }
    /// <summary>Compressed header size in bytes (0 for show_existing_frame).</summary>
    public int FirstPartitionSize { get; init; }
    /// <summary>Uncompressed header size in bytes.</summary>
    public int UncompressedHeaderBytes { get; init; }
    public bool UsePrevFrameMvs { get; init; }
    /// <summary>The buffer this frame decodes into (the shown buffer for show_existing_frame).</summary>
    public int NewBuffer { get; init; }
    /// <summary>Crop sizes of LAST, GOLDEN, ALTREF (inter frames).</summary>
    public (int Width, int Height)[] RefSizes { get; init; } = Array.Empty<(int, int)>();
}

/// <summary>
/// The host half of a VP9 decoder: reads each frame's uncompressed header and keeps the state libvpx keeps between
/// frames (reference map of frame buffers, loop filter deltas, segmentation parameters, last frame properties).
/// Call <see cref="ReadFrame"/> for every frame in decode order, decode it, then <see cref="FinishFrame"/>.
/// </summary>
public sealed class Vp9HeaderReader
{
    /// <summary>libvpx FRAME_BUFFERS (REF_FRAMES + 7 frames in flight in its frame-parallel design; plenty here).</summary>
    public const int FrameBuffers = 12;
    private const int RefFrames = 8, MaxQ = 255, MaxLoopFilter = 63;
    private static readonly int[] SegFeatureMax = { MaxQ, MaxLoopFilter, 3, 0 };
    private static readonly bool[] SegFeatureSigned = { true, true, false, false };

    /// <summary>Frame buffer properties the header logic needs (the pixels are on the accelerator).</summary>
    private sealed class Buffer
    {
        public int RefCount, Width, Height, SubsamplingX, SubsamplingY, BitDepth;
    }

    private readonly Buffer[] _buffers = Enumerable.Range(0, FrameBuffers).Select(_ => new Buffer()).ToArray();
    private readonly int[] _refMap = Enumerable.Repeat(-1, RefFrames).ToArray();
    private readonly int[] _nextRefMap = new int[RefFrames];
    private bool _needResync = true;
    private int _newBuffer = -1;
    private int _currentVideoFrame;
    // VP9_COMMON fields that persist between frames
    private int _profile, _bitDepth = 8, _colorSpace, _colorRange, _ssX = 1, _ssY = 1;
    private int _width, _height, _lastWidth, _lastHeight, _frameType, _lastFrameType, _miCols, _miRows;
    private bool _intraOnly, _lastIntraOnly, _showFrame, _lastShowFrame, _showExisting, _segEnabled;
    private int _refreshFrameFlags;
    private readonly int[] _signBias = new int[4];
    private int _lfLevel, _lfSharpness;
    private bool _lfDeltaEnabled, _lfDeltaUpdate;
    private readonly int[] _refDeltas = { 1, 0, -1, -1 };
    private readonly int[] _modeDeltas = new int[2];
    private bool _segUpdateMap, _segUpdateData, _segTemporal, _segAbs;
    // zero until a frame updates the map (libvpx's zero-allocated VP9_COMMON); only read after an update
    private readonly byte[] _segTree = new byte[7];
    private readonly byte[] _segPred = new byte[3];
    private readonly int[] _segMask = new int[8];
    private readonly int[,] _segData = new int[8, 4];

    /// <summary>The frame buffer each of the 8 reference slots holds (-1 = empty).</summary>
    public IReadOnlyList<int> ReferenceMap => _refMap;

    /// <summary>Read the uncompressed header of <paramref name="frame"/> (one frame, see <see cref="Vp9Packets.Split"/>).</summary>
    /// <exception cref="InvalidDataException">The header is invalid or refers to frames the decoder does not hold.</exception>
    public Vp9FrameInfo ReadFrame(ReadOnlySpan<byte> frame)
    {
        // vp9_receive_compressed_data: a free buffer for this frame
        _newBuffer = Array.FindIndex(_buffers, b => b.RefCount == 0);
        if (_newBuffer < 0) throw new InvalidDataException("VP9: no free frame buffer.");
        _buffers[_newBuffer].RefCount = 1;

        var rb = new BitBuffer(frame);
        _lastFrameType = _frameType;
        _lastIntraOnly = _intraOnly;
        if (rb.Literal(2) != 2) throw new InvalidDataException("VP9: invalid frame marker.");
        _profile = rb.Bit() | (rb.Bit() << 1);
        if (_profile > 2) _profile += rb.Bit();
        if (_profile >= 4) throw new InvalidDataException("VP9: unsupported bitstream profile.");

        _showExisting = rb.Bit() != 0;
        if (_showExisting)
        {
            int show = _refMap[rb.Literal(3)];
            if (show < 0 || _buffers[show].RefCount < 1) throw new InvalidDataException($"VP9: buffer {show} does not hold a decoded frame.");
            // ref_cnt_fb(frame_bufs, &new_fb_idx, frame_to_show): the new buffer becomes the shown one
            _buffers[_newBuffer].RefCount--;
            _newBuffer = show;
            _buffers[show].RefCount++;
            _refreshFrameFlags = 0;
            _lfLevel = 0;
            _showFrame = true;
            return new Vp9FrameInfo
            {
                Profile = _profile, BitDepth = _bitDepth, ShowExistingFrame = true, FrameToShowBuffer = show, ShowFrame = true,
                FrameType = _frameType, Width = _width, Height = _height, NewBuffer = show,
                UncompressedHeaderBytes = _profile <= 2 ? 1 : 2,
                LoopFilter = LoopFilterSnapshot(), Segmentation = SegmentationSnapshot(), Dequant = new short[1, 4],
            };
        }

        _frameType = rb.Bit();
        _showFrame = rb.Bit() != 0;
        bool errorRes = rb.Bit() != 0;
        int resetFrameContext = 0;
        int[] refBuffers = { -1, -1, -1 };
        bool allowHp = false;
        var interp = Vp9Interp.EightTap;
        if (_frameType == 0)
        {
            ReadSyncCode(ref rb);
            ReadBitDepthColorSpaceSampling(ref rb);
            _refreshFrameFlags = 0xFF;
            SetupFrameSize(ref rb);
            if (_needResync)
            {
                Array.Fill(_refMap, -1);
                _needResync = false;   // (flush_all_fb_on_key releases buffers libvpx no longer maps)
            }
        }
        else
        {
            _intraOnly = _showFrame ? false : rb.Bit() != 0;
            resetFrameContext = errorRes ? 0 : rb.Literal(2);
            if (_intraOnly)
            {
                ReadSyncCode(ref rb);
                if (_profile > 0) ReadBitDepthColorSpaceSampling(ref rb);
                else
                {
                    _colorSpace = 1;   // VPX_CS_BT_601
                    _colorRange = 0;   // studio
                    _ssX = _ssY = 1;
                    _bitDepth = 8;
                }
                _refreshFrameFlags = rb.Literal(RefFrames);
                SetupFrameSize(ref rb);
                if (_needResync)
                {
                    Array.Fill(_refMap, -1);
                    _needResync = false;
                }
            }
            else if (!_needResync)
            {
                _refreshFrameFlags = rb.Literal(RefFrames);
                for (int i = 0; i < 3; i++)
                {
                    int idx = _refMap[rb.Literal(3)];
                    if (idx < 0) throw new InvalidDataException("VP9: invalid reference frame index.");
                    refBuffers[i] = idx;
                    _signBias[1 + i] = rb.Bit();
                }
                SetupFrameSizeWithRefs(ref rb, refBuffers);
                allowHp = rb.Bit() != 0;
                interp = rb.Bit() != 0 ? Vp9Interp.Switchable
                    : rb.Literal(2) switch { 0 => Vp9Interp.EightTapSmooth, 1 => Vp9Interp.EightTap, 2 => Vp9Interp.EightTapSharp, _ => Vp9Interp.Bilinear };
            }
        }
        if (_needResync) throw new InvalidDataException("VP9: a key frame or intra-only frame is required to start decoding.");

        bool refreshCtx, frameParallel;
        if (!errorRes)
        {
            refreshCtx = rb.Bit() != 0;
            frameParallel = rb.Bit() != 0;
        }
        else
        {
            refreshCtx = false;
            frameParallel = true;
        }
        int ctxIdx = rb.Literal(2);

        // next_ref_frame_map (applied in FinishFrame)
        for (int i = 0; i < RefFrames; i++)
        {
            if (((_refreshFrameFlags >> i) & 1) != 0)
            {
                _nextRefMap[i] = _newBuffer;
                _buffers[_newBuffer].RefCount++;
            }
            else _nextRefMap[i] = _refMap[i];
            if (_refMap[i] >= 0) _buffers[_refMap[i]].RefCount++;   // held while decoding
        }

        bool intraOnlyFrame = _frameType == 0 || _intraOnly;
        bool pastIndependence = intraOnlyFrame || errorRes;
        var reset = Vp9ContextReset.None;
        if (pastIndependence)
        {
            // vp9_setup_past_independence (host half)
            Array.Clear(_segMask);
            Array.Clear(_segData);
            _segAbs = false;
            _lfDeltaEnabled = true;
            _lfDeltaUpdate = true;
            _refDeltas[0] = 1; _refDeltas[1] = 0; _refDeltas[2] = -1; _refDeltas[3] = -1;
            _modeDeltas[0] = 0; _modeDeltas[1] = 0;
            if (_frameType == 0 || errorRes || resetFrameContext == 3) reset = Vp9ContextReset.All;
            else if (resetFrameContext == 2) reset = Vp9ContextReset.One;   // the header's ctxIdx, before it is forced to 0
            Array.Clear(_signBias);
        }
        int resetIdx = ctxIdx;
        if (pastIndependence) ctxIdx = 0;

        // setup_loopfilter
        _lfLevel = rb.Literal(6);
        _lfSharpness = rb.Literal(3);
        _lfDeltaUpdate = false;
        _lfDeltaEnabled = rb.Bit() != 0;
        if (_lfDeltaEnabled)
        {
            _lfDeltaUpdate = rb.Bit() != 0;
            if (_lfDeltaUpdate)
            {
                for (int i = 0; i < 4; i++) if (rb.Bit() != 0) _refDeltas[i] = rb.SignedLiteral(6);
                for (int i = 0; i < 2; i++) if (rb.Bit() != 0) _modeDeltas[i] = rb.SignedLiteral(6);
            }
        }
        // setup_quantization
        int baseQ = rb.Literal(8);
        int yDc = DeltaQ(ref rb), uvDc = DeltaQ(ref rb), uvAc = DeltaQ(ref rb);
        bool lossless = baseQ == 0 && yDc == 0 && uvDc == 0 && uvAc == 0;
        // setup_segmentation
        _segUpdateMap = false;
        _segUpdateData = false;
        _segEnabled = rb.Bit() != 0;
        if (_segEnabled)
        {
            _segUpdateMap = rb.Bit() != 0;
            if (_segUpdateMap)
            {
                for (int i = 0; i < 7; i++) _segTree[i] = (byte)(rb.Bit() != 0 ? rb.Literal(8) : 255);
                _segTemporal = rb.Bit() != 0;
                for (int i = 0; i < 3; i++) _segPred[i] = (byte)(_segTemporal && rb.Bit() != 0 ? rb.Literal(8) : 255);
            }
            _segUpdateData = rb.Bit() != 0;
            if (_segUpdateData)
            {
                _segAbs = rb.Bit() != 0;
                Array.Clear(_segMask);
                Array.Clear(_segData);
                for (int s = 0; s < 8; s++)
                    for (int f = 0; f < 4; f++)
                    {
                        int data = 0;
                        if (rb.Bit() != 0)
                        {
                            _segMask[s] |= 1 << f;
                            int max = SegFeatureMax[f];
                            data = rb.Literal(max == 0 ? 0 : 32 - System.Numerics.BitOperations.LeadingZeroCount((uint)max));
                            if (data > max) data = max;
                            if (SegFeatureSigned[f] && rb.Bit() != 0) data = -data;
                        }
                        _segData[s, f] = data;
                    }
            }
        }
        // setup_segmentation_dequant
        int segments = _segEnabled ? 8 : 1;
        var dequant = new short[segments, 4];
        for (int s = 0; s < segments; s++)
        {
            int q = baseQ;
            if (_segEnabled && (_segMask[s] & 1) != 0)
                q = Math.Clamp(_segAbs ? _segData[s, 0] : baseQ + _segData[s, 0], 0, MaxQ);
            dequant[s, 0] = Vp9Dequantizer.DcQuant(q, yDc);
            dequant[s, 1] = Vp9Dequantizer.AcQuant(q, 0);
            dequant[s, 2] = Vp9Dequantizer.DcQuant(q, uvDc);
            dequant[s, 3] = Vp9Dequantizer.AcQuant(q, uvAc);
        }
        // setup_tile_info
        int sb64Cols = (_miCols + 7) >> 3;
        int minLog2 = 0;
        while ((64 << minLog2) < sb64Cols) minLog2++;
        int maxLog2 = 1;
        while ((sb64Cols >> maxLog2) >= 4) maxLog2++;
        maxLog2--;
        int log2TileCols = minLog2, maxOnes = maxLog2 - minLog2;
        while (maxOnes-- > 0 && rb.Bit() != 0) log2TileCols++;
        if (log2TileCols > 6) throw new InvalidDataException("VP9: invalid number of tile columns.");
        int log2TileRows = rb.Bit();
        if (log2TileRows != 0) log2TileRows += rb.Bit();
        int firstPartitionSize = rb.Literal(16);
        if (firstPartitionSize == 0) throw new InvalidDataException("VP9: invalid header size.");
        int headerBytes = rb.BytesRead;
        if (headerBytes + firstPartitionSize > frame.Length) throw new InvalidDataException("VP9: truncated packet or corrupt header length.");

        // vp9_decode_frame
        bool usePrevMvs = !errorRes && _width == _lastWidth && _height == _lastHeight && !_lastIntraOnly && _lastShowFrame && _lastFrameType != 0;

        var nb = _buffers[_newBuffer];
        nb.Width = _width; nb.Height = _height; nb.SubsamplingX = _ssX; nb.SubsamplingY = _ssY; nb.BitDepth = _bitDepth;
        return new Vp9FrameInfo
        {
            Profile = _profile, BitDepth = _bitDepth, FrameType = _frameType, LastFrameType = _lastFrameType, ShowFrame = _showFrame, ErrorResilient = errorRes,
            IntraOnly = _intraOnly, ResetFrameContext = resetFrameContext, ColorSpace = _colorSpace, ColorRange = _colorRange,
            SubsamplingX = _ssX, SubsamplingY = _ssY, RefreshFrameFlags = _refreshFrameFlags, RefBuffers = refBuffers,
            SignBias = (int[])_signBias.Clone(), Width = _width, Height = _height, RenderWidth = _renderWidth, RenderHeight = _renderHeight,
            MiCols = _miCols, MiRows = _miRows, AllowHighPrecisionMv = allowHp, InterpFilter = interp, RefreshFrameContext = refreshCtx,
            FrameParallel = frameParallel, FrameContextIdx = ctxIdx, PastIndependence = pastIndependence,
            ContextReset = reset, ContextResetIdx = resetIdx, LoopFilter = LoopFilterSnapshot(), BaseQIndex = baseQ, YDcDeltaQ = yDc,
            UvDcDeltaQ = uvDc, UvAcDeltaQ = uvAc, Lossless = lossless, Segmentation = SegmentationSnapshot(), Dequant = dequant,
            Log2TileCols = log2TileCols, Log2TileRows = log2TileRows, FirstPartitionSize = firstPartitionSize,
            UncompressedHeaderBytes = headerBytes, UsePrevFrameMvs = usePrevMvs, NewBuffer = _newBuffer,
            RefSizes = intraOnlyFrame ? Array.Empty<(int, int)>()
                : refBuffers.Select(b => (_buffers[b].Width, _buffers[b].Height)).ToArray(),
        };
    }

    /// <summary>After the frame is decoded: libvpx swap_frame_buffers + the end of vp9_receive_compressed_data.</summary>
    public void FinishFrame()
    {
        int i = 0;
        for (int mask = _refreshFrameFlags; mask != 0; mask >>= 1, i++)
        {
            int old = _refMap[i];
            Release(old);   // the hold taken while decoding
            if ((mask & 1) != 0) Release(old);
            _refMap[i] = _nextRefMap[i];
        }
        for (; i < RefFrames && !_showExisting; i++)
        {
            Release(_refMap[i]);
            _refMap[i] = _nextRefMap[i];
        }
        _buffers[_newBuffer].RefCount--;
        if (!_showExisting) _lastShowFrame = _showFrame;
        _lastWidth = _width;
        _lastHeight = _height;
        if (_showFrame) _currentVideoFrame++;
    }

    private void Release(int buffer)
    {
        if (buffer >= 0 && _buffers[buffer].RefCount > 0) _buffers[buffer].RefCount--;
    }

    private int _renderWidth, _renderHeight;

    private static void ReadSyncCode(ref BitBuffer rb)
    {
        if (rb.Literal(8) != 0x49 || rb.Literal(8) != 0x83 || rb.Literal(8) != 0x42)
            throw new InvalidDataException("VP9: invalid frame sync code.");
    }

    private void ReadBitDepthColorSpaceSampling(ref BitBuffer rb)
    {
        _bitDepth = _profile >= 2 ? (rb.Bit() != 0 ? 12 : 10) : 8;
        _colorSpace = rb.Literal(3);
        if (_colorSpace != 7)   // VPX_CS_SRGB
        {
            _colorRange = rb.Bit();
            if (_profile == 1 || _profile == 3)
            {
                _ssX = rb.Bit();
                _ssY = rb.Bit();
                if (_ssX == 1 && _ssY == 1) throw new InvalidDataException("VP9: 4:2:0 color not supported in profile 1 or 3.");
                if (rb.Bit() != 0) throw new InvalidDataException("VP9: reserved bit set.");
            }
            else _ssX = _ssY = 1;
        }
        else
        {
            _colorRange = 1;
            if (_profile == 1 || _profile == 3)
            {
                _ssX = _ssY = 0;
                if (rb.Bit() != 0) throw new InvalidDataException("VP9: reserved bit set.");
            }
            else throw new InvalidDataException("VP9: 4:4:4 color not supported in profile 0 or 2.");
        }
    }

    private void SetupFrameSize(ref BitBuffer rb)
    {
        int w = rb.Literal(16) + 1, h = rb.Literal(16) + 1;
        ResizeContext(w, h);
        SetupRenderSize(ref rb);
    }

    private void SetupFrameSizeWithRefs(ref BitBuffer rb, int[] refBuffers)
    {
        int w = 0, h = 0;
        bool found = false;
        for (int i = 0; i < 3; i++)
        {
            if (rb.Bit() != 0)
            {
                w = _buffers[refBuffers[i]].Width;
                h = _buffers[refBuffers[i]].Height;
                found = true;
                break;
            }
        }
        if (!found)
        {
            w = rb.Literal(16) + 1;
            h = rb.Literal(16) + 1;
        }
        if (w <= 0 || h <= 0) throw new InvalidDataException("VP9: invalid frame size.");
        bool anyValid = false;
        for (int i = 0; i < 3; i++)
        {
            var b = _buffers[refBuffers[i]];
            // valid_ref_frame_size: 2x downscaling to 16x upscaling
            anyValid |= 2 * w >= b.Width && 2 * h >= b.Height && w <= 16 * b.Width && h <= 16 * b.Height;
        }
        if (!anyValid) throw new InvalidDataException("VP9: referenced frame has invalid size.");
        for (int i = 0; i < 3; i++)
        {
            var b = _buffers[refBuffers[i]];
            if (b.BitDepth != _bitDepth || b.SubsamplingX != _ssX || b.SubsamplingY != _ssY)
                throw new InvalidDataException("VP9: referenced frame has incompatible color format.");
        }
        ResizeContext(w, h);
        SetupRenderSize(ref rb);
    }

    private void ResizeContext(int w, int h)
    {
        _width = w;
        _height = h;
        _miCols = (w + 7) >> 3;
        _miRows = (h + 7) >> 3;
    }

    private void SetupRenderSize(ref BitBuffer rb)
    {
        _renderWidth = _width;
        _renderHeight = _height;
        if (rb.Bit() != 0)
        {
            _renderWidth = rb.Literal(16) + 1;
            _renderHeight = rb.Literal(16) + 1;
        }
    }

    private static int DeltaQ(ref BitBuffer rb) => rb.Bit() != 0 ? rb.SignedLiteral(4) : 0;

    private Vp9LoopFilterInfo LoopFilterSnapshot() =>
        new(_lfLevel, _lfSharpness, _lfDeltaEnabled, _lfDeltaUpdate, (int[])_refDeltas.Clone(), (int[])_modeDeltas.Clone());

    private Vp9SegmentationInfo SegmentationSnapshot() =>
        new(_segEnabled, _segUpdateMap, _segUpdateData, _segTemporal, _segAbs, (byte[])_segTree.Clone(), (byte[])_segPred.Clone(),
            (int[])_segMask.Clone(), (int[,])_segData.Clone());

    /// <summary>libvpx vpx_read_bit_buffer: MSB-first bits; reading past the end is an error.</summary>
    private ref struct BitBuffer
    {
        private readonly ReadOnlySpan<byte> _data;
        private int _bit;

        public BitBuffer(ReadOnlySpan<byte> data) { _data = data; _bit = 0; }

        public readonly int BytesRead => (_bit + 7) >> 3;

        public int Bit()
        {
            int p = _bit >> 3;
            if (p >= _data.Length) throw new InvalidDataException("VP9: truncated frame header.");
            int b = (_data[p] >> (7 - (_bit & 7))) & 1;
            _bit++;
            return b;
        }

        public int Literal(int bits)
        {
            int v = 0;
            for (int i = bits - 1; i >= 0; i--) v |= Bit() << i;
            return v;
        }

        public int SignedLiteral(int bits)
        {
            int v = Literal(bits);
            return Bit() != 0 ? -v : v;
        }
    }
}
