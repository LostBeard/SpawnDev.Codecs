// A PSD reader for the tests only, written from Adobe's published Photoshop File Formats Specification (no GPL / LGPL reader was read). It reads what the writer writes - and,
// checked by hand, ComfyUI's compositor export (2026-11-25): header, layer records (rectangle, channels, blend key, opacity, flags, names), RLE channel data and the merged image.
// The writer is never its own judge: PackBits is decoded here, row by row, against the row byte counts of the file.

using System.Text;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

internal sealed class PsdTestReader
{
    internal sealed record Layer(int Top, int Left, int Bottom, int Right, string Key, int Opacity, int Flags, string PascalName, string? UnicodeName,
        byte[] Alpha, byte[] Red, byte[] Green, byte[] Blue, int[] ChannelIds)
    {
        public int Width => Right - Left;
        public int Height => Bottom - Top;
        public byte[] Rgba()
        {
            var o = new byte[Width * Height * 4];
            for (int i = 0; i < Width * Height; i++) { o[i * 4] = Red[i]; o[i * 4 + 1] = Green[i]; o[i * 4 + 2] = Blue[i]; o[i * 4 + 3] = Alpha[i]; }
            return o;
        }
    }

    public int Version, Channels, Width, Height, Depth, Mode, LayerCount;
    public List<Layer> Layers = new();
    /// <summary>The merged image: Width x Height, planes red, green, blue and (4 channels) alpha.</summary>
    public byte[][] Composite = System.Array.Empty<byte[]>();

    private readonly byte[] _d;
    private int _p;

    private PsdTestReader(byte[] d) { _d = d; }

    public static PsdTestReader Read(byte[] file)
    {
        var r = new PsdTestReader(file);
        r.Parse();
        return r;
    }

    private byte U8() => _d[_p++];
    private int U16() { int v = (_d[_p] << 8) | _d[_p + 1]; _p += 2; return v; }
    private int I16() { int v = (short)U16(); return v; }
    private long U32() { long v = ((long)U16() << 16) | (uint)U16(); return v; }
    private int I32() => (int)U32();
    private string Ascii(int n) { string s = Encoding.ASCII.GetString(_d, _p, n); _p += n; return s; }

    private void Parse()
    {
        Check(Ascii(4) == "8BPS", "signature");
        Version = U16();
        _p += 6;
        Channels = U16();
        Height = I32(); Width = I32();
        Depth = U16(); Mode = U16();
        long colorData = U32(); _p += (int)colorData;   // color mode data
        long res = U32();
        long resEnd = _p + res;
        while (_p + 12 <= resEnd)
        {
            Check(Ascii(4) == "8BIM", "resource signature");
            U16();
            int nameLength = U8();
            _p += nameLength;
            if ((1 + nameLength) % 2 != 0) _p++;
            long size = U32();
            _p += (int)size + (int)(size & 1);
        }
        Check(_p == resEnd, "image resources end");
        long lmLength = U32();
        long lmEnd = _p + lmLength;
        if (lmLength > 0)
        {
            long liLength = U32();
            long liEnd = _p + liLength;
            Check((liLength & 1) == 0, "the layer info length is even");
            LayerCount = I16();
            ReadLayers(System.Math.Abs(LayerCount), liEnd);
            Check(_p <= liEnd && liEnd - _p <= 1, "layer info ends where its length says");
            _p = (int)liEnd;
            long globalMask = U32();
            _p += (int)globalMask;
            Check(_p == lmEnd, $"layer and mask info ends at its length (at {_p}, {lmEnd})");
        }
        _p = (int)lmEnd;
        int compression = U16();
        Check(compression == 1, "the merged image is PackBits (1)");
        Composite = new byte[Channels][];
        var rowLengths = new int[Channels * Height];
        for (int i = 0; i < rowLengths.Length; i++) rowLengths[i] = U16();
        for (int c = 0, k = 0; c < Channels; c++)
        {
            Composite[c] = new byte[Width * Height];
            for (int y = 0; y < Height; y++, k++) UnpackRow(rowLengths[k], Composite[c].AsSpan(y * Width, Width));
        }
        Check(_p == _d.Length, $"the file ends with the merged image (at {_p} of {_d.Length})");
    }

    private void ReadLayers(int count, long end)
    {
        var infos = new List<(int Top, int Left, int Bottom, int Right, List<(int Id, long Length)> Chs, string Key, int Opacity, int Flags, string Pascal, string? Unicode)>();
        for (int i = 0; i < count; i++)
        {
            int top = I32(), left = I32(), bottom = I32(), right = I32();
            int nc = U16();
            var chs = new List<(int, long)>();
            for (int c = 0; c < nc; c++) { int id = I16(); chs.Add((id, U32())); }
            Check(Ascii(4) == "8BIM", "blend signature");
            string key = Ascii(4);
            int opacity = U8(), clipping = U8(), flags = U8(), filler = U8();
            Check(clipping == 0 && filler == 0, "clipping and filler are 0");
            long extra = U32();
            long extraEnd = _p + extra;
            long mask = U32(); _p += (int)mask;
            long ranges = U32(); _p += (int)ranges;
            int n = U8();
            string pascal = Encoding.ASCII.GetString(_d, _p, n);
            _p += n;
            _p += ((1 + n) % 4 == 0) ? 0 : 4 - (1 + n) % 4;
            string? unicode = null;
            while (_p + 12 <= extraEnd)
            {
                Check(Ascii(4) == "8BIM", "additional layer info signature");
                string k = Ascii(4);
                int len = I32();
                int body = _p;
                if (k == "luni")
                {
                    int chars = (_d[body] << 24) | (_d[body + 1] << 16) | (_d[body + 2] << 8) | _d[body + 3];
                    unicode = Encoding.BigEndianUnicode.GetString(_d, body + 4, chars * 2);
                }
                Check((len & 1) == 0, "additional layer info length is even");
                _p = body + len;
            }
            Check(_p == extraEnd, $"the extra data of layer {i} ends at its length");
            infos.Add((top, left, bottom, right, chs, key, opacity, flags, pascal, unicode));
        }
        for (int i = 0; i < count; i++)
        {
            var l = infos[i];
            int w = l.Right - l.Left, h = l.Bottom - l.Top;
            var planes = new Dictionary<int, byte[]>();
            foreach (var (id, length) in l.Chs)
            {
                int start = _p;
                int compression = U16();
                Check(compression == 1, "layer channel compression is 1");
                var lengths = new int[h];
                for (int y = 0; y < h; y++) lengths[y] = U16();
                var data = new byte[w * h];
                for (int y = 0; y < h; y++) UnpackRow(lengths[y], data.AsSpan(y * w, w));
                Check(_p - start == length, $"layer {i} channel {id}: length {length} in the record, {_p - start} in the file");
                planes[id] = data;
            }
            byte[] Plane(int id) => planes.TryGetValue(id, out var v) ? v : new byte[w * h];
            Layers.Add(new Layer(l.Top, l.Left, l.Bottom, l.Right, l.Key, l.Opacity, l.Flags, l.Pascal, l.Unicode,
                Plane(-1), Plane(0), Plane(1), Plane(2), l.Chs.Select(c => c.Id).ToArray()));
        }
        Check(end - _p <= 1, "channel data ends at the layer info length");
    }

    /// <summary>Decode one PackBits row of <paramref name="packed"/> bytes into exactly <paramref name="dst"/>.Length bytes; nothing may be left over on either side.</summary>
    private void UnpackRow(int packed, Span<byte> dst)
    {
        int end = _p + packed;
        int o = 0;
        while (_p < end)
        {
            sbyte control = (sbyte)U8();
            if (control >= 0)
            {
                int n = control + 1;
                Check(o + n <= dst.Length && _p + n <= end, "a literal fits the row");
                for (int i = 0; i < n; i++) dst[o + i] = _d[_p + i];
                o += n; _p += n;
            }
            else if (control != -128)
            {
                int n = 1 - control;
                Check(o + n <= dst.Length && _p < end, "a run fits the row");
                byte v = U8();
                for (int i = 0; i < n; i++) dst[o + i] = v;
                o += n;
            }
        }
        Check(_p == end && o == dst.Length, $"the packed row has exactly {dst.Length} pixels ({o}) and uses exactly its {packed} bytes");
    }

    private static void Check(bool ok, string what)
    {
        if (!ok) throw new InvalidDataException("PSD: " + what);
    }
}
