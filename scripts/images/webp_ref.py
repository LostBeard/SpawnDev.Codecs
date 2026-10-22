"""
Reference VP8L (WebP lossless) bitstream decoder + RIFF helpers, written straight from the WebP Lossless
Bitstream Specification (RFC 9649). It is a TEST TOOL, not part of the library:

  * make_webp_fixtures.py uses it to REPORT AND ASSERT which bitstream features the fixture files exercise
    (every transform, all 14 predictor modes, colour cache, meta prefix codes, LZ77 with 2-D distance codes,
    pixel bundling widths, simple and normal prefix codes ...), so a fixture set cannot silently miss a feature.
  * It is cross-checked against libwebp (through Pillow) on every fixture, so it is only believed where
    libwebp agrees. The truth the unit tests compare against is always libwebp's output / the source samples,
    never this decoder and never the library's own code.

Pure Python, slow on purpose-simple code: fine for the fixture sizes (the 1024x768 file takes a while).
"""
import struct

# (dx, dy) of the 120 short distance codes, RFC 9649 section 5.2.2.
PLANE_CODES = [
    (0, 1), (1, 0), (1, 1), (-1, 1), (0, 2), (2, 0), (1, 2), (-1, 2), (2, 1), (-2, 1), (2, 2), (-2, 2),
    (0, 3), (3, 0), (1, 3), (-1, 3), (3, 1), (-3, 1), (2, 3), (-2, 3), (3, 2), (-3, 2), (0, 4), (4, 0),
    (1, 4), (-1, 4), (4, 1), (-4, 1), (3, 3), (-3, 3), (2, 4), (-2, 4), (4, 2), (-4, 2), (0, 5), (3, 4),
    (-3, 4), (4, 3), (-4, 3), (5, 0), (1, 5), (-1, 5), (5, 1), (-5, 1), (2, 5), (-2, 5), (5, 2), (-5, 2),
    (4, 4), (-4, 4), (3, 5), (-3, 5), (5, 3), (-5, 3), (0, 6), (6, 0), (1, 6), (-1, 6), (6, 1), (-6, 1),
    (2, 6), (-2, 6), (6, 2), (-6, 2), (4, 5), (-4, 5), (5, 4), (-5, 4), (3, 6), (-3, 6), (6, 3), (-6, 3),
    (0, 7), (7, 0), (1, 7), (-1, 7), (5, 5), (-5, 5), (7, 1), (-7, 1), (4, 6), (-4, 6), (6, 4), (-6, 4),
    (2, 7), (-2, 7), (7, 2), (-7, 2), (3, 7), (-3, 7), (7, 3), (-7, 3), (5, 6), (-5, 6), (6, 5), (-6, 5),
    (8, 0), (4, 7), (-4, 7), (7, 4), (-7, 4), (8, 1), (8, 2), (6, 6), (-6, 6), (8, 3), (5, 7), (-5, 7),
    (7, 5), (-7, 5), (8, 4), (6, 7), (-6, 7), (7, 6), (-7, 6), (8, 5), (7, 7), (-7, 7), (8, 6), (8, 7),
]
CODE_LENGTH_ORDER = [17, 18, 0, 1, 2, 3, 4, 5, 16, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
NUM_LENGTH_CODES = 24
NUM_DIST_CODES = 40


class VP8LError(Exception):
    pass


class BitReader:
    def __init__(self, data):
        self.data = data
        self.pos = 0          # bit position
        self.nbits = len(data) * 8

    def read(self, n):
        if n == 0:
            return 0
        if self.pos + n > self.nbits:
            raise VP8LError("truncated")
        v = 0
        for i in range(n):
            p = self.pos + i
            v |= ((self.data[p >> 3] >> (p & 7)) & 1) << i
        self.pos += n
        return v


class Huffman:
    """Canonical prefix code. 'single' is set for a one-symbol code (zero bits per symbol)."""

    def __init__(self, lengths):
        used = [(l, s) for s, l in enumerate(lengths) if l > 0]
        if not used:
            raise VP8LError("empty prefix code")
        self.single = None
        self.count = [0] * 16
        if len(used) == 1:
            self.single = used[0][1]
            return
        for l, _ in used:
            self.count[l] += 1
        left = 1
        for l in range(1, 16):
            left = (left << 1) - self.count[l]
            if left < 0:
                raise VP8LError("over-subscribed prefix code")
        if left != 0:
            raise VP8LError("incomplete prefix code")
        used.sort()
        self.syms = [s for _, s in used]

    def read(self, br):
        if self.single is not None:
            return self.single
        code = first = index = 0
        for l in range(1, 16):
            code |= br.read(1)
            c = self.count[l]
            if code - c < first:
                return self.syms[index + (code - first)]
            index += c
            first = (first + c) << 1
            code <<= 1
        raise VP8LError("bad prefix code symbol")


def read_prefix_code(br, alphabet):
    if br.read(1):                                       # simple code lengths
        nsym = br.read(1) + 1
        first8 = br.read(1)
        s0 = br.read(8 if first8 else 1)
        lengths = [0] * alphabet
        lengths[s0] = 1
        if nsym == 2:
            s1 = br.read(8)
            lengths[s1] = 1
        return Huffman(lengths), "simple%d" % nsym
    ncl = br.read(4) + 4
    cl = [0] * 19
    for i in range(ncl):
        cl[CODE_LENGTH_ORDER[i]] = br.read(3)
    clh = Huffman(cl)
    if br.read(1):
        nbits = 2 + 2 * br.read(3)
        max_symbol = 2 + br.read(nbits)
    else:
        max_symbol = alphabet
    if max_symbol > alphabet:
        raise VP8LError("bad max_symbol")
    lengths = [0] * alphabet
    sym = 0
    prev = 8
    while sym < alphabet:
        if max_symbol == 0:
            break
        max_symbol -= 1
        c = clh.read(br)
        if c < 16:
            lengths[sym] = c
            sym += 1
            if c:
                prev = c
        else:
            extra, off = {16: (2, 3), 17: (3, 3), 18: (7, 11)}[c]
            rep = br.read(extra) + off
            if sym + rep > alphabet:
                raise VP8LError("bad repeat")
            v = prev if c == 16 else 0
            for _ in range(rep):
                lengths[sym] = v
                sym += 1
    return Huffman(lengths), "normal"


def ceil_div(a, b):
    return (a + b - 1) // b


def get_copy(br, sym):
    if sym < 4:
        return sym + 1
    extra = (sym - 2) >> 1
    off = (2 + (sym & 1)) << extra
    return off + br.read(extra) + 1


def hash_color(argb, bits):
    return ((0x1E35A7BD * argb) & 0xFFFFFFFF) >> (32 - bits)


class Stats:
    def __init__(self):
        self.transforms = []        # (kind, bits) in stream order
        self.cache_bits = 0
        self.meta_bits = 0
        self.groups = 0
        self.groups_used = 0
        self.literals = 0
        self.cache_hits = 0
        self.copies = 0
        self.plane_codes = set()    # short distance codes (1..120) used
        self.far_dists = 0
        self.predictor_modes = set()
        self.code_kinds = set()     # simple1 / simple2 / normal
        self.palette_size = 0
        self.bundle_bits = 0        # width_bits of colour indexing (0..3)
        self.alpha_is_used = 0
        self.has_overlap_copy = False
        self.cross_tiles = 0        # cross-colour transform tiles
        self.cross_nonzero = 0      # of those, tiles with at least one non-zero multiplier
    def summary(self):
        t = ",".join("%s%d" % (k, b) if b is not None else k for k, b in self.transforms) or "none"
        return ("xf[%s] cache=%d meta=%d groups=%d/%d lit=%d cache_hits=%d copies=%d short=%d far=%d "
                "pred_modes=%s pal=%d bundle=%d codes=%s cc_nz=%d/%d") % (
            t, self.cache_bits, self.meta_bits, self.groups_used, self.groups, self.literals, self.cache_hits,
            self.copies, len(self.plane_codes), self.far_dists, "".join("%x" % m for m in sorted(self.predictor_modes)),
            self.palette_size, self.bundle_bits, "/".join(sorted(self.code_kinds)), self.cross_nonzero, self.cross_tiles)


def _decode_stream(br, xsize, ysize, level0, st):
    transforms = []
    if level0:
        seen = set()
        while br.read(1):
            t = br.read(2)
            if t in seen:
                raise VP8LError("transform used twice")
            seen.add(t)
            if t == 0 or t == 1:
                bits = br.read(3) + 2
                data = _decode_stream(br, ceil_div(xsize, 1 << bits), ceil_div(ysize, 1 << bits), False, st)
                transforms.append((t, bits, data, xsize))
                st.transforms.append(("P" if t == 0 else "C", bits))
            elif t == 2:
                transforms.append((2, 0, None, xsize))
                st.transforms.append(("G", None))
            else:
                n = br.read(8) + 1
                tbl = _decode_stream(br, n, 1, False, st)
                pal = []
                prev = 0
                for c in tbl:
                    v = 0
                    for sh in (0, 8, 16, 24):
                        v |= ((((prev >> sh) & 255) + ((c >> sh) & 255)) & 255) << sh
                    pal.append(v)
                    prev = v
                wb = 3 if n <= 2 else 2 if n <= 4 else 1 if n <= 16 else 0
                transforms.append((3, wb, pal, xsize))
                st.transforms.append(("I", wb))
                st.palette_size = n
                st.bundle_bits = wb
                xsize = ceil_div(xsize, 1 << wb)
    cache_bits = 0
    if br.read(1):
        cache_bits = br.read(4)
        if cache_bits < 1 or cache_bits > 11:
            raise VP8LError("bad colour cache size")
    if level0:
        st.cache_bits = cache_bits
    meta = None
    meta_bits = 0
    num_groups = 1
    if level0 and br.read(1):
        meta_bits = br.read(3) + 2
        meta = _decode_stream(br, ceil_div(xsize, 1 << meta_bits), ceil_div(ysize, 1 << meta_bits), False, st)
        meta = [(m >> 8) & 0xFFFF for m in meta]
        num_groups = max(meta) + 1
        st.meta_bits = meta_bits
    if level0:
        st.groups = num_groups
        st.groups_used = len(set(meta)) if meta else 1
    cache_size = (1 << cache_bits) if cache_bits else 0
    groups = []
    for _ in range(num_groups):
        g = []
        for i, alpha in enumerate((256 + NUM_LENGTH_CODES + cache_size, 256, 256, 256, NUM_DIST_CODES)):
            h, kind = read_prefix_code(br, alpha)
            g.append(h)
            if level0:
                st.code_kinds.add(kind)
        groups.append(g)
    n = xsize * ysize
    out = [0] * n
    cache = [0] * cache_size if cache_size else None
    pos = 0
    tiles_per_row = ceil_div(xsize, 1 << meta_bits) if meta else 0
    while pos < n:
        y, x = divmod(pos, xsize)
        g = groups[meta[(y >> meta_bits) * tiles_per_row + (x >> meta_bits)]] if meta else groups[0]
        s = g[0].read(br)
        if s < 256:
            r = g[1].read(br)
            b = g[2].read(br)
            a = g[3].read(br)
            px = (a << 24) | (r << 16) | (s << 8) | b
            out[pos] = px
            if cache is not None:
                cache[hash_color(px, cache_bits)] = px
            pos += 1
            if level0:
                st.literals += 1
        elif s < 256 + NUM_LENGTH_CODES:
            length = get_copy(br, s - 256)
            dsym = g[4].read(br)
            dcode = get_copy(br, dsym)
            if dcode > 120:
                dist = dcode - 120
                if level0:
                    st.far_dists += 1
            else:
                dx, dy = PLANE_CODES[dcode - 1]
                dist = max(1, dx + dy * xsize)
                if level0:
                    st.plane_codes.add(dcode)
            if dist > pos or length > n - pos:
                raise VP8LError("bad backward reference")
            if level0:
                st.copies += 1
                if dist < length:
                    st.has_overlap_copy = True
            for _ in range(length):
                px = out[pos - dist]
                out[pos] = px
                if cache is not None:
                    cache[hash_color(px, cache_bits)] = px
                pos += 1
        else:
            idx = s - 256 - NUM_LENGTH_CODES
            if cache is None or idx >= cache_size:
                raise VP8LError("bad colour cache index")
            px = cache[idx]
            out[pos] = px
            cache[hash_color(px, cache_bits)] = px
            pos += 1
            if level0:
                st.cache_hits += 1
    if not level0:
        return out
    return out, transforms, xsize


def _clip(v):
    return 0 if v < 0 else 255 if v > 255 else v


def _avg2(a, b):
    r = 0
    for sh in (0, 8, 16, 24):
        r |= ((((a >> sh) & 255) + ((b >> sh) & 255)) >> 1) << sh
    return r


def _select(l, t, tl):
    pl = pt = 0
    for sh in (0, 8, 16, 24):
        L = (l >> sh) & 255
        T = (t >> sh) & 255
        TL = (tl >> sh) & 255
        p = L + T - TL
        pl += abs(p - L)
        pt += abs(p - T)
    return l if pl < pt else t


def _clamp_full(a, b, c):
    r = 0
    for sh in (0, 8, 16, 24):
        r |= _clip(((a >> sh) & 255) + ((b >> sh) & 255) - ((c >> sh) & 255)) << sh
    return r


def _clamp_half(a, b):
    r = 0
    for sh in (0, 8, 16, 24):
        A = (a >> sh) & 255
        B = (b >> sh) & 255
        d = A - B
        half = d // 2 if d >= 0 else -((-d) // 2)
        r |= _clip(A + half) << sh
    return r


def _predict(mode, l, t, tr, tl):
    if mode == 0 or mode > 13:
        return 0xFF000000
    if mode == 1: return l
    if mode == 2: return t
    if mode == 3: return tr
    if mode == 4: return tl
    if mode == 5: return _avg2(_avg2(l, tr), t)
    if mode == 6: return _avg2(l, tl)
    if mode == 7: return _avg2(l, t)
    if mode == 8: return _avg2(tl, t)
    if mode == 9: return _avg2(t, tr)
    if mode == 10: return _avg2(_avg2(l, tl), _avg2(t, tr))
    if mode == 11: return _select(l, t, tl)
    if mode == 12: return _clamp_full(l, t, tl)
    return _clamp_half(_avg2(l, t), tl)


def _add(a, b):
    r = 0
    for sh in (0, 8, 16, 24):
        r |= ((((a >> sh) & 255) + ((b >> sh) & 255)) & 255) << sh
    return r


def _to_s8(v):
    return v - 256 if v >= 128 else v


def _inverse(tr, pix, xsize, ysize, st):
    kind, bits, data, _ = tr
    if kind == 2:
        out = []
        for p in pix:
            g = (p >> 8) & 255
            r = (((p >> 16) & 255) + g) & 255
            b = ((p & 255) + g) & 255
            out.append((p & 0xFF00FF00) | (r << 16) | b)
        return out
    if kind == 1:
        tw = ceil_div(xsize, 1 << bits)
        st.cross_tiles += len(data)
        st.cross_nonzero += sum(1 for m in data if (m & 0xFFFFFF) != 0)
        out = []
        for i, p in enumerate(pix):
            y, x = divmod(i, xsize)
            m = data[(y >> bits) * tw + (x >> bits)]
            g2r = _to_s8(m & 255)
            g2b = _to_s8((m >> 8) & 255)
            r2b = _to_s8((m >> 16) & 255)
            green = _to_s8((p >> 8) & 255)
            red = (p >> 16) & 255
            nr = (red + ((g2r * green) >> 5)) & 255
            nb = ((p & 255) + ((g2b * green) >> 5) + ((r2b * _to_s8(nr)) >> 5)) & 255
            out.append((p & 0xFF00FF00) | (nr << 16) | nb)
        return out
    if kind == 0:
        tw = ceil_div(xsize, 1 << bits)
        out = [0] * len(pix)
        for y in range(ysize):
            for x in range(xsize):
                i = y * xsize + x
                if y == 0:
                    pred = 0xFF000000 if x == 0 else out[i - 1]
                elif x == 0:
                    pred = out[i - xsize]
                else:
                    mode = (data[(y >> bits) * tw + (x >> bits)] >> 8) & 15
                    st.predictor_modes.add(mode)
                    pred = _predict(mode, out[i - 1], out[i - xsize], out[i - xsize + 1], out[i - xsize - 1])
                out[i] = _add(pix[i], pred)
        return out
    raise VP8LError("bad transform")


def decode_vp8l(payload, want_stats=True):
    """payload = the VP8L chunk data. Returns (width, height, argb_list, Stats)."""
    br = BitReader(payload)
    if br.read(8) != 0x2F:
        raise VP8LError("bad signature")
    w = br.read(14) + 1
    h = br.read(14) + 1
    st = Stats()
    st.alpha_is_used = br.read(1)
    if br.read(3) != 0:
        raise VP8LError("bad version")
    pix, transforms, xs = _decode_stream(br, w, h, True, st)
    # transforms[k] records the xsize in effect when it was read; undo in reverse order
    for tr in reversed(transforms):
        kind, bits, data, tx = tr
        if kind == 3:
            pal = data
            ppb = 8 >> bits
            mask = (1 << bits) - 1
            out = []
            for y in range(h):
                for x in range(w):
                    p = pix[y * xs + (x >> bits)]
                    idx = (p >> 8) & 255
                    if bits:
                        idx = (idx >> ((x & mask) * ppb)) & ((1 << ppb) - 1)
                    out.append(pal[idx] if idx < len(pal) else 0)
            pix = out
            xs = w
        else:
            pix = _inverse(tr, pix, tx, h, st)
    return w, h, pix, st


# ---------------------------------------------------------------------------------------------------
# RIFF helpers
# ---------------------------------------------------------------------------------------------------

def riff_chunks(data):
    """Yield (fourcc, payload) of the top-level chunks of a RIFF/WEBP file."""
    assert data[0:4] == b"RIFF" and data[8:12] == b"WEBP"
    pos = 12
    while pos + 8 <= len(data):
        fcc = data[pos:pos + 4]
        size = struct.unpack("<I", data[pos + 4:pos + 8])[0]
        yield fcc, data[pos + 8:pos + 8 + size]
        pos += 8 + size + (size & 1)


def make_chunk(fcc, payload):
    out = fcc + struct.pack("<I", len(payload)) + payload
    if len(payload) & 1:
        out += b"\0"
    return out


def make_riff(chunks):
    body = b"WEBP" + b"".join(chunks)
    return b"RIFF" + struct.pack("<I", len(body)) + body
