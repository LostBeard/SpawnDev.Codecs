// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Meta prefix codes for the VP8L encoder: instead of one set of five prefix codes for the whole image, the image is cut
// into square tiles (2^bits pixels), each tile is assigned to one of a few groups, and every group has its own code set
// built from the tokens of its tiles. Regions with different statistics (flat sky next to foliage) then cost fewer bits.
// The tile -> group map is written as the "entropy image" (group index in red + green) in front of the codes, and a token
// is coded with the group of the tile its FIRST pixel lies in - the same rule the decoder applies.
//
// Both steps run as one GPU thread (they walk the token stream / do a small k-means over per-tile histograms, a
// dependency chain over little data); the code construction afterwards is one thread per code, in parallel.

using ILGPU;

namespace SpawnDev.Codecs.Images.WebP;

/// <summary>GPU-callable meta-prefix-code builders.</summary>
public static class Vp8lEncodeGroups
{
    /// <summary>Most groups the encoder forms.</summary>
    public const int MaxGroups = 8;
    /// <summary>Largest number of tiles.</summary>
    public const int MaxTiles = 1024;

    private const int Inf = 0x3FFFFFFF;

    /// <summary>Tile size bits for an image: at least 16 pixels square, at most <see cref="MaxTiles"/> tiles.</summary>
    public static int TileBitsFor(int width, int height)
    {
        int bits = 4;
        while (bits < 9 && Vp8lEncLayout.Tiles(width, height, bits) > MaxTiles) bits++;
        return bits;
    }

    /// <summary>Histogram offset of the cost table inside the misc region (after the length estimates).</summary>
    public static int CostTableOffset => MaxGroups * Vp8lEncLayout.SetAlphaMax;

    private static int BitLength(int v)
    {
        int n = 0;
        int t = v;
        while (t > 0)
        {
            n++;
            t >>= 1;
        }
        return n;
    }

    /// <summary>Per-tile histograms of the five codes, from the main token stream.</summary>
    public static void TileHistograms(ArrayView<int> w, ArrayView<int> tok, ArrayView<int> meta, ArrayView<int> tables)
    {
        int cacheBits = meta[WebPEncMeta.CacheBits];
        int cacheSize = cacheBits > 0 ? 1 << cacheBits : 0;
        int setAlpha = Vp8lEncodeGpu.SetAlpha(cacheSize);
        int xsize = meta[WebPEncMeta.PackedWidth];
        int height = meta[WebPEncMeta.Height];
        int bits = meta[WebPEncMeta.MetaBits];
        int tilesPerRow = (xsize + (1 << bits) - 1) >> bits;
        int tiles = tilesPerRow * ((height + (1 << bits) - 1) >> bits);
        int th = meta[WebPEncMeta.RegionTileHist];
        int total = tiles * setAlpha;
        for (int i = 0; i < total; i++) w[th + i] = 0;

        int gOff = Vp8lEncodeGpu.AlphaOff(0, cacheSize);
        int rOff = Vp8lEncodeGpu.AlphaOff(1, cacheSize);
        int bOff = Vp8lEncodeGpu.AlphaOff(2, cacheSize);
        int aOff = Vp8lEncodeGpu.AlphaOff(3, cacheSize);
        int dOff = Vp8lEncodeGpu.AlphaOff(4, cacheSize);

        int count = meta[WebPEncMeta.TokenCount];
        int x = 0;
        int y = 0;
        for (int t = 0; t < count; t++)
        {
            int hb = th + ((y >> bits) * tilesPerRow + (x >> bits)) * setAlpha;
            int t0 = tok[t * 2];
            int type = t0 & 3;
            int advance = 1;
            if (type == 0)
            {
                uint px = (uint)tok[t * 2 + 1];
                int g = hb + gOff + (int)((px >> 8) & 0xFFu);
                int r = hb + rOff + (int)((px >> 16) & 0xFFu);
                int b = hb + bOff + (int)(px & 0xFFu);
                int a = hb + aOff + (int)(px >> 24);
                w[g] = w[g] + 1;
                w[r] = w[r] + 1;
                w[b] = w[b] + 1;
                w[a] = w[a] + 1;
            }
            else if (type == 1)
            {
                int g = hb + gOff + 280 + (t0 >> 2);
                w[g] = w[g] + 1;
            }
            else
            {
                int len = t0 >> 2;
                int dist = tok[t * 2 + 1];
                int lp = Vp8lEncodeGpu.PrefixEncode(len);
                int dp = Vp8lEncodeGpu.PrefixEncode(Vp8lEncodeGpu.DistanceToPlaneCode(xsize, dist, tables));
                int g = hb + gOff + 256 + (lp & 0xFF);
                int d = hb + dOff + (dp & 0xFF);
                w[g] = w[g] + 1;
                w[d] = w[d] + 1;
                advance = len;
            }
            int nx = x + advance;
            int rows = nx / xsize;
            y += rows;
            x = nx - rows * xsize;
        }
    }

    /// <summary>Approximate code length of every symbol of a cluster histogram (an unseen symbol gets a long one).</summary>
    private static void LengthEstimates(ArrayView<int> w, int histBase, int setAlpha, int outBase)
    {
        int total = 0;
        for (int s = 0; s < setAlpha; s++) total += w[histBase + s];
        int tb = BitLength(total);
        for (int s = 0; s < setAlpha; s++)
        {
            int c = w[histBase + s];
            w[outBase + s] = c == 0 ? tb + 3 : tb - BitLength(c) + 1;
        }
    }

    private static int TileCost(ArrayView<int> w, int tileHist, int setAlpha, int lenBase)
    {
        int cost = 0;
        for (int s = 0; s < setAlpha; s++)
        {
            int h = w[tileHist + s];
            if (h != 0) cost += h * w[lenBase + s];
        }
        return cost;
    }

    /// <summary>
    /// Groups the tiles into at most <paramref name="k"/> clusters of similar histograms (farthest-point seeding, then a few
    /// k-means rounds with estimated code lengths as the distance). Writes the tile -> group map, the entropy image, the
    /// histograms of the groups (at the head of each group set) and the number of groups to <c>meta[Groups]</c>.
    /// </summary>
    public static void Cluster(ArrayView<int> w, ArrayView<int> meta, int k)
    {
        int cacheBits = meta[WebPEncMeta.CacheBits];
        int cacheSize = cacheBits > 0 ? 1 << cacheBits : 0;
        int setAlpha = Vp8lEncodeGpu.SetAlpha(cacheSize);
        int xsize = meta[WebPEncMeta.PackedWidth];
        int height = meta[WebPEncMeta.Height];
        int bits = meta[WebPEncMeta.MetaBits];
        int tilesPerRow = (xsize + (1 << bits) - 1) >> bits;
        int tiles = tilesPerRow * ((height + (1 << bits) - 1) >> bits);
        int th = meta[WebPEncMeta.RegionTileHist];
        int assign = meta[WebPEncMeta.RegionAssign];
        int entropy = meta[WebPEncMeta.RegionEntropy];
        int sets = meta[WebPEncMeta.RegionGroupSets];
        int misc = meta[WebPEncMeta.RegionGroupMisc];
        int tileMin = misc + CostTableOffset + 64;
        int stride = Vp8lEncLayout.SetInts;

        // ---- seeds: tile 0, then repeatedly the tile the current seeds serve worst ----
        for (int i = 0; i < setAlpha; i++) w[sets + i] = w[th + i];
        LengthEstimates(w, sets, setAlpha, misc);
        for (int t = 0; t < tiles; t++) w[tileMin + t] = TileCost(w, th + t * setAlpha, setAlpha, misc);
        int seeds = 1;
        bool more = k > 1;
        while (more)
        {
            int worst = 0;
            int worstCost = -1;
            for (int t = 0; t < tiles; t++)
            {
                if (w[tileMin + t] > worstCost)
                {
                    worstCost = w[tileMin + t];
                    worst = t;
                }
            }
            int cb = sets + seeds * stride;
            for (int i = 0; i < setAlpha; i++) w[cb + i] = w[th + worst * setAlpha + i];
            LengthEstimates(w, cb, setAlpha, misc + seeds * Vp8lEncLayout.SetAlphaMax);
            for (int t = 0; t < tiles; t++)
            {
                int c = TileCost(w, th + t * setAlpha, setAlpha, misc + seeds * Vp8lEncLayout.SetAlphaMax);
                if (c < w[tileMin + t]) w[tileMin + t] = c;
            }
            seeds++;
            more = seeds < k;
        }

        // ---- k-means rounds ----
        for (int round = 0; round < 4; round++)
        {
            for (int c = 0; c < seeds; c++)
                LengthEstimates(w, sets + c * stride, setAlpha, misc + c * Vp8lEncLayout.SetAlphaMax);
            for (int t = 0; t < tiles; t++)
            {
                int best = 0;
                int bestCost = Inf;
                for (int c = 0; c < seeds; c++)
                {
                    int cost = TileCost(w, th + t * setAlpha, setAlpha, misc + c * Vp8lEncLayout.SetAlphaMax);
                    if (cost < bestCost)
                    {
                        bestCost = cost;
                        best = c;
                    }
                }
                w[assign + t] = best;
            }
            for (int c = 0; c < seeds; c++)
                for (int i = 0; i < setAlpha; i++) w[sets + c * stride + i] = 0;
            for (int t = 0; t < tiles; t++)
            {
                int cb = sets + w[assign + t] * stride;
                for (int i = 0; i < setAlpha; i++) w[cb + i] = w[cb + i] + w[th + t * setAlpha + i];
            }
        }

        // ---- drop empty clusters, renumber the rest 0..n-1 ----
        for (int c = 0; c < seeds; c++) w[misc + CostTableOffset + c] = 0; // member counts
        for (int t = 0; t < tiles; t++)
        {
            int c = w[assign + t];
            w[misc + CostTableOffset + c] = w[misc + CostTableOffset + c] + 1;
        }
        int next = 0;
        for (int c = 0; c < seeds; c++)
        {
            int members = w[misc + CostTableOffset + c];
            if (members > 0)
            {
                // map[c] = next, kept in the member-count slot (negated to tell it from a count)
                w[misc + CostTableOffset + c] = -1 - next;
                if (next != c)
                    for (int i = 0; i < setAlpha; i++) w[sets + next * stride + i] = w[sets + c * stride + i];
                next++;
            }
        }
        for (int t = 0; t < tiles; t++)
        {
            int id = -1 - w[misc + CostTableOffset + w[assign + t]];
            w[assign + t] = id;
            w[entropy + t] = (int)(0xff000000u | ((uint)id << 8));
        }
        meta[WebPEncMeta.Groups] = next;
    }

    /// <summary>Builds the five codes of group <c>idx / 5</c>'s set (code <c>idx % 5</c>) and records its cost; threads past the last group return.</summary>
    public static void BuildGroupCode(ArrayView<int> w, ArrayView<int> meta, int idx)
    {
        int groups = meta[WebPEncMeta.Groups];
        int c = idx / 5;
        int k = idx - c * 5;
        int misc = meta[WebPEncMeta.RegionGroupMisc];
        if (c < groups)
        {
            int cacheBits = meta[WebPEncMeta.CacheBits];
            int cacheSize = cacheBits > 0 ? 1 << cacheBits : 0;
            int set = meta[WebPEncMeta.RegionGroupSets] + c * Vp8lEncLayout.SetInts;
            int scratch = meta[WebPEncMeta.RegionGroupBuild] + idx * Vp8lEncLayout.BuildScratch;
            Vp8lEncodeGpu.BuildOneCode(w, set, cacheSize, k, scratch);
            w[misc + CostTableOffset + idx] = Vp8lEncodeGpu.CodeCostBits(w, set, cacheSize, k);
        }
        else
        {
            w[misc + CostTableOffset + idx] = 0;
        }
    }
}
