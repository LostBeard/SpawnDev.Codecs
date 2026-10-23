// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Meta prefix codes for the VP8L encoder: instead of one set of five prefix codes for the whole image, the image is cut
// into square tiles (2^bits pixels), each tile is assigned to one of a few groups, and every group has its own code set
// built from the tokens of its tiles. Regions with different statistics (flat sky next to foliage) then cost fewer bits.
// The tile -> group map is written as the "entropy image" (group index in red + green) in front of the codes, and a token
// is coded with the group of the tile its FIRST pixel lies in - the same rule the decoder applies.
//
// The token walk is resumable (bounded launches of StepChunk tokens); the k-means is per-tile / per-(cluster, symbol)
// parallel launches; the code construction afterwards is one thread per code, in parallel.

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

    private static int TileCount(ArrayView<int> meta, out int tilesPerRow)
    {
        int xsize = meta[WebPEncMeta.PackedWidth];
        int height = meta[WebPEncMeta.Height];
        int bits = meta[WebPEncMeta.MetaBits];
        tilesPerRow = (xsize + (1 << bits) - 1) >> bits;
        return tilesPerRow * ((height + (1 << bits) - 1) >> bits);
    }

    /// <summary>
    /// Thread <paramref name="i"/> of the tile-histogram set-up (launch tiles x <see cref="Vp8lEncLayout.SetAlphaMax"/>
    /// threads): clears the per-tile histograms; thread 0 resets the resumable walk.
    /// </summary>
    public static void TileHistogramsInit(int i, ArrayView<int> w, ArrayView<int> meta)
    {
        int cacheBits = meta[WebPEncMeta.CacheBits];
        int setAlpha = Vp8lEncodeGpu.SetAlpha(cacheBits > 0 ? 1 << cacheBits : 0);
        int tiles = TileCount(meta, out _);
        if (i < tiles * setAlpha) w[meta[WebPEncMeta.RegionTileHist] + i] = 0;
        if (i == 0)
        {
            int s = Vp8lEncLayout.StepState;
            w[s + Vp8lEncLayout.SHistT] = 0;
            w[s + Vp8lEncLayout.SHistX] = 0;
            w[s + Vp8lEncLayout.SHistY] = 0;
        }
    }

    /// <summary>
    /// Per-tile histograms of the five codes, from the next <see cref="Vp8lEncLayout.StepChunk"/> tokens of the main
    /// token stream (resumable: <see cref="TileHistogramsInit"/> first; a launch after the last token does nothing).
    /// </summary>
    public static void TileHistogramsStep(ArrayView<int> w, ArrayView<int> tok, ArrayView<int> meta, ArrayView<int> tables)
    {
        int cacheBits = meta[WebPEncMeta.CacheBits];
        int cacheSize = cacheBits > 0 ? 1 << cacheBits : 0;
        int setAlpha = Vp8lEncodeGpu.SetAlpha(cacheSize);
        int xsize = meta[WebPEncMeta.PackedWidth];
        int bits = meta[WebPEncMeta.MetaBits];
        TileCount(meta, out int tilesPerRow);
        int th = meta[WebPEncMeta.RegionTileHist];

        int gOff = Vp8lEncodeGpu.AlphaOff(0, cacheSize);
        int rOff = Vp8lEncodeGpu.AlphaOff(1, cacheSize);
        int bOff = Vp8lEncodeGpu.AlphaOff(2, cacheSize);
        int aOff = Vp8lEncodeGpu.AlphaOff(3, cacheSize);
        int dOff = Vp8lEncodeGpu.AlphaOff(4, cacheSize);

        int s = Vp8lEncLayout.StepState;
        int count = meta[WebPEncMeta.TokenCount];
        int start = w[s + Vp8lEncLayout.SHistT];
        int x = w[s + Vp8lEncLayout.SHistX];
        int y = w[s + Vp8lEncLayout.SHistY];
        int end = start + Vp8lEncLayout.StepChunk;
        if (end > count) end = count;
        for (int t = start; t < end; t++)
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
        if (start < end)
        {
            w[s + Vp8lEncLayout.SHistT] = end;
            w[s + Vp8lEncLayout.SHistX] = x;
            w[s + Vp8lEncLayout.SHistY] = y;
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

    // Clustering: the tiles are grouped into at most k clusters of similar histograms (farthest-point seeding, then four
    // k-means rounds with estimated code lengths as the distance). The steps are separate launches: the per-tile and the
    // per-(cluster, symbol) work is one thread each, only the argmax of the seeding and the final renumbering are one thread
    // (over at most MaxTiles tiles / MaxGroups clusters). The host sequence is
    //   SeedFirst, TileMin, (k - 1) x [SeedNext, TileMin], 4 x [Lengths, Assign, Accumulate], Finish
    // and the result is the same as walking it in one thread (ties break the same way: first max, first min).

    private static int SetAlphaOf(ArrayView<int> meta)
    {
        int cacheBits = meta[WebPEncMeta.CacheBits];
        return Vp8lEncodeGpu.SetAlpha(cacheBits > 0 ? 1 << cacheBits : 0);
    }

    private static int TileMinBase(ArrayView<int> meta) => meta[WebPEncMeta.RegionGroupMisc] + CostTableOffset + 64;

    /// <summary>Seed 0 = tile 0's histogram, with its code length estimates.</summary>
    public static void ClusterSeedFirst(ArrayView<int> w, ArrayView<int> meta)
    {
        int setAlpha = SetAlphaOf(meta);
        int th = meta[WebPEncMeta.RegionTileHist];
        int sets = meta[WebPEncMeta.RegionGroupSets];
        for (int i = 0; i < setAlpha; i++) w[sets + i] = w[th + i];
        LengthEstimates(w, sets, setAlpha, meta[WebPEncMeta.RegionGroupMisc]);
        w[Vp8lEncLayout.StepState + Vp8lEncLayout.SClusterSeeds] = 1;
    }

    /// <summary>Tile <paramref name="t"/>: its cost under the newest seed, kept as the minimum over the seeds so far.</summary>
    public static void ClusterTileMin(int t, ArrayView<int> w, ArrayView<int> meta)
    {
        int tiles = TileCount(meta, out _);
        if (t < tiles)
        {
            int setAlpha = SetAlphaOf(meta);
            int seeds = w[Vp8lEncLayout.StepState + Vp8lEncLayout.SClusterSeeds];
            int lenBase = meta[WebPEncMeta.RegionGroupMisc] + (seeds - 1) * Vp8lEncLayout.SetAlphaMax;
            int c = TileCost(w, meta[WebPEncMeta.RegionTileHist] + t * setAlpha, setAlpha, lenBase);
            int slot = TileMinBase(meta) + t;
            bool replace = (seeds == 1) | (c < w[slot]);
            if (replace) w[slot] = c;
        }
    }

    /// <summary>The next seed: the tile the current seeds serve worst (first one on a tie).</summary>
    public static void ClusterSeedNext(ArrayView<int> w, ArrayView<int> meta)
    {
        int setAlpha = SetAlphaOf(meta);
        int tiles = TileCount(meta, out _);
        int th = meta[WebPEncMeta.RegionTileHist];
        int tileMin = TileMinBase(meta);
        int stateSlot = Vp8lEncLayout.StepState + Vp8lEncLayout.SClusterSeeds;
        int seeds = w[stateSlot];
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
        int cb = meta[WebPEncMeta.RegionGroupSets] + seeds * Vp8lEncLayout.SetInts;
        for (int i = 0; i < setAlpha; i++) w[cb + i] = w[th + worst * setAlpha + i];
        LengthEstimates(w, cb, setAlpha, meta[WebPEncMeta.RegionGroupMisc] + seeds * Vp8lEncLayout.SetAlphaMax);
        w[stateSlot] = seeds + 1;
    }

    /// <summary>Cluster <paramref name="c"/> (launch <see cref="MaxGroups"/>): code length estimates of its histogram.</summary>
    public static void ClusterLengths(int c, ArrayView<int> w, ArrayView<int> meta)
    {
        int seeds = w[Vp8lEncLayout.StepState + Vp8lEncLayout.SClusterSeeds];
        if (c < seeds)
            LengthEstimates(w, meta[WebPEncMeta.RegionGroupSets] + c * Vp8lEncLayout.SetInts, SetAlphaOf(meta),
                meta[WebPEncMeta.RegionGroupMisc] + c * Vp8lEncLayout.SetAlphaMax);
    }

    /// <summary>Tile <paramref name="t"/>: the cluster whose estimated code lengths code it cheapest (first one on a tie).</summary>
    public static void ClusterAssign(int t, ArrayView<int> w, ArrayView<int> meta)
    {
        int tiles = TileCount(meta, out _);
        if (t < tiles)
        {
            int setAlpha = SetAlphaOf(meta);
            int seeds = w[Vp8lEncLayout.StepState + Vp8lEncLayout.SClusterSeeds];
            int misc = meta[WebPEncMeta.RegionGroupMisc];
            int hist = meta[WebPEncMeta.RegionTileHist] + t * setAlpha;
            int best = 0;
            int bestCost = Inf;
            for (int c = 0; c < seeds; c++)
            {
                int cost = TileCost(w, hist, setAlpha, misc + c * Vp8lEncLayout.SetAlphaMax);
                if (cost < bestCost)
                {
                    bestCost = cost;
                    best = c;
                }
            }
            w[meta[WebPEncMeta.RegionAssign] + t] = best;
        }
    }

    /// <summary>
    /// Thread <paramref name="idx"/> = (cluster, symbol) (launch <see cref="MaxGroups"/> x <see cref="Vp8lEncLayout.SetAlphaMax"/>):
    /// the cluster's histogram entry = the sum over its member tiles.
    /// </summary>
    public static void ClusterAccumulate(int idx, ArrayView<int> w, ArrayView<int> meta)
    {
        int c = idx / Vp8lEncLayout.SetAlphaMax;
        int i = idx - c * Vp8lEncLayout.SetAlphaMax;
        int setAlpha = SetAlphaOf(meta);
        int seeds = w[Vp8lEncLayout.StepState + Vp8lEncLayout.SClusterSeeds];
        bool active = (c < seeds) & (i < setAlpha);
        if (active)
        {
            int tiles = TileCount(meta, out _);
            int th = meta[WebPEncMeta.RegionTileHist];
            int assign = meta[WebPEncMeta.RegionAssign];
            int sum = 0;
            for (int t = 0; t < tiles; t++)
                if (w[assign + t] == c) sum += w[th + t * setAlpha + i];
            w[meta[WebPEncMeta.RegionGroupSets] + c * Vp8lEncLayout.SetInts + i] = sum;
        }
    }

    /// <summary>
    /// Drops empty clusters and renumbers the rest 0..n-1: writes the tile -> group map, the entropy image, the histograms
    /// of the groups (at the head of each group set) and the number of groups to <c>meta[Groups]</c>.
    /// </summary>
    public static void ClusterFinish(ArrayView<int> w, ArrayView<int> meta)
    {
        int setAlpha = SetAlphaOf(meta);
        int tiles = TileCount(meta, out _);
        int assign = meta[WebPEncMeta.RegionAssign];
        int entropy = meta[WebPEncMeta.RegionEntropy];
        int sets = meta[WebPEncMeta.RegionGroupSets];
        int misc = meta[WebPEncMeta.RegionGroupMisc];
        int stride = Vp8lEncLayout.SetInts;
        int seeds = w[Vp8lEncLayout.StepState + Vp8lEncLayout.SClusterSeeds];

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
