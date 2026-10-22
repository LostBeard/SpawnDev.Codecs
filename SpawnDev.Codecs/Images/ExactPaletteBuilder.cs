// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Exact-colour discovery on the accelerator, shared by the palette encoders (GIF, indexed BMP): decides whether
// an image has at most N distinct opaque colours and, if so, produces the palette (sorted ascending by
// 0xRRGGBB, so the result does not depend on thread scheduling).
//
//   1. ScanAlpha  (1 thread per pixel)  flags whether any pixel is transparent (alpha < threshold).
//   2. SetLimit   (1 thread)            limit = maxColors - (transparent ? 1 : 0)  (one index is reserved).
//   3. Insert     (1 thread per pixel)  lock-free open-addressing hash set of 24-bit colours (CAS insert,
//                                       gives up as soon as more than `limit` distinct colours exist).
//   4. Finish     (1 thread)            compacts the set into a sorted palette, writes Exact / PaletteCount.
//
// The set has 1024 slots, so it never fills while it holds at most 256 colours; the insert loop is bounded.
// Colours are accumulated across several Insert calls (multi-frame global palette) until the next Reset.

using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.Codecs.Images;

/// <summary>Slots of the exact-palette state block.</summary>
internal static class ExactPaletteState
{
    public const int Size = 8;
    /// <summary>Distinct opaque colours inserted so far (stops growing once above the limit).</summary>
    public const int Distinct = 0;
    /// <summary>1 when an insert could not find a free slot.</summary>
    public const int Overflow = 1;
    /// <summary>1 when the image has at most <see cref="Limit"/> opaque colours (palette in <c>Palette</c>).</summary>
    public const int Exact = 2;
    /// <summary>Entries in the palette when <see cref="Exact"/> is 1.</summary>
    public const int PaletteCount = 3;
    /// <summary>1 when any pixel has alpha below the threshold.</summary>
    public const int HasTransparent = 4;
    /// <summary>Largest number of opaque colours the palette may hold.</summary>
    public const int Limit = 5;
}

internal sealed class ExactPaletteBuilder : IDisposable
{
    public const int TableSlots = 1024;
    public const int MaxPalette = 256;

    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>, int> _scanAlpha;
    private readonly Action<Index1D, ArrayView<int>, int> _setLimit;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, int, int> _setLimitFrom;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, int> _insert;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>> _finish;

    private readonly MemoryBuffer1D<int, Stride1D.Dense> _table;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _palette;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _state;

    public ExactPaletteBuilder(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _scanAlpha = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<int>, int>(ScanAlphaKernel);
        _setLimit = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, int>(SetLimitKernel);
        _setLimitFrom = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, int, int>(SetLimitFromKernel);
        _insert = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, int>(InsertKernel);
        _finish = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>>(FinishKernel);
        _table = accelerator.Allocate1D<int>(TableSlots);
        _palette = accelerator.Allocate1D<int>(MaxPalette);
        _state = accelerator.Allocate1D<int>(ExactPaletteState.Size);
    }

    /// <summary>Sorted palette (0xRRGGBB per entry); valid when <c>State[Exact] == 1</c>.</summary>
    public ArrayView1D<int, Stride1D.Dense> Palette => _palette.View;

    /// <summary>The state block (see <see cref="ExactPaletteState"/>).</summary>
    public ArrayView1D<int, Stride1D.Dense> State => _state.View;

    /// <summary>Forget everything (empty set, no transparency seen).</summary>
    public void Reset()
    {
        _table.View.MemSetToZero();
        _state.View.MemSetToZero();
    }

    /// <summary>Record whether any of <paramref name="pixelCount"/> RGBA pixels has alpha &lt; <paramref name="alphaThreshold"/>.</summary>
    public void ScanAlpha(ArrayView<byte> rgba, int pixelCount, int alphaThreshold)
    {
        if (pixelCount > 0) _scanAlpha(pixelCount, rgba, _state.View, alphaThreshold);
    }

    /// <summary>Fix the colour limit: <paramref name="maxColors"/> minus one reserved index when a transparent pixel was seen.</summary>
    public void SetLimit(int maxColors)
    {
        _setLimit(1, _state.View, maxColors);
    }

    /// <summary>
    /// Like <see cref="SetLimit"/>, but the reserved transparent index is decided by a flag that lives on the
    /// accelerator (<paramref name="flags"/>[<paramref name="flagIndex"/>] != 0), e.g. "some frame of the animation
    /// has a transparent pixel". Lets a per-frame builder follow an animation-wide decision without a host readback.
    /// </summary>
    public void SetLimitFrom(ArrayView<int> flags, int flagIndex, int maxColors)
    {
        _setLimitFrom(1, _state.View, flags, flagIndex, maxColors);
    }

    /// <summary>Add the opaque colours of <paramref name="pixelCount"/> RGBA pixels (alpha &gt;= threshold) to the set.</summary>
    public void Insert(ArrayView<byte> rgba, int pixelCount, int alphaThreshold)
    {
        if (pixelCount > 0) _insert(pixelCount, rgba, _table.View, _state.View, alphaThreshold);
    }

    /// <summary>Produce the sorted palette and the Exact / PaletteCount verdict.</summary>
    public void Finish()
    {
        _finish(1, _table.View, _palette.View, _state.View);
    }

    private static void ScanAlphaKernel(Index1D index, ArrayView<byte> rgba, ArrayView<int> state, int threshold)
    {
        int a = rgba[index.X * 4 + 3];
        if (a < threshold) state[ExactPaletteState.HasTransparent] = 1;
    }

    private static void SetLimitKernel(Index1D _, ArrayView<int> state, int maxColors)
    {
        state[ExactPaletteState.Limit] = maxColors - state[ExactPaletteState.HasTransparent];
    }

    private static void SetLimitFromKernel(Index1D _, ArrayView<int> state, ArrayView<int> flags, int flagIndex, int maxColors)
    {
        int reserved = 0;
        if (flags[flagIndex] != 0) reserved = 1;
        state[ExactPaletteState.Limit] = maxColors - reserved;
    }

    private static void InsertKernel(
        Index1D index, ArrayView<byte> rgba, ArrayView<int> table, ArrayView<int> state, int threshold)
    {
        int p = index.X * 4;
        int a = rgba[p + 3];
        int limit = state[ExactPaletteState.Limit];
        bool opaque = a >= threshold;
        bool room = state[ExactPaletteState.Distinct] <= limit;
        bool wanted = opaque & room;
        if (wanted)
        {
            // Bit 24 is a marker so the key is never 0 (0 = empty slot).
            int key = 0x1000000 | (rgba[p] << 16) | (rgba[p + 1] << 8) | rgba[p + 2];
            int slot = (int)(((uint)key * 2654435761u) >> 22);
            int step = 0;
            bool done = false;
            // Loop condition as a bool local built with non-short-circuit operators (WebGPU hangs on `a && b`).
            bool go = (!done) & (step < TableSlots);
            while (go)
            {
                int seen = Atomic.CompareExchange(ref table[slot], 0, key);
                if (seen == 0)
                {
                    Atomic.Add(ref state[ExactPaletteState.Distinct], 1);
                    done = true;
                }
                else if (seen == key)
                {
                    done = true;
                }
                else
                {
                    slot = (slot + 1) & (TableSlots - 1);
                    step++;
                    // Once over the limit the verdict is "not exact"; stop probing.
                    if (state[ExactPaletteState.Distinct] > limit) done = true;
                }
                go = (!done) & (step < TableSlots);
            }
            if (!done) state[ExactPaletteState.Overflow] = 1;
        }
    }

    private static void FinishKernel(Index1D _, ArrayView<int> table, ArrayView<int> palette, ArrayView<int> state)
    {
        int limit = state[ExactPaletteState.Limit];
        int n = 0;
        for (int s = 0; s < TableSlots; s++)
        {
            int k = table[s];
            if (k != 0)
            {
                if (n < MaxPalette) palette[n] = k & 0xFFFFFF;
                n++;
            }
        }
        bool exact = (state[ExactPaletteState.Overflow] == 0) & (n <= limit) & (n <= MaxPalette);
        if (exact)
        {
            // Insertion sort ascending: deterministic palette order whatever the thread schedule.
            for (int i = 1; i < n; i++)
            {
                int v = palette[i];
                int j = i - 1;
                bool moving = true;
                while (moving)
                {
                    // Nested ifs, not &&: palette[-1] must never be evaluated.
                    if (j >= 0)
                    {
                        if (palette[j] > v)
                        {
                            palette[j + 1] = palette[j];
                            j--;
                        }
                        else
                        {
                            moving = false;
                        }
                    }
                    else
                    {
                        moving = false;
                    }
                }
                palette[j + 1] = v;
            }
            state[ExactPaletteState.Exact] = 1;
            state[ExactPaletteState.PaletteCount] = n;
        }
        else
        {
            state[ExactPaletteState.Exact] = 0;
            state[ExactPaletteState.PaletteCount] = 0;
        }
    }

    public void Dispose()
    {
        _table.Dispose();
        _palette.Dispose();
        _state.Dispose();
    }
}
