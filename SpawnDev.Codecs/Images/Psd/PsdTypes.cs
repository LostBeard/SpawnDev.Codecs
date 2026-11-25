// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// What the layered PSD writer takes (PsdEncoderGpu). Written from Adobe's published Photoshop File Formats Specification and from ComfyUI's
// compositor export measured as a black box (Lore, 2026-11-25: SpawnGen's Export PSD button): blend keys, flags, names, rectangles,
// channel order and compression. No GPL or LGPL PSD code was read.

using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.Codecs.Images.Psd;

/// <summary>The blend modes of a layer. Each is one enum value; grain extract and grain merge have no key of their own in the format and are written as normal, as
/// ComfyUI writes them.</summary>
public enum PsdBlendMode
{
    /// <summary>Normal.</summary>
    Normal = 0,
    /// <summary>Multiply.</summary>
    Multiply,
    /// <summary>Screen.</summary>
    Screen,
    /// <summary>Overlay.</summary>
    Overlay,
    /// <summary>Darken.</summary>
    Darken,
    /// <summary>Lighten.</summary>
    Lighten,
    /// <summary>Color dodge.</summary>
    ColorDodge,
    /// <summary>Color burn.</summary>
    ColorBurn,
    /// <summary>Hard light.</summary>
    HardLight,
    /// <summary>Soft light.</summary>
    SoftLight,
    /// <summary>Difference.</summary>
    Difference,
    /// <summary>Exclusion.</summary>
    Exclusion,
    /// <summary>Linear dodge (add).</summary>
    LinearDodge,
    /// <summary>Linear burn.</summary>
    LinearBurn,
    /// <summary>Vivid light.</summary>
    VividLight,
    /// <summary>Pin light.</summary>
    PinLight,
    /// <summary>Linear light.</summary>
    LinearLight,
    /// <summary>Hard mix.</summary>
    HardMix,
    /// <summary>Subtract.</summary>
    Subtract,
    /// <summary>Divide.</summary>
    Divide,
    /// <summary>Grain extract (GIMP): the format has no key for it; written as normal.</summary>
    GrainExtract,
    /// <summary>Grain merge (GIMP): the format has no key for it; written as normal.</summary>
    GrainMerge,
    /// <summary>Hue.</summary>
    Hue,
    /// <summary>Saturation.</summary>
    Saturation,
    /// <summary>Color.</summary>
    Color,
    /// <summary>Luminosity.</summary>
    Luminosity,
}

/// <summary>
/// One layer of a <see cref="PsdDocument"/>: where its pixels are, how it blends, and its name. Give the pixels as RGBA8 (straight alpha, rows top to
/// bottom, tightly packed) either as managed bytes (<see cref="Rgba"/>) or as an accelerator view (<see cref="RgbaView"/>): exactly one.
/// </summary>
public sealed record PsdLayer
{
    /// <summary>The layer's name (Unicode; kept whole in the file).</summary>
    public string Name { get; init; } = "";

    /// <summary>Canvas column of the layer's left edge (may be negative).</summary>
    public int Left { get; init; }

    /// <summary>Canvas row of the layer's top edge (may be negative).</summary>
    public int Top { get; init; }

    /// <summary>Width of the layer's pixels (1..30000). May run past the canvas.</summary>
    public int Width { get; init; }

    /// <summary>Height of the layer's pixels (1..30000). May run past the canvas.</summary>
    public int Height { get; init; }

    /// <summary>The pixels as managed bytes: <see cref="Width"/> x <see cref="Height"/> x 4 RGBA8. Not used when <see cref="RgbaView"/> is set.</summary>
    public ReadOnlyMemory<byte> Rgba { get; init; }

    /// <summary>The pixels on the accelerator: at least <see cref="Width"/> x <see cref="Height"/> x 4 RGBA8 bytes.</summary>
    public ArrayView1D<byte, Stride1D.Dense>? RgbaView { get; init; }

    /// <summary>Blend mode. Default normal.</summary>
    public PsdBlendMode Blend { get; init; } = PsdBlendMode.Normal;

    /// <summary>Layer opacity, 0 (invisible) to 255 (opaque). Default 255.</summary>
    public byte Opacity { get; init; } = 255;

    /// <summary>True (default) when the layer shows; false hides it.</summary>
    public bool Visible { get; init; } = true;
}

/// <summary>The flattened picture of the document (what a viewer that ignores layers shows).</summary>
public sealed record PsdComposite
{
    /// <summary>The pixels: RGBA8 (managed bytes), canvas width x height x 4. The alpha is written as the merged transparency only when <see cref="HasAlpha"/>.</summary>
    public ReadOnlyMemory<byte> Rgba { get; init; }

    /// <summary>The pixels on the accelerator (same layout). Used instead of <see cref="Rgba"/> when set.</summary>
    public ArrayView1D<byte, Stride1D.Dense>? RgbaView { get; init; }

    /// <summary>True: write the alpha channel of the composite as the document's merged transparency (4 channels).
    /// False (default): write the 3 colour channels.</summary>
    public bool HasAlpha { get; init; }
}

/// <summary>A layered picture to write as a PSD file.</summary>
public sealed record PsdDocument
{
    /// <summary>Canvas width (1..30000).</summary>
    public int Width { get; init; }

    /// <summary>Canvas height (1..30000).</summary>
    public int Height { get; init; }

    /// <summary>The layers, bottom first. May be empty (a flat PSD).</summary>
    public IReadOnlyList<PsdLayer> Layers { get; init; } = Array.Empty<PsdLayer>();

    /// <summary>The flattened picture (required).</summary>
    public PsdComposite Composite { get; init; } = new();
}

internal static class PsdFormat
{
    /// <summary>Largest side the writer accepts: PSD version 1 stores dimensions as 32-bit but the format caps pixel sides at 30,000 and its row
    /// byte counts are 16-bit.</summary>
    public const int MaxSide = 30000;

    /// <summary>Largest number of layers.</summary>
    public const int MaxLayers = 8192;

    /// <summary>The 4-character blend key of <paramref name="mode"/> (measured on ComfyUI's export; Adobe's list).</summary>
    public static string BlendKey(PsdBlendMode mode) => mode switch
    {
        PsdBlendMode.Normal => "norm",
        PsdBlendMode.Multiply => "mul ",
        PsdBlendMode.Screen => "scrn",
        PsdBlendMode.Overlay => "over",
        PsdBlendMode.Darken => "dark",
        PsdBlendMode.Lighten => "lite",
        PsdBlendMode.ColorDodge => "div ",
        PsdBlendMode.ColorBurn => "idiv",
        PsdBlendMode.HardLight => "hLit",
        PsdBlendMode.SoftLight => "sLit",
        PsdBlendMode.Difference => "diff",
        PsdBlendMode.Exclusion => "smud",
        PsdBlendMode.LinearDodge => "lddg",
        PsdBlendMode.LinearBurn => "lbrn",
        PsdBlendMode.VividLight => "vLit",
        PsdBlendMode.PinLight => "pLit",
        PsdBlendMode.LinearLight => "lLit",
        PsdBlendMode.HardMix => "hMix",
        PsdBlendMode.Subtract => "fsub",
        PsdBlendMode.Divide => "fdiv",
        PsdBlendMode.GrainExtract => "norm",
        PsdBlendMode.GrainMerge => "norm",
        PsdBlendMode.Hue => "hue ",
        PsdBlendMode.Saturation => "sat ",
        PsdBlendMode.Color => "colr",
        PsdBlendMode.Luminosity => "lum ",
        _ => throw new ArgumentOutOfRangeException(nameof(mode)),
    };
}
