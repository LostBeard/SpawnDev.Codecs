// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// C# equivalent of libopus silk_NLSF_CB_struct from silk/structs.h.
//
// Upstream Copyright (c) 2006-2011 Skype Limited. BSD 3-Clause. See NOTICE.md.

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// NLSF codebook data container. Groups the vector counts, quantizer step sizes,
/// and table pointers that libopus packs into <c>silk_NLSF_CB_struct</c>. In SILK
/// there are two of these at runtime: NB/MB (10 LSF coefficients) and WB (16).
/// </summary>
internal sealed class SilkNlsfCodebook
{
    /// <summary>Number of entries in the first-stage codebook.</summary>
    public required short NVectors { get; init; }

    /// <summary>Filter order (matches LPC order; 10 for NB/MB, 16 for WB).</summary>
    public required short Order { get; init; }

    /// <summary>Quantizer step size in Q16.</summary>
    public required short QuantStepSizeQ16 { get; init; }

    /// <summary>Inverse quantizer step size in Q6.</summary>
    public required short InvQuantStepSizeQ6 { get; init; }

    /// <summary>First-stage codebook NLSF values in Q8. Length = <see cref="NVectors"/> * <see cref="Order"/>.</summary>
    public required byte[] Cb1NlsfQ8 { get; init; }

    /// <summary>First-stage codebook weights in Q9 (encoder uses this; decoder can have same ref).</summary>
    public required short[] Cb1WghtQ9 { get; init; }

    /// <summary>Inverse CDF for the first-stage codebook index.</summary>
    public required byte[] Cb1Icdf { get; init; }

    /// <summary>
    /// Residual predictor coefficients in Q8. Laid out as <c>2 * (Order - 1)</c>:
    /// the upper half is used when the sign bit of <c>ec_sel</c> is 1.
    /// </summary>
    public required byte[] PredQ8 { get; init; }

    /// <summary>
    /// Entropy-coder selector bits, one byte per (vector, coefficient pair). Length
    /// = <see cref="NVectors"/> * <see cref="Order"/> / 2. Encodes which entropy table
    /// + predictor variant to use for each NLSF coefficient.
    /// </summary>
    public required byte[] EcSel { get; init; }

    /// <summary>Inverse CDF tables for residual entropy decoding.</summary>
    public required byte[] EcIcdf { get; init; }

    /// <summary>Entropy-coding rate table in Q5.</summary>
    public required byte[] EcRatesQ5 { get; init; }

    /// <summary>Minimum allowed delta between adjacent NLSF values in Q15. Length <see cref="Order"/> + 1.</summary>
    public required short[] DeltaMinQ15 { get; init; }
}
