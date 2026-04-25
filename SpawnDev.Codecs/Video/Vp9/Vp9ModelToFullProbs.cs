// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Bridge between slice 142/143's stored coefficient probabilities and
// the VP9 entropy decoder's per-coefficient probability tree. libvpx
// stores only the 3 "unconstrained" probability nodes per (tx_size,
// plane, ref, band, ctx) tuple; the remaining 8 nodes of the 11-node
// tree are reconstructed at decode time from pareto8_full using the
// 3rd stored probability as the pivot index.
//
// Spec: VP9 Bitstream Specification sec 8.5.4 "Residual decoding".
// libvpx reference: vp9/common/vp9_entropy.c
//   vp9_model_to_full_probs / extend_to_full_distribution.

namespace SpawnDev.Codecs.Video.Vp9;

public static partial class Vp9CoefProbs
{
    /// <summary>Stored unconstrained-node count per coef-prob entry (libvpx UNCONSTRAINED_NODES).</summary>
    public const int UnconstrainedNodes = 3;

    /// <summary>Pareto8-derived node count per coef-prob entry (libvpx MODEL_NODES).</summary>
    public const int ModelNodes = 8;

    /// <summary>Total entropy-tree node count for VP9 coefficient decoding (libvpx ENTROPY_NODES).</summary>
    public const int EntropyNodes = UnconstrainedNodes + ModelNodes; // 11

    /// <summary>Index of the pivot probability within the 3-entry stored model (libvpx PIVOT_NODE).</summary>
    public const int PivotNode = 2;

    /// <summary>
    /// Expand a 3-entry stored model probability vector to the 11-entry
    /// full coefficient-tree probability vector. The first three full
    /// entries are copied verbatim from <paramref name="model"/>; the
    /// remaining eight come from
    /// <see cref="Pareto8Full"/>[<paramref name="model"/>[2] - 1, 0..7].
    ///
    /// Matches libvpx <c>vp9_model_to_full_probs</c> / <c>extend_to_full_distribution</c>
    /// bit-exactly. The pivot probability (model[2]) must be at least 1
    /// - libvpx asserts the same; a zero pivot would index Pareto8Full
    /// at -1 and is invalid in any decoded VP9 bitstream.
    /// </summary>
    public static void ModelToFullProbs(ReadOnlySpan<byte> model, Span<byte> full)
    {
        if (model.Length < UnconstrainedNodes)
            throw new ArgumentException(
                $"model must hold at least {UnconstrainedNodes} bytes",
                nameof(model));
        if (full.Length < EntropyNodes)
            throw new ArgumentException(
                $"full must hold at least {EntropyNodes} bytes",
                nameof(full));

        // Copy the 3 stored unconstrained nodes.
        full[0] = model[0];
        full[1] = model[1];
        full[2] = model[2];

        // Pivot must be >= 1 - the Pareto8 row index is `pivot - 1`.
        byte pivot = model[PivotNode];
        if (pivot == 0)
            throw new InvalidDataException(
                "pivot probability (model[2]) must be >= 1; zero indicates a corrupt bitstream");

        // Copy the 8 pareto8 nodes into full[3..10].
        int row = pivot - 1;
        for (int i = 0; i < ModelNodes; i++)
            full[UnconstrainedNodes + i] = Pareto8Full[row, i];
    }
}
