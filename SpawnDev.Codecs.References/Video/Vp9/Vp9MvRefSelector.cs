// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 MV reference selector. Given a populated
// <see cref="Vp9MvCandidatesList"/> and the block's chosen
// <see cref="Vp9InterMode"/>, picks the right reference MV.
//
// Mapping (libvpx vp9_decodemv.c assign_mv):
//   NEARESTMV -> candidates[0] (nearest)
//   NEARMV    -> candidates[1] (near)
//   ZEROMV    -> (0, 0)
//   NEWMV     -> candidates[0] is the reference for the diff
//                that the bitstream then transmits
//
// When the candidates list has fewer entries than expected (e.g.
// only 1 distinct neighbor MV found), missing slots fall back to
// <see cref="Vp9Mv.Zero"/>.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 reference MV selector by inter mode.</summary>
public static class Vp9MvRefSelector
{
    /// <summary>
    /// Nearest MV (candidates[0]), or zero if the list is empty.
    /// </summary>
    public static Vp9Mv Nearest(Vp9MvCandidatesList list)
    {
        ArgumentNullException.ThrowIfNull(list);
        return list.Count > 0 ? list[0] : Vp9Mv.Zero;
    }

    /// <summary>
    /// Near MV (candidates[1]), or zero if the list has fewer than 2
    /// entries.
    /// </summary>
    public static Vp9Mv Near(Vp9MvCandidatesList list)
    {
        ArgumentNullException.ThrowIfNull(list);
        return list.Count > 1 ? list[1] : Vp9Mv.Zero;
    }

    /// <summary>
    /// Resolve the reference MV for the given inter mode:
    /// <list type="bullet">
    /// <item><description>NearestMv: <see cref="Nearest"/></description></item>
    /// <item><description>NearMv: <see cref="Near"/></description></item>
    /// <item><description>ZeroMv: <see cref="Vp9Mv.Zero"/></description></item>
    /// <item><description>NewMv: <see cref="Nearest"/> (libvpx uses
    /// nearest as the reference for the transmitted diff)</description></item>
    /// </list>
    /// </summary>
    public static Vp9Mv ForInterMode(Vp9MvCandidatesList list, Vp9InterMode mode)
    {
        ArgumentNullException.ThrowIfNull(list);
        return mode switch
        {
            Vp9InterMode.NearestMv => Nearest(list),
            Vp9InterMode.NearMv => Near(list),
            Vp9InterMode.ZeroMv => Vp9Mv.Zero,
            Vp9InterMode.NewMv => Nearest(list),
            _ => throw new ArgumentOutOfRangeException(nameof(mode), mode,
                "inter mode must be one of NearestMv / NearMv / ZeroMv / NewMv."),
        };
    }
}
