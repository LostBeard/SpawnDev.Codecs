// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 fixed-point cosine constants used by the forward and inverse DCT
// implementations. Values are bit-exact copies of libvpx
// vpx_dsp/txfm_common.h cospi_<i>_64 macros = round(cos(pi*i/64) * 2^14).
//
// DCT_CONST_BITS = 14. fdct_round_shift / dct_const_round_shift =
// (input + (1 << 13)) >> 14.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 cosine constants in DCT_CONST_BITS=14 fixed-point.</summary>
public static class Vp9CospiConstants
{
    /// <summary>DCT_CONST_BITS from libvpx (precision of cospi values).</summary>
    public const int DctConstBits = 14;

    /// <summary>Rounding constant for fdct_round_shift.</summary>
    public const int DctConstRounding = 1 << (DctConstBits - 1);

    // cos(pi*i/64) * 2^14, i in 1..32
    public const int Cospi2_64  = 16305;
    public const int Cospi4_64  = 16069;
    public const int Cospi6_64  = 15679;
    public const int Cospi8_64  = 15137;
    public const int Cospi10_64 = 14449;
    public const int Cospi12_64 = 13623;
    public const int Cospi14_64 = 12665;
    public const int Cospi16_64 = 11585;
    public const int Cospi18_64 = 10394;
    public const int Cospi20_64 = 9102;
    public const int Cospi22_64 = 7723;
    public const int Cospi24_64 = 6270;
    public const int Cospi26_64 = 4756;
    public const int Cospi28_64 = 3196;
    public const int Cospi30_64 = 1606;

    // Sin variants (used by IADST).
    public const int SinpiAr1   = 5283;
    public const int SinpiBr1   = 13377;
    public const int SinpiCr1   = 15212;
    public const int SinpiDr1   = 9929;

    /// <summary>fdct_round_shift: (input + DctConstRounding) &gt;&gt; DctConstBits.</summary>
    public static int RoundShift(long input) =>
        (int)((input + DctConstRounding) >> DctConstBits);
}
