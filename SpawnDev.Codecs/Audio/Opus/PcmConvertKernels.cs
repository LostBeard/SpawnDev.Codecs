// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// PCM sample format conversion on the accelerator. FloatToS16Kernel converts float samples (full scale +-1.0) to int16
// the way C's lrintf(x * 32768) followed by a clip to int16 does on x86 (round half to even in the default rounding mode;
// a NaN or a product outside the int32 range gives the "integer indefinite" value 0x80000000, which clips to -32768).
// This is how FFmpeg's libswresample converts float input for an int16 encoder (observed with the ffmpeg binary), so the
// float input path of OggOpusEncoder encodes what FFmpeg encodes for the same floats.

using ILGPU;

namespace SpawnDev.Codecs.Audio.Opus;

/// <summary>PCM conversion kernels (one thread per sample).</summary>
public static class PcmConvertKernels
{
    /// <summary>
    /// output[i] = clip16(lrintf(input[i] * 32768)): round half to even; NaN and values whose product is outside the int32
    /// range give -32768. Load with LoadAutoGroupedStreamKernel; launch over the sample count.
    /// </summary>
    public static void FloatToS16Kernel(Index1D i, ArrayView<float> input, ArrayView<short> output)
        => output[i] = FloatToS16(input[i]);

    /// <summary>The conversion of one sample (see <see cref="FloatToS16Kernel"/>).</summary>
    public static short FloatToS16(float x)
    {
        float v = x * 32768f;                                                   // exact (a power of two) unless it overflows
        // NaN fails both comparisons; 2^31 and beyond (or below -2^31) is x86's integer indefinite.
        if (!(v >= -2147483648f && v < 2147483648f)) return -32768;
        if (v >= 32767.5f) return 32767;                                        // rounds to >= 32768, clipped
        if (v <= -32768.5f) return -32768;                                      // rounds to <= -32768, clipped
        int t = (int)v;                                                         // toward zero; |v| < 2^15 so v - t is exact
        float frac = v - t;
        if (frac > 0.5f || (frac == 0.5f && (t & 1) != 0)) t++;
        else if (frac < -0.5f || (frac == -0.5f && (t & 1) != 0)) t--;
        return (short)t;
    }
}
