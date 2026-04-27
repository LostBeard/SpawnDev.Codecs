// Verify the Vorbis encoder works across multiple block sizes.

#:project SpawnDev.Codecs/SpawnDev.Codecs.csproj
using System;
using System.Numerics;
using SpawnDev.Codecs.Audio.Vorbis;

const int SR = 44100;
const double Hz = 440.0;

foreach (int bs in new[] { 256, 512, 1024, 2048 })
{
    int total = SR;
    var input = new float[total];
    for (int n = 0; n < total; n++)
        input[n] = (float)(0.5 * Math.Sin(2 * Math.PI * Hz * n / SR));

    var enc = new VorbisAudioEncoder(new VorbisAudioEncoderOptions { SampleRateHz = SR, Channels = 1, BlockSize = bs });
    var ogg = enc.EncodeStream(input);
    var dec = VorbisOggDecoder.Decode(ogg);

    int searchN = Math.Min(dec.InterleavedSamples.Length, 8192);
    double peakHz = 0; double peakMag = 0;
    for (int hz = 50; hz <= 1500; hz++)
    {
        Complex sum = Complex.Zero;
        for (int n = 0; n < searchN; n++)
        {
            double phase = -2 * Math.PI * hz * n / SR;
            sum += new Complex(dec.InterleavedSamples[n] * Math.Cos(phase),
                                dec.InterleavedSamples[n] * Math.Sin(phase));
        }
        double mag = sum.Magnitude;
        if (mag > peakMag) { peakMag = mag; peakHz = hz; }
    }
    Console.WriteLine($"BS={bs,4}: encoded {ogg.Length}B, decoded {dec.InterleavedSamples.Length} samples, DFT peak {peakHz:F0} Hz");
}
