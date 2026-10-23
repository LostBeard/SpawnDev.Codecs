"""Fixtures for the Vorbis decoders (Audio/Vorbis): real libvorbis streams and libvorbis's own decode of them.

Truth:
  <name>.ogg        encoded by LIBVORBIS (ffmpeg -c:a libvorbis) from a deterministic ffmpeg lavfi signal.
  <name>.f32.gz     the interleaved float PCM of <name>.ogg decoded by libsndfile (libvorbis + vorbisfile semantics:
                    the first packet primes the overlap, the last page's granule position trims the end).
  manifest.tsv      name, channels, sample rate, frames, signal, quality
Nothing here comes from SpawnDev.Codecs.

The set covers what a real file has and a self round trip does not: two floors (short + long blocks) with block
switching (the click signal), coupled stereo (libvorbis codes stereo with residue type 2), low and high quality setups
and several sample rates.

Usage: python -I make_vorbis_fixtures.py <ffmpeg.exe>     (venv with numpy + soundfile)
"""
import gzip
import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "..", "SpawnDev.Codecs.Demo.Shared", "TestData", "Vorbis"))

SECONDS = 0.4

# lavfi expressions (deterministic: no random seed involved except anoisesrc's fixed seed)
SIGNALS = {
    "chirp": "aevalsrc='0.6*sin(2*PI*(200+4000*t)*t)':s={rate}:d={dur}",
    "tones": "aevalsrc='0.3*sin(2*PI*440*t)+0.2*sin(2*PI*1500*t)+0.1*sin(2*PI*6000*t)':s={rate}:d={dur}",
    # isolated clicks on silence: libvorbis switches to short blocks around each one
    "clicks": "aevalsrc='if(lt(mod(t\\,0.1)\\,0.002)\\,0.9*sin(2*PI*3000*t)\\,0)':s={rate}:d={dur}",
    "noise": "anoisesrc=color=pink:amplitude=0.5:seed=1234:r={rate}:d={dur}",
}

# name, signal, channels, rate, quality (libvorbis -q:a, -1 .. 10)
CASES = [
    ("mono_chirp_48k_q3", "chirp", 1, 48000, 3),
    ("mono_clicks_44k_q4", "clicks", 1, 44100, 4),
    ("mono_noise_22k_q0", "noise", 1, 22050, 0),
    ("stereo_tones_44k_q5", "tones", 2, 44100, 5),
    ("stereo_clicks_48k_qm1", "clicks", 2, 48000, -1),
    ("stereo_noise_48k_q10", "noise", 2, 48000, 10),
]


def run(args):
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"{' '.join(args)}\n{r.stderr}")


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    ffmpeg = sys.argv[1]
    os.makedirs(OUT, exist_ok=True)
    manifest = ["# name\tchannels\trate\tframes\tsignal\tquality"]
    with tempfile.TemporaryDirectory() as tmp:
        for name, signal, channels, rate, quality in CASES:
            src = SIGNALS[signal].format(rate=rate, dur=SECONDS)
            wav = os.path.join(tmp, name + ".wav")
            # the second channel of a stereo case is the signal delayed and attenuated: coupled stereo that is not mono
            if channels == 2:
                graph = f"{src},asplit[a][b];[b]adelay=7,volume=0.7[c];[a][c]amerge=inputs=2"
                run([ffmpeg, "-hide_banner", "-y", "-f", "lavfi", "-i", graph, "-ac", "2", "-c:a", "pcm_f32le", wav])
            else:
                run([ffmpeg, "-hide_banner", "-y", "-f", "lavfi", "-i", src, "-c:a", "pcm_f32le", wav])
            ogg = os.path.join(OUT, name + ".ogg")
            run([ffmpeg, "-hide_banner", "-y", "-i", wav, "-map_metadata", "-1", "-fflags", "+bitexact",
                 "-c:a", "libvorbis", "-q:a", str(quality), ogg])
            pcm, sr = sf.read(ogg, dtype="float32", always_2d=True)
            if sr != rate or pcm.shape[1] != channels:
                raise SystemExit(f"{name}: libsndfile says {sr} Hz x {pcm.shape[1]}")
            with open(os.path.join(OUT, name + ".f32.gz"), "wb") as raw, \
                    gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=9, mtime=0) as f:
                f.write(np.ascontiguousarray(pcm, dtype="<f4").tobytes())
            manifest.append(f"{name}\t{channels}\t{rate}\t{pcm.shape[0]}\t{signal}\t{quality}")
            print(f"{name}: {os.path.getsize(ogg)} bytes ogg, {pcm.shape[0]} frames")
    with open(os.path.join(OUT, "manifest.tsv"), "w", newline="\n") as f:
        f.write("\n".join(manifest) + "\n")


if __name__ == "__main__":
    main()
