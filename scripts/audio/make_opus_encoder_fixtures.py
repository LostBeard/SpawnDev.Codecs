"""Fixtures for the Opus ENCODER (Audio/Opus, CELT-only): exact PCM in, the packets LIBOPUS 1.5.2 makes from it, and the
encoder's final range after each packet.

Truth:
  <name>.pcmin      the interleaved int16 PCM given to opus_encode(), frame after frame (exactly what the encoder saw).
  <name>.opusp      the packets: per packet a little-endian uint16 length then the bytes.
  <name>.rng        per packet the encoder's OPUS_GET_FINAL_RANGE after it (uint32 LE).
  manifest.tsv      name, rate, channels, frame_us, application, bitrate, vbr, cvbr, complexity, bandwidth, packets, notes
With the instrumented debug build (_refs/opus-dbg: celt_encoder.c exports opus_dbg_celt_open), also, per stream:
  <name>.celtin     per celt_encode_with_ec call, 20 int32: call, frame_size, CC, nbCompressedBytes, tell on entry,
                    standalone (enc == NULL), then the CELT CTLs on entry: bitrate, vbr, cvbr, complexity, start, end,
                    C (stream channels), lsb_depth, disable_pf, force_intra, lfe, disable_inv, loss_rate, upsample; then
                    CC*frame_size interleaved int16 = the exact CELT input after the Opus layer (delay, dc_reject, ...).
  <name>.celtdbg    per call, text: the CTL values and encoder state on entry, then every decision (budget, silence, pitch,
                    transient, energies, dynalloc, tf, spread, boosts, stereo, trim, vbr, allocation, tells, final range).
Every stream is CELT-only (OPUS_SET_FORCE_MODE MODE_CELT_ONLY, or OPUS_APPLICATION_RESTRICTED_LOWDELAY), complexity <= 9 (libopus
1.5.2 runs its tonality analysis only at complexity >= 10, so the encode is the deterministic fixed-point path), signal type
and bandwidth left to the encoder unless stated. Nothing here comes from SpawnDev.Codecs or Concentus. libopus is 1.5.2 built
FIXED POINT with intrinsics off (cmake -DOPUS_FIXED_POINT=ON -DOPUS_DISABLE_INTRINSICS=ON -DBUILD_SHARED_LIBS=ON).

Usage: python -I make_opus_encoder_fixtures.py <path to opus.dll> [out dir]
"""
import ctypes
import math
import os
import random
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "..", "SpawnDev.Codecs.Demo.Shared", "TestData", "OpusEnc"))

APP = {"voip": 2048, "audio": 2049, "lowdelay": 2051}
OPUS_SET_BITRATE = 4002
OPUS_SET_VBR = 4006
OPUS_SET_BANDWIDTH = 4008
OPUS_SET_COMPLEXITY = 4010
OPUS_SET_VBR_CONSTRAINT = 4020
OPUS_SET_FORCE_CHANNELS = 4022
OPUS_GET_FINAL_RANGE = 4031
OPUS_SET_FORCE_MODE = 11002
MODE_CELT_ONLY = 1002
BW = {"auto": -1000, "NB": 1101, "WB": 1103, "SWB": 1104, "FB": 1105}


def music(rate, channels, samples, seed):
    """Chord with vibrato, sharp decaying noise bursts every 150 ms (transients), an HF shimmer."""
    rng = random.Random(seed)
    out = [0] * (samples * channels)
    period = int(0.15 * rate)
    for ch in range(channels):
        phases = [0.0] * 4
        freqs = [220.0, 277.18, 329.63, 440.0 * (1 + 0.01 * ch)]
        burst = 0.0
        for i in range(samples):
            t = i / rate
            v = 0.0
            for k in range(4):
                phases[k] += 2 * math.pi * freqs[k] * (1 + 0.003 * math.sin(2 * math.pi * 5 * t)) / rate
                v += 0.12 * math.sin(phases[k])
            v += 0.03 * math.sin(2 * math.pi * min(rate * 0.45, 9000 + 2000 * ch) * t)
            if (i + ch * period // 3) % period == 0:
                burst = 0.8
            v += burst * (rng.random() * 2 - 1)
            burst *= 0.985
            out[i * channels + ch] = max(-32768, min(32767, int(round(v * 32767))))
    return out


def speech(rate, channels, samples, seed):
    """Amplitude-modulated harmonics with a wandering pitch and a little noise, a near-silent stretch."""
    rng = random.Random(seed)
    out = [0] * (samples * channels)
    for ch in range(channels):
        f0, phase = 140 + 40 * ch, 0.0
        for i in range(samples):
            t = i / rate
            phase += 2 * math.pi * f0 * (1 + 0.15 * math.sin(2 * math.pi * 1.3 * t)) / rate
            env = 0.5 + 0.5 * math.sin(2 * math.pi * 3.1 * t + ch)
            v = 0.25 * env * sum(math.sin(h * phase) / h for h in range(1, 7)) + 0.01 * (rng.random() * 2 - 1)
            if 0.4 * samples < i < 0.5 * samples:
                v *= 0.002
            out[i * channels + ch] = max(-32768, min(32767, int(round(v * 32767))))
    return out


class Opus:
    def __init__(self, dll):
        self.lib = ctypes.CDLL(dll)
        self.lib.opus_encoder_create.restype = ctypes.c_void_p
        self.lib.opus_encoder_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        self.lib.opus_encode.restype = ctypes.c_int
        self.lib.opus_encode.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_short), ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
        self.lib.opus_encoder_ctl.restype = ctypes.c_int
        self.lib.opus_encoder_destroy.argtypes = [ctypes.c_void_p]
        self.lib.opus_get_version_string.restype = ctypes.c_char_p

    def encoder(self, rate, channels, app, ctl):
        err = ctypes.c_int(0)
        enc = self.lib.opus_encoder_create(rate, channels, APP[app], ctypes.byref(err))
        if err.value != 0:
            raise RuntimeError(f"opus_encoder_create {err.value}")
        for req, val in ctl.items():
            r = self.lib.opus_encoder_ctl(ctypes.c_void_p(enc), ctypes.c_int(req), ctypes.c_int(val))
            if r != 0:
                raise RuntimeError(f"opus_encoder_ctl({req}, {val}) = {r}")
        return enc

    def encode(self, enc, pcm, frame):
        buf = (ctypes.c_short * len(pcm))(*pcm)
        out = ctypes.create_string_buffer(1500)
        n = self.lib.opus_encode(ctypes.c_void_p(enc), buf, frame, out, 1500)
        if n < 0:
            raise RuntimeError(f"opus_encode {n}")
        rng = ctypes.c_uint32(0)
        self.lib.opus_encoder_ctl(ctypes.c_void_p(enc), ctypes.c_int(OPUS_GET_FINAL_RANGE), ctypes.byref(rng))
        return out.raw[:n], rng.value


def case(name, sig, rate, ch, us, app="audio", bitrate=64000, vbr=0, cvbr=0, cx=5, bw="auto", packets=None, notes=""):
    frames_per_s = 1_000_000 // us if us >= 1000 else 400
    n = packets if packets is not None else max(12, min(120, int(0.6 * 1_000_000 / us)))
    return dict(name=name, sig=sig, rate=rate, ch=ch, us=us, app=app, bitrate=bitrate, vbr=vbr, cvbr=cvbr, cx=cx, bw=bw, packets=n,
                notes=notes)


CASES = [
    # Step 1 (the first target): CBR, 20 ms, mono, complexity 5.
    case("enc_cbr20_mono_64k_c5", "music", 48000, 1, 20000, notes="CBR 20 ms mono 64 kb/s complexity 5: the first target"),
    case("enc_cbr20_mono_64k_c5_speech", "speech", 48000, 1, 20000, notes="as above, speech-like input"),
    # Step 2: every frame size, stereo, complexity 0..9.
    case("enc_cbr10_mono_48k_c5", "music", 48000, 1, 10000, bitrate=48000),
    case("enc_cbr5_mono_48k_c5", "music", 48000, 1, 5000, bitrate=48000),
    case("enc_cbr2p5_mono_64k_c5", "music", 48000, 1, 2500, bitrate=64000),
    case("enc_cbr40_mono_64k_c5", "music", 48000, 1, 40000, notes="40 ms = two 20 ms CELT frames, repacketized (code 1 / 3)"),
    case("enc_cbr60_mono_64k_c5", "music", 48000, 1, 60000, notes="60 ms = three 20 ms CELT frames, repacketized"),
    case("enc_cbr20_stereo_96k_c5", "music", 48000, 2, 20000, bitrate=96000),
    case("enc_cbr10_stereo_64k_c5", "music", 48000, 2, 10000, bitrate=64000),
    case("enc_cbr5_stereo_128k_c5", "music", 48000, 2, 5000, bitrate=128000),
    case("enc_cbr2p5_stereo_96k_c5", "music", 48000, 2, 2500, bitrate=96000),
    case("enc_cbr20_stereo_24k_c5_low", "music", 48000, 2, 20000, bitrate=24000, notes="low rate stereo: intensity / dual stereo, folding"),
    case("enc_cbr20_mono_12k_c5_low", "music", 48000, 1, 20000, bitrate=12000, notes="low rate mono: bands skip and fold"),
    *[case(f"enc_cbr20_stereo_64k_c{c}", "music", 48000, 2, 20000, cx=c, packets=30, notes=f"complexity {c}") for c in range(0, 10) if c != 5],
    # Step 3: VBR and constrained VBR.
    case("enc_vbr20_mono_64k_c5", "music", 48000, 1, 20000, vbr=1),
    case("enc_cvbr20_mono_64k_c5", "music", 48000, 1, 20000, vbr=1, cvbr=1),
    case("enc_vbr20_stereo_96k_c9", "music", 48000, 2, 20000, bitrate=96000, vbr=1, cx=9),
    case("enc_cvbr10_stereo_64k_c7", "music", 48000, 2, 10000, vbr=1, cvbr=1, cx=7),
    case("enc_vbr20_mono_32k_c5_speech", "speech", 48000, 1, 20000, bitrate=32000, vbr=1),
    # Other input rates, a forced bandwidth, the restricted-low-delay application.
    case("enc_cbr20_mono_24k_rate16k_c5", "music", 16000, 1, 20000, bitrate=24000, notes="16 kHz input (CELT downsampling, end band 17)"),
    case("enc_cbr20_stereo_48k_rate24k_c5", "music", 24000, 2, 20000, bitrate=48000, notes="24 kHz input (end band 19)"),
    case("enc_cbr20_mono_64k_c5_bwswb", "music", 48000, 1, 20000, bw="SWB", notes="forced SWB at 48 kHz"),
    case("enc_lowdelay20_stereo_64k_c5", "music", 48000, 2, 20000, app="lowdelay", notes="OPUS_APPLICATION_RESTRICTED_LOWDELAY"),
    case("enc_lowdelay5_mono_32k_vbr_c5", "music", 48000, 1, 5000, app="lowdelay", bitrate=32000, vbr=1),
]


def main():
    if len(sys.argv) not in (2, 3):
        print(__doc__)
        sys.exit(2)
    opus = Opus(sys.argv[1])
    version = opus.lib.opus_get_version_string().decode()
    out = sys.argv[2] if len(sys.argv) == 3 else OUT
    dbg = getattr(opus.lib, "opus_dbg_celt_open", None)  # only the instrumented debug build has it
    if dbg is not None:
        dbg.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    os.makedirs(out, exist_ok=True)
    manifest = ["name\trate\tchannels\tframe_us\tapplication\tbitrate\tvbr\tcvbr\tcomplexity\tbandwidth\tpackets\tnotes"]
    for c in CASES:
        frame = c["rate"] * c["us"] // 1_000_000
        total = frame * c["packets"]
        pcm = (music if c["sig"] == "music" else speech)(c["rate"], c["ch"], total, seed=len(c["name"]))
        ctl = {OPUS_SET_BITRATE: c["bitrate"], OPUS_SET_VBR: c["vbr"], OPUS_SET_VBR_CONSTRAINT: c["cvbr"], OPUS_SET_COMPLEXITY: c["cx"]}
        if c["app"] != "lowdelay":
            ctl[OPUS_SET_FORCE_MODE] = MODE_CELT_ONLY
        if c["bw"] != "auto":
            ctl[OPUS_SET_BANDWIDTH] = BW[c["bw"]]
        enc = opus.encoder(c["rate"], c["ch"], c["app"], ctl)
        stream, ranges = bytearray(), []
        if dbg is not None:
            dbg(os.path.join(out, c["name"] + ".celtdbg").encode(), os.path.join(out, c["name"] + ".celtin").encode())
        for p in range(c["packets"]):
            pkt, rng = opus.encode(enc, pcm[p * frame * c["ch"]:(p + 1) * frame * c["ch"]], frame)
            if (pkt[0] >> 7) == 0:
                raise RuntimeError(f"{c['name']} packet {p} is not CELT-only (TOC {pkt[0]:02X})")
            stream += struct.pack("<H", len(pkt)) + pkt
            ranges.append(rng)
        opus.lib.opus_encoder_destroy(ctypes.c_void_p(enc))
        if dbg is not None:
            dbg(None, None)
        with open(os.path.join(out, c["name"] + ".pcmin"), "wb") as f:
            f.write(struct.pack(f"<{len(pcm)}h", *pcm))
        with open(os.path.join(out, c["name"] + ".opusp"), "wb") as f:
            f.write(bytes(stream))
        with open(os.path.join(out, c["name"] + ".rng"), "wb") as f:
            f.write(struct.pack(f"<{len(ranges)}I", *ranges))
        manifest.append("\t".join(str(c[k]) for k in ("name", "rate", "ch", "us", "app", "bitrate", "vbr", "cvbr", "cx", "bw", "packets"))
                        + "\t" + c["notes"])
        print(f"{c['name']}: {c['packets']} packets, {len(stream)} bytes")
    with open(os.path.join(out, "manifest.tsv"), "w", newline="\n") as f:
        f.write(f"# {version}, fixed point, intrinsics off\n")
        f.write("\n".join(manifest) + "\n")


if __name__ == "__main__":
    main()
