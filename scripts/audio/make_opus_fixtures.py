"""Fixtures for the Opus decoders (Audio/Opus): streams encoded by LIBOPUS and libopus's own decode of them, including lost
packets (PLC / CNG), DTX, CELT-only and hybrid streams of every frame size, and mode switches.

Truth:
  <name>.opusp      the packet stream: per packet a little-endian uint16 length then the bytes; length 0xFFFF marks a
                    packet that is LOST (the decoder is asked to conceal one packet duration).
  <name>.pcm16      the interleaved int16 PCM libopus's opus_decode() returns for the whole stream, packet after packet,
                    concealment included (decode_fec = 0). The float API of a fixed-point libopus returns exactly these
                    values / 32768.
  <name>.rng        per packet the decoder's OPUS_GET_FINAL_RANGE after it (uint32 LE; 0 for a lost packet), the entropy
                    coder's state: it catches range-coder drift before the PCM shows it.
  manifest.tsv      name, api rate, api channels, packet duration in microseconds, packets, lost packets, notes
Nothing here comes from SpawnDev.Codecs or Concentus. libopus is 1.5.2 built FIXED POINT with intrinsics off
(cmake -DOPUS_FIXED_POINT=ON -DOPUS_DISABLE_INTRINSICS=ON -DBUILD_SHARED_LIBS=ON), the same arithmetic Concentus ports.

Why libopus and not Concentus: Concentus 2.2.2 ports libopus ~1.1. 1.5.2 differs (silk_CNG, the PLC synthesis update,
decode_core's voiced-PLC transition, CELT changes), and Concentus's silk_CNG never copies the frame's excitation into
CNG_exc_buf_Q14 and its enc_shrink moves the raw-bit tail the wrong way.

Usage: python -I make_opus_fixtures.py <path to opus.dll>
"""
import ctypes
import math
import os
import random
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "..", "SpawnDev.Codecs.Demo.Shared", "TestData", "Opus"))

OPUS_APPLICATION_VOIP = 2048
OPUS_SET_BITRATE = 4002
OPUS_SET_VBR = 4006
OPUS_SET_BANDWIDTH = 4008
OPUS_SET_COMPLEXITY = 4010
OPUS_SET_INBAND_FEC = 4012
OPUS_SET_PACKET_LOSS_PERC = 4014
OPUS_SET_DTX = 4016
OPUS_SET_FORCE_CHANNELS = 4022
OPUS_GET_FINAL_RANGE = 4031
OPUS_SET_FORCE_MODE = 11002
MODE_SILK_ONLY = 1000
MODE_HYBRID = 1001
MODE_CELT_ONLY = 1002
BW = {"NB": 1101, "MB": 1102, "WB": 1103, "SWB": 1104, "FB": 1105}


def speech(rate, channels, samples, seed, quiet=(0.4, 0.5)):
    """Speech-like: amplitude-modulated harmonics with a wandering pitch, a little noise, a quiet stretch (VAD off,
    DTX, mid-only stereo frames)."""
    rng = random.Random(seed)
    out = [0] * (samples * channels)
    for ch in range(channels):
        f0, phase = 140 + 40 * ch, 0.0
        for i in range(samples):
            t = i / rate
            pitch = f0 * (1 + 0.15 * math.sin(2 * math.pi * 1.3 * t))
            phase += 2 * math.pi * pitch / rate
            env = 0.5 + 0.5 * math.sin(2 * math.pi * 3.1 * t + ch)
            v = sum(math.sin(h * phase) / h for h in range(1, 7))
            v = 0.25 * env * v + 0.01 * (rng.random() * 2 - 1)
            if quiet[0] * samples < i < quiet[1] * samples:
                v *= 0.002
            out[i * channels + ch] = max(-32768, min(32767, int(round(v * 32767))))
    return out


def music(rate, channels, samples, seed):
    """Music-like with TRANSIENTS (CELT short blocks, tf changes, anti-collapse): a sustained chord with vibrato, sharp
    decaying noise bursts every 150 ms (offset per channel, so stereo differs), and a high-frequency shimmer."""
    rng = random.Random(seed)
    out = [0] * (samples * channels)
    period = int(0.15 * rate)
    for ch in range(channels):
        phases = [0.0, 0.0, 0.0, 0.0]
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


class Opus:
    def __init__(self, dll):
        self.lib = ctypes.CDLL(dll)
        self.lib.opus_encoder_create.restype = ctypes.c_void_p
        self.lib.opus_encoder_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        self.lib.opus_encode.restype = ctypes.c_int
        self.lib.opus_encode.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_short), ctypes.c_int,
                                         ctypes.c_char_p, ctypes.c_int]
        self.lib.opus_encoder_ctl.restype = ctypes.c_int
        self.lib.opus_decoder_ctl.restype = ctypes.c_int
        self.lib.opus_decoder_create.restype = ctypes.c_void_p
        self.lib.opus_decoder_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        self.lib.opus_decode.restype = ctypes.c_int
        self.lib.opus_decode.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int,
                                         ctypes.POINTER(ctypes.c_short), ctypes.c_int, ctypes.c_int]
        self.lib.opus_encoder_destroy.argtypes = [ctypes.c_void_p]
        self.lib.opus_decoder_destroy.argtypes = [ctypes.c_void_p]
        self.lib.opus_get_version_string.restype = ctypes.c_char_p

    def encoder(self, rate, channels, ctl):
        err = ctypes.c_int(0)
        enc = self.lib.opus_encoder_create(rate, channels, OPUS_APPLICATION_VOIP, ctypes.byref(err))
        if err.value != 0:
            raise RuntimeError(f"opus_encoder_create {err.value}")
        for req, val in ctl.items():
            self.ctl(enc, req, val)
        return enc

    def ctl(self, enc, req, val):
        r = self.lib.opus_encoder_ctl(ctypes.c_void_p(enc), ctypes.c_int(req), ctypes.c_int(val))
        if r != 0:
            raise RuntimeError(f"opus_encoder_ctl({req}, {val}) = {r}")

    def encode(self, enc, pcm, frame):
        buf = (ctypes.c_short * len(pcm))(*pcm)
        out = ctypes.create_string_buffer(1500)
        n = self.lib.opus_encode(ctypes.c_void_p(enc), buf, frame, out, 1500)
        if n < 0:
            raise RuntimeError(f"opus_encode {n}")
        return out.raw[:n]

    def decoder(self, rate, channels):
        err = ctypes.c_int(0)
        dec = self.lib.opus_decoder_create(rate, channels, ctypes.byref(err))
        if err.value != 0:
            raise RuntimeError(f"opus_decoder_create {err.value}")
        return dec

    def decode(self, dec, packet, frame, channels):
        pcm = (ctypes.c_short * (5760 * channels))()
        if packet is None:
            n = self.lib.opus_decode(ctypes.c_void_p(dec), None, 0, pcm, frame, 0)
        else:
            n = self.lib.opus_decode(ctypes.c_void_p(dec), packet, len(packet), pcm, frame, 0)
        if n < 0:
            raise RuntimeError(f"opus_decode {n}")
        rng = ctypes.c_uint32(0)
        r = self.lib.opus_decoder_ctl(ctypes.c_void_p(dec), ctypes.c_int(OPUS_GET_FINAL_RANGE), ctypes.byref(rng))
        if r != 0:
            raise RuntimeError(f"OPUS_GET_FINAL_RANGE {r}")
        return list(pcm[: n * channels]), rng.value


def celt(bitrate, bw="FB"):
    return {OPUS_SET_FORCE_MODE: MODE_CELT_ONLY, OPUS_SET_BITRATE: bitrate, OPUS_SET_BANDWIDTH: BW[bw]}


def hybrid(bitrate, bw="FB", fec=False):
    c = {OPUS_SET_FORCE_MODE: MODE_HYBRID, OPUS_SET_BITRATE: bitrate, OPUS_SET_BANDWIDTH: BW[bw]}
    if fec:
        c.update({OPUS_SET_INBAND_FEC: 1, OPUS_SET_PACKET_LOSS_PERC: 20})
    return c


def silk(bitrate, bw, fec=False, **extra):
    c = {OPUS_SET_FORCE_MODE: MODE_SILK_ONLY, OPUS_SET_BITRATE: bitrate, OPUS_SET_BANDWIDTH: BW[bw]}
    if fec:
        c.update({OPUS_SET_INBAND_FEC: 1, OPUS_SET_PACKET_LOSS_PERC: 20})
    c.update(extra)
    return c


# name, signal, enc rate, enc channels, packet us, api rate, api channels, packets, ctl, per-packet hook, lost, notes
CASES = [
    # ---- SILK ----
    ("silk_wb20_mono_48k_loss", "speech", 16000, 1, 20000, 48000, 1, 40, silk(20000, "WB", True),
     None, {6, 13, 14, 22, 31}, "SILK WB 20 ms mono, FEC on, single and burst losses (libopus switches MB -> WB itself)"),
    ("silk_mb20_stereo_48k_loss", "speech", 12000, 2, 20000, 48000, 2, 40, silk(48000, "MB", True, **{str(OPUS_SET_FORCE_CHANNELS): 2}),
     None, {5, 11, 12, 25}, "SILK MB 20 ms forced stereo -> 48 kHz stereo, losses"),
    ("silk_nb60_mono_16kst_loss", "speech", 8000, 1, 60000, 16000, 2, 14, silk(24000, "NB", True),
     None, {4, 9}, "SILK NB 60 ms (3 SILK frames) mono -> 16 kHz stereo, losses (concealed in 20 ms steps)"),
    ("silk_wb40_switch_24k_loss", "speech", 16000, 2, 40000, 24000, 2, 24, silk(24000, "WB", True),
     lambda p: {OPUS_SET_FORCE_CHANNELS: 2 if (p // 5) % 2 == 0 else 1}, {3, 8, 16}, "SILK WB 40 ms, stereo / mono switches, losses"),
    ("silk_wb20_dtx_16k", "speech_dtx", 16000, 1, 20000, 16000, 1, 50, silk(16000, "WB", False, **{str(OPUS_SET_DTX): 1}),
     None, set(), "SILK WB 20 ms with DTX over a 500 ms silence (TOC-only packets run PLC + CNG)"),
    ("silk_10ms_mono_8k_clean", "speech", 16000, 1, 10000, 8000, 1, 60, silk(16000, "WB"),
     None, set(), "SILK WB 10 ms -> 8 kHz, no loss (normal decode vs libopus)"),
    # ---- CELT-only: every frame size, mono / stereo, high and low bitrates (folding), transients, losses ----
    ("celt_fb20_stereo_96k_loss", "music", 48000, 2, 20000, 48000, 2, 40, celt(96000),
     None, {7, 15, 16, 28}, "CELT FB 20 ms stereo 96 kb/s, transients, losses (CELT PLC)"),
    ("celt_fb20_mono_12k_low", "music", 48000, 1, 20000, 48000, 1, 40, celt(12000),
     None, set(), "CELT FB 20 ms mono 12 kb/s: bands skip and fold"),
    ("celt_fb10_mono_48k_loss", "music", 48000, 1, 10000, 48000, 1, 60, celt(48000),
     None, {9, 31}, "CELT FB 10 ms mono 48 kb/s, losses"),
    ("celt_fb10_stereo_24k_low", "music", 48000, 2, 10000, 48000, 2, 60, celt(24000),
     None, set(), "CELT FB 10 ms stereo 24 kb/s (intensity / dual stereo decisions at a low rate)"),
    ("celt_fb5_stereo_128k", "music", 48000, 2, 5000, 48000, 2, 120, celt(128000),
     None, set(), "CELT FB 5 ms stereo 128 kb/s"),
    ("celt_fb5_mono_32k_loss", "music", 48000, 1, 5000, 48000, 1, 120, celt(32000),
     None, {40, 77, 78}, "CELT FB 5 ms mono 32 kb/s, losses"),
    ("celt_fb2p5_mono_64k", "music", 48000, 1, 2500, 48000, 1, 200, celt(64000),
     None, set(), "CELT FB 2.5 ms mono 64 kb/s (LM 0)"),
    ("celt_fb2p5_stereo_96k_loss", "music", 48000, 2, 2500, 48000, 2, 200, celt(96000),
     None, {50, 120, 121}, "CELT FB 2.5 ms stereo 96 kb/s, losses"),
    ("celt_wb20_mono_16k", "music", 16000, 1, 20000, 16000, 1, 30, celt(32000, "WB"),
     None, set(), "CELT WB 20 ms mono at 16 kHz (end band 17, downsampled output)"),
    ("celt_swb20_stereo_24k", "music", 24000, 2, 20000, 24000, 2, 30, celt(48000, "SWB"),
     None, set(), "CELT SWB 20 ms stereo at 24 kHz (end band 19)"),
    # ---- Hybrid (10 / 20 ms) ----
    ("hybrid_swb20_mono_24k_loss", "speech", 24000, 1, 20000, 24000, 1, 40, hybrid(32000, "SWB", True),
     None, {8, 19, 20}, "Hybrid SWB 20 ms mono at 24 kHz, FEC, losses"),
    ("hybrid_fb20_stereo_48k_loss", "speech", 48000, 2, 20000, 48000, 2, 40, hybrid(64000),
     None, {11, 25}, "Hybrid FB 20 ms stereo 64 kb/s, losses"),
    ("hybrid_fb10_mono_48k", "speech", 48000, 1, 10000, 48000, 1, 60, hybrid(40000),
     None, set(), "Hybrid FB 10 ms mono (celt_accum: frame >= 10 ms)"),
    ("hybrid_swb10_stereo_48k_low", "speech", 48000, 2, 10000, 48000, 2, 60, hybrid(28000, "SWB"),
     None, set(), "Hybrid SWB 10 ms stereo 28 kb/s"),
    # ---- mode switches: redundancy frames, transitions, CELT resets ----
    ("switch_modes_48k_stereo_loss", "speech", 48000, 2, 20000, 48000, 2, 54, {OPUS_SET_BITRATE: 64000},
     lambda p: {OPUS_SET_FORCE_MODE: (MODE_SILK_ONLY, MODE_HYBRID, MODE_CELT_ONLY, MODE_SILK_ONLY, MODE_CELT_ONLY, MODE_HYBRID)[(p // 6) % 6],
                OPUS_SET_BANDWIDTH: BW["WB"] if (p // 6) % 6 in (0, 3) else BW["FB"]},
     {13, 30, 37}, "SILK / hybrid / CELT switches every 6 packets (redundancy, transitions), losses next to switches"),
]


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    opus = Opus(sys.argv[1])
    version = opus.lib.opus_get_version_string().decode()
    os.makedirs(OUT, exist_ok=True)
    manifest = ["name\tapi_rate\tapi_channels\tpacket_us\tpackets\tlost\tnotes"]
    for (name, kind, enc_rate, enc_ch, us, api_rate, api_ch, packets, ctl, hook, lost, notes) in CASES:
        frame = enc_rate * us // 1_000_000
        total = frame * packets
        if kind == "music":
            pcm = music(enc_rate, enc_ch, total, seed=len(name))
        elif kind == "speech_dtx":
            pcm = speech(enc_rate, enc_ch, total, seed=len(name), quiet=(0.3, 0.8))
        else:
            pcm = speech(enc_rate, enc_ch, total, seed=len(name))
        enc = opus.encoder(enc_rate, enc_ch, {int(k): v for k, v in ctl.items()})
        dec = opus.decoder(api_rate, api_ch)
        api_frame = api_rate * us // 1_000_000
        stream, ranges, out = bytearray(), [], []
        for p in range(packets):
            if hook:
                for k, v in hook(p).items():
                    opus.ctl(enc, k, v)
            pkt = opus.encode(enc, pcm[p * frame * enc_ch:(p + 1) * frame * enc_ch], frame)
            if p in lost:
                stream += struct.pack("<H", 0xFFFF)
                samples, rng = opus.decode(dec, None, api_frame, api_ch)
                rng = 0
            else:
                stream += struct.pack("<H", len(pkt)) + pkt
                samples, rng = opus.decode(dec, pkt, api_frame, api_ch)
            out += samples
            ranges.append(rng)
        opus.lib.opus_encoder_destroy(ctypes.c_void_p(enc))
        opus.lib.opus_decoder_destroy(ctypes.c_void_p(dec))
        with open(os.path.join(OUT, name + ".opusp"), "wb") as f:
            f.write(bytes(stream))
        with open(os.path.join(OUT, name + ".pcm16"), "wb") as f:
            f.write(struct.pack(f"<{len(out)}h", *out))
        with open(os.path.join(OUT, name + ".rng"), "wb") as f:
            f.write(struct.pack(f"<{len(ranges)}I", *ranges))
        manifest.append(f"{name}\t{api_rate}\t{api_ch}\t{us}\t{packets}\t{','.join(map(str, sorted(lost)))}\t{notes}")
        print(f"{name}: {packets} packets, {len(lost)} lost, {len(out) // api_ch} samples per channel")
    with open(os.path.join(OUT, "manifest.tsv"), "w", newline="\n") as f:
        f.write(f"# {version}, fixed point, intrinsics off\n")
        f.write("\n".join(manifest) + "\n")


if __name__ == "__main__":
    main()
