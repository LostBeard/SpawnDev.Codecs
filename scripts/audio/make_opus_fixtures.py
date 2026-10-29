"""Fixtures for the Opus decoders (Audio/Opus): streams encoded by LIBOPUS and libopus's own decode of them, including lost
packets (PLC / CNG) and DTX.

Truth:
  <name>.opusp      the packet stream: per packet a little-endian uint16 length then the bytes; length 0xFFFF marks a
                    packet that is LOST (the decoder is asked to conceal one packet duration).
  <name>.pcm16      the interleaved int16 PCM libopus's opus_decode() returns for the whole stream, concealment included
                    (decode_fec = 0). The float API of a fixed-point libopus returns exactly these values / 32768.
  manifest.tsv      name, api rate, api channels, packet ms, packets, lost packets, notes
Nothing here comes from SpawnDev.Codecs or Concentus. libopus is 1.5.2 built FIXED POINT with intrinsics off
(cmake -DOPUS_FIXED_POINT=ON -DOPUS_DISABLE_INTRINSICS=ON -DBUILD_SHARED_LIBS=ON), the same arithmetic Concentus ports.

Why libopus and not Concentus: Concentus 2.2.2's silk_CNG never copies the frame's excitation into CNG_exc_buf_Q14
(libopus memmove + memcpy; Concentus only the memmove), so its comfort noise after a loss is wrong, and its enc_shrink
moves the raw-bit tail the wrong way.

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
OPUS_SET_FORCE_MODE = 11002
MODE_SILK_ONLY = 1000
BW = {"NB": 1101, "MB": 1102, "WB": 1103, "SWB": 1104, "FB": 1105}
OPUS_AUTO = -1000


def signal(rate, channels, samples, seed, quiet=(0.4, 0.5)):
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


class Opus:
    def __init__(self, dll):
        self.lib = ctypes.CDLL(dll)
        self.lib.opus_encoder_create.restype = ctypes.c_void_p
        self.lib.opus_encoder_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        self.lib.opus_encode.restype = ctypes.c_int
        self.lib.opus_encode.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_short), ctypes.c_int,
                                         ctypes.c_char_p, ctypes.c_int]
        self.lib.opus_encoder_ctl.restype = ctypes.c_int
        self.lib.opus_decoder_create.restype = ctypes.c_void_p
        self.lib.opus_decoder_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        self.lib.opus_decode.restype = ctypes.c_int
        self.lib.opus_decode.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int,
                                         ctypes.POINTER(ctypes.c_short), ctypes.c_int, ctypes.c_int]
        self.lib.opus_encoder_destroy.argtypes = [ctypes.c_void_p]
        self.lib.opus_decoder_destroy.argtypes = [ctypes.c_void_p]
        self.lib.opus_get_version_string.restype = ctypes.c_char_p

    def encoder(self, rate, channels, **ctl):
        err = ctypes.c_int(0)
        enc = self.lib.opus_encoder_create(rate, channels, OPUS_APPLICATION_VOIP, ctypes.byref(err))
        if err.value != 0:
            raise RuntimeError(f"opus_encoder_create {err.value}")
        for req, val in ctl.items():
            r = self.lib.opus_encoder_ctl(ctypes.c_void_p(enc), ctypes.c_int(int(req)), ctypes.c_int(int(val)))
            if r != 0:
                raise RuntimeError(f"opus_encoder_ctl({req}, {val}) = {r}")
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
        return list(pcm[: n * channels])


# name, enc rate, enc channels, packet ms, api rate, api channels, packets, ctl, per-packet hooks, lost packets, notes
CASES = [
    ("silk_wb20_mono_48k_loss", 16000, 1, 20, 48000, 1, 40,
     {OPUS_SET_BITRATE: 20000, OPUS_SET_BANDWIDTH: BW["WB"], OPUS_SET_INBAND_FEC: 1, OPUS_SET_PACKET_LOSS_PERC: 20},
     None, {6, 13, 14, 22, 31}, "SILK WB 20 ms mono, FEC on, single and burst losses"),
    ("silk_mb20_stereo_48k_loss", 12000, 2, 20, 48000, 2, 40,
     {OPUS_SET_BITRATE: 48000, OPUS_SET_BANDWIDTH: BW["MB"], OPUS_SET_FORCE_CHANNELS: 2, OPUS_SET_INBAND_FEC: 1,
      OPUS_SET_PACKET_LOSS_PERC: 20},
     None, {5, 11, 12, 25}, "SILK MB 20 ms forced stereo -> 48 kHz stereo, losses"),
    ("silk_nb60_mono_16kst_loss", 8000, 1, 60, 16000, 2, 14,
     {OPUS_SET_BITRATE: 24000, OPUS_SET_BANDWIDTH: BW["NB"], OPUS_SET_INBAND_FEC: 1, OPUS_SET_PACKET_LOSS_PERC: 20},
     None, {4, 9}, "SILK NB 60 ms (3 SILK frames) mono -> 16 kHz stereo, losses (concealed in 20 ms steps)"),
    ("silk_wb40_switch_24k_loss", 16000, 2, 40, 24000, 2, 24,
     {OPUS_SET_BITRATE: 24000, OPUS_SET_BANDWIDTH: BW["WB"], OPUS_SET_INBAND_FEC: 1, OPUS_SET_PACKET_LOSS_PERC: 20},
     lambda p: {OPUS_SET_FORCE_CHANNELS: 2 if (p // 5) % 2 == 0 else 1}, {3, 8, 16}, "SILK WB 40 ms, stereo / mono switches, losses"),
    ("silk_wb20_dtx_16k", 16000, 1, 20, 16000, 1, 50,
     {OPUS_SET_BITRATE: 16000, OPUS_SET_BANDWIDTH: BW["WB"], OPUS_SET_DTX: 1},
     None, set(), "SILK WB 20 ms with DTX over a 500 ms silence (TOC-only packets run PLC + CNG)"),
    ("silk_10ms_mono_8k_clean", 16000, 1, 10, 8000, 1, 60,
     {OPUS_SET_BITRATE: 16000, OPUS_SET_BANDWIDTH: BW["WB"]},
     None, set(), "SILK WB 10 ms -> 8 kHz, no loss (normal decode vs libopus)"),
]


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    opus = Opus(sys.argv[1])
    version = opus.lib.opus_get_version_string().decode()
    os.makedirs(OUT, exist_ok=True)
    manifest = ["name\tapi_rate\tapi_channels\tpacket_ms\tpackets\tlost\tnotes"]
    for (name, enc_rate, enc_ch, ms, api_rate, api_ch, packets, ctl, hook, lost, notes) in CASES:
        frame = enc_rate * ms // 1000
        pcm = signal(enc_rate, enc_ch, frame * packets, seed=len(name), quiet=(0.3, 0.8) if "dtx" in name else (0.4, 0.5))
        enc = opus.encoder(enc_rate, enc_ch, **{str(k): v for k, v in {OPUS_SET_FORCE_MODE: MODE_SILK_ONLY, **ctl}.items()})
        dec = opus.decoder(api_rate, api_ch)
        api_frame = api_rate * ms // 1000
        stream = bytearray()
        out = []
        for p in range(packets):
            if hook:
                for k, v in hook(p).items():
                    opus.ctl(enc, k, v)
            pkt = opus.encode(enc, pcm[p * frame * enc_ch:(p + 1) * frame * enc_ch], frame)
            if p in lost:
                stream += struct.pack("<H", 0xFFFF)
                out += opus.decode(dec, None, api_frame, api_ch)
            else:
                stream += struct.pack("<H", len(pkt)) + pkt
                out += opus.decode(dec, pkt, api_frame, api_ch)
        opus.lib.opus_encoder_destroy(ctypes.c_void_p(enc))
        opus.lib.opus_decoder_destroy(ctypes.c_void_p(dec))
        with open(os.path.join(OUT, name + ".opusp"), "wb") as f:
            f.write(bytes(stream))
        with open(os.path.join(OUT, name + ".pcm16"), "wb") as f:
            f.write(struct.pack(f"<{len(out)}h", *out))
        manifest.append(f"{name}\t{api_rate}\t{api_ch}\t{ms}\t{packets}\t{','.join(map(str, sorted(lost)))}\t{notes}")
        print(f"{name}: {packets} packets, {len(lost)} lost, {len(out) // api_ch} samples per channel")
    with open(os.path.join(OUT, "manifest.tsv"), "w", newline="\n") as f:
        f.write(f"# {version}, fixed point, intrinsics off\n")
        f.write("\n".join(manifest) + "\n")


if __name__ == "__main__":
    main()
