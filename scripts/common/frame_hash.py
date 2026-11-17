"""FrameHash: the 64-bit test hash an accelerator computes in parallel (SpawnDev.Codecs.Demo.Shared/UnitTests/FrameHash.cs,
scripts/video/libvpx_ref/libvpx_ref.c fh_*), for fixture scripts that write references.

The bytes are cut into 4096-byte chunks; chunk c gets two MurmurHash3 x86_32 values, seeds c * 0x9E3779B9 + 0x243F6A88
and c * 0x9E3779B9 + 0x85A308D3; the A values (little-endian words) are then hashed with seed = length and the B values
with seed = length ^ 0x13198A2E. Returned as "%08x%08x" (A, B). Not cryptographic: it detects a wrong output.
"""
import struct

M = 0xFFFFFFFF
CHUNK = 4096


def _rotl(x, r):
    return ((x << r) | (x >> (32 - r))) & M


def _murmur(data, seed):
    h = seed & M
    n = len(data)
    nb = n // 4
    for (k,) in struct.iter_unpack("<I", data[: nb * 4]):
        k = (k * 0xCC9E2D51) & M
        k = _rotl(k, 15)
        k = (k * 0x1B873593) & M
        h ^= k
        h = _rotl(h, 13)
        h = (h * 5 + 0xE6546B64) & M
    tail = data[nb * 4:]
    k = 0
    if len(tail) == 3:
        k ^= tail[2] << 16
    if len(tail) >= 2:
        k ^= tail[1] << 8
    if len(tail) >= 1:
        k ^= tail[0]
        k = (k * 0xCC9E2D51) & M
        k = _rotl(k, 15)
        k = (k * 0x1B873593) & M
        h ^= k
    h ^= n
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & M
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & M
    h ^= h >> 16
    return h


def frame_hash(data):
    data = bytes(data)
    n = len(data)
    a, b = [], []
    for c in range((n + CHUNK - 1) // CHUNK):
        chunk = data[c * CHUNK:(c + 1) * CHUNK]
        base = (c * 0x9E3779B9) & M
        a.append(_murmur(chunk, base + 0x243F6A88))
        b.append(_murmur(chunk, base + 0x85A308D3))
    ha = _murmur(struct.pack("<%dI" % len(a), *a), n & M)
    hb = _murmur(struct.pack("<%dI" % len(b), *b), (n & M) ^ 0x13198A2E)
    return "%08x%08x" % (ha, hb)
