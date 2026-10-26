# Truth fixtures for Compression/Filters (ShuffleFilterGpu): byte shuffle and bitshuffle outputs made by the REFERENCE
# libraries, never by our code.
#
#   byte shuffle : numcodecs.Shuffle (c-blosc's shuffle). numcodecs rejects a buffer that is not a whole number of
#                  elements, so a trailing partial element is appended raw, exactly as c-blosc2 shuffle_generic_inline
#                  does (memcpy of blocksize % typesize bytes at blocksize - rem).
#   bitshuffle   : the bitshuffle library (kiyo-masui/bitshuffle) through its HDF5 filter in hdf5plugin, no compression
#                  (cname='none'): the chunk bytes are bshuf_bitshuffle(in, out, size, elem_size, block_size).
#
# Inputs are procedural (gen() below; ShuffleFilterGpuTests.FilterInput rebuilds the same bytes), so only the truth is
# stored: TestData/Filters/<case>.bin for the small cases, and SHA-256 lines in large.txt for the 16 MiB cases.
#
# usage: python make_filter_fixtures.py <out dir>     (needs numpy, numcodecs, h5py, hdf5plugin)

import hashlib, io, os, sys
import numpy as np, numcodecs, h5py, hdf5plugin

def gen(n, seed):
    i = np.arange(n, dtype=np.uint64)
    x = (i * np.uint64(2654435761) + np.uint64(seed * 0x9E37)) & np.uint64(0xFFFFFFFF)
    v = ((x >> np.uint64(13)) ^ (i >> np.uint64(9)) ^ np.uint64(seed)) & np.uint64(0xFF)
    return v.astype(np.uint8).tobytes()

def shuffle_ref(raw, e):
    whole = len(raw) // e * e
    body = bytes(numcodecs.Shuffle(elementsize=e).encode(raw[:whole])) if whole else b''
    return body + raw[whole:]

def bitshuffle_ref(raw, e, block):
    n = len(raw) // e
    arr = np.frombuffer(raw, dtype=f'V{e}') if e not in (1, 2, 4, 8) else np.frombuffer(raw, dtype={1: 'u1', 2: '<u2', 4: '<u4', 8: '<u8'}[e])
    bio = io.BytesIO()
    with h5py.File(bio, 'w') as f:
        d = f.create_dataset('d', data=arr, chunks=(n,), **hdf5plugin.Bitshuffle(nelems=block, cname='none'))
        f.flush()
        _, chunk = d.id.read_direct_chunk((0,))
    chunk = bytes(chunk)
    assert len(chunk) == len(raw), (len(chunk), len(raw))
    return chunk

def main(out):
    os.makedirs(out, exist_ok=True)
    count = 0
    for e in (1, 2, 3, 4, 8, 12, 16):
        for nbytes in (0, 1, 17, 1000, 4099, 65537):
            seed = e * 31 + nbytes % 97
            open(os.path.join(out, f'shuf_e{e}_b{nbytes}.bin'), 'wb').write(shuffle_ref(gen(nbytes, seed), e))
            count += 1
    for e in (1, 2, 3, 4, 8, 12):
        for n in (1, 7, 8, 17, 1000, 2048, 4103, 9000):
            for block in ((0, 64) if (e == 4 and n >= 1000) else (0,)):
                seed = e * 7 + n % 89 + block
                open(os.path.join(out, f'bshuf_e{e}_n{n}_bs{block}.bin'), 'wb').write(bitshuffle_ref(gen(n * e, seed), e, block))
                count += 1
    big = 16 << 20
    with open(os.path.join(out, 'large.txt'), 'w', newline='\n') as f:
        for op, e in (('shuf', 4), ('bshuf', 4), ('bshuf', 2)):
            seed = 1000 + e
            raw = gen(big, seed)
            t = shuffle_ref(raw, e) if op == 'shuf' else bitshuffle_ref(raw, e, 0)
            f.write(f'{op} {e} {big} {seed} {hashlib.sha256(t).hexdigest()}\n')
            count += 1
    print(f'{count} fixtures -> {out}')

if __name__ == '__main__':
    main(sys.argv[1])
