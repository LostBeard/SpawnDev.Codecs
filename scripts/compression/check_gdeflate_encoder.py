# Checks GDeflateCompressorGpu against Microsoft's reference decoder: every input below is written by our encoder (the
# DemoConsole `gdeflate-encode` command) at every level, decoded by the reference (gdcli, built from source; see
# make_gdeflate_fixtures.py) and compared with the input. Also prints our size next to the reference encoder's at
# level 6 (sizes are not expected to match: a different matcher).
#
# usage: python check_gdeflate_encoder.py <TestData dir> <gdcli.exe> <DemoConsole.exe> [cpu|cuda|opencl] [extra files...]
# Exit 0 when every stream decodes to its input.

import os, subprocess, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_snappy_fixtures import content

def main():
    testdata, cli, console = sys.argv[1], sys.argv[2], sys.argv[3]
    backend = sys.argv[4] if len(sys.argv) > 4 else "cpu"
    extra = sys.argv[5:]
    tmp = tempfile.mkdtemp(prefix="gdenc_")
    inputs = [("text_1", content("text", 1, 8)), ("text_100", content("text", 100, 16)),
              ("text_65535", content("text", 65535, 9)), ("text_65537", content("text", 65537, 11)),
              ("text_300000", content("text", 300_000, 15)), ("noise_200000", content("noise", 200_000, 99)),
              ("struct_1000003", content("struct", 1_000_003, 5)), ("zeros_2000000", bytes(2_000_000)),
              ("mixed_200000", content("mixed", 200_000, 3)), ("far_700000", content("far", 700_000, 9))]
    for rel in ("Big_Buck_Bunny_180_10s.webm", "bbb_first_frame.yuv", "bbb_180_2s.ivf"):
        inputs.append((rel, open(os.path.join(testdata, rel), "rb").read()))
    for path in extra:
        inputs.append((os.path.basename(path), open(path, "rb").read()))
    bad = 0
    for name, data in inputs:
        src = os.path.join(tmp, "in.bin")
        open(src, "wb").write(data)
        ref = os.path.join(tmp, "ref.gdf")
        subprocess.run([cli, "c", "6", src, ref], check=True, capture_output=True)
        sizes = []
        for level in range(5):
            ours = os.path.join(tmp, f"ours{level}.gdf")
            subprocess.run([console, "gdeflate-encode", src, ours, str(level), backend], check=True, capture_output=True)
            back = os.path.join(tmp, "back.bin")
            r = subprocess.run([cli, "d", ours, back], capture_output=True)
            ok = r.returncode == 0 and open(back, "rb").read() == data
            if not ok:
                bad += 1
                print(f"FAIL {name} level {level}: reference {'refused' if r.returncode else 'decoded different bytes'}")
            sizes.append(os.path.getsize(ours))
        print(f"{name:32} {len(data):>10}  ours L0..4 {sizes}  reference L6 {os.path.getsize(ref)}")
    print("all streams decode with the reference" if bad == 0 else f"{bad} streams FAILED")
    return 0 if bad == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
