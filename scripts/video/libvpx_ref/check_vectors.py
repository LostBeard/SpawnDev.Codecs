"""Decode libvpx's VP9 test vectors with a decoder program and compare every frame's MD5 with the vector's .md5 file.

Usage: check_vectors.py <decoder.exe> <vectors dir> <ivf cache dir> [name prefix, default vp90-2-]
The decoder is run as "<decoder> in.ivf" and must print "<n> <w>x<h> <md5>" per output frame (libvpx_ref.c's format).
.webm vectors are remuxed to IVF once with "ffmpeg -c copy -copyinkf" (-copyinkf: vp90-2-16-intra-only has no keyframe and
plain stream copy drops everything before one) (stream copy: the frame bytes are not touched; a decoder
that matches every .md5 also proves the remux).
"""
import os, subprocess, sys

exe, vec_dir, ivf_dir = sys.argv[1].split("|"), sys.argv[2], sys.argv[3]  # "exe|arg|arg": extra decoder args
prefix = sys.argv[4] if len(sys.argv) > 4 else "vp90-2-"
os.makedirs(ivf_dir, exist_ok=True)
names = sorted(n for n in os.listdir(vec_dir) if n.startswith(prefix) and n.endswith((".webm", ".ivf")))
ok, bad = 0, []
for name in names:
    src = os.path.join(vec_dir, name)
    ivf = src if name.endswith(".ivf") else os.path.join(ivf_dir, name[:-5] + ".ivf")
    if not os.path.exists(ivf):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-c", "copy", "-copyinkf", "-f", "ivf", ivf], check=True)
    want = [l.split()[0] for l in open(src + ".md5") if l.strip()]
    r = subprocess.run(exe + [ivf], capture_output=True, text=True, timeout=600)
    got = [l.split()[2] for l in r.stdout.splitlines() if l.strip() and not l.startswith("H ")]
    if r.returncode != 0:
        bad.append(f"{name}: exit {r.returncode}: {r.stderr.strip()[:200]}")
    elif got != want:
        first = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
        bad.append(f"{name}: {len(got)} frames vs {len(want)}, first mismatch at frame {first + 1}")
    else:
        ok += 1
print(f"{ok}/{len(names)} vectors match every frame")
for b in bad:
    print("  ", b)
sys.exit(1 if bad else 0)
