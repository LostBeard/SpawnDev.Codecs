"""Write TestData/Vp9Conformance/libvpx_modeinfo.tsv.gz: libvpx's per-frame references for the VP9 decoder tests.

Usage: py -3 make_modeinfo_tsv.py <libvpx_ref.exe> <ivf cache dir> <out .tsv.gz> [old .tsv.gz to compare with]

Every VP9 test file of SpawnDev.Codecs.Demo.Shared/TestData (Vp9Conformance, Vp9ComfyUI, Vp9Synthetic,
Vp9ConformanceLarge) is remuxed to IVF once (ffmpeg -c copy -copyinkf, as check_vectors.py) and decoded with
"libvpx_ref -m". Each output line becomes "<file>\\t<line>":
  P <n> <md5> <hash>   the frame before the loop filter       L <n> <md5> <hash>   after it
  M <n> mi=<md5> counts=<md5|->                               A <n> <md5>          context after adaptation
  O <n> <md5> <hash>   output frame n (the frame libvpx shows; was only in the vectors' .md5 files)
<hash> is libvpx_ref.c's frame hash (FrameHash.cs on the accelerator). The script refuses to write when an output
frame's MD5 differs from the file's .md5, or (with an old file) when any MD5 token of the old file changed, so the
new hashes come from frames proven to be libvpx's.
"""
import gzip, os, subprocess, sys

exe, ivf_dir, out = sys.argv[1], sys.argv[2], sys.argv[3]
old = sys.argv[4] if len(sys.argv) > 4 else None
here = os.path.dirname(os.path.abspath(__file__))
data = os.path.normpath(os.path.join(here, "..", "..", "..", "SpawnDev.Codecs.Demo.Shared", "TestData"))
dirs = ["Vp9Conformance", "Vp9ComfyUI", "Vp9Synthetic", "Vp9ConformanceLarge"]
os.makedirs(ivf_dir, exist_ok=True)

files = []
for d in dirs:
    for n in sorted(os.listdir(os.path.join(data, d))):
        if n.endswith((".webm", ".ivf")):
            files.append((n, os.path.join(data, d, n)))
names = [n for n, _ in files]
assert len(names) == len(set(names)), "file names must be unique across the folders"

lines, problems = [], []
for name, src in files:
    ivf = src if name.endswith(".ivf") else os.path.join(ivf_dir, name[:-5] + ".ivf")
    if not os.path.exists(ivf):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-c", "copy", "-copyinkf", "-f", "ivf", ivf], check=True)
    want = [l.split()[0] for l in open(src + ".md5") if l.strip()]
    r = subprocess.run([exe, "-m", ivf], capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        problems.append(f"{name}: exit {r.returncode}: {r.stderr.strip()[:200]}")
        continue
    got = []
    for l in r.stdout.splitlines():
        if not l.strip():
            continue
        if l[:1].isdigit():                       # output frame: "<n> <w>x<h> <md5> <hash>"
            t = l.split()
            got.append(t[2])
            lines.append(f"{name}\tO {t[0]} {t[2]} {t[3]}")
        elif l[:1] in "PLMA":
            lines.append(f"{name}\t{l}")
    if got != want:
        first = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
        problems.append(f"{name}: output MD5s differ from .md5 ({len(got)} vs {len(want)} frames, first at {first + 1})")

if old:
    # every line of the old file must reappear with the same tokens (the new file only appends the hash)
    new_prefix = {}
    for l in lines:
        f, rest = l.split("\t", 1)
        t = rest.split()
        if t[0] in "PL":
            new_prefix[(f, t[0], t[1])] = " ".join(t[:3])
        elif t[0] in "MA":
            new_prefix[(f, t[0], t[1])] = rest
    with gzip.open(old, "rt") as g:
        for l in g:
            f, rest = l.rstrip("\n").split("\t", 1)
            t = rest.split()
            key = (f, t[0], t[1])
            if new_prefix.get(key) != rest:
                problems.append(f"{f}: old line '{rest}' -> new '{new_prefix.get(key)}'")

print(f"{len(files)} files, {len(lines)} lines, {sum(1 for l in lines if chr(9) + 'O ' in l)} output frames")
if problems:
    print(f"{len(problems)} problems, NOT written:")
    for p in problems[:30]:
        print("  ", p)
    sys.exit(1)
with gzip.open(out, "wt", newline="\n") as g:
    for l in lines:
        g.write(l + "\n")
print("written", out)
