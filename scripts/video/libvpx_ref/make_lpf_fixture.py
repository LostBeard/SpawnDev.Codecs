"""VP9 loop filter fixtures: libvpx v1.17.0's own C loop filter (BSD-3-Clause) on generated cases.

Usage: py -3 make_lpf_fixture.py <libvpx source dir> <libvpx build dir>

The build dir is build_libvpx_ref.py's (generated headers + the compiled libvpx objects). lpf_ref.c is compiled with
cl.exe from the newest Visual Studio's vcvars64.bat and linked with those objects, then run:
  lpf_ref edge -> TestData/Vp9Dsp/lpf.gz     every vpx_lpf_* edge filter
  lpf_ref sb   -> TestData/Vp9Dsp/lpf_sb.gz  whole superblocks as the decoder filters them
(single extensions: "x.bin.gz" would go into a satellite assembly, "bin" being a culture name).
"""
import glob, gzip, os, subprocess, sys

src, build = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
here = os.path.dirname(os.path.abspath(__file__))
repo = os.path.abspath(os.path.join(here, "..", "..", ".."))
out_dir = os.path.join(repo, "SpawnDev.Codecs.Demo.Shared", "TestData", "Vp9Dsp")
EDGE_CASES, SB_CASES = 12000, 2400

# libvpx_ref.obj holds libvpx_ref.c's main. The build dir can also hold objects of an earlier build_libvpx_ref.py
# that its current source list no longer compiles: vp9_decodeframe.obj (now patched_vp9_decodeframe.obj; both
# define vp9_decode_frame) and the files its EXCLUDED set leaves out of a decoder-only build (they reference
# encoder-only functions). Neither is part of the decoder these fixtures come from.
STALE = {"vp9_decodeframe.obj", "vp9_mfqe.obj", "vp9_postproc.obj", "variance.obj", "vpx_scale.obj", "gen_scalers.obj",
         "psnr.obj", "psnrhvs.obj", "ssim.obj"}
objs = [o for o in sorted(glob.glob(os.path.join(build, "*.obj")))
        if os.path.basename(o) not in {"libvpx_ref.obj", "lpf_ref.obj"} | STALE]
work = os.path.join(build, "lpf")
os.makedirs(work, exist_ok=True)
exe = os.path.join(work, "lpf_ref.exe")
vcvars = sorted(glob.glob(r"C:\Program Files\Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvars64.bat"))[-1]
rsp = os.path.join(work, "cl.rsp")
with open(rsp, "w") as f:
    f.write("/nologo /O2 /W3 /DNDEBUG /D_CRT_SECURE_NO_WARNINGS\n")
    f.write(f'/I"{build}" /I"{src}"\n')
    f.write(f'/Fo"{os.path.join(work, "lpf_ref.obj")}" /Fe"{exe}"\n')
    f.write(f'"{os.path.join(here, "lpf_ref.c")}"\n')
    for o in objs:
        f.write(f'"{o}"\n')
bat = os.path.join(work, "build.bat")
with open(bat, "w") as f:
    f.write(f'@call "{vcvars}" >nul\r\ncl @"{rsp}"\r\n')
r = subprocess.run(["cmd", "/c", bat], capture_output=True, text=True)
problems = [l for l in r.stdout.splitlines() if "error" in l or "warning" in l or "LNK" in l]
print("\n".join(problems[:40]))
if r.returncode != 0:
    sys.exit(f"build FAILED ({r.returncode})")

for mode, count, name in (("edge", EDGE_CASES, "lpf.gz"), ("sb", SB_CASES, "lpf_sb.gz")):
    raw = os.path.join(work, mode + ".raw")
    subprocess.run([exe, mode, str(count), raw], check=True)
    data = open(raw, "rb").read()
    dst = os.path.join(out_dir, name)
    # mtime 0: the same cases give the same bytes
    with open(dst, "wb") as f, gzip.GzipFile(fileobj=f, mode="wb", compresslevel=9, mtime=0, filename="") as g:
        g.write(data)
    print(f"{name}: {len(data)} bytes raw, {os.path.getsize(dst)} gzipped")
