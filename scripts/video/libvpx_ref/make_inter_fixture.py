"""Make TestData/Vp9Dsp/inter.gz: libvpx v1.17.0's VP9 inter prediction (dec_build_inter_predictors_sb) of
deterministic pseudo-random blocks, the reference for Vp9InterPred (CodecsTestBase.Vp9DspInterTests.cs).

Usage: py -3 make_inter_fixture.py <libvpx source dir> <libvpx build dir> <work dir>

The build dir is the one build_libvpx_ref.py compiled (its cl.rsp lists the sources); inter_ref.c is linked against
those objects (all but libvpx_ref.c's), with the static prediction functions of vp9/decoder/vp9_decodeframe.c copied
verbatim from the source into <work dir>/inter_dec_copy.h. The cases (fixed seed): every BLOCK_SIZE incl. 4x4 / 4x8 /
8x4 with four sub-block MVs, every interpolation filter, single and compound, MVs from zero over fractional and
full-pel to far outside the frame (every edge and corner, the MV range limits), blocks at frame edges at legal
partition positions, frame sizes that are and are not multiples of 8, 4:2:0 plus 4:4:4 / 4:2:2 / 4:4:0 (the q2 / q4
chroma MV averages), references of the same size and scaled (2x, 1.5x, 0.75x, one axis only, 16x upscale).
Reference pixels are not stored: the test regenerates them from (seed, kind) with inter_ref.c's LCG.
"""
import glob, gzip, os, random, re, subprocess, sys

src, build, work = (os.path.abspath(p) for p in sys.argv[1:4])
here = os.path.dirname(os.path.abspath(__file__))
repo = os.path.abspath(os.path.join(here, "..", "..", ".."))
os.makedirs(work, exist_ok=True)

# ---------------------------------------------------------------- verbatim copy of the decoder's static functions
text = open(os.path.join(src, "vp9", "decoder", "vp9_decodeframe.c"), newline="").read().replace("\r\n", "\n")


def span(start, end):
    assert text.count(start) == 1, start
    a = text.index(start)
    b = text.index(end, a)
    return text[a:b]


copy = ("/* copied verbatim by make_inter_fixture.py from libvpx v1.17.0 vp9/decoder/vp9_decodeframe.c (BSD-3-Clause) */\n"
        + span("static void set_plane_n4(", "\nstatic MODE_INFO *set_offsets_recon(") + "\n"
        + span("static void build_mc_border(", "static INLINE void dec_reset_skip_context("))
assert "dec_build_inter_predictors_sb" in copy and "extend_and_predict" in copy
open(os.path.join(work, "inter_dec_copy.h"), "w", newline="\n").write(copy)

# ---------------------------------------------------------------- build
rsp = open(os.path.join(build, "cl.rsp")).read()
objs = []
for s in re.findall(r'^"([^"]+\.c)"$', rsp, re.M):
    name = os.path.splitext(os.path.basename(s))[0]
    if name == "libvpx_ref":
        continue
    obj = os.path.join(build, name + ".obj")
    assert os.path.exists(obj), obj
    objs.append(obj)
exe = os.path.join(work, "inter_ref.exe")
vcvars = sorted(glob.glob(r"C:\Program Files\Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvars64.bat"))[-1]
with open(os.path.join(work, "inter_cl.rsp"), "w") as f:
    f.write("/nologo /O2 /W3 /DNDEBUG /D_CRT_SECURE_NO_WARNINGS\n")
    f.write(f'/I"{work}" /I"{build}" /I"{src}"\n')
    f.write(f'/Fo"{os.path.join(work, "inter_ref.obj")}" /Fe"{exe}"\n')
    f.write(f'"{os.path.join(here, "inter_ref.c")}"\n')
    for o in objs:
        f.write(f'"{o}"\n')
with open(os.path.join(work, "inter_build.bat"), "w") as f:
    f.write(f'@call "{vcvars}" >nul\r\ncl @"{os.path.join(work, "inter_cl.rsp")}"\r\n')
r = subprocess.run(["cmd", "/c", os.path.join(work, "inter_build.bat")], capture_output=True, text=True)
if r.returncode != 0:
    print(r.stdout[-4000:])
    sys.exit(f"build failed ({r.returncode})")

# ---------------------------------------------------------------- cases
rnd = random.Random(0x5639_1E7E)
# (ssx, ssy, cur w, cur h, (ref0 w, h, kind), (ref1 w, h, kind)); kind 0 noise, 1 gradient + noise, 2 black / white
SCENARIOS = [
    (1, 1, 70, 45, (70, 45, 0), (70, 45, 2)),       # unscaled, sizes not multiples of 8
    (1, 1, 64, 48, (64, 48, 1), (64, 48, 0)),       # unscaled, multiples of 8 (zero MV skips border extension)
    (1, 1, 70, 45, (140, 90, 0), (105, 67, 1)),     # reference 2x and 1.5x larger
    (1, 1, 70, 45, (52, 33, 1), (70, 90, 0)),       # 0.75x; y-only 2x (x_step_q4 16, y scaled)
    (1, 1, 160, 96, (10, 6, 0), (140, 96, 1)),      # 16x upscale; x-only 0.875x
    (1, 1, 37, 23, (74, 46, 2), (37, 23, 1)),       # tiny odd frame, 2x
    (0, 0, 50, 30, (50, 30, 0), (75, 45, 1)),       # 4:4:4
    (1, 0, 50, 30, (50, 30, 1), (75, 45, 0)),       # 4:2:2 (sub8x8 chroma: mi_mv_pred_q2 over block, block + 1)
    (0, 1, 50, 30, (50, 30, 0), (38, 23, 2)),       # 4:4:0 (mi_mv_pred_q2 over block, block + 2)
    (1, 1, 128, 72, (256, 144, 1), (64, 36, 0)),    # 2x down, 2x up
    (1, 1, 33, 17, (66, 34, 0), (17, 9, 1)),        # odd tiny, 2x and ~0.5x
    (1, 1, 96, 64, (96, 64, 0), (192, 128, 2)),     # multiples of 8, unscaled + 2x
    (1, 1, 40, 24, (40, 24, 0), (40, 24, 1)),       # ZERO_HEAVY: zero MVs on blocks hanging over the frame edge
    (0, 0, 24, 16, (24, 16, 1), (24, 16, 0)),       # ZERO_HEAVY, 4:4:4
]
# scenarios where most MV components are 0: unscaled, crop sizes multiples of 8 and no motion is the one path where
# libvpx reads its unfilled frame border (no build_mc_border), for the pixels of a block outside the visible frame
ZERO_HEAVY = {12, 13}
CASES_PER_SCENARIO = 130
N8W = [1, 1, 1, 1, 1, 2, 2, 2, 4, 4, 4, 8, 8]
N8H = [1, 1, 1, 1, 2, 1, 2, 4, 2, 4, 8, 4, 8]
BSIZE_WEIGHTS = [7, 5, 5, 6, 4, 4, 5, 3, 3, 3, 1.5, 1.5, 1.5]
MV_MAX = (1 << 14) - 1   # MV_UPP - 1: the decoder rejects |component| >= 1 << 14


def pack(row, col):
    v = (row & 0xFFFF) | ((col & 0xFFFF) << 16)
    return v - (1 << 32) if v >= 1 << 31 else v


def legal_block(mi_rows, mi_cols):
    """A block decode_partition can produce: a parent square of S mi at an S-aligned position with
    has_rows / has_cols deciding which partitions read_partition allows."""
    while True:
        bsize = rnd.choices(range(13), BSIZE_WEIGHTS)[0]
        bw, bh = N8W[bsize], N8H[bsize]
        S = max(bw, bh)
        r = rnd.randrange(0, (mi_rows + S - 1) // S) * S
        c = rnd.randrange(0, (mi_cols + S - 1) // S) * S
        if rnd.random() < 0.5:   # bias toward the right / bottom edges
            r = ((mi_rows - 1) // S) * S if rnd.random() < 0.5 else r
            c = ((mi_cols - 1) // S) * S if rnd.random() < 0.5 else c
        hbs = S // 2
        has_rows, has_cols = r + hbs < mi_rows, c + hbs < mi_cols
        if S == 1:
            return bsize, r, c
        if bw == bh:
            if has_rows and has_cols:
                return bsize, r, c
        elif bw > bh:   # PARTITION_HORZ: needs has_cols; the second half only when has_rows
            if has_cols:
                if has_rows and rnd.random() < 0.5:
                    r += hbs
                return bsize, r, c
        else:           # PARTITION_VERT: needs has_rows; the second half only when has_cols
            if has_rows:
                if has_cols and rnd.random() < 0.5:
                    c += hbs
                return bsize, r, c


def mv_comp(pos_px, size_px, ref_px):
    k = rnd.random()
    if k < 0.15:
        v = 0
    elif k < 0.40:
        v = rnd.randint(-64, 64)
    elif k < 0.55:
        v = 8 * rnd.randint(-16, 16)
    elif k < 0.78:   # land the block on / across an edge of the reference, or just outside it
        target = rnd.choice([-size_px - 6, -size_px, -size_px // 2, -4, -1, 0, 2,
                             ref_px - size_px - 2, ref_px - size_px // 2, ref_px - 1, ref_px + 3, ref_px + size_px])
        v = (target - pos_px) * 8 + rnd.randint(-7, 7)
    elif k < 0.92:
        v = rnd.randint(-MV_MAX, MV_MAX)
    else:
        v = rnd.choice([-MV_MAX, MV_MAX, -MV_MAX + 3, MV_MAX - 5, -(1 << 13), 1 << 13])
    return max(-MV_MAX, min(MV_MAX, v))


lines, cases = [], []
for si, (ssx, ssy, cw, ch, r0, r1) in enumerate(SCENARIOS):
    lines.append(f"{ssx} {ssy} {cw} {ch} {r0[0]} {r0[1]} {1000 + 2 * si} {r0[2]} {r1[0]} {r1[1]} {1001 + 2 * si} {r1[2]}")
    mi_rows, mi_cols = (ch + 7) >> 3, (cw + 7) >> 3
    for _ in range(CASES_PER_SCENARIO):
        bsize, r, c = legal_block(mi_rows, mi_cols)
        filt = rnd.randrange(4)
        compound = 1 if rnd.random() < 0.4 else 0
        bw_px, bh_px = 8 * N8W[bsize], 8 * N8H[bsize]
        mvs, bmis = [], []
        for ref in range(2):
            rw, rh = (r0, r1)[ref][:2]

            def mv_pair():
                if si in ZERO_HEAVY and rnd.random() < 0.6:
                    return pack(0, 0)
                return pack(mv_comp(r * 8, bh_px, rh), mv_comp(c * 8, bw_px, rw))
            sub = [mv_pair() for _ in range(4)]
            if bsize == 1:     # BLOCK_4X8: bmi[2] = bmi[0], bmi[3] = bmi[1]
                sub[2], sub[3] = sub[0], sub[1]
            elif bsize == 2:   # BLOCK_8X4: bmi[1] = bmi[0], bmi[3] = bmi[2]
                sub[1], sub[3] = sub[0], sub[2]
            if bsize < 3:
                mv = sub[3]    # read_inter_block_mode_info: mi->mv = bmi[3]
            else:
                mv = mv_pair()
                sub = [mv] * 4
            mvs.append(mv)
            bmis.append(sub)
        cases.append(f"{si} {r} {c} {bsize} {filt} {compound} {mvs[0]} {mvs[1]} "
                     + " ".join(str(v) for v in bmis[0] + bmis[1]))
params = os.path.join(work, "inter_params.txt")
with open(params, "w") as f:
    f.write(f"{len(SCENARIOS)} {len(cases)}\n" + "\n".join(lines) + "\n" + "\n".join(cases) + "\n")
out = os.path.join(work, "inter_out.bin")
r = subprocess.run([exe, params, out], capture_output=True, text=True)
print(r.stdout.strip(), r.stderr.strip())
if r.returncode != 0:
    sys.exit(f"inter_ref failed ({r.returncode})")
raw = open(out, "rb").read()
dst = os.path.join(repo, "SpawnDev.Codecs.Demo.Shared", "TestData", "Vp9Dsp", "inter.gz")
with open(dst, "wb") as f:
    with gzip.GzipFile(fileobj=f, mode="wb", compresslevel=9, mtime=0, filename="") as g:
        g.write(raw)
print(f"{len(cases)} cases, {len(raw)} bytes raw, {os.path.getsize(dst)} bytes gz -> {dst}")
