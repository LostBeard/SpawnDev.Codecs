"""Fixtures for Yuv420ToRgb16Gpu (ComfyUI LoadVideo's YUV -> RGB). See Docs/comfyui-loadvideo-yuv-to-rgb.md.

1. ComfyUI references -> per-frame MD5s. Input: the IMAGE dumps Lore made with ComfyUI's LoadVideo ->
   GetVideoComponents (float32 N,H,W,3, little-endian) for the Vp9ComfyUI test files. Output, next to each WebM:
   <name>.webm.loadvideo-f32.md5 = one MD5 per frame of that frame's float32 bytes (H x W x 3, interleaved).
2. Odd sizes against the black-box oracle: seeded random yuv420p pictures (TestData/Yuv420Rgb16/yuv_WxH.yuv) and
   the MD5 of ffmpeg's `-sws_flags bilinear -pix_fmt gbrp16le` output re-interleaved as R,G,B uint16 little-endian
   (manifest.tsv: name, width, height, md5).

Usage: python -I make_loadvideo_rgb_fixtures.py <refs dir> <repo root>
"""
import hashlib, os, random, subprocess, sys
from array import array

refs, repo = sys.argv[1], sys.argv[2]
td = os.path.join(repo, 'SpawnDev.Codecs.Demo.Shared', 'TestData')

for name, n, h, w in (('comfyui_vp9_512x512_crf32', 48, 512, 512), ('comfyui_vp9_1024x576_crf32', 48, 576, 1024),
                      ('comfyui_vp9_rgba_512x320_crf32', 12, 320, 512)):
    src = os.path.join(refs, f'{name}_images_f32_{n}x{h}x{w}x3.raw')
    size = h * w * 3 * 4
    lines = []
    with open(src, 'rb') as f:
        for k in range(n):
            lines.append(hashlib.md5(f.read(size)).hexdigest())
    with open(os.path.join(td, 'Vp9ComfyUI', f'{name}.webm.loadvideo-f32.md5'), 'w', newline='\n') as f:
        f.write('\n'.join(lines) + '\n')
    print(name, n, 'frames')

out = os.path.join(td, 'Yuv420Rgb16')
os.makedirs(out, exist_ok=True)
sizes = [(1, 1), (2, 3), (3, 3), (7, 5), (6, 6), (3, 170), (170, 3), (64, 6), (101, 133), (283, 77), (303, 170),
         (513, 67), (130, 67), (1023, 7), (9, 241)]
manifest = []
for w, h in sizes:
    cw, ch = (w + 1) // 2, (h + 1) // 2
    data = random.Random(w * 100003 + h).randbytes(w * h + 2 * cw * ch)
    name = f'yuv_{w}x{h}'
    yuv = os.path.join(out, name + '.yuv')
    open(yuv, 'wb').write(data)
    r = subprocess.run(['ffmpeg', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'yuv420p', '-s', f'{w}x{h}', '-i', yuv,
                        '-sws_flags', 'bilinear', '-f', 'rawvideo', '-pix_fmt', 'gbrp16le', '-'], capture_output=True, check=True)
    a = array('H'); a.frombytes(r.stdout); n = w * h
    rgb = array('H', bytes(6 * n))
    rgb[0::3] = a[2 * n:3 * n]; rgb[1::3] = a[:n]; rgb[2::3] = a[n:2 * n]
    if sys.byteorder != 'little': rgb.byteswap()
    manifest.append(f'{name}\t{w}\t{h}\t{hashlib.md5(rgb.tobytes()).hexdigest()}')
with open(os.path.join(out, 'manifest.tsv'), 'w', newline='\n') as f:
    f.write('\n'.join(manifest) + '\n')
print(len(sizes), 'random pictures')

# 3. WebM load with audio (TestData/LoadVideo): ComfyUI writes Opus only next to AV1 (SaveVideo), which we cannot decode
#    yet, so the ComfyUI file's Opus packets are remuxed unchanged (ffmpeg -c copy, a black box) next to a ComfyUI VP9
#    video. Truth: ComfyUI's AUDIO for those packets (float32 planar 2 x 96000, its own LoadVideo). (Our Opus decoder is
#    libopus FIXED POINT, bit-exact in its own tests; ComfyUI's ffmpeg decodes in float, so the values differ slightly.)
lv = os.path.join(td, 'LoadVideo')
os.makedirs(lv, exist_ok=True)
mux = os.path.join(lv, 'vp9_with_comfyui_opus.webm')
subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', os.path.join(td, 'Vp9ComfyUI', 'comfyui_vp9_512x512_crf32.webm'),
                '-i', os.path.join(refs, 'comfyui_webm_with_opus.webm'), '-map', '0:v', '-map', '1:a', '-c', 'copy',
                '-fflags', '+bitexact', mux], check=True)
comfy = open(os.path.join(refs, 'comfyui_webm_with_opus_audio_f32_1x2x96000.raw'), 'rb').read()
open(os.path.join(lv, 'comfyui_opus_audio_f32_2x96000.raw'), 'wb').write(comfy)
print('LoadVideo: ComfyUI audio', len(comfy) // 8, 'stereo samples')
