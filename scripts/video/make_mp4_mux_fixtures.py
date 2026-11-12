"""Fixtures for Mp4Writer (Container/Mp4), the ISO BMFF muxer for AV1 + Opus.

The ffmpeg command line tool (a black box; its source is not read) encodes short synthetic clips and muxes them into
MP4 with -fflags +bitexact. Each <case>.mp4 is the truth; <case>.json lists what a muxer is given, read back with
ffprobe: per track the codec, the codec configuration (av1C record / OpusHead fields), width / height or sample rate /
channels, the media timescale; per packet the track, pts, dts, duration (track timescale), key flag and the packet's
bytes as (offset, length) in the truth file.

Usage: python -I make_mp4_mux_fixtures.py <out dir>
"""
import json, os, re, subprocess, sys

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
tmp = os.path.join(out, '_tmp')
os.makedirs(tmp, exist_ok=True)

def ff(*args):
    subprocess.run(['ffmpeg', '-v', 'error', '-y', *args], check=True)

def probe(path, *args):
    r = subprocess.run(['ffprobe', '-v', 'error', *args, '-of', 'json', path], capture_output=True, check=True)
    return json.loads(r.stdout)

def video(name, w, h, rate, seconds, gop, extra=()):
    p = os.path.join(tmp, name + '.ivf')
    ff('-f', 'lavfi', '-i', f'testsrc2=size={w}x{h}:rate={rate}:duration={seconds}', '-c:v', 'libaom-av1', '-cpu-used', '8',
       '-g', str(gop), '-crf', '40', '-pix_fmt', 'yuv420p', *extra, p)
    return p

def audio(name, rate, channels, seconds):
    p = os.path.join(tmp, name + '.ogg')
    layout = 'mono' if channels == 1 else 'stereo'
    ff('-f', 'lavfi', '-i', f'sine=frequency=440:sample_rate={rate}:duration={seconds}', '-ac', str(channels),
       '-c:a', 'libopus', '-b:a', '64k', p)
    return p

def mux(case, inputs):
    dst = os.path.join(out, case + '.mp4')
    args = []
    for i in inputs: args += ['-i', i]
    for k in range(len(inputs)): args += ['-map', f'{k}']
    ff(*args, '-c', 'copy', '-fflags', '+bitexact', '-map_metadata', '-1', dst)
    streams = probe(dst, '-show_streams', '-show_entries', 'stream=index,codec_name,codec_type,width,height,sample_rate,channels,time_base,extradata,initial_padding:stream_tags')['streams']
    ext = {s['index']: s for s in probe(dst, '-show_streams', '-show_data')['streams']}
    pk = probe(dst, '-show_packets', '-show_entries', 'packet=stream_index,pts,dts,duration,flags,pos,size')['packets']
    tracks = []
    for s in streams:
        t = {'index': s['index'], 'type': s['codec_type'], 'codec': s['codec_name'], 'timebase': s['time_base']}
        if s['codec_type'] == 'video': t.update(width=s['width'], height=s['height'])
        else: t.update(sampleRate=int(s['sample_rate']), channels=s['channels'], initialPadding=s.get('initial_padding', 0))
        # extradata as hex (ffprobe -show_data prints a hex dump; keep only the bytes)
        dump = ext[s['index']].get('extradata', '')
        hexbytes = ''.join(''.join(tok for tok in line.split(':', 1)[1].split()[:8] if re.fullmatch('[0-9a-f]{2}|[0-9a-f]{4}', tok)) for line in dump.splitlines() if ':' in line)
        t['extradata'] = hexbytes
        tracks.append(t)
    packets = [{'track': p['stream_index'], 'pts': int(p['pts']), 'dts': int(p['dts']), 'duration': int(p['duration']),
                'key': 'K' in p['flags'], 'data': [int(p['pos']), int(p['size'])]} for p in pk]
    json.dump({'tracks': tracks, 'packets': packets}, open(os.path.join(out, case + '.json'), 'w'), indent=0)
    return case

cases = []
v10 = video('v10', 64, 48, 10, 2, 5)
cases.append(mux('av1_only', [v10]))
vkey = video('vkey', 64, 48, 10, 1, 1)
cases.append(mux('av1_keyframes', [vkey]))
a48 = audio('a48', 48000, 2, 2)
cases.append(mux('av1_opus', [v10, a48]))
cases.append(mux('opus_only', [a48]))
a441 = audio('a441', 44100, 1, 2)
cases.append(mux('opus_441_mono', [a441]))
for rate, tag in (('24', '24'), ('25', '25'), ('30000/1001', '2997'), ('16', '16')):
    v = video('v' + tag, 32, 32, rate, 1, 12)
    cases.append(mux('av1_fps' + tag, [v]))
def vp9(name, w, h, rate, seconds, gop):
    p = os.path.join(tmp, name + '.ivf')
    ff('-f', 'lavfi', '-i', f'testsrc2=size={w}x{h}:rate={rate}:duration={seconds}', '-c:v', 'libvpx-vp9', '-deadline', 'realtime',
       '-cpu-used', '8', '-g', str(gop), '-crf', '40', '-b:v', '0', '-pix_fmt', 'yuv420p', p)
    return p

p9 = vp9('p9', 64, 48, 24, 1, 12)
cases.append(mux('vp9_only', [p9]))
cases.append(mux('vp9_opus', [p9, a48]))
p9odd = vp9('p9odd', 70, 50, 25, 1, 25)
cases.append(mux('vp9_odd_size', [p9odd]))
for w, h in ((640, 360), (1280, 720), (1920, 1080)):   # the vpcC level ffmpeg derives from size and rate
    cases.append(mux(f'vp9_level_{h}p30', [vp9(f'p9l{h}', w, h, 30, 0.2, 6)]))
vlong = video('vlong', 32, 32, 24, 12, 48)
along = audio('along', 48000, 2, 12)
cases.append(mux('av1_opus_12s', [vlong, along]))
open(os.path.join(out, 'manifest.txt'), 'w', newline='\n').write('\n'.join(cases) + '\n')
for f in os.listdir(tmp): os.remove(os.path.join(tmp, f))
os.rmdir(tmp)
print(len(cases), 'cases:', ', '.join(cases))
