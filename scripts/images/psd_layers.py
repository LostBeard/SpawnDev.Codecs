"""Oracle for the PSD writer tests: opens a PSD with psd-tools (MIT licence, the standard PSD reader) and prints what it sees (argv: file outdir).
Exit 3 when psd-tools is not installed (pip install psd-tools into the oracle venv).
Output: <outdir>/layers.json = {width, height, layers: [{name, left, top, right, bottom, visible, opacity, blend, width, height}], composite: [w, h]}, pixels as
raw RGBA8 (straight alpha) in <outdir>/layer{i}.rgba (the layer's own rectangle) and composite.rgba.
Pillow's own PSD plugin is not used: it reads every layer's channels in the order R, G, B, A, while Photoshop and ComfyUI write alpha first, and
it reads only the short name."""
import json, os, sys
try:
    from psd_tools import PSDImage
except ImportError:
    print('MISSING psd-tools')
    sys.exit(3)

path, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
psd = PSDImage.open(path)
composite = psd.topil(apply_icc=False).convert('RGBA') if hasattr(psd, 'topil') else None
info = {'width': psd.width, 'height': psd.height, 'layers': [], 'composite': [composite.width, composite.height]}
open(os.path.join(out, 'composite.rgba'), 'wb').write(composite.tobytes())
for i, l in enumerate(psd):
    im = l.topil()
    rgba = im.convert('RGBA')
    open(os.path.join(out, f'layer{i}.rgba'), 'wb').write(rgba.tobytes())
    info['layers'].append({'name': l.name, 'left': l.left, 'top': l.top, 'right': l.right, 'bottom': l.bottom, 'visible': bool(l.visible),
                           'opacity': int(l.opacity), 'blend': l.blend_mode.value.decode('ascii') if isinstance(l.blend_mode.value, bytes) else str(l.blend_mode.value),
                           'width': rgba.width, 'height': rgba.height})
json.dump(info, open(os.path.join(out, 'layers.json'), 'w', encoding='utf-8'), ensure_ascii=False)
print(len(info['layers']), 'layers')
