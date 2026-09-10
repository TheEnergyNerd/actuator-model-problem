"""Package a detailed Allegro recording without changing its measured motion."""
import argparse,json,hashlib,shutil,zipfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('destination',type=Path);a=p.parse_args();a.destination.mkdir(parents=True,exist_ok=True)
for name in ['poses.bin','telemetry.json','result.json','video.mp4']:
 shutil.copy2(a.source/name,a.destination/name)
scene=json.loads((a.source/'scene.json').read_text())
for mesh in scene['meshes']:mesh['vertices']=[round(v,6) for v in mesh['vertices']]
scene['geometry_precision_m']=.000001
(a.destination/'scene.json').write_text(json.dumps(scene,separators=(',',':'),allow_nan=False))
project=Path(__file__).resolve().parents[1]
with zipfile.ZipFile(a.destination/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
 for name in ['record_dexterous.py','hand_studio.py','locomotion_replay.py','train_hand_precision.py']:
  z.write(project/'eval_scripts'/name,'eval_scripts/'+name)
 for f in (project/'atlas_actuators_ext/atlas_actuators').rglob('*.py'):
  z.write(f,f.relative_to(project))
 z.write(Path(__file__),'tools/package_hand.py')
hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in a.destination.iterdir() if f.is_file() and f.name!='SHA256.json'}
(a.destination/'SHA256.json').write_text(json.dumps(hashes,indent=2))
print('Packaged',len(scene['bodies']),'rigid bodies;',sum(len(m['indices'])//3 for m in scene['meshes']),'mesh triangles')
