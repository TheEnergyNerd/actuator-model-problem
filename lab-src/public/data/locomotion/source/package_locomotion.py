"""Package Isaac Lab videos and replay data for the GitHub Pages viewer."""
import argparse, json, shutil, subprocess, hashlib
from pathlib import Path
import numpy as np
import imageio_ffmpeg
p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('destination',type=Path);a=p.parse_args()
a.destination.mkdir(parents=True,exist_ok=True)
result=json.loads((a.source/'results.json').read_text())
scene=json.loads((a.source/'scene.json').read_text())
# Geometry-only precision reduction to 10 micrometres; recorded poses unchanged.
for mesh in scene['meshes']: mesh['vertices']=[round(v,5) for v in mesh['vertices']]
(a.destination/'scene.json').write_text(json.dumps(scene,separators=(',',':'),allow_nan=False))
shutil.copy2(a.source/'results.json',a.destination/'results.json')
labels={'torque_very_low':('Current-limit stress test','Only 15% of nominal peak current. This deliberately weak motor tests loss of torque authority.'),'kv_very_high':('Kv ×4 stress test','Kv ×4 and Kt ÷4, with resistance and inductance ÷16. A deliberately extreme fixed-copper rewind at the same peak current.'),'nominal':('Nominal motors','Reference actuator model and unchanged trained policy.'),'mass_added_double':('Heavier actuators','Add 0.5 kg per actuator: 18.5 kg across G1. Added carrier mass and inertia enter the physical simulation.'),'kv_high':('Higher Kv winding','Kv ×1.5, Kt ÷1.5, resistance and inductance ÷2.25 for a fixed-copper rewind. Current and torque-speed behavior change together.'),'gear_low':('Lower gearing','Reduce gear ratio to two thirds of nominal. Less joint torque, more available joint speed.'),'torque_low':('Lower current limit','Reduce peak current to 67% of nominal, reducing the available peak motor torque.')}
variants=[]
for i,row in enumerate(result['records']):
 name=row['design']['name'];dest=a.destination/name;dest.mkdir(exist_ok=True)
 for f in ['poses.bin','telemetry.json','result.json']:shutil.copy2(a.source/name/f,dest/f)
 info=json.loads((dest/'result.json').read_text());samples=json.loads((dest/'telemetry.json').read_text());poses=np.fromfile(dest/'poses.bin',dtype='<f4').reshape(info['frames'],info['bodies'],7)
 assert np.isfinite(poses).all() and len(samples)==len(poses)
 assert np.max(np.abs(np.linalg.norm(poses[:,:,3:],axis=-1)-1))<.001
 subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-i',str(a.source/'comparison_raw.mp4'),'-vf',f'crop=800:608:{800*i}:0','-c:v','libx264','-crf','24','-preset','fast','-movflags','+faststart','-an',str(dest/'video.mp4')],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
 label,description=labels[name];variants.append(dict(id=name,label=label,description=description))
manifest=dict(robot='G1' if result['robot']=='g1' else 'ANYmal',variants=variants,physics_hz=round(1/result['physics_dt_s']),fps=info['fps'],push_time=result['push_time_s'],description='Paired Isaac Lab recordings with one trained policy, the same initial physical properties and commands, and a 0.2-second lateral force pulse. These short trials show design sensitivity, not robustness across seeds or hardware efficiency.',geometry_precision_m=.00001,video_fps=info['fps']/3,video_first_physics_time_s=1/info['fps'])
(a.destination/'manifest.json').write_text(json.dumps(manifest,indent=2))
checks={str(f.relative_to(a.destination)):hashlib.sha256(f.read_bytes()).hexdigest() for f in a.destination.rglob('*') if f.is_file() and f.name!='SHA256.json'}
(a.destination/'SHA256.json').write_text(json.dumps(checks,indent=2))
print(json.dumps({'robot':result['robot'],'variants':len(variants),'frames':info['frames'],'bodies':info['bodies'],'mesh_triangles':sum(len(m['indices'])//3 for m in scene['meshes']),'verified_files':len(checks)}))
