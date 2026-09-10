"""Record measured Isaac Lab rigid-body trajectories alongside evaluator video."""
import json
from pathlib import Path
import numpy as np
import torch

class LocomotionRecorder:
    def __init__(self, output, env, cases):
        self.output, self.env, self.cases = Path(output), env, cases
        self.robot = env.scene['robot']
        self.frames, self.samples = [], [[] for _ in cases]
        self.export_mesh()

    def export_mesh(self):
        import omni.usd
        from pxr import Usd, UsdGeom, UsdPhysics, Gf, UsdShade
        stage = omni.usd.get_context().get_stage()
        root = stage.GetPrimAtPath('/World/envs/env_0/Robot')
        paths = {p.GetName(): p for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
                 if p.HasAPI(UsdPhysics.RigidBodyAPI) and p.GetName() in self.robot.body_names}
        meshes = []
        for i, name in enumerate(self.robot.body_names):
            body = paths[name]
            world = UsdGeom.Xformable(body).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            rigid = Gf.Matrix4d(1)
            rigid.SetRotate(Gf.Transform(world).GetRotation())
            rigid.SetTranslateOnly(world.ExtractTranslation())
            for p in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
                if not p.IsA(UsdGeom.Mesh) or 'collision' in str(p.GetPath()).lower(): continue
                if UsdGeom.Imageable(p).ComputeVisibility() == 'invisible': continue
                mesh = UsdGeom.Mesh(p)
                points, counts, indices = mesh.GetPointsAttr().Get(), mesh.GetFaceVertexCountsAttr().Get(), mesh.GetFaceVertexIndicesAttr().Get()
                if not points or not counts: continue
                transform = UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default()) * rigid.GetInverse()
                vertices = [float(v) for pt in points for v in transform.Transform(Gf.Vec3d(pt))]
                faces, offset = [], 0
                for count in counts:
                    for j in range(1, count-1): faces.extend([int(indices[offset]), int(indices[offset+j]), int(indices[offset+j+1])])
                    offset += count
                color = [.72,.76,.69]
                try:
                    material = UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial()[0]
                    shader = material.ComputeSurfaceSource()[0]
                    for key in ['diffuse_color_constant','diffuseColor']:
                        value = shader.GetInput(key).Get()
                        if value is not None:
                            color = list(value)[:3]; break
                except Exception: pass
                meshes.append(dict(body=i, vertices=vertices, indices=faces, color=color))
        if not meshes: raise RuntimeError('No robot visual meshes exported')
        scene = dict(kind='locomotion', bodies=self.robot.body_names, meshes=meshes, cube_size=0, table=None, follow_body=0)
        (self.output/'scene.json').write_text(json.dumps(scene, separators=(',',':'), allow_nan=False))

    def capture(self, t, target, latest, failures, force):
        r = self.robot
        poses = torch.cat((r.data.body_pos_w - self.env.scene.env_origins[:,None,:], r.data.body_quat_w), -1).detach().cpu().numpy()
        if not np.isfinite(poses).all(): raise RuntimeError('Non-finite recorded pose')
        self.frames.append(poses)
        for i in range(len(self.cases)):
            actuators = list(r.actuators.values())
            row = dict(t=t, phase='Force pulse' if abs(float(force[i,0,1])) > 0 else 'Walk',
                cube=poses[i,0,:3].tolist(), tips=[], grip=[],
                temperature=max(float(x._winding_T[i].max()) for x in actuators),
                torque=max(float(x.applied_effort[i].abs().max()) for x in actuators),
                requested_torque=max(float(x.computed_effort[i].abs().max()) for x in actuators),
                current=max(float(x._iq[i].abs().max()) for x in actuators),
                saturation=float(latest['torque_shortfall_fraction'][i]),
                speed=float(r.data.root_lin_vel_b[i,0]), target_speed=target,
                failures=int(failures[i]), force_N=float(force[i,0,1]),
                foot_slip=float(latest['contact_slip_m_s'][i]) if 'contact_slip_m_s' in latest else None)
            self.samples[i].append(row)

    def finish(self, result):
        poses = np.asarray(self.frames, dtype='<f4')
        for i, case in enumerate(self.cases):
            dest = self.output/case.name
            dest.mkdir()
            poses[:,i].tofile(dest/'poses.bin')
            (dest/'telemetry.json').write_text(json.dumps(self.samples[i], allow_nan=False))
            info = dict(result['records'][i], frames=len(poses), bodies=poses.shape[2], fps=1/self.env.step_dt,
                        duration=(len(poses)-1)*self.env.step_dt, physics_dt_s=result['physics_dt_s'],
                        checkpoint_sha256=result['checkpoint_sha256'], video_start_physics_s=self.env.step_dt,
                        reset_policy='Normal episode resets retained; failure counter is shown.', seed=result['seed'])
            (dest/'result.json').write_text(json.dumps(info, indent=2, allow_nan=False))
