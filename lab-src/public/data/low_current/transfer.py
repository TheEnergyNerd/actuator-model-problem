"""Continuous Isaac Lab contact-driven bimanual pick, rotate, handoff, place.
Only initial joint state is written. During the episode, only bounded actuator
commands are applied. The object is never attached, teleported, or reset.
"""
import argparse
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True)
parser.add_argument('--variant', default='nominal', choices=['nominal','high_kv','low_kv','low_current','heavy','hot','low_voltage'])
parser.add_argument('--duration', type=float, default=30.0)
parser.add_argument('--export-mesh', action='store_true')
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
import json, math, time
import numpy as np
import torch
from pxr import Usd, UsdGeom, Gf, UsdShade
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils.math import quat_apply, quat_mul, quat_conjugate, skew_symmetric_matrix
from isaaclab_assets import FRANKA_PANDA_CFG
from atlas_actuators.foc_actuator import AtlasActuatorFOCCfg

out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
dt = .0025
sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt, device=args.device, render_interval=8))
sim.set_camera_view([1.65, 1.6, 1.4], [.4, 0, .55])
mat = sim_utils.RigidBodyMaterialCfg(static_friction=1., dynamic_friction=.8, restitution=0.)
sim_utils.GroundPlaneCfg().func('/World/Ground', sim_utils.GroundPlaneCfg())
light = sim_utils.DomeLightCfg(intensity=1800., color=(.85,.9,1.))
light.func('/World/Light', light)
table = sim_utils.CuboidCfg(size=(1.35,1.65,.08), collision_props=sim_utils.CollisionPropertiesCfg(), physics_material=mat, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.16,.2,.23)))
table.func('/World/Table', table, translation=(.35,0,.26))
robots=[]; bases=[]
scale_kv = {'high_kv':2., 'low_kv':.5}.get(args.variant,1.)
scale_i = .12 if args.variant=='low_current' else 1.
for name,y in [('Left',-.55),('Right',.55)]:
    cfg = FRANKA_PANDA_CFG.replace(prim_path='/World/'+name)
    cfg.init_state.pos=(0.,y,.3)
    cfg.spawn.activate_contact_sensors=True
    cfg.spawn.articulation_props.solver_position_iteration_count=16
    cfg.spawn.articulation_props.solver_velocity_iteration_count=4
    cfg.actuators={
      'shoulder':AtlasActuatorFOCCfg(joint_names_expr=['panda_joint[1-4]'], stiffness=600.,damping=35., effort_limit=1000.,effort_limit_sim=1000.,velocity_limit=10.,velocity_limit_sim=10.,armature=.10,sim_dt=dt,pole_pairs=10,Rs_ohm=.03/scale_kv**2,Ld_H=85e-6/scale_kv**2,Lq_H=85e-6/scale_kv**2,lambda_m_Wb=(.25/15)/scale_kv,gear_ratio=12.,I_peak_A=55.*scale_i, I_cont_A=28.*scale_i,V_bus_V=6. if args.variant=='low_voltage' else 48.),
      'wrist':AtlasActuatorFOCCfg(joint_names_expr=['panda_joint[5-7]'], stiffness=300.,damping=12., effort_limit=1000.,effort_limit_sim=1000.,velocity_limit=10.,velocity_limit_sim=10.,armature=.08,sim_dt=dt,pole_pairs=10,Rs_ohm=.03/scale_kv**2,Ld_H=85e-6/scale_kv**2,Lq_H=85e-6/scale_kv**2,lambda_m_Wb=(.25/15)/scale_kv,gear_ratio=4.,I_peak_A=55.*scale_i, I_cont_A=28.*scale_i,V_bus_V=6. if args.variant=='low_voltage' else 48.),
      'gripper':cfg.actuators['panda_hand']}
    cfg.actuators['gripper'].effort_limit=30.
    cfg.actuators['gripper'].effort_limit_sim=30.
    cfg.actuators['gripper'].stiffness=800.
    cfg.actuators['gripper'].damping=40.
    cfg.spawn.rigid_props.max_depenetration_velocity=1.
    robots.append(Articulation(cfg)); bases.append([0,y,.3])
cube=RigidObject(RigidObjectCfg(prim_path='/World/Cube', init_state=RigidObjectCfg.InitialStateCfg(pos=(.45,-.30,.333)), spawn=sim_utils.CuboidCfg(size=(.06,.06,.06),mass_props=sim_utils.MassPropertiesCfg(mass=.12),rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=16,solver_velocity_iteration_count=4,max_depenetration_velocity=1.),collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.001,rest_offset=0.),physics_material=mat,visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.96,.35,.08)))))
contacts = [ContactSensor(ContactSensorCfg(prim_path=f'/World/{side}/{finger}', update_period=0., history_length=1, filter_prim_paths_expr=['/World/Cube'])) for side in ['Left','Right'] for finger in ['panda_leftfinger','panda_rightfinger']]
sim.reset()
for r in robots:
    r.write_joint_state_to_sim(r.data.default_joint_pos, r.data.default_joint_vel)
    r.reset(); r.update(dt)
    if args.variant=='heavy':
        masses=r.root_physx_view.get_masses(); inertia=r.root_physx_view.get_inertias()
        factor=torch.ones_like(masses); factor[:,1:8]=(masses[:,1:8]+2.)/masses[:,1:8]
        r.root_physx_view.set_masses(masses*factor, torch.tensor([0],device='cpu'))
        r.root_physx_view.set_inertias(inertia*factor[...,None], torch.tensor([0],device='cpu'))
    if args.variant=='hot':
        for a in r.actuators.values():
            if hasattr(a,'_winding_T'): a._winding_T.fill_(125.)
controllers=[DifferentialIKController(DifferentialIKControllerCfg(command_type='pose',use_relative_mode=False,ik_method='dls'),num_envs=1,device=sim.device) for _ in robots]
handids=[r.body_names.index('panda_hand') for r in robots]
armids=[r.find_joints('panda_joint[1-7]')[0] for r in robots]
fingerids=[r.find_joints('panda_finger_joint.*')[0] for r in robots]
print('BODY_NAMES',robots[0].body_names,flush=True)
print('JOINT_NAMES',robots[0].joint_names,flush=True)
# World gripper-tip poses. Quaternions use wxyz.
def qx(a): return [math.cos(a/2),math.sin(a/2),0,0]
def qdown(yaw=0): return [0,math.cos(yaw/2),math.sin(yaw/2),0]
D=qdown(); R=qdown(math.pi/2); SIDE=qx(math.pi/2)
# time, left tip, left orientation, left opening, right tip, right orientation, right opening, phase
keys=[
(0,[.45,-.30,.55],D,.04,[.43,.32,.64],D,.04,'Approach'),
(3,[.45,-.30,.55],D,.04,[.43,.32,.64],D,.04,'Approach'),
(5,[.45,-.30,.335],D,.04,[.43,.32,.64],D,.04,'Grasp'),
(6.5,[.45,-.30,.335],D,.022,[.43,.32,.64],D,.04,'Grasp'),
(9,[.48,-.20,.65],D,.022,[.43,.32,.72],SIDE,.04,'Lift'),
(12,[.48,-.12,.65],R,.022,[.48,.26,.65],SIDE,.04,'Reorient'),
(14,[.48,-.04,.65],R,.022,[.48,.13,.65],SIDE,.04,'Present'),
(16,[.48,-.04,.65],R,.022,[.48,-.04,.65],SIDE,.04,'Receive'),
(17.5,[.48,-.04,.65],R,.022,[.48,-.04,.65],SIDE,.022,'Receive'),
(18.5,[.48,-.04,.65],R,.04,[.48,-.04,.65],SIDE,.022,'Release'),
(20,[.42,-.28,.84],R,.04,[.48,.04,.65],SIDE,.022,'Transfer'),
(23,[.40,-.32,.75],D,.04,[.48,.30,.57],D,.022,'Carry'),
(25,[.40,-.32,.75],D,.04,[.48,.30,.360],D,.022,'Place'),
(26.5,[.40,-.32,.75],D,.04,[.48,.30,.360],D,.04,'Place'),
(28,[.40,-.32,.75],D,.04,[.48,.30,.57],D,.04,'Retreat'),
(30,[.40,-.32,.75],D,.04,[.48,.30,.57],D,.04,'Verify')]
# Ease from the actual initial posture; the first frame is not a joint-target jump.
initial=[]
for r,h in zip(robots,handids):
    initial.extend([(r.data.body_pos_w[:,h]+quat_apply(r.data.body_quat_w[:,h],torch.tensor([[0,0,.1034]],device=sim.device)))[0].tolist(),r.data.body_quat_w[0,h].tolist(),.04])
keys[0]=(0,*initial,'Approach')
def slerp(a,b,u):
    a=np.array(a);b=np.array(b); dot=np.dot(a,b)
    if dot<0:b=-b;dot=-dot
    if dot>.9995:q=a+(b-a)*u;return(q/np.linalg.norm(q)).tolist()
    angle=math.acos(np.clip(dot,-1,1));return((math.sin((1-u)*angle)*a+math.sin(u*angle)*b)/math.sin(angle)).tolist()
def target(t):
    k=next((i for i in range(1,len(keys)) if t<=keys[i][0]),len(keys)-1)
    a,b=keys[k-1],keys[k];u=np.clip((t-a[0])/(b[0]-a[0]),0,1);u=u*u*(3-2*u)
    return [((1-u)*np.array(a[1])+u*np.array(b[1])).tolist(),slerp(a[2],b[2],u),(1-u)*a[3]+u*b[3],((1-u)*np.array(a[4])+u*np.array(b[4])).tolist(),slerp(a[5],b[5],u),(1-u)*a[6]+u*b[6],b[7]]

# Export authored visual meshes in rigid-body coordinates for a faithful web replay.
if args.export_mesh:
    meshes=[]; body_paths=[]
    for r in robots:
        root=sim.stage.GetPrimAtPath(r.cfg.prim_path)
        paths={p.GetName():p.GetPath() for p in Usd.PrimRange(root,Usd.TraverseInstanceProxies()) if p.GetName() in r.body_names}
        body_paths.extend([str(paths[n]) for n in r.body_names])
    body_paths.append('/World/Cube')
    for bi,bp in enumerate(body_paths):
        body=sim.stage.GetPrimAtPath(bp);bw=UsdGeom.Xformable(body).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        for p in Usd.PrimRange(body,Usd.TraverseInstanceProxies()):
            if not p.IsA(UsdGeom.Mesh):continue
            if UsdGeom.Imageable(p).ComputeVisibility()=='invisible':continue
            if UsdGeom.Imageable(p).GetPurposeAttr().Get()=='guide':continue
            m=UsdGeom.Mesh(p); pts=m.GetPointsAttr().Get(); counts=m.GetFaceVertexCountsAttr().Get(); inds=m.GetFaceVertexIndicesAttr().Get()
            if not pts or not counts:continue
            # Exclude collision-only meshes, retaining authored render surfaces.
            if 'collision' in str(p.GetPath()).lower():continue
            trans=UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default())*bw.GetInverse()
            verts=[float(v) for pt in pts for v in trans.Transform(Gf.Vec3d(pt))]
            faces=[];n=0
            for count in counts:
                for k in range(1,count-1):faces.extend([int(inds[n]),int(inds[n+k]),int(inds[n+k+1])])
                n+=count
            color=[.78,.8,.82]
            try:
                bound=UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial()[0]
                shader=bound.ComputeSurfaceSource()[0]
                for inp in ['diffuse_color_constant','diffuseColor']:
                    v=shader.GetInput(inp).Get()
                    if v is not None:color=list(v)[:3];break
            except Exception:pass
            meshes.append(dict(body=bi,name=str(p.GetPath()),vertices=verts,indices=faces,color=color))
    (out/'scene.json').write_text(json.dumps(dict(bodies=[f'{side}/{n}' for side,r in zip(['left','right'],robots) for n in r.body_names]+['cube'],meshes=meshes,table=dict(size=[1.35,1.65,.08],position=[.35,0,.26]),cube_size=.06),separators=(',',':')))
    print('MESHES',len(meshes),flush=True)

frames=[]; telemetry=[]; last_des=[r.data.default_joint_pos[:,ids].clone() for r,ids in zip(robots,armids)]
start=time.time()
for step in range(int(args.duration/dt)):
    t=step*dt
    if step%4==0:
        targets=target(t)
        for ri,(r,c,h,ids,fids) in enumerate(zip(robots,controllers,handids,armids,fingerids)):
            pos,quat,opening=targets[ri*3:ri*3+3]
            p=torch.tensor([pos],device=sim.device);q=torch.tensor([quat],device=sim.device)
            # Control the actual finger-tip frame 103.4 mm from panda_hand.
            handpos=r.data.body_pos_w[:,h];handquat=r.data.body_quat_w[:,h]
            off=quat_apply(handquat,torch.tensor([[0,0,.1034]],device=sim.device))
            jac=r.root_physx_view.get_jacobians()[:,h-1,:,ids].clone()
            jac[:,:3,:]-=torch.bmm(skew_symmetric_matrix(off),jac[:,3:,:])
            c.set_command(torch.cat((p,q),-1))
            des=c.compute(handpos+off,handquat,jac,r.data.joint_pos[:,ids])
            # Bound requested joint changes to avoid singularity jumps.
            des=r.data.joint_pos[:,ids]+(des-r.data.joint_pos[:,ids]).clamp(-.10,.10)
            limits=r.data.soft_joint_pos_limits[:,ids]
            des=torch.maximum(torch.minimum(des,limits[:,:,1]),limits[:,:,0])
            last_des[ri]=des
            r.set_joint_position_target(des,joint_ids=ids)
            r.set_joint_position_target(torch.full((1,2),float(opening),device=sim.device),joint_ids=fids)
    for r,ids in zip(robots,armids):
        r.set_joint_effort_target(r.root_physx_view.get_gravity_compensation_forces()[:,ids],joint_ids=ids)
        r.write_data_to_sim()
    sim.step(render=False)
    for r in robots:r.update(dt)
    cube.update(dt)
    for sensor in contacts:sensor.update(dt)
    if step%8==0:
        poses=torch.cat([r.data.body_state_w[0,:,:7] for r in robots]+[cube.data.root_state_w[:,:7]],0)
        frames.append(poses.cpu().numpy())
        temps=[];torques=[];requested=[];currents=[];sat=[]
        for r in robots:
            for a in r.actuators.values():
                if hasattr(a,'_winding_T'):
                    temps.extend(a._winding_T[0].tolist());torques.extend(a.applied_effort[0].tolist());requested.extend(a.computed_effort[0].tolist());currents.extend(a._iq[0].tolist())
        tippos=[(r.data.body_pos_w[:,h]+quat_apply(r.data.body_quat_w[:,h],torch.tensor([[0,0,.1034]],device=sim.device)))[0].tolist() for r,h in zip(robots,handids)]
        telemetry.append(dict(t=round((step+1)*dt,4),phase=target(t)[6],cube=cube.data.root_state_w[0].tolist(),tips=tippos,temperature=max(temps),torque=max(abs(v) for v in torques),requested_torque=max(abs(v) for v in requested),current=max(abs(v) for v in currents),saturation=sum(abs(x-y)>max(.5,abs(y)*.1) for x,y in zip(torques,requested))/len(torques),contact_forces=[sensor.data.force_matrix_w[0,0,0].tolist() for sensor in contacts],grip=[r.data.joint_pos[0,f].tolist() for r,f in zip(robots,fingerids)]))
    if step%800==0:print('PROGRESS',round(t,2),'cube',cube.data.root_pos_w[0].tolist(),'tips',tippos,flush=True)
poses=np.asarray(frames,dtype='<f4');poses.tofile(out/'poses.bin')
(out/'telemetry.json').write_text(json.dumps(telemetry,separators=(',',':')))
final=cube.data.root_pos_w[0].cpu().numpy(); last=telemetry[-50:]
result=dict(variant=args.variant,physics='Isaac Lab 2.1 / PhysX',dt=dt,fps=50,frames=len(frames),bodies=poses.shape[1],duration=args.duration,wall_seconds=time.time()-start,object_state_writes_during_episode=0,object_attachments=0,resets=0,controller='Differential IK with bounded FOC arm actuators; 30 N implicit linear gripper drives',final_position=final.tolist(),placement_success=bool(np.linalg.norm(final[:2]-[.48,.30])<.05 and abs(final[2]-.33)<.02 and max(np.linalg.norm(x['cube'][7:10]) for x in last)<.03),peak_temperature=max(x['temperature'] for x in telemetry),peak_torque=max(x['torque'] for x in telemetry),kv_scale=scale_kv,current_scale=scale_i,added_mass_per_arm_kg=14 if args.variant=='heavy' else 0,initial_winding_C=125 if args.variant=='hot' else 25,source='transfer.py')
(out/'result.json').write_text(json.dumps(result,indent=2));print('RESULT',json.dumps(result),flush=True)
app.close()
