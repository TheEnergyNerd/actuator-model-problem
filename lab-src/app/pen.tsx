import { useEffect, useRef, useState } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine } from "recharts";
import { Replay, loadScene, type Recording, type SceneData } from "./replay";
import { loadRecording, SyncedVideo } from "./locomotion";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Play, Pause, RotateCcw } from "lucide-react";

type Outcome = {trial:number; seed:number; motion_passed:boolean; native_contact_gate_passed:boolean; net_turns:number; hold_ok:boolean; drop:boolean};
type Evaluation = {trials:Outcome[]; motion_successes:number; native_contact_successes:number; method:string};

type Comparison = {rows:{id:string;motion_successes:number;native_contact_successes:number;trials:number;selected_trial_third_turn_s:number}[]};
const labels:Record<string,string>={baseline:"Native Sharpa",weak:"25% torque ceiling",heavy:"+50% hand mass / inertia"};
const fingers=["Thumb","Index","Middle","Ring","Pinky"];
export default function PenSpinning() {
  const [variant,setVariant]=useState("baseline");
  const [comparison,setComparison]=useState<Comparison|null>(null);
  useEffect(()=>{let active=true;fetch("./data/pen/comparison.json").then(r=>r.json() as Promise<Comparison>).then(c=>{if(active)setComparison(c)}).catch(()=>{});return()=>{active=false}},[]);
  const root=`./data/pen/${variant}`;
  const selector=<div className="hand-trial-selector"><label htmlFor="pen-variant">Physical comparison</label><select id="pen-variant" value={variant} onChange={e=>setVariant(e.target.value)}><option value="baseline">Native Sharpa · supplied policy</option><option value="weak">Torque ceiling · 25% of baseline</option><option value="heavy">Hand mass and inertia · +50%</option></select></div>;
  const [recording,setRecording]=useState<Recording|null>(null),[scene,setScene]=useState<SceneData|null>(null),[evaluation,setEvaluation]=useState<Evaluation|null>(null);
  const [error,setError]=useState(""),[time,setTime]=useState(0),[playing,setPlaying]=useState(false),[mode,setMode]=useState("both");
  const clock=useRef({time:0,playing:false,speed:1});
  clock.current.playing=playing;
  useEffect(()=>{let active=true;setRecording(null);setError("");setPlaying(false);setTime(0);clock.current.time=0;Promise.all([loadScene("./data/pen/baseline/scene.json"),loadRecording(root),fetch(`${root}/evaluation.json`).then(r=>{if(!r.ok)throw Error("Evaluation unavailable");return r.json() as Promise<Evaluation>})]).then(([s,r,e])=>{if(active){setScene(s);setRecording(r);setEvaluation(e)}}).catch(e=>active&&setError(e.message));return()=>{active=false}},[root]);
  useEffect(()=>{let frame=0,last=performance.now();const tick=(now:number)=>{if(clock.current.playing&&recording){clock.current.time=Math.min(recording.result.duration,clock.current.time+Math.min((now-last)/1000,.1)*clock.current.speed);setTime(clock.current.time);if(clock.current.time>=recording.result.duration)setPlaying(false)}last=now;frame=requestAnimationFrame(tick)};frame=requestAnimationFrame(tick);return()=>cancelAnimationFrame(frame)},[recording]);
  const seek=(t:number)=>{clock.current.time=t;setTime(t)};
  if(error)return <>{selector}<p role="alert">The pen recording could not be loaded: {error}</p></>;
  if(!recording||!scene||!evaluation)return <>{selector}<p role="status">Loading the Sharpa pen evaluation…</p></>;
  const sample=recording.samples[Math.min(recording.samples.length-1,Math.floor(time*recording.result.fps))] as any;
  return <>{selector}<section className="motion-workbench"><div className="motion-stage">
    <div className="motion-heading"><span>Sharpa Wave · Pen spinning</span><span>PHYSX / 240 Hz · REPLAY / 60 FPS</span></div>
    <div className="motion-phase"><h2>{sample.phase}</h2><p>Five fingers, one learned policy. The pen moves through physical contact.</p></div>
    <div className={`motion-pair ${mode==="both"?"dual":""}`}>
      {mode!=="3d"&&<SyncedVideo src={`${root}/video.mp4`} clock={clock} label="Isaac measured-state render"/>}
      {mode!=="video"&&<Replay sceneData={scene} recording={recording} playback={clock} cameraView="overview" showPath={false} label="Orbit the measured 3D replay"/>}
    </div>
    <div className="motion-toolbar"><Button aria-label={playing?"Pause":"Play"} onClick={()=>{if(time>=recording.result.duration)seek(0);setPlaying(!playing)}}>{playing?<Pause size={16}/>:<Play size={16}/>}</Button><Button aria-label="Restart" variant="ghost" onClick={()=>seek(0)}><RotateCcw size={16}/></Button><select aria-label="Playback speed" defaultValue="1" onChange={e=>{clock.current.speed=Number(e.target.value)}}><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1">1×</option></select><span>{time.toFixed(1)} / {recording.result.duration.toFixed(1)} s</span></div>
    <div className="transport"><Slider aria-label="Replay time" min={0} max={recording.result.duration} step={1/60} value={[time]} onValueChange={v=>seek(typeof v==="number"?v:v[0])}/></div>
  </div><aside className="motion-inspector"><p className="eyebrow">Preselected trial {recording.result.trial}</p><h2>{recording.result.task_success?"Three turns. Then hold.":"Recorded attempt."}</h2>
    <div className="motion-modes">{["both","video","3d"].map(v=><Button key={v} variant={mode===v?"default":"ghost"} onClick={()=>setMode(v)}>{v==="both"?"Both":v==="3d"?"3D replay":"Video"}</Button>)}</div>
    <dl className="motion-values"><dt>Net rotations</dt><dd>{sample.turns.toFixed(2)}</dd><dt>Pen angular speed</dt><dd>{sample.angular_speed_deg_s.toFixed(0)} °/s</dd><dt>Estimated drive torque now</dt><dd>{sample.torque.toFixed(3)} Nm</dd>{fingers.map((f,i)=><span key={f} style={{display:"contents"}}><dt>{f} contact</dt><dd>{sample.finger_forces[i].toFixed(3)} N</dd></span>)}</dl>
    <p>{evaluation.motion_successes}/{evaluation.trials.length} trials complete three timed rotations and a supported one-second hold, within the motion limits. {evaluation.native_contact_successes}/{evaluation.trials.length} also pass the additional native contact and finger checks.</p>
    <p>The grasp is prepared before the recording. Trial 0 was selected before evaluation; every trial is reported below.</p>
  </aside></section>
  <section className="motion-analysis"><div><h2>Rotations through time</h2><ResponsiveContainer width="100%" height={240}><LineChart data={recording.samples}><XAxis dataKey="t" type="number" domain={[0,recording.result.duration]}/><YAxis/><Tooltip/><Line name="Net rotations" dataKey="turns" stroke="#177960" dot={false} isAnimationActive={false}/><ReferenceLine y={3} stroke="#a48043" strokeDasharray="4 4"/><ReferenceLine x={time} stroke="#314f42"/></LineChart></ResponsiveContainer></div><div><h2>First establish the skill</h2><p>This is a reproduction of the supplied <a href="https://github.com/jianglongye/dexterous-astra">Dexterous Astra</a> checkpoint with official Sharpa geometry. Atlas records and displays the Isaac body states; it did not train this checkpoint.</p><p>The comparisons retain the same policy and native Sharpa position controller. One reduces the physical joint torque ceiling to 25%; another increases hand-body mass and inertia by 50%. Changes apply after the seeded grasp setup. These are sensitivity tests, not complete motor designs. Kv, thermal models and per-design retraining are not included. Contact checks use PhysX separations, so they differ from the reference’s independent collision audit.</p></div></section>
  <section className="precision-results"><h2>Same policy. Different physical limits.</h2>{comparison&&<div style={{overflowX:"auto"}}><table><thead><tr><th>Configuration</th><th>Motion successes</th><th>Additional contact checks</th><th>Trial 0 · third turn</th></tr></thead><tbody>{comparison.rows.map(r=><tr key={r.id}><td><Button variant="ghost" onClick={()=>setVariant(r.id)}>{labels[r.id]}</Button></td><td>{r.motion_successes}/{r.trials}</td><td>{r.native_contact_successes}/{r.trials}</td><td>{r.selected_trial_third_turn_s.toFixed(2)} s</td></tr>)}</tbody></table></div>}<p>All three runs use identical recorded starting poses, pen mass and friction for each seed. These 32 development trials show sensitivity, not a reliable ranking of motor designs. Extra hand mass also increases inertia and does not necessarily reduce success on this task.</p><a href="./data/pen/comparison.json">Matched comparison data</a><h2>Every evaluation trial · {labels[variant]}</h2><div style={{overflowX:"auto"}}><table><thead><tr><th>Seed</th><th>Max net turns</th><th>Hold</th><th>Drop</th><th>Motion</th><th>Native contact gate</th></tr></thead><tbody>{evaluation.trials.map(r=><tr key={r.seed}><td>{r.seed}</td><td>{r.net_turns.toFixed(2)}</td><td>{r.hold_ok?"Yes":"No"}</td><td>{r.drop?"Yes":"No"}</td><td>{r.motion_passed?"Pass":"Fail"}</td><td>{r.native_contact_gate_passed?"Pass":"Fail"}</td></tr>)}</tbody></table></div><p>{evaluation.method}</p><a href={`${root}/evaluation.json`}>All numerical checks</a> · <a href={`${root}/result.json`}>Recording provenance</a> · <a href={`${root}/telemetry.json`}>Recorded telemetry</a> · <a href="./data/pen/SHARPA-LICENSE.txt">Sharpa asset license</a></section></>;
}
