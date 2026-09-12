import ActuatorDesign from "./actuator-design";
import { useEffect, useRef, useState } from "react";
import { Replay, loadScene, type Recording, type SceneData } from "./replay";
import { loadRecording, SyncedVideo } from "./locomotion";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Play, Pause, RotateCcw } from "lucide-react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine } from "recharts";

type Case = {id: string; label: string; description: string; passed: boolean; duration: number; parameters?: {kv:number;gear:number;torque:number;rpm:number;mass:number}};
export default function RoughTerrain() {
  const [track, setTrack] = useState("course"), [variant, setVariant] = useState("stock");
  const [nominal, setNominal] = useState<Recording | null>(null);
  useEffect(() => {loadRecording("./data/course/nominal").then(setNominal).catch(()=>{});}, []);
  const [cases, setCases] = useState<Case[]>([]);
  const root = track === "course" ? `./data/course/${variant}` : "./data/rough-terrain";
  const [recording, setRecording] = useState<Recording | null>(null), [scene, setScene] = useState<SceneData | null>(null);
  const [error, setError] = useState(""), [playing, setPlaying] = useState(false), [time, setTime] = useState(0), [speed, setSpeed] = useState(1);
  const [mode, setMode] = useState("both"), [camera, setCamera] = useState("overview"), [metric, setMetric] = useState("speed");
  const clock = useRef({time: 0, playing: false, speed: 1});
  clock.current.playing = playing;
  clock.current.speed = speed;
  useEffect(() => {fetch("./data/course/manifest.json").then(r => {if (!r.ok) throw Error("Course comparison unavailable"); return r.json() as Promise<{variants: Case[]}>;}).then(m => setCases(m.variants)).catch(e => setError(e.message));}, []);
  useEffect(() => {
    let active = true;
    setRecording(null); setError(""); setMetric("speed"); setPlaying(false); setTime(0); clock.current.time = 0;
    Promise.all([loadScene(track === "course" ? "./data/course/stock/scene.json" : `${root}/scene.json`), loadRecording(root)])
      .then(([s, r]) => {if (active) {setScene(s); setRecording(r);}}).catch(e => active && setError(e.message));
    return () => {active = false;};
  }, [root, track]);
  useEffect(() => {
    let last = performance.now(), frame = 0;
    const tick = (now: number) => {
      if (clock.current.playing && recording) {
        clock.current.time = Math.min(recording.result.duration, clock.current.time + Math.min((now-last)/1000, .1)*clock.current.speed);
        setTime(clock.current.time);
        if (clock.current.time >= recording.result.duration) setPlaying(false);
      }
      last = now; frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick); return () => cancelAnimationFrame(frame);
  }, [recording]);
  const seek = (value: number) => {clock.current.time = value; setTime(value);};
  const selector = <div className="hand-trial-selector"><label htmlFor="terrain-route">Route</label><select id="terrain-route" value={track} onChange={e => setTrack(e.target.value)}><option value="course">Stairs, platform and blocks</option><option value="pilot">Random-terrain pilot</option></select>{track === "course" && <><label htmlFor="course-motor">Actuator design</label><select id="course-motor" value={variant} onChange={e => setVariant(e.target.value)}>{cases.map(c => <option key={c.id} value={c.id}>{c.label}</option>)}</select></>}</div>;
  if (error) return <>{selector}<section className="precision-results" role="alert">{error}</section></>;
  if (!recording || !scene) return <>{selector}<p role="status">Loading measured terrain and motion…</p></>;
  const sample = recording.samples[Math.min(recording.samples.length-1, Math.floor(time*recording.result.fps))] as any;
  const chosen = cases.find(c => c.id === variant);
  const design = recording.result.design?.specifications?.legs;
  const x = sample?.cube?.[0] ?? 0;
  const phase = track !== "course" ? "Native rough terrain" : x < 2 ? "Approach stairs" : x < 3.75 ? "Climb stairs" : x < 5.25 ? "Cross platform" : x < 7 ? "Descend stairs" : x < 10.8 ? "Cross uneven blocks" : x < 12.5 ? "Reach the finish" : "Stop and hold";
  const chartKey = metric === "speed" ? "speed" : metric === "torque" ? "torque" : metric === "current" ? "current" : "temperature";
  return <>{selector}
    <section className="motion-workbench">
      <div className="motion-stage">
        <div className="motion-heading"><span>ANYmal · {track === "course" ? "Authored obstacle course" : "Native rough terrain"}</span><span>PHYSX / 200 Hz · REPLAY / 50 FPS</span></div>
        <div className="motion-phase"><h2>{phase}</h2><p>{track === "course" ? "A 12.5 m route. Five steps up, a raised platform, five steps down, uneven blocks, then a stopped hold." : "The original eight-environment rough-terrain pilot, showing environment 0."}</p></div>
        <div className={`motion-pair ${mode === "both" ? "dual" : ""}`}>
          {mode !== "3d" && <SyncedVideo src={`${root}/video.mp4`} clock={clock} label="Measured-state render" />}
          {mode !== "video" && <Replay sceneData={scene} recording={recording} playback={clock} cameraView={camera} showPath={true} label="Measured 3D replay" />}
        </div>
        <div className="motion-toolbar">
          <Button aria-label={playing ? "Pause" : "Play"} onClick={() => {if (time >= recording.result.duration) seek(0); setPlaying(!playing);}}>{playing ? <Pause size={16}/> : <Play size={16}/>}</Button>
          <Button aria-label="Restart" variant="ghost" onClick={() => seek(0)}><RotateCcw size={16}/></Button>
          <span>{time.toFixed(1)} / {recording.result.duration.toFixed(1)} s</span>
          <select aria-label="Playback speed" value={speed} onChange={e => setSpeed(Number(e.target.value))}>{[.25,.5,1,2].map(v => <option key={v} value={v}>{v}×</option>)}</select>
          <select aria-label="Camera" value={camera} onChange={e => setCamera(e.target.value)}><option value="follow">Follow robot</option><option value="overview">Course overview / orbit</option></select>
        </div>
        <div className="transport"><Slider aria-label="Replay time" min={0} max={recording.result.duration} step={.02} value={[time]} onValueChange={v => seek(typeof v === "number" ? v : v[0])}/></div>
      </div>
      <aside className="motion-inspector">
        <p className="eyebrow">{track === "course" ? "Recorded course outcome" : "Pilot baseline"}</p>
        <h2>{track === "course" ? recording.result.task_success ? "Course completed." : "Course not completed." : "Uneven footing."}</h2>
        <p>{track === "course" ? chosen?.description : "Eight environments, one development seed, 20 seconds each."}</p>
        <div className="motion-modes">{["both","video","3d"].map(v => <Button key={v} variant={mode===v ? "default":"ghost"} onClick={() => setMode(v)}>{v === "3d" ? "3D replay" : v === "both" ? "Both" : "Video"}</Button>)}</div>
        <dl className="motion-values"><dt>Forward speed</dt><dd>{sample?.speed?.toFixed(2)} m/s</dd><dt>Command</dt><dd>{sample?.target_speed?.toFixed(2)} m/s</dd><dt>Position along route</dt><dd>{x.toFixed(2)} m</dd><dt>Failure terminations</dt><dd>{sample?.failure_terminations ?? 0}</dd><dt>Episode resets</dt><dd>{sample?.failures ?? 0}</dd>{design && <><dt>Kv · phase-peak convention</dt><dd>{design.Kv_phase_peak_rpm_per_V.toFixed(1)} rpm/V</dd><dt>Gear ratio</dt><dd>{design.gear_ratio.toFixed(1)}:1</dd><dt>Peak current limit</dt><dd>{design.peak_current_A.toFixed(1)} A</dd><dt>Added robot mass</dt><dd>{recording.result.design.measured_added_mass_kg[0].toFixed(1)} kg</dd></>}</dl>
        <p>Completion requires staying within the route corridor, reaching 12.5 m and holding below 0.1 m/s for one second, without a failure or timeout. These are single-seed demonstrations, not reliability estimates.</p>
        <p>Video is rendered in Isaac Sim from the recorded body states and mesh geometry. Materials and lighting are presentation choices.</p>
      </aside>
    </section>
    <ActuatorDesign groups={recording.result.design?.specifications} baseline={nominal?.result.design?.specifications} addedMass={recording.result.design?.measured_added_mass_kg?.[0]} builder={track === "course"} notice={track === "course" && variant === "stock" ? "Native ANYdrive is a separate actuator/controller baseline. Select an Atlas design to inspect winding, gearing and mass differences." : undefined}/>
    <section className="motion-analysis"><div><h2>Measured through the route</h2><label>Signal <select value={metric} onChange={e=>setMetric(e.target.value)}><option value="speed">Forward speed · m/s</option>{sample?.torque != null && <option value="torque">Peak joint torque · Nm</option>}{design && <><option value="current">Peak motor current · A</option><option value="temperature">Hottest winding · °C</option></>}</select></label><ResponsiveContainer width="100%" height={240}><LineChart data={recording.samples}><XAxis dataKey="t" type="number" domain={[0,recording.result.duration]} tickFormatter={v=>`${v}s`}/><YAxis width={45}/><Tooltip/><Line dataKey={chartKey} dot={false} stroke="#177960" strokeWidth={2} isAnimationActive={false}/>{metric === "speed" && <Line dataKey="target_speed" dot={false} stroke="#a48043" strokeDasharray="4 4" isAnimationActive={false}/>}<ReferenceLine x={time} stroke="#314f42"/></LineChart></ResponsiveContainer></div><div><h2>What the comparison holds fixed</h2><p>The course, command, checkpoint, and development seed stay fixed. Atlas variants change one design axis: winding, gearing, current limit, or physical carrier mass and inertia. The native ANYdrive reference also changes controller behavior, so it is a separate baseline.</p><p>Different winding designs change Kt, resistance and inductance with Kv. Motor constants use the model’s phase-peak convention. Added mass is a stated simulation intervention, not a manufacturer-qualified motor.</p><div className="motion-downloads"><a href={`${root}/result.json`}>Recording metadata</a><a href={`${root}/telemetry.json`}>Measured telemetry</a><a href="./data/course/manifest.json">Course comparisons</a></div></div></section>
    {track === "course" && <section className="precision-results"><h2>Every recorded design</h2><p>Model parameters beside measured outcomes. RPM is ideal unloaded joint speed; mass is total added actuator mass.</p><div className="actuator-table"><table><thead><tr><th>Actuator design</th><th>Kv · rpm/V</th><th>Gearing</th><th>Peak stall · Nm</th><th>Ideal RPM</th><th>Added kg</th><th>Recorded outcome</th><th>Run duration</th></tr></thead><tbody>{cases.map(c=><tr key={c.id}><td><button onClick={()=>setVariant(c.id)}>{c.label}</button></td><td>{c.parameters?.kv.toFixed(1) ?? "—"}</td><td>{c.parameters?.gear.toFixed(1) ?? "—"}</td><td>{c.parameters?.torque.toFixed(1) ?? "—"}</td><td>{c.parameters?.rpm.toFixed(1) ?? "—"}</td><td>{c.parameters?.mass.toFixed(1) ?? "—"}</td><td>{c.passed ? "Completed" : "Not completed"}</td><td>{c.duration.toFixed(2)} s</td></tr>)}</tbody></table></div></section>}
  </>;
}
