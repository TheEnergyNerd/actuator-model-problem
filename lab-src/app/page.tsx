"use client";
import Locomotion from "./locomotion";
import DexterousHand from "./hand";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { NativeSelect, NativeSelectOption } from "@/components/ui/native-select";
import { ChartContainer } from "@/components/ui/chart";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine } from "recharts";
import {
  ArrowUpRight,
  Boxes,
  Play,
  Pause,
  RotateCcw,
  FlaskConical,
  Move3d,
  Check,
  Download,
  GitCompareArrows,
  ArrowRight,
  Activity,
  Thermometer,
  Zap,
  Target,
} from "lucide-react";
import { Replay, loadScene, type SceneData, type Recording, type Sample } from "./replay";
type Variant = {
  id: string;
  label: string;
  short: string;
  description: string;
  observed: string;
  kv: number;
  kt: number;
  current: number;
  mass: number;
  voltage: number;
  temperature: number;
  peakTorque: number;
  resistance: number;
  result: any;
  envelope: { speed: number; torque: number }[];
};
type Manifest = {
  variants: Variant[];
  protocol: string;
  source: string;
  scene: string;
};
const cache = new Map<string, Promise<Recording>>();
function loadRun(id: string) {
  if (!cache.has(id))
    cache.set(
      id,
      Promise.all([
        fetch(`./data/${id}/poses.bin`).then((r) => {
          if (!r.ok) throw Error("Recording unavailable");
          return r.arrayBuffer();
        }),
        fetch(`./data/${id}/telemetry.json`).then((r) => r.json() as Promise<Sample[]>),
        fetch(`./data/${id}/result.json`).then((r) => r.json() as Promise<any>),
      ])
        .then(([poses, samples, result]) => {
          if (
            poses.byteLength !== result.frames * result.bodies * 7 * 4 ||
            samples.length !== result.frames
          )
            throw Error("Recording integrity check failed");
          return { poses: new Float32Array(poses), samples, result };
        })
        .catch((e) => {
          cache.delete(id);
          throw e;
        }),
    );
  return cache.get(id)!;
}
const phases = [
  ["Grasp & lift", 8],
  ["Reorient", 12],
  ["Handoff", 18],
  ["Place & release", 27],
] as const;
const fmt = (v: number | undefined, n = 1) => (v === undefined ? "—" : v.toFixed(n));
export default function Home() {
  const [manifest, setManifest] = useState<Manifest | null>(null),
    [scene, setScene] = useState<SceneData | null>(null),
    [run, setRun] = useState<Recording | null>(null),
    [baseline, setBaseline] = useState<Recording | null>(null),
    [id, setId] = useState("nominal"),
    [error, setError] = useState(""),
    [loading, setLoading] = useState(true),
    [playing, setPlaying] = useState(false),
    [time, setTime] = useState(0),
    [speed, setSpeed] = useState(1),
    [camera, setCamera] = useState("overview"),
    [compare, setCompare] = useState(false),
    [path, setPath] = useState(true),
    [metric, setMetric] = useState("torque"),
    [task, setTask] = useState(() =>
      ["transfer", "g1", "anymal", "allegro"].includes(window.location.hash.slice(1))
        ? window.location.hash.slice(1)
        : "g1",
    );
  const clock = useRef({ time: 0, playing: false, speed: 1, duration: 30 });
  clock.current.playing = playing;
  clock.current.speed = speed;
  clock.current.duration = run?.result.duration || 30;
  useEffect(() => {
    let disposed = false;
    Promise.all([
      fetch("./data/manifest.json").then((r) => {
        if (!r.ok)
          throw Error(
            "Recordings are being prepared. The nominal transfer video is available below.",
          );
        return r.json() as Promise<Manifest>;
      }),
      loadRun("nominal"),
    ])
      .then(async ([m, r]) => {
        const s = await loadScene(m.scene);
        if (!disposed) {
          setManifest(m);
          setScene(s);
          setBaseline(r);
          setRun(r);
          if (!window.matchMedia("(prefers-reduced-motion: reduce)").matches) setPlaying(true);
          setLoading(false);
        }
      })
      .catch((e) => {
        if (!disposed) {
          setError(e.message);
          setLoading(false);
        }
      });
    return () => {
      disposed = true;
    };
  }, []);
  useEffect(() => {
    if (!manifest) return;
    let disposed = false;
    setLoading(true);
    setError("");
    loadRun(id)
      .then((r) => {
        if (!disposed) {
          setRun(r);
          setLoading(false);
        }
      })
      .catch((e) => {
        if (!disposed) {
          setError(e.message);
          setLoading(false);
        }
      });
    return () => {
      disposed = true;
    };
  }, [id, manifest]);
  useEffect(() => {
    let last = performance.now(),
      lastUI = last,
      raf = 0;
    function tick(now: number) {
      const c = clock.current;
      if (c.playing) {
        const next = Math.min(
          c.duration - 0.02,
          c.time + Math.min((now - last) / 1000, 0.1) * c.speed,
        );
        clock.current.time = next;
        if (now - lastUI > 80 || next >= c.duration - 0.02) {
          setTime(next);
          lastUI = now;
        }
        if (next >= c.duration - 0.02) setPlaying(false);
      }
      last = now;
      raf = requestAnimationFrame(tick);
    }
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);
  const seek = useCallback((t: number) => {
    clock.current.time = t;
    setTime(t);
  }, []);
  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).closest("input,select,textarea,button,[role=slider]")) return;
      if (e.code === "Space") {
        e.preventDefault();
        setPlaying((p) => !p);
      }
      if (e.code === "ArrowRight") {
        e.preventDefault();
        seek(Math.min(clock.current.duration - 0.02, clock.current.time + 1));
      }
      if (e.code === "ArrowLeft") {
        e.preventDefault();
        seek(Math.max(0, clock.current.time - 1));
      }
    };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [seek]);
  const selected = manifest?.variants.find((v) => v.id === id),
    sample = run?.samples[Math.min((run?.samples.length || 1) - 1, Math.floor(time * 50))];
  const chart = useMemo(
    () =>
      run?.samples
        .filter((_, i) => i % 10 === 0)
        .map((s, i) => ({
          t: s.t,
          selected: (s as any)[metric],
          nominal: (baseline?.samples[i * 10] as any)?.[metric],
        })) || [],
    [run, baseline, metric],
  );
  const units: Record<string, string> = {
    temperature: "°C",
    torque: "N·m",
    current: "A",
    saturation: "fraction",
  };
  const totalForce = (start: number) =>
    sample?.contact_forces?.slice(start, start + 2).reduce((s, v) => s + Math.hypot(...v), 0);
  const isFinal = time > 28,
    success = run?.result.checks?.full_sequence_success;
  return (
    <main className="atlas">
      <header>
        <a className="brand" href="../">
          ATLAS<span>/ ACTUATOR LAB</span>
        </a>
        <span className="status">
          <i /> ISAAC LAB · RECORDED PHYSICS
        </span>
        <a className="quiet-link" href="#method">
          How this works <ArrowUpRight size={13} />
        </a>
      </header>
      <section className="intro">
        <div>
          <p className="eyebrow">
            {task === "transfer"
              ? "CONTACT-DRIVEN MANIPULATION"
              : task === "allegro"
                ? "IN-HAND REORIENTATION"
                : "LEARNED LOCOMOTION / ACTUATOR DESIGN"}
          </p>
          <h1>
            Every motor choice
            <br />
            has a consequence.
          </h1>
        </div>
        <div className="intro-right">
          <p className="lede">
            One task. The same controller.
            <br />
            Different hardware, visible consequences.
          </p>
          <div className="task-switch" role="group" aria-label="Experiment">
            {[
              ["transfer", "Bimanual transfer"],
              ["g1", "G1 walking"],
              ["anymal", "ANYmal"],
              ["allegro", "Dexterous hand"],
            ].map(([v, l]) => (
              <Button
                key={v}
                size="sm"
                variant={task === v ? "default" : "ghost"}
                onClick={() => {
                  setTask(v);
                  history.replaceState(null, "", `#${v}`);
                  setPlaying(false);
                }}
              >
                {l}
              </Button>
            ))}
          </div>
        </div>
      </section>
      {task === "transfer" ? (
        <>
          <section className="workbench">
            <div className="stage">
              <div className="stage-heading">
                <span>
                  <Boxes size={16} /> Two arms. One continuous sequence.
                </span>
                <span>PHYSX / 400 Hz · REPLAY / 50 FPS</span>
              </div>
              <div className={`scene-wrap ${compare ? "split" : ""}`}>
                {scene && run ? (
                  <>
                    <Replay
                      sceneData={scene}
                      recording={run}
                      playback={clock}
                      cameraView={camera}
                      showPath={path}
                      label={selected?.label || "Nominal"}
                    />
                    {compare && baseline && (
                      <Replay
                        sceneData={scene}
                        recording={baseline}
                        playback={clock}
                        cameraView={camera}
                        showPath={path}
                        label="Nominal reference"
                      />
                    )}
                  </>
                ) : (
                  <video
                    src="./nominal.mp4"
                    poster="./transfer-poster.png"
                    controls
                    muted
                    playsInline
                    preload="metadata"
                  />
                )}
                {loading && (
                  <div className="scene-notice" role="status">
                    Loading recorded physics…
                  </div>
                )}
                {error && (
                  <div className="scene-notice" role="alert">
                    {error}
                  </div>
                )}
                {sample && !loading && (
                  <div className="phase-overlay">
                    <span className="eyebrow">{isFinal ? "FINAL OUTCOME" : "COMMANDED PHASE"}</span>
                    <strong>
                      {isFinal
                        ? success
                          ? "Placed & released"
                          : "Placement failed"
                        : sample.phase}
                    </strong>
                    <span>
                      {isFinal
                        ? success
                          ? "Object rests at the destination."
                          : "Inspect the missed stage in the replay."
                        : "Actual body motion is shown in the scene."}
                    </span>
                  </div>
                )}
              </div>
              <div className="view-controls">
                <span>
                  <Move3d size={13} /> Drag to orbit · scroll to zoom
                </span>
                <div>
                  <NativeSelect
                    aria-label="Camera view"
                    value={camera}
                    onChange={(e) => setCamera(e.target.value)}
                  >
                    {[
                      ["overview", "Overview"],
                      ["handoff", "Handoff close-up"],
                      ["front", "Front"],
                      ["follow", "Follow cube"],
                      ["top", "Overhead"],
                    ].map(([v, l]) => (
                      <NativeSelectOption key={v} value={v}>
                        {l}
                      </NativeSelectOption>
                    ))}
                  </NativeSelect>
                  <Button
                    variant={path ? "secondary" : "ghost"}
                    size="sm"
                    aria-pressed={path}
                    onClick={() => setPath(!path)}
                  >
                    Object trail
                  </Button>
                  <Button
                    variant={compare ? "secondary" : "ghost"}
                    size="sm"
                    aria-pressed={compare}
                    onClick={() => setCompare(!compare)}
                  >
                    <GitCompareArrows />
                    Compare
                  </Button>
                </div>
              </div>
              <div className="transport">
                <Button
                  disabled={!run || loading}
                  size="icon"
                  aria-label={playing ? "Pause replay" : "Play replay"}
                  onClick={() => {
                    if (time > 29.9) seek(0);
                    setPlaying(!playing);
                  }}
                >
                  {playing ? <Pause /> : <Play />}
                </Button>
                <Button
                  variant="ghost"
                  size="icon"
                  aria-label="Restart replay"
                  onClick={() => {
                    seek(0);
                    setPlaying(true);
                  }}
                >
                  <RotateCcw />
                </Button>
                <span className="timecode">
                  {fmt(time)} <small>/ {run?.result.duration || 30}.0 s</small>
                </span>
                <Slider
                  aria-label="Replay time"
                  min={0}
                  max={29.98}
                  step={0.02}
                  value={[time]}
                  onValueChange={(v) => seek(Array.isArray(v) ? v[0] : v)}
                />
                <NativeSelect
                  aria-label="Playback speed"
                  value={speed}
                  onChange={(e) => setSpeed(Number(e.target.value))}
                >
                  {[0.25, 0.5, 1, 2].map((v) => (
                    <NativeSelectOption key={v} value={v}>
                      {v}×
                    </NativeSelectOption>
                  ))}
                </NativeSelect>
              </div>
              <div className="phase-strip">
                {phases.map(([name, t], i) => (
                  <Button
                    variant="ghost"
                    key={name}
                    onClick={() => {
                      seek(t);
                      setPlaying(false);
                    }}
                    className={time >= t ? "visited" : ""}
                  >
                    <span>0{i + 1}</span>
                    {name}
                    <ArrowRight size={12} />
                  </Button>
                ))}
              </div>
            </div>
            <aside>
              <div className="aside-title">
                <p className="eyebrow">ACTUATOR DESIGN</p>
                <span className="small-tag">RECORDED RUNS</span>
              </div>
              <h2>
                Change the motor.
                <br />
                Follow the result.
              </h2>
              <div className="variants" role="group" aria-label="Actuator variants">
                {manifest?.variants.map((v) => (
                  <Button
                    key={v.id}
                    variant={id === v.id ? "default" : "outline"}
                    onClick={() => {
                      setId(v.id);
                      setPlaying(false);
                    }}
                    aria-pressed={id === v.id}
                  >
                    {v.short}
                  </Button>
                ))}
              </div>
              <p className="variant-description">
                {selected?.description || "The nominal transfer is being evaluated."}
              </p>
              <dl className="parameters">
                <div>
                  <dt>
                    Motor Kv <small>phase peak</small>
                  </dt>
                  <dd>
                    {fmt(selected?.kv, 1)} <span>rpm/V</span>
                  </dd>
                </div>
                <div>
                  <dt>Torque constant Kt</dt>
                  <dd>
                    {fmt(selected?.kt, 3)} <span>N·m/A</span>
                  </dd>
                </div>
                <div>
                  <dt>Peak current</dt>
                  <dd>
                    {fmt(selected?.current, 1)} <span>A</span>
                  </dd>
                </div>
                <div>
                  <dt>Added mass / arm</dt>
                  <dd>
                    {fmt(selected?.mass, 1)} <span>kg</span>
                  </dd>
                </div>
                <div>
                  <dt>Bus voltage</dt>
                  <dd>
                    {selected?.voltage || "—"} <span>V</span>
                  </dd>
                </div>
                <div>
                  <dt>Cold shoulder limit</dt>
                  <dd>
                    {fmt(selected?.peakTorque, 1)} <span>N·m</span>
                  </dd>
                </div>
              </dl>
              <div className="result-block">
                <span className="eyebrow">COMPLETE-RUN OUTCOME</span>
                <strong className={success ? "positive" : "negative"}>
                  {run ? (
                    success ? (
                      <>
                        <Check size={17} /> Sequence passed
                      </>
                    ) : (
                      <>Sequence failed</>
                    )
                  ) : (
                    "Awaiting recording"
                  )}
                </strong>
                <span>
                  {run
                    ? `${fmt(Math.hypot(run.result.final_position[0] - 0.48, run.result.final_position[1] - 0.3) * 1000)} mm from target center`
                    : "One continuous 30-second attempt"}
                </span>
              </div>
              <div className="stage-checks" aria-label="Measured stage checks">
                {[
                  ["lifted", "Lift"],
                  ["rotated_90_degrees", "Rotate"],
                  ["handoff_supported_by_right_only", "Handoff"],
                  ["placed_and_released", "Place"],
                ].map(([k, l]) => (
                  <span key={k} className={run?.result.checks?.[k] ? "positive" : "negative"}>
                    {run?.result.checks?.[k] ? "✓" : "×"} {l}
                  </span>
                ))}
              </div>
              <p className="observed-effect">
                <b>Observed:</b> {selected?.observed}
              </p>
              <p className="micro">
                Each choice loads a separate Isaac Lab run. The camera and timeline are interactive;
                changing a motor does not run physics in your browser.
              </p>
            </aside>
          </section>
          <section className="telemetry">
            <div className="live-stat">
              <span>
                <Target size={14} /> OBJECT HEIGHT
              </span>
              <strong>
                {fmt(sample ? (sample.cube[2] - 0.3) * 100 : undefined)}
                <small> cm above table</small>
              </strong>
              <p>Actual cube center</p>
            </div>
            <div className="live-stat">
              <span>
                <Activity size={14} /> GRIP CONTACT
              </span>
              <strong>
                {fmt(totalForce(0))}
                <small> / </small>
                {fmt(totalForce(2))}
                <small> N</small>
              </strong>
              <p>Left / right · summed normal forces</p>
            </div>
            <div className="live-stat">
              <span>
                <Zap size={14} /> DELIVERED TORQUE
              </span>
              <strong>
                {fmt(sample?.torque)}
                <small> N·m</small>
              </strong>
              <p>Largest absolute arm-joint torque</p>
            </div>
            <div className="live-stat">
              <span>
                <Thermometer size={14} /> HOTTEST WINDING
              </span>
              <strong>
                {fmt(sample?.temperature)}
                <small> °C</small>
              </strong>
              <p>Motor-model estimate</p>
            </div>
          </section>
          <section className="analysis-grid">
            <article className="plot-card">
              <div className="plot-heading">
                <div>
                  <p className="eyebrow">WHAT CHANGED OVER TIME</p>
                  <h3>Motion has an electrical cost.</h3>
                </div>
                <NativeSelect
                  aria-label="Telemetry metric"
                  value={metric}
                  onChange={(e) => setMetric(e.target.value)}
                >
                  {[
                    ["temperature", "Winding temperature"],
                    ["torque", "Delivered torque"],
                    ["current", "Torque current Iq"],
                    ["saturation", "Torque shortfall fraction"],
                  ].map(([v, l]) => (
                    <NativeSelectOption key={v} value={v}>
                      {l}
                    </NativeSelectOption>
                  ))}
                </NativeSelect>
              </div>
              <ChartContainer
                config={{
                  selected: {
                    label: selected?.label || "Selected",
                    color: "#19735e",
                  },
                  nominal: { label: "Nominal", color: "#b18a57" },
                }}
                className="time-chart"
              >
                <LineChart data={chart} margin={{ top: 10, right: 15, left: 0, bottom: 0 }}>
                  <CartesianGrid vertical={false} strokeDasharray="3 5" />
                  <XAxis
                    dataKey="t"
                    type="number"
                    domain={[0, 30]}
                    tickCount={7}
                    tickFormatter={(v) => `${v}s`}
                  />
                  <YAxis
                    width={45}
                    tickFormatter={(v) => Number(v).toFixed(metric === "saturation" ? 1 : 0)}
                  />
                  <Tooltip
                    formatter={(v: any) => `${Number(v).toFixed(2)} ${units[metric]}`}
                    labelFormatter={(v) => `${Number(v).toFixed(1)} s`}
                  />
                  {id !== "nominal" && (
                    <Line
                      type="monotone"
                      dataKey="nominal"
                      name="Nominal"
                      stroke="#b18a57"
                      strokeDasharray="4 4"
                      dot={false}
                      isAnimationActive={false}
                    />
                  )}
                  <Line
                    type="monotone"
                    dataKey="selected"
                    name={selected?.label || "Selected"}
                    stroke="#19735e"
                    strokeWidth={2}
                    dot={false}
                    isAnimationActive={false}
                  />
                  <ReferenceLine x={time} stroke="#d2793c" strokeDasharray="3 3" />
                </LineChart>
              </ChartContainer>
              <p className="micro">
                {id !== "nominal" ? "Green: selected design. Dashed: nominal. " : ""}
                Values are maxima across the arm motors, except shortfall, which is the fraction of
                joints missing requested torque by more than 10% or 0.5 N·m.
              </p>
            </article>
            <article className="plot-card envelope">
              <p className="eyebrow">THE DESIGN TRADEOFF</p>
              <h3>Torque available at speed.</h3>
              <ChartContainer
                config={{
                  torque: { label: "Available torque", color: "#d2793c" },
                }}
                className="envelope-chart"
              >
                <LineChart
                  data={selected?.envelope || []}
                  margin={{ top: 12, right: 12, left: 0, bottom: 0 }}
                >
                  <CartesianGrid vertical={false} strokeDasharray="3 5" />
                  <XAxis
                    dataKey="speed"
                    type="number"
                    domain={[0, 30]}
                    tickCount={4}
                    tickFormatter={(v) => `${v}`}
                  />
                  <YAxis width={45} />
                  <Tooltip
                    formatter={(v: any) => `${Number(v).toFixed(1)} N·m`}
                    labelFormatter={(v) => `${Number(v).toFixed(1)} rad/s`}
                  />
                  <Line
                    dataKey="torque"
                    name="Shoulder torque"
                    stroke="#d2793c"
                    strokeWidth={2}
                    dot={false}
                    isAnimationActive={false}
                  />
                </LineChart>
              </ChartContainer>
              <p className="micro">
                FOC model · shoulder gearing 12:1 · {selected?.temperature || 25} °C initial
                temperature. Speed in rad/s, torque in N·m. This is a modeled envelope, not a
                measured motor curve.
              </p>
            </article>
          </section>
          <section className="results-section">
            <p className="eyebrow">SAME SEQUENCE / ALL RECORDED OUTCOMES</p>
            <h3>Successful and failed runs stay visible.</h3>
            <p className="micro results-note">
              Target error measures final cube placement. Arm pose error measures the worse
              gripper’s distance from its commanded final pose, so a successful placement can still
              reveal poor tracking.
            </p>
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>Design</th>
                    <th>Kv / rpm·V⁻¹</th>
                    <th>Added kg / arm</th>
                    <th>Full sequence</th>
                    <th>Target error</th>
                    <th>Arm pose error</th>
                    <th>Peak winding</th>
                    <th>Inspect</th>
                  </tr>
                </thead>
                <tbody>
                  {manifest?.variants.map((v) => (
                    <tr key={v.id} className={v.id === id ? "selected-row" : ""}>
                      <td>{v.label}</td>
                      <td>{fmt(v.kv)}</td>
                      <td>{fmt(v.mass)}</td>
                      <td
                        className={v.result.checks?.full_sequence_success ? "positive" : "negative"}
                      >
                        {v.result.checks?.full_sequence_success ? "Passed" : "Failed"}
                      </td>
                      <td>
                        {fmt(
                          Math.hypot(
                            v.result.final_position[0] - 0.48,
                            v.result.final_position[1] - 0.3,
                          ) * 1000,
                        )}{" "}
                        mm
                      </td>
                      <td>{fmt(v.result.checks?.final_arm_pose_error_mm)} mm</td>
                      <td>{fmt(v.result.peak_temperature)} °C</td>
                      <td>
                        <Button
                          variant="ghost"
                          size="sm"
                          aria-label={`Inspect ${v.label}`}
                          onClick={() => {
                            setId(v.id);
                            seek(18);
                            setCamera("handoff");
                            window.scrollTo({ top: 100, behavior: "smooth" });
                          }}
                        >
                          <ArrowUpRight />
                        </Button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </>
      ) : task === "g1" || task === "anymal" ? (
        <Locomotion key={task} robot={task} />
      ) : task === "allegro" ? (
        <DexterousHand />
      ) : (
        <section className="existing-task">
          <div>
            <p className="eyebrow">
              {task === "g1"
                ? "EXISTING G1 EVALUATION"
                : task === "anymal"
                  ? "EXISTING ANYMAL EVALUATION"
                  : "EXISTING ALLEGRO DIAGNOSTIC"}
            </p>
            <h2>
              {task === "g1"
                ? "From shuffling to sustained steps."
                : task === "anymal"
                  ? "Gearing changes the walking task."
                  : "Dexterous reorientation baseline."}
            </h2>
            <p>
              {task === "g1"
                ? "Before / after, recorded in Isaac Lab. The updated gait reduced foot slip from 11.0 to 3.5 cm/s in the matched 20-second evaluation. It still has heading drift."
                : task === "anymal"
                  ? "Matched Isaac Lab runs with nominal and 6:1 gearing. The video includes a real force pulse and measured motor response."
                  : "The existing 40-second test uses changing orientation targets and a physical push. It is not a completed grasp–transfer–place task."}
            </p>
            <video
              src={
                task === "g1" ? "./g1.mp4" : task === "anymal" ? "./anymal.mp4" : "./allegro.mp4"
              }
              controls
              muted
              playsInline
              preload="metadata"
            />
          </div>
          <aside>
            <p className="eyebrow">ACTUATOR STUDY</p>
            <h3>
              {task === "g1"
                ? "Mass changes the gait."
                : task === "anymal"
                  ? "Torque, speed and recovery."
                  : "Motor design changes control."}
            </h3>
            <p>
              {task === "g1"
                ? "Adding 18.5 kg of actuator housing mass reduced world-forward speed from 0.605 to 0.365 m/s and increased foot slip from 3.5 to 15.1 cm/s with the same walking policy."
                : task === "anymal"
                  ? "The earlier study used 18 designs, 32 replicas per design and 120-second trials. Gear ratio affects available joint torque, motor speed and reflected inertia; the policy is held fixed. The full study’s 2 m/s stage fails repeatedly across all variants; this video is a separate walking demonstration."
                  : "The earlier study compared 18 designs with the same hand policy, including Kv, torque, gearing, winding resistance, mass and cooling."}
            </p>
            <a
              className="download-link"
              href={task === "g1" ? "./data/g1-study.md" : "./data/design-study.md"}
              download
            >
              <Download size={16} />
              Download measured results
            </a>
            <p className="micro">
              Fixed-policy simulation results. These measure sensitivity, not the best achievable
              performance after retraining each design.
            </p>
            <Button onClick={() => setTask("transfer")}>
              <ArrowRight />
              Open the complete transfer
            </Button>
          </aside>
        </section>
      )}
      {task === "transfer" && (
        <section className="method" id="method">
          <div>
            <p className="eyebrow">REPRODUCIBLE, WITH LIMITS</p>
            <h3>
              Inspect the physics
              <br />
              behind the presentation.
            </h3>
            <a
              href="https://dex-rubik-cube.yanjieze.com/"
              target="_blank"
              rel="noreferrer"
              className="quiet-link"
            >
              Presentation reference: dex-rubik-cube <ArrowUpRight size={13} />
            </a>
          </div>
          <div>
            <p>
              Two Franka arms execute a fixed task-space sequence with bounded Atlas FOC motor
              models. The grippers use 30 N linear drives. The cube moves through gravity and
              contact; it has no actuator or attachment. All meshes and body poses in the viewer
              come from the Isaac Lab scene.
            </p>
            <p>
              These are scripted-control simulation demonstrations, not a learned Rubik’s Cube
              solver or a hardware test. A successful placement requires the cube within 5 cm of the
              destination, near the table surface, and moving below 3 cm/s over the final second.
              Finger glows indicate measured contact with the cube.
            </p>
            <p>
              Kv variants use a fixed-copper rewind model: Kt changes inversely with Kv, and
              resistance and inductance change with the square of turns. Arm mass changes are
              applied to rigid bodies and their inertia. The path, gains and nominal gripper drives
              stay the same. Gravity compensation uses each variant’s actual modeled mass.
              Temperatures are model estimates; a 30-second cold run does not establish thermal
              endurance.
            </p>
            <div className="downloads">
              <a className="download-link" href="./nominal.mp4" target="_blank" rel="noreferrer">
                <Play size={14} />
                Watch nominal video
              </a>
              <a className="download-link" href="./data/protocol.json" download>
                <Download size={14} />
                Protocol & checks
              </a>
              <a className="download-link" href="./data/transfer.py" download>
                <Download size={14} />
                Isaac Lab source
              </a>
              <a className="download-link" href="./data/source.zip" download>
                <Download size={14} />
                Source bundle
              </a>
              <a className="download-link" href={`./data/${id}/telemetry.json`} download>
                <Download size={14} />
                Selected telemetry
              </a>
              <a className="download-link" href={`./data/${id}/poses.bin`} download>
                <Download size={14} />
                Body poses
              </a>
            </div>
          </div>
        </section>
      )}
      <footer>
        <FlaskConical size={15} /> ATLAS / Isaac Lab actuator experiments
        <span>Simulation evidence · Keep the failures visible.</span>
      </footer>
    </main>
  );
}
