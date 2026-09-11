import { useEffect, useRef, useState } from "react";
import { Replay, loadScene, type SceneData, type Recording } from "./replay";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { NativeSelect, NativeSelectOption } from "@/components/ui/native-select";
import { Play, Pause, RotateCcw, GitCompareArrows, Boxes } from "lucide-react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
type Variant = { id: string; label: string; description: string };
type Manifest = {
  robot: string;
  variants: Variant[];
  physics_hz: number;
  fps: number;
  push_time: number;
  description: string;
};
const cache = new Map<string, Promise<Recording>>();
export async function loadRecording(path: string) {
  if (!cache.has(path))
    cache.set(
      path,
      Promise.all(
        ["poses.bin", "telemetry.json", "result.json"].map(async (name): Promise<any> => {
          const r = await fetch(`${path}/${name}`);
          if (!r.ok) throw Error(`Recording unavailable (${r.status})`);
          return name.endsWith(".bin") ? r.arrayBuffer() : r.json();
        }),
      )
        .then(([buffer, samples, result]) => {
          if (
            buffer.byteLength !== result.frames * result.bodies * 28 ||
            samples.length !== result.frames
          )
            throw Error("Recording integrity check failed");
          return { poses: new Float32Array(buffer), samples, result };
        })
        .catch((e) => {
          cache.delete(path);
          throw e;
        }),
    );
  return cache.get(path)!;
}
export function SyncedVideo({
  src,
  clock,
  label,
}: {
  src: string;
  clock: React.RefObject<{ time: number; playing: boolean; speed: number }>;
  label: string;
}) {
  const ref = useRef<HTMLVideoElement>(null);
  useEffect(() => {
    let frame = 0;
    const sync = () => {
      const v = ref.current,
        c = clock.current;
      if (v && v.readyState >= 1) {
        if (!v.seeking && Math.abs(v.currentTime - c.time) > 0.12) v.currentTime = c.time;
        v.playbackRate = c.speed;
        if (c.playing && v.paused) void v.play().catch(() => {});
        if (!c.playing && !v.paused) v.pause();
      }
      frame = requestAnimationFrame(sync);
    };
    sync();
    return () => cancelAnimationFrame(frame);
  }, [src, clock]);
  return (
    <div className="motion-video">
      <video key={src} ref={ref} src={src} muted playsInline preload="auto" aria-label={label} />
      <span className="canvas-label">{label} · Isaac Lab video</span>
    </div>
  );
}
export default function Locomotion({ robot }: { robot: "g1" | "anymal" }) {
  const [manifest, setManifest] = useState<Manifest | null>(null),
    [scene, setScene] = useState<SceneData | null>(null),
    [run, setRun] = useState<Recording | null>(null),
    [baseline, setBaseline] = useState<Recording | null>(null);
  const [id, setId] = useState("nominal"),
    [mode, setMode] = useState("both"),
    [compare, setCompare] = useState(false),
    [camera, setCamera] = useState("follow"),
    [trail, setTrail] = useState(true),
    [error, setError] = useState(""),
    [time, setTime] = useState(0),
    [playing, setPlaying] = useState(false),
    [speed, setSpeed] = useState(1),
    [metric, setMetric] = useState("speed");
  const clock = useRef({ time: 0, playing: false, speed: 1 });
  const root = `./data/locomotion/${robot}`;
  useEffect(() => {
    let cancelled = false;
    setManifest(null);
    setRun(null);
    setError("");
    setId("nominal");
    setTime(0);
    setPlaying(false);
    clock.current.time = 0;
    Promise.all([
      fetch(`${root}/manifest.json`).then((r) => {
        if (!r.ok) throw Error("Robot recording unavailable");
        return r.json() as Promise<Manifest>;
      }),
      loadScene(`${root}/scene.json`),
      loadRecording(`${root}/nominal`),
    ])
      .then(([m, s, r]) => {
        if (!cancelled) {
          setManifest(m);
          setScene(s);
          setBaseline(r);
          setRun(r);
        }
      })
      .catch((e) => !cancelled && setError(e.message));
    return () => {
      cancelled = true;
    };
  }, [root]);
  useEffect(() => {
    if (!manifest) return;
    let cancelled = false;
    setRun(null);
    setPlaying(false);
    setError("");
    loadRecording(`${root}/${id}`)
      .then((r) => {
        if (!cancelled) setRun(r);
      })
      .catch((e) => !cancelled && setError(e.message));
    return () => {
      cancelled = true;
    };
  }, [id, manifest, root]);
  useEffect(() => {
    clock.current.playing = playing;
    clock.current.speed = speed;
  }, [playing, speed]);
  useEffect(() => {
    let raf = 0,
      last = performance.now(),
      ui = 0;
    const tick = (now: number) => {
      const dt = Math.min(0.1, (now - last) / 1000);
      last = now;
      if (clock.current.playing && run) {
        clock.current.time = Math.min(
          run.result.duration,
          clock.current.time + dt * clock.current.speed,
        );
        if (clock.current.time >= run.result.duration) setPlaying(false);
      }
      if (now - ui > 60) {
        setTime(clock.current.time);
        ui = now;
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [run]);
  const seek = (t: number) => {
    clock.current.time = t;
    setTime(t);
  };
  if (error)
    return (
      <section className="existing-task" role="alert">
        {error}
        <Button onClick={() => window.location.reload()}>Reload recordings</Button>
      </section>
    );
  if (!manifest || !scene || !run || !baseline)
    return (
      <section className="existing-task" aria-live="polite">
        Loading {robot === "g1" ? "G1" : "ANYmal"} geometry and recorded physics…
      </section>
    );
  const variant = manifest.variants.find((v) => v.id === id)!;
  const index = Math.min(run.samples.length - 1, Math.floor(time * run.result.fps)),
    sample = run.samples[index] as any;
  const result = run.result,
    group = Object.values(result.groups)[0] as any;
  const chart = run.samples
    .filter((_, i) => i % 5 === 0)
    .map((s: any, i) => ({ ...s, baseline: (baseline.samples[i * 5] as any)?.[metric] }));
  const units: Record<string, string> = {
    speed: "m/s",
    torque: "Nm",
    current: "A",
    temperature: "°C",
    saturation: "fraction",
  };
  const surface = (r: Recording, label: string, key: string) => (
    <div className={`motion-pair ${mode === "both" ? "dual" : ""}`}>
      {mode !== "video" && (
        <Replay
          sceneData={scene}
          recording={r}
          playback={clock}
          cameraView={camera}
          showPath={trail}
          label={label}
        />
      )}
      {mode !== "3d" && (
        <SyncedVideo src={`${root}/${key}/video.mp4`} clock={clock} label={label} />
      )}
    </div>
  );
  return (
    <>
      <section className="motion-workbench">
        <div className="motion-stage">
          <div className="motion-heading">
            <span>
              <Boxes size={20} />
              {manifest.robot} · One policy. Different motors.
            </span>
            <span>
              PHYSX / {manifest.physics_hz} Hz · REPLAY / {manifest.fps} FPS
            </span>
          </div>
          <div className="motion-phase">
            <p className="eyebrow">COMMANDED PHASE</p>
            <h2>{sample.phase}</h2>
            <p>
              {sample.target_speed.toFixed(1)} m/s forward command ·{" "}
              {sample.force_N
                ? `${sample.force_N.toFixed(0)} N lateral force`
                : "Recorded body motion"}
            </p>
          </div>
          <div className={compare ? "motion-compare" : ""}>
            {surface(run, variant.label, id)}
            {compare && surface(baseline, "Nominal reference", "nominal")}
          </div>
          <div className="motion-toolbar">
            <span>Drag to orbit · scroll to zoom</span>
            <NativeSelect
              aria-label="Camera"
              value={camera}
              onChange={(e) => setCamera(e.target.value)}
            >
              <NativeSelectOption value="follow">Follow robot</NativeSelectOption>
              <NativeSelectOption value="overview">Overview</NativeSelectOption>
              <NativeSelectOption value="front">Front</NativeSelectOption>
              <NativeSelectOption value="top">Top</NativeSelectOption>
            </NativeSelect>
            <Button variant="ghost" aria-pressed={trail} onClick={() => setTrail(!trail)}>
              Body trail {trail ? "on" : "off"}
            </Button>
            <Button
              variant={compare ? "default" : "ghost"}
              aria-pressed={compare}
              onClick={() => setCompare(!compare)}
            >
              <GitCompareArrows />
              Compare
            </Button>
          </div>
          <div className="transport">
            <Button aria-label={playing ? "Pause" : "Play"} onClick={() => setPlaying(!playing)}>
              {playing ? <Pause /> : <Play />}
            </Button>
            <Button variant="ghost" aria-label="Restart" onClick={() => seek(0)}>
              <RotateCcw />
            </Button>
            <Slider
              value={[time]}
              min={0}
              max={result.duration}
              step={1 / result.fps}
              onValueChange={(v) => seek(Array.isArray(v) ? v[0] : v)}
              aria-label="Synchronized video and 3D timeline"
            />
            <span>
              {time.toFixed(1)} / {result.duration.toFixed(1)} s
            </span>
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
        </div>
        <aside className="motion-inspector">
          <p className="eyebrow">ACTUATOR DESIGN</p>
          <h2>Watch the motor change the motion.</h2>
          <div className="motion-modes">
            {["both", "3d", "video"].map((m) => (
              <Button
                key={m}
                variant={mode === m ? "default" : "outline"}
                onClick={() => setMode(m)}
                aria-pressed={mode === m}
              >
                {m === "both" ? "3D + video" : m === "3d" ? "3D replay" : "Video"}
              </Button>
            ))}
          </div>
          <label className="eyebrow" htmlFor="motor-design">
            RECORDED DESIGN
          </label>
          <NativeSelect id="motor-design" value={id} onChange={(e) => setId(e.target.value)}>
            {manifest.variants.map((v) => (
              <NativeSelectOption key={v.id} value={v.id}>
                {v.label}
              </NativeSelectOption>
            ))}
          </NativeSelect>
          <h3>{variant.label}</h3>
          <p>{variant.description}</p>
          <dl className="motion-values">
            <dt>Body-forward speed</dt>
            <dd>{sample.speed.toFixed(2)} m/s</dd>
            <dt>Peak joint torque now</dt>
            <dd>{sample.torque.toFixed(1)} Nm</dd>
            <dt>Maximum winding temperature</dt>
            <dd>{sample.temperature.toFixed(1)} °C</dd>
            <dt>Joints with torque shortfall</dt>
            <dd>{(sample.saturation * 100).toFixed(0)}%</dd>
            <dt>Falls / resets so far</dt>
            <dd>{sample.failures}</dd>
            <dt>Added actuator mass</dt>
            <dd>{result.added_total_mass_kg.toFixed(1)} kg</dd>
          </dl>
          <p className="micro">
            One recorded trial per design. The trained policy and commanded task are held fixed.
            Normal episode resets are retained and counted.
          </p>
          <Button variant="outline" onClick={() => seek(manifest.push_time)}>
            Jump to force pulse
          </Button>
        </aside>
      </section>
      <section className="motion-analysis" id="method">
        <div>
          <p className="eyebrow">SYNCHRONIZED TELEMETRY</p>
          <h2>See when the response changes.</h2>
          <NativeSelect
            aria-label="Telemetry metric"
            value={metric}
            onChange={(e) => setMetric(e.target.value)}
          >
            {Object.keys(units).map((m) => (
              <NativeSelectOption key={m} value={m}>
                {m} ({units[m]})
              </NativeSelectOption>
            ))}
          </NativeSelect>
          <div style={{ height: 260, width: "100%" }}>
            <ResponsiveContainer initialDimension={{ width: 600, height: 260 }}>
              <LineChart data={chart}>
                <XAxis dataKey="t" type="number" domain={[0, result.duration]} unit="s" />
                <YAxis width={55} />
                <Tooltip />
                <Line
                  dataKey={metric}
                  name={variant.label}
                  stroke="#17725c"
                  dot={false}
                  isAnimationActive={false}
                />
                {compare && (
                  <Line
                    dataKey="baseline"
                    name="Nominal"
                    stroke="#c47a40"
                    dot={false}
                    isAnimationActive={false}
                  />
                )}
                <ReferenceLine x={time} stroke="#203d34" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
        <div>
          <p className="eyebrow">MEASURED RUN</p>
          <h3>
            {manifest.robot} · {variant.label}
          </h3>
          <p>{manifest.description}</p>
          <dl className="motion-values">
            <dt>Mean speed</dt>
            <dd>{result.forward_speed_m_s_mean.toFixed(2)} m/s</dd>
            <dt>Velocity RMSE</dt>
            <dd>{result.velocity_rmse_m_s.toFixed(2)} m/s</dd>
            <dt>Failure events</dt>
            <dd>{result.failures}</dd>
            <dt>Peak winding temperature</dt>
            <dd>{result.peak_temperature_C.toFixed(1)} °C</dd>
            <dt>First actuator group: Kv</dt>
            <dd>{group.Kv_phase_peak_rpm_per_V.toFixed(1)} rpm/V</dd>
            <dt>Gear ratio</dt>
            <dd>{group.gear_ratio.toFixed(1)}:1</dd>
            <dt>Peak current limit</dt>
            <dd>{group.peak_current_A.toFixed(1)} A</dd>
            <dt>Bus voltage</dt>
            <dd>{group.V_bus_V.toFixed(1)} V</dd>
          </dl>
          <p className="micro">
            Kv uses the phase-peak back-EMF convention. Full group parameters are included in the
            result file. Thermal values are modeled, not measured hardware temperatures.
          </p>
          <div className="motion-downloads">
            {["result.json", "telemetry.json", "poses.bin", "video.mp4"].map((name) => (
              <a key={name} href={`${root}/${id}/${name}`} download>
                {name}
              </a>
            ))}
            <a href={`${root}/results.json`} download>
              Full protocol
            </a>
            <a href="./data/locomotion/source.zip" download>
              Isaac Lab source
            </a>
          </div>
        </div>
      </section>
    </>
  );
}
