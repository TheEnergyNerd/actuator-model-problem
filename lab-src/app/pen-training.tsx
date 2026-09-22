import { useEffect, useMemo, useRef, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import { Replay, loadScene, type Recording, type SceneData } from "./replay";
import { loadRecording, SyncedVideo } from "./locomotion";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";

type Row = {
  policy: string;
  model: string;
  motion_successes: number;
  native_contact_successes: number;
  trials: number;
};
type Manifest = {
  rows: Row[];
  summary: string[];
  training: { ideal: any[]; motor: any[] };
  transitions: number;
  test_seed0: number;
  test_trials: number;
};
const policies: Record<string, string> = {
  frozen: "Original frozen policy",
  ideal: "Fine-tuned · simplified model",
  motor: "Fine-tuned · motor model",
};
const models: Record<string, string> = {
  ideal: "Fixed torque limits",
  motor: "Motor model · 25°C start",
  "motor-hot": "Motor model · 100°C start",
};
export default function PenTraining({
  base = "./data/pen/training",
  bench = false,
}: {
  base?: string;
  bench?: boolean;
}) {
  const [left, setLeft] = useState("ideal"),
    [right, setRight] = useState("motor"),
    [model, setModel] = useState(bench ? "motor" : "motor-hot");
  const [mode, setMode] = useState("compare"),
    [playing, setPlaying] = useState(false),
    [time, setTime] = useState(0),
    [joint, setJoint] = useState(0);
  const [manifest, setManifest] = useState<Manifest | null>(null),
    [scene, setScene] = useState<SceneData | null>(null),
    [pair, setPair] = useState<Recording[] | null>(null),
    [error, setError] = useState("");
  const clock = useRef({ time: 0, playing: false, speed: 1 });
  clock.current.playing = playing;
  useEffect(() => {
    let active = true;
    fetch(`${base}/comparison.json`)
      .then((r) => {
        if (!r.ok) throw Error("Training comparison unavailable");
        return r.json() as Promise<Manifest>;
      })
      .then((m) => active && setManifest(m))
      .catch((e) => active && setError(e.message));
    return () => {
      active = false;
    };
  }, [base]);
  useEffect(() => {
    let active = true;
    setPair(null);
    setError("");
    setTime(0);
    setPlaying(false);
    clock.current.time = 0;
    Promise.all([
      loadScene("./data/pen/baseline/scene.json"),
      loadRecording(`${base}/${left}-${model}`),
      loadRecording(`${base}/${right}-${model}`),
    ])
      .then(([s, a, b]) => {
        if (active) {
          setScene(s);
          setPair([a, b]);
        }
      })
      .catch((e) => active && setError(e.message));
    return () => {
      active = false;
    };
  }, [left, right, model, base]);
  useEffect(() => {
    let frame = 0,
      last = performance.now();
    function tick(now: number) {
      if (clock.current.playing && pair) {
        clock.current.time = Math.min(
          pair[0].result.duration,
          clock.current.time + Math.min((now - last) / 1000, 0.1) * clock.current.speed,
        );
        setTime(clock.current.time);
        if (clock.current.time >= pair[0].result.duration) setPlaying(false);
      }
      last = now;
      frame = requestAnimationFrame(tick);
    }
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [pair]);
  const curve = useMemo(
    () =>
      manifest?.training.ideal.map((r, i) => ({
        iteration: r.iteration,
        ideal: r.reward_mean,
        motor: manifest.training.motor[i]?.reward_mean,
      })) ?? [],
    [manifest],
  );
  const rotationData = useMemo(
    () =>
      pair?.[0].samples.map((s: any, i: number) => ({
        t: s.t,
        left: s.turns,
        right: pair[1].samples[i]?.turns,
      })) ?? [],
    [pair],
  );
  const torqueData = useMemo(
    () =>
      pair?.[1].samples.map((s: any) => ({
        t: s.t,
        requested: s.motor_joints.requested[joint],
        delivered: s.motor_joints.applied[joint],
      })) ?? [],
    [pair, joint],
  );
  const seek = (t: number) => {
    clock.current.time = t;
    setTime(t);
  };
  const selectors = (
    <div className="hand-trial-selector">
      <label htmlFor="training-left">Left policy</label>
      <select id="training-left" value={left} onChange={(e) => setLeft(e.target.value)}>
        {Object.entries(policies).map(([id, label]) => (
          <option key={id} value={id}>
            {label}
          </option>
        ))}
      </select>
      <label htmlFor="training-right">Right policy</label>
      <select id="training-right" value={right} onChange={(e) => setRight(e.target.value)}>
        {Object.entries(policies).map(([id, label]) => (
          <option key={id} value={id}>
            {label}
          </option>
        ))}
      </select>
      <label htmlFor="training-world">Shared test conditions</label>
      <select id="training-world" value={model} onChange={(e) => setModel(e.target.value)}>
        {Object.entries(models).map(([id, label]) => (
          <option key={id} value={id}>
            {label}
          </option>
        ))}
      </select>
    </div>
  );
  if (error)
    return (
      <>
        {selectors}
        <p role="alert">{error}</p>
      </>
    );
  if (!manifest || !scene || !pair)
    return (
      <>
        {selectors}
        <p role="status">Loading the matched training experiment…</p>
      </>
    );
  const sample = pair[1].samples[Math.min(719, Math.floor(time * 60))] as any;
  const j = sample.motor_joints;
  const row = (p: string) => manifest.rows.find((r) => r.policy === p && r.model === model)!;
  return (
    <>
      {!bench && (      <section className="motion-analysis">
        <div>
          <p className="eyebrow">MATCHED POLICY TRAINING / ISAAC LAB</p>
          <h2>Train with motor limits. Test under the same conditions.</h2>
          <p>
            {bench
              ? "Replay pair from preselected training seed 44000000,"
              : "Two copies of the supplied Sharpa policy,"}{" "}
            {manifest.transitions.toLocaleString()} training transitions each. One learns with fixed
            torque limits; the other learns with current, speed and thermal constraints. The
            original frozen policy is a separate control.
          </p>
        </div>
        <div>
          {manifest.summary.map((s) => (
            <p key={s}>{s}</p>
          ))}
        </div>
      </section>
      )}
      {selectors}
      <section className="motion-workbench">
        <div className="motion-stage">
          <div className="motion-heading">
            <span>Same test seed · {manifest.test_seed0}</span>
            <span>PHYSX / 240 Hz · REPLAY / 60 FPS</span>
          </div>
          <div className="motion-phase">
            <h2>{models[model]}</h2>
            <p>
              Trial 0 was selected before training. Both sides start from the same prepared grasp.
            </p>
          </div>
          <div className="motion-pair dual">
            {mode === "both" ? (
              <>
                <SyncedVideo
                  src={`${base}/${right}-${model}/video.mp4`}
                  clock={clock}
                  label={policies[right] + " · Isaac render"}
                />
                <Replay
                  sceneData={scene}
                  recording={pair[1]}
                  playback={clock}
                  cameraView="overview"
                  showPath={false}
                  label={policies[right] + " · measured replay"}
                />
              </>
            ) : (
              pair.map((r, i) =>
                mode === "video" ? (
                  <SyncedVideo
                    key={`${i}-${left}-${right}-${model}`}
                    src={`${base}/${i === 0 ? left : right}-${model}/video.mp4`}
                    clock={clock}
                    label={policies[i === 0 ? left : right]}
                  />
                ) : (
                  <Replay
                    key={i}
                    sceneData={scene}
                    recording={r}
                    playback={clock}
                    cameraView="overview"
                    showPath={false}
                    label={`${policies[i === 0 ? left : right]} · ${r.result.task_success ? "motion pass" : "motion fail"}`}
                  />
                ),
              )
            )}
          </div>
          <div className="motion-toolbar">
            <Button
              aria-label={playing ? "Pause training replay" : "Play training replay"}
              onClick={() => {
                if (time >= pair[0].result.duration) seek(0);
                setPlaying(!playing);
              }}
            >
              {playing ? "Pause" : "Play"}
            </Button>
            <Button
              variant="ghost"
              onClick={() => {
                setPlaying(false);
                seek(0);
              }}
            >
              Restart
            </Button>
            <select
              aria-label="Training replay speed"
              defaultValue="1"
              onChange={(e) => (clock.current.speed = Number(e.target.value))}
            >
              <option value="1">1×</option>
              <option value="0.5">0.5×</option>
              <option value="0.25">0.25×</option>
            </select>
            <span>
              {time.toFixed(1)} / {pair[0].result.duration.toFixed(1)} s
            </span>
          </div>
          <div className="transport">
            <Slider
              aria-label="Training replay time"
              min={0}
              max={pair[0].result.duration}
              step={1 / 60}
              value={[time]}
              onValueChange={(v) => seek(typeof v === "number" ? v : v[0])}
            />
          </div>
        </div>
        <aside className="motion-inspector">
          <p className="eyebrow">Held-out motion results</p>
          <h2>
            {row(left).motion_successes}/{manifest.test_trials} vs {row(right).motion_successes}/
            {manifest.test_trials}
          </h2>
          <p>Left versus right, across all test seeds. The displayed attempt is only one trial.</p>
          <div className="motion-modes">
            {[
              ["compare", "Paired 3D"],
              ["video", "Paired videos"],
              ["both", "Right video + 3D"],
            ].map(([id, label]) => (
              <Button
                key={id}
                variant={mode === id ? "default" : "ghost"}
                onClick={() => setMode(id)}
              >
                {label}
              </Button>
            ))}
          </div>
          <label htmlFor="training-joint">Inspect right-hand joint</label>
          <select
            id="training-joint"
            value={joint}
            onChange={(e) => setJoint(Number(e.target.value))}
          >
            {pair[1].result.motor_model.joint_names.map((n: string, i: number) => (
              <option key={n} value={i}>
                {n.replace("right_", "")}
              </option>
            ))}
          </select>
          <dl className="motion-values">
            <dt>Net rotations</dt>
            <dd>{sample.turns.toFixed(2)}</dd>
            <dt>Requested torque</dt>
            <dd>{j.requested[joint].toFixed(3)} Nm</dd>
            <dt>Delivered torque</dt>
            <dd>{j.applied[joint].toFixed(3)} Nm</dd>
            <dt>Motor speed</dt>
            <dd>{Math.abs(j.motor_rpm[joint]).toFixed(0)} RPM</dd>
            {j.current && (
              <>
                <dt>Model current · peak dq</dt>
                <dd>{j.current[joint].toFixed(2)} A</dd>
                <dt>Model winding temperature</dt>
                <dd>{j.temperature[joint].toFixed(1)} °C</dd>
              </>
            )}
          </dl>
          <p>
            {bench
              ? "mj5208 electrical calibration; virtual remote drive with the documented gearing and thermal settings. Both policies trained with 25°C starts. Temperature is not observed directly."
              : "Generic assumed motor parameters; no Sharpa hardware calibration. Neither policy sees temperature directly. Training motor starts span 25–100°C."}
          </p>
        </aside>
      </section>
      <section className="motion-analysis">
        <div>
          <h2>Rotations · same test conditions</h2>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={rotationData}>
              <XAxis dataKey="t" type="number" domain={[0, pair[0].result.duration]} />
              <YAxis />
              <Tooltip />
              <Line
                dataKey="left"
                name={policies[left]}
                stroke="#b58b45"
                dot={false}
                isAnimationActive={false}
              />
              <Line
                dataKey="right"
                name={policies[right]}
                stroke="#177960"
                dot={false}
                isAnimationActive={false}
              />
              <ReferenceLine x={time} />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div>
          <h2>Right joint · torque delivery</h2>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={torqueData}>
              <XAxis dataKey="t" type="number" domain={[0, pair[0].result.duration]} />
              <YAxis unit=" Nm" width={65} />
              <Tooltip />
              <Line dataKey="requested" stroke="#b58b45" dot={false} isAnimationActive={false} />
              <Line dataKey="delivered" stroke="#177960" dot={false} isAnimationActive={false} />
              <ReferenceLine x={time} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </section>
      <section className="precision-results">
        <h2>
          {bench ? "Selected training seed · all test conditions" : "All nine held-out evaluations"}
        </h2>
        <div style={{ overflowX: "auto" }}>
          <table>
            <thead>
              <tr>
                <th>Policy</th>
                <th>Test model</th>
                <th>Motion</th>
                <th>Additional contact gate</th>
                <th>Full outcomes</th>
              </tr>
            </thead>
            <tbody>
              {manifest.rows.map((r) => (
                <tr key={`${r.policy}-${r.model}`}>
                  <td>{policies[r.policy]}</td>
                  <td>{models[r.model]}</td>
                  <td>
                    {r.motion_successes}/{r.trials}
                  </td>
                  <td>
                    {r.native_contact_successes}/{r.trials}
                  </td>
                  <td>
                    <a href={`${base}/${r.policy}-${r.model}/evaluation.json`}>Every seed</a>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p>
          Shared evaluation seeds {manifest.test_seed0}–
          {manifest.test_seed0 + manifest.test_trials - 1}, separate from training and the earlier
          development batch. Same frozen motion gate; additional contact checks use PhysX
          separations and differ from the reference's independent collision audit.{" "}
          {bench
            ? "The three-seed study is reported above; this replay panel shows the preselected first training seed."
            : "One training seed per condition cannot establish a robust advantage across repeated training runs."}
        </p>
      </section>
      <section className="motion-analysis">
        <div>
          <h2>Training reward through updates</h2>
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={curve}>
              <XAxis dataKey="iteration" />
              <YAxis />
              <Tooltip />
              <Line name="Simplified training" dataKey="ideal" stroke="#b58b45" dot={false} />
              <Line name="Motor-model training" dataKey="motor" stroke="#177960" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div>
          <h2>What was actually trained</h2>
          <p>
            Atlas PPO fine-tuning from the{" "}
            <a href="https://github.com/jianglongye/dexterous-astra">Dexterous Astra</a> checkpoint.
            The reference does not distribute its trainer or reward implementation. This experiment
            uses a new reward for supported rotation, quiet holding and avoiding drops, with a small
            penalty for departing from the original policy. It is not training from scratch or a
            reproduction of upstream training.
          </p>
          <p>
            Same budget and optimizer settings; final checkpoint selected by iteration, not test
            performance. Training reward is a surrogate, not the reported motion success rate.
            Prepared training grasps repeat; test grasps use separate seeds.
          </p>
          <a href={`${base}/comparison.json`}>Training logs, protocol and checkpoint hashes</a> ·{" "}
          <a href={`${base}/${right}-${model}/result.json`}>Selected replay provenance</a>
          {" · "}
          <a href="./data/pen/SHARPA-LICENSE.txt">Sharpa asset license</a>
        </div>
      </section>
    </>
  );
}
