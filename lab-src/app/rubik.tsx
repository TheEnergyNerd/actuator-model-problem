import ActuatorDesign from "./actuator-design";
import { useEffect, useRef, useState } from "react";
import { Replay, loadScene, type SceneData, type Recording, type Sample } from "./replay";
import { SyncedVideo, loadRecording } from "./locomotion";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Play, Pause, RotateCcw, Hand } from "lucide-react";
type RubikSample = Sample & {
  move: string;
  move_index: number;
  attempt: number;
  expected_facelets: string;
  decoded: { aligned: boolean; facelets: string | null };
  max_target_error_deg: number;
  max_anchor_error_m: number;
  joint_torque_max_nm: number;
};
export default function Rubik() {
  const [selected, setSelected] = useState("free");
  const path = selected === "free" ? "./data/rubik/free-turn" : selected === "single" ? "./data/rubik" : "./data/rubik/full-attempt";
  return (
    <>
      <div className="hand-trial-selector" role="group" aria-label="Rubik recording">
        <Button variant={selected === "free" ? "default" : "outline"} onClick={() => setSelected("free")} aria-pressed={selected === "free"}>Two-hand free-cube turn</Button>
        <Button
          variant={selected === "single" ? "default" : "outline"}
          onClick={() => setSelected("single")}
          aria-pressed={selected === "single"}
        >
          Fixture quarter-turn
        </Button>
        <Button
          variant={selected === "full" ? "default" : "outline"}
          onClick={() => setSelected("full")}
          aria-pressed={selected === "full"}
        >
          Fixture scramble attempt
        </Button>
      </div>
      <RubikRecording key={path} path={path} />
    </>
  );
}
function RubikRecording({ path }: { path: string }) {
  const [scene, setScene] = useState<SceneData | null>(null),
    [run, setRun] = useState<Recording | null>(null);
  const [error, setError] = useState(""),
    [time, setTime] = useState(0),
    [playing, setPlaying] = useState(false),
    [mode, setMode] = useState("both");
  const clock = useRef({ time: 0, playing: false, speed: 1 });
  useEffect(() => {
    let gone = false;
    Promise.all([loadScene(path.endsWith("free-turn") ? `${path}/scene.json` : "./data/rubik/scene.json"), loadRecording(path)])
      .then(([s, r]) => {
        if (!gone) {
          setScene(s);
          setRun(r);
        }
      })
      .catch((e) => {
        if (!gone) setError(e.message);
      });
    return () => {
      gone = true;
    };
  }, [path]);
  useEffect(() => {
    clock.current.playing = playing;
  }, [playing]);
  useEffect(() => {
    let frame = 0,
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
      if (now - ui > 50) {
        setTime(clock.current.time);
        ui = now;
      }
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [run]);
  const seek = (t: number) => {
    clock.current.time = t;
    setTime(t);
  };
  if (error)
    return (
      <section className="existing-task" role="alert">
        Rubik recording unavailable: {error}
      </section>
    );
  if (!run || !scene)
    return (
      <section className="existing-task">Loading Wuji Hand 2 and the articulated cube…</section>
    );
  const sample = run.samples[
    Math.min(run.samples.length - 1, Math.floor(time * run.result.fps))
  ] as RubikSample;
  const matched = sample.decoded?.facelets && sample.decoded.facelets === sample.expected_facelets;
  const milestones = (run.samples as RubikSample[]).filter(
    (row, index, rows) =>
      !index ||
      row.move_index !== rows[index - 1].move_index ||
      row.attempt !== rows[index - 1].attempt,
  );
  return (
    <>
    <section className="motion-workbench hand-workbench">
      <div className="motion-stage">
        <div className="motion-heading">
          <span>
            <Hand size={20} /> WUJI HAND 2 / 5 fingers · 20 actuated joints per hand
          </span>
          <span>PHYSX / 1,000 Hz · REPLAY / 50 FPS</span>
        </div>
        <div className="motion-phase">
          <p className="eyebrow">
            {run.result.fixture
              ? "FIXTURE TEST / CONTACT-DRIVEN FACE TURNS"
              : "BIMANUAL RUBIK MANIPULATION"}
          </p>
          <h2>
            {sample.phase || "Turn"} · {sample.move || "U"}
          </h2>
          <p>Native hand geometry. Articulated cube. Measured motion.</p>
        </div>
        <div className={`motion-pair ${mode === "both" ? "dual" : ""}`}>
          {mode !== "video" && (
            <Replay
              sceneData={scene}
              recording={run}
              playback={clock}
              cameraView="overview"
              showPath={false}
              label="Wuji Hand 2 · drag to inspect"
            />
          )}
          {mode !== "3d" && (
            <SyncedVideo
              src={`${path}/video.mp4`}
              clock={clock}
              label="Isaac render · same measured poses"
            />
          )}
        </div>
        <div className="motion-toolbar">
          <span>Drag to orbit · scroll to zoom</span>
          {["both", "3d", "video"].map((m) => (
            <Button
              key={m}
              variant={mode === m ? "default" : "ghost"}
              onClick={() => setMode(m)}
              aria-pressed={mode === m}
            >
              {m === "both" ? "3D + video" : m === "3d" ? "3D replay" : "Video"}
            </Button>
          ))}
        </div>
        <div className="transport">
          <Button
            aria-label={playing ? "Pause" : "Play"}
            onClick={() => {
              if (!playing && time >= run.result.duration) seek(0);
              setPlaying(!playing);
            }}
          >
            {playing ? <Pause /> : <Play />}
          </Button>
          <Button variant="ghost" aria-label="Restart" onClick={() => seek(0)}>
            <RotateCcw />
          </Button>
          <Slider
            aria-label="Rubik video and replay timeline"
            value={[time]}
            min={0}
            max={run.result.duration}
            step={0.02}
            onValueChange={(v) => seek(typeof v === "number" ? v : v[0])}
          />
          <span>
            {time.toFixed(1)} / {run.result.duration.toFixed(1)} s
          </span>
          <select
            aria-label="Playback speed"
            defaultValue="1"
            onChange={(event) => {
              clock.current.speed = Number(event.target.value);
            }}
          >
            {[0.5, 1, 2, 4].map((speed) => (
              <option key={speed} value={speed}>
                {speed}×
              </option>
            ))}
          </select>
        </div>
        {milestones.length > 1 && (
          <div className="hand-milestones rubik-moves" role="group" aria-label="Jump to a turn">
            {milestones.map((row) => (
              <Button
                key={`${row.move_index}-${row.attempt}`}
                variant={
                  sample.move_index === row.move_index && sample.attempt === row.attempt
                    ? "default"
                    : "outline"
                }
                onClick={() => seek(row.t)}
              >
                {row.move_index + 1}. {row.move}
                {row.attempt > 1 ? ` · retry ${row.attempt - 1}` : ""}
              </Button>
            ))}
          </div>
        )}
      </div>
      <aside className="motion-inspector">
        <p className="eyebrow">MEASURED CUBE STATE</p>
        <h2>
          {matched ? "Target aligned" : sample.decoded?.aligned ? "Aligned" : "Turning / unaligned"}
        </h2>
        <p>
          Measured stickers must match the expected legal state within 3°, with joint anchor error
          below 0.5 mm, for a continuous 0.4-second final hold{run.result.fixture ? " after hand release" : " while supported by the hands"}.
        </p>
        <dl className="rubik-metrics">
          <div>
            <dt>Turn</dt>
            <dd>
              {sample.move_index + 1} / {run.result.moves?.length || 1}
            </dd>
          </div>
          <div>
            <dt>Stable hold validation</dt>
            <dd>{run.result.passed ? "Passed" : "Not yet passed"}</dd>
          </div>
          <div>
            <dt>Completed quarter-turns</dt>
            <dd>
              {run.result.completed_moves} / {run.result.moves?.length || 1}
            </dd>
          </div>
          <div>
            <dt>Final sticker state</dt>
            <dd>{run.result.final.decoded.solved ? "Solved" : "Unsolved"}</dd>
          </div>
          <div>
            <dt>Largest cubie target error</dt>
            <dd>{sample.max_target_error_deg.toFixed(1)}°</dd>
          </div>
          <div>
            <dt>Peak finger torque now</dt>
            <dd>{sample.joint_torque_max_nm.toFixed(3)} Nm</dd>
          </div>
          <div>
            <dt>Joint anchor error</dt>
            <dd>{(sample.max_anchor_error_m * 1000).toFixed(3)} mm</dd>
          </div>
          <div>
            <dt>Cube motors</dt>
            <dd>0</dd>
          </div>
        </dl>
        <p>
          Starting scramble: <code>{run.result.scramble}</code>
        </p>
        <p>
          {run.result.fixture
            ? "The core is fixed for this test. This is not yet a free, two-hand scramble-and-solve demonstration."
            : "A fixture prepares the grasp for the first two seconds, then releases. The turn and final hold are supported through hand contact, with no cube motors or pose resets. This is one quarter-turn, not a full-scramble solve."}
        </p>
        <p>
          Finger position control and bounded wrist force/torque servos. Passive cube detents depend
          on current geometry, never the requested move.
        </p>
        <a href={`${path}/result.json`} target="_blank" rel="noreferrer">
          Run configuration and validation ↗
        </a>
        <br />
        <a
          href="https://github.com/wuji-technology/wuji-description"
          target="_blank"
          rel="noreferrer"
        >
          Native Wuji hand assets ↗
        </a>
      </aside>
    </section>
    <ActuatorDesign notice="The Wuji recordings test grasp and turning controllers. A matched Atlas winding, gearing or mass sweep has not been recorded for these hands."/>
    </>
  );
}
