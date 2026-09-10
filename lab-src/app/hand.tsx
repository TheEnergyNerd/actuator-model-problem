import { useEffect, useRef, useState } from "react";
import { Replay, loadScene, type SceneData, type Recording } from "./replay";
import { SyncedVideo, loadRecording } from "./locomotion";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { NativeSelect, NativeSelectOption } from "@/components/ui/native-select";
import { Play, Pause, RotateCcw, Hand, Target } from "lucide-react";
export default function DexterousHand() {
  const [run, setRun] = useState<Recording | null>(null),
    [scene, setScene] = useState<SceneData | null>(null),
    [error, setError] = useState(""),
    [time, setTime] = useState(0),
    [playing, setPlaying] = useState(false),
    [speed, setSpeed] = useState(1),
    [mode, setMode] = useState("both"),
    [camera, setCamera] = useState("overview");
  const clock = useRef({ time: 0, playing: false, speed: 1 });
  useEffect(() => {
    let gone = false;
    Promise.all([loadScene("./data/hand/scene.json"), loadRecording("./data/hand")])
      .then(([s, r]) => {
        if (!gone) {
          setScene(s);
          setRun(r);
        }
      })
      .catch((e) => !gone && setError(e.message));
    return () => {
      gone = true;
    };
  }, []);
  useEffect(() => {
    clock.current.playing = playing;
    clock.current.speed = speed;
  }, [playing, speed]);
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
        {error}
        <Button onClick={() => window.location.reload()}>Reload hand recording</Button>
      </section>
    );
  if (!run || !scene)
    return <section className="existing-task">Loading the articulated Allegro hand…</section>;
  const sample = run.samples[
      Math.min(run.samples.length - 1, Math.floor(time * run.result.fps))
    ] as any,
    result = run.result;
  const milestones = run.samples.filter((s: any) => s.goal_reached) as any[];
  return (
    <>
      <section className="motion-workbench hand-workbench">
        <div className="motion-stage">
          <div className="motion-heading">
            <span>
              <Hand size={20} />
              ALLEGRO / 4 fingers · 16 actuated joints
            </span>
            <span>
              PHYSX / {Math.round(1 / result.physics_dt_s)} Hz · REPLAY / {result.fps.toFixed(0)}{" "}
              FPS
            </span>
          </div>
          <div className="motion-phase">
            <p className="eyebrow">LEARNED IN-HAND MANIPULATION</p>
            <h2>Regrasp. Rotate. Reach the next target.</h2>
            <p>
              Actual joint geometry and motion · orange wireframe shows the desired cube
              orientation.
            </p>
          </div>
          <div className={`motion-pair ${mode === "both" ? "dual" : ""}`}>
            {mode !== "video" && (
              <Replay
                sceneData={scene}
                recording={run}
                playback={clock}
                cameraView={camera}
                showPath={false}
                label="Articulated hand · drag to inspect"
              />
            )}
            {mode !== "3d" && (
              <SyncedVideo
                src="./data/hand/video.mp4"
                clock={clock}
                label="Corrected motor-trained policy"
              />
            )}
          </div>
          <div className="motion-toolbar">
            <span>Drag to orbit · scroll for finger detail</span>
            <NativeSelect
              aria-label="Hand camera"
              value={camera}
              onChange={(e) => setCamera(e.target.value)}
            >
              <NativeSelectOption value="overview">Close-up</NativeSelectOption>
              <NativeSelectOption value="top">Overhead</NativeSelectOption>
              <NativeSelectOption value="front">Palm view</NativeSelectOption>
            </NativeSelect>
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
            <Button aria-label={playing ? "Pause" : "Play"} onClick={() => setPlaying(!playing)}>
              {playing ? <Pause /> : <Play />}
            </Button>
            <Button variant="ghost" aria-label="Restart" onClick={() => seek(0)}>
              <RotateCcw />
            </Button>
            <Slider
              aria-label="Hand video and 3D timeline"
              value={[time]}
              min={0}
              max={result.duration}
              step={1 / result.fps}
              onValueChange={(v) => seek(Array.isArray(v) ? v[0] : v)}
            />
            <span>
              {time.toFixed(1)} / {result.duration.toFixed(1)} s
            </span>
            <NativeSelect
              value={speed}
              aria-label="Hand playback speed"
              onChange={(e) => setSpeed(Number(e.target.value))}
            >
              {[0.25, 0.5, 1, 2].map((v) => (
                <NativeSelectOption value={v} key={v}>
                  {v}×
                </NativeSelectOption>
              ))}
            </NativeSelect>
          </div>
        </div>
        <aside className="motion-inspector">
          <p className="eyebrow">DEXTERITY, MEASURED</p>
          <h2>Watch each finger do its work.</h2>
          <p>
            The completed corrected policy operates the real-motor model. Finger articulation comes
            directly from Isaac Lab.
          </p>
          <dl className="motion-values">
            <dt>Completed targets</dt>
            <dd>{sample.goals}</dd>
            <dt>Orientation error</dt>
            <dd>{sample.orientation_error_deg.toFixed(1)}°</dd>
            <dt>Drops / resets</dt>
            <dd>{sample.failures}</dd>
            <dt>Peak joint torque now</dt>
            <dd>{sample.torque.toFixed(2)} Nm</dd>
            <dt>Maximum winding temperature</dt>
            <dd>{sample.temperature.toFixed(1)} °C</dd>
            <dt>Torque shortfall</dt>
            <dd>{(100 * sample.saturation).toFixed(0)}%</dd>
          </dl>
          <p className="micro">
            A target counts when Isaac Lab registers success. The next target then changes. This is
            continuous reorientation with a four-finger Allegro hand; two-hand transfer remains
            unfinished.
          </p>
          <p className="eyebrow">REACHED TARGETS</p>
          <div className="hand-milestones">
            {milestones.length ? (
              milestones.map((s, i) => (
                <Button
                  size="sm"
                  variant="outline"
                  key={s.t}
                  onClick={() => seek(Math.max(0, s.t - 0.5))}
                >
                  <Target size={12} />
                  {i + 1} · {s.t.toFixed(1)}s
                </Button>
              ))
            ) : (
              <span>No completed targets in this recording.</span>
            )}
          </div>
        </aside>
      </section>
      <section className="hand-joints">
        <div>
          <p className="eyebrow">LIVE JOINT DETAIL</p>
          <h2>Sixteen motors. One coordinated grasp.</h2>
          <p>
            Each tile shows measured simulated joint torque and modeled winding temperature at the
            current frame. Bars use the same torque scale.
          </p>
        </div>
        <div className="joint-grid">
          {result.joint_names.map((name: string, i: number) => (
            <div className="joint-tile" key={name}>
              <span>{name.replace(/_/g, " ")}</span>
              <strong>
                {sample.joint_torque[i].toFixed(2)} <small>Nm</small>
              </strong>
              <div className="joint-meter">
                <i
                  style={{
                    width: `${Math.min(100, (Math.abs(sample.joint_torque[i]) / 1.875) * 100)}%`,
                  }}
                />
              </div>
              <small>
                {sample.joint_temperature[i].toFixed(1)} °C ·{" "}
                {((sample.joint_position[i] * 180) / Math.PI).toFixed(0)}°
              </small>
            </div>
          ))}
        </div>
      </section>
      <section className="motion-analysis" id="method">
        <div>
          <p className="eyebrow">COMPLETE RECORDED TRIAL</p>
          <h2>
            {result.goals} orientation targets. {result.drops} drops.
          </h2>
          <p>
            24 seconds, one seed, no cuts. Success markers let you inspect the approach to each
            reached orientation. This is a learned policy, not a scripted finger animation.
          </p>
          <p>
            Rendered surfaces preserve the authored Allegro mesh geometry. Close-up cameras and
            separate finger-pad materials make the mechanism legible; all robot and cube transforms
            are measured simulation output.
          </p>
          <div className="motion-downloads">
            {[
              "video.mp4",
              "poses.bin",
              "telemetry.json",
              "result.json",
              "scene.json",
              "source.zip",
            ].map((n) => (
              <a key={n} href={`./data/hand/${n}`} download>
                {n}
              </a>
            ))}
          </div>
        </div>
        <div>
          <p className="eyebrow">MODELED ACTUATOR</p>
          <dl className="motion-values">
            <dt>Kv · phase-peak convention</dt>
            <dd>{result.Kv.toFixed(1)} rpm/V</dd>
            <dt>Kt</dt>
            <dd>{result.Kt.toFixed(3)} Nm/A</dd>
            <dt>Gearing</dt>
            <dd>{result.gear}:1</dd>
            <dt>Peak phase current</dt>
            <dd>{result.I_peak} A</dd>
            <dt>Shared bus voltage</dt>
            <dd>{result.V_bus} V</dd>
          </dl>
          <p className="micro">
            The policy was trained for 6,000 iterations against the corrected actuator model.
            Temperatures are model estimates. This demonstration does not reproduce the referenced
            Rubik's Cube solver or establish hardware performance.
          </p>
        </div>
      </section>
    </>
  );
}
