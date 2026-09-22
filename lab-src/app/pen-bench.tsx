import { useEffect, useState } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from "recharts";
import PenTraining from "./pen-training";
const base = "./data/pen/bench";
const labels: Record<string, string> = {
  ideal: "Fixed torque",
  motor: "Cold motor",
  "motor-hot": "100°C start",
};
export default function PenBench() {
  const [data, setData] = useState<any>(null),
    [error, setError] = useState("");
  useEffect(() => {
    let active = true;
    fetch(`${base}/study.json`)
      .then((r) => {
        if (!r.ok) throw Error("Study results unavailable");
        return r.json();
      })
      .then((d) => active && setData(d))
      .catch((e) => active && setError(e.message));
    return () => {
      active = false;
    };
  }, []);
  if (error) return <p role="alert">{error}</p>;
  if (!data) return <p role="status">Loading the bench-informed study…</p>;
  return (
    <>
      <section className="motion-analysis">
        <div>
          <p className="eyebrow">ELECTRICAL CALIBRATION → VIRTUAL DESIGN → POLICY TEST</p>
          <h2>What changes when the motor model starts with measurements?</h2>
          <p>
            Three paired training seeds. 4.19 million transitions per run. Fresh test grasps. Both
            training conditions start cold; hot starts are a separate stress test.
          </p>
          <p>
            This is a virtual remote-drive hand using the mj5208 electrical calibration. It is not
            the measured actuator inside a Sharpa hand.
          </p>
        </div>
        <div>
          {data.summary.map((s: string) => (
            <p key={s}>{s}</p>
          ))}
          <a href={`${base}/study.json`}>Complete results and training logs</a> ·{" "}
          <a href={`${base}/protocol.md`}>Protocol fixed before training</a>
        </div>
      </section>
      <PenTraining base={`${base}/replay`} bench />
      <section className="motion-analysis">
        <div>
          <h2>Measured electrical inputs</h2>
          <dl className="motion-values">
            <dt>Phase resistance</dt>
            <dd>65.08 mΩ</dd>
            <dt>Kv · peak line-to-line</dt>
            <dd>310.68 RPM/V</dd>
            <dt>d / q inductance</dt>
            <dd>27.01 / 45.24 μH</dd>
            <dt>Pole pairs</dt>
            <dd>7</dd>
            <dt>Derived Kt · peak q current</dt>
            <dd>0.02662 Nm/A</dd>
          </dl>
          <p>
            Controller calibration checked against the raw log. Kv conversion checked against the
            recorded moteus tool and firmware. Kt is derived, not load-cell measured.
          </p>
          <a href={`${base}/evidence.json`}>Calibration provenance and source hashes</a>
        </div>
        <div>
          <h2>Drive settings and thermal tests</h2>
          <p>
            12 V supply, 2 A peak current, 1 A continuous current, 80% transmission efficiency.
            Gearing matches each native joint’s nominal cold torque ceiling. Native link mass is
            retained, assuming remote motors; rotor inertia, transmission compliance and packaging
            are not validated.
          </p>
          <p>
            This pen experiment uses thermal settings of 10 K/W and 8 J/K. Displayed temperature
            and copper heat are simulation outputs. The separate completed Allegro cooling sweep
            tests the effect on sustained task performance; newer physical bench results will be
            added with their test record. The 100°C pen test starts hot.
          </p>
          <p><a href="../#thermal-results">Cooling experiment and thermal results</a></p>
          <p>
            Neither training condition gets hidden randomized starting temperature. Both retain the
            same policy observations, reward, initialization and training budget.
          </p>
        </div>
      </section>
      <section className="precision-results">
        <h2>Every training seed, under the same test plant</h2>
        <p>
          Each cell is motion successes out of 128 fresh test grasps. Compare paired training seeds;
          these are three training repetitions, not 384 independent policies.
        </p>
        <div style={{ overflowX: "auto" }}>
          <table>
            <thead>
              <tr>
                <th>Training seed</th>
                <th>Test plant</th>
                <th>Simplified training</th>
                <th>Motor training</th>
                <th>Difference</th>
                <th>Outcomes</th>
              </tr>
            </thead>
            <tbody>
              {data.pairs.map((r: any) => (
                <tr key={`${r.seed}-${r.model}`}>
                  <td>{r.seed}</td>
                  <td>{labels[r.model]}</td>
                  <td>{r.ideal}/128</td>
                  <td>{r.motor}/128</td>
                  <td>
                    {r.motor - r.ideal > 0 ? "+" : ""}
                    {r.motor - r.ideal}
                  </td>
                  <td>
                    <a href={`${base}/outcomes/${r.seed}-ideal-${r.model}.json`}>Simplified</a> ·{" "}
                    <a href={`${base}/outcomes/${r.seed}-motor-${r.model}.json`}>Motor</a>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p>
          Original frozen policy:{" "}
          {data.frozen.map((r: any) => `${labels[r.model]} ${r.motion_successes}/128`).join(" · ")}.
          Additional contact-gate outcomes, all failures and paired differences are included in the
          full results.
        </p>
      </section>
      <section className="motion-analysis">
        <div>
          <h2>Validation success through training</h2>
          <ResponsiveContainer width="100%" height={270}>
            <LineChart data={data.validation_curve}>
              <XAxis dataKey="iteration" />
              <YAxis domain={[0, 100]} unit="%" />
              <Tooltip />
              <Legend />
              <Line name="Simplified · mean of 3 seeds" dataKey="ideal" stroke="#b58b45" />
              <Line name="Motor · mean of 3 seeds" dataKey="motor" stroke="#177960" />
            </LineChart>
          </ResponsiveContainer>
          <p>
            Separate 32-grasp validation set, cold motor plant. Checkpoints at 64, 128 and 256
            updates. Final checkpoint fixed in advance; these curves are diagnostics, not a test-set
            selection rule.
          </p>
        </div>
        <div>
          <h2>Do the constraints actually bind?</h2>
          {data.constraint_summary.map((s: string) => (
            <p key={s}>{s}</p>
          ))}
          <p>
            The fast model was checked against a 24 kHz substepped electrical solver over 0–3,400
            motor RPM; this checks software consistency, not measured motor dynamics. A larger model
            does not guarantee a larger effect. If requested torque is feasible, both drives can
            produce almost the same motion. Calibration improves the assumptions we can defend; it
            does not predetermine which policy wins.
          </p>
        </div>
      </section>
      <p>
        <a href={`${base}/dq-audit.json`}>Electrical solver comparison traces</a>
      </p>
      <figure style={{ margin: "24px 0" }}>
        <img
          src={`${base}/electrical-model.svg`}
          alt="Predicted cold and hot shaft torque versus speed, and comparison of the fast actuator approximation with substepped electrical dynamics"
          style={{ width: "100%", height: "auto" }}
        />
        <figcaption>
          12 V, 2 A peak drive scenario. Torque and derating curves are model predictions; the
          measured inputs are electrical calibration values.
        </figcaption>
      </figure>
    </>
  );
}
