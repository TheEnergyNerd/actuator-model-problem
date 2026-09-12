import { useState } from "react";
type Group = Record<string, number | string>;
const rows: [string, string, string, number][] = [
  ["Kv_phase_peak_rpm_per_V", "Motor Kv", "rpm/V", 1],
  ["Kt_Nm_per_peak_q_A", "Torque per amp · Kt", "Nm/A", 3],
  ["gear_ratio", "Reduction ratio", ":1", 1],
  ["peak_current_A", "Peak current limit", "A", 1],
  ["nominal_stall_peak_joint_Nm", "Peak stall joint torque", "Nm", 2],
  ["joint_rpm", "Ideal unloaded joint speed", "RPM", 1],
  ["motor_rpm", "Ideal unloaded motor speed", "RPM", 0],
  ["V_bus_V", "Bus voltage", "V", 1],
  ["phase_R_ohm", "Winding resistance", "Ω", 4],
  ["added_mass_per_actuator_kg", "Added mass per actuator", "kg", 3],
  ["added_joint_armature_kg_m2", "Added reflected rotor inertia", "kg·m²", 5],
  ["thermal_R_K_per_W", "Thermal resistance", "K/W", 2],
  ["thermal_C_J_per_K", "Thermal capacity", "J/K", 1],
  ["initial_winding_C", "Starting winding temperature", "°C", 1],
];
function values(g: Group) {
  const speed =
    g.base_speed_joint_rad_s_ideal_no_field_weakening != null
      ? Number(g.base_speed_joint_rad_s_ideal_no_field_weakening)
      : (((Number(g.Kv_phase_peak_rpm_per_V) * Number(g.V_bus_V)) /
          Math.sqrt(3) /
          Number(g.gear_ratio)) *
          2 *
          Math.PI) /
        60;
  return {
    ...g,
    joint_rpm: Number.isFinite(speed) ? (speed * 60) / (2 * Math.PI) : undefined,
    motor_rpm: Number.isFinite(speed)
      ? (speed * Number(g.gear_ratio) * 60) / (2 * Math.PI)
      : undefined,
  };
}
export default function ActuatorDesign({
  groups = {},
  baseline = {},
  addedMass,
  baselineMass = 0,
  notice,
  builder = false,
}: {
  groups?: Record<string, Group>;
  baseline?: Record<string, Group>;
  addedMass?: number;
  baselineMass?: number;
  notice?: string;
  builder?: boolean;
}) {
  const [selected, setSelected] = useState("");
  const keys = Object.keys(groups);
  const key = keys.includes(selected) ? selected : keys[0];
  const g = key ? groups[key] : undefined;
  const b = key ? baseline[key] : undefined;
  const now = g ? values(g) : null,
    ref = b ? values(b) : null;
  const changes: string[] = [];
  if (g && b) {
    const ratio = (k: string) => Number(g[k]) / Number(b[k]);
    if (Math.abs(ratio("Kv_phase_peak_rpm_per_V") - 1) > 0.005)
      changes.push(
        `Kv is ${ratio("Kv_phase_peak_rpm_per_V").toFixed(2)}× nominal. This rewind changes torque per amp inversely and changes winding resistance and inductance; more RPM per volt does not mean more joint torque.`,
      );
    if (Math.abs(ratio("gear_ratio") - 1) > 0.005)
      changes.push(
        `Gearing is ${ratio("gear_ratio").toFixed(2)}× nominal: reduction trades joint speed for torque. Reflected rotor inertia scales with the square of the ratio when rotor inertia is included.`,
      );
    if (Math.abs(ratio("peak_current_A") - 1) > 0.005)
      changes.push(
        `Peak current is ${ratio("peak_current_A").toFixed(2)}× nominal. This changes the current-limited torque ceiling; sustained capability also depends on voltage and temperature.`,
      );
    if (
      Number(g.initial_winding_C) !== Number(b.initial_winding_C) &&
      Number.isFinite(Number(g.initial_winding_C))
    )
      changes.push(
        "Starting winding temperature changed. This is an initial-condition experiment, not a different manufactured actuator.",
      );
    if (Math.abs(ratio("V_bus_V") - 1) > 0.005)
      changes.push("Bus voltage changed, shifting voltage-limited speed and torque availability.");
    if (Math.abs(ratio("thermal_R_K_per_W") - 1) > 0.005)
      changes.push(
        "Thermal resistance changed: lower resistance removes heat more readily for the same temperature difference.",
      );
  }
  if (addedMass != null && addedMass > baselineMass + 0.001)
    changes.push(
      `${(addedMass - baselineMass).toFixed(2)} kg more physical carrier mass than nominal. Its placement changes the load the joints must accelerate and support.`,
    );
  return (
    <section className="actuator-design" aria-label="Actuator design and motion effects">
      <div className="actuator-design-heading">
        <div>
          <p className="eyebrow">What changes the motion</p>
          <h2>Inside the selected actuator design</h2>
        </div>
        {keys.length > 1 && (
          <label>
            Joint group{" "}
            <select value={key} onChange={(e) => setSelected(e.target.value)}>
              {keys.map((k) => (
                <option key={k}>{k}</option>
              ))}
            </select>
          </label>
        )}
      </div>
      {notice && <p>{notice}</p>}
      {now ? (
        <>
          <div className="actuator-design-grid">
            <div>
              <h3>Design → expected mechanical effect</h3>
              {changes.length ? (
                changes.map((s, i) => <p key={i}>{s}</p>)
              ) : (
                <p>
                  {b
                    ? "Nominal parameters for this actuator group. Select another recorded design to see the differences."
                    : "This recording supplies one actuator configuration; a matched nominal comparison is not available."}
                </p>
              )}
              <p>
                These explain model changes. The replay and trial outcome show what actually
                happened; a single run does not establish a general causal effect.
              </p>
              {addedMass != null && (
                <p>
                  <strong>Total added actuator mass: {addedMass.toFixed(2)} kg</strong>
                  {baselineMass !== addedMass ? ` · nominal ${baselineMass.toFixed(2)} kg` : ""}
                </p>
              )}
              <p className="micro">
                RPM below is calculated from the model’s ideal unloaded speed, not measured motor
                RPM. Actual motor RPM was not exported in these recordings. Kv uses phase-peak
                back-EMF and Kt uses peak q-axis current.
              </p>
            </div>
            <div className="actuator-table">
              <table>
                <thead>
                  <tr>
                    <th>Parameter</th>
                    {ref && <th>Nominal</th>}
                    <th>Selected</th>
                    {ref && <th>Change</th>}
                  </tr>
                </thead>
                <tbody>
                  {rows.map(([field, label, unit, digits]) => {
                    const n = Number(now[field as keyof typeof now]),
                      v = ref ? Number(ref[field as keyof typeof ref]) : NaN;
                    if (!Number.isFinite(n)) return null;
                    const delta = Number.isFinite(v) ? n - v : NaN;
                    return (
                      <tr key={field}>
                        <td>{label}</td>
                        {ref && <td>{Number.isFinite(v) ? v.toFixed(digits) : "—"}</td>}
                        <td>
                          {n.toFixed(digits)} {unit}
                        </td>
                        {ref && (
                          <td className={Math.abs(delta) > 1e-7 ? "design-changed" : ""}>
                            {!Number.isFinite(delta)
                              ? "—"
                              : Math.abs(delta) < 1e-7
                                ? "Same"
                                : `${delta > 0 ? "+" : ""}${delta.toFixed(digits)}`}
                          </td>
                        )}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </>
      ) : (
        <p>
          Detailed electrical design parameters and matched actuator variants are not available for
          this recording. No Kv or RPM values are inferred from its appearance.
        </p>
      )}
      {builder && baseline.legs && <DesignBuilder base={baseline.legs} />}
    </section>
  );
}
function DesignBuilder({ base }: { base: Group }) {
  const [kv, setKv] = useState(1),
    [gear, setGear] = useState(1),
    [current, setCurrent] = useState(1),
    [mass, setMass] = useState(0);
  const candidate = {
    schema: "atlas-course-design-v1",
    design: {
      name: "custom_course",
      axis: "custom winding, gearing, current and carrier mass",
      kv_factor: kv,
      gear_factor: gear,
      peak_current_factor: current,
      added_mass_factor: mass / 0.25,
    },
  };
  const download = () => {
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(candidate, null, 2) + "\n"], { type: "application/json" }),
    );
    const a = document.createElement("a");
    a.href = url;
    a.download = "atlas-course-design.json";
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  const presets = [
    { name: "Torque-focused", kv: 0.8, gear: 1.2, current: 1, mass: 0.1 },
    { name: "Speed-focused", kv: 1.25, gear: 0.85, current: 1, mass: 0 },
    { name: "Higher current + added carrier mass", kv: 1, gear: 1, current: 1.2, mass: 0.25 },
  ];
  return (
    <details className="design-builder">
      <summary>Create another actuator design · not yet simulated</summary>
      <p>
        These controls create a new candidate relative to the nominal Atlas course motor. The video
        remains the selected recorded run. New task performance requires an Isaac Lab evaluation.
      </p>
      <div className="design-presets">
        {presets.map((p) => (
          <button
            key={p.name}
            onClick={() => {
              setKv(p.kv);
              setGear(p.gear);
              setCurrent(p.current);
              setMass(p.mass);
            }}
          >
            {p.name}
          </button>
        ))}
      </div>
      <div className="design-inputs">
        {[
          { label: "Kv factor", value: kv, set: setKv, min: 0.5, max: 1.5, step: 0.05 },
          { label: "Gear ratio factor", value: gear, set: setGear, min: 0.5, max: 1.5, step: 0.05 },
          {
            label: "Peak current factor",
            value: current,
            set: setCurrent,
            min: 0.5,
            max: 1.5,
            step: 0.05,
          },
          {
            label: "Added mass / actuator · kg",
            value: mass,
            set: setMass,
            min: 0,
            max: 0.5,
            step: 0.025,
          },
        ].map((x) => (
          <label key={x.label}>
            {x.label}
            <input
              type="range"
              min={x.min}
              max={x.max}
              step={x.step}
              value={x.value}
              onChange={(e) => x.set(Number(e.target.value))}
            />
            <output>{x.value.toFixed(3)}</output>
          </label>
        ))}
      </div>
      <p>
        <strong>{(Number(base.Kv_phase_peak_rpm_per_V) * kv).toFixed(1)} rpm/V</strong> · Kt{" "}
        {(Number(base.Kt_Nm_per_peak_q_A) / kv).toFixed(3)} Nm/A ·{" "}
        {(Number(base.gear_ratio) * gear).toFixed(2)}:1 ·{" "}
        {(Number(base.peak_current_A) * current).toFixed(1)} A · {(12 * mass).toFixed(2)} kg added
        across 12 actuators
      </p>
      <p>
        Winding resistance becomes {(Number(base.phase_R_ohm) / kv ** 2).toFixed(4)} Ω; inductance
        scales by {(1 / kv ** 2).toFixed(3)}×. Peak stall joint torque:{" "}
        {((Number(base.nominal_stall_peak_joint_Nm) * current * gear) / kv).toFixed(1)} Nm. Ideal
        unloaded joint speed:{" "}
        {(
          (((Number(base.base_speed_joint_rad_s_ideal_no_field_weakening) * 60) / (2 * Math.PI)) *
            kv) /
          gear
        ).toFixed(1)}{" "}
        RPM.
      </p>
      <p>
        Mass is a separate physical intervention at the existing carrier COM with
        spherical-equivalent inertia; it is not automatically a manufacturable housing. Native joint
        armature stays fixed; this course intervention adds no rotor inertia. The exporter does not
        establish packaging, cooling or structural feasibility.
      </p>
      <button onClick={download}>Download candidate for Isaac Lab</button>
      <p className="micro">
        Run with the course evaluator’s <code>--atlas --design-file atlas-course-design.json</code>{" "}
        options. This importer targets the ANYmal course; hardware-specific designs for LeKiwi
        require its motor and mounting measurements.
      </p>
    </details>
  );
}
