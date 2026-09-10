# Bench assembly & hookup guide

Software harness: ../../../../bench/ (flash, calibrate, sweep, overlay).

Zero-solder validation bench: moteus-c1 dev kit (drive) + second mj5208
(brake) + NAU7802/load-cell chain (torque measurement). Every connection is
a plug, screw terminal, or lever nut. See fig_bench_hookup.png.

## Phase 0 — smoke test the drive (15 min, day of unboxing)

1. The dev kit arrives with the moteus-c1 already mounted on the mj5208 and
   set on its desk stand. Do not disassemble.
2. PSU barrel/XT30 → moteus power input (XT30 plug, one way only).
3. CAN-USB adapter → laptop USB; JST PH-3 cable from adapter → moteus CAN.
4. `pip install moteus` on the laptop, then `python -m moteus_gui.tview`.
   Calibrate (`python -m moteus.moteus_tool --target 1 --calibrate`) and
   command a slow spin. If it spins, the whole drive chain works.

## Phase 1 — stall / low-speed torque (validates Kt and the ceiling at ω≈0)

1. Bolt the motor's base (stator) to the base plate — the mj5208 has M3
   holes on the stator face; the dev-kit bracket can screw straight down.
2. Bolt the aluminum arm to the ROTOR face bolt circle (M3 x 2). Measure
   center-of-shaft → contact-point distance precisely (this is `r`).
3. Mount the load cell flat on the base plate by its FIXED end (M4/M5
   through-holes), free end floating over a spacer so the beam can flex.
   Position so the arm's tip presses down on the free end's marked spot.
4. Load cell wires → NAU7802 screw terminals: red→E+, black→E−,
   green→A+ (O+), white→A− (O−). Screwdriver only.
5. NAU7802 → QT Py via STEMMA QT cable (plug). QT Py → laptop USB-C.
6. Calibrate: with the motor unpowered, rest a known mass (kitchen-scale a
   water bottle) on the cell → counts per newton. Torque N·m = force × r.
7. Test: command torque steps via moteus; log commanded vs cell-measured.

## Phase 2 — dyno (torque at speed, traces the full torque-speed curve)

1. Couple the two rotors face-to-face with the printed 3-bolt coupling
   disc (M3 into each rotor's bolt circle). Shafts must be coaxial —
   shim the mounts until the pair spins freely by hand.
2. The BRAKE motor's stator mounts on the pivot bearing (lazy-susan or
   printed bushing) with its own arm pressing the load cell: at steady
   state the stator reaction torque equals the transmitted shaft torque.
3. Brake motor's 3 phase wires → WAGO 221 lever nuts → resistor bank in a
   Y: phase A→R1, B→R2, C→R3, far ends joined in one 5-port WAGO.
   Screw-lug resistors: wires under the lugs, no solder.
4. Sweep: command the drive to velocity setpoints (moteus velocity mode),
   log cell force + moteus telemetry (phase current, bus voltage, temp).
   Swap resistor values for different load lines.
5. CAUTION: resistors get hot (they are rated 50 W — bolt them to a scrap
   of aluminum), keep fingers off the coupling, tape the arm's travel so
   an over-torque can't slap the cell, and start every run at low current
   limits (`moteus_tool` config) before opening them up.

## The plot that settles it

Feed the mj5208's parameters (R, L, Kv from mjbots specs, or measured)
through the toolkit → predicted torque-speed curve. Overlay Phase 1 + 2
measurements. Then flash the emitted firmware (STLINK-V3MINIE on the
moteus SWD pads) and repeat: same law in sim, on the bench, and shipping.

## Printed fixtures

Generate the reaction-dyno printables with:

```bash
.venv-mfg/bin/python bench/bench_cad.py --out bench/printables
```

Parts (STEP + STL in bench/printables/, params at the top of bench_cad.py —
MEASURE your rotor bolt-circle and shaft first):
- **coupling_disc** — bolts the drive rotor to the brake rotor face-to-face
  (two 3-bolt patterns clocked 60 deg so screws never collide).
- **torque_arm** — hub bolts to a rotor (Phase 1) or the brake carrier
  (Phase 2); the arm reaches the calibrated radius `r` and its foot presses
  the load cell.
- **pivot_carrier** + **pivot_base** — the brake motor bolts to the carrier;
  its boss rotates in the fixed base bushing, coaxial with the shaft, so the
  brake stator is free to react (reaction torque = transmitted shaft torque).
  Plain bearing: grease the boss.
