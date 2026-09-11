# Interactive Isaac Lab replays

The existing research report remains at `index.html`. The new static application is in `lab/`, linked from the report. It is compatible with GitHub Pages project subpaths, with no server or private service required.

The expanded scope includes ANYmal obstacle courses, a two-hand tool workshop, precision assembly, and actuator optimization across all three. Implementation stages and measured readiness checks live in [isaac/suite/README.md](isaac/suite/README.md). These new tracks are not yet published demonstrations.

- `lab/#g1`: corrected G1 walking policy, video and orbitable replay.
- `lab/#anymal`: ANYmal walking, video and orbitable replay.
- `lab/#transfer`: eight existing two-Franka actuator variants.
- `lab/#allegro`: detailed four-finger Allegro reorientation, with synchronized video/3D and all 16 joint torques, temperatures, and angles. This is not a completed dexterous two-hand transfer.

`lab-src/` contains the React/Three.js source. Rebuild with Node 22.13+ using `npm ci` and `npm run build` inside that directory. The build writes `lab/`; commit both source and generated output for the existing branch-based GitHub Pages setup.

The shared locomotion timeline controls both views. Video was recorded directly from Isaac Lab at every third control step; rigid-body transforms and telemetry were sampled every control step. Both start after the same first physics step. Videos were cropped from the paired recordings, without retiming. 3D interpolates measured transforms; it does not rerun or approximate the physics. Geometry is rounded to 10 micrometres for delivery; recorded poses are unchanged. Presentation materials and lighting differ from the Isaac Lab video.

Motor variants select completed experiments, not arbitrary live parameter values. Parameters, checkpoint hashes, failure counts, binary poses, and telemetry are downloadable from each robot view. The same trained policy is held fixed across each robot's designs. Startup properties are paired; normal episode resets remain and are counted. These single-seed, 16-second runs demonstrate conditional design sensitivity, not hardware validation or statistical robustness. Temperature is modeled. Absolute electrical efficiency is not claimed.

G1 uses `Isaac-Velocity-Flat-G1-AtlasFOC-Walking-v3`, checkpoint SHA256 `16cd9f934f5129735d1be9c07386bc0f96ed0c07d93173c5af99cc898ab7783c`. It has residual heading drift. ANYmal uses the established flat-ground policy. The 0.2-second lateral pulse begins at 9 seconds.

The implementation is prepared locally. Publishing to TheEnergyNerd requires GitHub write access, which was not configured in the workspace during preparation. No changes have been pushed to the public repository.

Validation: 35 Python tests passed, including decoding every new video to verify frame counts and replay time alignment, quaternion normalization, artifact hashes, geometry indices, and failure-count consistency. TypeScript checks and the static production build passed. All eight variants' video, pose, result, and telemetry URLs were checked under the `/lab/` subpath. Browser interaction testing was not performed.

Observed results from these recordings:

| Robot / design | Mean body-forward speed | Failure/reset events |
| --- | ---: | ---: |
| G1 nominal | 0.638 m/s | 0 |
| G1 +18.5 kg actuator mass | 0.555 m/s | 0 |
| G1 Kv ×1.5 | 0.625 m/s | 0 |
| ANYmal nominal | 0.997 m/s | 0 |
| ANYmal gearing ×2/3 | 0.995 m/s | 0 |
| ANYmal peak current ×0.67 | 0.998 m/s | 0 |
| ANYmal peak current ×0.15 (stress case) | 0.474 m/s | 35 |
| ANYmal Kv ×4 (stress case) | 0.470 m/s | 11 |

The mild ANYmal changes remain within the task's operating limits; their near-identical results are retained. The deliberately extreme stress cases are labeled separately. Failure counts include repeated episode resets and must not be interpreted as independent trial counts. Replay interpolation and body trails break at resets.

## Detailed dexterous hand update

The Allegro tab now uses the completed corrected-v1 REAL seed-42 policy (6,000 training iterations), checkpoint SHA256 `295e2fdb2868069afd690e9b785815ef7256586e021d06e60107e32b0885282f`. It reaches 13 randomized orientation goals in one continuous 24-second recording, with zero drops. The hand has 16 actuated joints, 21 recorded hand bodies, and 39,128 authored mesh triangles; the cube is a measured 60 mm rigid object.

The video is 1440 × 1440 at 30 FPS, with brighter studio illumination and a closer camera. Physics runs at 120 Hz. The viewer adds synchronized 3D, slow motion, palm/overhead/close-up cameras, a clearly labeled wireframe orientation target, success jump points, and each joint's torque, winding temperature, and angle. The 3D materials are presentation finishes; they are not a recreation of the video cube's texture atlas.

A separate initial recording and the final lighting/camera refinement produced exactly identical pose arrays (maximum absolute component difference 0). Neither rendering refinement changes physics. Timeouts and the maximum-success termination are disabled for the continuous recording; drop resets remain enabled and counted. The task is learned in-hand reorientation, not a Rubik solver or two-hand transfer. The Franka transfer tab still uses two-finger grippers.

Four focused delivery tests passed, including full video decoding, frame-time alignment, pose quaternion normalization, actual movement in at least 12 joints, success-event count consistency, and regression checks of the locomotion recordings. TypeScript and the production build also passed. No browser interaction test was performed. New artifacts and Isaac Lab source are downloadable from the hand view.
