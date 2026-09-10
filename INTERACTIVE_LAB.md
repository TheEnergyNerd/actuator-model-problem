# Interactive Isaac Lab replays

The existing research report remains at `index.html`. The new static application is in `lab/`, linked from the report. It is compatible with GitHub Pages project subpaths, with no server or private service required.

- `lab/#g1`: corrected G1 walking policy, video and orbitable replay.
- `lab/#anymal`: ANYmal walking, video and orbitable replay.
- `lab/#transfer`: eight existing two-Franka actuator variants.
- `lab/#allegro`: existing Allegro diagnostic video, not a completed dexterous transfer.

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
