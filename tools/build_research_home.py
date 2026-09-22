"""Build the research landing page from the recorded course manifest.

No simulation or generated motion is used. Existing full videos remain in lab/data.
"""
from html import escape
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
data = json.loads((ROOT / "assets/research/data/candidates.json").read_text())
labels = {"nominal":"Nominal actuator", "kv_high":"Higher-Kv winding", "gear_low":"Lower gearing · 6:1", "mass_added_double":"Added carrier mass · +6 kg", "torque_low":"Lower current limit", "torque_focused":"Combined actuator design"}
order = ["nominal", "kv_high", "gear_low", "mass_added_double", "torque_low", "torque_focused"]
cases = [next(c for c in data["variants"] if c["id"] == name) for name in order]
cards=[]
for i,c in enumerate(cases,1):
    name=c["id"];passed=c["passed"]
    status="Completed" if passed else "Not completed"
    cards.append(f'''<article class="candidate{' active' if name=='kv_high' else ''}">
<button type="button" data-candidate="{name}" aria-pressed="{'true' if name=='kv_high' else 'false'}" aria-label="Inspect {escape(labels[name])}: {status.lower()}">
<div class="candidate-media"><video muted playsinline preload="none" poster="assets/research/posters/{name}.jpg" data-src="lab/data/challenge/{name}/video.mp4" aria-label="Recorded {escape(labels[name])} trial"></video><span class="tile-index">DESIGN {i:02d}</span></div>
<div class="candidate-info"><h3>{escape(labels[name])}</h3><div><span class="status{'' if passed else ' failed'}">{status}</span><span>{c['duration']:.2f} s recorded</span></div></div></button></article>''')
options=''.join(f'<option value="{c["id"]}"{" selected" if c["id"]=="kv_high" else ""}>{escape(labels[c["id"]])}</option>' for c in cases)
max_time=max(c['recording_duration'] for c in cases)
html='''<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Atlas — Actuator Design, Tested in Motion</title>
<meta name="description" content="We built Atlas to connect actuator design with the tasks robots need to perform. Explore recorded design comparisons, motor measurements, thermal experiments, interactive replays and the companion paper.">
<meta property="og:title" content="Atlas — Actuator Design, Tested in Motion">
<meta property="og:description" content="Different windings. Different gearing. Different motion. Explore the experiments, interactive replays and research paper.">
<meta property="og:image" content="https://theenergynerd.github.io/actuator-model-problem/assets/research/posters/kv_high.jpg">
<meta property="og:type" content="website"><meta name="theme-color" content="#f5f4ee">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Crect width='64' height='64' rx='14' fill='%23183b32'/%3E%3Cpath d='M15 49L32 13L49 49M22 35H42' fill='none' stroke='%23f5f4ee' stroke-width='5'/%3E%3C/svg%3E">
<link rel="stylesheet" href="assets/research/site.css">
<script src="assets/research/site.js" defer></script>
</head><body>
<a class="skip" href="#main">Skip to research</a>
<div class="wrap">
<header class="topbar"><a href="./" class="brand" aria-label="Atlas research home"><span class="brand-mark" aria-hidden="true"></span>atlas<span class="nav-tag">Motion Systems / Research</span></a><nav class="topnav" aria-label="Main navigation"><a href="#candidates">The experiment</a><a href="#paper">Paper ↗</a><a href="#findings">Findings</a><a href="#demo-update">Explore ↗</a></nav></header>
<main id="main">
<section class="hero" aria-labelledby="hero-title">
<div><p class="eyebrow">Actuator design, tested in motion</p><h1 id="hero-title">Build the actuator<br>around <em>the task.</em></h1></div>
<div class="hero-copy"><p>We built <strong>Atlas</strong> to connect actuator design with the work a robot needs to do. Change the winding, transmission or mass. Test what changes in the movement. Keep the evidence.</p><div class="actions"><a class="button" href="paper/atlas-paper-draft.pdf">Read the paper <span aria-hidden="true">↗</span></a><a class="button secondary" href="#candidates">Compare designs <span aria-hidden="true">↓</span></a></div></div>
</section>
<div class="caption-line"><span>Atlas Motion Systems · September 2026</span><span>Isaac Lab + PhysX · Paper, code &amp; recorded experiments</span></div>

<section class="section candidate-section" id="candidates" aria-labelledby="candidate-title">
<div class="candidate-top"><h2 id="candidate-title">One robot. Different actuators.</h2><span class="micro">Recorded trials · Same policy · Seed 201</span></div>
<div class="candidate-grid" id="candidate-grid">__CARDS__</div>
<div class="transport" aria-label="Synchronized candidate recordings"><button class="small-button" id="grid-play" type="button" aria-pressed="false">Play six recordings</button><button class="small-button" id="grid-reset" type="button">Restart</button><label for="grid-time" class="visually-hidden">Candidate recording time in seconds</label><input type="range" id="grid-time" min="0" max="__MAXTIME__" step=".02" value="0"><output id="grid-clock" for="grid-time">0.0 s</output></div>
<p id="grid-message" class="grid-message" role="status" aria-live="polite"></p>
<p class="note">Six completed development evaluations on the same 19 m challenge, shown as synchronized recordings. Select a tile to inspect its design. A stopped recording holds its final frame. These runs show design sensitivity; independent design-search and unseen-course evaluations are the next study.</p>
<div class="numbers" aria-label="Completed design sensitivity study"><div class="number"><strong>18</strong><p>actuator variants in the broader design sweep</p></div><div class="number"><strong>32</strong><p>physical simulation replicas per design</p></div><div class="number"><strong>3</strong><p>robot embodiments: ANYmal, G1 and Allegro</p></div></div>
</section>

<section class="section" id="designs" aria-labelledby="design-heading">
<div class="section-head"><div><p class="eyebrow">01 / From design to movement</p><h2 id="design-heading">See what changed.<br>Then watch what happened.</h2></div><p>Actuator design is a set of connected choices. A faster winding changes the electrical model. Gearing trades speed for torque. Added mass changes the robot’s dynamics.</p></div>
<div class="design-layout"><article class="design-description"><span class="design-number" id="design-outcome">Completed · 28.68 s</span><h3 id="design-title">A faster winding. A different outcome.</h3><p id="design-text">The higher-Kv winding completes this recorded challenge with a lower ideal stall torque than the nominal design. Inspect the same run in the video and 3D replay.</p><a class="button" id="design-replay" href="lab/?design=kv_high#terrain">Open this design’s replay <span aria-hidden="true">↗</span></a></article>
<div><div class="selector"><label for="design-select">Compare with nominal</label><select id="design-select">__OPTIONS__</select></div><div class="table-wrap"><table><thead><tr><th>Design parameter</th><th>Nominal</th><th>Selected</th></tr></thead><tbody id="design-rows"><tr><td>Kv · phase-peak rpm/V</td><td>57.3</td><td class="changed">85.9</td></tr><tr><td>Gear ratio</td><td>9.0</td><td>9.0</td></tr><tr><td>Peak current · A</td><td>55.00</td><td>55.00</td></tr><tr><td>Phase resistance · mΩ</td><td>30.0</td><td class="changed">13.3</td></tr><tr><td>Ideal stall torque · N·m</td><td>111.4</td><td class="changed">74.2</td></tr><tr><td>Added carrier mass · kg</td><td>0.0</td><td>0.0</td></tr></tbody></table></div><p class="note" style="margin-top:15px">These are the evaluated model parameters. Stall torque is a model rating; the replay shows delivered torque and recorded joint speed through the task. <a href="assets/research/data/candidates.json">Download the design records ↗</a></p></div></div>
<div class="design-details"><article><h3>Windings</h3><p>Kv, Kt, resistance and inductance change together under the stated rewind model. A single torque number cannot capture the tradeoff.</p></article><article><h3>Gearing</h3><p>Compare 9:1 with 6:1 reduction and inspect the resulting motion. These course runs retain native joint armature.</p></article><article><h3>Mass &amp; placement</h3><p>Recorded mass changes affect carrier-body inertia. Relocating motors and modeling the associated transmission is part of the next design study.</p></article></div>
</section>

<section class="section" id="findings" aria-labelledby="findings-title"><span id="experiment-notes"></span>
<div class="section-head"><div><p class="eyebrow">02 / Results, with their evidence</p><h2 id="findings-title">What the experiments<br>actually found.</h2></div><p>Electrical calibration, repeated cooling tests in simulation, and matched policy training answer different questions. Each result links to its conditions and full record.</p></div>
<div class="result-cards">
<article class="result-card"><span class="scope-tag">Physical motor · controller calibration</span><h3>The measured motor<br>changes the inputs.</h3><div class="result-stat">45 → 65.1 <small style="font-size:17px">mΩ</small></div><p>Nominal versus calibrated phase resistance. At matched current and resistance conditions, that is 44.6% more predicted copper loss.</p><a href="lab/data/pen/bench/evidence.json">Electrical calibration &amp; provenance ↗</a></article>
<article class="result-card"><span class="scope-tag">Isaac Lab · 32 replicas per design</span><h3>Cooling changes<br>sustained performance.</h3><div class="result-stat">40.92 → 20.62</div><p>Reorientation goals per hand per minute when thermal resistance doubles. Peak modeled winding temperature rises from 69.0°C to 107.7°C.</p><a href="#thermal-results">Inspect the cooling experiment ↓</a></article>
<article class="result-card"><span class="scope-tag">Isaac Lab · 3 paired training seeds</span><h3>Model detail matters<br>when the task needs it.</h3><div class="result-stat">95.6% / 95.1%</div><p>Cold pen-test motion success after simplified versus motor-model training. This workload shows no clear motor-aware advantage; stricter contact results are reported separately.</p><a href="lab/#pen">Every seed, video &amp; replay ↗</a></article>
</div>
<div class="thermal-block" id="thermal-results"><figure><img src="assets/research/cooling-results.svg" alt="Recorded Allegro cooling sweep: task throughput and modeled winding temperature for three thermal resistance settings" loading="lazy"></figure><div><p class="eyebrow">Completed cooling experiment</p><h3>Same hand policy.<br>Different thermal response.</h3><p>The 120-second tests compare three cooling configurations within the 18-design sweep. Doubling thermal resistance reduces average throughput by 49.6%. Halving it leaves throughput close to the reference, with more drops.</p><p>These are recorded Isaac Lab outcomes under one fixed policy and paired startup conditions. The chart identifies the simulated temperature signal and reports all three cooling variants.</p><div class="thermal-links"><a href="assets/research/data/cooling-source.json">Full run data ↗</a><a href="assets/research/data/cooling-summary.json">Comparison &amp; conditions ↗</a></div><div class="data-update"><strong>Physical thermal bench results</strong><br>A newer bench test has been completed. Its results will be added here with the test record and measurement conditions.</div></div></div>
</section>

<section class="section paper-section" id="paper" aria-labelledby="paper-title"><a class="paper-cover" href="paper/atlas-paper-draft.pdf" aria-label="Read the Atlas paper draft"><img src="assets/research/paper-cover.jpg" alt="First page of Atlas: Task-Driven Actuator Design with Measured Motor Models" loading="lazy"></a><div class="paper-copy"><div class="paper-label"><span class="eyebrow" style="margin:0">03 / The companion paper</span><span class="pill">Working manuscript</span></div><h2 id="paper-title">Task-Driven Actuator Design<br>with Measured Motor Models.</h2><p>The technical account behind the demonstrations: motor inputs, simulation methods, completed experiments and the evaluation of Atlas’s design choices.</p><ul><li>Measured electrical parameters and their model conventions.</li><li>Cooling-sweep outcomes and matched hand-policy training.</li><li>Design baselines, controller adaptation and unseen-test protocol.</li><li>Figures, source data and reproducible experiment records.</li></ul><div class="actions"><a class="button" href="paper/atlas-paper-draft.pdf">Read PDF <span aria-hidden="true">↗</span></a><a class="button secondary" href="paper/manuscript.html">Read online ↗</a><a class="button secondary" href="paper/experiment-protocol.md">Study protocol ↗</a></div><p class="note" style="margin-top:17px">Completed findings are reported as results. The independent design-search comparison remains a planned experiment.</p></div></section>

<section class="section" id="demo-update" aria-labelledby="demo-title"><div class="section-head"><div><p class="eyebrow">04 / Open the experiment</p><h2 id="demo-title">Watch it. Orbit it.<br>Inspect the actuator.</h2></div><p>Video and the recorded 3D body states share a timeline. Select an experiment, scrub the motion and inspect the corresponding design and joint signals.</p></div>
<nav class="experiment-nav" aria-label="Choose an interactive experiment"><button type="button" data-demo="terrain" aria-pressed="true">ANYmal course</button><button type="button" data-demo="pen" aria-pressed="false">Sharpa hand</button><button type="button" data-demo="g1" aria-pressed="false">G1</button><button type="button" data-demo="anymal" aria-pressed="false">ANYmal walking</button><button type="button" data-demo="transfer" aria-pressed="false">Two arms</button><button type="button" data-demo="allegro" aria-pressed="false">Allegro</button><button type="button" data-demo="rubik" aria-pressed="false">Wuji cube</button><button type="button" data-demo="assembly" aria-pressed="false">Assembly</button></nav>
<div class="viewer-box"><div class="viewer-placeholder" id="viewer-placeholder"><span class="micro">VIDEO + INTERACTIVE 3D REPLAY</span><h3 id="viewer-title">ANYmal obstacle course</h3><p id="viewer-description">A 19 m course with stairs, a gap, narrow crossing, rubble, slope and a physical push. Select a recorded design and inspect the matching motion.</p><button class="button" id="load-viewer" type="button">Load interactive replay <span aria-hidden="true">↗</span></button></div></div>
<div class="viewer-bottom"><p id="demo-description" aria-live="polite">The course, checkpoint and development seed are fixed across the Atlas design variants. All recorded successes and failures remain available.</p><a href="lab/#terrain" id="demo-full" target="_blank" rel="noopener">Open in a full window ↗</a></div>
</section>

<section class="section" id="next-study" aria-labelledby="next-title"><div class="section-head"><div><p class="eyebrow">05 / The design question</p><h2 id="next-title">From candidate designs<br>to an unseen challenge.</h2></div><p>Atlas is the system our team built for actuator design. The next study tests whether its selected hardware performs better under the same resource limits.</p></div>
<div class="roadmap"><article><div class="micro"><span class="dot"></span> Recorded comparisons</div><h3>A grid of candidates.</h3><p>Different actuator designs face the same task. The grid above opens the actual evaluated runs and their parameter records.</p></article><article><div class="micro"><span class="dot"></span> Existing + extended study</div><h3>Design changes. Motion changes.</h3><p>Connect winding, gearing and mass changes to the movement. Extend the mechanical model to test motor placement and transmission inertia.</p></article><article><div class="micro"><span class="dot pending"></span> Next independent evaluation</div><h3>A payoff that survives a new test.</h3><p>Freeze the selected design, compare against the original under matched limits, then evaluate on tasks excluded from the search. Report every outcome.</p></article></div>
</section>

<section class="section" id="resources" aria-labelledby="resources-title"><p class="eyebrow">06 / Continue into the work</p><h2 id="resources-title">The paper is the beginning<br>of the evidence.</h2><div class="resource-grid"><a href="https://github.com/TheEnergyNerd/actuator-model-problem">Source repository ↗<span>Actuator models, experiments and public data.</span></a><a href="paper/experiment-protocol.md">Evaluation protocol ↗<span>Baselines, budgets, task splits and reporting.</span></a><a href="lab/data/pen/bench/study.json">Matched training results ↗<span>All training seeds and independent test grasps.</span></a><a href="report.html">Detailed research report ↗<span>The longer article, earlier experiments and references.</span></a></div></section>
</main><footer><span>Atlas Motion Systems · Research, September 2026</span><span>Built on Isaac Lab &amp; PhysX · <a href="paper/manuscript.html#references">Sources &amp; acknowledgments</a></span></footer>
<noscript><p class="noscript-note">The paper and results are available without JavaScript. Open the <a href="lab/#terrain">interactive lab</a> with JavaScript enabled to play synchronized videos and 3D replays.</p></noscript>
</div></body></html>
'''
html=html.replace('__CARDS__','\n'.join(cards)).replace('__OPTIONS__',options).replace('__MAXTIME__',f'{max_time:.2f}')
(ROOT/'index.html').write_text(html)
print('Built index.html with',len(cards),'recorded candidates.')
