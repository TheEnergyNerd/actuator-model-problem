"""Build both outputs from src/report-src.html:
  --site      → ../index.html + ../assets/ (relative paths, cropped figs)
  --artifact  → /tmp scratch single-file with base64 assets (path printed)
Asset sources live in ~/Downloads/atlas_ab_experiment.
"""
import base64, os, pathlib, re, shutil, sys
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
SITE = HERE.parent
AB = pathlib.Path.home() / "Downloads/atlas_ab_experiment"
MAP = {
    "FIG0": ("figs/fig0_anatomy.png", "fig-anatomy.png"),
    "FIG1": ("figs/fig1_torque_speed.png", "fig-torque-speed.png"),
    "FIG2": ("figs/fig2_field_weakening.png", "fig-field-weakening.png"),
    "FIG3": ("figs/fig3_training_curves.png", "fig-training-curves.png"),
    "FIG4": ("figs/fig4_eval.png", "fig-eval.png"),
    "FIG5": ("figs/fig5_compounding.png", "fig-compounding.png"),
    "FIG6": ("figs/fig6_push_two_gen.png", "fig-push-two-gen.png"),
    "FIG7": ("figs/fig7_sweep.png", "fig-gear-sweep.png"),
    "FIG_SPRINT": ("figs/fig_sprint_pareto.png", "fig-sprint-pareto.png"),
    "FIG_THERMAL": ("figs/fig_thermal.png", "fig-thermal.png"),
    "FIG_CAD_PAIR": ("figs/fig_cad_pair.png", "fig-cad-pair.png"),
    "FIG9": ("figs/fig9_g1_curves.png", "fig-g1-curves.png"),
    "FIG10": ("figs/fig10_g1_falls.png", "fig-g1-falls.png"),
    "FIG11": ("figs/fig_standing_trap.png", "fig-standing-trap.png"),
    "FIG12": ("figs/fig_hard_curves.png", "fig-hard-curves.png"),
    "FIG13": ("figs/fig_codesign_loop.png", "fig-codesign-loop.png"),
    "VID_A_IDEAL": ("v3/A_ideal_fwd_web.mp4", "vid-a-ideal.mp4"),
    "VID_A_FOC": ("v3/A_foc_fwd_web.mp4", "vid-a-foc.mp4"),
    "VID_B_FOC": ("v3/B_foc_fwd_web.mp4", "vid-b-foc.mp4"),
    "VID_PUSH": ("v3/AB_shove_hard35_pg.mp4", "vid-quad-shove.mp4"),
    "VID_QUAD_FALLS": ("quadfalls3/film/quadfalls3_AB_pg.mp4", "vid-quad-falls.mp4"),
    "VID_SWEEP": ("sweep_montage_web.mp4", "vid-gear-sweep.mp4"),
    "VID_G1_REF": ("v3/G1_ref_pg.mp4", "vid-g1-ref.mp4"),
    "VID_G1_COLLAPSE": ("v3/G1_collapse_pg.mp4", "vid-g1-collapse.mp4"),
    "VID_G1_WALK": ("v3/G1_walk_pg.mp4", "vid-g1-walk.mp4"),
    "VID_G1_SHOVE": ("v3/G1_shove_sync_pg.mp4", "vid-g1-shove.mp4"),
    "VID_GAINS_AB": ("v3/G1_gains_ab_pg.mp4", "vid-g1-gains-ab.mp4"),
    "VID_GAIT_AB": ("v3/G1_gait_ab_pg.mp4", "vid-g1-gait-ab.mp4"),
}
SITE_HEAD = """<!doctype html>
<html lang="en">
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="Every RL policy trains against an actuator model, and the industry default is fiction. A/B experiments on a quadruped and a humanoid, from Atlas Motion Systems.">
<meta property="og:title" content="The Actuator Model Problem: Why Sim-to-Real Fails at the Motor">
<meta property="og:description" content="A humanoid trained on stock simulation actuators falls in 4,554 of 4,554 episodes on real motor physics. Trained on a design-derived model, it walks with zero falls.">
<meta property="og:image" content="https://pranavatlas.github.io/actuator-model-problem/assets/fig-anatomy.png">
<meta property="og:type" content="article">
<link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>⚙️</text></svg>">
"""

def tokens_ok(t):
    left = re.findall(r"\{\{[A-Z0-9_]+\}\}", t)
    assert not left, left

def build_site():
    t = (HERE / "report-src.html").read_text()
    (SITE / "assets").mkdir(exist_ok=True)
    for tok, (src, dst) in MAP.items():
        s, d = AB / src, SITE / "assets" / dst
        if dst.endswith(".png"):
            im = Image.open(s); w, h = im.size
            im.crop((0, 0, w, int(h * 0.966))).save(d)
        else:
            shutil.copyfile(s, d)
        assert t.count("{{" + tok + "}}") == 1, tok
        t = t.replace("{{" + tok + "}}", f"assets/{dst}")
    t = t.replace("<!--SITE_ONLY-->", "").replace("<!--/SITE_ONLY-->", "")
    tokens_ok(t)
    t = SITE_HEAD + t
    t = t.replace("</style>", "</style>\n</head>\n<body>", 1) + "\n</body>\n</html>\n"
    (SITE / "index.html").write_text(t)
    viewer = HERE.parent / "motor-explorer.html"
    if viewer.exists() and viewer.resolve() != (SITE / "motor-explorer.html").resolve():
        shutil.copyfile(viewer, SITE / "motor-explorer.html")
    print("site built")

def build_artifact():
    t = (HERE / "report-src.html").read_text()
    t = re.sub(r"<!--SITE_ONLY-->.*?<!--/SITE_ONLY-->", "", t, flags=re.S)
    for tok, (src, _) in MAP.items():
        data = (AB / src).read_bytes()
        mime = "image/png" if src.endswith(".png") else "video/mp4"
        uri = f"data:{mime};base64,{base64.b64encode(data).decode()}"
        assert t.count("{{" + tok + "}}") == 1, tok
        t = t.replace("{{" + tok + "}}", uri)
    tokens_ok(t)
    out = pathlib.Path("/tmp/atlas-evidence-artifact.html")
    out.write_text(t)
    print(f"artifact built: {out} ({out.stat().st_size/1e6:.2f} MB)")

if __name__ == "__main__":
    if "--artifact" in sys.argv: build_artifact()
    else: build_site()
