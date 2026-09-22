"""Publish the completed cooling sweep, keeping physical-bench results separate."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'assets/research'
source=OUT/'data/cooling-source.json'
d=json.loads(source.read_text())
uncertainty=json.loads((OUT/'data/design-uncertainty.json').read_text())
ids=['cooling_better','nominal','cooling_worse']
records=[next(r for r in d['records'] if r['design']['name']==name) for name in ids]
assert d['replicas_per_design']==32 and d['seconds']==120
assert all(len(r['goals_per_replica'])==32 for r in records)
for r in records:
    assert np.isclose(np.mean(r['goals_per_replica'])/(d['seconds']/60),r['goals_per_hand_minute'])
summary=dict(
    experiment='Allegro cooling-resistance sweep in Isaac Lab',
    date='2026-09-10', evidence_type='Recorded simulation experiment',
    replicas_per_design=32,seconds_per_replica=120,seed=d['seed'],
    policy_checkpoint_sha256=d['checkpoint_sha256'],
    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    protocol=d['protocol'],
    physical_bench_status='A newer physical thermal test has been reported by Atlas; its source record has not yet been supplied for this release.',
    variants=[dict(id=r['design']['name'],thermal_R_K_per_W=r['groups']['fingers']['thermal_R_K_per_W'],
                   thermal_C_J_per_K=r['groups']['fingers']['thermal_C_J_per_K'],
                   goals_per_hand_minute=r['goals_per_hand_minute'],peak_modeled_temperature_C=r['peak_temperature_C'],
                   drops_across_replicas=r['failures'],goals_per_replica=r['goals_per_replica']) for r in records],
    uncertainty_scope=uncertainty['scope'],
    uncertainty=[r for r in uncertainty['robots']['allegro'] if r['design'] in ids],
)
(OUT/'data/cooling-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'text.color':'#183b32','axes.labelcolor':'#64746a','xtick.color':'#64746a','ytick.color':'#64746a','svg.fonttype':'none'})
fig,axes=plt.subplots(1,2,figsize=(11.4,6),facecolor='#fffefa')
colors=['#88a78b','#216d53','#b77542']; labels=['2.5 K/W','5 K/W\nReference','10 K/W']
for ax in axes:
    ax.set_facecolor('#fffefa');ax.spines[['top','right','left']].set_visible(False)
    ax.spines['bottom'].set_color('#d4dacf');ax.grid(axis='y',color='#e8ece4',linewidth=.7);ax.set_axisbelow(True);ax.tick_params(length=0)
    ax.set_xticks(range(3),labels,fontsize=10)
goals=[r['goals_per_hand_minute'] for r in records]
peaks=[r['peak_temperature_C'] for r in records]
for ax,vals,title,unit,ymax in zip(axes,[goals,peaks],['Sustained task throughput','Peak modeled winding temperature'],['Goals / hand / minute','°C · maximum across replicas'],[55,135]):
    ax.bar(range(3),vals,color=colors,width=.56)
    for i,v in enumerate(vals):ax.text(i,v+ymax*.025,f'{v:.2f}' if ax==axes[0] else f'{v:.1f}°',ha='center',fontsize=13,weight='bold')
    ax.set_ylim(0,ymax);ax.set_title(title,loc='left',fontsize=12,pad=19);ax.set_ylabel(unit,fontsize=9)
fig.text(.06,.935,'Cooling changed sustained performance.',fontsize=22,weight='bold')
fig.text(.06,.875,'Allegro · one fixed policy · 32 paired replicas per design · 120 seconds',fontsize=11,color='#64746a')
fig.text(.06,.13,'Drop events across all replicas: 9 at 2.5 K/W · 2 at 5 K/W · 19 at 10 K/W.',fontsize=10,color='#64746a')
fig.text(.06,.065,'Recorded Isaac Lab outcomes. Thermal capacitance: 6 J/K. The 32 replicas do not represent independent policy training.\nThese data are separate from the newer physical thermal bench test.',fontsize=9,color='#64746a',linespacing=1.6)
fig.subplots_adjust(left=.09,right=.98,top=.76,bottom=.25,wspace=.32)
for ext in ['svg','png']:fig.savefig(OUT/f'cooling-results.{ext}',dpi=170,facecolor=fig.get_facecolor())
print(json.dumps(summary['variants'],indent=2))
