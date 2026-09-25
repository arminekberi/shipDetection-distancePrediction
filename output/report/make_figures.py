"""Rebuild report figures from saved project evidence. Run with the project venv."""
import sys, json, csv, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pool_sim

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'output/report/assets'
BASE = ROOT/'results/report_20260925'
BLUE='#184991'; CYAN='#4093bf'; INK='#192638'; GRAY='#64748b'; PALE='#e9f0f9'; WARM='#be693f'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
 'axes.spines.right':False,'axes.labelcolor':INK,'text.color':INK,'axes.edgecolor':'#b8c4d3',
 'xtick.color':GRAY,'ytick.color':GRAY,'axes.titleweight':'bold','figure.facecolor':'white'})
def save(fig,name):
 fig.savefig(OUT/name,dpi=220,bbox_inches='tight',facecolor='white'); plt.close(fig)
def readcsv(p): return list(csv.DictReader(open(p)))

# Architecture is a diagram of the inspected code, not a benchmark.
fig,ax=plt.subplots(figsize=(9,3.4)); ax.set_xlim(0,10);ax.set_ylim(0,4);ax.axis('off')
def box(x,y,w,h,txt,blue=False):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08,rounding_size=0.07',fc=BLUE if blue else PALE,ec='none'))
 ax.text(x+w/2,y+h/2,txt,ha='center',va='center',color='white' if blue else INK,fontsize=10)
def arr(x,y,x2,y2):ax.annotate('',(x2,y2),(x,y),arrowprops={'arrowstyle':'->','color':GRAY,'lw':1.5})
box(.1,2.7,1.8,.9,'Video or raw\ncolour frames');box(2.55,2.7,1.8,.9,'YOLO detection\noptional proposals');box(5,2.7,1.8,.9,'Association\nand tracking');box(7.45,2.7,2.35,.9,'Annotated video\nand per track CSV',True)
for x in [1.95,4.4,6.85]:arr(x,3.15,x+.52,3.15)
box(.1,.35,1.8,1.1,'UWB and IMU\ntelemetry');box(2.55,.35,1.8,1.1,'Time alignment\nand calibration');box(5,.35,1.8,1.1,'Range and\nmotion estimates');box(7.45,.35,2.35,1.1,'Dataset review\nand evaluation')
arr(1.95,.9,2.45,.9);arr(4.4,.9,4.9,.9);arr(5.9,2.6,5.9,1.55);arr(6.85,.9,8.6,2.6)
ax.text(5,3.95,'Optional segmentation and tag observations enter fusion before tracking',ha='center',fontsize=9,color=GRAY)
save(fig,'architecture.png')

manifest=json.loads((ROOT/'results/reviewed_yolo_20260917/dataset/manifest.json').read_text())
c=manifest['counts']; labels=['Train','Validation','Test','Held','Excluded'];keys=['train','val','test','hold','excluded']
fig,ax=plt.subplots(figsize=(8.6,2.7));y=np.arange(5)
pos=np.array([c[k]['positive'] for k in keys]);bg=np.array([c[k]['background'] for k in keys])
ax.barh(y,pos,color=BLUE,label='Positive frames');ax.barh(y,bg,left=pos,color='#b6cbe8',label='Background frames')
for i,k in enumerate(keys): ax.text(pos[i]+bg[i]+28,i,f"{c[k]['frames']:,}",va='center',fontsize=9)
ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlim(0,2120);ax.set_xlabel('Reviewed source frames');ax.legend(frameon=False,ncol=2,loc='lower right');ax.grid(axis='x',alpha=.17);ax.set_axisbelow(True)
save(fig,'dataset.png')

train=readcsv(ROOT/'weights/boat_reviewed_20260917_results.csv');epochs=[int(r['epoch']) for r in train]
fig,axes=plt.subplots(1,2,figsize=(9,3.1))
for key,label,col in [('metrics/mAP50(B)','mAP50',BLUE),('metrics/mAP50-95(B)','mAP50-95',CYAN)]:axes[0].plot(epochs,[float(r[key]) for r in train],label=label,color=col,lw=2)
axes[0].axvline(7,color=GRAY,ls=':',lw=1);axes[0].set_ylim(0,.6);axes[0].set_ylabel('Validation average precision');axes[0].legend(frameon=False)
for key,label,col in [('train/box_loss','Training box loss',BLUE),('val/box_loss','Validation box loss',WARM)]:axes[1].plot(epochs,[float(r[key]) for r in train],label=label,color=col,lw=2)
axes[1].set_ylabel('Box loss');axes[1].legend(frameon=False)
for ax in axes:ax.set_xlabel('Epoch');ax.set_xticks([1,4,7,10,13,17]);ax.grid(alpha=.17)
fig.tight_layout();save(fig,'training.png')

fig,axes=plt.subplots(1,2,figsize=(9,4))
for ax,scenario in zip(axes,['weave','range-sweep']):
 session=json.loads((BASE/f'sim/{scenario}/session.json').read_text());p=session['pool']
 ax.add_patch(plt.Rectangle((p['x_min'],p['y_min']),p['x_max']-p['x_min'],p['y_max']-p['y_min'],fc='#f3f7fc',ec='#b8c4d3'))
 for ship,col in [(1,BLUE),(2,CYAN)]:
  rows=readcsv(BASE/f'sim/{scenario}/truth_ship{ship}.csv');xs=[float(r['x_m']) for r in rows];ys=[float(r['y_m']) for r in rows]
  ax.plot(xs,ys,color=col,lw=2,label=f'Ship {ship}');ax.scatter(xs[0],ys[0],color=col,s=35,zorder=4);ax.plot(xs[-1],ys[-1],marker='x',color=col,ms=8)
 for _,(x,y,z) in pool_sim.ANCHORS.items():ax.plot(x,y,marker='s' if z>5 else '^',color=GRAY,ms=4)
 ax.set_aspect('equal');ax.set_title(scenario);ax.set_xlabel('x in anchor frame (m)');ax.set_ylabel('y (m)');ax.set_xlim(2,32);ax.set_ylim(-2,31);ax.legend(frameon=False,loc='lower right');ax.grid(alpha=.12)
fig.tight_layout();save(fig,'trajectories.png')

# Re-run only the deterministic camera scorer to export its rows for the report.
folder=BASE/'sim/weave';session,records,truths=pool_sim.load(folder); camera,rows=pool_sim.check_camera(session,folder,truths)
pool_sim.write_csv(BASE/'weave_tma_trace.csv',rows)
t,lo,hi,est,true=pool_sim.broken([r['t_sim'] for r in rows],*[ [r[k] for r in rows] for k in ['low_m','high_m','range_m','true_range_m']])
fig,ax=plt.subplots(figsize=(9,3.4));ax.fill_between(t,lo,hi,color='#c4d6ee',alpha=.8,label='Reported 90% range interval');ax.plot(t,est,color=BLUE,lw=1.8,label='Estimated range');ax.plot(t,true,color=INK,lw=2,label='True range')
conv=[r for r in rows if r['status']=='converged'];ax.scatter([r['t_sim'] for r in conv],[r['range_m'] for r in conv],s=22,facecolors='none',edgecolors=WARM,label='Reported converged',zorder=5)
ax.set_ylim(0,65);ax.set_xlim(0,90);ax.set_xlabel('Simulation time (s)');ax.set_ylabel('Range (m)');ax.grid(alpha=.17);ax.legend(frameon=False,ncol=2,fontsize=9);save(fig,'tma_failure.png')

summary=json.loads((ROOT/'results/audit_20260914/models/summary.json').read_text()); hist=[]
for r in summary['results']:
 if r['split']!='test':continue
 groups=[v for k,v in r['by_recording'].items() if k!='renkliTekneTekne'];tp=sum(v['tp'] for v in groups);fp=sum(v['fp'] for v in groups);fn=sum(v['fn'] for v in groups)
 hist.append({'model':Path(r['weights']).stem,'tp':tp,'fp':fp,'fn':fn,'precision':tp/(tp+fp),'recall':tp/(tp+fn),'f1':2*tp/(2*tp+fp+fn)})
(BASE/'historical_recomputed.json').write_text(json.dumps(hist,indent=2))

evaluation=BASE/'reviewed_test/summary.json'
if evaluation.exists():
 data=json.loads(evaluation.read_text())
 if len(data['results'])==2:
  fig,ax=plt.subplots(figsize=(8.7,3));x=np.arange(3);w=.33
  for i,(r,label,col) in enumerate(zip(data['results'],['Default v4','Reviewed fine tune'],[BLUE,CYAN])):
   m=r['metrics'][1];values=[m['precision'],m['recall'],m['f1']]; bars=ax.bar(x+(i-.5)*w,values,w,color=col,label=label)
   ax.bar_label(bars,labels=[f'{v:.3f}' for v in values],padding=4,fontsize=10)
  ax.set_xticks(x,['Precision','Recall','F1']);ax.set_ylim(0,1);ax.set_ylabel('Score at confidence 0.45');ax.legend(frameon=False,loc='upper right');ax.grid(axis='y',alpha=.17);ax.set_axisbelow(True);save(fig,'detector_comparison.png')

sources=['README.md','HANDOFF.md','boatdet/pipeline.py','boatdet/tma.py','boatdet/config.py','boatdet/twin.py','pool_sim.py','evaluate_models.py','weights/boat_reviewed_20260917_results.csv','weights/boat_reviewed_20260917_args.yaml','results/reviewed_yolo_20260917/dataset/manifest.json','results/audit_20260914/models/summary.json']
(BASE/'report_source_hashes.json').write_text(json.dumps({p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},indent=2))
print('Figures saved to',OUT)
