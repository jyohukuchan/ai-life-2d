"""Combine throughput, local numerical checks, and new short learning trials."""
import json,statistics
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


root=Path('runs/acceleration-20261011');out=root/'analysis';out.mkdir(exist_ok=True)
report={'hardware':{},'long_trajectory_exact_match_required':False,'paired_map_session':9,'learning_ticks':3000}
variants=['fp32-eager','fp32-compiled','bf16-compiled','fp16-compiled']
colors=['#777777','#287898','#bf8036','#628954'];labels=['FP32 eager','FP32 compiled','BF16 compiled','FP16 compiled']
for gpu in ['v620','r9700']:
 data=json.loads((root/(gpu+'-bench/results.json')).read_text())
 extra=root/(gpu+'-low-bench/results.json')
 if extra.exists():data['variants'].update(json.loads(extra.read_text())['variants'])
 if gpu=='r9700':
  isolated=root/'r9700-eager-isolated/results.json'
  if isolated.exists():data['variants'].update(json.loads(isolated.read_text())['variants'])
  else:data['variants']['bf16-eager']={'status':'failed_process','returncode':134,'error':'ROCm illegal memory access in a BF16 eager BLAS kernel; excluded','log':'r9700-bench.log'}
 base=data['variants']['reference']['median_ms_per_tick']
 for name,d in data['variants'].items():
  if d['status']=='ok':d['speedup_vs_reference']=base/d['median_ms_per_tick']
 learnpath=root/(gpu+'-learning/results.json')
 learning=json.loads(learnpath.read_text()) if learnpath.exists() else {'variants':{}}
 diag={}
 for variant,conditions in learning['variants'].items():
  diag[variant]={}
  for condition,r in conditions.items():
   trial=Path('runs')/r['run'];metrics=list(map(json.loads,(trial/'metrics.jsonl').read_text().splitlines()))
   events=list(map(json.loads,(trial/'events.jsonl').read_text().splitlines()))
   def decisions(t):return 3*32*t+sum((t-e['tick'])*(1 if e['event']=='birth' else -1) for e in events if e['tick']<=t)
   previous=0;count=0;windows=[]
   for m in metrics:
    d=decisions(m['tick']);total=m['mean_prediction_loss']*d
    windows.append({'tick':m['tick'],'loss':(total-previous)/max(d-count,1),'food':m['eaten'],'alive':m['alive'],'elapsed_seconds':m['elapsed_seconds']})
    previous=total;count=d
   integrity=r['integrity'];summary=r['summary']
   first=[w['loss'] for w in windows if w['tick']<=500];last=[w['loss'] for w in windows if w['tick']>=2500]
   row={'run':r['run'],'windows':windows,'loss_first_500':float(np.mean(first)),'loss_last_750':float(np.mean(last)),
        'food_at_1000':next(w['food'] for w in windows if w['tick']==1000),'food_at_3000':summary['eaten'],
        'alive_at_3000':summary['alive'],'elapsed_seconds':summary['elapsed_seconds'],'world_ticks_per_second_including_setup':summary['world_ticks_per_second'],
        'all_expected_updates_applied':integrity['decisions']==integrity['prediction_updates'], 'integrity':integrity}
   row['prediction_decreased']=row['loss_last_750']<row['loss_first_500']
   diag[variant][condition]=row
  if {'cold','warm'}<=diag[variant].keys():
   cold,warm=diag[variant]['cold'],diag[variant]['warm']
   diag[variant]['paired']={'warm_prediction_lower_at_1000':bool(np.mean([w['loss'] for w in warm['windows'][:4]])<np.mean([w['loss'] for w in cold['windows'][:4]])),
    'mean_food_cold_1000':float(np.mean(cold['food_at_1000'])),'mean_food_warm_1000':float(np.mean(warm['food_at_1000']))}
 report['hardware'][gpu]={'benchmark':data,'learning':diag}
(out/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
for ax,gpu in zip(axs,['v620','r9700']):
 data=report['hardware'][gpu]['benchmark']['variants'];ys=[data[v]['median_ms_per_tick'] for v in variants];xs=np.arange(4)
 ax.bar(xs,ys,color=colors,width=.7)
 for x,v,y in zip(xs,variants,ys):
  timings=data[v]['ms_per_tick'];ax.errorbar(x,y,yerr=[[y-min(timings)],[max(timings)-y]],color='black',capsize=3)
  ax.text(x,y+.6,f"{y:.2f} ms\n{data[v]['speedup_vs_reference']:.2f}x",ha='center',fontsize=9)
 ax.set_xticks(xs,['FP32\neager','FP32\ncompiled','BF16\ncompiled','FP16\ncompiled']);ax.set(title=gpu.upper(),ylabel='Milliseconds / 3-world step');ax.set_ylim(0,max(max(data[v]['ms_per_tick']) for v in variants)+6);ax.grid(axis='y',alpha=.2)
fig.suptitle('Matched pre-birth workload | median of 5 repeats | compile/warmup excluded')
fig.savefig(out/'throughput.png',dpi=170);plt.close(fig)
fig,axs=plt.subplots(2,2,figsize=(11,7),layout='constrained')
for row,gpu in enumerate(['v620','r9700']):
 for col,condition in enumerate(['cold','warm']):
  ax=axs[row,col]
  for variant,label,color in zip(variants,labels,colors):
   d=report['hardware'][gpu]['learning'].get(variant,{}).get(condition)
   if d:ax.plot([w['tick'] for w in d['windows']],[w['loss'] for w in d['windows']],'-o',color=color,label=label,markersize=3)
  ax.set(title=f'{gpu.upper()} / {condition}',xlabel='Simulation tick',ylabel='Prediction MSE / live decision');ax.set_yscale('log');ax.grid(alpha=.2)
axs[0,0].legend(fontsize=9);fig.suptitle('New matched-map 3000-tick trials | 250-tick local loss windows')
fig.savefig(out/'learning.png',dpi=170);plt.close(fig)
print(json.dumps({g:{v:{k:d[k] for k in ['median_ms_per_tick','speedup_vs_reference']} for v,d in x['benchmark']['variants'].items() if d['status']=='ok'} for g,x in report['hardware'].items()},indent=2))
