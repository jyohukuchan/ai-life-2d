"""Compare exact birth/death population traces at matching simulation ticks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_population import analyze


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',default='runs/comparison.json')
    p.add_argument('--output',default='runs/population-comparison')
    args=p.parse_args()
    study_path=Path(args.study).resolve();study=json.loads(study_path.read_text())
    labels={'learned':study.get('labels',{}).get('learned','L3'),'control':study.get('labels',{}).get('control','RANDOM')}
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    summaries={};traces={}
    for role in ['learned','control']:
        paths=[study_path.parent/name for name in study[role+'_chain']]
        summaries[role],traces[role]=analyze(paths,output/role)
    end=min(s['through_tick'] for s in summaries.values())
    a,b=traces['learned'][:end+1],traces['control'][:end+1]
    seeds_a=[w['seed'] for w in summaries['learned']['worlds']]
    seeds_b=[w['seed'] for w in summaries['control']['worlds']]
    if seeds_a!=seeds_b:raise ValueError('Paired world seeds differ')
    begin=max(0,end-9999)
    result={'common_tick':end,'time_basis':'same_simulation_tick','tail_start':begin,
            'learned_status':summaries['learned']['status'],'control_status':summaries['control']['status'],
            'labels':labels,'worlds':[],'milestones':[]}
    for i,seed in enumerate(seeds_a):
        row={'world':i,'seed':seed}
        for name,trace in [(labels['learned'],a),(labels['control'],b)]:
            tail=trace[begin:,i];zero=np.flatnonzero(trace[:,i]==0)
            row[name]={'population':int(trace[-1,i]),'tail_mean':float(tail.mean()),
                       'tail_min':int(tail.min()),'tail_max':int(tail.max()),
                       'extinction_tick':int(zero[0]) if len(zero) else None}
        result['worlds'].append(row)
    for tick in sorted(set([0,1800,10000,20000,40000,60000,80000,100000,end])):
        if tick<=end:result['milestones'].append({'tick':tick,labels['learned']:a[tick].tolist(),labels['control']:b[tick].tolist()})
    (output/'comparison-summary.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savetxt(output/'comparison.csv',np.column_stack((np.arange(end+1),a,b)),fmt='%d',delimiter=',',
               header='tick,'+','.join(f'{name}_world_{i}' for name in [labels['learned'],labels['control']] for i in range(a.shape[1])),comments='')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    ticks=np.arange(end+1)
    colors={labels['learned']:'#197f69',labels['control']:'#c06d2e'}
    ax=axes.flat[0]
    for name,trace in [(labels['learned'],a),(labels['control'],b)]:
        ax.plot(ticks,trace.mean(1),color=colors[name],lw=1.5,label=name+f' mean ({a.shape[1]} worlds)')
        ax.fill_between(ticks,trace.min(1),trace.max(1),color=colors[name],alpha=.12)
    ax.set_title('Mean population; shading = range across 3 worlds')
    for i,ax in enumerate(list(axes.flat)[1:]):
        ax.plot(ticks,a[:,i],color=colors[labels['learned']],lw=1,label=labels['learned'])
        ax.plot(ticks,b[:,i],color=colors[labels['control']],lw=1,label=labels['control'])
        ax.set_title(f'World {i} (paired seed {seeds_a[i]})')
    for ax in axes.flat:
        ax.set(xlabel='Simulation tick',ylabel='Living individuals',ylim=(0,None))
        ax.grid(alpha=.2);ax.legend(frameon=False,loc='upper right',fontsize=9)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle(f"{labels['learned']} versus {labels['control']} — through tick {end:,}",fontsize=14)
    fig.savefig(output/'comparison.png',dpi=160);plt.close(fig)
    print(json.dumps(result))


if __name__=='__main__':main()
