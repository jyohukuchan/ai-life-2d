"""Evaluate random-agent survival windows with optional mandatory feeding."""
import argparse
import json
from pathlib import Path

import numpy as np

from analyze_population import analyze


def select_candidate(results,min_eaten,require_all,preference="feeding"):
    eligible=[r for r in results if not require_all or r['qualified']]
    if not eligible:return None
    if min_eaten and preference=='max_supply':
        rank=lambda r:(r['early_feeding_worlds']/r['worlds'],r['regrowth']*r['food_energy'],r['height']*r['width'],r['early_eaten_range'][0])
    elif min_eaten:
        rank=lambda r:(r['joint_success_worlds']/r['worlds'],r['early_feeding_worlds']/r['worlds'],r['height']*r['width'],r['early_eaten_range'][0])
    else:
        rank=lambda r:(r['joint_success_worlds']/r['worlds'],r['short_population_median'],-abs(r['metabolism']-.02))
    return max(eligible,key=rank)['name']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',default='runs/random-feeding-current-study.json')
    parser.add_argument('--output',default='runs/random-feeding-analysis/current')
    args=parser.parse_args();study_path=Path(args.study).resolve();study=json.loads(study_path.read_text())
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    short,long=study['target']['short_tick'],study['target']['long_tick']
    min_eaten=study['target'].get('min_total_eaten',1)
    if not isinstance(min_eaten,int) or min_eaten<0:raise ValueError('Invalid feeding threshold')
    require_all=study.get('require_all_worlds',min_eaten>0)
    results=[];traces={};feeding_traces={}
    for candidate in study['candidates']:
        summary,trace=analyze([study_path.parent/candidate['run']],output/candidate['name'])
        if summary['status']!='complete' or summary['through_tick']<long:raise ValueError('Wait for complete horizon before selecting a condition')
        seeds=[w['seed'] for w in summary['worlds']]
        if seeds!=study['seeds']:raise ValueError('Unexpected seeds')
        metrics=[json.loads(line) for line in (study_path.parent/candidate['run']/'metrics.jsonl').read_text().splitlines()]
        by_tick={r['tick']:r for r in metrics}
        if short not in by_tick or long not in by_tick:raise ValueError('Need exact metrics at both target ticks')
        cfg=json.loads((study_path.parent/candidate['run']/'config.json').read_text())
        early=trace[short];late=trace[long]
        eaten=np.array(by_tick[long]['eaten']);early_eaten=np.array(by_tick[short]['eaten'])
        temporal=(early>0)&(late==0);fed=eaten>=min_eaten;joint=temporal&fed
        extinction=[w['extinction_tick'] for w in summary['worlds']]
        observed=np.array([t for t in extinction if t is not None])
        row={**candidate,'height':cfg.get('height') or cfg['size'],'width':cfg.get('width') or cfg['size'],
             'regrowth':cfg['regrowth'],'food_energy':cfg['food_energy'],'metabolism':cfg['metabolism'],'worlds':len(seeds),'short_surviving_worlds':int((early>0).sum()),'long_extinct_worlds':int((late==0).sum()),
             'temporal_success_worlds':int(temporal.sum()),'feeding_success_worlds':int(fed.sum()),
             'early_feeding_worlds':int((early_eaten>=max(1,min_eaten)).sum()),'zero_eating_worlds':int((eaten==0).sum()),
             'joint_success_worlds':int(joint.sum()),'qualified':bool(joint.all()),
             'short_population':early.tolist(),'long_population':late.tolist(),
             'short_population_median':float(np.median(early)),'short_population_range':[int(early.min()),int(early.max())],
             'extinction_tick':extinction,'extinction_median':float(np.median(observed)) if len(observed) else None,
             'extinction_range':[int(observed.min()),int(observed.max())] if len(observed) else None,
             'total_eaten':eaten.tolist(),'total_eaten_median':float(np.median(eaten)),'total_eaten_range':[int(eaten.min()),int(eaten.max())],
             'early_eaten':early_eaten.tolist(),'early_eaten_median':float(np.median(early_eaten)),'early_eaten_range':[int(early_eaten.min()),int(early_eaten.max())],
             'failed_worlds':[{'world':i,'seed':seed,'short_population':int(early[i]),'long_population':int(late[i]),'total_eaten':int(eaten[i])} for i,seed in enumerate(seeds) if not joint[i]],
             'all_metrics_verified_against_birth_death_events':True}
        results.append(row);traces[candidate['name']]=trace
        feeding_traces[candidate['name']]=(np.array([0]+[r['tick'] for r in metrics]),np.array([0.]+[float((np.array(r['eaten'])>=max(1,min_eaten)).mean()) for r in metrics]))
    result={'target':study['target'],'seeds':study['seeds'],'results':results,
            'selected':select_candidate(results,min_eaten,require_all,study.get('preference','feeding')),'require_all_worlds':require_all,
            'selection_rule':study['selection_rule'],'sampling':'paired world seeds across candidate conditions; worlds are the replication unit; no world is filtered out'}
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3 if min_eaten else 2,figsize=(15 if min_eaten else 11,4),constrained_layout=True)
    for name,trace in traces.items():
        ticks=np.arange(len(trace));alive=(trace>0).mean(1)
        axes[0].plot(ticks,trace.mean(1),label=name)
        axes[1].plot(ticks,alive,label=name)
        if min_eaten:axes[2].plot(*feeding_traces[name],label=name,marker='.',ms=3)
    for ax in axes:
        ax.axvline(short,color='#333',ls='--',lw=1);ax.axvline(long,color='#333',ls=':',lw=1)
        ax.set(xlabel='Simulation tick',xlim=(0,long));ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=8)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set(ylabel='Mean living population',title='Population, including extinct worlds')
    axes[1].set(ylabel='Fraction of worlds still alive',ylim=(-.02,1.02),title=f'Survive at {short:,}; extinct by {long:,}')
    if min_eaten:axes[2].set(ylabel='Fraction of worlds meeting feeding minimum',ylim=(-.02,1.02),title=f'Each world must eat >= {min_eaten} food')
    fig.savefig(output/'survival-window.png',dpi=160);plt.close(fig)
    print(json.dumps({'selected':result['selected'],'results':[{k:r[k] for k in ['name','joint_success_worlds','worlds','short_population_median','extinction_range','zero_eating_worlds','early_feeding_worlds','total_eaten_range']} for r in results]}))


if __name__=='__main__':main()
