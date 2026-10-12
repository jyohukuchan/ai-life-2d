"""Summarize observed evaluator terms, including non-additive group-removal effects."""
import argparse
import json
from pathlib import Path

import numpy as np


def analyze(directory,output):
    root=Path(directory);source=root/'reward-components'
    schema=json.loads((source/'schema.json').read_text())
    paths=sorted(source.glob('*.npz'))
    if not paths:raise ValueError('No completed reward log block')
    blocks=[]
    for path in paths:
        with np.load(path,allow_pickle=False) as data:blocks.append({k:data[k] for k in data.files})
    data={k:np.concatenate([b[k] for b in blocks]) for k in blocks[0]}
    terms=data['terms'].astype(np.float64);total=terms.sum(-1)
    _,identity=np.unique(np.column_stack((data['world'],data['id'])),axis=0,return_inverse=True)
    counts=np.bincount(identity)
    np.testing.assert_allclose(np.tanh(total),data['raw_reward'],rtol=2e-6,atol=1e-8)
    def stats(values,mask):
        means=np.bincount(identity,weights=values)/counts
        residual=values-means[identity];selected=values[mask]
        return {'mean':float(selected.mean()),'mean_abs':float(np.abs(selected).mean()),'rms':float(np.sqrt(np.square(selected).mean())),
                'std':float(selected.std()),'within_individual_rms':float(np.sqrt(np.square(residual[mask]).mean()))}
    groups=schema['groups']|{'all_action':sum([schema['groups'][k] for k in ['movement','mouth','sent_sound','reproduction']],[])}
    summaries=[]
    for world in [None]+sorted(np.unique(data['world']).tolist()):
        mask=np.ones(len(total),dtype=bool) if world is None else data['world']==world
        rows={}
        for name,indices in groups.items():
            contribution=terms[:,indices].sum(-1)
            effect=np.tanh(total)-np.tanh(total-contribution)
            rows[name]={'pre_tanh':stats(contribution,mask),'group_removal':stats(effect,mask),
                        'group_removal_fixed_scale':stats(effect/data['scale'],mask)}
        events={}
        for event in ['ate','moved','birth_parent','terminal']:
            selected=mask&data[event]
            events[event]={'count':int(selected.sum()),'mean_raw_reward':float(data['raw_reward'][selected].mean()) if selected.any() else None}
        summaries.append({'world':world,'active_transitions':int(mask.sum()),'components':rows,'events':events,
                          'positive_energy_coefficient_transition_fraction':float((data['energy_coefficient'][mask]>0).mean()),
                          'raw_reward':stats(data['raw_reward'],mask),'scaled_reward':stats(data['scaled_reward'],mask)})
    result={'run':root.name,'first_tick':int(data['tick'].min()),'last_tick':int(data['tick'].max()),
            'covered_through_tick':int(paths[-1].stem.split('-')[-1]),'log_files':len(paths),
            'amplitudes':'RMS includes constant offsets; within_individual_rms removes each world/ID mean. Neither measures policy influence or causal learning benefit.',
            'weighting':'one equally weighted old-live-agent transition, not one vote per individual or lineage',
            'group_removal':'hold observed phi, other terms and current normalization scale fixed; effects do not add and do not simulate retraining',
            'event_conditioning':'descriptive, not causal; actions, observations and lineages are confounded',
            'summaries':summaries}
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    (output/'reward-summary.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=summaries[0]['components'];names=list(schema['groups'])
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for ax,key,title in zip(axes,['pre_tanh','group_removal_fixed_scale'],['Additive terms before tanh','Group removal, with recorded scale held fixed']):
        values=[rows[name][key]['rms'] for name in names]
        ax.barh(names,values,color=['#197f69']+['#416bb0']*2+['#c06d2e']*4)
        ax.set(xlabel='RMS amplitude (observed active transitions)',title=title,xscale='log')
        ax.invert_yaxis();ax.grid(axis='x',alpha=.2);ax.spines[['top','right']].set_visible(False)
    fig.suptitle(root.name+' — actual evaluator component magnitudes')
    fig.savefig(output/'reward-components.png',dpi=160);plt.close(fig)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run');parser.add_argument('--output',required=True)
    args=parser.parse_args();result=analyze(args.run,args.output)
    print(json.dumps({k:result[k] for k in ['run','first_tick','last_tick','log_files']}))
