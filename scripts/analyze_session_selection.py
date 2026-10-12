"""Summarize the bounded session-selection experiment without changing its selection rule."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ARMS=['acquired','genome','fresh']
LABELS=['Acquired model','Birth-time genome','Fresh founders']
COLORS=['#30789e','#d18538','#76836b']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='runs')
    parser.add_argument('--output',default='runs/session-selection-analysis')
    args=parser.parse_args();root=Path(args.root);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((root/'session-selection-protocol.json').read_text())
    manifests={arm:json.loads((root/f'session-selection-{arm}.json').read_text()) for arm in ARMS}
    rows=[];report={'protocol':protocol,'arms':{},'independent_chains_per_arm':protocol['worlds_per_arm'],
                   'note':'Session 1 is shared. Subsequent sessions are dependent observations in 3 selection chains, not 30 independent replicates.'}
    for arm,m in manifests.items():
        sessions={int(i):r for i,r in m['sessions'].items() if r.get('status')=='complete'}
        ordered=[sessions[i] for i in sorted(sessions)]
        food=np.array([r['food'] for r in ordered]);ext=np.array([r['extinction_tick'] for r in ordered])
        late=np.array([r['late_food_windows'] for r in ordered]);alive=np.array([r['alive'] for r in ordered])
        persistence=np.where(ext>=0,ext,protocol['config']['ticks'])
        food_positive=[];total_elites=[];early=[];prediction=[]
        for index,r in zip(sorted(sessions),ordered):
            metrics={row['tick']:row for row in map(json.loads,(root/r['run']/'metrics.jsonl').read_text().splitlines())}
            prediction.append(metrics[1000]['mean_prediction_loss'])
            early.append([metrics[t]['eaten'] for t in [1000,2000,3000]])
            elites=r.get('selected',protocol['source_selected'] if index==0 else [[]]*food.shape[1])
            food_positive.append([sum(e['eaten']>0 for e in es) for es in elites])
            total_elites.append([len(es) for es in elites])
            for wi in range(food.shape[1]):
                rows.append({'arm':arm,'session':index+1,'world':wi,'seed':r['seed'],'run':r['run'],
                             'food':r['food'][wi],'extinction_tick':r['extinction_tick'][wi],
                             'alive':r['alive'][wi],**{f'food_at_{t}':metrics[t]['eaten'][wi] for t in [1000,2000,3000]},**{f'late_window_{j+1}_food':x for j,x in enumerate(r['late_food_windows'][wi])},
                             'stable_this_session':r['stable_this_session'][wi],
                             'selected_food_positive':food_positive[-1][wi],'selected_count':total_elites[-1][wi]})
        stability=[]
        for wi in range(food.shape[1]):
            streak=best=0
            for r in ordered:
                streak=streak+1 if r['stable_this_session'][wi] else 0;best=max(best,streak)
            stability.append({'world':wi,'best_consecutive_stable_sessions':best,'criterion_met':best>=3})
        report['arms'][arm]={'status':m['status'],'completed_sessions':len(ordered),
            'food_per_session':food.tolist(),'food_mean_per_session':food.mean(1).tolist(),
            'extinction_per_session':ext.tolist(),'persistence_mean_per_session':persistence.mean(1).tolist(),
            'alive_per_session':alive.tolist(),'late_food_per_session':late.tolist(),
            'early_food_per_session':early,'prediction_loss_first_1000_per_session':prediction,
            'early_food_mean_per_session':np.array(early).mean(2).tolist(),
            'selected_food_positive_per_session':food_positive,'selected_count_per_session':total_elites,
            'stability':stability,
            'first_3_mean_food':float(food[:3].mean()),'last_3_mean_food':float(food[-3:].mean()),
            'first_3_mean_persistence':float(persistence[:3].mean()),'last_3_mean_persistence':float(persistence[-3:].mean())}
    common=set.intersection(*[{int(i) for i,r in m['sessions'].items() if r.get('status')=='complete'} for m in manifests.values()])
    recent=sorted(common)[-3:]
    matched={}
    for arm,m in manifests.items():
        rs=[m['sessions'][str(i)] for i in recent];data=report['arms'][arm]
        matched[arm]={'food_at_1000':float(np.mean([data['early_food_per_session'][i][0] for i in recent])),
                      'total_food':float(np.mean([r['food'] for r in rs])),
                      'observed_persistence_capped_10000':float(np.mean([[t if t>=0 else 10000 for t in r['extinction_tick']] for r in rs])),
                      'horizon_survival_count':sum(n>0 for r in rs for n in r['alive'])}
    report['matched_recent']={'sessions':[i+1 for i in recent],'arms':matched,'note':'Identical session/map seeds across arms; persistence is capped at 10000, not an extinction time for a surviving world.'}
    (out/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    with (out/'world-sessions.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for arm,label,color in zip(ARMS,LABELS,COLORS):
        d=report['arms'][arm];x=np.arange(1,d['completed_sessions']+1)
        food=np.array(d['food_per_session']);ext=np.array(d['extinction_per_session']);late=np.array(d['late_food_per_session'])
        persistence=np.where(ext>=0,ext,10000)
        positive=np.array(d['selected_food_positive_per_session']);count=np.array(d['selected_count_per_session'])
        for ax,data in [(axs[0,0],food),(axs[0,1],persistence)]:
            ax.plot(x,data.mean(1),'-o',label=label,color=color,markersize=4)
            for wi in range(data.shape[1]):ax.scatter(x+(wi-1)*.06,data[:,wi],s=13,color=color,alpha=.35)
        axs[1,0].plot(x,late.sum(2).mean(1),'-o',label=label,color=color,markersize=4)
        axs[1,1].plot(x,positive.sum(1)/count.sum(1).clip(1),'-o',label=label,color=color,markersize=4)
    for ax,title,ylabel in zip(axs.flatten(),['Food acquired in each session','Population persistence (10k censoring)','Food acquired during ticks 7001-10000','Selected individuals that had acquired food'],['Food / world','Tick','Food / world','Fraction']):
        ax.set(title=title,xlabel='Session',ylabel=ylabel);ax.set_xticks(range(1,protocol['sessions']+1));ax.grid(alpha=.2)
    axs[1,1].set_ylim(-.03,1.03);axs[0,1].set_ylim(0,10500)
    axs[0,0].legend(loc='best');fig.suptitle('Last-survivor session selection | 3 chains per condition | session 1 shared')
    fig.savefig(out/'session-selection.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for arm,label,color in zip(ARMS,LABELS,COLORS):
        d=report['arms'][arm];x=np.arange(1,d['completed_sessions']+1);early=np.array(d['early_food_per_session'])
        for j,ax in enumerate(axs):
            ax.plot(x,early[:,j].mean(1),'-o',label=label,color=color,markersize=4)
            for wi in range(early.shape[2]):ax.scatter(x+(wi-1)*.06,early[:,j,wi],s=12,color=color,alpha=.35)
    for ax,t in zip(axs,[1000,2000,3000]):
        ax.set(title=f'Food acquired by tick {t}',xlabel='Session',ylabel='Food / world');ax.set_xticks(range(1,protocol['sessions']+1));ax.grid(alpha=.2)
    axs[0].legend();fig.savefig(out/'early-food.png',dpi=170);plt.close(fig)
    print(json.dumps({arm:{k:d[k] for k in ['status','completed_sessions','first_3_mean_food','last_3_mean_food','stability']} for arm,d in report['arms'].items()}))


if __name__=='__main__':main()
