"""Audit disk states against the declared session-boundary transfer rules."""
import json
from pathlib import Path

import torch


def main():
    torch.set_num_threads(4);root=Path('runs');initial=root/'session-selection-initial'
    originals=torch.load(root/'compass-ablation-initial/absolute.pt',map_location='cpu',weights_only=True,mmap=True)
    reports=[];digests={};selections=[]
    for arm in ['acquired','genome','fresh']:
        manifest=json.loads((root/f'session-selection-{arm}.json').read_text())
        for key,record in manifest['sessions'].items():
            index=int(key)
            if index==0:continue
            if record.get('status')=='complete':
                trial=root/record['run']
                selected=torch.load(trial/'elites.pt',map_location='cpu',weights_only=True)
                snapshot=json.loads((trial/'latest.json').read_text())
                events=list(map(json.loads,(trial/'events.jsonl').read_text().splitlines()))
                for wi,donors in enumerate(selected['worlds']):
                    current=snapshot['worlds'][wi]['agents']
                    if current:
                        assert {d['id'] for d in donors}<={a['id'] for a in current}
                        assert all(d['selected_alive'] for d in donors)
                    else:
                        dead=sorted([e for e in events if e['world']==wi and e['event']=='death'],key=lambda e:(e['tick'],e['id']))[-selected['config']['elite_count']:]
                        assert [(d['death_tick'],d['id']) for d in donors]==[(e['tick'],e['id']) for e in dead]
                selections.append({'arm':arm,'session':index+1,'matches_selection_rule':True})
            path=initial/f'{arm}-s{index:02d}.pt'
            if not path.exists():continue
            state=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
            cfg=state['config'];learner=state['learners'];world=state['world'];info=state['initialization']
            assert int(world['clock'])==0
            assert world['alive'].sum(1).tolist()==[cfg['initial']]*cfg['worlds']
            assert not world['age'].any()
            assert world['energy'][world['alive']].unique().tolist()==[cfg['founder_energy']]
            for name in ['hidden','last_action','reward_u','reward_n','reward_mean','reward_m2','adam_count']:
                assert not learner[name].any(),(arm,index,name)
            for pair in learner['moments'].values():
                assert not any(x.any() for x in pair)
            for group in ['traces','bounds']:assert not any(x.any() for x in learner[group].values())
            digest=info['initial_world_digest']
            if index in digests:assert digests[index]==digest,(arm,index,'physical initial state differs')
            digests[index]=digest
            if arm=='fresh':
                for name in learner['params']:assert torch.equal(learner['params'][name],originals['learners']['params'][name])
                assert not any(x.any() for x in learner['normalization_genome'].values())
            else:
                packpath=initial/'source-elites.pt' if index==1 else root/manifest['sessions'][str(index-1)]['run']/'elites.pt'
                pack=torch.load(packpath,map_location='cpu',weights_only=True)
                donor_maps=[{d['id']:d for d in rows} for rows in pack['worlds']]
                for assignment in info['assignments']:
                    wi=assignment['world'];idx=wi*cfg['capacity']+assignment['founder_id']-1
                    donor=donor_maps[wi][assignment['donor_id']]
                    weights=donor['params' if arm=='acquired' else 'genome']
                    for name,v in weights.items():
                        assert torch.equal(learner['params'][name][idx],v),(arm,index,idx,name)
                        assert torch.equal(learner['genome'][name][idx],v),(arm,index,idx,'genome',name)
                    norm=donor['normalization' if arm=='acquired' else 'normalization_genome']
                    for name,v in norm.items():
                        assert torch.equal(learner[name][idx],v)
                        assert torch.equal(learner['normalization_genome'][name][idx],v)
                    for dest,source in [('log_rates','log_rates' if arm=='acquired' else 'genome_rates'),('eval_w','eval_w' if arm=='acquired' else 'genome_eval')]:
                        assert torch.equal(learner[dest][idx],donor[source])
            reports.append({'arm':arm,'session':index+1,'exact_transfer':True,'reset_correct':True,'world_digest':digest})
    report={'checked_initial_states':len(reports),'all_exact':True,'reports':reports,'completed_selection_audits':selections}
    out=root/'session-selection-analysis';out.mkdir(exist_ok=True)
    (out/'initialization-audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'checked_initial_states':len(reports),'all_exact':True}))


if __name__=='__main__':main()
