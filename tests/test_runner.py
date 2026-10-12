import dataclasses
import json
import signal
import subprocess
import sys
import time

import pytest
import torch

from ailife.config import Config
from ailife.runner import run


@pytest.mark.parametrize('compass',['none','masked','absolute'])
def test_disk_checkpoint_resume_matches_uninterrupted_run(tmp_path,compass):
    cfg=Config(worlds=1,initial=2,capacity=4,size=12,hidden=8,radius=1,ticks=8,threads=1,compass=compass,replay_count=1,replay_length=3,log_ticks=4,checkpoint_ticks=4)
    run(cfg,directory=tmp_path/'first',observe=False)
    extended=dataclasses.replace(cfg,ticks=12)
    run(extended,directory=tmp_path/'resumed',resume=tmp_path/'first'/'checkpoint.pt',observe=False)
    run(extended,directory=tmp_path/'reference',observe=False)
    a=torch.load(tmp_path/'resumed'/'checkpoint.pt',weights_only=True)
    b=torch.load(tmp_path/'reference'/'checkpoint.pt',weights_only=True)
    for name,value in a['world'].items():assert torch.equal(value,b['world'][name])
    def equal(x,y):
        if isinstance(x,torch.Tensor):assert torch.equal(x,y)
        elif isinstance(x,dict):
            for key in x:equal(x[key],y[key])
        elif isinstance(x,tuple):
            for left,right in zip(x,y):equal(left,right)
    equal(a['learners'],b['learners'])
    assert all(c['start']>=8 for c in a['replay_schedule'])
    metadata=json.loads((tmp_path/'resumed'/'metadata.json').read_text())
    assert metadata['start_tick']==8
    assert metadata['resumed_from']==str(tmp_path/'first'/'checkpoint.pt')


def test_sigterm_stops_after_a_complete_transition_and_saves(tmp_path):
    directory=tmp_path/'stopped'
    with (tmp_path/'process.log').open('w') as log:
        process=subprocess.Popen([sys.executable,'-m','ailife.cli','run','--mode','L3','--worlds','1','--initial','2','--capacity','4','--size','12','--hidden','8','--threads','1','--ticks','100000','--log-ticks','25','--checkpoint-ticks','0','--replay-count','0','--output',str(directory)],stdout=log,stderr=log)
        try:
            deadline=time.monotonic()+15
            while time.monotonic()<deadline:
                latest=directory/'latest.json'
                if latest.exists() and json.loads(latest.read_text())['tick']>0:break
                time.sleep(.05)
            else:raise AssertionError('simulation did not publish')
            process.send_signal(signal.SIGTERM)
            process.wait(timeout=15)
        finally:
            if process.poll() is None:process.kill();process.wait()
    summary=json.loads((directory/'summary.json').read_text())
    snapshot=json.loads((directory/'latest.json').read_text())
    state=torch.load(directory/'checkpoint.pt',weights_only=True)
    assert summary['status']==snapshot['status']=='interrupted'
    assert summary['tick']==snapshot['tick']==int(state['world']['clock'])
    # Both old founders have exactly one optimizer update per completed tick.
    assert state['learners']['adam_count'][:2].tolist()==[summary['tick']]*2
