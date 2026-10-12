"""Profile the frozen pre-optimization L3 on the established compass world."""
import argparse,dataclasses,importlib.util,json,time
from pathlib import Path
import torch
from ailife.config import Config
from ailife.world import World
from ailife.runner import iteration,sync

p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:0');p.add_argument('--output',required=True);args=p.parse_args()
root=Path('runs/acceleration-20261011');out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
spec=importlib.util.spec_from_file_location('ailife.reference_learner',root/'reference_learner.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
cfg=dataclasses.replace(Config.load('configs/experiments/2026-10-11-compass/compass-absolute.yaml'),log_reward_components=False,elite_count=0,replay_count=0)
torch.set_num_threads(cfg.threads);state=torch.load('runs/compass-ablation-initial/absolute.pt',map_location='cpu',weights_only=True)
def initialize():
 w=World(cfg,args.device);l=mod.Learners(cfg,args.device);w.load_state_dict(state['world']);l.load_state_dict(state['learners']);return w,l
results=[]
for repeat in range(3):
 w,l=initialize()
 for _ in range(32):iteration(w,l)
 sync(args.device);start=time.perf_counter()
 for _ in range(128):iteration(w,l)
 sync(args.device);elapsed=time.perf_counter()-start
 row={'repeat':repeat,'seconds':elapsed,'world_ticks_per_second':cfg.worlds*128/elapsed,'ms_per_tick':elapsed/128*1000};results.append(row);print(json.dumps(row),flush=True)
w,l=initialize()
for _ in range(32):iteration(w,l)
sync(args.device)
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as prof:
 for _ in range(12):iteration(w,l)
 sync(args.device)
prof.export_chrome_trace(str(out/'trace.json'))
(out/'operators.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=35))
result={'config':cfg.as_dict(),'torch':torch.__version__,'hip':torch.version.hip,'device':torch.cuda.get_device_name(),'timings':results,'profiler_steps':12}
(out/'baseline.json').write_text(json.dumps(result,indent=2)+'\n');print((out/'operators.txt').read_text(),flush=True)
