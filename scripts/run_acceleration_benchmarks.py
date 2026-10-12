"""Isolate each GPU candidate so a driver/kernel failure preserves other results."""
import argparse,json,os,resource,subprocess,sys
from pathlib import Path
from ailife.observer import atomic_json


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
 p.add_argument('--variants',default='reference,fp32-eager,fp32-compiled,bf16-compiled,fp16-compiled')
 p.add_argument('--steps',type=int,default=160);p.add_argument('--warmup',type=int,default=32);p.add_argument('--repeats',type=int,default=5);p.add_argument('--profile',action='store_true')
 args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
 resource.setrlimit(resource.RLIMIT_CORE,(0,0))
 report={'process_per_variant':True,'cpu_affinity':sorted(os.sched_getaffinity(0)), 'visible_devices':os.environ.get('ROCR_VISIBLE_DEVICES'),'variants':{}}
 for name in args.variants.split(','):
  command=[sys.executable,'-u','scripts/benchmark_acceleration.py','--output',str(out/name),'--variants',name,'--steps',str(args.steps),'--warmup',str(args.warmup),'--repeats',str(args.repeats)]
  if args.profile:command.append('--profile')
  with (out/(name+'.log')).open('w') as log:
   try:code=subprocess.run(command,stdout=log,stderr=log,timeout=600).returncode
   except subprocess.TimeoutExpired:code='timeout'
  if code==0:
   result=json.loads((out/name/'results.json').read_text());report['variants'].update(result['variants'])
   for k in ['device','arch','torch','hip','config','steps','warmup','repeats']:report[k]=result[k]
  else:report['variants'][name]={'status':'failed_process','returncode':code,'log':str(out/(name+'.log'))}
  atomic_json(out/'results.json',report);print(json.dumps({'variant':name,**report['variants'][name]}),flush=True)


if __name__=='__main__':main()
