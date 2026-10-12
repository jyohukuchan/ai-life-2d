from __future__ import annotations

import argparse
import dataclasses
import json

from .config import Config


def main():
    parser=argparse.ArgumentParser(description='Fast independent-learning artificial life')
    sub=parser.add_subparsers(dest='command',required=True)
    run=sub.add_parser('run')
    run.add_argument('--config')
    run.add_argument('--device',default='cpu')
    run.add_argument('--output')
    run.add_argument('--resume')
    run.add_argument('--no-live',action='store_true')
    run.add_argument('--no-checkpoint',action='store_true')
    for name in ['worlds','initial','capacity','size','height','width','hidden','encoder_width','seed','ticks','threads','log_ticks','checkpoint_ticks','replay_count','replay_length']:
        run.add_argument('--'+name.replace('_','-'),type=int)
    run.add_argument('--mode',choices=['L0','L1','L2','L3','RANDOM'])
    run.add_argument('--obstacle-density',type=float)
    run.add_argument('--compass',choices=['none','masked','absolute'])
    run.add_argument('--precision',choices=['fp32','fp16','bf16'])
    run.add_argument('--execution',choices=['eager','kernels','compiled'])
    run.add_argument('--log-reward-components',action=argparse.BooleanOptionalAction,default=None)
    serve=sub.add_parser('serve')
    serve.add_argument('--root',default='runs')
    serve.add_argument('--host',default='0.0.0.0')
    serve.add_argument('--port',type=int,default=8765)
    bench=sub.add_parser('bench')
    bench.add_argument('--device',default='cpu')
    bench.add_argument('--worlds',type=int,default=1)
    bench.add_argument('--initial',type=int,default=32)
    bench.add_argument('--capacity',type=int,default=128)
    bench.add_argument('--hidden',type=int,default=64)
    bench.add_argument('--encoder-width',type=int,default=32)
    bench.add_argument('--mode',choices=['L0','L1','L2','L3','RANDOM'],default='L3')
    bench.add_argument('--steps',type=int,default=100)
    bench.add_argument('--warmup',type=int,default=10)
    bench.add_argument('--threads',type=int,default=4)
    bench.add_argument('--output',default='runs/benchmarks.jsonl')
    bench.add_argument('--observe',action='store_true')
    bench.add_argument('--compass',choices=['none','masked','absolute'],default='none')
    bench.add_argument('--precision',choices=['fp32','fp16','bf16'],default='fp32')
    bench.add_argument('--execution',choices=['eager','kernels','compiled'],default='eager')
    bench.add_argument('--obstacle-density',type=float,default=0.04)
    args=parser.parse_args()
    if args.command=='serve':
        import uvicorn
        from .server import create_app
        uvicorn.run(create_app(args.root),host=args.host,port=args.port)
    elif args.command=='bench':
        from .benchmark import benchmark
        print(json.dumps(benchmark(args),ensure_ascii=False),flush=True)
    else:
        from .runner import run as execute
        if args.config:cfg=Config.load(args.config)
        elif args.resume:
            import torch
            cfg=Config.from_saved(torch.load(args.resume,map_location='cpu',weights_only=True)['config'])
        else:cfg=Config()
        values={f.name:getattr(args,f.name) for f in dataclasses.fields(cfg) if getattr(args,f.name,None) is not None}
        if args.initial is not None and args.size is None and cfg.height==cfg.width==cfg.size:values['size']=0
        if (args.size is not None or 'size' in values) and args.height is None and args.width is None:
            values.update(height=0,width=0)
        cfg=dataclasses.replace(cfg,**values)
        print(json.dumps(execute(cfg,args.device,args.output,args.resume,not args.no_live,not args.no_checkpoint)),flush=True)


if __name__=='__main__':main()
