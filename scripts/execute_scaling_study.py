"""Run one persistent process per model/GPU, then publish the final analysis."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='runs')
    parser.add_argument('--study', default='scale40-20261011')
    parser.add_argument('--devices', default='0,1,2')
    parser.add_argument('--cpu-sets', default='16-19,24-27,32-35')
    parser.add_argument('--analysis-python', default='/usr/bin/python3')
    parser.add_argument('--source', help='frozen source tree with src/ and scripts/')
    args = parser.parse_args()
    root = Path(args.root).resolve()
    source = Path(args.source).resolve() if args.source else Path(__file__).resolve().parents[1]
    protocol = json.loads((root / f'{args.study}-protocol.json').read_text())
    devices, cpu_sets = args.devices.split(','), args.cpu_sets.split(',')
    if len(devices) != len(protocol['sizes']) or len(cpu_sets) != len(devices):
        raise ValueError('each model needs one device and CPU set')
    workers = []
    execution = {'study': args.study, 'started_unix': time.time(),
        'source_snapshot': str(source), 'workers': []}
    for hidden, gpu, cores in zip(protocol['sizes'], devices, cpu_sets):
        env = os.environ.copy()
        env['ROCR_VISIBLE_DEVICES'] = gpu
        env['TORCHINDUCTOR_COMPILE_THREADS'] = '4'
        env['PYTHONPATH'] = str(source / 'src')
        command = ['taskset', '-c', cores, sys.executable, '-u',
            str(source / 'scripts/run_scaling_study.py'), '--root', str(root),
            '--study', args.study, '--hidden', str(hidden), '--device', 'cuda:0']
        logfile = root / f'{args.study}-h{hidden:03d}.log'
        with logfile.open('a') as log:
            child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        workers.append(child)
        execution['workers'].append({'hidden': hidden, 'rocr_visible_devices': gpu,
            'cpu_set': cores, 'pid': child.pid, 'log': str(logfile), 'command': command})
    path = root / f'{args.study}-execution.json'
    path.write_text(json.dumps(execution, indent=2) + '\n')
    print(json.dumps(execution), flush=True)
    analyze = [args.analysis_python, str(source / 'scripts/analyze_scaling_study.py'),
        '--root', str(root), '--study', args.study]
    last_analysis = 0.
    while any(child.poll() is None for child in workers):
        if time.monotonic() - last_analysis >= 60:
            result = subprocess.run(analyze)
            if result.returncode:
                print(json.dumps({'analysis_error_returncode': result.returncode,
                    'training_continues': True}), flush=True)
            last_analysis = time.monotonic()
        time.sleep(10)
    execution['finished_unix'] = time.time()
    execution['returncodes'] = [child.returncode for child in workers]
    path.write_text(json.dumps(execution, indent=2) + '\n')
    subprocess.run(analyze, check=True)
    if any(child.returncode for child in workers):
        raise SystemExit('One or more model workers failed; results and checkpoints are preserved.')


if __name__ == '__main__':
    main()
