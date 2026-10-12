"""Run matched trial blocks on fixed GPUs; publish progress and final estimates."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def worker(root, plan, source, gpu):
    progress = {'rocr': gpu, 'status': 'running', 'jobs': {}}
    path = root / f'{plan["study"]}-worker{gpu}.json'
    for job in [j for j in plan['jobs'] if j['rocr'] == gpu]:
        env = os.environ.copy()
        env.update(ROCR_VISIBLE_DEVICES=gpu, TORCHINDUCTOR_COMPILE_THREADS='4',
            PYTHONPATH=str(source / 'src'))
        command = ['taskset', '-c', job['cores'], sys.executable, '-u',
            str(source / 'scripts/run_scaling_study.py'), '--root', str(root),
            '--study', job['study'], '--hidden', str(job['hidden']), '--device', 'cuda:0']
        logfile = root / f'{plan["study"]}-{job["id"]}.log'
        progress['current_job'] = job['id']
        progress['jobs'][job['id']] = {'status': 'running', 'log': str(logfile),
            'started_unix': time.time(), 'command': command}
        save(path, progress)
        with logfile.open('a') as log:
            code = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        progress['jobs'][job['id']].update(status='complete' if code == 0 else 'failed',
            returncode=code, finished_unix=time.time())
        save(path, progress)
        if code:
            progress['status'] = 'failed'
            save(path, progress)
            return code
    progress['status'] = 'complete'
    save(path, progress)
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='runs')
    p.add_argument('--study', default='rep20-20261012')
    p.add_argument('--source', required=True)
    p.add_argument('--worker-gpu')
    p.add_argument('--analysis-python', default='/usr/bin/python3')
    args = p.parse_args()
    root, source = Path(args.root).resolve(), Path(args.source).resolve()
    plan = json.loads((root / f'{args.study}-plan.json').read_text())
    if args.worker_gpu is not None:
        raise SystemExit(worker(root, plan, source, args.worker_gpu))
    processes = []
    execution = {'study': args.study, 'started_unix': time.time(),
        'source_snapshot': str(source), 'workers': []}
    for gpu in sorted({j['rocr'] for j in plan['jobs']}):
        command = [sys.executable, '-u', str(source / 'scripts/execute_replicate_study.py'),
            '--root', str(root), '--study', args.study, '--source', str(source), '--worker-gpu', gpu]
        child = subprocess.Popen(command)
        processes.append(child)
        execution['workers'].append({'gpu': gpu, 'pid': child.pid})
    path = root / f'{args.study}-execution.json'
    save(path, execution)
    print(json.dumps(execution), flush=True)
    analyze = [args.analysis_python, str(source / 'scripts/analyze_replicate_study.py'),
        '--root', str(root), '--study', args.study]
    last = 0.
    while any(child.poll() is None for child in processes):
        if time.monotonic() - last >= 60:
            result = subprocess.run(analyze)
            if result.returncode:
                print(json.dumps({'analysis_error': result.returncode, 'training_continues': True}), flush=True)
            last = time.monotonic()
        time.sleep(10)
    execution.update(finished_unix=time.time(), returncodes=[child.returncode for child in processes])
    save(path, execution)
    subprocess.run(analyze, check=True)
    if any(child.returncode for child in processes):
        raise SystemExit('A trial block failed; results and checkpoints are preserved.')


if __name__ == '__main__':
    main()
