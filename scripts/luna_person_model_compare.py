"""Fixed-model crossover on frozen inputs; no HTTP listener or business writes."""
import argparse
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path

from PIL import Image
from luna_person_ablation import payload_for, resource_ok


MODEL = 'gpt-5.6-sol'


def plan(units):
    # Reverse representation order in round 2 to reduce ordering confounds.
    return [(f'SOL-{arm}{round_no}', unit, arm) for round_no in (1, 2)
            for unit in units for arm in (('A', 'H') if round_no == 1 else ('H', 'A'))]


def image_contract(command, expected, model):
    assert command[command.index('-m')+1] == model
    paths = [Path(command[i+1]) for i, x in enumerate(command) if x == '-i']
    hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    if len(paths) != 3 or hashes != expected:
        raise ValueError('image bytes differ from frozen Luna input')
    sizes = []
    for path in paths:
        with Image.open(path) as image:
            sizes.append(list(image.size))
    return {'sha256': hashes, 'sizes': sizes,
            'image_count': len(paths), 'requested_model': model}


def run(args):
    spec = importlib.util.spec_from_file_location('frozen_worker', args.worker)
    worker = importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)
    worker.MODEL = MODEL
    worker.PROMPT_DOCUMENT = worker.load_prompt('p000-pure-class-cards')
    units = json.loads((args.root/'experiment-units.json').read_text())
    expected = json.loads((args.root/'expected-inputs.json').read_text())
    output = args.root/'results'; output.mkdir(exist_ok=True)
    ledger = output/'ledger.jsonl'
    history = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []

    def append(row):
        row['time'] = time.time()
        with ledger.open('a') as f:
            f.write(json.dumps(row, ensure_ascii=False)+'\n'); f.flush(); os.fsync(f.fileno())
        history.append(row)

    original_popen = worker.subprocess.Popen
    completed = 0
    for entry, unit, arm in plan(units):
        destination = output/f'{entry}-{unit["unit"]}.json'
        if destination.exists():
            continue
        payload = payload_for(args.root, unit, 'P' if arm == 'A' else 'H')
        prompt_hash = hashlib.sha256(worker.build_prompt(worker.PROMPT_DOCUMENT, payload['candidates']).encode()).hexdigest()
        if prompt_hash != expected[unit['unit']][arm]['prompt_sha256']:
            raise ValueError('prompt differs from frozen contract')
        attempts = sum(r.get('kind') == 'attempt' and r.get('entry') == entry and r.get('unit') == unit['unit'] for r in history)
        for attempt in range(attempts, 2):
            if (args.root/'STOP').exists() or not resource_ok():
                raise RuntimeError('stop or insufficient memory')
            if sum(r.get('kind') == 'attempt' for r in history) >= 80:
                raise RuntimeError('attempt budget reached')
            audit, contract = {}, {}

            def capture(command, **kwargs):
                contract.update(image_contract(command, expected[unit['unit']][arm]['view_sha256'], MODEL))
                return original_popen(command, **kwargs)

            append({'kind': 'attempt', 'entry': entry, 'unit': unit['unit'], 'attempt': attempt})
            started = time.monotonic(); worker.subprocess.Popen = capture
            try:
                record = {'success': True, 'result': worker.execute(payload, audit)}
            except Exception as exc:
                record = {'success': False, 'error_type': type(exc).__name__,
                          'failure_category': audit.get('failure_category'),
                          'safe_error': audit.get('cli_stderr')}
            finally:
                worker.subprocess.Popen = original_popen
            record.update(entry=entry, unit=unit['unit'], attempt=attempt,
                          elapsed=time.monotonic()-started, input_contract=contract,
                          prompt_sha256=prompt_hash, requested_model=MODEL,
                          underlying_model_identity='not independently observable through gateway')
            append({'kind': 'result', 'entry': entry, 'unit': unit['unit'],
                    'attempt': attempt, 'success': record['success']})
            temp = destination.with_suffix('.tmp'); temp.write_text(json.dumps(record, ensure_ascii=False)); temp.replace(destination)
            print(json.dumps({'entry': entry, 'done': len(list(output.glob(entry+'-*.json'))),
                              'success': record['success']}), flush=True)
            if record['success']:
                break
            # Fail closed on setup/schema errors; do not spend a whole cohort on an unavailable model.
            if not contract or audit.get('failure_category') not in ('cli_timeout', 'cli_exit_nonzero', 'cli_error'):
                raise RuntimeError('non-transient preflight or response failure')
        if not record['success']:
            raise RuntimeError('model unavailable after bounded retry')
        completed += 1
        if args.limit and completed >= args.limit:
            break


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--worker', type=Path, required=True)
    p.add_argument('--limit', type=int, default=0)
    run(p.parse_args())
