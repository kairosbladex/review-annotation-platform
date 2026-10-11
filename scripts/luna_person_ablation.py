"""Bounded offline view/prompt ablation using the frozen worker's actual execute path.

No HTTP listener, database access, service replacement, or label changes.
Run only inside a resource-capped systemd unit with the existing CLI environment.
"""
import argparse
import base64
import copy
import hashlib
import importlib.util
import io
import json
import math
import os
import time
from pathlib import Path

from PIL import Image, ImageDraw


def zoom_sheet(image, candidates):
    """Display existing pixels larger; no inpainting or super-resolution."""
    cols = min(3, len(candidates))
    canvas = Image.new('RGB', (cols * 256, math.ceil(len(candidates) / cols) * 280), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, candidate in enumerate(candidates):
        b = candidate['bbox']
        x1, y1, x2, y2 = b[0]*image.width, b[1]*image.height, b[2]*image.width, b[3]*image.height
        mx, my = max(20, (x2-x1)*.25), max(20, (y2-y1)*.25)
        rect = (max(0, x1-mx), max(0, y1-my), min(image.width, x2+mx), min(image.height, y2+my))
        crop = image.crop(rect)
        scale = min(4, 256/max(crop.size))
        resampling = getattr(Image, 'Resampling', Image)
        crop = crop.resize((max(1, round(crop.width*scale)), max(1, round(crop.height*scale))), resampling.BICUBIC)
        px, py = i%cols*256, i//cols*280
        canvas.paste(crop, (px, py+24))
        draw.text((px+4, py+4), str(candidate['target_index']), fill='black')
    return canvas


def project_box(box, matrix, low_size, high_size):
    import numpy as np
    x1, y1, x2, y2 = box
    w, h = low_size
    points = np.array([[x1*w,y1*h,1],[x2*w,y1*h,1],[x2*w,y2*h,1],[x1*w,y2*h,1]]) @ np.array(matrix).T
    points = points[:,:2]/points[:,2:]
    minimum, maximum = points.min(axis=0), points.max(axis=0)
    if not (minimum >= 0).all() or not (maximum <= high_size).all():
        raise ValueError('projected box outside original')
    return [minimum[0]/high_size[0],minimum[1]/high_size[1],maximum[0]/high_size[0],maximum[1]/high_size[1]]


def zoom_payload(payload):
    """Use the exact decoded input pixels, without an intermediate JPEG."""
    source = Image.open(io.BytesIO(base64.b64decode(payload['image'], validate=True))).convert('RGB')
    return zoom_sheet(source, payload['candidates'])


def payload_for(root, unit, arm):
    if arm not in ('P','S','H','Z','C'):
        raise ValueError('unknown arm')
    path = root/'images'/f'{unit["unit"]}.jpg' if arm == 'P' else (
        root/'images'/f'{unit["source_event"]}-clean.jpg' if arm == 'S' else root/'frames'/f'{unit["source_event"]}.png')
    raw = path.read_bytes()
    candidates = []
    for i, c in enumerate(unit['candidates']):
        bbox = c['bbox'] if arm in ('P','S') else project_box(c['bbox'],unit['H'],[960,720],unit['native_size'])
        candidates.append({'target_index':i,'class_name':c['class_name'],'bbox':bbox})
    return {'image':base64.b64encode(raw).decode(),'candidates':candidates}


def resource_ok():
    data = {line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
    return data['MemAvailable']/data['MemTotal'] >= .15


def run(args):
    spec = importlib.util.spec_from_file_location('frozen_worker', args.worker)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    base = worker.load_prompt('p000-pure-class-cards')
    card = (args.root/'person_candidate.txt').read_text().strip()
    units = json.loads((args.root/'experiment-units.json').read_text())
    output = args.root/'ablation'
    output.mkdir(exist_ok=True)
    ledger = output/'ledger.jsonl'
    history = [json.loads(l) for l in ledger.read_text().splitlines()] if ledger.exists() else []

    def append(row):
        row['time'] = time.time()
        with ledger.open('a') as f:
            f.write(json.dumps(row,ensure_ascii=False)+'\n'); f.flush(); os.fsync(f.fileno())
        history.append(row)

    original_popen = worker.subprocess.Popen
    for entry in args.plan.split(','):
        arm, repeat = entry[0], int(entry[1:])
        for unit in units:
            dest = output/f'{arm}{repeat}-{unit["unit"]}.json'
            if dest.exists():
                continue
            payload = payload_for(args.root, unit, arm)
            worker.PROMPT_DOCUMENT = copy.deepcopy(base)
            if arm in ('P','C'):
                assert all(worker._class_key(c['class_name']) == '人员' for c in payload['candidates'])
                worker.PROMPT_DOCUMENT['cards']['人员'] = card
            attempts = sum(x.get('kind')=='attempt' and x.get('entry')==entry and x.get('unit')==unit['unit'] for x in history)
            for attempt in range(attempts, 2):
                if (args.root/'STOP').exists() or not resource_ok():
                    raise RuntimeError('stop or low resources')
                if sum(x.get('kind')=='attempt' for x in history) >= 120:
                    raise RuntimeError('hard call budget reached')
                view_hashes = []

                def capture(command, **kwargs):
                    images = [Path(command[i+1]) for i,x in enumerate(command) if x=='-i']
                    if arm == 'Z':
                        # Match the worker's crop source exactly. Reopening source.jpg
                        # would add a JPEG generation before the tiny crop is enlarged.
                        zoom_payload(payload).save(images[2], quality=95)
                    view_hashes.extend(hashlib.sha256(p.read_bytes()).hexdigest() for p in images)
                    return original_popen(command, **kwargs)

                append({'kind':'attempt','entry':entry,'unit':unit['unit'],'attempt':attempt})
                audit = {}
                started = time.monotonic()
                worker.subprocess.Popen = capture
                try:
                    result = worker.execute(payload, audit)
                    record = {'success':True,'result':result}
                except Exception as exc:
                    record = {'success':False,'error_type':type(exc).__name__,'failure_category':audit.get('failure_category')}
                finally:
                    worker.subprocess.Popen = original_popen
                record.update(entry=entry,unit=unit['unit'],attempt=attempt,elapsed=time.monotonic()-started,
                              view_sha256=view_hashes,prompt_sha256=hashlib.sha256(worker.build_prompt(worker.PROMPT_DOCUMENT,payload['candidates']).encode()).hexdigest())
                append({'kind':'result','entry':entry,'unit':unit['unit'],'attempt':attempt,'success':record['success']})
                temporary = dest.with_suffix('.tmp')
                temporary.write_text(json.dumps(record,ensure_ascii=False)); temporary.replace(dest)
                print(json.dumps({'entry':entry,'done':len(list(output.glob(entry+'-*.json'))),'success':record['success']}),flush=True)
                if record['success']:
                    break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--worker',type=Path,required=True)
    parser.add_argument('--plan',default='P1,S1,H1,Z1')
    run(parser.parse_args())
