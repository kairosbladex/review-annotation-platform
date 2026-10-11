"""Synthetic three-image delivery probe; no business images or labels."""
import argparse
import base64
import hashlib
import importlib.util
import io
import json
import os
import secrets
import time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from luna_person_ablation import resource_ok


def make_images(root):
    manifest=root/'probe-images.json'
    if manifest.exists():return json.loads(manifest.read_text())
    fonts=[Path('/System/Library/Fonts/Supplemental/Arial Bold.ttf'),
           Path('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf')]
    font_path=next((p for p in fonts if p.exists()),None)
    if font_path is None:raise RuntimeError('legible probe font unavailable')
    font=ImageFont.truetype(str(font_path),100)
    rows=[]
    for i,color in enumerate(('red','green','blue')):
        code=''.join(secrets.choice('ACDEFHJKLMNPRTWXYZ') for _ in range(5))
        tile=Image.new('RGB',(800,400),color);draw=ImageDraw.Draw(tile)
        draw.rectangle((60,80,739,319),fill='white');draw.text((200,140),code,font=font,fill='black')
        path=root/f'probe-{i}.jpg';tile.save(path,quality=95)
        rows.append({'index':i,'color':color,'code':code,'path':path.name,
                     'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest.write_text(json.dumps(rows,indent=2)+'\n');return rows


def run(args):
    out=args.root/'probe';out.mkdir(exist_ok=True)
    rows=make_images(out)
    spec=importlib.util.spec_from_file_location('probe_worker',args.worker)
    worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
    prompt=('这是图片传递检查，不是业务识别。你应收到三张独立图片。按附件顺序，'
            '逐张读取白色区域中的5位大写字母/数字，以及背景颜色。'
            '返回results，target_index分别为0、1、2；observed_object只写图片中读到的5位字符；'
            'reasoning只写背景颜色英文red、green或blue；能读取则decision为true，'
            '读不到则uncertain；image_limitation写限制，没有则写无。不得推测其他图片的字符。')
    worker.build_prompt=lambda *_:prompt
    source=Image.new('RGB',(64,64),'gray');buf=io.BytesIO();source.save(buf,format='PNG')
    payload={'image':base64.b64encode(buf.getvalue()).decode(),
             'candidates':[{'target_index':i,'class_name':'图像传递检查','bbox':[.1,.1,.9,.9]} for i in range(3)]}
    original=worker.subprocess.Popen
    for model in ('gpt-5.6-luna','gpt-5.6-sol'):
        destination=out/(model+'.json')
        if destination.exists():continue
        if not resource_ok() or (args.root/'STOP').exists():raise RuntimeError('resource stop')
        worker.MODEL=model;audit={};image_hashes=[]
        def capture(command,**kwargs):
            positions=[i+1 for i,x in enumerate(command) if x=='-i'];assert len(positions)==3
            for pos,row in zip(positions,rows):
                command[pos]=str(out/row['path']);image_hashes.append(row['sha256'])
            return original(command,**kwargs)
        worker.subprocess.Popen=capture;started=time.monotonic()
        try:
            result=worker.execute(payload,audit)
            actual={x['target_index']:x for x in result['results']}
            checks=[{'index':r['index'],'code_match':actual[r['index']]['observed_object'].strip()==r['code'],
                     'color_match':actual[r['index']]['reasoning'].strip().lower()==r['color']} for r in rows]
            record={'success':True,'result':result,'checks':checks,
                    'all_three_passed':all(x['code_match'] and x['color_match'] for x in checks)}
        except Exception as exc:
            record={'success':False,'error_type':type(exc).__name__,'failure_category':audit.get('failure_category')}
        finally:worker.subprocess.Popen=original
        record.update(requested_model=model,elapsed=time.monotonic()-started,image_sha256=image_hashes,
                      prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
        destination.write_text(json.dumps(record,ensure_ascii=False)+'\n')
        print(json.dumps({'model':model,'success':record['success'],'all_three_passed':record.get('all_three_passed')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--worker',type=Path,required=True)
    run(p.parse_args())
