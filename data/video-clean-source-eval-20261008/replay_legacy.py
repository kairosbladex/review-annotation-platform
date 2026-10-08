"""断网验证容器内回放：只上传带框截图 + clean_video_sources，与 Java 视频事件请求一致。"""
import json, sys, time, urllib.request, uuid, os
out = sys.argv[1]
man = [m for m in json.load(open("/work/manifest.json")) if any(d["class_name"] != "人员" for d in m["detections"])]
for _ in range(180):
    try:
        urllib.request.urlopen('http://127.0.0.1:8189/api/v1/health', timeout=5); break
    except Exception:
        time.sleep(5)
for m in man:
    p = f"{out}/{m['event_id']}.json"
    if os.path.exists(p): continue
    b = uuid.uuid4().hex
    body = (f'--{b}\r\nContent-Disposition: form-data; name="image"; filename="shot.jpg"\r\nContent-Type: image/jpeg\r\n\r\n'.encode()
            + open(f"/annotated/{m['event_id']}.jpg", 'rb').read() + b'\r\n')
    dets = [{k: v for k, v in d.items() if k != 'candidate_id'} for d in m['detections']]
    for name, value in (("detections", dets),):
        body += f'--{b}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{json.dumps(value, ensure_ascii=False)}\r\n'.encode()
    body += f'--{b}--\r\n'.encode()
    req = urllib.request.Request('http://127.0.0.1:8189/api/v1/review-agent/review', data=body,
                                 headers={'Content-Type': f'multipart/form-data; boundary={b}'})
    t = time.time(); r = json.load(urllib.request.urlopen(req, timeout=3600))
    json.dump({'event_id': m['event_id'], 'candidate_ids': [d['candidate_id'] for d in m['detections']],
               'elapsed': time.time() - t, 'response': r}, open(p, 'w'), ensure_ascii=False)
    v = r.get('clean_source_video') or {}
    print(m['event_id'], round(time.time() - t, 1), v.get('status'), v.get('reasonCode'), v.get('matchedTime'), flush=True)
print('done')
