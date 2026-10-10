#!/usr/bin/env python3
"""只读核对新航次道路裂纹事件和 MinIO 对象；不读取复检结论，不下载视频。"""
import argparse
import datetime as dt
import json
from pathlib import Path
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import pymysql

DEVICE = 'AF5AFA566B175B72867FD565F9E5436B'
BASE = 'https://lxtest.zhuoyilingxi.com:8088/minio-src/zm-zq-test'
NS = {'s': 'http://s3.amazonaws.com/doc/2006-03-01/'}


def listing(prefix, delimiter=None):
    objects, prefixes, token = [], [], None
    while True:
        params = {'list-type': '2', 'prefix': prefix}
        if delimiter: params['delimiter'] = delimiter
        if token: params['continuation-token'] = token
        with urllib.request.urlopen(BASE + '?' + urllib.parse.urlencode(params), timeout=30) as res:
            root = ET.fromstring(res.read())
        for row in root.findall('s:Contents', NS):
            objects.append({k: row.findtext('s:' + k, namespaces=NS) for k in ('Key', 'Size', 'LastModified', 'ETag')})
        prefixes += [p.findtext('s:Prefix', namespaces=NS) for p in root.findall('s:CommonPrefixes', NS)]
        if root.findtext('s:IsTruncated', namespaces=NS) != 'true': break
        token = root.findtext('s:NextContinuationToken', namespaces=NS)
        if not token: raise RuntimeError('truncated listing without continuation token')
    return objects, prefixes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--env-file', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    env = {}
    for line in a.env_file.read_text().splitlines():
        if '=' in line and not line.lstrip().startswith('#'):
            k, v = line.split('=', 1); env[k.strip()] = v.strip().strip('\"').strip("'")
    p = 'KTD_SOURCE_MYSQL_'
    if env[p + 'DATABASE'] != 'ktd_lx': raise RuntimeError('expected ktd_lx')
    conn = pymysql.connect(host=env[p+'HOST'], port=int(env[p+'PORT']), user=env[p+'USER'],
                           password=env[p+'PASSWORD'], database=env[p+'DATABASE'],
                           connect_timeout=10, read_timeout=60, cursorclass=pymysql.cursors.DictCursor)
    audit = {'captured_at': dt.datetime.now(dt.timezone.utc).isoformat(), 'device': DEVICE,
             'start_inclusive': '2026-10-08 00:00:00', 'connection': {'host': env[p+'HOST'], 'port': int(env[p+'PORT']), 'database': 'ktd_lx'}, 'queries': []}
    try:
        with conn.cursor() as q:
            q.execute('SET SESSION TRANSACTION READ ONLY')
            q.execute('START TRANSACTION READ ONLY')
            def query(name, sql, params=()):
                q.execute(sql, params); rows = q.fetchall()
                audit['queries'].append({'name': name, 'sql': sql, 'params': params, 'rows': rows})
                print(name, len(rows), flush=True)
                return rows
            query('identity', 'SELECT DATABASE() db,@@hostname server,@@port port,NOW() db_now,@@session.transaction_read_only readonly')
            query('event_extent', 'SELECT COUNT(*) n,MAX(event_time) latest_event,MAX(create_time) latest_create FROM bo_task_event')
            query('new_event_categories', "SELECT upload_data_type,upload_data_type_name,COUNT(*) n FROM bo_task_event WHERE event_time >= %s GROUP BY 1,2", ('2026-10-08',))
            query('historical_crack_categories', "SELECT upload_data_type,upload_data_type_name,COUNT(*) n,MAX(event_time) latest_event,MAX(create_time) latest_create FROM bo_task_event WHERE upload_data_type_name LIKE %s GROUP BY 1,2", ('%裂%',))
            query('newly_created_categories', "SELECT upload_data_type,upload_data_type_name,COUNT(*) n,MIN(event_time) earliest_event,MAX(event_time) latest_event FROM bo_task_event WHERE create_time >= %s GROUP BY 1,2", ('2026-10-08',))
            events = query('new_crack_events_all_devices', "SELECT id,event_time,create_time,terminal_id,task_id,upload_data_type,upload_data_type_name,imgxy FROM bo_task_event WHERE (event_time >= %s OR create_time >= %s) AND (upload_data_type_name LIKE %s OR upload_data_type IN (SELECT DISTINCT upload_data_type FROM bo_task_event WHERE upload_data_type_name LIKE %s)) ORDER BY event_time,id", ('2026-10-08', '2026-10-08', '%裂%', '%裂%'))
            query('device_new_annex_summary', "SELECT busi_type_code,file_type,COUNT(*) n,MIN(create_time) earliest,MAX(create_time) latest,SUM(imgxy IS NOT NULL AND imgxy <> '') with_boxes FROM sys_annex_file WHERE file_path LIKE %s AND (create_time >= %s OR file_path LIKE %s OR file_path LIKE %s OR file_path LIKE %s) GROUP BY 1,2", ('%'+DEVICE+'%', '2026-10-08', '%/20261008/%', '%/20261009/%', '%/20261010/%'))
            query('device_new_shots', "SELECT id,busi_id,event_id,busi_type_code,file_type,file_path,create_time,date_time_original,imgxy FROM sys_annex_file WHERE file_path LIKE %s AND (create_time >= %s OR file_path LIKE %s OR file_path LIKE %s OR file_path LIKE %s) AND (busi_type_code='SHOTIMG' OR file_type='SHOTIMG') ORDER BY create_time,id", ('%'+DEVICE+'%', '2026-10-08', '%/20261008/%', '%/20261009/%', '%/20261010/%'))
            query('new_shot_event_links', "SELECT e.upload_data_type_name,a.busi_type_code,COUNT(*) n FROM sys_annex_file a JOIN bo_task_event e ON a.busi_id=e.id OR a.event_id=e.id WHERE e.event_time >= %s GROUP BY 1,2", ('2026-10-08',))
            query('device_new_boxed_annex', "SELECT id,busi_id,event_id,busi_type_code,file_path,create_time,imgxy FROM sys_annex_file WHERE file_path LIKE %s AND create_time >= %s AND imgxy IS NOT NULL AND imgxy <> '' ORDER BY create_time,id", ('%'+DEVICE+'%', '2026-10-08'))
            for e in events:
                query('event_annex_'+e['id'], 'SELECT id,busi_id,event_id,busi_type_code,file_type,file_path,create_time,date_time_original,imgxy FROM sys_annex_file WHERE busi_id=%s OR event_id=%s', (e['id'], e['id']))
    finally:
        conn.rollback(); conn.close()
    audit['transaction_end'] = 'ROLLBACK'
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out/'database-audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=str)+'\n')
    _, dates = listing('ktdTest/'+DEVICE+'/', '/')
    objects = []
    for date in sorted(dates):
        if date.rstrip('/').rsplit('/', 1)[-1] >= '20261008':
            found, _ = listing(date); objects += found; print(date, len(found), flush=True)
    (a.out/'minio-inventory.json').write_text(json.dumps({'captured_at': dt.datetime.now(dt.timezone.utc).isoformat(), 'base_url': BASE, 'objects': objects}, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
