-- 仅在测试库执行；各 SELECT 使用同一只读一致性快照。
SET SESSION MAX_EXECUTION_TIME=20000;
SET TRANSACTION READ ONLY;
START TRANSACTION WITH CONSISTENT SNAPSHOT, READ ONLY;
SELECT DATABASE() AS db,@@hostname AS server_host,@@port AS server_port,NOW() AS snapshot_time,@@session.time_zone AS time_zone;
SELECT recheck_status,COUNT(*) AS event_count,MAX(recheck_time) AS last_recheck,MAX(update_time) AS last_update FROM bo_task_event GROUP BY recheck_status;
SELECT step_type_name,COUNT(*) AS track_count,MAX(occurred_time) AS last_occurred FROM bo_event_work_order WHERE step_type_name LIKE '%复核%' GROUP BY step_type_name;
SELECT COUNT(*) AS image_result_count,MAX(recheck_time) AS last_recheck FROM bo_task_event_image_recheck_result;
SELECT e.id,e.event_no,e.event_time,e.upload_data_type_name,e.recheck_status,e.recheck_time,CASE WHEN JSON_VALID(e.imgxy) THEN JSON_LENGTH(e.imgxy) ELSE NULL END AS event_box_count FROM bo_task_event e WHERE (e.recheck_status IS NULL OR e.recheck_status<>'2' OR e.recheck_result IS NULL OR TRIM(e.recheck_result) IN ('','{}','null')) AND (e.recheck_time BETWEEN '2026-10-08 16:43:03' AND '2026-10-10 10:26:45' OR EXISTS (SELECT 1 FROM bo_event_work_order w WHERE w.event_id=e.id AND w.step_type_name='提交慧眼复核' AND w.occurred_time BETWEEN '2026-10-08 16:43:03' AND '2026-10-10 10:26:45')) ORDER BY e.event_time,e.id;
SELECT e.id,e.event_no,e.event_time,e.create_time,e.update_time,e.upload_data_type_name,e.recheck_status,e.recheck_time,e.del_flag+0 AS deleted,CASE WHEN JSON_VALID(e.imgxy) THEN JSON_LENGTH(e.imgxy) ELSE NULL END AS event_box_count,(SELECT COUNT(*) FROM bo_event_work_order w WHERE w.event_id=e.id AND w.step_type_name='提交慧眼复核') AS submit_tracks FROM bo_task_event e WHERE (e.create_time>'2026-10-08 16:43:02' OR e.update_time>'2026-10-08 16:43:02') AND (e.recheck_result IS NULL OR TRIM(e.recheck_result) IN ('','{}','null')) ORDER BY e.create_time,e.id;
ROLLBACK;
