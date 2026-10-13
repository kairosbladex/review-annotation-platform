"""从只读调查快照生成汇总；不连接数据库，不修改业务数据。"""

import argparse
import json
from collections import Counter
from pathlib import Path


def summarize(root):
    def read(name):
        return json.loads((root / name).read_text())

    snapshot = read("database-private.json")
    annexes = [a for a in snapshot["annexes"] if a["del_flag"] in (0, None)]
    relations = snapshot["sourceTargetMappings"]
    groups = Counter()
    no_direct_map = []
    for event in snapshot["events"]:
        own = [a for a in annexes if a["busi_id"] == event["id"]]
        shots = [a for a in own if a["busi_type_code"] == "SHOTIMG"]
        primary = [a for a in own if a["busi_type_code"] == "PRIMARYIMG"]
        shot_paths = {a["file_path"] for a in shots}
        mapped = any(r["target_id"] in {a["id"] for a in shots} for r in relations)
        different = any(a["file_path"] not in shot_paths for a in primary)
        if not shots:
            group = "no_shot"
        elif mapped:
            group = "direct_shot_relation"
        elif different:
            group = "different_primary_no_direct_relation"
        else:
            group = "same_primary_and_shot_no_direct_relation"
        groups[group] += 1
        if shots and not mapped:
            no_direct_map.append(event)

    media = read("task-media-stat.json")
    existing = [a for a in media if a["status"] == "exists"]
    videos = [a for a in existing if a["isVideo"]]
    photos = [a for a in existing if not a["isVideo"] and a["declaredPixels"]]
    photo_tasks = {a["taskId"] for a in photos}
    video_tasks = {a["taskId"] for a in videos}
    recovered = read("recovered-primary-lineage.json")
    direct = read("minio-direct-check.json")
    extended = read("extended-database-private.json")
    result = {
        "capturedAt": snapshot["capturedAt"],
        "scope": "Previously frozen 358-event test cohort, not all events in the test database",
        "readOnly": True,
        "eventCount": len(snapshot["events"]),
        "taskCount": len(snapshot["tasks"]),
        "attachmentSnapshotCount": len(snapshot["annexes"]),
        "sourceTargetRelationCount": len(relations),
        "eventGroups": dict(groups),
        "additionalOriginalLineageEvents": len(recovered),
        "additionalOriginalObjectCount": len({p for r in recovered for p in r["originalPaths"]}),
        "additionalOriginalPixels": sorted({tuple(o["pixels"]) for r in recovered for o in r["originals"]}),
        "newFullyValidatedRecheckPairs": 0,
        "targetedMinioAttachmentChecks": len(direct),
        "targetedMinioUniqueKeys": len({a["objectKey"] for a in direct}),
        "targetedMinioStatuses": dict(Counter(a["status"] for a in direct)),
        "taskMediaChecked": len(media),
        "taskMediaStatuses": dict(Counter(a["status"] for a in media)),
        "taskCameraPhotosWithResolutionMetadata": len(photos),
        "taskCameraPhotoDeclaredResolutions": dict(Counter(a["declaredPixels"] for a in photos)),
        "taskVideoObjects": len(videos),
        "eventsWithSameTaskPhotos": sum(e["task_id"] in photo_tasks for e in no_direct_map),
        "eventsWithSameTaskVideo": sum(e["task_id"] in video_tasks for e in no_direct_map),
        "eventsWithEitherSameTaskMedia": sum(e["task_id"] in photo_tasks | video_tasks for e in no_direct_map),
        "sameTaskMediaIsNotExactFrameMapping": True,
        "annexEventRelationCount": len(snapshot["annexEventRelations"]),
        "taskFilePathRows": len(snapshot["taskFiles"]),
        "upstreamArchiveRows": len(snapshot["upstreamArchives"]),
        "samePathAliasesAcrossDatabase": len(extended["samePathAliases"]),
        "extraAliasRelationCount": len(extended["extraAliasMappings"]),
        "nonemptyCallbackParams": sum(bool(a["callback_param"]) for a in extended["attachmentParams"]),
        "videoProbe": read("video-probe.json"),
        "videoRecoveryExperiment": read("video-recovery-candidate.json"),
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot_directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = summarize(args.snapshot_directory)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ["eventCount", "eventGroups", "additionalOriginalLineageEvents", "taskCameraPhotosWithResolutionMetadata", "taskVideoObjects"]}, ensure_ascii=False))
