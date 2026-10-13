"""人员视图实验的固定预算评测。

原始图片、框和标签不变。基线走现有 thumbnail；候选 A 只改人员视图；候选 B 只加人员卡和开发集参考图。
每张图只调用一次，不重试，不挑选较好一轮。缺失和执行失败保持为失败，不算判假。
本脚本不写业务数据，也不启动共享 worker。
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import io
import json
import os
import time
from pathlib import Path

from PIL import Image

import input_evidence
import person_candidate
import person_view
import promptlib
import view_protocol
import luna_experiment_worker as worker

ROUNDS = (
    ('baseline-1', 'legacy', 'baseline'),
    ('baseline-2', 'legacy', 'baseline'),
    ('candidate-a-1', 'fixed_view_v1', 'candidate-a'),
    ('candidate-a-2', 'fixed_view_v1', 'candidate-a'),
    ('candidate-b-1', 'card_references_v1', 'candidate-b'),
    ('candidate-b-2', 'card_references_v1', 'candidate-b'),
)
EXPOSURE = {
    'reason': '冻结人员集已被反复查看，且没有未参与本实验的独立人员正负样本；留出集只作回归',
    'exposed_source_ids': ['luna-current-flow-20261012-75', 'luna-person-diagnosis-24'],
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_assets(assets: Path):
    items = json.loads((assets / 'items.json').read_text(encoding='utf-8'))
    labels = json.loads((assets / 'labels.json').read_text(encoding='utf-8'))
    return items, labels


def resolve_path(item, annotated_root: Path | None, gold_dir: Path | None):
    if item['image'].startswith('snapshot:') and annotated_root is not None:
        return annotated_root / f"{item['unit']}.jpg"
    if gold_dir is not None:
        path = gold_dir / f"{item['unit']}.jpg"
        if path.is_file():
            return path
    return None


def plan_calls(items, labels):
    """确定参考例、排除分组和实际调用清单。参考例不进入计分。"""
    group_of = {item['unit']: item['group'] for item in items}
    selected = person_candidate.select_references(items, labels, group_of)
    labeled = [item for item in items if item['split'] in {'dev', 'holdout'}
               and any(person_view.is_person(c['class_name']) for c in item['candidates'])]
    excluded = set(selected.get('excluded_groups') or [])
    scored = [item for item in labeled if item['group'] not in excluded]
    references = []
    for row in selected.get('references') or []:
        references.append({'unit': row['unit'], 'group': row['group'], 'key': row['key'],
                           'caption': row['caption'], 'role': row['role'], 'bbox': row['bbox']})
    return {
        'references_ok': selected.get('ok') is True,
        'reference_reason': selected.get('reason') or '',
        'references': references,
        'excluded_groups': sorted(excluded),
        'blocked_groups': selected.get('blocked_groups') or [],
        'labeled_units': len(labeled),
        'scored_units': scored,
        'call_budget': len(scored) * len(ROUNDS),
    }


def _bbox_ok(bbox):
    return (isinstance(bbox, list) and len(bbox) == 4
            and 0 <= bbox[0] < bbox[2] <= 1 and 0 <= bbox[1] < bbox[3] <= 1)


def check_required_images(units, labels, annotated_root, gold_dir):
    errors = []
    resolved = {}
    for item in units:
        path = resolve_path(item, annotated_root, gold_dir)
        if path is None or not path.is_file():
            errors.append(f'{item["unit"]} 找不到冻结图片')
            continue
        digest = sha256_file(path)
        if digest != item['sha256']:
            errors.append(f'{item["unit"]} 图片哈希与冻结清单不一致')
            continue
        for index, candidate in enumerate(item['candidates']):
            if not _bbox_ok(candidate['bbox']):
                errors.append(f'{item["unit"]} 第{index}个框超出合法范围')
            label = labels.get(candidate['key'])
            if not person_view.is_person(candidate['class_name']):
                errors.append(f'{item["unit"]} 含非人员候选，人员实验不应悄悄丢弃')
            elif not label or label.get('label') not in {'true', 'false'} or label.get('split') != item['split']:
                # 请求仍保留原框和原编号。没有金标的框不计入分母，也不猜标签。
                errors.append(('unlabeled', candidate['key']))
        resolved[item['unit']] = path
    unlabeled = [item[1] for item in errors if isinstance(item, tuple) and item[0] == 'unlabeled']
    failures = [item for item in errors if isinstance(item, str)]
    return failures, resolved, unlabeled


def nonperson_pixel_errors(items, annotated_root, gold_dir):
    """没有人员的冻结图必须与旧拼图像素一致。找不到文件就记为未核对，不假装通过。"""
    errors, unchecked = [], []
    checked = 0
    for item in items:
        if item['split'] not in {'dev', 'holdout'}:
            continue
        if any(person_view.is_person(c['class_name']) for c in item['candidates']):
            continue
        path = resolve_path(item, annotated_root, gold_dir)
        if path is None or not path.is_file():
            unchecked.append(item['unit'])
            continue
        if item.get('sha256') and sha256_file(path) != item['sha256']:
            errors.append(f'{item["unit"]} 非人员图哈希不一致，不能用来证明视图未变')
            continue
        with Image.open(path) as source:
            image = source.convert('RGB')
        candidates = input_evidence.request_candidates(item)
        legacy, _ = person_view.legacy_sheet(image, candidates)
        rendered = person_view.render_sheet(image, candidates)
        checked += 1
        if rendered.changed or rendered.sheet.tobytes() != legacy.tobytes():
            errors.append(f'{item["unit"]} 没有人员但固定视图改变了拼图')
    return {'checked': checked, 'unchecked': len(unchecked), 'errors': errors}


def project_evidence(evidence):
    """把基线的 v1 证据投影成视图协议可比较的附件摘要。不改变 G0。"""
    if evidence.get('protocol') == view_protocol.PROTOCOL:
        return evidence
    clean = evidence.get('clean_binding')
    return {
        'source_sha256': evidence['source']['sha256'],
        'candidates_sha256': evidence['candidates_sha256'],
        'source_kind': 'screenshot' if clean in {None, 'not_requested'} else 'unverified',
        'prompt_sha256': evidence['prompt_sha256'],
        'person_changed': False,
        'attachments': [{'role': item['role'], 'sha256': item['sha256']} for item in evidence['attachments']],
    }


def p000_document():
    return json.loads((Path(promptlib.__file__).parent / 'prompts' / 'p000.json').read_text(encoding='utf-8'))


def prompts_for(candidates, reference_sheet: str | None, mode: str):
    if mode == 'card_references_v1':
        prompt = person_candidate.build_prompt(candidates)
        payload_mode = mode
    elif mode == 'fixed_view_v1':
        prompt = promptlib.build(p000_document(), candidates)
        payload_mode = mode
    elif mode == 'legacy':
        prompt = promptlib.build(p000_document(), candidates)
        payload_mode = None
    else:
        raise ValueError('未知评测模式')
    return prompt, payload_mode


def reference_sheet_b64(plan, paths):
    examples = []
    for row in plan['references']:
        with Image.open(paths[row['unit']]) as source:
            image = source.convert('RGB')
        examples.append({'image': image, 'bbox': row['bbox'], 'caption': row['caption']})
    sheet, visible = person_candidate.render_reference_sheet(examples)
    if visible != ['P1', 'P2', 'N1', 'N2']:
        raise RuntimeError('参考图标题不是固定的两组正反例')
    buf = io.BytesIO()
    sheet.save(buf, format='JPEG', quality=95)
    raw = buf.getvalue()
    return base64.b64encode(raw).decode('ascii'), hashlib.sha256(raw).hexdigest()


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding='utf-8')
    temporary.replace(path)


def build_plan(assets, annotated_root, gold_dir):
    items, labels = load_assets(assets)
    plan = plan_calls(items, labels)
    required = list(plan['scored_units'])
    by_unit = {item['unit']: item for item in items}
    for row in plan['references']:
        if row['unit'] not in {item['unit'] for item in required}:
            required.append(by_unit[row['unit']])
    errors, paths, unlabeled = check_required_images(required, labels, annotated_root, gold_dir)
    pixels = nonperson_pixel_errors(items, annotated_root, gold_dir)
    errors.extend(pixels['errors'])
    if not plan['references_ok']:
        errors.append(plan['reference_reason'] or '参考例不足')
    scored_keys = []
    for item in plan['scored_units']:
        for candidate in item['candidates']:
            if candidate['key'] in unlabeled:
                continue
            scored_keys.append(candidate['key'])
    summary = {
        'protocol': view_protocol.PROTOCOL,
        'references_ok': plan['references_ok'],
        'references': [{'unit': row['unit'], 'caption': row['caption'], 'role': row['role']} for row in plan['references']],
        'excluded_groups': plan['excluded_groups'],
        'labeled_person_units': plan['labeled_units'],
        'scored_units': len(plan['scored_units']),
        'scored_boxes': len(scored_keys),
        'unlabeled_boxes_kept_in_request': unlabeled,
        'call_budget': plan['call_budget'],
        'rounds': [name for name, _, _ in ROUNDS],
        'nonperson_checked': pixels['checked'],
        'nonperson_unchecked': pixels['unchecked'],
        'errors': errors,
        'model': worker.MODEL,
        'pillow': Image.__version__,
        'provider': 'https://sub.butterfly2sea.us',
        'source_kind': 'screenshot',
        'clean_frame': '未使用。没有可信原帧绑定',
    }
    return summary, plan, paths, labels, errors, unlabeled


def _payload(item, prompt, mode, image_b64, sheet):
    payload = {'image': image_b64, 'candidates': input_evidence.request_candidates(item), 'prompt': prompt}
    if mode == 'legacy':
        return payload
    payload['view_mode'] = mode
    if mode == 'card_references_v1':
        payload['reference_sheet'] = sheet
    return payload


def _call(payload, mode):
    started = time.perf_counter()
    if mode == 'legacy':
        result = worker.execute(payload)
    else:
        result = worker.execute_view_experiment(payload)
    return result, round(time.perf_counter() - started, 3)


def _scored_labels(plan, labels, unlabeled):
    skipped = set(unlabeled)
    scored = {}
    for item in plan['scored_units']:
        for candidate in item['candidates']:
            key = candidate['key']
            if key in skipped or key not in labels:
                continue
            scored[key] = labels[key]
    return scored


def _label_digests(scored_labels):
    return {split: view_protocol.labels_digest(scored_labels, split) for split in ('dev', 'holdout')}


def execute_budget(out: Path, assets: Path, annotated_root, gold_dir, max_calls: int):
    summary, plan, paths, labels, errors, unlabeled = build_plan(assets, annotated_root, gold_dir)
    if errors:
        raise SystemExit('预检失败：' + '；'.join(errors[:8]))
    label_digests = _label_digests(_scored_labels(plan, labels, unlabeled))
    if max_calls != plan['call_budget']:
        raise SystemExit(f'调用预算 {max_calls} 与清单 {plan["call_budget"]} 不一致，拒绝开跑')
    binary = os.environ.get('CODEX_REVIEW_BINARY', '')
    home = Path(os.environ.get('CODEX_REVIEW_HOME', ''))
    if not binary or not Path(binary).is_file() or not (home / 'auth.json').is_file():
        raise SystemExit('未找到已授权的评测 CLI 或凭据目录，拒绝改用其他通道')
    sheet, sheet_sha = reference_sheet_b64(plan, paths)
    summary['reference_sheet_sha256'] = sheet_sha
    summary['cli_binary_sha256'] = sha256_file(Path(binary))
    _write_json(out / 'plan.json', summary)
    calls = 0
    fast_failures = 0
    for round_name, mode, _role in ROUNDS:
        for item in plan['scored_units']:
            destination = out / 'calls' / round_name / f'{item["unit"]}.json'
            if destination.is_file():
                calls += 1
                continue
            if calls >= max_calls:
                raise SystemExit('调用次数达到预先固定的预算，已停止')
            candidates = input_evidence.request_candidates(item)
            prompt, _ = prompts_for(candidates, sheet, mode)
            leaks = person_candidate.prompt_contains_audit_secrets(prompt, plan['references'], labels)
            if leaks:
                raise SystemExit('提示词含有参考例编号或金标，已停止且未继续调用')
            image_b64 = base64.b64encode(paths[item['unit']].read_bytes()).decode('ascii')
            payload = _payload(item, prompt, mode, image_b64, sheet)
            record = {'unit': item['unit'], 'split': item['split'], 'group': item['group'],
                      'mode': mode, 'round': round_name, 'source_sha256': item['sha256']}
            if item['split'] in label_digests:
                record['labels_sha256'] = label_digests[item['split']]
            started = time.perf_counter()
            try:
                result, elapsed = _call(payload, mode)
                decisions = {}
                for candidate, row in zip(item['candidates'], result['results']):
                    decisions[candidate['key']] = row['decision']
                record.update({'status': 'ok', 'elapsed_s': elapsed, 'decisions': decisions,
                               'evidence': project_evidence(result['input_evidence']),
                               'execution_id': result.get('execution_id'),
                               'requested_model': result.get('requested_model')})
                if isinstance(result.get('model_config'), dict):
                    record['model_config'] = result['model_config']
                fast_failures = 0
            except Exception as exc:
                elapsed = round(time.perf_counter() - started, 3)
                record.update({'status': 'failed', 'elapsed_s': elapsed, 'decisions': {},
                               'error': type(exc).__name__ + ': ' + str(exc)[:300]})
                fast_failures = fast_failures + 1 if elapsed < 15 and 'CLI failed' in str(exc) else 0
            _write_json(destination, record)
            calls += 1
            print(f'PROGRESS {calls}/{max_calls} {round_name} {item["unit"]} {record["status"]}', flush=True)
            if fast_failures >= 2:
                print('FAILED channel rejected two calls; remaining units were not called', flush=True)
                raise SystemExit(2)
    return calls


def _load_rounds(out: Path, plan):
    """只搬运调用记录里已经写下的模型和标签摘要。各轮不一致或有缺口时不补造。"""
    grouped = {role: [] for _, _, role in ROUNDS}
    expected = {}
    for item in plan['scored_units']:
        expected[item['split']] = expected.get(item['split'], 0) + 1
    for round_name, _mode, role in ROUNDS:
        results, elapsed, evidence = {}, {}, {}
        configs = []
        config_gaps = False
        digests = {}
        for item in plan['scored_units']:
            path = out / 'calls' / round_name / f'{item["unit"]}.json'
            if not path.is_file():
                continue
            record = json.loads(path.read_text(encoding='utf-8'))
            if record.get('evidence'):
                evidence[item['unit']] = record['evidence']
                config = record.get('model_config')
                if isinstance(config, dict):
                    configs.append(config)
                else:
                    config_gaps = True
            digest = record.get('labels_sha256')
            if isinstance(digest, str):
                digests.setdefault(item['split'], []).append(digest)
            for key, decision in (record.get('decisions') or {}).items():
                results[key] = decision
                if record.get('elapsed_s') is not None:
                    elapsed[key] = record['elapsed_s']
        run = {'round': round_name, 'results': results, 'elapsed_s': elapsed, 'evidence_by_unit': evidence}
        if configs and not config_gaps and all(item == configs[0] for item in configs):
            run['model_config'] = configs[0]
        agreed = {}
        for split, values in digests.items():
            if expected.get(split) and len(values) == expected[split] and all(item == values[0] for item in values):
                agreed[split] = values[0]
        if agreed:
            run['labels_sha256'] = agreed
        grouped[role].append(run)
    return grouped


def write_report(out: Path, assets: Path, annotated_root, gold_dir):
    summary, plan, _paths, labels, errors, unlabeled = build_plan(assets, annotated_root, gold_dir)
    if errors:
        raise SystemExit('汇总时预检失败：' + '；'.join(errors[:8]))
    grouped = _load_rounds(out, plan)
    scored_labels = _scored_labels(plan, labels, unlabeled)
    reports = {}
    for split in ('dev', 'holdout'):
        reports[split] = {}
        for factor, role in (('person_view', 'candidate-a'), ('person_card_references', 'candidate-b')):
            reports[split][factor] = view_protocol.evaluate(
                grouped[role], grouped['baseline'], scored_labels, factor=factor, split=split, exposure=EXPOSURE)
    factor_errors = []
    for role, factor in (('candidate-a', 'person_view'), ('candidate-b', 'person_card_references')):
        base_units = grouped['baseline'][0].get('evidence_by_unit') or {}
        cand_units = grouped[role][0].get('evidence_by_unit') or {}
        for unit in sorted(set(base_units) & set(cand_units)):
            found = view_protocol.factor_errors(base_units[unit], cand_units[unit], factor)
            factor_errors.extend(f'{role} {unit} {item}' for item in found)
    paired = {}
    for split in ('dev', 'holdout'):
        paired[split] = {
            'candidate-a': view_protocol.pair_boxes(grouped['candidate-a'], grouped['baseline'], scored_labels, split),
            'candidate-b': view_protocol.pair_boxes(grouped['candidate-b'], grouped['baseline'], scored_labels, split),
        }
    report = {
        'plan': json.loads((out / 'plan.json').read_text(encoding='utf-8')) if (out / 'plan.json').is_file() else summary,
        'factor_errors': factor_errors,
        'splits': {
            split: {
                factor: {key: value[factor][key] for key in (
                    'checks_passed', 'no_regression', 'improvement', 'measurement_passed', 'release_allowed',
                    'independence', 'errors', 'person', 'baseline_person', 'flips', 'elapsed_s')}
                for factor in value
            } for split, value in reports.items()
        },
        'paired': paired,
        'exposure': EXPOSURE,
    }
    _write_json(out / 'report.json', report)
    return report


def write_label_template(out: Path, assets: Path, unlabeled=()):
    items, _labels = load_assets(assets)
    exposed = sorted(item['unit'] for item in items if item['split'] in {'dev', 'holdout'}
                     and any(person_view.is_person(c['class_name']) for c in item['candidates']))
    _write_json(out / 'independent-label-template.json', {
        'purpose': '独立人员正负样本仍缺。本清单不填标签，也不能用冻结集里的历史结论补齐。',
        'fields': ['event_or_flight_id', 'candidate_id', 'class_name', 'bbox', 'label', 'reviewer'],
        'rows': [{'candidate_id': key, 'class_name': '人员', 'label': '', 'reviewer': ''} for key in unlabeled],
        'unlabeled_frozen_boxes': list(unlabeled),
        'excluded_exposed_units': exposed,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('preflight', 'execute', 'report'))
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--annotated-root', type=Path)
    parser.add_argument('--gold-dir', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-calls', type=int)
    args = parser.parse_args()
    if args.command == 'preflight':
        summary, _plan, _paths, _labels, errors, _unlabeled = build_plan(args.assets, args.annotated_root, args.gold_dir)
        args.out.mkdir(parents=True, exist_ok=True)
        _write_json(args.out / 'preflight.json', summary)
        write_label_template(args.out, args.assets, _unlabeled)
        print(json.dumps({key: summary[key] for key in (
            'scored_units', 'scored_boxes', 'call_budget', 'excluded_groups', 'nonperson_checked',
            'nonperson_unchecked', 'errors')}, ensure_ascii=False))
        if errors:
            raise SystemExit(1)
        return
    if args.command == 'execute':
        if args.max_calls is None:
            raise SystemExit('execute 必须显式给出 --max-calls')
        try:
            calls = execute_budget(args.out, args.assets, args.annotated_root, args.gold_dir, args.max_calls)
        finally:
            if (args.out / 'calls').exists():
                write_report(args.out, args.assets, args.annotated_root, args.gold_dir)
        print(f'DONE calls={calls}', flush=True)
        return
    write_report(args.out, args.assets, args.annotated_root, args.gold_dir)


if __name__ == '__main__':
    main()
