"""Prepare an opt-in overlay for the verified 8189 VL comparison router."""
from __future__ import annotations

import argparse
import ast
import hashlib
from pathlib import Path
import shutil

LEGACY_SHA256 = 'e7233688c6ac0aa0b74560594c96884b06cc8c5d4f2b9d4b7ac03e5743b9ff69'


def patch_router(source: bytes, expected_sha: str = LEGACY_SHA256) -> str:
    if hashlib.sha256(source).hexdigest() != expected_sha:
        raise ValueError('Legacy router digest changed; inspect the deployed version before preparing an overlay')
    text = source.decode('utf-8-sig')
    tree = ast.parse(text)
    names = {'_compare_pair': ('current', 'history', True),
             '_compare_baseline_compare_pair': ('baseline', 'compare', False)}
    edits = []
    for node in tree.body:
        if not isinstance(node, ast.AsyncFunctionDef) or node.name not in names:
            continue
        first, second, single = names[node.name]
        injected = f'''    if _env("VL_COMPARE_ENGINE", "legacy") == "registered":
        from history_compare_runtime.services.history_compare_legacy import ComparisonBusy, compare_legacy_pair
        try:
            return await asyncio.to_thread(
                compare_legacy_pair, {first}.image, {second}.image,
                one_to_many={single!r},
                baseline_is_current=_env("VL_COMPARE_BASELINE_IS_CURRENT", "false").lower() == "true",
                base_url=client.base_url, model=client.model,
                timeout=max(1.0, min(20.0, _env_float("VL_COMPARE_SEMANTIC_TIMEOUT", 20.0))),
            )
        except ComparisonBusy as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
'''
        edits.append((node.body[0].lineno - 1, injected))
    if len(edits) != len(names):
        raise ValueError('Expected comparison helpers were not found')
    lines = text.splitlines(keepends=True)
    for line, addition in sorted(edits, reverse=True):
        lines.insert(line, addition)
    patched = ''.join(lines)
    compile(patched, 'vl_compare.py', 'exec')
    return patched


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy-router', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    patched = patch_router(args.legacy_router.read_bytes())
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'vl_compare.py').write_text(patched, encoding='utf-8')
    root = Path(__file__).resolve().parents[1]
    runtime = args.output / 'history_compare_runtime'
    (runtime / 'services').mkdir(parents=True)
    (runtime / 'prompts').mkdir()
    for name in ('history_compare.py', 'history_compare_legacy.py'):
        shutil.copy2(root / 'backend/app/services' / name, runtime / 'services' / name)
    shutil.copy2(root / 'backend/app/prompts/history_compare.txt', runtime / 'prompts/history_compare.txt')
    print(f'Prepared overlay at {args.output}; source SHA256={LEGACY_SHA256}')


if __name__ == '__main__':
    main()
