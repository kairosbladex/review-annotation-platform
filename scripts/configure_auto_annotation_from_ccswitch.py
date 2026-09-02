#!/usr/bin/env python3
"""Copy non-sensitive Codex routing plus the selected API key into a local env file.

The generated file is intentionally gitignored and mode 0600.  The key is
never printed; use ``--dry-run`` to inspect only the selected provider/model.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import stat
import tomllib
from pathlib import Path


def _provider(db_path: Path, provider_name: str) -> tuple[dict, dict]:
    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            "SELECT name, settings_config, meta FROM providers "
            "WHERE app_type='codex' AND lower(name)=lower(?) "
            "ORDER BY is_current DESC LIMIT 1",
            (provider_name,),
        ).fetchone()
    if row is None:
        raise RuntimeError(f"CC Switch Codex provider not found: {provider_name}")
    try:
        settings = json.loads(row[1])
        config = tomllib.loads(settings.get("config", ""))
    except (TypeError, ValueError, tomllib.TOMLDecodeError) as exc:
        raise RuntimeError("CC Switch provider configuration is invalid") from exc
    return settings, config


def build_env(db_path: Path, provider_name: str) -> tuple[str, str]:
    settings, config = _provider(db_path, provider_name)
    provider_key = str(config.get("model_provider") or provider_name)
    provider = config.get("model_providers", {}).get(provider_key, {})
    base_url = str(provider.get("base_url") or "").rstrip("/")
    model = str(config.get("model") or "gpt-5.6-luna")
    actor = str(provider.get("http_headers", {}).get("x-openai-actor-authorization") or "local-image-extension")
    auth = settings.get("auth") or {}
    api_key = str(auth.get("OPENAI_API_KEY") or "")
    if not base_url or not api_key:
        raise RuntimeError("CC Switch provider has no usable base_url or API key")
    env = "\n".join(
        [
            f"AUTO_ANNOTATION_API_KEY={api_key}",
            f"AUTO_ANNOTATION_BASE_URL={base_url}",
            f"AUTO_ANNOTATION_MODEL={model}",
            "AUTO_ANNOTATION_ENDPOINT=/responses",
            "AUTO_ANNOTATION_REASONING_EFFORT=low",
            "AUTO_ANNOTATION_TIMEOUT_SECONDS=120",
            "AUTO_ANNOTATION_VERIFY_TLS=true",
            "AUTO_ANNOTATION_USE_CODEX_CLI=true",
            "AUTO_ANNOTATION_CODEX_BINARY=codex",
            f"AUTO_ANNOTATION_ACTOR_AUTHORIZATION={actor}",
            "",
        ]
    )
    return env, model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path.home() / ".cc-switch" / "cc-switch.db")
    parser.add_argument("--provider", default="zhuoyi")
    parser.add_argument("--output", type=Path, default=Path(".env.auto-annotation.local"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="replace an existing output file")
    args = parser.parse_args()

    env, model = build_env(args.db.expanduser(), args.provider)
    if args.dry_run:
        print(f"provider={args.provider} model={model} output={args.output}")
        return 0
    output = args.output.expanduser()
    if output.exists() and not args.force:
        raise SystemExit(f"refusing to replace existing file; pass --force: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(env, encoding="utf-8")
    os.chmod(output, stat.S_IRUSR | stat.S_IWUSR)
    print(f"wrote {output} (mode 0600; credential value not displayed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
