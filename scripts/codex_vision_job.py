"""Single offline vision job inside a resource-limited disposable container.

Read a JSON request from stdin; credentials never enter argv or persistent files.
The caller mounts only /job (input images) and /tools/codex, both read-only.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time


def validate_request(request, image_root=Path("/job")):
    if request.get("task") not in {"smoke", "review", "geolocation"}:
        raise ValueError("unsupported task")
    images = request.get("images", [])
    if not 1 <= len(images) <= 3:
        raise ValueError("one to three images required")
    for name in images:
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError("image must be a direct child of the job directory")
        p = image_root / name
        if p.is_symlink() or not p.is_file() or p.stat().st_size > 5_000_000:
            raise ValueError("missing or oversized image")
    if not isinstance(request.get("prompt"), str) or len(request["prompt"]) > 12000:
        raise ValueError("invalid prompt")
    if not request.get("api_key"):
        raise ValueError("credential required")
    return images


def main():
    request = json.loads(sys.stdin.buffer.read(100_000))
    images = validate_request(request)
    key = request.pop("api_key")
    home = Path("/tmp/codex-home")
    home.mkdir(mode=0o700)
    model = request.get("model", "gpt-5.6-sol")
    config = (
        'model_provider = "company"\n'
        f'model = {json.dumps(model)}\n'
        'model_reasoning_effort = "low"\n'
        'approval_policy = "never"\n'
        'sandbox_mode = "read-only"\n'
        'web_search = "disabled"\n'
        '[features]\nshell_tool = false\n'
        '[model_providers.company]\n'
        'name = "Company relay"\n'
        'base_url = "https://sub.butterfly2sea.us/v1"\n'
        'wire_api = "responses"\n'
        'env_key = "VISION_RELAY_API_KEY"\n'
        'request_max_retries = 0\nstream_max_retries = 0\n'
    )
    (home / "config.toml").write_text(config)
    schema = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "decision": {"type": "string", "enum": [
                "true", "false", "uncertain", "A", "B", "neither", "red_square_blue_circle"]},
            "reason": {"type": "string"},
        }, "required": ["decision", "reason"],
    }
    (home / "schema.json").write_text(json.dumps(schema))
    output = home / "answer.json"
    command = ["/tools/codex", "exec", "--ephemeral", "--skip-git-repo-check",
               "--sandbox", "read-only", "--json", "-C", "/job",
               "--output-schema", str(home / "schema.json"), "-o", str(output)]
    for name in images:
        command.extend(["-i", "/job/" + name])
    command.append("-")
    env = {**os.environ, "CODEX_HOME": str(home), "VISION_RELAY_API_KEY": key}
    start = time.monotonic()
    result = {"task": request["task"], "requested_model": model,
              "business_writes": 0, "device_actions": 0}
    try:
        completed = subprocess.run(command, input=request["prompt"], text=True,
                                   capture_output=True, env=env, timeout=150)
        result.update(exit_code=completed.returncode)
        events = []
        for line in completed.stdout.splitlines():
            try:
                events.append(json.loads(line))
            except ValueError:
                pass
        result["usage"] = [e.get("usage") for e in events if e.get("type") == "turn.completed"]
        result["tool_calls"] = [e.get("item", {}).get("type") for e in events
                                if e.get("type") == "item.started"]
        if completed.returncode == 0 and output.is_file():
            answer = json.loads(output.read_text())
            allowed = {"smoke": {"red_square_blue_circle", "uncertain"},
                       "review": {"true", "false", "uncertain"},
                       "geolocation": {"A", "B", "neither", "uncertain"}}
            if answer.get("decision") not in allowed[request["task"]] or not isinstance(answer.get("reason"), str):
                raise ValueError("invalid decision contract")
            if any(t not in {"reasoning", "agent_message"} for t in result["tool_calls"]):
                raise ValueError("unexpected tool use")
            result["answer"] = answer
        else:
            result["error"] = (completed.stderr + "\n" + completed.stdout)[-6000:]
    except subprocess.TimeoutExpired:
        result["error"] = "job exceeded 150 seconds; container exits without retry"
    except (ValueError, OSError) as exc:
        result["error"] = str(exc)
    result["elapsed_seconds"] = round(time.monotonic() - start, 2)
    print(json.dumps(result, ensure_ascii=False).replace(key, "[REDACTED]"))
    return 0 if "answer" in result else 1


if __name__ == "__main__":
    raise SystemExit(main())
