#!/usr/bin/env bash
# Run on AI-lingxi. Stdin is one JSON request including the per-call credential.
set -euo pipefail
root=/data/workspace/wangxiaolin/codex-vision-pilot
binary=/home/user/.local/opt/codex-0.155.1/codex-x86_64-unknown-linux-musl
image=python@sha256:8d9d0b8bcf6506481eae4907c18f5e3e7902e629f5f6d684f9e7c32e85e3ddf0
job=${1:?job directory name required}
[[ "$job" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 64
[[ -d "$root/jobs/$job" ]] || exit 66
[[ ! -L "$root/jobs/$job" ]] || exit 64
expected=0753dfe1d8b87a52436deb13eb1c549661ef4c84fee2c5aa688385eebeccb761
actual=$(sha256sum "$binary")
[[ "${actual%% *}" == "$expected" ]] || { echo 'CLI binary changed' >&2; exit 65; }
exec 9>"$root/worker.lock"
flock -n 9 || { echo 'worker busy' >&2; exit 75; }
available_kb=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
total_kb=$(awk '/MemTotal:/ {print $2}' /proc/meminfo)
(( available_kb > total_kb * 15 / 100 && available_kb > 4194304 )) || exit 69
exec docker run --rm -i --init --name agent-codex-vision-job \
  --label owner=agent --label purpose=codex-vision-pilot \
  --memory=2g --memory-swap=2g --cpus=2 --pids-limit=128 \
  --user 1000:1000 --cap-drop=ALL --security-opt=no-new-privileges \
  --read-only --tmpfs /tmp:rw,nosuid,nodev,size=256m,mode=1777 \
  --mount "type=bind,src=$binary,dst=/tools/codex,readonly" \
  --mount "type=bind,src=$root/jobs/$job,dst=/job,readonly" \
  --mount "type=bind,src=$root/codex_vision_job.py,dst=/runner.py,readonly" \
  "$image" python /runner.py
