"""SSH executor for remote job nodes (paramiko).

Key auth first (GPU_SSH_KEY or the ssh agent), GPU_SSH_PASSWORD as fallback.
Commands executed here come from fixed pipeline templates, never from LLM
output. Secrets stay in env, never in RunState or checkpoints.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import paramiko


@dataclass
class ExecResult:
    rc: int
    stdout: str
    stderr: str


class RemoteError(RuntimeError):
    pass


class RemoteExecutor:
    def __init__(
        self,
        host: str,
        user: str,
        *,
        key_path: str | None = None,
        password: str | None = None,
        port: int = 22,
        connect_timeout: float = 10.0,
    ):
        self._host = host
        self._user = user
        self._port = port
        self._key_path = key_path
        self._password = password
        self._connect_timeout = connect_timeout
        self._client: paramiko.SSHClient | None = None

    @classmethod
    def from_env(cls) -> "RemoteExecutor":
        host = os.environ.get("GPU_SSH_HOST")
        user = os.environ.get("GPU_SSH_USER")
        if not host or not user:
            raise RemoteError("GPU_SSH_HOST / GPU_SSH_USER are not set")
        return cls(
            host,
            user,
            key_path=os.environ.get("GPU_SSH_KEY") or None,
            password=os.environ.get("GPU_SSH_PASSWORD") or None,
        )

    def _connect(self) -> paramiko.SSHClient:
        transport = self._client.get_transport() if self._client is not None else None
        if transport is not None and transport.is_active():
            return self._client  # type: ignore[return-value]
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            self._host,
            port=self._port,
            username=self._user,
            key_filename=self._key_path,
            password=self._password,
            allow_agent=True,
            look_for_keys=self._key_path is None,
            timeout=self._connect_timeout,
            banner_timeout=self._connect_timeout,
        )
        client.get_transport().set_keepalive(30)  # type: ignore[union-attr]
        self._client = client
        return client

    def run(self, command: str, timeout: float = 60.0) -> ExecResult:
        for attempt in range(2):
            client = self._connect()
            try:
                _, stdout, stderr = client.exec_command(command, timeout=timeout)
                rc = stdout.channel.recv_exit_status()
                return ExecResult(
                    rc=rc,
                    stdout=stdout.read().decode("utf-8", "replace"),
                    stderr=stderr.read().decode("utf-8", "replace"),
                )
            except (paramiko.SSHException, OSError):
                self.close()  # stale transport: reconnect once, then give up
                if attempt == 1:
                    raise
        raise RemoteError(f"unreachable retry state for command: {command!r}")

    def run_ok(self, command: str, timeout: float = 60.0) -> str:
        result = self.run(command, timeout=timeout)
        if result.rc != 0:
            raise RemoteError(
                f"remote command failed rc={result.rc}: {command!r}\n{result.stderr.strip()}"
            )
        return result.stdout

    def get_file(self, remote_path: str, local_path: Path) -> None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect().open_sftp() as sftp:
            sftp.get(remote_path, str(local_path))

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
