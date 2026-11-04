"""Local stand-in for the DeepInfra sandbox HTTP API, used by the sandbox tests.

It speaks the same wire format the ``deepinfra`` SDK uses (see
``deepinfra/sandbox_api.py`` and ``deepinfra/_streaming.py``):

- ``POST /v1/sandboxes`` and ``GET``/``DELETE /v1/sandboxes/{id}``
- ``POST /v1/sandboxes/{id}/exec`` streaming NDJSON ``{"stdout": ...}``,
  ``{"stderr": ...}`` and a terminal ``{"returncode": n}`` or ``{"error": ...}``
- ``GET``/``PUT /v1/sandboxes/{id}/fs/content?path=...``, which like the real
  service rejects any path outside ``/workspace`` with HTTP 400

Commands run as host subprocesses. ``/workspace`` is mapped to a temporary
directory (in fs paths and in exec arguments) so the workspace-only fs rule can
be enforced while commands see the same paths. This is a test double, not a
sandbox: only point it at test code.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

_PREFIX = "/v1/sandboxes"
_WORKSPACE = "/workspace"


class SandboxEmulator:
    """Run the emulator on a background thread: ``with SandboxEmulator() as e``."""

    def __init__(self) -> None:
        self.workspace_root = tempfile.mkdtemp(prefix="di-workspace-")
        self.sandboxes: dict[str, dict] = {}
        self.exec_calls: list[list[str]] = []
        self.fs_calls: list[tuple[str, str]] = []
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def __enter__(self) -> SandboxEmulator:
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._server.shutdown()
        self._server.server_close()
        shutil.rmtree(self.workspace_root, ignore_errors=True)

    def host_path(self, path: str) -> str:
        return self.workspace_root + path[len(_WORKSPACE) :]

    def _map_arg(self, arg: str) -> str:
        return arg.replace(_WORKSPACE, self.workspace_root)

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        emulator = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: object) -> None:  # keep test output quiet
                pass

            def _json(self, status: int, payload: object) -> None:
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _body(self) -> bytes:
                length = int(self.headers.get("Content-Length") or 0)
                return self.rfile.read(length) if length else b""

            def _route(self) -> tuple[str | None, str, dict[str, list[str]]]:
                url = urlparse(self.path)
                parts = url.path[len(_PREFIX) :].strip("/").split("/", 1)
                sandbox_id = unquote(parts[0]) if parts[0] else None
                rest = parts[1] if len(parts) > 1 else ""
                return sandbox_id, rest, parse_qs(url.query)

            def _info(self, sandbox_id: str) -> dict:
                return {
                    "sandbox_id": sandbox_id,
                    "plan": "medium",
                    "state": emulator.sandboxes[sandbox_id]["state"],
                    "tags": {},
                }

            def do_POST(self) -> None:  # noqa: N802
                sandbox_id, rest, _ = self._route()
                payload = json.loads(self._body() or b"{}")
                if sandbox_id is None:
                    new_id = f"sb-{len(emulator.sandboxes) + 1}"
                    emulator.sandboxes[new_id] = {"state": "running"}
                    return self._json(200, {"sandbox_id": new_id})
                if sandbox_id not in emulator.sandboxes:
                    return self._json(404, {"error": "not found"})
                if rest == "exec":
                    return self._exec(payload)
                return self._json(404, {"error": "unknown route"})

            def _exec(self, payload: dict) -> None:
                command = [emulator._map_arg(a) for a in payload["command"]]
                emulator.exec_calls.append(payload["command"])
                timeout = payload.get("timeout_seconds") or None
                try:
                    proc = subprocess.run(  # noqa: S603 - test double
                        command,
                        capture_output=True,
                        timeout=timeout,
                        check=False,
                    )
                    events = []
                    if proc.stdout:
                        events.append({"stdout": proc.stdout.decode(errors="replace")})
                    if proc.stderr:
                        events.append({"stderr": proc.stderr.decode(errors="replace")})
                    events.append({"returncode": proc.returncode})
                except subprocess.TimeoutExpired:
                    events = [{"error": f"command timed out after {timeout}s"}]
                body = b"".join(json.dumps(e).encode() + b"\n" for e in [{}, *events])
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _fs_path(self, query: dict[str, list[str]]) -> str | None:
                path = query.get("path", [""])[0]
                if path != _WORKSPACE and not path.startswith(_WORKSPACE + "/"):
                    self._json(400, {"error": "ERR_PATH_OUTSIDE_WORKSPACE"})
                    return None
                return emulator.host_path(path)

            def do_GET(self) -> None:  # noqa: N802
                sandbox_id, rest, query = self._route()
                if sandbox_id not in emulator.sandboxes:
                    return self._json(404, {"error": "not found"})
                if rest == "":
                    return self._json(200, self._info(sandbox_id))
                if rest == "fs/content":
                    emulator.fs_calls.append(("read", query.get("path", [""])[0]))
                    host = self._fs_path(query)
                    if host is None:
                        return
                    if not os.path.isfile(host):
                        return self._json(404, {"error": "ERR_NOT_FOUND"})
                    with open(host, "rb") as f:
                        data = f.read()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                return self._json(404, {"error": "unknown route"})

            def do_PUT(self) -> None:  # noqa: N802
                sandbox_id, rest, query = self._route()
                data = self._body()
                if sandbox_id not in emulator.sandboxes or rest != "fs/content":
                    return self._json(404, {"error": "not found"})
                emulator.fs_calls.append(("write", query.get("path", [""])[0]))
                host = self._fs_path(query)
                if host is None:
                    return
                os.makedirs(os.path.dirname(host), exist_ok=True)
                with open(host, "wb") as f:
                    f.write(data)
                return self._json(200, {})

            def do_DELETE(self) -> None:  # noqa: N802
                sandbox_id, _, _ = self._route()
                if emulator.sandboxes.pop(sandbox_id or "", None) is None:
                    return self._json(404, {"error": "not found"})
                return self._json(200, {})

        return Handler
