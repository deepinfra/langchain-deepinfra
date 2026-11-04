"""DeepInfra Sandboxes backend for LangChain Deep Agents.

Lets a Deep Agent run shell commands and edit files inside a DeepInfra sandbox
(an isolated microVM) instead of on the host:

    from deepagents import create_deep_agent
    from deepinfra import Sandbox

    from langchain_deepinfra import DeepInfraSandbox

    sandbox = Sandbox.create(plan="medium", timeout="30m")
    backend = DeepInfraSandbox(sandbox=sandbox)
    agent = create_deep_agent(model=..., backend=backend)
    ...
    sandbox.terminate()

Requires the ``sandbox`` extra: ``pip install "langchain-deepinfra[sandbox]"``.
"""

from __future__ import annotations

import shlex
import uuid
from collections.abc import Iterable
from typing import TYPE_CHECKING

try:
    from deepagents.backends.protocol import (
        ExecuteResponse,
        FileDownloadResponse,
        FileUploadResponse,
    )
    from deepagents.backends.sandbox import BaseSandbox
    from deepinfra import APIConnectionError, APIStatusError, SandboxExecError
except ImportError as e:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "DeepInfraSandbox needs the optional sandbox dependencies. Install them "
        'with: pip install "langchain-deepinfra[sandbox]"'
    ) from e

if TYPE_CHECKING:
    from deepinfra import ExecResult, Sandbox

# DeepInfra's fs endpoints only accept paths under /workspace, so files bound
# for anywhere else are staged here and moved into place with one exec call.
_STAGING_DIR = "/workspace/.deepagents-transfer"
_DEFAULT_TIMEOUT = 30 * 60
# Exit code reported when the sandbox kills a command or the exec stream
# breaks, matching the convention of coreutils `timeout`.
_EXEC_FAILURE_EXIT_CODE = 124
_STATUS_OK = "ok"


class DeepInfraSandbox(BaseSandbox):
    """Deep Agents sandbox backend that runs inside a DeepInfra sandbox.

    The backend wraps an existing :class:`deepinfra.Sandbox`; creating,
    stopping and terminating the sandbox stays with the caller, so one sandbox
    can be shared across agents or reused between runs.

    Commands run with ``bash -c``. File transfers go through the sandbox's fs
    API. Because that API only accepts paths under ``/workspace``, uploads and
    downloads for other absolute paths are staged under
    ``/workspace/.deepagents-transfer`` and moved with a single command per
    batch, so every path an agent can see is also transferable.
    """

    def __init__(
        self,
        *,
        sandbox: Sandbox,
        default_timeout: int = _DEFAULT_TIMEOUT,
    ) -> None:
        """Create a backend bound to a running DeepInfra sandbox.

        Args:
            sandbox: A running sandbox, e.g. from ``deepinfra.Sandbox.create()``
                or ``deepinfra.Sandbox.from_id()``.
            default_timeout: Seconds a command may run when ``execute`` is
                called without a timeout. Defaults to 30 minutes.
        """
        self._sandbox = sandbox
        self._default_timeout = default_timeout

    @property
    def id(self) -> str:
        """Return the DeepInfra sandbox id."""
        return self._sandbox.id

    # -- execution ---------------------------------------------------------

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        """Run a shell command in the sandbox.

        Args:
            command: Shell command string, run with ``bash -c``.
            timeout: Maximum seconds to wait. ``None`` uses the backend default.

        Returns:
            Combined stdout/stderr and the exit code. If the sandbox kills the
            command (for example on timeout) the exit code is 124.
        """
        try:
            result = self._sandbox.exec(
                "bash", "-c", command, timeout=self._timeout(timeout)
            )
        except SandboxExecError as e:
            return ExecuteResponse(
                output=f"[DeepInfra: {e}]", exit_code=_EXEC_FAILURE_EXIT_CODE
            )
        return _to_response(result)

    async def aexecute(
        self,
        command: str,
        *,
        timeout: int | None = None,  # noqa: ASYNC109 - forwarded to the API
    ) -> ExecuteResponse:
        """Async version of :meth:`execute` using the SDK's native async client."""
        try:
            result = await self._sandbox.aexec(
                "bash", "-c", command, timeout=self._timeout(timeout)
            )
        except SandboxExecError as e:
            return ExecuteResponse(
                output=f"[DeepInfra: {e}]", exit_code=_EXEC_FAILURE_EXIT_CODE
            )
        return _to_response(result)

    def _timeout(self, timeout: int | None) -> int:
        return self._default_timeout if timeout is None else timeout

    # -- file transfer -----------------------------------------------------

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload files into the sandbox, creating parent directories.

        Partial success is reported per file; relative paths are rejected with
        ``invalid_path`` without touching the sandbox.
        """
        responses = [
            FileUploadResponse(path=path, error=_path_error(path)) for path, _ in files
        ]
        pending = [i for i, r in enumerate(responses) if r.error is None]
        if not pending:
            return responses

        batch = uuid.uuid4().hex
        staged = {i: f"{_STAGING_DIR}/{batch}-{i}" for i in pending}
        try:
            self._sandbox.exec("mkdir", "-p", _STAGING_DIR)
            for i in pending:
                self._sandbox.fs.write(staged[i], files[i][1])
            statuses = self._run_batch(_move_script(staged, files))
        except (APIStatusError, APIConnectionError, SandboxExecError) as e:
            for i in pending:
                responses[i].error = f"DeepInfra sandbox error: {e}"
            self._cleanup(staged.values())
            return responses

        for i in pending:
            status = statuses.get(i, "unknown upload failure")
            if status != _STATUS_OK:
                responses[i].error = status
        self._cleanup(staged.values())
        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download files from the sandbox.

        Missing files, directories and unreadable files are reported per path
        as ``file_not_found``, ``is_directory`` and ``permission_denied``.
        """
        responses = [
            FileDownloadResponse(path=path, content=None, error=_path_error(path))
            for path in paths
        ]
        pending = [i for i, r in enumerate(responses) if r.error is None]
        if not pending:
            return responses

        batch = uuid.uuid4().hex
        staged = {i: f"{_STAGING_DIR}/{batch}-{i}" for i in pending}
        try:
            statuses = self._run_batch(_copy_script(staged, paths))
            for i in pending:
                status = statuses.get(i, "unknown download failure")
                if status == _STATUS_OK:
                    responses[i].content = self._sandbox.fs.read(staged[i])
                else:
                    responses[i].error = status
        except (APIStatusError, APIConnectionError, SandboxExecError) as e:
            for i in pending:
                if responses[i].content is None and responses[i].error is None:
                    responses[i].error = f"DeepInfra sandbox error: {e}"
        self._cleanup(staged.values())
        return responses

    def _run_batch(self, script: str) -> dict[int, str]:
        """Run a transfer script and parse its ``<index> <status>`` lines."""
        result = self._sandbox.exec("bash", "-c", script, timeout=self._default_timeout)
        statuses: dict[int, str] = {}
        for line in result.stdout.splitlines():
            index, _, status = line.partition(" ")
            if index.isdigit() and status:
                statuses[int(index)] = status
        return statuses

    def _cleanup(self, staged_paths: Iterable[str]) -> None:
        paths = list(staged_paths)
        if not paths:
            return
        try:
            self._sandbox.exec("rm", "-f", "--", *paths)
        except (APIStatusError, APIConnectionError, SandboxExecError):
            pass  # best effort; the staging dir only holds transfer leftovers


def _to_response(result: ExecResult) -> ExecuteResponse:
    output = result.stdout
    if result.stderr:
        output = f"{output}\n{result.stderr}" if output else result.stderr
    return ExecuteResponse(output=output, exit_code=result.returncode, truncated=False)


def _path_error(path: str) -> str | None:
    return None if path.startswith("/") else "invalid_path"


def _move_script(staged: dict[int, str], files: list[tuple[str, bytes]]) -> str:
    """Shell script that moves staged uploads to their targets."""
    lines = []
    for i, src in staged.items():
        dst = shlex.quote(files[i][0])
        lines.append(
            f"if [ -d {dst} ]; then echo '{i} is_directory'; "
            f'elif mkdir -p -- "$(dirname -- {dst})" 2>/dev/null '
            f"&& mv -f -- {shlex.quote(src)} {dst} 2>/dev/null; "
            f"then echo '{i} {_STATUS_OK}'; "
            f"else echo '{i} permission_denied'; fi"
        )
    return "\n".join(lines)


def _copy_script(staged: dict[int, str], paths: list[str]) -> str:
    """Shell script that classifies each path and copies readable files out."""
    lines = []
    for i, dst in staged.items():
        src = shlex.quote(paths[i])
        lines.append(
            f"if [ ! -e {src} ]; then echo '{i} file_not_found'; "
            f"elif [ -d {src} ]; then echo '{i} is_directory'; "
            f"elif [ ! -r {src} ]; then echo '{i} permission_denied'; "
            f"elif cp -- {src} {shlex.quote(dst)} 2>/dev/null; "
            f"then echo '{i} {_STATUS_OK}'; "
            f"else echo '{i} permission_denied'; fi"
        )
    return f"mkdir -p {_STAGING_DIR}\n" + "\n".join(lines)
