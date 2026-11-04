"""Tests for DeepInfraSandbox against a local emulator of the sandbox API.

The emulator (``tests/sandbox_emulator.py``) speaks the SDK's wire format over
real HTTP, so these tests go through the actual ``deepinfra`` client. The
LangChain standard sandbox suite runs on top of it.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

pytest.importorskip("deepagents")  # requires Python 3.11+

from deepinfra import Sandbox  # noqa: E402
from deepinfra.clients.deepinfra import DeepInfraClient  # noqa: E402
from langchain_tests.integration_tests import SandboxIntegrationTests  # noqa: E402

from langchain_deepinfra import DeepInfraSandbox  # noqa: E402
from tests.sandbox_emulator import SandboxEmulator  # noqa: E402

_UPSTREAM_DRIFT = pytest.mark.xfail(
    reason="deepagents BaseSandbox and langchain-tests disagree on glob/write",
    strict=False,
)


@pytest.fixture(scope="module")
def emulator() -> Iterator[SandboxEmulator]:
    with SandboxEmulator() as emu:
        yield emu


def _sandbox(emulator: SandboxEmulator) -> Sandbox:
    client = DeepInfraClient(api_key="test-key", base_url=emulator.base_url)
    return Sandbox.create(client=client)


# The SDK keeps one httpx.AsyncClient per client object, bound to the event
# loop that first used it, so the class-scoped sandbox needs a class-scoped loop.
@pytest.mark.asyncio(loop_scope="class")
class TestDeepInfraSandboxStandard(SandboxIntegrationTests):
    @pytest.fixture(scope="class")
    def sandbox(self) -> Iterator[DeepInfraSandbox]:
        with SandboxEmulator() as emu:
            sandbox = _sandbox(emu)
            try:
                yield DeepInfraSandbox(sandbox=sandbox)
            finally:
                sandbox.terminate()

    # langchain-tests 1.1.9 expects relative glob paths and refusing to
    # overwrite in write(); deepagents 0.7.32's BaseSandbox (which implements
    # glob/write for every sandbox backend) returns absolute paths and
    # overwrites. A plain local BaseSandbox subclass fails these the same way.

    @_UPSTREAM_DRIFT
    def test_glob(self, sandbox_backend, sandbox_test_root) -> None:
        super().test_glob(sandbox_backend, sandbox_test_root)

    @_UPSTREAM_DRIFT
    def test_glob_basic_pattern(self, sandbox_backend, sandbox_test_root) -> None:
        super().test_glob_basic_pattern(sandbox_backend, sandbox_test_root)

    @_UPSTREAM_DRIFT
    def test_glob_with_directories(self, sandbox_backend, sandbox_test_root) -> None:
        super().test_glob_with_directories(sandbox_backend, sandbox_test_root)

    @_UPSTREAM_DRIFT
    def test_glob_hidden_files_explicitly(
        self, sandbox_backend, sandbox_test_root
    ) -> None:
        super().test_glob_hidden_files_explicitly(sandbox_backend, sandbox_test_root)

    @_UPSTREAM_DRIFT
    def test_glob_with_character_class(
        self, sandbox_backend, sandbox_test_root
    ) -> None:
        super().test_glob_with_character_class(sandbox_backend, sandbox_test_root)

    @_UPSTREAM_DRIFT
    def test_glob_with_question_mark(self, sandbox_backend, sandbox_test_root) -> None:
        super().test_glob_with_question_mark(sandbox_backend, sandbox_test_root)

    @_UPSTREAM_DRIFT
    def test_write_existing_file_fails(
        self, sandbox_backend, sandbox_test_root
    ) -> None:
        super().test_write_existing_file_fails(sandbox_backend, sandbox_test_root)


@pytest.fixture
def backend(emulator: SandboxEmulator) -> DeepInfraSandbox:
    emulator.exec_calls.clear()
    emulator.fs_calls.clear()
    return DeepInfraSandbox(sandbox=_sandbox(emulator))


def test_id_is_sandbox_id(backend: DeepInfraSandbox) -> None:
    assert backend.id.startswith("sb-")


def test_execute_runs_bash_and_merges_stderr(
    backend: DeepInfraSandbox, emulator: SandboxEmulator
) -> None:
    result = backend.execute("echo out; echo err >&2; exit 3")
    assert result.exit_code == 3
    assert result.output == "out\n\nerr\n"
    assert emulator.exec_calls[-1][:2] == ["bash", "-c"]


def test_execute_timeout_reports_124(backend: DeepInfraSandbox) -> None:
    result = backend.execute("sleep 5", timeout=1)
    assert result.exit_code == 124
    assert "timed out" in result.output


async def test_aexecute_uses_async_client(backend: DeepInfraSandbox) -> None:
    result = await backend.aexecute("printf hi")
    assert result.exit_code == 0
    assert result.output == "hi"


def test_upload_outside_workspace_is_staged(
    backend: DeepInfraSandbox, emulator: SandboxEmulator, tmp_path
) -> None:
    target = str(tmp_path / "nested" / "dir" / "file.bin")
    responses = backend.upload_files([(target, b"\x00\x01payload")])
    assert responses[0].error is None
    with open(target, "rb") as f:
        assert f.read() == b"\x00\x01payload"
    # The fs API only ever saw /workspace paths, as the real service requires.
    assert emulator.fs_calls
    assert all(path.startswith("/workspace/") for _, path in emulator.fs_calls)


def test_transfers_clean_up_staging(
    backend: DeepInfraSandbox, emulator: SandboxEmulator, tmp_path
) -> None:
    target = str(tmp_path / "f.txt")
    backend.upload_files([(target, b"x")])
    backend.download_files([target])
    staging = emulator.host_path("/workspace/.deepagents-transfer")
    assert os.listdir(staging) == []


def test_workspace_paths_roundtrip(backend: DeepInfraSandbox) -> None:
    path = "/workspace/project/notes.md"
    assert backend.upload_files([(path, b"# notes\n")])[0].error is None
    [download] = backend.download_files([path])
    assert download.content == b"# notes\n"
    assert backend.execute(f"cat {path}").output == "# notes\n"


def test_relative_paths_never_reach_the_sandbox(
    backend: DeepInfraSandbox, emulator: SandboxEmulator
) -> None:
    uploads = backend.upload_files([("rel/a.txt", b"a")])
    downloads = backend.download_files(["rel/a.txt"])
    assert uploads[0].error == "invalid_path"
    assert downloads[0].error == "invalid_path"
    assert emulator.exec_calls == []
    assert emulator.fs_calls == []


def test_batch_reports_partial_success_in_order(
    backend: DeepInfraSandbox, tmp_path
) -> None:
    good = str(tmp_path / "good.txt")
    directory = str(tmp_path)
    missing = str(tmp_path / "missing.txt")
    uploads = backend.upload_files([(good, b"ok"), (directory, b"nope"), ("x", b"")])
    assert [r.error for r in uploads] == [None, "is_directory", "invalid_path"]

    downloads = backend.download_files([missing, good, directory])
    assert [d.path for d in downloads] == [missing, good, directory]
    assert [d.error for d in downloads] == ["file_not_found", None, "is_directory"]
    assert downloads[1].content == b"ok"


def test_paths_with_shell_metacharacters(backend: DeepInfraSandbox, tmp_path) -> None:
    path = str(tmp_path / "it's a $file; rm -rf x.txt")
    assert backend.upload_files([(path, b"safe")])[0].error is None
    assert backend.download_files([path])[0].content == b"safe"
