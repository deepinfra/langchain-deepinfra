"""Live sandbox tests. Require ``DEEPINFRA_API_KEY`` and create a real sandbox."""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

pytest.importorskip("deepagents")  # requires Python 3.11+

from deepinfra import Sandbox  # noqa: E402
from langchain_tests.integration_tests import SandboxIntegrationTests  # noqa: E402

from langchain_deepinfra import DeepInfraSandbox  # noqa: E402

pytestmark = pytest.mark.skipif(
    not os.getenv("DEEPINFRA_API_KEY"), reason="DEEPINFRA_API_KEY is not set"
)


@pytest.mark.asyncio(loop_scope="class")
class TestDeepInfraSandboxIntegration(SandboxIntegrationTests):
    @pytest.fixture(scope="class")
    def sandbox(self) -> Iterator[DeepInfraSandbox]:
        sandbox = Sandbox.create(timeout="15m")
        try:
            yield DeepInfraSandbox(sandbox=sandbox)
        finally:
            sandbox.terminate()
