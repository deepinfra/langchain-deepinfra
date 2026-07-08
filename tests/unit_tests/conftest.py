import pytest


@pytest.fixture(autouse=True)
def _deepinfra_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide a fake token so models can be constructed offline."""
    monkeypatch.setenv("DEEPINFRA_API_TOKEN", "test-token")
    monkeypatch.delenv("DEEPINFRA_API_BASE", raising=False)
