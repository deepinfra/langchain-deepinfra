from langchain_core.load import dumpd
from langchain_tests.unit_tests import ChatModelUnitTests

from langchain_deepinfra import ChatDeepInfra

MODEL = "meta-llama/Llama-3.3-70B-Instruct"


class TestChatDeepInfraUnit(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> type[ChatDeepInfra]:
        return ChatDeepInfra

    @property
    def chat_model_params(self) -> dict:
        return {"model": MODEL}


def test_default_base_url() -> None:
    llm = ChatDeepInfra(model=MODEL, api_key="k")
    assert llm.deepinfra_api_base == "https://api.deepinfra.com/v1/openai"
    assert str(llm.root_client.base_url).rstrip("/") == (
        "https://api.deepinfra.com/v1/openai"
    )


def test_api_key_from_env() -> None:
    # `DEEPINFRA_API_TOKEN=test-token` is set by the autouse fixture.
    llm = ChatDeepInfra(model=MODEL)
    assert llm.deepinfra_api_key is not None
    assert llm.deepinfra_api_key.get_secret_value() == "test-token"


def test_base_url_override() -> None:
    llm = ChatDeepInfra(model=MODEL, api_key="k", base_url="https://example.com/v1")
    assert str(llm.root_client.base_url).rstrip("/") == "https://example.com/v1"


def test_secret_not_leaked_in_repr() -> None:
    llm = ChatDeepInfra(model=MODEL, api_key="super-secret")
    assert "super-secret" not in repr(llm)
    assert "super-secret" not in str(dumpd(llm))


def test_lc_serialization_metadata() -> None:
    llm = ChatDeepInfra(model=MODEL, api_key="k")
    assert llm.is_lc_serializable() is False
    assert llm.get_lc_namespace() == ["langchain_deepinfra", "chat_models"]
    assert llm.lc_secrets == {"deepinfra_api_key": "DEEPINFRA_API_TOKEN"}
    assert llm._llm_type == "deepinfra-chat"


def test_ls_params_provider() -> None:
    llm = ChatDeepInfra(model=MODEL, api_key="k")
    assert llm._get_ls_params()["ls_provider"] == "deepinfra"
