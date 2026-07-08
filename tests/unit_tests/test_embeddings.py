from langchain_tests.unit_tests import EmbeddingsUnitTests

from langchain_deepinfra import DeepInfraEmbeddings

MODEL = "Qwen/Qwen3-Embedding-8B"


class TestDeepInfraEmbeddingsUnit(EmbeddingsUnitTests):
    @property
    def embeddings_class(self) -> type[DeepInfraEmbeddings]:
        return DeepInfraEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {"model": MODEL}


def test_default_base_url() -> None:
    emb = DeepInfraEmbeddings(model=MODEL, api_key="k")
    assert emb.deepinfra_api_base == "https://api.deepinfra.com/v1/openai"
    assert str(emb.client._client.base_url).rstrip("/") == (
        "https://api.deepinfra.com/v1/openai"
    )


def test_ctx_length_check_disabled() -> None:
    # Must be off for non-OpenAI models (no tiktoken tokenization).
    emb = DeepInfraEmbeddings(model=MODEL, api_key="k")
    assert emb.check_embedding_ctx_length is False


def test_api_key_from_env() -> None:
    emb = DeepInfraEmbeddings(model=MODEL)
    assert emb.deepinfra_api_key is not None
    assert emb.deepinfra_api_key.get_secret_value() == "test-token"


def test_lc_serialization_metadata() -> None:
    emb = DeepInfraEmbeddings(model=MODEL, api_key="k")
    assert emb.get_lc_namespace() == ["langchain_deepinfra", "embeddings"]
    assert emb.lc_secrets == {"deepinfra_api_key": "DEEPINFRA_API_TOKEN"}
