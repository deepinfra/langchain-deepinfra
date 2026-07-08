"""Live embedding tests. Require ``DEEPINFRA_API_TOKEN``."""

from langchain_tests.integration_tests import EmbeddingsIntegrationTests

from langchain_deepinfra import DeepInfraEmbeddings

MODEL = "BAAI/bge-base-en-v1.5"


class TestDeepInfraEmbeddingsIntegration(EmbeddingsIntegrationTests):
    @property
    def embeddings_class(self) -> type[DeepInfraEmbeddings]:
        return DeepInfraEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {"model": MODEL}
