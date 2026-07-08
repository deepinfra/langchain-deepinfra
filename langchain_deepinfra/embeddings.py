"""DeepInfra embeddings integration.

`DeepInfraEmbeddings` is a thin wrapper over `OpenAIEmbeddings`. DeepInfra
exposes an OpenAI-compatible embeddings endpoint at
``https://api.deepinfra.com/v1/openai/embeddings``.
"""

from __future__ import annotations

from typing import Any

import openai
from langchain_core.utils import from_env, secret_from_env
from langchain_openai import OpenAIEmbeddings
from pydantic import ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

DEFAULT_API_BASE = "https://api.deepinfra.com/v1/openai"


class DeepInfraEmbeddings(OpenAIEmbeddings):
    """DeepInfra embedding model.

    Setup:
        Install ``langchain-deepinfra`` and set the environment variable
        ``DEEPINFRA_API_TOKEN``.

        .. code-block:: bash

            pip install -U langchain-deepinfra
            export DEEPINFRA_API_TOKEN="your-api-token"

    Instantiate:
        .. code-block:: python

            from langchain_deepinfra import DeepInfraEmbeddings

            embeddings = DeepInfraEmbeddings(model="Qwen/Qwen3-Embedding-8B")
            embeddings.embed_query("Hello, world!")
    """

    model: str = Field()
    """Name of the DeepInfra embedding model, e.g. ``Qwen/Qwen3-Embedding-8B``."""

    deepinfra_api_key: SecretStr | None = Field(
        alias="api_key",
        default_factory=secret_from_env("DEEPINFRA_API_TOKEN", default=None),
    )
    """DeepInfra API token, read from ``DEEPINFRA_API_TOKEN`` if not provided."""

    deepinfra_api_base: str = Field(
        alias="base_url",
        default_factory=from_env("DEEPINFRA_API_BASE", default=DEFAULT_API_BASE),
    )
    """Base URL for API requests, read from ``DEEPINFRA_API_BASE`` if not provided."""

    # DeepInfra models are not OpenAI models, so the tiktoken-based context-length
    # bookkeeping in the parent must be disabled; raw strings are sent to the API.
    check_embedding_ctx_length: bool = False

    # Neutralize the inherited OpenAI-named credential fields (see chat_models).
    openai_api_key: SecretStr | None = None
    openai_api_base: str | None = None

    model_config = ConfigDict(populate_by_name=True)

    @property
    def lc_secrets(self) -> dict[str, str]:
        return {"deepinfra_api_key": "DEEPINFRA_API_TOKEN"}

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        return ["langchain_deepinfra", "embeddings"]

    @classmethod
    def is_lc_serializable(cls) -> bool:
        # See ChatDeepInfra.is_lc_serializable for rationale.
        return False

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate configuration and build the OpenAI-compatible clients."""
        api_key = (
            self.deepinfra_api_key.get_secret_value()
            if self.deepinfra_api_key
            else None
        )
        if api_key is None:
            msg = (
                "DeepInfra API token is not set. Please set it in the "
                "`api_key` field or in the `DEEPINFRA_API_TOKEN` environment "
                "variable."
            )
            raise ValueError(msg)

        client_params: dict[str, Any] = {
            "api_key": api_key,
            "base_url": self.deepinfra_api_base,
            "timeout": self.request_timeout,
            "max_retries": self.max_retries,
            "default_headers": self.default_headers,
            "default_query": self.default_query,
        }
        if not self.client:
            self.client = openai.OpenAI(
                **client_params,
                http_client=self.http_client,  # type: ignore[arg-type]
            ).embeddings
        if not self.async_client:
            self.async_client = openai.AsyncOpenAI(
                **client_params,
                http_client=self.http_async_client,  # type: ignore[arg-type]
            ).embeddings
        return self
