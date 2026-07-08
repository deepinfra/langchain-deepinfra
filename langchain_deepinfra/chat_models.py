"""DeepInfra chat model integration.

`ChatDeepInfra` is a thin wrapper over `BaseChatOpenAI`. DeepInfra exposes an
OpenAI-compatible Chat Completions endpoint at
``https://api.deepinfra.com/v1/openai``, so tool calling, structured output,
streaming, async, and vision are all inherited from ``langchain-openai``.
"""

from __future__ import annotations

from typing import Any

import openai
from langchain_core.language_models.chat_models import LangSmithParams
from langchain_core.utils import from_env, secret_from_env
from langchain_openai.chat_models.base import BaseChatOpenAI
from pydantic import ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

from langchain_deepinfra._version import __version__

DEFAULT_API_BASE = "https://api.deepinfra.com/v1/openai"


class ChatDeepInfra(BaseChatOpenAI):
    """DeepInfra chat model.

    Setup:
        Install ``langchain-deepinfra`` and set the environment variable
        ``DEEPINFRA_API_TOKEN``.

        .. code-block:: bash

            pip install -U langchain-deepinfra
            export DEEPINFRA_API_TOKEN="your-api-token"

    Instantiate:
        .. code-block:: python

            from langchain_deepinfra import ChatDeepInfra

            llm = ChatDeepInfra(
                model="meta-llama/Llama-3.3-70B-Instruct",
                temperature=0,
                # api_key="...",  # or set DEEPINFRA_API_TOKEN
            )

    Invoke:
        .. code-block:: python

            llm.invoke("What is the capital of France?")
    """

    model_name: str = Field(alias="model")
    """Name of the DeepInfra model, e.g. ``meta-llama/Llama-3.3-70B-Instruct``."""

    deepinfra_api_key: SecretStr | None = Field(
        alias="api_key",
        default_factory=secret_from_env("DEEPINFRA_API_TOKEN", default=None),
    )
    """DeepInfra API token.

    Automatically read from the ``DEEPINFRA_API_TOKEN`` environment variable if
    not provided.
    """

    deepinfra_api_base: str = Field(
        alias="base_url",
        default_factory=from_env("DEEPINFRA_API_BASE", default=DEFAULT_API_BASE),
    )
    """Base URL for API requests.

    Automatically read from the ``DEEPINFRA_API_BASE`` environment variable if
    not provided; defaults to DeepInfra's OpenAI-compatible endpoint.
    """

    # Neutralize the inherited OpenAI-named fields so there is exactly one field
    # per (``api_key``, ``base_url``) alias and the OpenAI client is built from
    # the DeepInfra-named fields in ``validate_environment`` below.
    openai_api_key: SecretStr | None = None
    openai_api_base: str | None = None

    model_config = ConfigDict(populate_by_name=True)

    @property
    def lc_secrets(self) -> dict[str, str]:
        return {"deepinfra_api_key": "DEEPINFRA_API_TOKEN"}

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        return ["langchain_deepinfra", "chat_models"]

    @classmethod
    def is_lc_serializable(cls) -> bool:
        # DeepInfra is not (yet) in langchain-core's built-in deserialization
        # allowlist, so `load()` cannot round-trip it without an explicit
        # `additional_import_mappings`. Advertise it as non-serializable rather
        # than claim a guarantee we cannot honor.
        return False

    @property
    def _llm_type(self) -> str:
        return "deepinfra-chat"

    def _get_ls_params(
        self, stop: list[str] | None = None, **kwargs: Any
    ) -> LangSmithParams:
        params = super()._get_ls_params(stop=stop, **kwargs)
        params["ls_provider"] = "deepinfra"
        return params

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate configuration and build the OpenAI-compatible clients."""
        if self.n is not None and self.n < 1:
            msg = "n must be at least 1."
            raise ValueError(msg)
        if self.n is not None and self.n > 1 and self.streaming:
            msg = "n must be 1 when streaming."
            raise ValueError(msg)

        client_params: dict[str, Any] = {
            "api_key": (
                self.deepinfra_api_key.get_secret_value()
                if self.deepinfra_api_key
                else None
            ),
            "base_url": self.deepinfra_api_base,
            "timeout": self.request_timeout,
            "default_headers": self.default_headers,
            "default_query": self.default_query,
        }
        if self.max_retries is not None:
            client_params["max_retries"] = self.max_retries

        if client_params["api_key"] is None:
            msg = (
                "DeepInfra API token is not set. Please set it in the "
                "`api_key` field or in the `DEEPINFRA_API_TOKEN` environment "
                "variable."
            )
            raise ValueError(msg)

        if not (self.client or None):
            sync_specific = {"http_client": self.http_client}
            self.root_client = openai.OpenAI(**client_params, **sync_specific)  # type: ignore[arg-type]
            self.client = self.root_client.chat.completions
        if not (self.async_client or None):
            async_specific = {"http_client": self.http_async_client}
            self.root_async_client = openai.AsyncOpenAI(
                **client_params,
                **async_specific,  # type: ignore[arg-type]
            )
            self.async_client = self.root_async_client.chat.completions

        if self.stream_usage is not False:
            self.stream_usage = True

        return self

    @model_validator(mode="after")
    def _set_deepinfra_version(self) -> Self:
        """Record the package version in run metadata.

        Named uniquely so it does not shadow ``BaseChatOpenAI``'s own version
        validator (Pydantic replaces same-named validators rather than chaining
        them).
        """
        self._add_version("langchain-deepinfra", __version__)
        return self
