"""DeepInfra rerank integration.

`DeepInfraRerank` is a `BaseDocumentCompressor` backed by DeepInfra's native
inference endpoint (``POST /v1/inference/{model}``). Unlike chat and embeddings,
rerank is not OpenAI-compatible: the request carries parallel ``queries`` and
``documents`` arrays and the response returns one relevance ``score`` per
document.
"""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from typing import Any

import httpx
from langchain_core.callbacks.base import Callbacks
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_core.utils import from_env, secret_from_env
from pydantic import ConfigDict, Field, SecretStr

DEFAULT_API_BASE = "https://api.deepinfra.com/v1/openai"


def _inference_url(api_base: str, model: str) -> str:
    """Build the ``/inference/{model}`` URL, tolerating an ``/openai`` suffix.

    The chat/embeddings base is ``.../v1/openai`` but the rerank endpoint lives
    at ``.../v1/inference/{model}``, so a trailing ``openai`` segment is
    stripped. This lets a single ``DEEPINFRA_API_BASE`` value work for all three.
    """
    base = api_base.rstrip("/")
    if base.endswith("/openai"):
        base = base[: -len("/openai")]
    return f"{base}/inference/{model}"


class DeepInfraRerank(BaseDocumentCompressor):
    """DeepInfra reranker for document compression.

    Setup:
        Install ``langchain-deepinfra`` and set the environment variable
        ``DEEPINFRA_API_TOKEN``.

        .. code-block:: bash

            pip install -U langchain-deepinfra
            export DEEPINFRA_API_TOKEN="your-api-token"

    Instantiate and use:
        .. code-block:: python

            from langchain_deepinfra import DeepInfraRerank

            reranker = DeepInfraRerank(model="Qwen/Qwen3-Reranker-4B", top_n=3)
            reranker.compress_documents(documents, "my query")
    """

    model: str = Field()
    """Name of the DeepInfra reranker model, e.g. ``Qwen/Qwen3-Reranker-4B``."""

    top_n: int | None = 3
    """Number of top documents to return. ``None`` returns all, re-ranked."""

    instruction: str | None = None
    """Optional task instruction passed to the reranker model."""

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

    request_timeout: float | None = 60.0
    """Timeout (seconds) for the HTTP request."""

    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        return ["langchain_deepinfra", "rerank"]

    def _headers(self) -> dict[str, str]:
        if self.deepinfra_api_key is None:
            msg = (
                "DeepInfra API token is not set. Please set it in the "
                "`api_key` field or in the `DEEPINFRA_API_TOKEN` environment "
                "variable."
            )
            raise ValueError(msg)
        return {
            "Authorization": f"Bearer {self.deepinfra_api_key.get_secret_value()}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _payload(self, query: str, docs: list[str]) -> dict[str, Any]:
        # DeepInfra pairs `queries` and `documents` element-wise, so the single
        # query is repeated once per document.
        payload: dict[str, Any] = {"queries": [query] * len(docs), "documents": docs}
        if self.instruction is not None:
            payload["instruction"] = self.instruction
        return payload

    @staticmethod
    def _to_text(doc: str | Document) -> str:
        return doc.page_content if isinstance(doc, Document) else doc

    def _ranked(self, scores: list[float], top_n: int | None) -> list[dict[str, Any]]:
        results = [
            {"index": i, "relevance_score": float(score)}
            for i, score in enumerate(scores)
        ]
        results.sort(key=lambda r: r["relevance_score"], reverse=True)
        if top_n is not None:
            results = results[:top_n]
        return results

    def rerank(
        self,
        documents: Sequence[str | Document],
        query: str,
        *,
        top_n: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return ``[{"index": i, "relevance_score": s}, ...]`` sorted descending."""
        if len(documents) == 0:
            return []
        docs = [self._to_text(d) for d in documents]
        url = _inference_url(self.deepinfra_api_base, self.model)
        with httpx.Client(timeout=self.request_timeout) as client:
            resp = client.post(
                url, headers=self._headers(), json=self._payload(query, docs)
            )
            resp.raise_for_status()
            scores = resp.json()["scores"]
        return self._ranked(scores, self.top_n if top_n is None else top_n)

    async def arerank(
        self,
        documents: Sequence[str | Document],
        query: str,
        *,
        top_n: int | None = None,
    ) -> list[dict[str, Any]]:
        """Async variant of :meth:`rerank`."""
        if len(documents) == 0:
            return []
        docs = [self._to_text(d) for d in documents]
        url = _inference_url(self.deepinfra_api_base, self.model)
        async with httpx.AsyncClient(timeout=self.request_timeout) as client:
            resp = await client.post(
                url, headers=self._headers(), json=self._payload(query, docs)
            )
            resp.raise_for_status()
            scores = resp.json()["scores"]
        return self._ranked(scores, self.top_n if top_n is None else top_n)

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
        """Rerank ``documents`` against ``query`` and return the top ``top_n``."""
        compressed = []
        for res in self.rerank(documents, query):
            doc = documents[res["index"]]
            doc_copy = Document(
                page_content=doc.page_content, metadata=deepcopy(doc.metadata)
            )
            doc_copy.metadata["relevance_score"] = res["relevance_score"]
            compressed.append(doc_copy)
        return compressed

    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
        """Async variant of :meth:`compress_documents`."""
        compressed = []
        for res in await self.arerank(documents, query):
            doc = documents[res["index"]]
            doc_copy = Document(
                page_content=doc.page_content, metadata=deepcopy(doc.metadata)
            )
            doc_copy.metadata["relevance_score"] = res["relevance_score"]
            compressed.append(doc_copy)
        return compressed
