"""Live rerank tests. Require ``DEEPINFRA_API_TOKEN``."""

import os

import pytest
from langchain_core.documents import Document

from langchain_deepinfra import DeepInfraRerank

MODEL = "Qwen/Qwen3-Reranker-4B"

DOCS = [
    Document(page_content="Bananas are yellow."),
    Document(page_content="The capital of France is Paris."),
    Document(page_content="Cats are mammals."),
]
QUERY = "What is the capital of France?"

pytestmark = pytest.mark.skipif(
    not os.environ.get("DEEPINFRA_API_TOKEN"),
    reason="DEEPINFRA_API_TOKEN not set",
)


def test_rerank_orders_relevant_first() -> None:
    reranker = DeepInfraRerank(model=MODEL, top_n=None)
    results = reranker.rerank(DOCS, QUERY)
    assert len(results) == len(DOCS)
    # The Paris document (index 1) should rank first.
    assert results[0]["index"] == 1


def test_compress_documents_top_n() -> None:
    reranker = DeepInfraRerank(model=MODEL, top_n=1)
    compressed = reranker.compress_documents(DOCS, QUERY)
    assert len(compressed) == 1
    assert "Paris" in compressed[0].page_content
    assert "relevance_score" in compressed[0].metadata


async def test_arerank() -> None:
    reranker = DeepInfraRerank(model=MODEL, top_n=None)
    results = await reranker.arerank(DOCS, QUERY)
    assert results[0]["index"] == 1
