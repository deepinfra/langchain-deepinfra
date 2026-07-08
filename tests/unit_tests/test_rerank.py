import json

import httpx
import pytest
import respx
from langchain_core.documents import Document

from langchain_deepinfra import DeepInfraRerank
from langchain_deepinfra.rerank import _inference_url

MODEL = "Qwen/Qwen3-Reranker-4B"
URL = f"https://api.deepinfra.com/v1/inference/{MODEL}"

DOCS = [
    Document(page_content="Paris is the capital of France.", metadata={"id": 0}),
    Document(page_content="The Eiffel Tower is in Paris.", metadata={"id": 1}),
    Document(page_content="Bananas are yellow.", metadata={"id": 2}),
]
QUERY = "What is the capital of France?"


def test_inference_url_strips_openai() -> None:
    assert _inference_url("https://api.deepinfra.com/v1/openai", MODEL) == URL
    assert _inference_url("https://api.deepinfra.com/v1", MODEL) == URL
    assert _inference_url("https://api.deepinfra.com/v1/openai/", MODEL) == URL


@respx.mock
def test_rerank_request_shape() -> None:
    route = respx.post(URL).mock(
        return_value=httpx.Response(200, json={"scores": [0.2, 0.9, 0.01]})
    )
    reranker = DeepInfraRerank(model=MODEL, api_key="k")
    reranker.rerank(DOCS, QUERY)

    assert route.called
    payload = json.loads(route.calls.last.request.content)
    # queries repeated once per document, paired element-wise with documents
    assert payload["queries"] == [QUERY] * len(DOCS)
    assert payload["documents"] == [d.page_content for d in DOCS]
    assert "instruction" not in payload
    auth = route.calls.last.request.headers["authorization"]
    assert auth == "Bearer k"


@respx.mock
def test_rerank_sorts_and_maps_indices() -> None:
    respx.post(URL).mock(
        return_value=httpx.Response(200, json={"scores": [0.2, 0.9, 0.01]})
    )
    reranker = DeepInfraRerank(model=MODEL, api_key="k", top_n=None)
    results = reranker.rerank(DOCS, QUERY)
    assert [r["index"] for r in results] == [1, 0, 2]
    assert results[0]["relevance_score"] == pytest.approx(0.9)


@respx.mock
def test_rerank_top_n() -> None:
    respx.post(URL).mock(
        return_value=httpx.Response(200, json={"scores": [0.2, 0.9, 0.01]})
    )
    reranker = DeepInfraRerank(model=MODEL, api_key="k", top_n=2)
    results = reranker.rerank(DOCS, QUERY)
    assert len(results) == 2
    assert [r["index"] for r in results] == [1, 0]


@respx.mock
def test_compress_documents_attaches_score_and_ranks() -> None:
    respx.post(URL).mock(
        return_value=httpx.Response(200, json={"scores": [0.2, 0.9, 0.01]})
    )
    reranker = DeepInfraRerank(model=MODEL, api_key="k", top_n=2)
    compressed = reranker.compress_documents(DOCS, QUERY)
    assert len(compressed) == 2
    assert compressed[0].page_content == DOCS[1].page_content
    assert compressed[0].metadata["relevance_score"] == pytest.approx(0.9)
    # original documents are not mutated
    assert "relevance_score" not in DOCS[1].metadata


@respx.mock
def test_instruction_included_when_set() -> None:
    route = respx.post(URL).mock(
        return_value=httpx.Response(200, json={"scores": [0.5, 0.5, 0.5]})
    )
    reranker = DeepInfraRerank(model=MODEL, api_key="k", instruction="Rank passages")
    reranker.rerank(DOCS, QUERY)
    payload = json.loads(route.calls.last.request.content)
    assert payload["instruction"] == "Rank passages"


def test_empty_documents_short_circuits() -> None:
    reranker = DeepInfraRerank(model=MODEL, api_key="k")
    assert reranker.rerank([], QUERY) == []
    assert reranker.compress_documents([], QUERY) == []


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPINFRA_API_TOKEN", raising=False)
    reranker = DeepInfraRerank(model=MODEL)
    with pytest.raises(ValueError, match="DEEPINFRA_API_TOKEN"):
        reranker.rerank(DOCS, QUERY)


@respx.mock
async def test_arerank() -> None:
    respx.post(URL).mock(
        return_value=httpx.Response(200, json={"scores": [0.2, 0.9, 0.01]})
    )
    reranker = DeepInfraRerank(model=MODEL, api_key="k", top_n=None)
    results = await reranker.arerank(DOCS, QUERY)
    assert [r["index"] for r in results] == [1, 0, 2]
