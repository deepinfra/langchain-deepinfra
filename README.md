# langchain-deepinfra

[![PyPI version](https://img.shields.io/pypi/v/langchain-deepinfra.svg)](https://pypi.org/project/langchain-deepinfra/)

The official [LangChain](https://github.com/langchain-ai/langchain) integration for
[DeepInfra](https://deepinfra.com) — chat models, embeddings, and reranking.

## Installation

```bash
pip install -U langchain-deepinfra
export DEEPINFRA_API_TOKEN="your-api-token"
```

Get an API token from the [DeepInfra dashboard](https://deepinfra.com/dash/api_keys).

## Chat

`ChatDeepInfra` wraps DeepInfra's OpenAI-compatible Chat Completions endpoint, so it
supports tool calling, structured output, streaming, async, and multimodal inputs.

```python
from langchain_deepinfra import ChatDeepInfra

llm = ChatDeepInfra(model="meta-llama/Llama-3.3-70B-Instruct", temperature=0)
llm.invoke("What is the capital of France?")
```

## Embeddings

```python
from langchain_deepinfra import DeepInfraEmbeddings

embeddings = DeepInfraEmbeddings(model="Qwen/Qwen3-Embedding-8B")
embeddings.embed_query("Hello, world!")
embeddings.embed_documents(["doc one", "doc two"])
```

## Rerank

`DeepInfraRerank` is a `BaseDocumentCompressor`, so it drops into a
`ContextualCompressionRetriever`.

```python
from langchain_deepinfra import DeepInfraRerank

reranker = DeepInfraRerank(model="Qwen/Qwen3-Reranker-4B", top_n=3)
reranked = reranker.compress_documents(documents, "my query")
# each returned Document carries metadata["relevance_score"]
```

## Configuration

| Argument             | Env var              | Default                                  |
| -------------------- | -------------------- | ---------------------------------------- |
| `api_key`            | `DEEPINFRA_API_TOKEN`| — (required)                             |
| `base_url`           | `DEEPINFRA_API_BASE` | `https://api.deepinfra.com/v1/openai`    |

## Architecture

DeepInfra's chat and embeddings APIs are OpenAI-compatible, so those classes are thin
subclasses of `langchain-openai`:

- **`ChatDeepInfra`** subclasses `BaseChatOpenAI`. It overrides only the credential /
  base-URL fields (`DEEPINFRA_API_TOKEN`, `https://api.deepinfra.com/v1/openai`) and
  rebuilds the OpenAI client in `validate_environment`. All chat features are inherited.
- **`DeepInfraEmbeddings`** subclasses `OpenAIEmbeddings` the same way, with
  `check_embedding_ctx_length=False` (DeepInfra models are not tokenized with tiktoken).
- **`DeepInfraRerank`** is a standalone `BaseDocumentCompressor`. Rerank is *not*
  OpenAI-compatible: it calls `POST /v1/inference/{model}` with parallel `queries` /
  `documents` arrays (the query is repeated once per document) and reads one relevance
  `score` per document from the response.

## Development

```bash
uv sync                 # install
make unit_test          # offline unit tests
make lint type_check    # ruff + mypy
make integration_test   # live tests (needs DEEPINFRA_API_TOKEN)
```

## License

MIT
