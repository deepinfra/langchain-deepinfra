# langchain-deepinfra

[![PyPI version](https://img.shields.io/pypi/v/langchain-deepinfra.svg)](https://pypi.org/project/langchain-deepinfra/)

The official [LangChain](https://github.com/langchain-ai/langchain) integration for
[DeepInfra](https://deepinfra.com) — chat models, embeddings, reranking, and a
[Deep Agents](https://github.com/langchain-ai/deepagents) sandbox backend.

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

## Sandbox (Deep Agents)

`DeepInfraSandbox` is a [Deep Agents](https://github.com/langchain-ai/deepagents)
sandbox backend, so an agent's shell commands and file edits run inside a
[DeepInfra Sandbox](https://deepinfra.com/blog/sandboxes) (an isolated microVM)
instead of on your machine.

```bash
pip install -U "langchain-deepinfra[sandbox]"   # Python 3.11+
export DEEPINFRA_API_KEY="your-api-token"       # read by the deepinfra SDK
```

```python
from deepagents import create_deep_agent
from deepinfra import Sandbox

from langchain_deepinfra import ChatDeepInfra, DeepInfraSandbox

sandbox = Sandbox.create(plan="medium", timeout="30m")
try:
    agent = create_deep_agent(
        model=ChatDeepInfra(model="meta-llama/Llama-3.3-70B-Instruct"),
        backend=DeepInfraSandbox(sandbox=sandbox),
    )
    agent.invoke(
        {"messages": [{"role": "user", "content": "Write and run fizzbuzz.py"}]}
    )
finally:
    sandbox.terminate()
```

The backend wraps a sandbox you own, so you decide its plan and lifetime and can
reuse one sandbox across runs (`Sandbox.from_id(...)`). Commands run with
`bash -c`; `execute(..., timeout=...)` is forwarded to the sandbox, and a command
the sandbox kills comes back with exit code 124. DeepInfra's file API only
accepts paths under `/workspace` (the directory that survives stop/start), so
uploads and downloads for other absolute paths are staged under
`/workspace/.deepagents-transfer` and moved with one command per batch. Missing
files, directories, unreadable files and relative paths are reported per file
as `file_not_found`, `is_directory`, `permission_denied` and `invalid_path`.

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
- **`DeepInfraSandbox`** subclasses Deep Agents' `BaseSandbox`, which builds `ls`, `read`,
  `write`, `edit`, `grep` and `glob` on top of three primitives. It implements those with
  the `deepinfra` SDK: `execute` via `Sandbox.exec`/`aexec`, and `upload_files` /
  `download_files` via the `/workspace` fs API plus one staging command per batch.

## Development

```bash
uv sync                 # install
make unit_test          # offline unit tests
make lint type_check    # ruff + mypy
make integration_test   # live tests (needs DEEPINFRA_API_TOKEN)
```

## License

MIT
