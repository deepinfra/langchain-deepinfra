"""LangChain integration for DeepInfra (chat, embeddings, rerank, and sandboxes)."""

from typing import TYPE_CHECKING, Any

from langchain_deepinfra._version import __version__
from langchain_deepinfra.chat_models import ChatDeepInfra
from langchain_deepinfra.embeddings import DeepInfraEmbeddings
from langchain_deepinfra.rerank import DeepInfraRerank

if TYPE_CHECKING:
    from langchain_deepinfra.sandbox import DeepInfraSandbox

__all__ = [
    "ChatDeepInfra",
    "DeepInfraEmbeddings",
    "DeepInfraRerank",
    "DeepInfraSandbox",
    "__version__",
]


def __getattr__(name: str) -> Any:
    # Imported lazily so the base install works without the `sandbox` extra.
    if name == "DeepInfraSandbox":
        from langchain_deepinfra.sandbox import DeepInfraSandbox

        return DeepInfraSandbox
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
