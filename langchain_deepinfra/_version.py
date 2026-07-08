"""Package version, resolved from installed metadata."""

from importlib import metadata

try:
    __version__ = metadata.version("langchain-deepinfra")
except metadata.PackageNotFoundError:  # pragma: no cover - editable/source runs
    __version__ = "0.0.0"

__all__ = ["__version__"]
