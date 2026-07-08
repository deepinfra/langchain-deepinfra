from langchain_deepinfra import (
    ChatDeepInfra,
    DeepInfraEmbeddings,
    DeepInfraRerank,
    __version__,
)

EXPECTED = ["ChatDeepInfra", "DeepInfraEmbeddings", "DeepInfraRerank", "__version__"]


def test_all_exports() -> None:
    import langchain_deepinfra

    assert sorted(langchain_deepinfra.__all__) == sorted(EXPECTED)


def test_classes_importable() -> None:
    assert ChatDeepInfra is not None
    assert DeepInfraEmbeddings is not None
    assert DeepInfraRerank is not None


def test_version_is_string() -> None:
    assert isinstance(__version__, str)
