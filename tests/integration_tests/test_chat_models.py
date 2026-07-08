"""Live chat tests. Require ``DEEPINFRA_API_TOKEN`` and hit the DeepInfra API."""

from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_deepinfra import ChatDeepInfra

MODEL = "meta-llama/Llama-3.3-70B-Instruct"


class TestChatDeepInfraIntegration(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> type[ChatDeepInfra]:
        return ChatDeepInfra

    @property
    def chat_model_params(self) -> dict:
        return {"model": MODEL, "temperature": 0}

    @property
    def has_tool_calling(self) -> bool:
        return True

    @property
    def has_structured_output(self) -> bool:
        return True

    @property
    def supports_image_inputs(self) -> bool:
        return False
