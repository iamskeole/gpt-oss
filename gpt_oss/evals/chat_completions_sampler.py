import time
from typing import Any

import openai
from openai import OpenAI

from .types import MessageList, SamplerBase, SamplerResponse

OPENAI_SYSTEM_MESSAGE_API = "You are a helpful assistant."
OPENAI_SYSTEM_MESSAGE_CHATGPT = (
    "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture."
    + "\nKnowledge cutoff: 2023-12\nCurrent date: 2024-04-01"
)


class ChatCompletionsSampler(SamplerBase):
    """Sample from a Chat Completions compatible API."""

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        system_message: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 1024,
        reasoning_model: bool = False,
        reasoning_effort: str | None = None,
        base_url: str = "http://localhost:8000/v1",
        enable_browser_tool: bool = False,
        enable_python_tool: bool = False,
        seed: int | None = None,
    ):
        self.client = OpenAI(base_url=base_url, timeout=24 * 60 * 60)
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_model = reasoning_model
        self.reasoning_effort = reasoning_effort
        self.image_format = "url"
        self.enable_browser_tool = enable_browser_tool
        self.enable_python_tool = enable_python_tool
        self.seed = seed

    def _pack_message(self, role: str, content: Any) -> dict[str, Any]:
        return {"role": str(role), "content": content}

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        if self.system_message:
            message_list = [
                self._pack_message("system", self.system_message)
            ] + message_list
        trial = 0
        tools: list[dict[str, Any]] = []
        if self.enable_browser_tool:
            tools.append(
                {
                    "type": "web_search",
                    "external_access": True,
                    "web_search_enabled": True,
                }
            )
        if self.enable_python_tool:
            tools.append({"type": "code_interpreter"})
        while True:
            try:
                if trial > 0:
                    print("Breaking for trial > 0")
                    return SamplerResponse(
                        response_text="",
                        response_metadata={"usage": None},
                        actual_queried_message_list=message_list,
                        n_errors=1,
                    )
                ts_start = time.time()
                if self.reasoning_model:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=message_list,
                        reasoning_effort=self.reasoning_effort,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        tools=tools,
                        seed=self.seed,
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=message_list,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        tools=tools,
                        seed=self.seed,
                    )
                latency = time.time() - ts_start

                choice = response.choices[0]
                content = choice.message.content
                usage = response.usage
                tool_calls = choice.message.tool_calls or []

                n_input_tokens = usage.prompt_tokens
                n_reasoning_tokens = 0
                if hasattr(usage.completion_tokens_details, "reasoning_tokens"):
                    n_reasoning_tokens = (
                        usage.completion_tokens_details.reasoning_tokens
                    )
                n_output_tokens = usage.completion_tokens
                n_response_tokens = n_output_tokens - n_reasoning_tokens

                n_tool_calls_browser = 0
                n_tool_calls_python = 0

                for tc in tool_calls:
                    if tc.function.name == "browser":
                        n_tool_calls_browser += 1
                    if tc.function.name == "python":
                        n_tool_calls_python += 1

                if getattr(choice.message, "reasoning", None):
                    message_list.append(
                        self._pack_message("assistant", choice.message.reasoning)
                    )

                if not content:
                    raise ValueError("OpenAI API returned empty response; retrying")
                return SamplerResponse(
                    response_text=content,
                    response_metadata={"usage": response.usage},
                    actual_queried_message_list=message_list,
                    n_input_tokens=n_input_tokens,
                    n_reasoning_tokens=n_reasoning_tokens,
                    n_response_tokens=n_response_tokens,
                    n_output_tokens=n_output_tokens,
                    n_tool_calls_browser=n_tool_calls_browser,
                    n_tool_calls_python=n_tool_calls_python,
                    latency=latency,
                )
            except openai.BadRequestError as e:
                print("Bad Request Error", e)
                return SamplerResponse(
                    response_text="No response (bad request).",
                    response_metadata={"usage": None},
                    actual_queried_message_list=message_list,
                )
            except Exception as e:
                exception_backoff = 2**trial  # exponential back off
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
            # unknown error shall throw exception
