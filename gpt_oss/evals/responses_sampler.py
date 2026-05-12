import time
from typing import Any

import openai
from openai import OpenAI

from .types import MessageList, SamplerBase, SamplerResponse


class ResponsesSampler(SamplerBase):
    """
    Sample from OpenAI's responses API
    """

    def __init__(
        self,
        model: str,
        developer_message: str | None = None,
        temperature: float = 1.0,
        max_tokens: int = 131_072,
        reasoning_model: bool = False,
        reasoning_effort: str | None = None,
        base_url: str = "http://localhost:8000/v1",
        enable_browser_tool: bool = False,
        enable_python_tool: bool = False,
        seed: int | None = None,
    ):
        self.client = OpenAI(base_url=base_url, timeout=24 * 60 * 60)
        self.model = model
        self.developer_message = developer_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.image_format = "url"
        self.reasoning_model = reasoning_model
        self.reasoning_effort = reasoning_effort
        self.enable_browser_tool = enable_browser_tool
        self.enable_python_tool = enable_python_tool
        self.seed = seed

    def _pack_message(self, role: str, content: Any) -> dict[str, Any]:
        return {"role": role, "content": content}

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        if self.developer_message:
            message_list = [
                self._pack_message("developer", self.developer_message)
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
                request_kwargs = {
                    "model": self.model,
                    "input": message_list,
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens,
                    "tools": tools,
                    "extra_body": {
                        "seed": self.seed,
                    },
                }
                if self.reasoning_model:
                    request_kwargs["reasoning"] = (
                        {"effort": self.reasoning_effort}
                        if self.reasoning_effort
                        else None
                    )

                ts_start = time.time()
                response = self.client.responses.create(**request_kwargs)
                latency = time.time() - ts_start
                usage = response.usage

                n_errors = trial
                n_input_tokens = usage.input_tokens
                n_reasoning_tokens = 0
                if hasattr(usage.output_tokens_details, "reasoning_tokens"):
                    n_reasoning_tokens = usage.output_tokens_details.reasoning_tokens
                n_output_tokens = usage.output_tokens
                n_response_tokens = n_output_tokens - n_reasoning_tokens

                n_tool_calls_browser = 0
                n_tool_calls_python = 0

                for output in response.output:
                    if output.type == "web_search_call":
                        n_tool_calls_browser += 1
                    if output.type == "code_interpreter_call":
                        n_tool_calls_python += 1
                    if output.type == "reasoning":
                        message_list.append(
                            self._pack_message("assistant", output.content[0].text)
                        )
                    elif hasattr(output, "content"):
                        for c in output.content:
                            # c.text handled below
                            pass

                output_text = response.output_text
                if not output_text:
                    n_errors += 1

                return SamplerResponse(
                    response_text=response.output_text,
                    response_metadata={"usage": usage},
                    actual_queried_message_list=message_list,
                    n_input_tokens=n_input_tokens,
                    n_reasoning_tokens=n_reasoning_tokens,
                    n_response_tokens=n_response_tokens,
                    n_output_tokens=n_output_tokens,
                    latency=latency,
                    n_errors=n_errors,
                )
            except openai.BadRequestError as e:
                print("Bad Request Error", e)
                return SamplerResponse(
                    response_text="",
                    response_metadata={"usage": None},
                    actual_queried_message_list=message_list,
                )
            except Exception as e:
                exception_backoff = 2**trial  # expontial back off
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
            # unknown error shall throw exception
